"""ROS-independent Isaac wheel navigation helpers and a local convex QP.

No module import creates a Kit app or imports ROS/MuJoCo. The caller owns every
physics step, native torque actuation, contact/load/sensor gates and recovery.
Nav2 is an isolated global path subprocess, NOT the local controller. These
helpers do not make the nonlinear rolling/contact task a convex problem.
"""
from __future__ import annotations

from dataclasses import dataclass
import json
import math
import os
from pathlib import Path
import subprocess
import time

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
CONFIG = ROOT / "configs/isaac/warehouse_task.json"


def _finite_vector(values, count, label):
    result = np.asarray(values, dtype=float)
    if result.shape != (count,) or not np.isfinite(result).all():
        raise ValueError(f"{label} must contain {count} finite values")
    return result


def _positive(value, label):
    if isinstance(value, bool) or not math.isfinite(float(value)) or float(value) <= 0:
        raise ValueError(f"{label} must be positive and finite")
    return float(value)


def load_config(path=CONFIG):
    return json.loads(Path(path).read_text())


def wrap_angle(angle):
    return math.atan2(math.sin(angle), math.cos(angle))


def quaternion_matrix_wxyz(quaternion):
    q = _finite_vector(quaternion, 4, "quaternion_wxyz")
    norm = float(np.linalg.norm(q))
    if norm < 1e-12:
        raise ValueError("Quaternion norm is zero")
    w, x, y, z = q / norm
    return np.array([[1-2*(y*y+z*z), 2*(x*y-z*w), 2*(x*z+y*w)],
                     [2*(x*y+z*w), 1-2*(x*x+z*z), 2*(y*z-x*w)],
                     [2*(x*z-y*w), 2*(y*z+x*w), 1-2*(x*x+y*y)]])


class NativeBaseStateUnavailable(RuntimeError):
    """Native PhysX measurements unavailable; never substitute a USD pose."""


def require_native_base_view(base_prim):
    """Check the pinned SingleRigidPrim's native view without reading state.

Isaac5.1 SingleRigidPrim does not expose a public handle-validity method;
its underlying RigidPrim does. A STOP/reset invalidates that handle.
"""
    checker = getattr(base_prim, 'is_physics_handle_valid', None)
    if not callable(checker):
        checker = getattr(getattr(base_prim, '_rigid_prim_view', None),
                          'is_physics_handle_valid', None)
    if not callable(checker):
        raise NativeBaseStateUnavailable('Native PhysX base view validity cannot be checked')
    try:
        valid = bool(checker())
    except Exception as error:
        raise NativeBaseStateUnavailable('Native PhysX base view validity check failed: '+str(error)) from error
    if not valid:
        raise NativeBaseStateUnavailable('Native PhysX base view is invalid/unavailable after STOP/reset; measurement unavailable')


def gt_base_state(base_prim, axle_offset_base_m=(0.228, 0.0, 0.0)):
    """Read valid native SingleRigidPrim pose/twist; this is GT localization.

GUI STOP/reset may occur while rendering. Check before AND after the reads;
None/malformed native values fail explicitly, with no stale USD fallback.
"""
    local_offset = _finite_vector(axle_offset_base_m, 3, "axle offset")
    require_native_base_view(base_prim)
    try:
        pose = base_prim.get_world_pose()
        if pose is None or len(pose) != 2:
            raise ValueError('native world pose is absent/malformed')
        position, quaternion = pose
        position = _finite_vector(position, 3, "base position")
        rotation = quaternion_matrix_wxyz(quaternion)
        offset = rotation @ local_offset
        linear = _finite_vector(base_prim.get_linear_velocity(), 3, "base linear velocity")
        angular = _finite_vector(base_prim.get_angular_velocity(), 3, "base angular velocity")
        # PhysX velocity is measured at COM. Express COM->axle in world axes.
        com = base_prim.get_com()
        if com is None or len(com) != 2:
            raise ValueError('native base COM is absent/malformed')
        local_com, _ = com
        com_to_axle = offset - rotation @ _finite_vector(np.asarray(local_com).reshape(-1), 3, "base local COM")
        require_native_base_view(base_prim)
    except Exception as error:
        raise NativeBaseStateUnavailable('Native PhysX base measurement unavailable: '+str(error)) from error
    axle_velocity = linear + np.cross(angular, com_to_axle)
    yaw = math.atan2(rotation[1, 0], rotation[0, 0])
    return {"base_position_m": position.tolist(), "base_yaw_rad": yaw,
            "axle_xy_m": (position + offset)[:2].tolist(),
            "axle_linear_velocity_m_s": axle_velocity.tolist(),
            "measured_planar_speed_m_s": float(np.linalg.norm(axle_velocity[:2])),
            "measured_yaw_rate_rad_s": float(angular[2]),
            "localization": "simulator_ground_truth"}


def wheel_rates_for_twist(v_m_s, omega_rad_s, *, radius_m=0.1,
                          separation_m=0.53, native_forward_sign=-1.0):
    """Return native left/right qdot, not wheel torque or an actuation call.

    Pinned A USD has both revolute axes along base -Y. Therefore native negative
    rates roll toward base +X. This geometric inference needs a drive probe.
    """
    v, omega = _finite_vector([v_m_s, omega_rad_s], 2, "twist")
    radius, separation = _positive(radius_m, "wheel radius"), _positive(separation_m, "wheel separation")
    if native_forward_sign not in (-1.0, 1.0):
        raise ValueError("native_forward_sign must be -1 or +1")
    return {"left_wheel": float(native_forward_sign * (v - separation * omega / 2) / radius),
            "right_wheel": float(native_forward_sign * (v + separation * omega / 2) / radius)}


def cpp_wheel_reference(cpp_joint_names, position_reference, v_m_s, omega_rad_s, **geometry):
    """Build vendor PD targets in its CPP order (A starts right,left).

    Return (integrated_velocity_mode, target). Pass these through the task's
    _build_reference_target/PD update_target. It never sets q or teleports base.
    """
    names = list(cpp_joint_names)
    target = _finite_vector(position_reference, len(names), "CPP reference").copy()
    modes = np.zeros(len(names), dtype=bool)
    rates = wheel_rates_for_twist(v_m_s, omega_rad_s, **geometry)
    for name, rate in rates.items():
        index = names.index(name)
        modes[index], target[index] = True, rate
    return modes, target


def point_rectangle_distance(point_xy, rectangle):
    """Signed distance and outward gradient for an axis-aligned rectangle."""
    p = _finite_vector(point_xy, 2, "point")
    bounds = _finite_vector(rectangle, 4, "rectangle")
    low, high = bounds[:2], bounds[2:]
    if np.any(low >= high):
        raise ValueError("Rectangle minima must be below maxima")
    closest = np.clip(p, low, high)
    delta = p - closest
    distance = float(np.linalg.norm(delta))
    if distance > 1e-12:
        return distance, delta / distance
    distances = np.r_[p-low, high-p]
    face = int(np.argmin(distances))
    gradient = np.array([[-1.,0.], [0.,-1.], [1.,0.], [0.,1.]])[face]
    return -float(distances[face]), gradient


def extract_static_collision_rectangles(stage, *, prefix="/World/Warehouse",
                                        height_range_m=(0.06, 1.9), slice_mesh=False):
    """Conservative loaded-stage CollisionAPI AABB snapshot, no engine step.

    Include invisible physical colliders. Ignore render-only meshes. The
    caller must obtain this from the actual loaded, composed stage. A single
    floor+walls mesh whose AABB spans free space can over-block; do not silently
    discard it. Moving warehouse bodies remain conservative initial obstacles,
    not dynamic perception. Returns coverage diagnostics even when empty.
    """
    from pxr import Usd, UsdGeom, UsdPhysics
    root = stage.GetPrimAtPath(prefix)
    if not root.IsValid() or not root.GetChildren():
        raise ValueError(f"Loaded warehouse prefix absent/empty: {prefix}")
    zlow, zhigh = _finite_vector(height_range_m, 2, "height_range")
    if zlow >= zhigh:
        raise ValueError("Invalid collision height range")
    cache = UsdGeom.BBoxCache(Usd.TimeCode.Default(),
                             [UsdGeom.Tokens.default_, UsdGeom.Tokens.render,
                              UsdGeom.Tokens.proxy, UsdGeom.Tokens.guide],
                             useExtentsHint=False, ignoreVisibility=True)
    boxes, skipped, invalid = [], [], []
    collider_count = 0
    for prim in Usd.PrimRange(root):
        if not prim.HasAPI(UsdPhysics.CollisionAPI):
            continue
        enabled = UsdPhysics.CollisionAPI(prim).GetCollisionEnabledAttr().Get()
        if enabled is False:
            continue
        collider_count += 1
        bound = cache.ComputeWorldBound(prim).ComputeAlignedRange()
        lo, hi = np.array(bound.GetMin()), np.array(bound.GetMax())
        if bound.IsEmpty() or not np.isfinite(np.r_[lo, hi]).all():
            invalid.append(str(prim.GetPath()))
            continue
        if slice_mesh and prim.IsA(UsdGeom.Mesh):
            from static_mesh_slice import mesh_slice_rectangles
            sliced=mesh_slice_rectangles(prim,zlow,zhigh)
            if sliced is not None:
                if not sliced:skipped.append(str(prim.GetPath()))
                boxes.extend(sliced)
                continue
        if hi[2] < zlow or lo[2] > zhigh:
            skipped.append(str(prim.GetPath()))
            continue
        if np.any(lo[:2] >= hi[:2]):
            invalid.append(str(prim.GetPath()))
            continue
        boxes.append({"path": str(prim.GetPath()), "xy_bounds_m": [*lo[:2], *hi[:2]],
                      "z_bounds_m": [float(lo[2]), float(hi[2])]})
    return {"source": "loaded_stage_CollisionAPI_height_sliced_mesh_faces_and_other_world_AABBs" if slice_mesh else "loaded_stage_CollisionAPI_world_AABBs",
            "prefix": prefix, "height_range_m": [float(zlow), float(zhigh)],
            "enabled_collider_count": collider_count, "rectangles": boxes,
            "height_excluded_paths": skipped, "invalid_bounds_paths": invalid,
            "coverage_admitted": bool(collider_count > 0 and boxes and not invalid),
            "physical_execution_validated": False}


def workcell_rectangles(config):
    result = []
    for label, cell in config["workcells"].items():
        center, size = np.asarray(cell["table_center_m"]), np.asarray(cell["table_size_m"])
        result.append({"path": "workcell_" + label,
                       "xy_bounds_m": [*(center[:2]-size[:2]/2), *(center[:2]+size[:2]/2)]})
    return result


def _rectangles(values):
    result = []
    for entry in values:
        value = entry["xy_bounds_m"] if isinstance(entry, dict) else entry
        rect = _finite_vector(value, 4, "obstacle rectangle")
        if np.any(rect[:2] >= rect[2:]):
            raise ValueError("Obstacle rectangle minima must be below maxima")
        result.append(rect)
    return result


def write_nav2_map(output_dir, obstacle_rectangles, bounds_xy_m, resolution_m, radius_m):
    """Author an occupied/free PGM and Humble NavFn parameters from admitted geometry."""
    output = Path(output_dir).resolve()
    output.mkdir(parents=True, exist_ok=False)
    bounds = _finite_vector(bounds_xy_m, 4, "map bounds")
    resolution, radius = _positive(resolution_m, "resolution"), _positive(radius_m, "radius")
    if np.any(bounds[:2] >= bounds[2:]):
        raise ValueError("Invalid map bounds")
    width, height = np.ceil((bounds[2:]-bounds[:2])/resolution).astype(int)
    if width * height > 2_000_000:
        raise ValueError("Refusing oversized static map")
    grid = np.full((height, width), 254, dtype=np.uint8)
    for xmin, ymin, xmax, ymax in _rectangles(obstacle_rectangles):
        low = np.floor((np.array([xmin,ymin])-bounds[:2])/resolution).astype(int)
        high = np.ceil((np.array([xmax,ymax])-bounds[:2])/resolution).astype(int)
        x0,y0 = np.maximum(low,0); x1,y1 = np.minimum(high,[width,height])
        if x0 < x1 and y0 < y1:
            grid[y0:y1,x0:x1] = 0
    # Occupied boundary makes outside-map risk visible to both Nav2 and admission.
    grid[0,:] = grid[-1,:] = grid[:,0] = grid[:,-1] = 0
    pgm = output / "warehouse.pgm"
    pgm.write_bytes(f"P5\n{width} {height}\n255\n".encode() + grid[::-1].tobytes())
    yaml = output / "warehouse.yaml"
    yaml.write_text(f"image: warehouse.pgm\nmode: trinary\nresolution: {resolution}\n"
                    f"origin: [{bounds[0]}, {bounds[1]}, 0.0]\nnegate: 0\n"
                    "occupied_thresh: 0.65\nfree_thresh: 0.196\n")
    params = output / "nav2_planner.yaml"
    template = (ROOT / "configs/nav2_planner.yaml").read_text()
    params.write_text(template.replace("robot_radius: 0.2", f"robot_radius: {radius}")
                      .replace("inflation_radius: 0.5", f"inflation_radius: {radius + 0.3}")
                      .replace("resolution: 0.1", f"resolution: {resolution}"))
    return {"map_yaml": str(yaml), "params_yaml": str(params), "map_pgm": str(pgm),
            "bounds_xy_m": bounds.tolist(), "resolution_m": resolution,
            "planning_radius_m": radius, "width": int(width), "height": int(height)}


def request_nav2_route(output_dir, warehouse_snapshot, *, measured_envelope_radius_m,
                       config=None, domain_id=174, timeout_s=80.0):
    """Request a fresh real Humble global path, separate from the Isaac process.

    Existing core adapter runs in core .venv; its child uses system Python and
    /opt/ros/humble. Clean ROS/Conda/Isaac library variables only in that child.
    An unmeasured footprint or absent warehouse collision inventory rejects.
    """
    config = load_config() if config is None else config
    nav = config["navigation"]
    radius = _positive(measured_envelope_radius_m, "measured Isaac envelope radius")
    if radius > nav["design_envelope_radius_m"] + 1e-9:
        raise ValueError("Measured transport envelope exceeds the design cap; re-plan layout/radius")
    if not warehouse_snapshot.get("coverage_admitted"):
        raise ValueError("Warehouse collision coverage not admitted")
    output = Path(output_dir).resolve()
    output.mkdir(parents=True, exist_ok=False)
    obstacles = [*warehouse_snapshot["rectangles"], *workcell_rectangles(config)]
    planning_radius = radius + nav["planning_clearance_margin_m"]
    files = write_nav2_map(output / "map", obstacles, nav["map_bounds_xy_m"],
                           nav["map_resolution_m"], planning_radius)
    offset = float(config["robot"]["axle_offset_base_m"][0])
    start = np.asarray(nav["A_undocked_base_pose_m_rad"][:2]) + [offset,0.]
    goal = np.asarray(nav["B_predock_base_pose_m_rad"][:2]) + [offset,0.]
    (output / "geometry_snapshot.json").write_text(json.dumps(
        {"warehouse": warehouse_snapshot, "workcells": workcell_rectangles(config),
         "measured_envelope_radius_m": radius, "planning_radius_m": planning_radius}, indent=2) + "\n")
    env = os.environ.copy()
    for key in ("PYTHONPATH", "PYTHONHOME", "LD_LIBRARY_PATH", "VIRTUAL_ENV", "CONDA_PREFIX",
                "AMENT_PREFIX_PATH", "COLCON_PREFIX_PATH", "CMAKE_PREFIX_PATH", "ROS_PACKAGE_PATH",
                "ROS_DISTRO", "ROS_VERSION", "ROS_PYTHON_VERSION"):
        env.pop(key, None)
    env["PATH"], env["PYTHONNOUSERSITE"] = "/usr/bin:/bin", "1"
    code = ("import json,sys; sys.path.insert(0,sys.argv[1]); "
            "from opti_robot.nav2_adapter import request_factory_path; "
            "r=request_factory_path(sys.argv[2],sys.argv[3],json.loads(sys.argv[4]),"
            "json.loads(sys.argv[5]),sys.argv[6],float(sys.argv[7]),timeout_s=float(sys.argv[8]),"
            "domain_id=int(sys.argv[9])); print(json.dumps(r)); sys.exit(0 if r['accepted'] else 1)")
    argv = [str(ROOT / ".venv/bin/python"), "-c", code, str(ROOT / "src"),
            files["map_yaml"], files["params_yaml"], json.dumps(start.tolist()),
            json.dumps(goal.tolist()), str(output / "request"), str(planning_radius),
            str(timeout_s), str(domain_id)]
    started = time.monotonic()
    with (output / "client.log").open("w") as log:
        # The inner adapter owns deadlines/cleanup. Do not interrupt it with an
        # earlier outer timeout and abandon independently sessioned ROS children.
        process = subprocess.run(argv, cwd=ROOT, env=env, stdout=log, stderr=subprocess.STDOUT)
    summary_path = output / "request/adapter_summary.json"
    result = json.loads(summary_path.read_text()) if summary_path.is_file() else {
        "accepted": False, "failure": "Adapter did not write a summary; inspect client.log"}
    result.update({"client_exit_code": process.returncode, "client_wall_s": time.monotonic()-started,
                   "map": files, "obstacle_rectangles": obstacles,
                   "physical_execution_validated": False})
    if process.returncode != 0:
        result["accepted"] = False
    (output / "navigation_plan.json").write_text(json.dumps(result, indent=2) + "\n")
    return result


@dataclass(frozen=True)
class QPCommand:
    accepted: bool
    v_m_s: float
    omega_rad_s: float
    diagnostics: dict


class BaseVelocityQP:
    """Two-variable convex velocity projection at a fixed measured axle pose.

    Hard inequalities: command speed/yaw/slew/wheel-rate and static circle
    separation CBF linearized at the present pose. No slack or retry relaxation.
    A zero fallback is a requested stop; physical braking must be measured.
    """
    def __init__(self, obstacle_rectangles, *, envelope_radius_m, clearance_margin_m=0.15,
                 limits=None, wheel_radius_m=0.1, wheel_separation_m=0.53,
                 bounds_xy_m=None):
        import cvxpy as cp
        self.cp, self.rectangles = cp, _rectangles(obstacle_rectangles)
        self.bounds = None if bounds_xy_m is None else _finite_vector(bounds_xy_m,4,"map bounds")
        if self.bounds is not None and np.any(self.bounds[:2]>=self.bounds[2:]):
            raise ValueError("Invalid map bounds")
        self.radius = _positive(envelope_radius_m, "envelope radius") + float(clearance_margin_m)
        if not math.isfinite(clearance_margin_m) or clearance_margin_m < 0:
            raise ValueError("Clearance margin must be finite and nonnegative")
        self.limits = dict(load_config()["navigation"]["limits"])
        if limits:
            unknown = set(limits)-set(self.limits)
            if unknown:
                raise ValueError(f"Unsupported QP options: {sorted(unknown)}")
            self.limits.update(limits)
        for name,value in self.limits.items():
            _positive(value,name)
        self.wheel_radius = _positive(wheel_radius_m,"wheel radius")
        self.wheel_separation = _positive(wheel_separation_m,"wheel separation")
        self.previous = np.zeros(2)
        self.u = cp.Variable(2)
        self.reference = cp.Parameter(2)
        self.previous_parameter = cp.Parameter(2)
        self.slew = cp.Parameter(2,nonneg=True)
        self.bound = cp.Parameter(2,nonneg=True)
        barrier_count = len(self.rectangles)+(4 if self.bounds is not None else 0)
        self.a = cp.Parameter((max(1,barrier_count),2))
        self.b = cp.Parameter(max(1,barrier_count))
        l = self.limits
        wheel_matrix = np.array([[1.,-self.wheel_separation/2],
                                 [1.,self.wheel_separation/2]]) / self.wheel_radius
        objective = (l["tracking_weight_v"]*cp.square(self.u[0]-self.reference[0])
                     + l["tracking_weight_omega"]*cp.square(self.u[1]-self.reference[1])
                     + l["smoothing_weight"]*cp.sum_squares(self.u-self.previous_parameter))
        self.problem = cp.Problem(cp.Minimize(objective),[
            cp.abs(self.u)<=self.bound, cp.abs(self.u-self.previous_parameter)<=self.slew,
            cp.abs(wheel_matrix@self.u)<=l["wheel_rate_cap_rad_s"], self.a@self.u>=self.b])

    def step(self, axle_pose_xy_yaw, reference_v_omega, dt_s):
        pose = _finite_vector(axle_pose_xy_yaw,3,"axle pose")
        reference = _finite_vector(reference_v_omega,2,"reference")
        dt = _positive(dt_s,"QP command interval")
        l = self.limits
        heading = np.array([math.cos(pose[2]),math.sin(pose[2])])
        a,b,h = [],[],[]
        for rectangle in self.rectangles:
            distance,gradient = point_rectangle_distance(pose[:2],rectangle)
            gap = distance-self.radius
            a.append([float(gradient@heading),0.])
            b.append(-l["barrier_rate_s_inv"]*gap)
            h.append(gap)
        if self.bounds is not None:
            xmin,ymin,xmax,ymax = self.bounds
            gaps = [pose[0]-xmin-self.radius,pose[1]-ymin-self.radius,
                    xmax-pose[0]-self.radius,ymax-pose[1]-self.radius]
            for gradient,gap in zip(([1.,0.],[0.,1.],[-1.,0.],[0.,-1.]),gaps):
                a.append([float(np.asarray(gradient)@heading),0.])
                b.append(-l["barrier_rate_s_inv"]*gap)
                h.append(float(gap))
        if not a:
            a,b = [[0.,0.]],[-1.]
        self.a.value, self.b.value = np.array(a),np.array(b)
        self.reference.value, self.previous_parameter.value = reference,self.previous
        self.slew.value = np.array([l["command_accel_m_s2"],l["command_alpha_rad_s2"]])*dt
        self.bound.value = [l["vmax_m_s"],l["omega_max_rad_s"]]
        started = time.monotonic()
        failure = None
        try:
            self.problem.solve(solver=self.cp.OSQP,warm_start=True,verbose=False,
                               eps_abs=l["solver_eps_abs"],eps_rel=l["solver_eps_rel"],
                               max_iter=int(l["solver_max_iter"]),polish=True)
            command = np.asarray(self.u.value,dtype=float) if self.u.value is not None else None
            accepted = (self.problem.status in ("optimal","optimal_inaccurate") and command is not None
                        and command.shape==(2,) and np.isfinite(command).all())
            if accepted:
                # Never admit a solver status without numerical hard-constraint checks.
                tolerance = max(5e-5,5*l["solver_eps_abs"])
                bound_error = float(np.max(np.abs(command)-self.bound.value))
                slew_error = float(np.max(np.abs(command-self.previous)-self.slew.value))
                barrier_error = float(np.max(self.b.value-self.a.value@command))
                rates = wheel_rates_for_twist(*command,radius_m=self.wheel_radius,
                                               separation_m=self.wheel_separation)
                wheel_error = max(abs(v) for v in rates.values())-l["wheel_rate_cap_rad_s"]
                accepted = max(bound_error,slew_error,barrier_error,wheel_error)<=tolerance
                if not accepted:
                    failure = "Numerical hard constraint violation"
            else:
                failure = f"QP status {self.problem.status}"
        except Exception as exc:
            accepted,command = False,None
            failure = f"{type(exc).__name__}: {exc}"
        diagnostic = {"solver_status": self.problem.status,"solve_wall_s": time.monotonic()-started,
                      "reference_v_omega":reference.tolist(),"previous_command":self.previous.tolist(),
                      "minimum_static_circle_gap_m":min(h) if h else None,
                      "command_interval_s":dt,"obstacle_count":len(self.rectangles),
                      "failure":failure,"constraint_scope":"command velocity/slew; static circle barriers",
                      "actual_acceleration_or_braking_guaranteed":False}
        if not accepted:
            self.previous = np.zeros(2)
            return QPCommand(False,0.,0.,diagnostic)
        self.previous = command.copy()
        diagnostic["command_v_omega"] = command.tolist()
        return QPCommand(True,float(command[0]),float(command[1]),diagnostic)


class PolylineReference:
    """GT axle pure-pursuit reference; feed its output through BaseVelocityQP.

    Enter final heading alignment inside position_tolerance_m and retain it
    until position_release_tolerance_m (default: entry + 3 mm). This avoids
    switching between opposite heading goals during small wheel slip. It is
    reference-state hysteresis only: the caller's measured arrival/rest gates
    are unchanged, and exceeding the release distance resumes path tracking.
    """
    def __init__(self, points_xy, goal_yaw=0.0, limits=None):
        points = np.asarray(points_xy,dtype=float)
        if points.ndim!=2 or points.shape[1]!=2 or len(points)<2 or not np.isfinite(points).all():
            raise ValueError("Polyline must contain >=2 finite XY points")
        points = points[np.r_[True,np.linalg.norm(np.diff(points,axis=0),axis=1)>1e-8]]
        if len(points)<2:
            raise ValueError("Polyline has zero length")
        self.points,self.vectors = points,np.diff(points,axis=0)
        self.lengths = np.linalg.norm(self.vectors,axis=1)
        self.starts = np.r_[0.,np.cumsum(self.lengths)]
        self.progress,self.goal_yaw = 0.,float(goal_yaw)
        self.limits = dict(load_config()["navigation"]["limits"])
        if limits:
            self.limits.update(limits)
        entry = _positive(self.limits["position_tolerance_m"], "position tolerance")
        release = _positive(self.limits.get("position_release_tolerance_m", entry + .003),
                            "position release tolerance")
        if release <= entry:
            raise ValueError("Position release tolerance must exceed alignment entry tolerance")
        self.limits["position_tolerance_m"] = entry
        self.limits["position_release_tolerance_m"] = release
        self.arrival_latched = False

    def reference(self, axle_pose_xy_yaw):
        pose = _finite_vector(axle_pose_xy_yaw,3,"axle pose")
        l = self.limits
        goal_distance = float(np.linalg.norm(self.points[-1]-pose[:2]))
        if self.arrival_latched:
            if goal_distance >= l["position_release_tolerance_m"]:
                self.arrival_latched = False
        elif goal_distance <= l["position_tolerance_m"]:
            self.arrival_latched = True
        if self.arrival_latched:
            error = wrap_angle(self.goal_yaw-pose[2])
            omega = 0. if abs(error)<=l["yaw_tolerance_rad"] else np.clip(l["heading_gain_s_inv"]*error,
                              -l["omega_max_rad_s"],l["omega_max_rad_s"])
            return np.array([0.,omega]),{"phase":"align_or_rest","goal_distance_m":goal_distance,
                                       "goal_yaw_error_rad":error,"arrival_latched":True,
                                       "position_release_tolerance_m":l["position_release_tolerance_m"]}
        fractions = np.clip(np.sum((pose[:2]-self.points[:-1])*self.vectors,axis=1)/self.lengths**2,0.,1.)
        projected = self.points[:-1]+fractions[:,None]*self.vectors
        distances = np.linalg.norm(projected-pose[:2],axis=1)
        candidate_progress = self.starts[:-1]+fractions*self.lengths
        allowed = (candidate_progress>=self.progress-.1)&(candidate_progress<=self.progress+.7)
        distances[~allowed] = np.inf
        if not np.isfinite(distances).any():
            raise RuntimeError("No local path projection; reject rather than jump to another segment")
        index = int(np.argmin(distances))
        self.progress = max(self.progress,float(candidate_progress[index]))
        target_progress = min(self.progress+l["lookahead_m"],float(self.starts[-1]))
        segment = min(int(np.searchsorted(self.starts,target_progress,side="right")-1),len(self.lengths)-1)
        target = self.points[segment]+self.vectors[segment]*((target_progress-self.starts[segment])/self.lengths[segment])
        delta = target-pose[:2]
        error = wrap_angle(math.atan2(delta[1],delta[0])-pose[2])
        omega = float(np.clip(l["heading_gain_s_inv"]*error,-l["omega_max_rad_s"],l["omega_max_rad_s"]))
        v = min(l["vmax_m_s"],l["position_gain_s_inv"]*goal_distance)*max(0.,math.cos(error))
        if abs(error)>.65:
            v=0.
        return np.array([v,omega]),{"phase":"track","progress_m":self.progress,
                                   "tracking_distance_m":float(distances[index]),
                                   "goal_distance_m":goal_distance,"heading_error_rad":error,
                                   "lookahead_target_xy_m":target.tolist(),"arrival_latched":False,
                                   "position_release_tolerance_m":l["position_release_tolerance_m"]}


def straight_dock_reference(base_pose_xy_yaw, target_base_xy_yaw, *, backwards=False,
                            limits=None):
    """Reference for separate yaw-zero undock/dock; no clearance exemption/gate.

    Circle-map table barriers cannot certify this close workcell motion. Caller
    must retain per-step real contact/load checks and measured 3D swept geometry.
    """
    pose,target = _finite_vector(base_pose_xy_yaw,3,"base pose"),_finite_vector(target_base_xy_yaw,3,"dock target")
    l = dict(load_config()["navigation"]["limits"])
    if limits:
        l.update(limits)
    error = target[:2]-pose[:2]
    distance = float(np.linalg.norm(error))
    yaw_error = wrap_angle(target[2]-pose[2])
    forward = np.array([math.cos(target[2]),math.sin(target[2])])
    lateral = np.array([-forward[1],forward[0]])
    along=float(error@forward)
    cross=float((pose[:2]-target[:2])@lateral)
    if abs(cross)>.02:
        raise RuntimeError("Straight docking corridor lateral error exceeds 20 mm; reposition outside the workcell")
    # Fixed workcell orientation: never chase the final XY point by spinning
    # the carried tray next to a table. Correct only a small heading offset.
    if abs(along)<=.004:
        v=0.; angular_error=yaw_error
    else:
        direction=-1. if backwards else 1.
        if direction*along<-.004:raise RuntimeError("Straight dock target was overshot")
        heading_offset=float(np.clip(-direction*2.*cross,-.02,.02))
        angular_error=wrap_angle(target[2]+heading_offset-pose[2])
        v=direction*min(l["vmax_m_s"],l["position_gain_s_inv"]*abs(along))
        if abs(yaw_error)>.04:v=0.
    omega=float(np.clip(l["heading_gain_s_inv"]*angular_error,-l["omega_max_rad_s"],l["omega_max_rad_s"]))
    if abs(along)<=.004 and abs(yaw_error)<=l["yaw_tolerance_rad"]:omega=0.
    return np.array([v,omega]),{"goal_distance_m":distance,"goal_yaw_error_rad":yaw_error,
                               "lateral_error_m":cross,"along_error_m":along,
                               "backwards":bool(backwards),"physical_rest_confirmed":False}
