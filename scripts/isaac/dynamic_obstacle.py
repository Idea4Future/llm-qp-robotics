"""Opt-in external kinematic cart for an Isaac stop/wait/resume experiment.

Importing this module never launches Kit or steps physics. Author the cart
before World.reset, initialize its native view after reset, then select a
scenario from the freshly computed Nav2 polyline. setup() performs ONE logged
external-obstacle placement before navigation; step() subsequently uses native
PhysX kinematic targets, not dynamic-body pose edits. No robot, gripper, tray or
transported-object transform is changed by this helper.

    metadata = add_dynamic_cart(world.stage)             # before reset
    cart = ScriptedCart(world.stage, metadata)
    world.reset()
    cart.initialize()                                   # after reset
    scenario = plan_cart_interruption(path_xy, radius_m=carry_radius,
                                      static_rectangles=map_rectangles)
    setup_event = cart.setup(scenario, world.current_time,
                            robot_axle_xy_m=actual_axle_xy)
    # Each physics step, BEFORE world.step:
    event = cart.step(world.current_time, release=stop_has_been_measured)
    world.step(render=False)
    actual_cart = cart.measure(world.current_time)       # evaluation only

The caller owns LiDAR/QP behavior, contact/stop/load checks and event storage.
Do not pass scenario/cart GT positions to the robot's local controller. A
kinematic object follows an imposed script and has effectively infinite mass
in contacts; the authored10kg is not a force-driven cart-motion model. The
0.6m-high collider intersects the project's0.4m planar LiDAR scan. This does
not validate obstacles that the scan plane misses, or dynamic safety.

Native API checked against installed omni.physics.tensors107.3.26 api.py:
RigidBodyView.set_kinematic_targets(data, indices), transforms XYZ+XYZW.
The helper has not been tested in a live PhysX scene by its author.
"""
from __future__ import annotations

import copy
import math

import numpy as np


def _vector(value, n, label):
    result = np.asarray(value, dtype=float)
    if result.shape != (n,) or not np.isfinite(result).all():
        raise ValueError(f"{label} must contain {n} finite values")
    return result


def _positive(value, label):
    if isinstance(value, bool) or not math.isfinite(float(value)) or float(value) <= 0:
        raise ValueError(f"{label} must be finite and positive")
    return float(value)


def _external_path(path):
    if not isinstance(path, str) or not path.startswith("/World/ProjectDynamic"):
        raise ValueError("Only a new /World/ProjectDynamic* external obstacle is permitted")
    if any(not part.isidentifier() for part in path.strip("/").split("/")):
        raise ValueError("Invalid external-obstacle USD path")
    return path


def add_dynamic_cart(stage, prim_path="/World/ProjectDynamicCart", *,
                     initial_position_m=(4.5, -2., .3), size_m=(.45, .55, .6),
                     mass_kg=10., friction=.6):
    """Author one native kinematic rigid box BEFORE physics initialization.

    Initial parking is scenario setup outside the task workcells; the caller
    must verify that this position is clear in its actual warehouse. It is not
    inserted into the static map automatically. Original assets stay intact.
    """
    from pxr import Gf, PhysxSchema, UsdGeom, UsdPhysics, UsdShade

    path = _external_path(prim_path)
    position = _vector(initial_position_m, 3, "initial_position_m")
    size = _vector(size_m, 3, "size_m")
    mass_kg = _positive(mass_kg, "mass_kg")
    if np.any(size <= 0) or not math.isfinite(float(friction)) or friction < 0:
        raise ValueError("Cart size must be positive and friction nonnegative")
    if not math.isclose(UsdGeom.GetStageMetersPerUnit(stage), 1., abs_tol=1e-8):
        raise ValueError("This helper requires meter stage units")
    if UsdGeom.GetStageUpAxis(stage) != UsdGeom.Tokens.z:
        raise ValueError("This helper requires a Z-up stage")
    material_path = path + "Material"
    if stage.GetPrimAtPath(path).IsValid() or stage.GetPrimAtPath(material_path).IsValid():
        raise ValueError("External cart or its material already exists")

    root = UsdGeom.Xform.Define(stage, path)
    root.AddTranslateOp().Set(Gf.Vec3d(*position))
    root.AddOrientOp().Set(Gf.Quatf(1.))
    body = UsdPhysics.RigidBodyAPI.Apply(root.GetPrim())
    body.CreateRigidBodyEnabledAttr(True)
    body.CreateKinematicEnabledAttr(True)
    PhysxSchema.PhysxRigidBodyAPI.Apply(root.GetPrim()).CreateDisableGravityAttr(True)
    mass = UsdPhysics.MassAPI.Apply(root.GetPrim())
    mass.CreateMassAttr(mass_kg)
    mass.CreateCenterOfMassAttr(Gf.Vec3f(0., 0., 0.))
    inertia = mass_kg / 12. * np.array([size[1]**2 + size[2]**2,
                                       size[0]**2 + size[2]**2,
                                       size[0]**2 + size[1]**2])
    mass.CreateDiagonalInertiaAttr(Gf.Vec3f(*inertia))

    material = UsdShade.Material.Define(stage, material_path)
    material_api = UsdPhysics.MaterialAPI.Apply(material.GetPrim())
    material_api.CreateStaticFrictionAttr(float(friction))
    material_api.CreateDynamicFrictionAttr(float(friction))
    material_api.CreateRestitutionAttr(0.)
    geom = UsdGeom.Cube.Define(stage, path + "/collision_box")
    geom.CreateSizeAttr(1.)
    geom.AddScaleOp().Set(Gf.Vec3d(*size))
    geom.CreateDisplayColorAttr([Gf.Vec3f(.83, .28, .07)])
    UsdPhysics.CollisionAPI.Apply(geom.GetPrim()).CreateCollisionEnabledAttr(True)
    collision = PhysxSchema.PhysxCollisionAPI.Apply(geom.GetPrim())
    collision.CreateContactOffsetAttr(.002)
    collision.CreateRestOffsetAttr(0.)
    UsdShade.MaterialBindingAPI.Apply(geom.GetPrim()).Bind(material, materialPurpose="physics")
    return {"prim_path": path, "collision_path": str(geom.GetPath()),
            "initial_position_m": position.tolist(), "size_m": size.tolist(),
            "authored_mass_kg": mass_kg, "inertia_kg_m2": inertia.tolist(),
            "kinematic": True, "gravity_disabled": True,
            "friction_assumption": float(friction), "restitution": 0.,
            "scope": "optional project box-cart approximation; externally scripted obstacle",
            "motion_model": "native PhysX kinematic target; effective infinite contact mass",
            "static_map_policy": "separate dynamic obstacle; do not include its parked/setup pose as an immutable map obstacle",
            "physical_model_status": "authored_not_live_validated",
            "robot_object_pose_edits": False}


def _rectangles(items):
    rectangles = []
    for item in items or []:
        bounds = item["xy_bounds_m"] if isinstance(item, dict) else item
        value = _vector(bounds, 4, "rectangle")
        if np.any(value[:2] >= value[2:]):
            raise ValueError("Static rectangle minima must be below maxima")
        rectangles.append(value)
    return rectangles


def _clear_center(point, rectangle, radius):
    low, high = rectangle[:2], rectangle[2:]
    return np.linalg.norm(point - np.clip(point, low, high)) > radius


def plan_cart_interruption(path_xy, *, radius_m, margin_m=.15,
                           cart_size_m=(.45, .55, .6), static_rectangles=None,
                           map_bounds_xy_m=None, progress_fraction=.5,
                           minimum_end_distance_m=.8, retreat_clearance_m=.15,
                           maximum_speed_m_s=.12):
    """Select a path blocker with a conservatively clear sideways retreat.

    Tries the requested arclength fraction and nearby middle fractions, both
    retreat sides. Conservative cart circumcircle tests its entire swept
    center segment against supplied rectangles and optional map bounds. It
    rejects rather than placing/retreating through a known wall. The supplied
    rectangles must represent relevant cart-height obstacles, not merely floor
    AABBs. Full3D collision and contact checks remain the caller's responsibility.
    """
    points = np.asarray(path_xy, dtype=float)
    if points.ndim != 2 or points.shape[1] != 2 or len(points) < 2 or not np.isfinite(points).all():
        raise ValueError("path_xy must contain >=2 finite XY points")
    points = points[np.r_[True, np.linalg.norm(np.diff(points, axis=0), axis=1) > 1e-8]]
    if len(points) < 2:
        raise ValueError("Path has zero length")
    radius_m = _positive(radius_m, "radius_m")
    margin_m = _positive(margin_m, "margin_m")
    maximum_speed = _positive(maximum_speed_m_s, "maximum_speed_m_s")
    minimum_end = _positive(minimum_end_distance_m, "minimum_end_distance_m")
    retreat_clearance = _positive(retreat_clearance_m, "retreat_clearance_m")
    size = _vector(cart_size_m, 3, "cart_size_m")
    if np.any(size <= 0) or not 0 < float(progress_fraction) < 1:
        raise ValueError("Cart size/fraction is invalid")
    rectangles = _rectangles(static_rectangles)
    bounds = None if map_bounds_xy_m is None else _rectangles([map_bounds_xy_m])[0]
    vectors = np.diff(points, axis=0)
    lengths = np.linalg.norm(vectors, axis=1)
    starts = np.r_[0., np.cumsum(lengths)]
    cart_radius = float(np.linalg.norm(size[:2]) / 2.)
    displacement = radius_m + margin_m + cart_radius + retreat_clearance
    # Cubic smoothstep has peak velocity1.5*distance/duration, zero endpoint speed.
    duration = 1.5 * displacement / maximum_speed
    fractions = list(dict.fromkeys([float(progress_fraction), .4, .6, .3, .7, .5]))
    for fraction in fractions:
        progress = fraction * starts[-1]
        if min(progress, starts[-1] - progress) < minimum_end:
            continue
        i = min(int(np.searchsorted(starts, progress, side="right") - 1), len(lengths) - 1)
        direction = vectors[i] / lengths[i]
        center = points[i] + (progress - starts[i]) * direction
        normal = np.array([-direction[1], direction[0]])
        for sign in (1, -1):
            clear = center + sign * displacement * normal
            samples = center + np.linspace(0., 1., max(2, math.ceil(displacement / .025) + 1))[:, None] * (clear - center)
            clearance = cart_radius + .025  # conservatism beyond sampled swept path
            if bounds is not None and (np.any(samples - clearance < bounds[:2]) or np.any(samples + clearance > bounds[2:])):
                continue
            if any(not _clear_center(sample, rectangle, clearance) for rectangle in rectangles for sample in samples):
                continue
            # Also ensure the final cart is clear of OTHER nearby path segments.
            t = np.clip(np.sum((clear - points[:-1]) * vectors, axis=1) / lengths**2, 0., 1.)
            distance_to_path = float(np.min(np.linalg.norm(clear - (points[:-1] + t[:, None] * vectors), axis=1)))
            if distance_to_path < radius_m + margin_m + cart_radius:
                continue
            return {"blocking_position_m": [*center, size[2] / 2.],
                    "clear_position_m": [*clear, size[2] / 2.],
                    "yaw_rad": math.atan2(direction[1], direction[0]),
                    "size_m": size.tolist(), "path_progress_m": float(progress),
                    "path_fraction": fraction, "retreat_side": sign,
                    "clear_duration_s": float(duration), "maximum_script_speed_m_s": maximum_speed,
                    "maximum_script_acceleration_m_s2": float(6. * displacement / duration**2),
                    "cart_circumradius_m": cart_radius, "robot_radius_m": radius_m,
                    "margin_m": margin_m, "final_distance_to_path_m": distance_to_path,
                    "static_rectangles_checked": len(rectangles),
                    "setup": "one logged external-obstacle placement BEFORE navigation",
                    "release": "caller triggers after actual stop/detection; cart GT never enters robot QP",
                    "scope": "conservative2D scenario selection, not a physical success/safety result"}
    raise ValueError("No path-middle cart placement has a clear sideways retreat; use another scenario/open bay")


class ScriptedCart:
    """Own only one external kinematic body; caller owns each physics step."""

    def __init__(self, stage, metadata, *, maximum_update_gap_s=.05):
        from pxr import UsdPhysics

        self.metadata = copy.deepcopy(metadata)
        self.path = _external_path(metadata["prim_path"])
        prim = stage.GetPrimAtPath(self.path)
        if not prim.IsValid() or not prim.HasAPI(UsdPhysics.RigidBodyAPI):
            raise ValueError("External cart rigid body missing")
        if not UsdPhysics.RigidBodyAPI(prim).GetKinematicEnabledAttr().Get():
            raise ValueError("Cart must be a native kinematic body")
        self.maximum_gap = _positive(maximum_update_gap_s, "maximum_update_gap_s")
        self._view = self._simulation_view = None
        self._scenario = self._release_time = self._last_time = self._last_position = None

    def initialize(self, simulation_view=None, *, body_view=None):
        """Bind AFTER World.reset; optional injected body_view is for CPU mocks."""
        if self._view is not None:
            raise RuntimeError("Cart already initialized")
        if body_view is None:
            if simulation_view is None:
                import omni.physics.tensors as tensors
                simulation_view = tensors.create_simulation_view("numpy")
                simulation_view.set_subspace_roots("/")
            simulation_view.initialize_kinematic_bodies()
            body_view = simulation_view.create_rigid_body_view(self.path)
        if body_view.count != 1 or list(body_view.prim_paths) != [self.path]:
            raise RuntimeError("Native view must contain exactly this external cart")
        self._simulation_view, self._view = simulation_view, body_view

    @staticmethod
    def _transform(position, yaw):
        return np.array([[*position, 0., 0., math.sin(yaw / 2.), math.cos(yaw / 2.)]], dtype=np.float32)

    def setup(self, scenario, simulation_time, *, robot_axle_xy_m=None):
        """One explicit scenario setup jump of the external cart, before navigation.

        No hidden jump is allowed in step(). Provide actual robot axle position
        to reject an initially overlapping cart, not as a robot-control input.
        """
        if self._view is None or self._scenario is not None:
            raise RuntimeError("Initialize first and perform cart setup only once")
        scenario = copy.deepcopy(scenario)
        start = _vector(scenario["blocking_position_m"], 3, "blocking_position_m")
        end = _vector(scenario["clear_position_m"], 3, "clear_position_m")
        duration = _positive(scenario["clear_duration_s"], "clear_duration_s")
        speed = _positive(scenario["maximum_script_speed_m_s"], "maximum_script_speed_m_s")
        yaw, now = float(scenario["yaw_rad"]), float(simulation_time)
        if not np.isfinite([yaw, now]).all() or now < 0 or abs(start[2] - end[2]) > 1e-9:
            raise ValueError("Invalid cart time/yaw or nonplanar retreat")
        if not np.allclose(_vector(scenario["size_m"], 3, "scenario size"), self.metadata["size_m"]):
            raise ValueError("Scenario cart size differs from authored collider")
        if 1.5 * np.linalg.norm(end - start) / duration > speed + 1e-9:
            raise ValueError("Clear duration would exceed the specified script speed")
        if robot_axle_xy_m is not None:
            robot = _vector(robot_axle_xy_m, 2, "robot_axle_xy_m")
            exclusion = scenario["robot_radius_m"] + scenario["margin_m"] + scenario["cart_circumradius_m"]
            if np.linalg.norm(start[:2] - robot) <= exclusion:
                raise ValueError("Scenario setup would overlap the admitted robot circle")
        transforms = self._transform(start, yaw)
        indices = np.array([0], dtype=np.uint32)
        self._view.set_transforms(transforms, indices)  # external-obstacle setup ONLY
        self._view.set_kinematic_targets(transforms, indices)
        self._scenario, self._last_time, self._last_position = scenario, now, start
        return {"type": "external_obstacle_scenario_setup", "timestamp_s": now,
                "prim_path": self.path, "from_parked_position_m": self.metadata["initial_position_m"],
                "to_position_m": start.tolist(), "scenario": copy.deepcopy(scenario),
                "external_pose_jump": True, "robot_or_load_pose_changed": False}

    def step(self, simulation_time, *, release=False):
        """Set the NEXT PhysX step's cart target; does not call World.step.

        The first release=True starts the cubic sideways retreat. A false flag
        afterwards does not rewind motion. Missing/backwards/sparse timestamps
        raise; the caller should stop the robot on any scenario/control failure.
        """
        if self._scenario is None:
            raise RuntimeError("Cart scenario has not been set up")
        now = float(simulation_time)
        if not math.isfinite(now) or now < self._last_time - 1e-9:
            raise ValueError("Invalid or backwards cart timestamp")
        dt = now - self._last_time
        if dt > self.maximum_gap + 1e-9:
            raise RuntimeError("Cart target updates were missing; reject a discontinuous obstacle script")
        if not isinstance(release, (bool, np.bool_)):
            raise ValueError("release must be boolean")
        began = bool(release) and self._release_time is None
        if began:
            self._release_time = now
        start = np.asarray(self._scenario["blocking_position_m"])
        end = np.asarray(self._scenario["clear_position_m"])
        u = 0. if self._release_time is None else np.clip((now - self._release_time) / self._scenario["clear_duration_s"], 0., 1.)
        blend = 3. * u**2 - 2. * u**3
        target = start + blend * (end - start)
        if np.linalg.norm(target - self._last_position) > self._scenario["maximum_script_speed_m_s"] * dt + 1e-7:
            raise RuntimeError("Cart target exceeded admitted script speed")
        self._view.set_kinematic_targets(self._transform(target, self._scenario["yaw_rad"]), np.array([0], dtype=np.uint32))
        self._last_position, self._last_time = target, now
        state = "blocking" if self._release_time is None else "clear" if u >= 1. else "clearing"
        return {"timestamp_s": now, "prim_path": self.path, "state": state,
                "position_target_m": target.tolist(), "release_started": began,
                "release_timestamp_s": self._release_time, "motion_fraction": float(u),
                "native_target_only": True, "external_obstacle_gt_for_evaluation_only": True}

    def measure(self, simulation_time):
        """Read actual native cart state for evaluation/logging, never robot control."""
        if self._view is None:
            raise RuntimeError("Cart native view is missing")
        transforms = np.asarray(self._view.get_transforms(), dtype=float)
        velocities = np.asarray(self._view.get_velocities(), dtype=float)
        if transforms.shape != (1, 7) or velocities.shape != (1, 6) or not np.isfinite(np.r_[transforms.ravel(), velocities.ravel()]).all():
            raise RuntimeError("Native cart measurement invalid")
        return {"timestamp_s": float(simulation_time), "prim_path": self.path,
                "position_m": transforms[0, :3].tolist(), "quaternion_xyzw": transforms[0, 3:].tolist(),
                "linear_velocity_m_s": velocities[0, :3].tolist(), "angular_velocity_rad_s": velocities[0, 3:].tolist(),
                "source": "native PhysX external-obstacle GT; evaluation only"}
