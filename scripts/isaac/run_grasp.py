#!/usr/bin/env python3
"""Stationary contact-only grasp gate; no object attachment, QP, LLM or navigation."""
from __future__ import annotations
import argparse
import json
import sys
import time
import traceback
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
parser.add_argument("--output", type=Path, required=True)
parser.add_argument("--probe-only", action="store_true", help="Only settle and inspect native FK/Jacobian")
parser.add_argument("--scene", choices=("minimal", "warehouse"), default="minimal")
parser.add_argument("--video", action="store_true", help="Record actual rendered physics frames at 25 simulation fps")
parser.add_argument("--inspect-gripper", action="store_true", help="Close-up camera and finger visual/physics diagnostics")
from isaaclab.app import AppLauncher
AppLauncher.add_app_launcher_args(parser)
parser.set_defaults(device="cpu")
args = parser.parse_args()
args.enable_cameras = True
out = args.output.resolve()
out.mkdir(parents=True, exist_ok=False)
app = AppLauncher(dict(vars(args)), multi_gpu=False, width=640, height=480, fast_shutdown=True).app
samples = []
video_writer = None
summary = {"passed": False, "scope": "stationary contact grasp with known initial pose; no navigation/QP/LLM", "probe_only": args.probe_only}
started = time.monotonic()

try:
    import numpy as np
    from scipy.spatial.transform import Rotation
    from PIL import Image
    from pxr import Gf, UsdGeom, UsdLux, UsdPhysics
    from isaacsim.core.api import World
    from isaacsim.core.api.materials import PhysicsMaterial
    from isaacsim.core.prims import SingleRigidPrim, RigidPrim
    from isaacsim.core.utils.extensions import enable_extension
    enable_extension("isaacsim.sensors.camera")
    from isaacsim.sensors.camera import Camera
    from vendor_task import load_standalone_task
    from grasp_scene import add_finger_contact_geometry, add_stationary_grasp_scene, GRASP_FRAME_IN_WRIST_M
    from validate_grasp import validate
    sys.path.insert(0, str(ROOT / "third_party/rby1_isaac/src"))
    from gripper_servers.rb_gripper import RbGripperServer
    VendorTask = load_standalone_task(ROOT)

    class GraspTask(VendorTask):
        def _add_gripper_to_robot(self, stage):
            super()._add_gripper_to_robot(stage)
            for side in ("left", "right"):
                stage.GetPrimAtPath(self._gripper_side_root_path(side)).RemoveAPI(UsdPhysics.ArticulationRootAPI)
            self.pad_metadata = add_finger_contact_geometry(stage)

        def set_up_scene(self, scene):
            super().set_up_scene(scene)
            self.scene_metadata = add_stationary_grasp_scene(scene.stage)
            if args.scene == "warehouse":
                from isaacsim.storage.native import get_assets_root_path
                from isaacsim.core.utils.stage import add_reference_to_stage
                asset_root = get_assets_root_path()
                if not asset_root:
                    raise RuntimeError("Warehouse asset server unavailable")
                url = asset_root + "/Isaac/Environments/Simple_Warehouse/warehouse.usd"
                add_reference_to_stage(url, "/World/Warehouse")
                if not scene.stage.GetPrimAtPath("/World/Warehouse").GetChildren():
                    raise RuntimeError("Warehouse did not compose")
                UsdGeom.Imageable(scene.stage.GetPrimAtPath("/World/defaultGroundPlane")).MakeInvisible()
                self.scene_metadata["warehouse_url"] = url
            ground_material = PhysicsMaterial("/World/ProjectGroundMaterial", static_friction=0.8, dynamic_friction=0.8, restitution=0.0)
            scene.get_object("default_ground_plane").apply_physics_material(ground_material)
            self.robot.set_solver_position_iteration_count(16)
            self.robot.set_solver_velocity_iteration_count(4)

        def _set_default_robot_state(self):
            super()._set_default_robot_state()
            q = np.zeros(self.num_joints)
            init = {"left": [-.4, .15, 0, -1, 0, 1.4, 0],
                    "right": [-.4591, -.3124, -.1156, -1.4535, -.2808, 1.9189, -.0087]}
            for side, values in init.items():
                for i, value in enumerate(values):
                    q[self.dof_names.index(f"{side}_arm_{i}")] = value
                letter = side[0]
                q[self.dof_names.index(f"gripper_finger_{letter}1")] = -.05
                q[self.dof_names.index(f"gripper_finger_{letter}2")] = .05
            # Initial state only; all later motion uses physics and bounded effort.
            self.robot.set_joint_positions(q)
            self.reference = q[self.joint_indices].copy()
            self.finger_target = -.05
            indices = np.array(self.gripper_command_indices)
            self.robot._articulation_view.set_gains(kps=np.array([[2000., 2000.]]), kds=np.array([[20., 20.]]), joint_indices=indices)
            self.robot._articulation_view.set_max_efforts(np.array([[12., 12.]]), joint_indices=indices)

        def _build_reference_target(self, simulation_time):
            mode = np.zeros(len(self.model_config.cpp_joint_names), dtype=bool)
            mode[:self.model_config.mobility_dof] = True
            target = self.reference.copy()
            target[:self.model_config.mobility_dof] = 0.0
            self.robot._articulation_view.set_joint_position_targets(
                np.array([[-.05, self.finger_target]]), joint_indices=self.gripper_command_indices)
            # Sim 5.1 gravity wrapper indexes a generalized floating-base vector.
            gravity = self.robot._articulation_view.get_generalized_gravity_forces(
                joint_indices=np.arange(self.num_joints, dtype=np.int64) + 6)[0]
            self.pd_controller.update_feedforward_term(gravity[self.joint_indices])
            return mode, target

    world = World(physics_dt=.002, rendering_dt=.04, backend="numpy", device="cpu",
                  stage_units_in_meters=1.0, sim_params={"use_gpu_pipeline": False, "enable_scene_query_support": True})
    task = GraspTask(robot_model="a", gripper_enabled=True, gripper_server=RbGripperServer())
    world.add_task(task)
    world.reset()
    robot, view = task.robot, task.robot._articulation_view
    obj_path, table_path = task.scene_metadata["object_prim"], task.scene_metadata["table_top_prim"]
    obj = SingleRigidPrim(obj_path, name="test_object")
    obj.initialize()
    base = SingleRigidPrim("/World/RBY1/base", name="measured_base")
    base.initialize()
    fingers = RigidPrim("/World/RBY1/right_gripper/ee_finger_r[12]",
                       name="finger_contacts", reset_xform_properties=False,
                       contact_filter_prim_paths_expr=[obj_path], max_contact_count=64)
    fingers.initialize()
    support = RigidPrim(obj_path, name="object_support_contact", reset_xform_properties=False,
                        contact_filter_prim_paths_expr=[table_path], max_contact_count=32)
    support.initialize()
    wrist = robot.link_right_arm_6
    arm_indices = np.array([robot.dof_names.index(f"right_arm_{i}") for i in range(7)])
    cpp_indices = np.array([task.model_config.cpp_joint_names.index(f"right_arm_{i}") for i in range(7)])
    body_index = view.get_body_index("link_right_arm_6")
    limits = view.get_dof_limits()[0, arm_indices]
    offset = np.array(GRASP_FRAME_IN_WRIST_M)
    source = np.array([.4, -.3, .83])
    destination = np.array([.38, -.43, .83])
    hover, lifted, dst_hover = source + [0, 0, .14], source + [0, 0, .12], destination + [0, 0, .12]

    def tcp_pose():
        p, quat = wrist.get_world_pose()
        r = Rotation.from_quat(np.asarray(quat)[[1, 2, 3, 0]]).as_matrix()
        return np.asarray(p) + r @ offset, r

    def command_tcp(target):
        tcp, rotation = tcp_pose()
        error = np.r_[target - tcp, Rotation.from_matrix(rotation.T).as_rotvec()]
        jacobians = view.get_jacobians()
        if jacobians.shape[-1] != robot.num_dof + 6:
            raise RuntimeError(f"Unexpected floating Jacobian shape: {jacobians.shape}")
        jac = jacobians[0, body_index][:, arm_indices + 6].copy()
        r = rotation @ offset
        skew = np.array([[0, -r[2], r[1]], [r[2], 0, -r[0]], [-r[1], r[0], 0]])
        jac[:3] -= skew @ jac[3:]
        jac[3:] *= .3
        error[3:] *= .3
        dq = jac.T @ np.linalg.solve(jac @ jac.T + .015**2 * np.eye(6), error)
        dq *= min(1.0, .012 / max(np.abs(dq).max(), 1e-9))
        current = robot.get_joint_positions()[arm_indices]
        task.reference[cpp_indices] = np.clip(current + dq, limits[:, 0] + .005, limits[:, 1] - .005)

    light = UsdLux.DomeLight.Define(world.stage, "/World/Light")
    light.CreateIntensityAttr(1200.)
    camera = Camera("/World/GraspCamera", resolution=(640, 480))
    camera.initialize()
    eye, focus = Gf.Vec3d(2.3, -2.6, 1.8), Gf.Vec3d(.25, -.2, .85)
    if args.inspect_gripper:
        eye, focus = Gf.Vec3d(.85, -1.0, 1.02), Gf.Vec3d(.4, -.3, .91)
    quat = Gf.Matrix4d().SetLookAt(eye, focus, Gf.Vec3d(0, 0, 1)).GetInverse().ExtractRotationQuat()
    camera.set_world_pose(np.array(eye), np.array([quat.GetReal(), *quat.GetImaginary()]), camera_axes="usd")
    camera.prim.GetAttribute("focalLength").Set(28.)
    camera.set_clipping_range(0.005, 100.0)
    if not args.headless:
        from isaacsim.core.utils.viewports import set_active_viewport_camera
        set_active_viewport_camera(camera.prim_path)
    world.set_simulation_dt(.002, .04)
    initial_tcp, initial_rot = tcp_pose()
    summary.update(pad_geometry=task.pad_metadata, scene=task.scene_metadata,
                   initial_tcp=initial_tcp.tolist(), initial_rotation=initial_rot.tolist(),
                   joint_names=list(robot.dof_names), arm_indices=arm_indices.tolist(),
                   jacobian_shape=list(view.get_jacobians().shape), wrist_body_index=int(body_index),
                   gripper_leader_force_cap_n=12., friction_is_assumption=True,
                   self_collision_enabled=False, object_attachment=False, gravity_enabled=True,
                   object_pose_overrides_during_execution=False)
    if args.video:
        import imageio.v2 as imageio
        video_writer = imageio.get_writer(str(out / "replay.mp4"), fps=25, codec="libx264", quality=7)
        summary["video_timing"] = "25 fps in simulation time; does not represent measured wall-time speed"
    (out / "setup.json").write_text(json.dumps(summary, indent=2) + "\n")
    phases = [("settle", 2., hover, hover, -.05),
              ("approach", 3., hover, source, -.05),
              ("close", 2., source, source, -.015),
              ("lift", 3., source, lifted, -.015),
              ("hold", 3., lifted, lifted, -.015),
              ("transfer", 3., lifted, dst_hover, -.015),
              ("lower", 3., dst_hover, destination, -.015),
              ("open", 1., destination, destination, -.05),
              ("retreat", 2., destination, destination + [0, 0, .14], -.05),
              ("place_settle", 1., destination + [0, 0, .14], destination + [0, 0, .14], -.05)]
    if args.probe_only:
        phases = phases[:1]
    hold_rel, hold_rot = None, None
    base_initial = np.asarray(base.get_world_pose()[0])
    for phase, seconds, begin, end, finger in phases:
        print("Phase:", phase, flush=True)
        for step in range(round(seconds / .002)):
            t = min(1., (step + 1) * .002 / seconds)
            target = begin + (end - begin) * t * t * (3 - 2 * t)
            task.finger_target = finger
            if step % 10 == 0:
                command_tcp(target)
            world.step(render=False)
            if step % 20 == 19:
                world.render()
                if video_writer is not None:
                    rgba = camera.get_rgba()
                    if rgba is not None and rgba.size:
                        video_writer.append_data(np.asarray(rgba)[..., :3].astype(np.uint8))
            tcp, rot = tcp_pose()
            p, quat = obj.get_world_pose()
            objrot = Rotation.from_quat(np.asarray(quat)[[1, 2, 3, 0]]).as_matrix()
            min_z = float(p[2] - np.abs(objrot[2]) @ (np.array([.04, .05, .06]) / 2))
            force = np.linalg.norm(fingers.get_contact_force_matrix(dt=.002)[:, 0, :], axis=1)
            table_force = float(np.linalg.norm(support.get_contact_force_matrix(dt=.002)))
            bp, bq = base.get_world_pose()
            up_z = float(1 - 2 * (bq[1]**2 + bq[2]**2))
            rel = rot.T @ (p - tcp)
            rel_rot = rot.T @ objrot
            if phase == "hold" and hold_rel is None:
                hold_rel, hold_rot = rel.copy(), rel_rot.copy()
            row = {"time": float(world.current_time), "phase": phase, "object": np.asarray(p).tolist(),
                   "tcp": tcp.tolist(), "tcp_error_m": float(np.linalg.norm(target - tcp)),
                   "tcp_rotation_error_rad": float(np.linalg.norm(Rotation.from_matrix(rot).as_rotvec())),
                   "clearance_m": min_z - .8, "finger_normal_force_n": force.tolist(), "support_force_n": table_force,
                   "relative_translation_drift_m": 0. if hold_rel is None else float(np.linalg.norm(rel - hold_rel)),
                   "relative_rotation_drift_rad": 0. if hold_rot is None else float(np.linalg.norm(Rotation.from_matrix(rel_rot @ hold_rot.T).as_rotvec())),
                   "base_xy_drift_m": float(np.linalg.norm(np.asarray(bp)[:2] - base_initial[:2])), "base_up_z": up_z,
                   "object_velocity": np.asarray(obj.get_linear_velocity()).tolist(),
                   "object_angular_velocity": np.asarray(obj.get_angular_velocity()).tolist()}
            samples.append(row)
            if not np.isfinite(np.r_[p, tcp, force, robot.get_joint_positions()]).all():
                raise RuntimeError("Non-finite physical state")
            if row["base_xy_drift_m"] > .03 or up_z < .98:
                raise RuntimeError("Base exceeded stationary drift/tilt bounds")
        rgba = camera.get_rgba()
        if rgba is not None and rgba.size:
            Image.fromarray(np.asarray(rgba)[..., :3].astype(np.uint8)).save(out / f"{phase}.png")
        if args.inspect_gripper:
            from pxr import Usd
            cache = UsdGeom.XformCache()
            inspection = []
            for i in (1, 2):
                body_path = f"/World/RBY1/right_gripper/ee_finger_r{i}"
                body = world.stage.GetPrimAtPath(body_path)
                entries = []
                for prim in Usd.PrimRange(body, Usd.TraverseInstanceProxies()):
                    if prim.IsA(UsdGeom.Mesh) or prim.IsA(UsdGeom.Cube):
                        imageable = UsdGeom.Imageable(prim)
                        entries.append({"path": str(prim.GetPath()), "type": prim.GetTypeName(),
                                        "visibility": str(imageable.ComputeVisibility()),
                                        "purpose": str(imageable.ComputePurpose()),
                                        "world_transform": np.asarray(cache.GetLocalToWorldTransform(prim)).tolist()})
                positions, orientations = fingers.get_world_poses()
                body_index_in_view = fingers.prim_paths.index(body_path)
                position, orientation = positions[body_index_in_view], orientations[body_index_in_view]
                inspection.append({"body": body_path, "position": np.asarray(position).tolist(),
                                   "quaternion_wxyz": np.asarray(orientation).tolist(), "geometries": entries})
            (out / f"{phase}_finger_visuals.json").write_text(json.dumps(inspection, indent=2) + "\n")
        if phase in ("settle", "approach", "lower") and samples[-1]["tcp_error_m"] > .004:
            raise RuntimeError(f"{phase} TCP did not reach target: {samples[-1]['tcp_error_m']}")
        if phase == "close" and min(samples[-1]["finger_normal_force_n"]) < .02:
            raise RuntimeError("No bilateral finger/object contact before lift")
    if args.probe_only:
        summary["passed"] = True
        summary["scope"] = "native FK/Jacobian and stationary pose probe only; not a grasp"
    else:
        evaluated = validate(samples, task.scene_metadata)
        summary.update(metrics=evaluated["metrics"], checks=evaluated["checks"],
                       passed=evaluated["passed"], validation_scope=evaluated["scope"])
except Exception as error:
    traceback.print_exc()
    summary.update(passed=False, error=str(error), error_type=type(error).__name__)
finally:
    try:
        summary.update(wall_seconds=time.monotonic() - started, physical_samples=len(samples))
        # Persist the raw measurements first, including after a gate failure.
        (out / "state_samples.json").write_text(json.dumps(samples) + "\n")
        (out / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
        print(json.dumps(summary), flush=True)
    finally:
        try:
            if video_writer is not None:
                video_writer.close()
        finally:
            app.close(skip_cleanup=True)
