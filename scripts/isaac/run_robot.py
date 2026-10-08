#!/usr/bin/env python3
"""RB-Y1 installation demo: physical joint control, RGB image, optional ROS clock.

This is a runtime/integration check, not a grasp, navigation or LLM benchmark.
The pinned vendor task keeps self collisions disabled; see the source manifest.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import subprocess
import sys
import time
import traceback
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
parser.add_argument("--output", type=Path, required=True)
parser.add_argument("--scene", choices=("minimal", "warehouse"), default="warehouse")
parser.add_argument("--seconds", type=float, default=5.0)
parser.add_argument("--ros2", action="store_true")
parser.add_argument("--check-ros-clock", action="store_true", help="Run and verify a separate system Humble listener")
parser.add_argument("--realtime", action="store_true", help="Pace physics to wall time for interactive viewing")
from isaaclab.app import AppLauncher

AppLauncher.add_app_launcher_args(parser)
parser.set_defaults(device="cpu")
args = parser.parse_args()
args.enable_cameras = True
if args.check_ros_clock:
    args.ros2 = True
if not math.isfinite(args.seconds) or args.seconds < 1.0:
    parser.error("--seconds must be finite and at least 1 second")
out = args.output.resolve()
out.mkdir(parents=True, exist_ok=False)
started = time.monotonic()
app = AppLauncher(dict(vars(args)), multi_gpu=False, width=640, height=480, fast_shutdown=True).app
listener = None
listener_log = None

try:
    import numpy as np
    from PIL import Image
    from pxr import Gf, UsdGeom, UsdLux, UsdPhysics
    from isaacsim.core.prims import SingleRigidPrim
    from isaacsim.core.api import World
    from isaacsim.core.utils.stage import add_reference_to_stage
    from isaacsim.core.utils.viewports import set_camera_view, set_active_viewport_camera
    from isaacsim.core.utils.extensions import enable_extension
    enable_extension("isaacsim.sensors.camera")
    from isaacsim.sensors.camera import Camera

    sys.path.insert(0, str(ROOT / "third_party/rby1_isaac/src"))
    from vendor_task import load_standalone_task
    RBY1Task = load_standalone_task(ROOT)
    from gripper_servers.rb_gripper import RbGripperServer

    class StationaryDemo(RBY1Task):
        """Keep wheels still and move only the unloaded right wrist gently."""
        def _add_gripper_to_robot(self, stage):
            super()._add_gripper_to_robot(stage)
            # The grippers are joined into one robot articulation. Their
            # standalone root APIs would otherwise produce PhysX errors.
            for side in ("left", "right"):
                stage.GetPrimAtPath(self._gripper_side_root_path(side)).RemoveAPI(UsdPhysics.ArticulationRootAPI)

        def _build_reference_target(self, simulation_time):
            wheel_velocity_mode = np.zeros(len(self.model_config.cpp_joint_names), dtype=bool)
            wheel_velocity_mode[:self.model_config.mobility_dof] = True
            target = np.zeros(len(self.model_config.cpp_joint_names), dtype=float)
            i = self.model_config.cpp_joint_names.index("right_arm_6")
            target[i] = 0.12 * (1.0 - math.cos(2.0 * math.pi * simulation_time / 4.0)) / 2.0
            return wheel_velocity_mode, target

    world = World(physics_dt=0.002, rendering_dt=1.0 / 25.0, backend="numpy", device="cpu",
                  stage_units_in_meters=1.0, sim_params={"use_gpu_pipeline": False, "enable_scene_query_support": True})
    gripper = RbGripperServer()  # no .start(): no network server for this demo
    task = StationaryDemo(robot_model="a", gripper_enabled=True, gripper_server=gripper)
    world.add_task(task)
    # Task composition creates its own local ground and robot.
    world.reset()
    warehouse_url = None
    if args.scene == "warehouse":
        from isaacsim.storage.native import get_assets_root_path
        asset_root = get_assets_root_path()
        if not asset_root:
            raise RuntimeError("Isaac asset server unavailable; --scene minimal works with local assets")
        warehouse_url = asset_root + "/Isaac/Environments/Simple_Warehouse/warehouse.usd"
        add_reference_to_stage(warehouse_url, "/World/Warehouse")
        warehouse = world.stage.GetPrimAtPath("/World/Warehouse")
        if not warehouse.IsValid() or not warehouse.GetChildren():
            raise RuntimeError("Warehouse reference did not compose")
        # Hide the vendor grid's visual mesh, retaining its collision plane.
        # Warehouse collision/map validation is a later task-integration gate.
        UsdGeom.Imageable(world.stage.GetPrimAtPath("/World/defaultGroundPlane")).MakeInvisible()
    light = UsdLux.DomeLight.Define(world.stage, "/World/DemoLight")
    light.CreateIntensityAttr(1000.0)
    eye, focus = Gf.Vec3d(4.0, -5.0, 2.8), Gf.Vec3d(0.0, 0.0, 0.8)
    set_camera_view(eye=np.array(eye), target=np.array(focus))
    # Process every explicitly rendered frame; don't assume Kit's UI-loop
    # rateLimitFrequency equals the physical render cadence.
    camera = Camera(prim_path="/World/DemoCamera", resolution=(640, 480))
    camera.initialize()
    # USD focalLength and aperture use the same tenths-of-stage-unit convention.
    # Author the attribute directly to avoid the wrapper's stage-unit conversion.
    camera.prim.GetAttribute("focalLength").Set(24.0)
    rotation = Gf.Matrix4d().SetLookAt(eye, focus, Gf.Vec3d(0, 0, 1)).GetInverse().ExtractRotationQuat()
    quat = np.array([rotation.GetReal(), *rotation.GetImaginary()])
    camera.set_world_pose(position=np.array(eye), orientation=quat, camera_axes="usd")
    camera.set_clipping_range(0.01, 1000.0)
    if not args.headless:
        set_active_viewport_camera(camera.prim_path)

    if args.ros2:
        from isaacsim.core.utils.extensions import enable_extension
        enable_extension("isaacsim.ros2.bridge")
        app.update()
        import omni.graph.core as og
        og.Controller.edit(
            {"graph_path": "/World/ClockGraph", "evaluator_name": "execution"},
            {og.Controller.Keys.CREATE_NODES: [
                 ("Tick", "omni.graph.action.OnPlaybackTick"),
                 ("Time", "isaacsim.core.nodes.IsaacReadSimulationTime"),
                 ("Context", "isaacsim.ros2.bridge.ROS2Context"),
                 ("Clock", "isaacsim.ros2.bridge.ROS2PublishClock")],
             og.Controller.Keys.CONNECT: [
                 ("Tick.outputs:tick", "Clock.inputs:execIn"),
                 ("Context.outputs:context", "Clock.inputs:context"),
                 ("Time.outputs:simulationTime", "Clock.inputs:timeStamp")],
             og.Controller.Keys.SET_VALUES: [("Clock.inputs:topicName", "/clock"),
                                             ("Context.inputs:useDomainIDEnvVar", True)]},
        )
    if args.check_ros_clock:
        listener_env = dict(os.environ)
        for key in ("PYTHONPATH", "PYTHONHOME", "LD_LIBRARY_PATH", "AMENT_PREFIX_PATH", "COLCON_PREFIX_PATH", "CMAKE_PREFIX_PATH"):
            listener_env.pop(key, None)
        listener_env.update(ROS_DOMAIN_ID=os.environ.get("ROS_DOMAIN_ID", "172"), ROS_LOCALHOST_ONLY="1", RMW_IMPLEMENTATION="rmw_fastrtps_cpp")
        listener_log = (out / "ros_listener.log").open("w")
        listener = subprocess.Popen(
            ["/bin/bash", "--noprofile", "--norc", "-c",
             'source /opt/ros/humble/setup.bash; exec /usr/bin/python3 "$@"',
             "isaac-clock-listener", str(ROOT / "scripts/isaac/ros_clock_listener.py"),
             "--output", str(out / "ros_clock"), "--timeout-s", "45"],
            env=listener_env, stdout=listener_log, stderr=subprocess.STDOUT,
        )
        time.sleep(1.0)  # Give the independent node time to discover the publisher.

    # Extension/asset initialization can reset the timeline's display rate.
    # Reapply the public timing API after all startup app.update() calls.
    world.set_simulation_dt(physics_dt=0.002, rendering_dt=1.0 / 25.0)
    robot = task.robot
    base = SingleRigidPrim(prim_path="/World/RBY1/base", name="measured_base")
    base.initialize()
    names = list(robot.dof_names)
    idx = names.index("right_arm_6")
    q0 = np.asarray(robot.get_joint_positions()).copy()
    p0, _ = base.get_world_pose()
    samples = []
    gpu_memory_samples = []
    simulation_start = float(world.current_time)
    run_started = time.monotonic()
    steps = round(args.seconds / 0.002)
    for step in range(steps):
        if not app.is_running():
            raise RuntimeError("Application closed before demo completed")
        world.step(render=False)
        if step % 20 == 19:
            world.render()
        if step % 500 == 0:
            try:
                measured = subprocess.check_output(["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"], text=True, timeout=3)
                gpu_memory_samples.append(int(measured.strip().splitlines()[0]))
            except (OSError, ValueError, subprocess.SubprocessError):
                pass
        if step % 50 == 0:
            q = np.asarray(robot.get_joint_positions())
            pos, orientation = base.get_world_pose()
            up_z = float(1.0 - 2.0 * (orientation[1] ** 2 + orientation[2] ** 2))
            samples.append({"time": world.current_time, "wrist": float(q[idx]), "base": np.asarray(pos).tolist(), "base_up_z": up_z})
            if not np.isfinite(q).all() or not np.isfinite(pos).all() or not np.isfinite(orientation).all():
                raise RuntimeError("Non-finite robot state")
            if np.linalg.norm(np.asarray(pos)[:2] - np.asarray(p0)[:2]) > 0.03 or up_z < math.cos(math.radians(10)) or not -0.02 < pos[2] < 0.06:
                raise RuntimeError("Stationary robot moved or tilted beyond the demo bounds")
        if args.realtime:
            time.sleep(max(0.0, (step + 1) * 0.002 - (time.monotonic() - run_started)))
    rgba = camera.get_rgba()
    if rgba is None or rgba.size == 0:
        raise RuntimeError("Camera did not produce RGB pixels")
    rgb = np.asarray(rgba)[..., :3].astype(np.uint8)
    if float(rgb.std()) < 2.0:
        raise RuntimeError("Rendered image is blank or nearly uniform")
    Image.fromarray(rgb).save(out / "camera.png")
    wrist_range = float(np.ptp([x["wrist"] for x in samples]))
    if wrist_range < 0.01:
        raise RuntimeError(f"Wrist did not follow a changing physical command: range={wrist_range}")
    loop_sim_seconds = float(world.current_time) - simulation_start
    if abs(loop_sim_seconds - steps * 0.002) > 0.002:
        raise RuntimeError(f"Unexpected physics time advance: {loop_sim_seconds}")
    ros_received = None
    if listener is not None:
        exit_code = listener.wait(timeout=50)
        ros_received = exit_code == 0
        if not ros_received:
            raise RuntimeError(f"External ROS clock check failed: exit {exit_code}; inspect ros_listener.log")
    summary = {
        "passed": True, "scope": "installation demo only; no grasp/navigation/LLM validation",
        "scene": args.scene, "warehouse_url": warehouse_url, "headless": args.headless,
        "physics_device": "cpu", "physics_dt": 0.002,
        "stage_time_codes_per_second": world.stage.GetTimeCodesPerSecond(),
        "simulation_seconds": float(world.current_time),
        "loop_simulation_seconds": loop_sim_seconds,
        "execution_and_capture_wall_seconds": time.monotonic() - run_started,
        "total_wall_seconds": time.monotonic() - started,
        "joint_names": names, "wrist_motion_range_rad": wrist_range,
        "initial_base": np.asarray(p0).tolist(), "final_base": np.asarray(base.get_world_pose()[0]).tolist(),
        "max_sampled_base_xy_drift_m": max(float(np.linalg.norm(np.asarray(x["base"])[:2] - np.asarray(p0)[:2])) for x in samples),
        "minimum_sampled_base_up_z": min(x["base_up_z"] for x in samples),
        "rgb_shape": list(rgb.shape), "rgb_std": float(rgb.std()),
        "ros_clock_requested": args.ros2, "self_collisions_enabled": False,
        "external_humble_clock_received": ros_received,
        "vendor_source_commit": "2417a2b2c83bc80b3ad605ab14f4d508d90089a9",
        "vendor_adapter": "unused SDK-only wire-codec import omitted in memory; upstream sources unchanged",
        "articulation_root_pose_note": "Vendor 1.31m refers to the PhysX-selected root link, not the base-link height",
        "gripper_nested_root_apis_removed": True,
        "ground_note": "Vendor collision plane retained; visual grid hidden in warehouse. Warehouse collision/map not validated for navigation.",
        "total_gpu_memory_sampled_mib": gpu_memory_samples,
        "gpu_memory_note": "Whole GPU including desktop/other processes; samples during loop, not a continuous peak measurement",
    }
    (out / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    (out / "state_samples.json").write_text(json.dumps(samples, indent=2) + "\n")
    print(json.dumps(summary), flush=True)
except Exception as error:
    traceback.print_exc()
    (out / "summary.json").write_text(json.dumps({"passed": False, "error": str(error), "error_type": type(error).__name__}, indent=2) + "\n")
finally:
    if listener is not None and listener.poll() is None:
        listener.terminate()
        try:
            listener.wait(timeout=5)
        except subprocess.TimeoutExpired:
            listener.kill()
            listener.wait()
    if listener_log is not None:
        listener_log.close()
    # Supported immediate shutdown: no asynchronous writers remain. The shell
    # entry point independently checks the saved result to preserve failure status.
    app.close(skip_cleanup=True)
