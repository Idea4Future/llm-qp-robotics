#!/usr/bin/env python3
"""Isaac Lab 2.3.0: one CPU-physics drop and one actual RTX RGB camera.

Run with the isolated Isaac Python environment, never the core ROS/MuJoCo venv::

    python scripts/isaac/smoke_lab.py --output results/isaac_lab_smoke_01

The output directory must be new. This primitive scene uses no downloaded USD
assets or learning/task packages. CPU physics still needs an RTX GPU for RGB.
This is a startup/contact/render smoke test, not a robot grasp or realtime test.

API references (official v2.3.0 source): run_rigid_object.py, camera/camera.py,
camera/camera_cfg.py, sim/simulation_cfg.py, and app/app_launcher.py in IsaacLab.
"""

from __future__ import annotations

import argparse
import importlib.metadata
import json
import math
from pathlib import Path
import struct
import sys
import time
import traceback
import zlib


def _write_png(path: Path, rgb) -> None:
    """Write uint8 HxWx3 RGB with the standard library, without extra installs."""
    height, width, channels = rgb.shape
    if channels != 3 or str(rgb.dtype) != "uint8":
        raise ValueError("PNG input must be uint8 HxWx3 RGB")

    def chunk(kind: bytes, data: bytes) -> bytes:
        checksum = zlib.crc32(kind + data) & 0xFFFFFFFF
        return struct.pack(">I", len(data)) + kind + data + struct.pack(">I", checksum)

    # Each PNG scanline starts with filter type 0 (no filtering).
    scanlines = b"".join(b"\x00" + row.tobytes() for row in rgb)
    payload = (
        b"\x89PNG\r\n\x1a\n"
        + chunk(b"IHDR", struct.pack(">IIBBBBB", width, height, 8, 2, 0, 0, 0))
        + chunk(b"IDAT", zlib.compress(scanlines))
        + chunk(b"IEND", b"")
    )
    with path.open("xb") as stream:
        stream.write(payload)


def _json_safe(value):
    """Keep a diagnostic summary serializable even when a state becomes NaN."""
    if isinstance(value, dict):
        return {key: _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def _run_trial(args, simulation_app, output: Path) -> dict:
    # Isaac/Omniverse imports must follow AppLauncher initialization.
    import numpy as np
    import torch

    import isaaclab.sim as sim_utils
    from isaaclab.assets import RigidObject, RigidObjectCfg
    from isaaclab.sensors import Camera, CameraCfg

    scene_started = time.perf_counter()
    dt = 1.0 / 120.0
    cube_size = 0.06
    initial_height = 0.50
    sim = sim_utils.SimulationContext(
        sim_utils.SimulationCfg(
            device="cpu",
            dt=dt,
            render_interval=1,
            gravity=(0.0, 0.0, -9.81),
            render=sim_utils.RenderCfg(rendering_mode="performance"),
        )
    )

    # A static primitive slab avoids GroundPlaneCfg's default remote USD asset.
    ground = sim_utils.CuboidCfg(
        size=(2.0, 2.0, 0.02),
        collision_props=sim_utils.CollisionPropertiesCfg(),
        physics_material=sim_utils.RigidBodyMaterialCfg(
            static_friction=0.8, dynamic_friction=0.6, restitution=0.0
        ),
        visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.20, 0.28, 0.34)),
    )
    ground.func("/World/Ground", ground, translation=(0.0, 0.0, -0.01))
    light = sim_utils.DomeLightCfg(intensity=1500.0, color=(1.0, 1.0, 1.0))
    light.func("/World/Light", light)
    cube = RigidObject(
        RigidObjectCfg(
            prim_path="/World/DropCube",
            spawn=sim_utils.CuboidCfg(
                size=(cube_size, cube_size, cube_size),
                rigid_props=sim_utils.RigidBodyPropertiesCfg(),
                mass_props=sim_utils.MassPropertiesCfg(mass=0.100),
                collision_props=sim_utils.CollisionPropertiesCfg(),
                physics_material=sim_utils.RigidBodyMaterialCfg(
                    static_friction=0.8, dynamic_friction=0.6, restitution=0.0
                ),
                visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.9, 0.08, 0.02)),
            ),
            init_state=RigidObjectCfg.InitialStateCfg(pos=(0.0, 0.0, initial_height)),
        )
    )
    camera = Camera(
        CameraCfg(
            prim_path="/World/Camera",
            update_period=0.0,
            height=args.camera_height,
            width=args.camera_width,
            data_types=["rgb"],
            spawn=sim_utils.PinholeCameraCfg(
                focal_length=18.0,
                horizontal_aperture=24.0,
                clipping_range=(0.01, 10.0),
            ),
        )
    )
    sim.reset()
    cube.reset()
    camera.set_world_poses_from_view(
        eyes=torch.tensor([[0.55, -0.55, 0.45]], dtype=torch.float32, device=sim.device),
        targets=torch.tensor([[0.0, 0.0, 0.20]], dtype=torch.float32, device=sim.device),
    )
    camera.reset()
    cube.update(0.0)
    measured_mass_kg = float(cube.root_physx_view.get_masses().detach().cpu().numpy().reshape(-1)[0])
    states = [cube.data.root_state_w[0].detach().cpu().numpy().copy()]
    setup_wall_s = time.perf_counter() - scene_started

    loop_started = time.perf_counter()
    completed_steps = 0
    for step in range(args.steps):
        if not simulation_app.is_running() or not sim.is_playing():
            raise RuntimeError(f"Simulator stopped or paused after {completed_steps} steps")
        cube.write_data_to_sim()
        sim.step(render=True)
        cube.update(dt)
        camera.update(dt)
        states.append(cube.data.root_state_w[0].detach().cpu().numpy().copy())
        completed_steps = step + 1
    loop_wall_s = time.perf_counter() - loop_started

    # Render-only refresh: these frames do not add physical timesteps or reset
    # the object. This also gives the RTX annotator a bounded warmup opportunity.
    for _ in range(8):
        sim.render()
    camera.update(0.0, force_recompute=True)
    raw_rgb = camera.data.output["rgb"][0].detach().cpu().numpy()
    rgb = np.ascontiguousarray(raw_rgb[..., :3])
    image_shape_ok = rgb.shape == (args.camera_height, args.camera_width, 3)
    rgb_finite = bool(np.isfinite(rgb).all())
    rgb_uint8 = str(rgb.dtype) == "uint8"
    rgb_nonempty = bool(rgb.size and rgb_finite and float(rgb.max()) > 0.0)
    rgb_varied = bool(rgb_nonempty and float(rgb.std()) > 1.0)
    if image_shape_ok and rgb_finite and rgb_uint8:
        _write_png(output / "rgb.png", rgb)

    states_array = np.asarray(states)
    np.savez_compressed(
        output / "states.npz",
        time_s=np.arange(len(states), dtype=np.float64) * dt,
        root_state_w=states_array,
    )
    finite_states = bool(np.isfinite(states_array).all())
    z = states_array[:, 2]
    linear_speeds = np.linalg.norm(states_array[:, 7:10], axis=1)
    angular_speeds = np.linalg.norm(states_array[:, 10:13], axis=1)
    tail_count = int(round(0.5 / dt))
    tail = states_array[-tail_count:]
    expected_rest_height = cube_size / 2.0
    drop_m = float(z[0] - np.min(z))
    checks = {
        "completed_requested_steps": completed_steps == args.steps,
        "cpu_physics": str(sim.device) == "cpu",
        "mass_is_100_g": bool(math.isfinite(measured_mass_kg) and abs(measured_mass_kg - 0.100) < 1e-6),
        "finite_object_states": finite_states,
        "started_above_floor": bool(z[0] > 0.40),
        "gravity_drop_over_0_20_m": bool(drop_m > 0.20),
        "observed_downward_velocity": bool(np.min(states_array[:, 9]) < -0.10),
        "no_center_below_floor": bool(np.min(z) >= 0.0),
        "settled_height_last_0_5_s": bool(
            np.max(np.abs(tail[:, 2] - expected_rest_height)) <= 0.015
        ),
        "settled_linear_speed_last_0_5_s": bool(np.max(linear_speeds[-tail_count:]) < 0.05),
        "settled_angular_speed_last_0_5_s": bool(np.max(angular_speeds[-tail_count:]) < 0.20),
        "rgb_expected_shape": image_shape_ok,
        "rgb_finite": rgb_finite,
        "rgb_uint8": rgb_uint8,
        "rgb_nonempty": rgb_nonempty,
        "rgb_varied": rgb_varied,
        "png_written": (output / "rgb.png").is_file(),
    }
    return {
        "accepted": all(checks.values()),
        "checks": checks,
        "physics": {
            "device": str(sim.device),
            "configured_mass_kg": 0.100,
            "measured_mass_kg": measured_mass_kg,
            "cube_size_m": [cube_size] * 3,
            "ground_top_z_m": 0.0,
            "gravity_m_s2": [0.0, 0.0, -9.81],
            "dt_s": dt,
            "steps": completed_steps,
            "sim_duration_s": completed_steps * dt,
            "initial_root_state_w": states_array[0].tolist(),
            "final_root_state_w": states_array[-1].tolist(),
            "root_state_columns": ["x", "y", "z", "qw", "qx", "qy", "qz", "vx", "vy", "vz", "wx", "wy", "wz"],
            "max_drop_m": drop_m,
            "min_center_z_m": float(np.min(z)),
            "min_vertical_velocity_m_s": float(np.min(states_array[:, 9])),
            "settle_window_s": 0.5,
            "tail_max_linear_speed_m_s": float(np.max(linear_speeds[-tail_count:])),
            "tail_max_angular_speed_rad_s": float(np.max(angular_speeds[-tail_count:])),
        },
        "camera": {
            "type": "Isaac Lab Camera / RTX RGB",
            "rendering_uses_gpu": True,
            "shape": list(rgb.shape),
            "dtype": str(rgb.dtype),
            "minimum": float(rgb.min()) if rgb.size else None,
            "maximum": float(rgb.max()) if rgb.size else None,
            "std": float(rgb.std()) if rgb.size else None,
            "frame_counter": camera.frame.detach().cpu().tolist(),
            "post_physics_render_only_frames": 8,
        },
        "timing": {"scene_setup_wall_s": setup_wall_s, "physics_and_render_loop_wall_s": loop_wall_s},
        "artifacts": {"rgb": str(output / "rgb.png"), "states": str(output / "states.npz")},
    }


def main() -> int:
    # Use the documented launcher parser, but reserve the mandatory output
    # first so initialization failures can also leave a diagnostic summary.
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True, help="New output directory; overwriting is refused")
    parser.add_argument("--steps", type=int, default=240, help="CPU physics steps (120..240)")
    parser.add_argument("--camera-width", type=int, default=320, choices=(320, 512))
    parser.add_argument("--camera-height", type=int, default=240, choices=(240, 384))
    parser.add_argument("--gui", action="store_true", help="Show GUI; headless is the default")
    preliminary, _ = parser.parse_known_args()
    if not 120 <= preliminary.steps <= 240:
        parser.error("--steps must be between 120 and 240")
    if (preliminary.camera_width, preliminary.camera_height) not in ((320, 240), (512, 384)):
        parser.error("Use camera resolution 320x240 or 512x384")
    output = preliminary.output.expanduser().resolve()
    try:
        output.mkdir(parents=True, exist_ok=False)
    except FileExistsError:
        parser.error(f"Output already exists: {output}")

    started = time.perf_counter()
    simulation_app = None
    summary = {
        "accepted": False,
        "experiment": "Isaac Lab primitive CPU physics and RTX camera smoke",
        "output": str(output),
        "limitations": [
            "A single freefall/contact/render smoke; no robot or gripper validation.",
            "CPU physics does not remove the RTX GPU requirement for camera rendering.",
            "Measured wall time is not a realtime guarantee or warehouse performance result.",
            "Floor settling is checked from pose and velocity; contact forces are not measured.",
        ],
    }
    try:
        from isaaclab.app import AppLauncher

        AppLauncher.add_app_launcher_args(parser)
        parser.set_defaults(headless=True, enable_cameras=True, device="cpu", livestream=0)
        args = parser.parse_args()
        if args.device != "cpu":
            raise ValueError("This smoke requires --device cpu; RTX rendering still uses GPU")
        args.headless = not args.gui
        args.enable_cameras = True
        summary["versions"] = {}
        for distribution in ("isaacsim", "isaaclab", "torch"):
            try:
                summary["versions"][distribution] = importlib.metadata.version(distribution)
            except importlib.metadata.PackageNotFoundError:
                summary["versions"][distribution] = None
        launch_started = time.perf_counter()
        # AppLauncher mutates its input mapping. Use Isaac's standard fast
        # shutdown after synchronous result writes; the parent checks exit code.
        launcher = AppLauncher(
            dict(vars(args)),
            multi_gpu=False,
            fast_shutdown=True,
            width=args.camera_width,
            height=args.camera_height,
        )
        simulation_app = launcher.app
        summary["app_launch_wall_s"] = time.perf_counter() - launch_started
        summary["headless"] = args.headless
        summary["driver_check_override_requested"] = "verifyDriverVersion" in args.kit_args
        summary.update(_run_trial(args, simulation_app, output))
    except Exception as error:
        summary["accepted"] = False
        summary["error"] = {"type": type(error).__name__, "message": str(error)}
        traceback.print_exc()
    finally:
        # Preserve physics/render evidence before native extension shutdown;
        # a native crash cannot be caught by Python's exception handler.
        summary["app_close_requested"] = simulation_app is not None
        summary["wall_elapsed_s"] = time.perf_counter() - started
        with (output / "summary.json").open("x", encoding="utf-8") as stream:
            json.dump(_json_safe(summary), stream, ensure_ascii=False, indent=2, allow_nan=False)
            stream.write("\n")
        close_started = time.perf_counter()
        if simulation_app is not None:
            try:
                # All required pixels are already synchronously saved. There
                # is no asynchronous Replicator writer to wait for.
                simulation_app.close(skip_cleanup=True)
                summary["app_closed"] = True
            except Exception as error:
                summary["accepted"] = False
                summary["app_closed"] = False
                summary["close_error"] = {"type": type(error).__name__, "message": str(error)}
                traceback.print_exc()
        else:
            summary["app_closed"] = False
        summary["app_close_wall_s"] = time.perf_counter() - close_started
        summary["wall_elapsed_s"] = time.perf_counter() - started
        with (output / "summary.json").open("w", encoding="utf-8") as stream:
            json.dump(_json_safe(summary), stream, ensure_ascii=False, indent=2, allow_nan=False)
            stream.write("\n")
        print(json.dumps({"accepted": summary["accepted"], "summary": str(output / "summary.json")}), flush=True)
    return 0 if summary["accepted"] else 1


if __name__ == "__main__":
    sys.exit(main())
