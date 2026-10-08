"""Verify the installed Humble planner stack without physical robot control.

Run after sourcing /opt/ros/humble/setup.bash, using /usr/bin/python3. The
synthetic map, fixed map->base_link TF, wall clock and 0.2 m radius are probe
assumptions. This does not exercise sensors, odometry, localization, wheel
commands, a local controller, the RB-Y1 footprint, or collision avoidance.
"""

import argparse
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import shutil
import signal
import subprocess
import sys
import time
import traceback


ROOT = Path(__file__).resolve().parents[1]
START = (2.0, 5.0)
GOAL = (8.0, 5.0)
PROBE_RADIUS = 0.2


def process_group_members(group_id):
    """Inspect only process identity/state; never collect command/environment."""
    members = []
    for entry in Path("/proc").iterdir():
        if not entry.name.isdecimal():
            continue
        try:
            stat = (entry / "stat").read_text()
            fields = stat[stat.rfind(")") + 2:].split()
            if int(fields[2]) == group_id and fields[0] != "Z":
                members.append({"pid": int(entry.name), "state": fields[0], "start_ticks": int(fields[19])})
        except (OSError, ValueError, IndexError):
            continue
    return members


def source_label(path):
    try:
        return str(path.relative_to(ROOT))
    except ValueError:
        return str(path)


def sample_segments(points, spacing=0.025):
    for start, end in zip(points, points[1:]):
        count = max(1, math.ceil(math.dist(start, end) / spacing))
        for i in range(count):
            alpha = i / count
            yield (start[0] * (1 - alpha) + end[0] * alpha,
                   start[1] * (1 - alpha) + end[1] * alpha)
    if points:
        yield points[-1]


def minimum_map_clearance(points, grid):
    """Distance of densely sampled path centers to occupied cell rectangles.

    Include unknown cells and map boundaries as obstacles. The synthetic map
    has zero origin yaw; reject other orientations rather than silently use a
    wrong coordinate conversion. Radius is checked separately.
    """
    origin = grid.info.origin
    if (abs(origin.orientation.x) > 1e-9 or abs(origin.orientation.y) > 1e-9
            or abs(origin.orientation.z) > 1e-9 or abs(origin.orientation.w - 1) > 1e-9):
        raise RuntimeError("Probe collision check requires an axis-aligned map")
    resolution = grid.info.resolution
    width, height = grid.info.width, grid.info.height
    ox, oy = origin.position.x, origin.position.y
    occupied = []
    for index, value in enumerate(grid.data):
        if value < 0 or value >= 65:
            x = ox + index % width * resolution
            y = oy + index // width * resolution
            occupied.append((x, y, x + resolution, y + resolution))
    minimum = float("inf")
    for x, y in sample_segments(points):
        distance = min(x - ox, ox + width * resolution - x,
                       y - oy, oy + height * resolution - y)
        for xmin, ymin, xmax, ymax in occupied:
            dx = max(xmin - x, 0.0, x - xmax)
            dy = max(ymin - y, 0.0, y - ymax)
            distance = min(distance, math.hypot(dx, dy))
        minimum = min(minimum, distance)
    return minimum


def costmap_snapshot_readiness(snapshot, grid, start, goal):
    """Check an actual published master costmap before the single action.

    This probe supports a fully known, axis-aligned static map. Lifecycle
    active and StaticLayer.isCurrent() can precede the first master update
    after a resize. A matching published grid must retain every occupied
    source cell and contain neither unknown cells nor blocked endpoints.
    OccupancyGrid cost 99 denotes the inscribed inflation band; 100 is lethal.
    The check does not establish physical driving or dynamic-map safety.
    """
    source, actual = grid.info, snapshot.info
    source_values = (source.resolution, source.origin.position.x,
                     source.origin.position.y, source.origin.position.z,
                     source.origin.orientation.x, source.origin.orientation.y,
                     source.origin.orientation.z, source.origin.orientation.w)
    actual_values = (actual.resolution, actual.origin.position.x,
                     actual.origin.position.y, actual.origin.position.z,
                     actual.origin.orientation.x, actual.origin.orientation.y,
                     actual.origin.orientation.z, actual.origin.orientation.w)
    finite = all(math.isfinite(value) for value in (*source_values, *actual_values))
    axis_aligned = (finite and all(abs(value) <= 1e-9 for value in source_values[4:7])
                    and abs(source_values[7] - 1.0) <= 1e-9)
    geometry_matches = (finite and axis_aligned and source.resolution > 0
                        and source.width > 0 and source.height > 0
                        and grid.header.frame_id == snapshot.header.frame_id == "map"
                        and source.width == actual.width and source.height == actual.height
                        and all(math.isclose(a, b, rel_tol=0.0, abs_tol=1e-7)
                                for a, b in zip(source_values, actual_values)))
    lengths_valid = (len(grid.data) == source.width * source.height
                     and len(snapshot.data) == actual.width * actual.height)
    source_unknown = sum(value < 0 for value in grid.data)
    actual_unknown = sum(value < 0 for value in snapshot.data)
    values_valid = all(-1 <= value <= 100 for values in (grid.data, snapshot.data) for value in values)
    occupied_preserved = (geometry_matches and lengths_valid
        and all(snapshot.data[index] >= 99
                for index, value in enumerate(grid.data) if value >= 65))
    costs = [None, None]
    if geometry_matches and lengths_valid:
        for index, xy in enumerate((start, goal)):
            if len(xy) != 2 or not all(math.isfinite(value) for value in xy):
                continue
            ix = math.floor((xy[0] - actual.origin.position.x) / actual.resolution)
            iy = math.floor((xy[1] - actual.origin.position.y) / actual.resolution)
            if 0 <= ix < actual.width and 0 <= iy < actual.height:
                costs[index] = int(snapshot.data[iy * actual.width + ix])
    checks = {
        "geometry_matches_map": geometry_matches,
        "data_lengths_valid": lengths_valid,
        "occupancy_values_valid": values_valid,
        "source_map_fully_known": source_unknown == 0,
        "costmap_fully_known": actual_unknown == 0,
        "source_occupied_cells_preserved": occupied_preserved,
        "start_goal_nonlethal": all(cost is not None and 0 <= cost < 99 for cost in costs),
    }
    return {"ready": all(checks.values()), "checks": checks,
            "unknown_cells": actual_unknown, "source_unknown_cells": source_unknown,
            "start_goal_cell_costs": costs}


def main():
    global START, GOAL, PROBE_RADIUS
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path,
                        default=ROOT / "results" / ("nav2_probe_" + datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")))
    parser.add_argument("--timeout", type=float, default=80.0,
                        help="Wall seconds allowed for startup and action completion")
    parser.add_argument("--map", type=Path, default=ROOT / "maps/nav2_probe.yaml")
    parser.add_argument("--params", type=Path, default=ROOT / "configs/nav2_planner.yaml")
    parser.add_argument("--start", nargs=2, type=float, default=START)
    parser.add_argument("--goal", nargs=2, type=float, default=GOAL)
    parser.add_argument("--radius", type=float, default=PROBE_RADIUS)
    parser.add_argument("--domain-id", type=int, default=171,
                        help="Isolated ROS domain; adapter uses 173")
    parser.add_argument("--allow-direct-route", action="store_true",
                        help="Reuse as a path API without requiring the straight route to be blocked")
    args = parser.parse_args()
    START, GOAL, PROBE_RADIUS = tuple(args.start), tuple(args.goal), args.radius
    if PROBE_RADIUS <= 0 or not all(math.isfinite(v) for v in (*START, *GOAL, PROBE_RADIUS)):
        raise SystemExit("Finite start/goal and a positive radius are required")
    if not 0 <= args.domain_id <= 232 or not math.isfinite(args.timeout) or args.timeout <= 0:
        raise SystemExit("A domain in [0,232] and finite positive timeout are required")
    output = args.output.resolve()
    if output.exists() and any(output.iterdir()):
        raise SystemExit(f"Choose an empty output directory to preserve evidence: {output}")
    output.mkdir(parents=True, exist_ok=True)
    os.environ["ROS_DOMAIN_ID"] = str(args.domain_id)
    os.environ["ROS_LOCALHOST_ONLY"] = "1"
    ros_log_dir = output / "ros_logs"
    ros_log_dir.mkdir()
    os.environ["ROS_LOG_DIR"] = str(ros_log_dir)
    params, map_file = args.params.resolve(), args.map.resolve()
    shutil.copy2(params, output / params.name)
    shutil.copy2(map_file, output / map_file.name)
    shutil.copy2(map_file.with_suffix(".pgm"), output / map_file.with_suffix(".pgm").name)
    sources = [Path(__file__), params, map_file, map_file.with_suffix(".pgm")]
    result = {
        "accepted": False, "probe": "Humble NavFn ComputePathToPose software compatibility",
        "ros_domain_id": args.domain_id, "ros_localhost_only": True,
        "python_executable": sys.executable, "python_version": sys.version,
        "started_utc": datetime.now(timezone.utc).isoformat(),
        "source_sha256": {source_label(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in sources},
        "assumptions": {"start_xy_m": START, "goal_xy_m": GOAL,
                        "probe_circle_radius_m": PROBE_RADIUS,
                        "tf": "fixed map->base_link at start; no odom or AMCL",
                        "clock": "wall clock, use_sim_time=false"},
        "excludes": ["physical driving", "RB-Y1 footprint validation", "sensor integration",
                     "local controller", "user QP", "LLM", "dynamic obstacle avoidance"],
        "processes": [],
    }
    children = []
    node = None
    initialized = False
    started = time.monotonic()
    deadline = started + args.timeout

    def launch(name, command):
        log_path = output / f"{name}.log"
        log = log_path.open("w")
        process = subprocess.Popen(command, cwd=ROOT, env=os.environ.copy(),
                                   stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
        record = {"name": name, "command": command, "pid": process.pid,
                  "process_group_id": process.pid, "stdout_stderr": log_path.name}
        group = process_group_members(process.pid)
        record["start_ticks"] = next((p["start_ticks"] for p in group if p["pid"] == process.pid), None)
        result["processes"].append(record)
        children.append((process, log, record))
        (output / "process_manifest.json").write_text(json.dumps(result["processes"], indent=2) + "\n")

    def check_children():
        for process, _, record in children:
            if process.poll() is not None:
                raise RuntimeError(f"{record['name']} exited early with {process.returncode}; see {record['stdout_stderr']}")
        if time.monotonic() >= deadline:
            raise TimeoutError("ROS startup/action deadline exceeded")

    def wait_future(future):
        while not future.done():
            check_children()
            rclpy.spin_once(node, timeout_sec=0.2)
        return future.result()

    try:
        if sys.version_info[:2] != (3, 10):
            raise RuntimeError("Use /usr/bin/python3 (3.10) for the installed Humble binary ABI")
        import rclpy
        from action_msgs.msg import GoalStatus
        from geometry_msgs.msg import PoseStamped
        from lifecycle_msgs.srv import GetState
        from nav2_msgs.action import ComputePathToPose
        from nav_msgs.msg import OccupancyGrid
        from rclpy.action import ActionClient
        from rclpy.qos import DurabilityPolicy, QoSProfile, ReliabilityPolicy

        result["rclpy_module"] = rclpy.__file__
        result["ros_distro"] = os.environ.get("ROS_DISTRO")
        if result["ros_distro"] != "humble":
            raise RuntimeError("Source /opt/ros/humble/setup.bash before running this probe")
        rclpy.init()
        initialized = True
        def interrupted(signum, _frame):
            raise InterruptedError(f"Planner request interrupted by signal {signum}; cleaning owned children")
        signal.signal(signal.SIGTERM, interrupted)
        signal.signal(signal.SIGINT, interrupted)
        node = rclpy.create_node("opti_nav2_probe")
        maps = []
        qos = QoSProfile(depth=1, durability=DurabilityPolicy.TRANSIENT_LOCAL,
                         reliability=ReliabilityPolicy.RELIABLE)
        map_sub = node.create_subscription(OccupancyGrid, "/map", maps.append, qos)
        costmap_state = {"latest": None, "messages_seen": 0}

        def on_costmap(message):
            costmap_state["latest"] = message
            costmap_state["messages_seen"] += 1

        costmap_sub = node.create_subscription(
            OccupancyGrid, "/global_costmap/costmap", on_costmap, qos)
        launch("static_tf", ["ros2", "run", "tf2_ros", "static_transform_publisher",
                             "--x", str(START[0]), "--y", str(START[1]), "--z", "0",
                             "--roll", "0", "--pitch", "0", "--yaw", "0",
                             "--frame-id", "map", "--child-frame-id", "base_link"])
        launch("map_server", ["ros2", "run", "nav2_map_server", "map_server", "--ros-args",
                              "--params-file", str(params), "-p", f"yaml_filename:={map_file}"])
        launch("planner_server", ["ros2", "run", "nav2_planner", "planner_server", "--ros-args",
                                  "--params-file", str(params)])
        launch("lifecycle_manager", ["ros2", "run", "nav2_lifecycle_manager", "lifecycle_manager",
                                     "--ros-args", "-r", "__node:=lifecycle_manager_probe",
                                     "--params-file", str(params)])
        lifecycle_states = {}
        for name in ("map_server", "planner_server"):
            client = node.create_client(GetState, f"/{name}/get_state")
            while not client.wait_for_service(timeout_sec=0.2):
                check_children()
            while True:
                state = wait_future(client.call_async(GetState.Request())).current_state
                lifecycle_states[name] = {"id": state.id, "label": state.label}
                if state.id == 3:  # lifecycle_msgs/State.PRIMARY_STATE_ACTIVE
                    break
                check_children()
                rclpy.spin_once(node, timeout_sec=0.2)
            node.destroy_client(client)
        result["lifecycle_states"] = lifecycle_states
        while not maps:
            check_children()
            rclpy.spin_once(node, timeout_sec=0.2)
        grid = maps[-1]
        result["map"] = {"frame_id": grid.header.frame_id, "width": grid.info.width,
                         "height": grid.info.height, "resolution_m": grid.info.resolution,
                         "occupied_cells": sum(v >= 65 for v in grid.data),
                         "unknown_cells": sum(v < 0 for v in grid.data)}
        readiness_started = time.monotonic()
        while True:
            snapshot = costmap_state["latest"]
            readiness = (costmap_snapshot_readiness(snapshot, grid, START, GOAL)
                         if snapshot is not None else {"ready": False, "checks": {"snapshot_received": False}})
            readiness.update({"topic": "/global_costmap/costmap",
                              "waited_wall_s": time.monotonic() - readiness_started,
                              "messages_seen": costmap_state["messages_seen"]})
            if snapshot is not None:
                readiness.update({"width": snapshot.info.width, "height": snapshot.info.height,
                                  "header_stamp": [snapshot.header.stamp.sec, snapshot.header.stamp.nanosec]})
                if readiness["checks"]["geometry_matches_map"]:
                    readiness["origin_xy"] = [snapshot.info.origin.position.x, snapshot.info.origin.position.y]
            result["costmap_readiness"] = readiness
            check_children()  # The existing startup/action deadline also bounds this gate.
            if readiness["ready"]:
                break
            rclpy.spin_once(node, timeout_sec=0.2)
        action = ActionClient(node, ComputePathToPose, "/compute_path_to_pose")
        while not action.wait_for_server(timeout_sec=0.2):
            check_children()
            rclpy.spin_once(node, timeout_sec=0.2)

        def pose(xy):
            message = PoseStamped()
            message.header.frame_id = "map"
            message.header.stamp = node.get_clock().now().to_msg()
            message.pose.position.x, message.pose.position.y = xy
            message.pose.orientation.w = 1.0
            return message

        request = ComputePathToPose.Goal()
        request.start, request.goal = pose(START), pose(GOAL)
        request.planner_id = "GridBased"
        request.use_start = True
        handle = wait_future(action.send_goal_async(request))
        if not handle.accepted:
            raise RuntimeError("ComputePathToPose goal rejected")
        response = wait_future(handle.get_result_async())
        path = response.result.path
        points = [(p.pose.position.x, p.pose.position.y) for p in path.poses]
        if len(points) < 2:
            raise RuntimeError(f"Planner returned {len(points)} poses, status={response.status}")
        minimum = minimum_map_clearance(points, grid)
        straight_minimum = minimum_map_clearance([START, GOAL], grid)
        checks = {
            "costmap_ready": result["costmap_readiness"]["ready"],
            "action_succeeded": response.status == GoalStatus.STATUS_SUCCEEDED,
            "map_frame": grid.header.frame_id == "map" and path.header.frame_id == "map",
            "finite_path": all(math.isfinite(v) for point in points for v in point),
            "start_error": math.dist(points[0], START) <= 0.2,
            "goal_error": math.dist(points[-1], GOAL) <= 0.1,
            "probe_circle_clearance": minimum >= PROBE_RADIUS - 1e-5,
        }
        if not args.allow_direct_route:
            checks["direct_route_blocked"] = straight_minimum < PROBE_RADIUS
        result.update({"accepted": all(checks.values()), "checks": checks,
                       "action_status": response.status, "path_pose_count": len(points),
                       "path_length_m": sum(math.dist(a, b) for a, b in zip(points, points[1:])),
                       "minimum_center_obstacle_clearance_m": minimum,
                       "minimum_probe_circle_margin_m": minimum - PROBE_RADIUS,
                       "straight_route_clearance_m": straight_minimum,
                       "start_error_m": math.dist(points[0], START),
                       "goal_error_m": math.dist(points[-1], GOAL),
                       "planner_reported_time_s": response.result.planning_time.sec
                       + response.result.planning_time.nanosec * 1e-9})
        (output / "path.json").write_text(json.dumps({"frame_id": path.header.frame_id,
                                                      "xy_m": points}, indent=2) + "\n")
        node.destroy_subscription(map_sub)
        node.destroy_subscription(costmap_sub)
        action.destroy()
    except Exception as error:
        result["failure"] = f"{type(error).__name__}: {error}"
        (output / "probe_error.log").write_text(traceback.format_exc())
    finally:
        # A second parent signal must not interrupt cleanup partway through.
        signal.signal(signal.SIGTERM, signal.SIG_IGN)
        signal.signal(signal.SIGINT, signal.SIG_IGN)
        try:
            if node is not None:
                node.destroy_node()
            if initialized:
                rclpy.shutdown()
        except Exception as error:
            result["node_cleanup_error"] = f"{type(error).__name__}: {error}"
        for process, log, record in reversed(children):
            try:
                if process.poll() is None:
                    os.killpg(process.pid, signal.SIGTERM)
                    try:
                        process.wait(timeout=5.0)
                        record["shutdown"] = "SIGTERM process group"
                    except subprocess.TimeoutExpired:
                        os.killpg(process.pid, signal.SIGKILL)
                        process.wait(timeout=5.0)
                        record["shutdown"] = "SIGKILL process group after timeout"
                remaining = process_group_members(process.pid)
                if remaining:
                    os.killpg(process.pid, signal.SIGTERM)
                    until = time.monotonic() + 2.
                    while process_group_members(process.pid) and time.monotonic() < until:
                        time.sleep(.05)
                    if process_group_members(process.pid):
                        os.killpg(process.pid, signal.SIGKILL)
                        record["shutdown"] = "SIGKILL remaining owned process group"
                if process.poll() is None:
                    process.wait(timeout=5.0)
            except ProcessLookupError:
                pass
            except Exception as error:
                record["cleanup_error"] = f"{type(error).__name__}: {error}"
            record["exit_code"] = process.returncode
            record["remaining_process_group_members"] = process_group_members(process.pid)
            log.close()
        result["cleanup_verified"] = (not result.get("node_cleanup_error")
            and all(not record.get("remaining_process_group_members") and not record.get("cleanup_error")
                    for _, _, record in children))
        if not result["cleanup_verified"]:
            result["accepted"] = False
        result["wall_duration_s"] = time.monotonic() - started
        (output / "summary.json").write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
        print(json.dumps({"output": str(output), "accepted": result["accepted"],
                          "failure": result.get("failure"), "wall_duration_s": result["wall_duration_s"]}, indent=2), flush=True)
    return 0 if result["accepted"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
