"""Isolated Humble ComputePathToPose adapter callable from the core .venv.

No ROS Python module is imported in this process. The existing probe uses
/usr/bin/python3 with child-only Humble setup, fixed GT start TF and a static
map. This is a global path service, not a Nav2 local controller, localization
system, sensor topic loop, or physical clearance/braking guarantee.
"""
from __future__ import annotations

import fcntl
import json
import math
import os
from pathlib import Path
import signal
import subprocess
import time

ROOT = Path(__file__).resolve().parents[2]
ROS_SETUP = Path("/opt/ros/humble/setup.bash")
PROBE = ROOT / "scripts/check_nav2_planner.py"


def _xy(values, label):
    if len(values) != 2 or any(isinstance(v, bool) for v in values):
        raise ValueError(f"{label} must contain two finite coordinates")
    result = tuple(float(v) for v in values)
    if not all(math.isfinite(v) for v in result):
        raise ValueError(f"{label} must contain two finite coordinates")
    return result


def _group_members(group_id):
    """Read PID/state identity only, never environment or command lines."""
    result = []
    for entry in Path("/proc").iterdir():
        if not entry.name.isdecimal():
            continue
        try:
            raw = (entry / "stat").read_text()
            fields = raw[raw.rfind(")") + 2:].split()
            if int(fields[2]) == group_id and fields[0] != "Z":
                result.append({"pid": int(entry.name), "state": fields[0], "start_ticks": int(fields[19])})
        except (OSError, ValueError, IndexError):
            continue
    return result


def _manifest(path):
    try:
        records = json.loads(path.read_text())
        return records if isinstance(records, list) else []
    except (OSError, ValueError):
        return []


def _cleanup_owned_groups(records):
    """Last-resort cleanup of only the child sessions in this request manifest."""
    cleanup = []
    for record in records:
        group_id = record.get("process_group_id", record.get("pid"))
        if not isinstance(group_id, int) or group_id <= 1:
            continue
        before = _group_members(group_id)
        entry = {"process_group_id": group_id, "before": before}
        leader = next((p for p in before if p["pid"] == group_id), None)
        if leader is not None and record.get("start_ticks") not in (None, leader["start_ticks"]):
            entry["failure"] = "PID identity changed; did not signal an unrelated process"
        elif before:
            try:
                os.killpg(group_id, signal.SIGTERM)
                deadline = time.monotonic() + 3.
                while _group_members(group_id) and time.monotonic() < deadline:
                    time.sleep(.05)
                if _group_members(group_id):
                    os.killpg(group_id, signal.SIGKILL)
                    deadline = time.monotonic() + 2.
                    while _group_members(group_id) and time.monotonic() < deadline:
                        time.sleep(.05)
            except ProcessLookupError:
                pass
            except OSError as exc:
                entry["failure"] = f"{type(exc).__name__}: {exc}"
        entry["remaining"] = _group_members(group_id)
        cleanup.append(entry)
    return cleanup


def _stop_probe(process, manifest_path):
    record = {"requested_stop": "SIGTERM; probe handles it and cleans its own ROS sessions"}
    if process.poll() is None:
        try:
            os.killpg(process.pid, signal.SIGTERM)
        except ProcessLookupError:
            pass
        try:
            process.wait(timeout=30.)
        except subprocess.TimeoutExpired:
            record["fallback_child_cleanup"] = _cleanup_owned_groups(_manifest(manifest_path))
            try:
                os.killpg(process.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
            process.wait(timeout=5.)
            record["probe_shutdown"] = "SIGKILL only after graceful cleanup deadline"
    record["probe_exit_code"] = process.returncode
    return record


def request_factory_path(map_yaml, params_yaml, start_axle_xy, goal_axle_xy,
                         output_dir, clearance_radius=.75, *, domain_id=173,
                         timeout_s=80.):
    """Make one actual Nav2 request and return an audited path or failure.

    ``output_dir`` must be empty/new. Factory maps currently use a same-stem
    PGM alongside their YAML. Radius is a static center-path admission value,
    not a measured footprint or runtime safety certificate. Concurrent calls
    within this project cannot reuse the same ROS domain because of flock.
    Caller must check accepted before using path_file.
    """
    result = {"accepted": False, "path_file": None, "summary": None,
              "ros_domain_id": domain_id, "ros_localhost_only": True,
              "api": "Humble ComputePathToPose; static map, fixed GT start TF, wall clock",
              "control_called": False, "physical_execution": False, "cleanup_verified": False}
    process = lock_file = None
    created_output = False
    output = Path(output_dir).resolve()
    manifest_path = output / "nav2/process_manifest.json"
    start_time = time.monotonic()
    try:
        start, goal = _xy(start_axle_xy, "start_axle_xy"), _xy(goal_axle_xy, "goal_axle_xy")
        if isinstance(clearance_radius, bool) or not math.isfinite(clearance_radius) or clearance_radius <= 0:
            raise ValueError("clearance_radius must be finite and positive")
        if isinstance(timeout_s, bool) or not math.isfinite(timeout_s) or timeout_s <= 0:
            raise ValueError("timeout_s must be finite and positive")
        if isinstance(domain_id, bool) or not isinstance(domain_id, int) or not 0 <= domain_id <= 232:
            raise ValueError("domain_id must be an integer in [0,232]")
        map_file, params_file = Path(map_yaml).resolve(), Path(params_yaml).resolve()
        for path in (map_file, map_file.with_suffix(".pgm"), params_file, ROS_SETUP, PROBE, Path("/usr/bin/python3")):
            if not path.is_file():
                raise FileNotFoundError(f"Required existing file is absent: {path}")
        if output.exists() and any(output.iterdir()):
            raise FileExistsError(f"Refusing to overwrite previous evidence: {output}")
        locks = ROOT / "results/.nav2_domain_locks"
        locks.mkdir(parents=True, exist_ok=True)
        lock_file = (locks / f"domain_{domain_id}.lock").open("a+")
        try:
            fcntl.flock(lock_file.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise RuntimeError(f"ROS domain {domain_id} is already used by another adapter request") from exc
        output.mkdir(parents=True, exist_ok=True)
        created_output = True
        child_output = output / "nav2"
        log_path = output / "adapter.log"
        arguments = ["/usr/bin/python3", str(PROBE), "--output", str(child_output),
                     "--map", str(map_file), "--params", str(params_file),
                     "--start", *map(str, start), "--goal", *map(str, goal),
                     "--radius", str(clearance_radius), "--timeout", str(timeout_s),
                     "--domain-id", str(domain_id), "--allow-direct-route"]
        # Paths are argv values, not interpolated shell code.
        command = ["/bin/bash", "--noprofile", "--norc", "-c",
                   'set -e\nsource "$1"\nshift\nexec "$@"', "opti-nav2-child", str(ROS_SETUP), *arguments]
        child_env = os.environ.copy()
        for key in ("PYTHONPATH", "PYTHONHOME", "VIRTUAL_ENV"):
            child_env.pop(key, None)
        child_env["ROS_DOMAIN_ID"], child_env["ROS_LOCALHOST_ONLY"] = str(domain_id), "1"
        result.update({"start_axle_xy_m": start, "goal_axle_xy_m": goal,
                       "clearance_radius_m": float(clearance_radius), "timeout_s": float(timeout_s),
                       "adapter_log": str(log_path), "summary_file": str(child_output / "summary.json"),
                       "output_dir": str(output), "command": command})
        with log_path.open("w") as log:
            process = subprocess.Popen(command, cwd=ROOT, env=child_env, stdout=log,
                                       stderr=subprocess.STDOUT, start_new_session=True)
            result["probe_pid"] = process.pid
            try:
                process.wait(timeout=timeout_s + 45.)
            except subprocess.TimeoutExpired:
                result["failure"] = "Adapter deadline exceeded including child cleanup allowance"
                result["adapter_timeout_cleanup"] = _stop_probe(process, manifest_path)
        result["subprocess_exit_code"] = process.returncode
        summary_path = child_output / "summary.json"
        if summary_path.is_file():
            result["summary"] = json.loads(summary_path.read_text())
        records = _manifest(manifest_path)
        remaining = {str(r["pid"]): _group_members(r.get("process_group_id", r["pid"]))
                     for r in records if isinstance(r, dict) and isinstance(r.get("pid"), int)}
        if any(remaining.values()):
            result["remaining_group_cleanup"] = _cleanup_owned_groups(records)
            remaining = {str(r["pid"]): _group_members(r.get("process_group_id", r["pid"]))
                         for r in records if isinstance(r, dict) and isinstance(r.get("pid"), int)}
        result["remaining_process_groups"] = remaining
        summary = result["summary"] or {}
        result["cleanup_verified"] = bool(summary.get("cleanup_verified") and not any(remaining.values()))
        path_file = child_output / "path.json"
        if path_file.is_file():
            result["candidate_path_file"] = str(path_file)
        if (process.returncode == 0 and summary.get("accepted") and result["cleanup_verified"]
                and not result.get("failure") and path_file.is_file()):
            path = json.loads(path_file.read_text())
            points = [_xy(p, "path point") for p in path.get("xy_m", [])]
            if path.get("frame_id") != "map" or len(points) < 2:
                raise ValueError("Returned path must contain at least two finite map-frame points")
            result["accepted"], result["path_file"] = True, str(path_file)
        else:
            result.setdefault("failure", summary.get("failure") or
                              f"Nav2 request not admitted; exit={process.returncode}, cleanup={result['cleanup_verified']}; inspect adapter.log")
    except Exception as exc:
        result["failure"] = f"{type(exc).__name__}: {exc}"
    finally:
        if process is not None and process.poll() is None:
            result["finally_cleanup"] = _stop_probe(process, manifest_path)
        if lock_file is not None:
            fcntl.flock(lock_file.fileno(), fcntl.LOCK_UN)
            lock_file.close()
        result["wall_duration_s"] = time.monotonic() - start_time
        if created_output:
            result["adapter_summary_file"] = str(output / "adapter_summary.json")
            (output / "adapter_summary.json").write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n")
    return result
