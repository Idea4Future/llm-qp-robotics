#!/usr/bin/env python3
"""Receive Isaac /clock in system ROS Humble Python 3.10 via DDS.

Run in a separate system-ROS terminal, never in the Isaac Python environment.
This checks transport and advancing clock values only; it does not establish
sensor/robot integration, real-time rate, physics, or general ROS compatibility.
An optional --output writes local diagnostic artifacts; no prior results ship.
"""
from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path
import sys
import time


def run(args):
    started = time.monotonic()
    report = {"accepted": False, "topic": args.topic, "messages": 0,
              "python": sys.version, "executable": sys.executable,
              "environment": {key: os.environ.get(key) for key in
                              ("ROS_DISTRO", "ROS_DOMAIN_ID", "ROS_LOCALHOST_ONLY", "RMW_IMPLEMENTATION")},
              "qos": {"depth": 10, "reliability": "best_effort", "durability": "volatile"},
              "timeout_wall_s": args.timeout_s, "minimum_messages": args.min_messages,
              "minimum_clock_span_s": args.min_span_s,
              "scope": "DDS transport and advancing clock only; not full robot/performance validation"}
    output = Path(args.output).resolve() if args.output else None
    if output is not None:
        if output.exists() and any(output.iterdir()):
            raise FileExistsError(f"Refusing to overwrite existing diagnostics: {output}")
        output.mkdir(parents=True, exist_ok=True)
    stream = node = rclpy = None
    observations = []
    malformed = backwards = 0
    publisher_count_max = 0
    try:
        if sys.version_info[:2] != (3, 10) or sys.prefix != sys.base_prefix:
            raise RuntimeError("Use /usr/bin/python3 from system ROS Humble, outside venv/Isaac Conda")
        if os.environ.get("ROS_DISTRO") != "humble":
            raise RuntimeError("Source /opt/ros/humble/setup.bash in a clean system-ROS shell")
        import rclpy
        from rclpy.qos import QoSProfile, ReliabilityPolicy, DurabilityPolicy
        from rosgraph_msgs.msg import Clock
        report["rclpy_module"] = rclpy.__file__
        if output is not None:
            stream = (output / "clock_messages.jsonl").open("w")
        rclpy.init(args=[])
        node = rclpy.create_node(f"isaac_clock_listener_{os.getpid()}")
        qos = QoSProfile(depth=10, reliability=ReliabilityPolicy.BEST_EFFORT,
                         durability=DurabilityPolicy.VOLATILE)

        def receive(msg):
            nonlocal malformed, backwards
            sec, nanosec = int(msg.clock.sec), int(msg.clock.nanosec)
            value = sec + nanosec * 1e-9
            valid = sec >= 0 and 0 <= nanosec < 1_000_000_000 and math.isfinite(value)
            if not valid:
                malformed += 1
            if observations and value < observations[-1]["clock_s"]:
                backwards += 1
            row = {"clock_s": value, "sec": sec, "nanosec": nanosec,
                   "arrival_wall_elapsed_s": time.monotonic() - started, "valid": valid}
            observations.append(row)
            if stream is not None:
                stream.write(json.dumps(row, allow_nan=False) + "\n")

        subscription = node.create_subscription(Clock, args.topic, receive, qos)
        deadline = started + args.timeout_s
        while rclpy.ok() and time.monotonic() < deadline:
            rclpy.spin_once(node, timeout_sec=.1)
            publisher_count_max = max(publisher_count_max, node.count_publishers(args.topic))
            span = observations[-1]["clock_s"] - observations[0]["clock_s"] if len(observations) >= 2 else 0
            if len(observations) >= args.min_messages and span >= args.min_span_s:
                break
        report.update(messages=len(observations), malformed_messages=malformed,
                      clock_backwards_events=backwards, publisher_count_max=publisher_count_max)
        report["discovered_publishers"] = [
            {"node_name": info.node_name, "node_namespace": info.node_namespace, "topic_type": info.topic_type}
            for info in node.get_publishers_info_by_topic(args.topic)]
        if observations:
            report.update(first_clock_s=observations[0]["clock_s"], last_clock_s=observations[-1]["clock_s"],
                          clock_span_s=observations[-1]["clock_s"] - observations[0]["clock_s"],
                          first_message_wall_s=observations[0]["arrival_wall_elapsed_s"])
            arrival_span = observations[-1]["arrival_wall_elapsed_s"] - observations[0]["arrival_wall_elapsed_s"]
            report["arrival_span_wall_s"] = arrival_span
            report["observed_arrival_hz"] = (len(observations) - 1) / arrival_span if arrival_span > 0 else None
        report["accepted"] = (len(observations) >= args.min_messages and malformed == 0 and backwards == 0
                              and report.get("clock_span_s", 0) >= args.min_span_s and publisher_count_max >= 1)
        report["reason"] = "clock_received_and_advancing" if report["accepted"] else "timeout_or_invalid_clock"
        report["exit_code"] = 0 if report["accepted"] else 2
        # Keep the subscription alive through the complete receive loop.
        del subscription
    except KeyboardInterrupt:
        report.update(reason="interrupted", exit_code=130)
    except Exception as exc:
        report.update(reason="listener_environment_or_runtime_error", error=f"{type(exc).__name__}: {exc}", exit_code=1)
    finally:
        if node is not None:
            node.destroy_node()
        if rclpy is not None and rclpy.ok():
            rclpy.shutdown()
        if stream is not None:
            stream.close()
        report["duration_wall_s"] = time.monotonic() - started
        if output is not None:
            (output / "summary.json").write_text(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False) + "\n")
    print(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False))
    return report["exit_code"]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--topic", default="/clock")
    parser.add_argument("--timeout-s", type=float, default=60)
    parser.add_argument("--min-messages", type=int, default=10)
    parser.add_argument("--min-span-s", type=float, default=.05)
    parser.add_argument("--output", help="Optional fresh directory for local diagnostics")
    args = parser.parse_args()
    if (not math.isfinite(args.timeout_s) or not math.isfinite(args.min_span_s)
            or args.timeout_s <= 0 or args.min_messages < 2 or args.min_span_s <= 0):
        parser.error("positive timeout/span and at least two messages are required")
    return run(args)


if __name__ == "__main__":
    raise SystemExit(main())
