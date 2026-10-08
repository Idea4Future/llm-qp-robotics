#!/usr/bin/env python3
"""Actual local-Qwen planning followed by an independently executed Isaac job.

Importing this module does not load an LLM, Isaac, ROS or a solver. Planning
admission never means physical success. The supported natural-language subset
and deterministic skill compiler are the existing command_executor contract.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import signal
import subprocess
import sys
import time
import traceback

ROOT = Path(__file__).resolve().parents[2]
REGISTRY = ROOT / "configs/isaac/world_registry.json"
ASSUMPTIONS = [
    "각 명령은 새 장면의 초기 상태에서 시작합니다(선반 환경: 홈, legacy: A 작업대). 앞선 명령의 종료 상태를 이어 제어하지 않습니다.",
    "LLM은 등록된 ID·목적 가중치·명령 한계를 구성하며 현재 물체 좌표를 스스로 알지 않습니다.",
    "계획 검토 시 관측은 미관측 상태입니다. 최신 관측·도달·접촉·적재 검사는 Isaac 실행기가 별도로 수행합니다.",
    "표시한 가속도 조건은 QP 명령의 변화율이며 실제 로봇 가속도·즉시 제동을 보장하지 않습니다.",
    "명세 한계와 실행기의 더 엄격한 한계는 교집합으로 적용합니다. 전체 접촉·운반 작업은 하나의 convex 문제가 아닙니다.",
]


def write_json(path, value):
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n")


def append_event(events_path, event_type, **fields):
    """One O_APPEND write per JSONL event; safe for the worker/physics writers."""
    record = {"type": event_type, "emitted_at": datetime.now(timezone.utc).isoformat(), **fields}
    data = (json.dumps(record, ensure_ascii=False, allow_nan=False, separators=(",", ":")) + "\n").encode()
    descriptor = os.open(Path(events_path), os.O_WRONLY | os.O_CREAT | os.O_APPEND, 0o600)
    try:
        if os.write(descriptor, data) != len(data):
            raise OSError("Incomplete event write")
    finally:
        os.close(descriptor)
    return record


def _latex_number(value):
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
        raise ValueError("Formula value must be finite and non-boolean")
    text = repr(value)
    if "e" in text.lower():
        mantissa, exponent = text.lower().split("e")
        return mantissa + r"\times 10^{" + str(int(exponent)) + "}"
    return text[:-2] if text.endswith(".0") else text


def formulas_for_task_spec(spec, *, arm_qp_active=False, base_qp_active=False, layout="rack-v1"):
    """Only call with a scope active after the physical runtime admits that QP.

    LLM weights and requested bounds are copied without repair. Effective limits
    are their explicit intersection with run_transport's fixed execution caps;
    these are command design limits, not measured hardware ratings. The formulas
    describe convex_control's shared QPs at dt=.02, including its normalizations,
    wheel rows, joint boxes and reference-aware velocity dampers. USD joint limits,
    measured carry radius, Jacobian and barrier geometry remain runtime symbols.
    Missing objective weights mean zero, matching task_spec_qp_config. A planning
    event with neither QP active still returns [], never an applied-QP claim.
    """
    weights = {item["term_id"]: item["weight"] for item in spec["objective_terms"]}
    bounds = {item["constraint_id"]: item["bound"] for item in spec["constraints"]}
    if layout not in {"rack-v1", "legacy"}:
        raise ValueError("Unsupported formula execution layout")
    base_speed_cap = .25 if layout == "rack-v1" else .15
    number = lambda key: _latex_number(weights.get(key, 0))
    formulas = []

    def add(identifier, title, latex, scope):
        formulas.append({"id": identifier, "title": title, "latex": latex, "scope": scope})

    def intersection(symbol, key, cap):
        return (symbol + r"=\min(" + _latex_number(bounds[key]) + ","
                + _latex_number(cap) + ")=" + _latex_number(min(bounds[key], cap)))

    if base_qp_active:
        v = _latex_number(min(bounds["base_speed_max"], base_speed_cap))
        omega = _latex_number(min(bounds["base_yaw_rate_max"], .3))
        add("base_velocity_qp", "주행 QP 목적: 기준 명령 추종·명령 크기·부드러움",
            r"\begin{aligned}\min_{u_k=(v_k,\omega_k)}\quad &"
            + number("base_path_tracking") + r"\|D_b(u_k-u_{\rm ref})\|_2^2\\&+"
            + number("base_effort") + r"\|D_bu_k\|_2^2\\&+"
            + number("base_command_change") + r"\|D_b(u_k-u_{k-1})\|_2^2\\"
            + r"D_b&=\operatorname{diag}\left(\frac{1}{" + v
            + r"},\frac{1}{" + omega + r"}\right)\end{aligned}",
            "Active shared velocity-command QP. Reference is the upstream path follower's command; all three terms use the same effective-limit normalization.")
        add("base_command_limits", "주행 명령 한계: LLM 요청과 실행 상한의 교집합",
            r"\begin{aligned}" + intersection(r"\bar v", "base_speed_max", base_speed_cap)
            + r"\quad &\mathrm{m/s}\\"
            + intersection(r"\bar a", "base_accel_max", .1) + r"\quad &\mathrm{m/s^2}\\"
            + intersection(r"\bar\omega", "base_yaw_rate_max", .3) + r"\quad &\mathrm{rad/s}\\"
            + r"|v_k|&\le\bar v,\qquad |\omega_k|\le\bar\omega\\"
            + r"|v_k-v_{k-1}|&\le\bar a\,\Delta t\\"
            + r"|\omega_k-\omega_{k-1}|&\le0.25\,\Delta t\\"
            + r"\Delta t&=0.02\;\mathrm{s}\end{aligned}",
            "Bounds apply to requested axle/yaw velocity and command slew, not physical acceleration or emergency braking.")
        add("base_wheel_limits", "양쪽 바퀴 명령 속도 한계",
            r"\left|\frac{v_k-0.265\,\omega_k}{0.1}\right|\le3.14,\qquad"
            + r"\left|\frac{v_k+0.265\,\omega_k}{0.1}\right|\le3.14"
            + r"\quad(\mathrm{rad/s})",
            "Shared QP wheel radius=.1 m, separation=.53 m, requested wheel-rate cap=3.14 rad/s. Native wheel sign changes neither absolute-value bound.")
        add("base_clearance_barrier", "국소 장애물 여유: 장애물 바깥쪽 방향을 기준으로",
            r"\begin{aligned}n_j^T\begin{bmatrix}\cos\psi_k\\\sin\psi_k\end{bmatrix}v_k"
            + r"&\ge-\kappa(d_j-r_{\rm carry}-m)\\"
            + r"d_j&=\|p_j\|_2,\quad n_j=-\frac{p_j}{d_j}\\"
            + r"m&=" + _latex_number(bounds["obstacle_clearance_min"]) + r"\;\mathrm{m},"
            + r"\quad\kappa=1\;\mathrm{s}^{-1}\end{aligned}",
            "p_j=obstacle point minus current axle in world axes; n_j points away from that obstacle. Current heading/points are frozen and carry radius is measured by the executor. Static instantaneous circle barrier only; sensor freshness/contact/braking remain separate runtime checks.")
    if arm_qp_active:
        qspeed = _latex_number(min(bounds["arm_joint_speed_max"], .3))
        add("arm_velocity_qp", "팔 QP 목적: 손끝 추종·관절 명령 크기·부드러움",
            r"\begin{aligned}\min_{\dot q_k\in\mathbb R^7}\quad &"
            + number("arm_tcp_tracking") + r"\|S(J_k\dot q_k-\dot x_{\rm ref})\|_2^2\\&+"
            + number("arm_joint_velocity") + r"\|D_q\dot q_k\|_2^2\\&+"
            + number("arm_command_change") + r"\|D_q(\dot q_k-\dot q_{k-1})\|_2^2\\"
            + r"S&=\operatorname{diag}\left(\frac1{0.1},\frac1{0.1},\frac1{0.1},"
            + r"\frac1{0.3},\frac1{0.3},\frac1{0.3}\right)\\"
            + r"D_q&=\frac1{" + qspeed + r"}I_7\end{aligned}",
            "Active shared right-arm velocity QP; J_k is the current world-frame 6x7 Jacobian, linear XYZ then angular XYZ. S uses .1 m/s linear and .3 rad/s angular scales.")
        add("arm_command_limits", "팔 관절 명령 속도·변화율 한계",
            r"\begin{aligned}"
            + intersection(r"\bar{\dot q}", "arm_joint_speed_max", .3) + r"\quad &\mathrm{rad/s}\\"
            + intersection(r"\bar a_q", "arm_joint_accel_max", .5) + r"\quad &\mathrm{rad/s^2}\\"
            + r"|\dot q_{k,i}|&\le\bar{\dot q}\\"
            + r"|\dot q_{k,i}-\dot q_{k-1,i}|&\le\bar a_q\,\Delta t\\"
            + r"\Delta t&=0.02\;\mathrm{s},\quad i=1,\ldots,7\end{aligned}",
            "Normal QP joint command slew; this is not a measured torque-servo qvel/qacc guarantee.")
        add("arm_joint_position_bounds", "관절 위치 한계와 누적 위치 명령 적분",
            r"\begin{aligned}q_{\min,i}&\le q_{k,i}+\Delta t\,\dot q_{k,i}\le q_{\max,i}\\"
            + r"q_{\min,i}&\le q^{\rm ref}_{k,i}+\Delta t\,\dot q_{k,i}\le q_{\max,i}\\"
            + r"q^{\rm ref}_{k+1}&=q^{\rm ref}_{k}+\Delta t\,\dot q_k,\qquad\Delta t=0.02\;\mathrm{s}\end{aligned}",
            "q_min/q_max are the actual per-joint USD limits admitted by the runtime. The first Euler box is kinematic; the second bounds integrated position-servo commands, not the predicted physical next state. Reference tracking/contact guards are separate.")
        add("arm_joint_velocity_damper", "관절 한계 근처 감속: 실제 자세와 위치 명령 모두 반영",
            r"\begin{aligned}-k_q\min(q_{k,i}-q_{\min,i},\;q^{\rm ref}_{k,i}-q_{\min,i})"
            + r"&\le\dot q_{k,i}\\\dot q_{k,i}&\le "
            + r"k_q\min(q_{\max,i}-q_{k,i},\;q_{\max,i}-q^{\rm ref}_{k,i})\\"
            + r"k_q&=1\;\mathrm{s}^{-1},\qquad i=1,\ldots,7\end{aligned}",
            "Shared QP affine velocity damper with joint_position_gain=1.0. Current state and reference are frozen, so each min is a constant. No recursive feasibility, grasp force, torque, self-collision or global safety claim.")
    return formulas


def emit_spec_event(path, spec, *, arm_qp_active=False, base_qp_active=False, layout="rack-v1"):
    return append_event(path, "spec", spec=spec, validated=True,
                        formulas=formulas_for_task_spec(spec, arm_qp_active=arm_qp_active,
                                                        base_qp_active=base_qp_active, layout=layout),
                        active_qp_scopes=(["arm"] if arm_qp_active else [])
                        + (["base"] if base_qp_active else []), assumptions=ASSUMPTIONS)


def _process_identity(pid):
    try:
        raw = Path(f"/proc/{pid}/stat").read_text()
        fields = raw[raw.rfind(")") + 2:].split()
        return {"pid": int(pid), "state": fields[0], "parent_pid": int(fields[1]),
                "process_group_id": int(fields[2]), "start_ticks": int(fields[19])}
    except (OSError, ValueError, IndexError):
        return None


def _group_members(group):
    result = []
    for entry in Path("/proc").iterdir():
        if entry.name.isdecimal():
            identity = _process_identity(int(entry.name))
            if identity and identity["process_group_id"] == group and identity["state"] != "Z":
                result.append(identity)
    return result


def _stop_owned_group(record):
    """Signal only a session/manifest group this job created; verify cleanup."""
    group = record["process_group_id"]
    if not isinstance(group, int) or group <= 1 or group == os.getpgrp():
        return {"verified": False, "failure": "Unsafe process group refused"}
    leader = _process_identity(group)
    if leader and record.get("start_ticks") not in (None, leader["start_ticks"]):
        return {"verified": False, "failure": "PID identity changed; refused unrelated group"}
    for sig, duration in ((signal.SIGTERM, 2.0), (signal.SIGKILL, 2.0)):
        if not _group_members(group):
            break
        try:
            os.killpg(group, sig)
        except ProcessLookupError:
            break
        deadline = time.monotonic() + duration
        while _group_members(group) and time.monotonic() < deadline:
            time.sleep(.05)
    remaining = _group_members(group)
    return {"process_group_id": group, "remaining": remaining, "verified": not remaining}


def _cleanup_nav2_groups(output):
    result = []
    # Read only manifests under this newly reserved job; never search other runs.
    for path in (Path(output) / "physics").glob("**/process_manifest.json"):
        try:
            records = json.loads(path.read_text())
            for record in records:
                if not isinstance(record, dict):
                    continue
                group = record.get("process_group_id", record.get("pid"))
                if isinstance(group, int) and group > 1:
                    result.append(_stop_owned_group({**record, "process_group_id": group}))
        except (OSError, ValueError, TypeError) as error:
            result.append({"verified": False, "failure": f"{type(error).__name__}: {error}"})
    return result


def _ensure_llm_stopped(plan):
    """Use the planner's cleanup, with a narrowly owned-child fallback."""
    cleanup = plan.get("cleanup") or {"verified": False}
    if cleanup.get("verified"):
        return cleanup
    pid = plan.get("server_pid")
    if not isinstance(pid, int) or pid <= 1:
        return cleanup
    identity = _process_identity(pid)
    if identity is None:
        if not _group_members(pid):
            return {**cleanup, "fallback": "No live owned server group remains", "verified": True}
        return {**cleanup, "fallback_failure": "Leader unavailable; remaining group ownership cannot be confirmed"}
    if identity["parent_pid"] != os.getpid() or identity["process_group_id"] != pid:
        return {**cleanup, "fallback_failure": "Server is not this worker's independent child; refused signaling"}
    fallback = _stop_owned_group(identity)
    return {**cleanup, "fallback": fallback, "verified": fallback["verified"]}


def physical_result_admitted(exit_code, summary):
    return (exit_code == 0 and isinstance(summary, dict) and summary.get("passed") is True
            and summary.get("mode") == "full" and summary.get("independent_validation_passed") is True
            and summary.get("navigation", {}).get("passed") is True)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    parser.add_argument("--command", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--events", type=Path)
    parser.add_argument("--backend", choices=("cuda", "cpu"), default="cuda")
    parser.add_argument("--plan-only", action="store_true", help="Planning admission only; final accepted remains false")
    parser.add_argument("--vision", action="store_true")
    parser.add_argument("--show-sim", action="store_true", help="Open the Isaac Sim GUI; headless remains the default")
    parser.add_argument("--layout", choices=("rack-v1", "legacy"), default="rack-v1")
    parser.add_argument("--moving-person", action="store_true", help="Include the optional moving-person experiment")
    parser.add_argument("--people-count", type=int, choices=(1, 3), default=3,
                        help="Number of people in the optional experiment; ignored without --moving-person")
    args = parser.parse_args(argv)
    output = args.output.resolve()
    if output.exists() and any(path.name != "worker.log" for path in output.iterdir()):
        parser.error("Output must be new/empty except for the UI-owned worker.log")
    output.mkdir(parents=True, exist_ok=True)
    events = (args.events or output / "events.jsonl").resolve()
    if not events.is_relative_to(output) or events.exists():
        parser.error("Events must be a new file inside this job directory")
    events.parent.mkdir(parents=True, exist_ok=True)
    result = {"accepted": False, "command": args.command, "mode": "full",
              "layout": args.layout, "show_sim": args.show_sim,
              "moving_person": args.moving_person, "people_count": args.people_count,
              "effective_people_count": args.people_count if args.moving_person else 0,
              "planner_mode": "typed-thinking", "planning_admitted": False,
              "physical_execution_called": False, "cancelled": False,
              "events": str(events.relative_to(output)), "artifacts": {},
              "source_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
    started, process, process_record = time.monotonic(), None, None
    cancelled = {"value": False}
    handlers = {}
    exit_code = 2

    def interrupted(signum, frame):
        cancelled["value"] = True
        raise KeyboardInterrupt(f"Received signal {signum}")

    for sig in (signal.SIGTERM, signal.SIGINT):
        handlers[sig] = signal.signal(sig, interrupted)
    try:
        sys.path.insert(0, str(ROOT / "src"))
        from opti_robot.command_executor import plan_factory_command
        from opti_robot.world_registry import WorldRegistry
        registry_source = REGISTRY if args.layout == "rack-v1" else ROOT / "configs/isaac/world_registry_legacy.json"
        catalog = json.loads(registry_source.read_text())
        registry = WorldRegistry(catalog)
        write_json(output / "registry_snapshot.json", catalog)
        result["registry_source_sha256"] = hashlib.sha256(registry_source.read_bytes()).hexdigest()
        append_event(events, "stage", stage="llm", detail="지원되는 지시인지 확인하고 로컬 Qwen의 작업·최적화 명세 생성을 준비합니다.")
        plan = plan_factory_command(args.command, output / "planner", registry=registry,
                                    backend=args.backend, planner_mode="typed-thinking")
        llm_cleanup = _ensure_llm_stopped(plan)
        result.update(planning_admitted=plan["accepted"], model_request_count=plan["model_request_count"],
                      llm_cleanup=llm_cleanup)
        if not llm_cleanup.get("verified"):
            raise RuntimeError("Owned model-server cleanup was not verified; physical execution blocked")
        if not plan["accepted"]:
            feedback = plan.get("feedback") or {}
            result["feedback"] = feedback
            cancelled["value"] = cancelled["value"] or feedback.get("reason_code") == "interrupted"
            result["failure"] = feedback.get("message", "계획이 거부되었습니다.")
        else:
            append_event(events, "stage", stage="validation", detail="등록된 물체·자리·하드 조건을 검토했습니다. 실행 관측은 별도로 확인합니다.")
            spec = plan["task_spec"]
            write_json(output / "task_spec.json", spec)
            # Runtime may later emit the same admitted spec with active-QP formulas.
            emit_spec_event(events, spec, layout=args.layout)
            if args.plan_only:
                result["failure"] = "계획 검토만 수행했습니다. 물리 실행 성공으로 표시하지 않습니다."
            else:
                transport = ROOT / "scripts/isaac/run_transport.py"
                if not transport.is_file():
                    raise FileNotFoundError("Isaac transport entry is absent; no physical execution occurred")
                command = ["bash", str(ROOT / "scripts/isaac/run_python.sh"), str(transport),
                           "--mode", "full", "--scene", "warehouse", "--video",
                           "--output", str(output / "physics"), "--task-spec", str(output / "task_spec.json"),
                           "--events", str(events), "--layout", args.layout]
                if not args.show_sim:
                    command.append("--headless")
                if args.moving_person:
                    command.extend(["--moving-person", "--people-count", str(args.people_count)])
                if args.vision:
                    command.append("--vision")
                env = os.environ.copy()
                env["OPTI_ISAAC_WORLD_REGISTRY"] = str(output / "registry_snapshot.json")
                append_event(events, "stage", stage="physics", detail="Isaac에서 실제 접촉 집기·트레이·주행·배치를 실행합니다.")
                result["physical_execution_called"] = True
                with (output / "physics_stdout.log").open("w") as log:
                    # Capture the new session identity before a cancellation
                    # handler can unwind this launch and lose child ownership.
                    previous_mask = signal.pthread_sigmask(signal.SIG_BLOCK, set(handlers))
                    try:
                        process = subprocess.Popen(command, cwd=ROOT, env=env, stdout=log,
                                                   stderr=subprocess.STDOUT, start_new_session=True)
                        process_record = _process_identity(process.pid) or {
                            "pid": process.pid, "process_group_id": process.pid}
                        write_json(output / "worker_processes.json", {"physics": process_record})
                    finally:
                        signal.pthread_sigmask(signal.SIG_SETMASK, previous_mask)
                    result["physics_exit_code"] = process.wait()
                append_event(events, "stage", stage="verify", detail="실제 실행 종료 코드와 물리 검증 결과를 함께 확인합니다.")
                path = output / "physics/summary.json"
                physics_summary = json.loads(path.read_text()) if path.is_file() else None
                result["physics_summary"] = physics_summary
                result["accepted"] = physical_result_admitted(process.returncode, physics_summary)
                if not result["accepted"]:
                    result["failure"] = (physics_summary or {}).get("error") or (physics_summary or {}).get("failure") or "full 물리 검증·정상 종료 조건을 모두 충족하지 못했습니다."
                exit_code = 0 if result["accepted"] else 2
    except KeyboardInterrupt:
        cancelled["value"] = True
        result["failure"] = "중지 요청을 받아 자식 프로세스를 정리합니다. 자동 재시도는 하지 않습니다."
    except Exception as error:
        result["failure"] = f"{type(error).__name__}: {error}"
        (output / "worker_error.log").write_text(traceback.format_exc())
    finally:
        # A second TERM must not interrupt owned-child cleanup.
        for sig in handlers:
            signal.signal(sig, lambda signum, frame: cancelled.update(value=True))
        if process_record is not None:
            result["physics_cleanup"] = _stop_owned_group(process_record)
            if process is not None and process.poll() is None:
                try:
                    process.wait(timeout=1.)
                except subprocess.TimeoutExpired:
                    result["physics_cleanup"]["verified"] = False
            if not result["physics_cleanup"]["verified"]:
                result["accepted"] = False
                result["failure"] = "Physics process cleanup was not verified"
        result["nav2_cleanup"] = _cleanup_nav2_groups(output)
        if any(not item["verified"] for item in result["nav2_cleanup"]):
            result["accepted"] = False
            result["failure"] = "Nav2 child cleanup was not verified"
        result["cancelled"] = cancelled["value"]
        if result["cancelled"]:
            result["accepted"], exit_code = False, 130
        elif not result["accepted"]:
            exit_code = 2
        result.update(exit_code=exit_code, wall_duration_s=time.monotonic()-started)
        for key, relative in {"task_spec": "task_spec.json", "planner_summary": "planner/planner_summary.json",
                              "physics_summary": "physics/summary.json", "video": "physics/replay.mp4",
                              "camera": "physics/camera.png", "physics_log": "physics_stdout.log",
                              "registry": "registry_snapshot.json"}.items():
            if (output / relative).is_file():
                result["artifacts"][key] = relative
        result["artifacts"]["summary"] = "command_summary.json"
        write_json(output / "command_summary.json", result)
        detail = ("실제 물리 운반 검증과 정상 종료를 모두 통과했습니다." if result["accepted"]
                  else "작업이 중지됐습니다." if result["cancelled"] else result.get("failure", "실행이 검증되지 않았습니다."))
        append_event(events, "result", accepted=result["accepted"], cancelled=result["cancelled"],
                     exit_code=exit_code, detail=detail,
                     summary={"planning_admitted": result["planning_admitted"],
                              "physical_execution_called": result["physical_execution_called"],
                              "physics_exit_code": result.get("physics_exit_code"),
                              "physical_full_passed": result["accepted"]}, artifacts=result["artifacts"])
        for sig, handler in handlers.items():
            signal.signal(sig, handler)
    print(json.dumps({"accepted": result["accepted"], "exit_code": exit_code,
                      "output": str(output), "failure": result.get("failure")}, ensure_ascii=False), flush=True)
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
