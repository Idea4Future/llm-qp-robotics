"""Restricted Korean command admission and one ephemeral local-LLM plan.

This is deliberately NOT a general natural-language semantic parser. The
selected catalog admits either legacy redbin01 A->B or one rack/color pair to
the outbound station. Unsupported/ambiguous forms are rejected explicitly.
Planning admission is separate from fresh sensing and physical execution.
"""

from copy import deepcopy
import hashlib
import json
import os
from pathlib import Path
import re
import signal
import socket
import subprocess
import threading
import time
import urllib.error
import urllib.request

from .local_planner import LocalPlanner
from .world_registry import RegistryError, WorldRegistry


ROOT = Path(__file__).resolve().parents[2]
SUPPORTED_TEMPLATE = "A 입고대의 빨간 통을 B 조립대 첫 번째 자리에 가져다 놓아줘."
SUPPORTED_SKILLS = ["navigate", "observe", "pick", "place_on_tray", "stow_arms",
                    "navigate", "observe", "pick_from_tray", "place", "verify_place"]
CAPABILITIES = {"object_id": "redbin01", "source_id": "station_A", "destination_id": "station_B",
                "destination_slot_id": "slot_B_1", "arm": "right", "grasp_face_id": "side_x",
                "tray_slot_id": "tray_slot_1", "actual_approach_opening_m": .08,
                "maximum_supported_clearance_margin_m": .15,
                "stow_arms_meaning": "Hold the tested carry posture; no complete arm-folding trajectory is executed.",
                "initial_condition": "Already at A in the tested grasp-hover posture; navigate-to-A is an arrival precondition.",
                "status": "Restricted executor capabilities and design limits, not robot hardware ratings."}


def execution_capabilities(registry=None):
    """Return explicit per-layout software capabilities, never measured success.

    Old factory catalogs without an execution_profile retain the legacy IDs.
    The profile also tells the client that rack navigate-to-source is a real
    empty-tray motion from home, while legacy arrival at A is a precondition.
    """
    registry = registry or WorldRegistry.load()
    catalog = registry.llm_context(0)["static_catalog"]
    profile = deepcopy(catalog.get("execution_profile", {}))
    if not profile:
        profile = {"layout": "legacy", "source_by_object": {"redbin01": "station_A"},
                   "destination_id": "station_B", "destination_slot_id": "slot_B_1",
                   "initial_station_id": "station_A", "source_navigation": "arrival_precondition",
                   "normal_motion_design_defaults": {"base_speed_max": .15, "base_accel_max": .10,
                       "base_yaw_rate_max": .30, "obstacle_clearance_min": .15,
                       "arm_joint_speed_max": .50, "arm_joint_accel_max": 1.50},
                   "base_speed_command_cap_m_s": .15, "slow_base_speed_max_m_s": .10}
    if profile.get("layout") not in ("legacy", "rack-v1") or not isinstance(profile.get("source_by_object"), dict):
        _reject("unsupported_execution_profile", "등록된 legacy/rack-v1 실행 profile이 필요합니다.")
    for object_id, source_id in profile["source_by_object"].items():
        registry.entity(object_id, "object"); registry.entity(source_id, "station")
    registry.entity(profile["destination_id"], "station")
    registry.entity(profile["destination_slot_id"], "slot")
    profile.update(arm="right", grasp_face_id="side_x", tray_slot_id="tray_slot_1",
                   actual_approach_opening_m=.08, maximum_supported_clearance_margin_m=.15,
                   stow_arms_meaning="Hold the admitted carry posture; no complete arm-folding trajectory is implied.",
                   status="Restricted software contract and design limits; physical execution is validated separately.")
    return profile


def _reject(code, message):
    raise RegistryError(code, message)


def extract_supported_intent(command, registry=None):
    """Conservative rules for the documented command subset, without an LLM.

    Negative corridor clauses become mandatory requirements, not model hints.
    Optional numeric bounds support labeled SI-unit '이하/이상' clauses. Any
    remaining number or unsupported direction/object/slot/arm blocks execution.
    False rejections outside this narrow subset are intentional; this is not a
    held-out language-understanding claim.
    """
    capabilities = execution_capabilities(registry)
    rack = capabilities["layout"] == "rack-v1"
    if not isinstance(command, str) or not command.strip():
        _reject("unsupported_command_scope", "명확한 운반 지시가 필요합니다.")
    text = re.sub(r"\s+", " ", command.strip())
    if re.search(r"아니라|대신|반대로|또는|왼\s*(?:팔|손)|양팔", text) or re.search(r"(?:redbin01|bluebin01|greenbin01|(?:빨간|파란|초록|녹색)\s*(?:부품\s*)?(?:통|용기))\s*(?:은|는|을|를)?\s*말고", text):
        _reject("ambiguous_or_unsupported_command", "대체·선택·왼팔·양팔 지시는 현재 실행 범위 밖입니다.")
    if re.search(r"노란|드럼|큰\s*상자|두\s*개|둘", text) or (not rack and re.search(r"파란|녹색|초록", text)):
        _reject("unknown_or_unsupported_object", "다른 물체 또는 복수 물체가 포함된 요청은 현재 실행하지 않습니다.")
    if re.search(r"(?:팔.*접|옮기지|옮겨.*(?:마|않)|놓지|운반하지|이동하지)", text):
        _reject("ambiguous_or_unsupported_command", "팔 접기 또는 운반 자체의 부정 지시는 현재 실행하지 않습니다.")
    for match in re.finditer(r"(\d+)\s*번\s*(?:자리|슬롯|칸)", text):
        if int(match.group(1)) != 1:
            _reject("unknown_destination_slot", "현재 선택된 목적지의 첫 번째 자리만 지원합니다.")
    if re.search(r"(?:(?:두|세|네)\s*번째|둘째|셋째|넷째|마지막|왼쪽|오른쪽|옆)\s*(?:자리|슬롯|칸)", text):
        _reject("unknown_destination_slot", "등록되지 않은 자리나 방향을 첫 번째 자리로 대신 해석하지 않습니다.")
    if rack:
        sources = list(re.finditer(r"station_rack_[ABC]|[ABC]\s*선반", text))
        object_patterns = {"redbin01": r"redbin01|빨간\s*(?:부품\s*)?(?:통|용기)",
                           "bluebin01": r"bluebin01|파란\s*(?:부품\s*)?(?:통|용기)",
                           "greenbin01": r"greenbin01|(?:초록|녹색)\s*(?:부품\s*)?(?:통|용기)"}
        objects = [(key, match) for key, pattern in object_patterns.items()
                   for match in re.finditer(pattern, text)]
        source_ids = {"station_rack_" + re.search(r"[ABC]", match.group()).group() for match in sources}
        object_ids = {key for key, _ in objects}
        if len(source_ids) != 1 or len(object_ids) != 1:
            _reject("ambiguous_or_unsupported_object", "선반 A/B/C와 그 선반의 용기 한 개를 명시해야 합니다.")
        object_id = next(iter(object_ids)); source_id = next(iter(source_ids))
        if capabilities["source_by_object"].get(object_id) != source_id:
            _reject("source_object_mismatch", "A 선반=빨간 용기, B 선반=파란 용기, C 선반=초록 용기입니다. 물체를 대신 선택하지 않습니다.")
        source = sources[0]; object_match = objects[0][1]
        destination = re.search(r"(?:station_out|출고\s*(?:작업대|대|구역))\s*(?:의\s*)?(?:(?:첫\s*번째|첫\s*째|첫|1\s*번)\s*(?:자리|슬롯|칸))?", text)
        if re.search(r"station_[AB](?![A-Za-z_])|입고대|조립대", text):
            _reject("layout_intent_mismatch", "rack-v1은 선반에서 출고대로 운반합니다. 기존 입고대/조립대 작업은 legacy를 선택하세요.")
    else:
        object_id, source_id = "redbin01", "station_A"
        source = re.search(r"(?:station_A|A\s*(?:입고\s*작업대|입고대|작업대|테이블)?|입고대)\s*(?:의|에서|에\s*있는)", text)
        object_match = re.search(r"redbin01|빨간\s*(?:부품\s*)?(?:통|용기)", text)
        destination = re.search(r"(?:station_B|B\s*(?:조립\s*작업대|조립대|작업대|테이블)?|조립대)\s*(?:의\s*)?(?:첫\s*번째|첫\s*째|첫|1\s*번)\s*(?:자리|슬롯)", text)
    if object_match is None:
        _reject("unknown_or_unsupported_object", "선택한 실행 catalog의 용기를 명시해야 합니다.")
    if source is None or destination is None or not source.start() < object_match.start() < destination.start():
        example = "B 선반의 파란 용기를 출고대에 가져다 놓아줘." if rack else SUPPORTED_TEMPLATE
        _reject("ambiguous_transport_direction", "출발지·물체·목적지 순으로 운반 방향을 명시해야 합니다. 예: " + example)
    if not re.search(r"가져|옮겨|운반해|이동해|놓아|갖다", text):
        _reject("missing_transport_request", "지원되는 명시적 운반 요청이 필요합니다.")
    requirements = {"object_id": object_id, "source_id": source_id, "destination_id": capabilities["destination_id"],
                    "destination_slot_id": capabilities["destination_slot_id"], "forbidden_zone_ids": [], "constraint_bounds": []}
    if re.search(r"천천히\s*(?:하지|하진|말|마)|느리게\s*(?:하지|말|마)", text):
        _reject("ambiguous_speed_instruction", "저속 요청의 부정 표현은 자동 실행하지 않습니다.")
    if re.search(r"천천히|느리게|저속으로", text):
        requirements["constraint_bounds"].append({"constraint_id": "base_speed_max", "unit": "m/s",
                                                 "bound": capabilities["slow_base_speed_max_m_s"]})
    routes = list(re.finditer(r"(아래|하부|위|상부)\s*통로", text))
    if len(routes)>1 and re.search(r'통로\s*(?:와|과|랑|및)',text):
        _reject('ambiguous_corridor_instruction','여러 통로를 한 부정문으로 묶은 요청은 자동 실행하지 않습니다. 각 금지를 따로 명시하세요.')
    positives = []
    for i, match in enumerate(routes):
        clause = text[match.end():routes[i + 1].start() if i + 1 < len(routes) else len(text)]
        zone = "corridor_lower" if match.group(1) in ("아래", "하부") else "corridor_upper"
        if re.search(r"금지하지|않아도|피하지\s*마", clause):
            _reject("ambiguous_corridor_instruction", "통로의 이중 부정 표현은 현재 자동 실행하지 않습니다.")
        negative = re.search(r"금지|쓰지\s*(?:마|말|않)|(?:사용|이용)하지\s*(?:마|말|않)|사용하지", clause)
        if negative:
            requirements["forbidden_zone_ids"].append(zone)
        else:
            positives.append(zone)
    requirements["forbidden_zone_ids"] = sorted(set(requirements["forbidden_zone_ids"]))
    for zone in positives:
        opposite = "corridor_upper" if zone == "corridor_lower" else "corridor_lower"
        if opposite not in requirements["forbidden_zone_ids"]:
            _reject("unsupported_corridor_instruction", "현재 통로 조건은 '아래/위 통로는 쓰지 말아줘'처럼 금지 형태로 명시해야 합니다.")
    if len(requirements["forbidden_zone_ids"]) == 2:
        _reject("no_allowed_route", "두 통로가 모두 금지되어 운반할 수 없습니다. 금지를 완화하지 않았습니다.")
    # Recognized entities/slot numerals do not count as numeric motion bounds.
    masked = list(text)
    for match in (source, object_match, destination):
        masked[match.start():match.end()] = " " * (match.end() - match.start())
    # Velocity-QP differences constrain requested commands, not measured
    # servo acceleration. Do not accept an ordinary physical acceleration
    # request under a weaker, silently substituted command-slew contract.
    if re.search(r"(?:실제|측정|물리)\s*(?:관절|팔|주행|이동|베이스)?\s*(?:명령\s*)?가속도", text):
        _reject("physical_acceleration_bound_unsupported", "실제 측정 가속도의 상한은 현재 보장하지 않습니다. QP 명령 변화율과 구분해야 합니다.")
    if re.search(r"가속도\s*(?:는|은|를|을)?\s*(?:최대|최소)?\s*[+-]?(?:\d|\.\d)", text) and not re.search(r"명령\s*가속도", text):
        _reject("ambiguous_acceleration_contract", "숫자 가속도 조건은 '주행/팔 명령 가속도'로 명시해야 합니다. 실제 가속도 제한으로 바꾸어 해석하지 않습니다.")
    labels = [
        (r"(?:주행\s*|이동\s*|베이스\s*)?속도", "m/s", "base_speed_max", "upper"),
        (r"(?:주행\s*|이동\s*|베이스\s*)?명령\s*가속도", "m/s^2", "base_accel_max", "upper"),
        (r"(?:회전|yaw|각)\s*속도", "rad/s", "base_yaw_rate_max", "upper"),
        (r"(?:관절|팔)\s*속도", "rad/s", "arm_joint_speed_max", "upper"),
        (r"(?:관절|팔)\s*명령\s*가속도", "rad/s^2", "arm_joint_accel_max", "upper"),
        (r"(?:장애물\s*)?(?:여유|간격|거리)", "m", "obstacle_clearance_min", "lower"),
    ]
    for label, unit, name, sense in labels:
        unit_pattern = re.escape(unit).replace(r"\^2", r"(?:\^2|²|2)")
        pattern = r"(?<![가-힣A-Za-z])" + label + r"\s*(?:는|은|를|을)?\s*(최대|최소)?\s*([+-]?(?:\d+(?:\.\d+)?|\.\d+))\s*" + unit_pattern + r"(?![A-Za-z0-9²^])\s*(이하|이상)?"
        for match in re.finditer(pattern, text):
            expected = "이하" if sense == "upper" else "이상"
            if not (match.group(3) == expected or match.group(1) == ("최대" if sense == "upper" else "최소")):
                _reject("ambiguous_numeric_bound", f"{name}의 상한/하한을 {expected}로 명시해야 합니다.")
            value = float(match.group(2))
            if not 0 < value < float("inf"):
                _reject("invalid_numeric_bound", "양의 유한한 제약값이 필요합니다.")
            requirements["constraint_bounds"].append({"constraint_id": name, "unit": unit, "bound": value})
            masked[match.start():match.end()] = " " * (match.end() - match.start())
    if re.search(r"\d", "".join(masked)):
        _reject("unsupported_numeric_instruction", "해석되지 않은 숫자 조건은 자동 실행하지 않습니다. 지원 단위와 상한/하한을 명시하세요.")
    return requirements


def validate_execution_task(task, requirements, registry=None):
    """Check supported executor semantics AFTER planning/registry admission."""
    registry = registry or WorldRegistry.load()
    capabilities = execution_capabilities(registry)
    task = registry.validate_task_spec(task, require_fresh_observations=False)
    object_id, source_id = task["object_id"], task["source_id"]
    destination_id, slot_id = capabilities["destination_id"], capabilities["destination_slot_id"]
    if (capabilities["source_by_object"].get(object_id) != source_id
            or task["destination_id"] != destination_id):
        _reject("unsupported_executor_entity", "선택된 실행 profile의 물체·출발 선반·목적지 조합이 아닙니다.")
    for field in ("object_id", "source_id", "destination_id"):
        if requirements.get(field, task[field]) != task[field]:
            _reject("intent_mismatch", "명세의 물체·출발지·목적지가 원문과 다릅니다.")
    if requirements.get("destination_slot_id", slot_id) != slot_id:
        _reject("intent_mismatch", "명세의 목적지 자리가 원문과 다릅니다.")
    if sorted(task["forbidden_zone_ids"]) != sorted(requirements["forbidden_zone_ids"]):
        _reject("forbidden_zone_intent_mismatch", "명세의 통로 금지가 원문의 명시적 조건과 다릅니다.")
    skills = task["ordered_skills"]
    if [step["skill_id"] for step in skills] != SUPPORTED_SKILLS:
        _reject("unsupported_executor_sequence", "현재 실행기는 고정된 10단계 운반 template만 지원합니다.")
    expected = {0: {"station_id": source_id}, 1: {"object_id": object_id},
                2: {"object_id": object_id, "arm": "right", "grasp_face_id": "side_x"},
                3: {"object_id": object_id, "slot_id": "tray_slot_1"},
                5: {"station_id": destination_id}, 6: {"object_id": object_id},
                7: {"object_id": object_id, "slot_id": "tray_slot_1", "arm": "right", "grasp_face_id": "side_x"},
                8: {"object_id": object_id, "station_id": destination_id, "slot_id": slot_id},
                9: {"object_id": object_id, "station_id": destination_id, "slot_id": slot_id}}
    for index, arguments in expected.items():
        if any(skills[index].get(key) != value for key, value in arguments.items()):
            _reject("unsupported_executor_arguments", f"단계 {index + 1}의 물체·팔·자리는 현재 executor 범위 밖입니다.")
    for index in (2, 7):
        if skills[index]["gripper_opening_m"] > capabilities["actual_approach_opening_m"] + 1e-9:
            _reject("unsupported_gripper_opening", "최소 접근 opening이 현재 80 mm보다 커서 실행할 수 없습니다.")
    bounds = {item["constraint_id"]: item["bound"] for item in task["constraints"]}
    if bounds["obstacle_clearance_min"] > capabilities["maximum_supported_clearance_margin_m"] + 1e-9:
        _reject("map_replan_required", "0.15 m보다 큰 여유는 지도를 다시 팽창·검증해야 합니다. 조건을 완화하지 않았습니다.")
    return task


def _sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _request(url, data=None, timeout=3.):
    payload = None if data is None else json.dumps(data).encode()
    req = urllib.request.Request(url, data=payload, headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as response:
        return json.load(response)


def _group_members(group_id):
    members = []
    for item in Path("/proc").iterdir():
        if not item.name.isdecimal():
            continue
        try:
            raw = (item / "stat").read_text(); fields = raw[raw.rfind(")") + 2:].split()
            if int(fields[2]) == group_id and fields[0] != "Z":
                members.append(int(item.name))
        except (OSError, ValueError, IndexError):
            pass
    return members


def _stop_server(process):
    record = {"process_group_id": process.pid, "remaining_pids": []}
    if _group_members(process.pid):
        try:
            os.killpg(process.pid, signal.SIGTERM)
            deadline = time.monotonic() + 5.
            while _group_members(process.pid) and time.monotonic() < deadline:
                time.sleep(.05)
            if _group_members(process.pid):
                os.killpg(process.pid, signal.SIGKILL)
                deadline = time.monotonic() + 5.
                while _group_members(process.pid) and time.monotonic() < deadline:
                    time.sleep(.05)
            process.wait(timeout=5.)
        except ProcessLookupError:
            process.wait(timeout=5.)
    elif process.poll() is None:
        process.wait(timeout=5.)
    record["exit_code"] = process.returncode
    record["remaining_pids"] = _group_members(process.pid)
    record["verified"] = not record["remaining_pids"]
    return record


def plan_factory_command(command, output, *, requirements=None, registry=None,
                         backend="cuda", planner_mode="direct", startup_timeout_s=180., request_timeout_s=300.):
    """Start one pinned loopback server, request one plan, save raw data, stop.

    No robot, Nav2, model download, training, retry or automatic repair occurs.
    Returns accepted/task_spec plus the planning-only report and provenance.
    Output must be empty/new. A valid plan is not fresh-sensor execution approval.
    """
    output = Path(output).resolve()
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"Refusing to overwrite prior planner evidence: {output}")
    output.mkdir(parents=True, exist_ok=True)
    report = {"accepted": False, "task_spec": None, "command": command,
              "planner_mode": planner_mode,
              "planning_review_only": True, "execution_called": False,
              "automatic_retries": 0, "model_request_count": 0, "capabilities": CAPABILITIES,
              "command_executor_source_sha256": _sha256(Path(__file__)),
              "output": str(output), "cleanup": {"verified": True, "server_started": False}}
    process = log = None
    handlers = {}
    started = time.monotonic()
    try:
        registry = registry or WorldRegistry.load()
        report["capabilities"] = execution_capabilities(registry)
        extracted = extract_supported_intent(command, registry)
        if requirements is not None:
            # Caller additions are explicit hard requirements; fixed transport
            # IDs and the NL-extracted forbidden zones cannot be overridden.
            for key in ("object_id", "source_id", "destination_id", "destination_slot_id"):
                if requirements.get(key, extracted[key]) != extracted[key]:
                    _reject("intent_override_rejected", "추출된 운반 대상·방향·자리는 덮어쓸 수 없습니다.")
            extracted["forbidden_zone_ids"] = sorted(set(extracted["forbidden_zone_ids"]) | set(requirements.get("forbidden_zone_ids", [])))
            extracted["constraint_bounds"] += deepcopy(requirements.get("constraint_bounds", []))
        report["requirements"] = extracted
        if len(extracted["forbidden_zone_ids"]) >= 2:
            _reject("no_allowed_route", "두 통로가 모두 금지되었습니다.")
        if planner_mode not in ("direct", "typed-thinking"):
            raise ValueError("planner_mode must be direct or typed-thinking")
        config = json.loads((ROOT / "configs/local_llm.json").read_text())
        if backend not in ("cuda", "cpu"):
            raise ValueError("backend must be cuda or cpu")
        binary = ROOT / config[f"runtime_{backend}_path"]
        model = ROOT / config["model_path"]
        if not binary.is_file() or not os.access(binary, os.X_OK) or not model.is_file():
            raise FileNotFoundError("Existing pinned local runtime/model are required; no automatic download occurs")
        model_hash = _sha256(model)
        if model_hash != config["model_sha256"]:
            raise ValueError("Pinned model SHA256 mismatch")
        sources = ["src/opti_robot/command_executor.py", "src/opti_robot/local_planner.py", "src/opti_robot/world_registry.py",
                   "configs/local_llm.json", "configs/factory_world.json", "configs/task_spec.schema.json",
                   "configs/planner_response.schema.json", "prompts/task_planner.txt"]
        if planner_mode == "typed-thinking":
            sources += ["src/opti_robot/task_proposal.py", "prompts/task_proposal.txt",
                        "prompts/task_proposal_typed.txt", "configs/task_proposal.schema.json",
                        "configs/task_proposal_typed.schema.json"]
        for relative in sources:
            snapshot = output / "source_snapshot" / relative
            snapshot.parent.mkdir(parents=True, exist_ok=True)
            snapshot.write_bytes((ROOT / relative).read_bytes())
        # Persist the exact selected catalog, including per-layout defaults,
        # independently of its original filename or caller-created object.
        catalog = registry.llm_context(0)["static_catalog"]
        registry_snapshot = output / "registry_snapshot.json"
        registry_snapshot.write_text(json.dumps(catalog, ensure_ascii=False, indent=2) + "\n")
        report["provenance"] = {"source_sha256": {p: _sha256(ROOT / p) for p in sources},
                                "model_repository": config["model_repository"], "model_revision": config["model_revision"],
                                "model_sha256_actual": model_hash, "runtime_sha256": _sha256(binary),
                                "runtime_tag": config["runtime_tag"], "backend": backend, "additional_training": False,
                                "execution_layout": report["capabilities"]["layout"],
                                "registry_snapshot_sha256": _sha256(registry_snapshot)}
        with socket.socket() as reservation:
            reservation.bind(("127.0.0.1", 0)); port = reservation.getsockname()[1]
        base_url = f"http://127.0.0.1:{port}"
        server_command = [str(binary), "--model", str(model), "--host", "127.0.0.1", "--port", str(port),
                          "--ctx-size", str(config["context_tokens"]), "--parallel", "1", "--threads", "6",
                          "--jinja", "--reasoning", "on" if planner_mode == "typed-thinking" else "off",
                          "--n-gpu-layers", "all" if backend == "cuda" else "0"]
        if planner_mode == "typed-thinking":
            server_command += ["--reasoning-format", "deepseek", "--reasoning-budget", "3072",
                               "--chat-template-kwargs", json.dumps({"enable_thinking": True})]
        log = (output / "llama_server.log").open("w")
        process = subprocess.Popen(server_command, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
        report.update(server_command=server_command, server_pid=process.pid, server_url=base_url,
                      cleanup={"verified": False, "server_started": True})
        if threading.current_thread() is threading.main_thread():
            def interrupted(signum, frame):
                raise KeyboardInterrupt(f"Received signal {signum}")
            for signum in (signal.SIGTERM, signal.SIGINT):
                handlers[signum] = signal.signal(signum, interrupted)
        deadline = time.monotonic() + startup_timeout_s
        while True:
            if process.poll() is not None:
                raise RuntimeError(f"Local server exited before health readiness: {process.returncode}")
            try:
                if _request(base_url + "/health", timeout=2.).get("status") == "ok":
                    break
            except (urllib.error.URLError, TimeoutError):
                pass
            if time.monotonic() >= deadline:
                raise TimeoutError("Local server startup deadline exceeded")
            time.sleep(.2)
        report["startup_seconds"] = time.monotonic() - started
        planner_class = LocalPlanner
        if planner_mode == "typed-thinking":
            from .task_proposal import TypedThinkingLocalPlanner
            planner_class = TypedThinkingLocalPlanner
        planner = planner_class(registry=registry, base_url=base_url, timeout_s=request_timeout_s)
        messages = planner.messages(command, now_s=0.)
        # Keep the exact inference inputs, not only mutable source paths/hashes.
        (output / "messages.json").write_text(json.dumps(messages, ensure_ascii=False, indent=2) + "\n")
        (output / "response_schema.json").write_text(json.dumps(planner.response_schema, ensure_ascii=False, indent=2) + "\n")
        template_request = {"messages": messages}
        if planner_mode == "typed-thinking":
            template_request["chat_template_kwargs"] = {"enable_thinking": True}
        rendered = _request(base_url + "/apply-template", template_request)["prompt"]
        token_count = len(_request(base_url + "/tokenize", {"content": rendered, "add_special": False})["tokens"])
        max_tokens = min(4096 if planner_mode == "typed-thinking" else 1800,
                         config["context_tokens"] - token_count - 64)
        report.update(prompt_tokens=token_count, maximum_output_tokens=max_tokens)
        if max_tokens < (4096 if planner_mode == "typed-thinking" else 900):
            _reject("model_context_budget", "명세 출력에 필요한 문맥 공간이 부족합니다. 실행하지 않았습니다.")
        report["model_request_count"] = 1
        plan_report = planner.plan(command, now_s=0., requirements=extracted, max_tokens=max_tokens)
        report["planner_report"] = plan_report
        (output / "raw_plan_report.json").write_text(json.dumps(plan_report, ensure_ascii=False, indent=2) + "\n")
        if plan_report["status"] != "task":
            report["feedback"] = plan_report["response"]
        else:
            task = validate_execution_task(plan_report["response"]["task"], extracted, registry)
            report.update(accepted=True, task_spec=task)
            (output / "task_spec.json").write_text(json.dumps(task, ensure_ascii=False, indent=2) + "\n")
    except RegistryError as error:
        report["feedback"] = error.as_feedback()
    except KeyboardInterrupt as error:
        report["feedback"] = {"status": "blocked", "reason_code": "interrupted", "message": str(error), "hard_constraints_relaxed": False}
    except Exception as error:
        report["feedback"] = {"status": "blocked", "reason_code": "planner_pipeline_failed",
                              "message": f"{type(error).__name__}: {error}", "hard_constraints_relaxed": False}
    finally:
        if process is not None:
            try:
                report["cleanup"] = _stop_server(process)
            except Exception as error:
                report["cleanup"] = {"verified": False, "failure": f"{type(error).__name__}: {error}"}
            if not report["cleanup"]["verified"]:
                report["accepted"] = False
                report["feedback"] = {"status": "blocked", "reason_code": "server_cleanup_failed",
                                      "message": "모델 서버 종료를 확인하지 못해 실행을 차단했습니다.", "hard_constraints_relaxed": False}
        if log is not None:
            log.close()
        for signum, handler in handlers.items():
            signal.signal(signum, handler)
        report["wall_duration_s"] = time.monotonic() - started
        (output / "planner_summary.json").write_text(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False) + "\n")
    return report
