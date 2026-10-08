"""Loopback-only local LLM planner with a task/blocked response contract.

Outputs are planning candidates. This module never calls a robot, solver,
sensor, executor, or automatic repair. Structured decoding cannot establish
meaning or feasibility, so registry and supplied hard intent requirements are
checked after generation. Requirement extraction itself remains a separate
interface; no general natural-language semantic guarantee is claimed.
"""
from __future__ import annotations

from copy import deepcopy
import json
import math
from pathlib import Path
import time
from urllib.parse import urlsplit
import urllib.error
import urllib.request

from .world_registry import RegistryError, WorldRegistry

ROOT = Path(__file__).resolve().parents[2]
REVIEW_CONTRACT = """API_RESPONSE_CONTRACT (위 정상 출력 설명을 이 envelope 계약으로 감싼다):
성공 후보는 {\"status\":\"task\",\"task\":<완전한 TaskSpec>}이다.
알 수 없는 물체, 미등록 자리, 불가능한 하드 조건은 {\"status\":\"blocked\",\"reason_code\":\"...\",\"message\":\"한국어 이유와 필요한 다음 조치\",\"hard_constraints_relaxed\":false}로 응답한다.
사용자가 요구한 물체/자리가 catalog에 없으면 비슷한 등록 물체/자리로 대체하지 않는다.
예: 녹색 드럼통이 목록에 없다면 unknown_entity로 거부한다. B 조립대 99번 자리가 없으면 unknown_destination_slot로 거부한다.
이번 호출은 PLANNING_REVIEW_ONLY이다. 미관측 상태를 그대로 유지하며 현재 실행하거나 실행을 승인하지 않는다.
정상 TaskSpec 후보는 require_fresh_observations=False로 검토한다. observe 스킬과 모든 하드 제약을 유지한다.
실제 실행은 별도의 관측 신선도, 도달 범위, 접촉, 주행, executor 검증을 필요로 한다.
USER_COMMAND에 없는 통로 금지를 추가하지 않는다. 좌표를 만들지 않는다. 설명문이나 코드 없이 envelope 하나만 출력한다.
/no_think"""


def _reject(code, message):
    raise RegistryError(code, message)


def _exact_keys(value, required):
    if not isinstance(value, dict) or set(value) != set(required):
        _reject("invalid_planner_response", f"expected exactly {sorted(required)}")


def validate_intent_requirements(spec, requirements, registry):
    """Check explicit caller-supplied intent, without pretending to parse all NL.

    Caller requirements can identify object/stations/destination slot, retain
    forbidden zones, and impose tighter numeric bounds. Catalog-mandatory
    constraints are already enforced by WorldRegistry.validate_task_spec.
    """
    if requirements is None:
        return
    allowed = {"object_id", "source_id", "destination_id", "destination_slot_id",
               "forbidden_zone_ids", "constraint_bounds"}
    if not isinstance(requirements, dict) or set(requirements) - allowed:
        _reject("invalid_intent_requirements", "unsupported requirement field")
    for field in ("object_id", "source_id", "destination_id"):
        if field in requirements and spec[field] != requirements[field]:
            _reject("intent_mismatch", f"required {field}={requirements[field]}")
    if "destination_slot_id" in requirements:
        slot_id = requirements["destination_slot_id"]
        for skill_id in ("place", "verify_place"):
            if not any(s["skill_id"] == skill_id and s.get("slot_id") == slot_id
                       for s in spec["ordered_skills"]):
                _reject("intent_mismatch", f"{skill_id} must use destination slot {slot_id}")
    zones = requirements.get("forbidden_zone_ids", [])
    if not isinstance(zones, list) or not all(isinstance(z, str) for z in zones):
        _reject("invalid_intent_requirements", "forbidden_zone_ids must be strings")
    for zone_id in zones:
        registry.entity(zone_id, "zone")
        if zone_id not in spec["forbidden_zone_ids"]:
            _reject("hard_constraint_missing", f"requested forbidden zone {zone_id} omitted")
    requested_bounds = requirements.get("constraint_bounds", [])
    if not isinstance(requested_bounds, list):
        _reject("invalid_intent_requirements", "constraint_bounds must be a list")
    definitions = {c["id"]: c for c in registry.llm_context(0)["static_catalog"]["constraint_types"]}
    actual_bounds = {c["constraint_id"]: c for c in spec["constraints"]}
    for item in requested_bounds:
        _exact_keys(item, {"constraint_id", "unit", "bound"})
        cid = item["constraint_id"]
        if not isinstance(cid, str) or cid not in definitions:
            _reject("unknown_constraint", str(cid))
        value = item["bound"]
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            _reject("invalid_intent_requirements", "bound must be finite")
        try:
            if not math.isfinite(float(value)):
                _reject("invalid_intent_requirements", "bound must be finite")
        except OverflowError:
            _reject("invalid_intent_requirements", "bound is too large")
        if item["unit"] != definitions[cid]["unit"]:
            _reject("unit_mismatch", cid)
        if cid not in actual_bounds:
            _reject("hard_constraint_missing", f"requested {cid} omitted")
        actual = actual_bounds[cid]["bound"]
        relaxed = (actual > value if definitions[cid]["sense"] == "upper" else actual < value)
        if relaxed:
            _reject("hard_constraint_relaxed", f"requested {cid} bound {value}, candidate {actual}")


def admit_response(envelope, registry, *, requirements=None):
    """Return an admitted task candidate or an explicit, unrelaxed rejection."""
    try:
        if not isinstance(envelope, dict):
            _reject("invalid_planner_response", "response must be an envelope object")
        status = envelope.get("status")
        if status == "blocked":
            _exact_keys(envelope, {"status", "reason_code", "message", "hard_constraints_relaxed"})
            if (not isinstance(envelope["reason_code"], str) or not envelope["reason_code"].strip()
                    or not isinstance(envelope["message"], str) or not envelope["message"].strip()
                    or envelope["hard_constraints_relaxed"] is not False):
                _reject("invalid_planner_response", "blocked feedback must be explicit and cannot relax constraints")
            return deepcopy(envelope)
        if status != "task":
            _reject("invalid_planner_response", "status must be task or blocked")
        _exact_keys(envelope, {"status", "task"})
        task = registry.validate_task_spec(envelope["task"], require_fresh_observations=False)
        validate_intent_requirements(task, requirements, registry)
        return {"status": "task", "task": task}
    except RegistryError as exc:
        return exc.as_feedback()


class LocalPlanner:
    """One request, no retries, no repair, no execution; audit data is returned."""

    def __init__(self, registry=None, *, base_url="http://127.0.0.1:8085", timeout_s=300):
        parsed = urlsplit(base_url)
        if (parsed.scheme != "http" or parsed.hostname not in ("127.0.0.1", "localhost", "::1")
                or parsed.username is not None or parsed.password is not None or parsed.query or parsed.fragment):
            raise ValueError("LocalPlanner requires a loopback HTTP URL without credentials")
        self.registry = registry or WorldRegistry.load()
        self.base_url = base_url.rstrip("/")
        self.timeout_s = timeout_s
        self.system_prompt = (ROOT / "prompts/task_planner.txt").read_text() + "\n\n" + REVIEW_CONTRACT
        self.response_schema = json.loads((ROOT / "configs/planner_response.schema.json").read_text())

    def messages(self, command, *, now_s=0):
        if not isinstance(command, str) or not command.strip():
            raise ValueError("command must be nonempty Korean or other natural-language text")
        context = json.dumps(self.registry.llm_context(now_s), ensure_ascii=False, separators=(",", ":"))
        return [{"role": "system", "content": self.system_prompt},
                {"role": "user", "content": "WORLD_REGISTRY_CONTEXT: " + context
                 + "\n\nUSER_COMMAND: " + command
                 + "\nApply this command. Corridor prohibitions and handling preferences are separate. Return the response envelope."}]

    def plan(self, command, *, now_s=0, requirements=None, max_tokens=1800):
        payload = {"model": "local-qwen3-4b", "messages": self.messages(command, now_s=now_s),
                   "temperature": 0.2, "top_p": 0.8, "seed": 42, "max_tokens": max_tokens,
                   "response_format": {"type": "json_schema", "json_schema": {
                       "name": "factory_planner_response", "strict": True, "schema": self.response_schema}}}
        report = {"command": command, "requirements": deepcopy(requirements),
                  "planning_review_only": True, "request_count": 1, "automatic_retries": 0,
                  "execute_called": False, "raw_api_response": None, "raw_content": None}
        start = time.monotonic()
        try:
            request = urllib.request.Request(self.base_url + "/v1/chat/completions",
                                             data=json.dumps(payload).encode(),
                                             headers={"Content-Type": "application/json"})
            with urllib.request.urlopen(request, timeout=self.timeout_s) as response:
                api_response = json.load(response)
            report["raw_api_response"] = api_response
            if not isinstance(api_response, dict):
                raise ValueError("model API response must be an object")
            report["usage"] = api_response.get("usage")
            report["server_timings"] = api_response.get("timings")
            content = api_response["choices"][0]["message"]["content"]
            report["raw_content"] = content
            candidate = json.loads(content)
            report["candidate_response"] = candidate
            report["response"] = admit_response(candidate, self.registry, requirements=requirements)
            report["rejected_after_generation"] = (candidate.get("status") == "task"
                                                    and report["response"]["status"] == "blocked") if isinstance(candidate, dict) else True
        except urllib.error.HTTPError as exc:
            report["http_status"] = exc.code
            report["raw_http_error"] = exc.read().decode("utf-8", errors="replace")
            report["response"] = {"status": "blocked", "reason_code": "model_request_failed",
                                  "message": f"Local model returned HTTP {exc.code}; no retry or constraint changes were made.",
                                  "hard_constraints_relaxed": False}
        except (urllib.error.URLError, TimeoutError, ValueError, KeyError, TypeError, IndexError) as exc:
            report["client_error"] = f"{type(exc).__name__}: {exc}"
            report["response"] = {"status": "blocked", "reason_code": "invalid_model_response",
                                  "message": report["client_error"], "hard_constraints_relaxed": False}
        report["request_seconds"] = time.monotonic() - start
        report["status"] = report["response"]["status"]
        return report
