"""Compact LLM objectives/bounds -> fixed skills, with no numeric repair.

The LLM proposes entity IDs, forbidden zones, and six local-QP weights/bounds.
The compiler expands only the selected catalog's fixed right-arm transport
template; it does NOT make an LLM-generated task-order claim. Its fixed 60-mm
minimum grasp opening is a tested-template parameter, not an LLM output.
LLM costs and constraint values are copied exactly, or the proposal is rejected.
Acceleration bounds describe QP command slew, not measured robot acceleration.
Later runtime cap intersection, sensing, dynamics and stopping stay external.
"""
from __future__ import annotations

from copy import deepcopy
import json
import math
from pathlib import Path
import time
import urllib.error
import urllib.request

from .local_planner import LocalPlanner, admit_response
from .world_registry import RegistryError, WorldRegistry

ROOT = Path(__file__).resolve().parents[2]
TERMS = ("base_path_tracking", "base_effort", "base_command_change", "arm_tcp_tracking",
         "arm_joint_velocity", "arm_command_change")
CONSTRAINT_UNITS = {"base_speed_max": "m/s", "base_accel_max": "m/s^2", "base_yaw_rate_max": "rad/s",
                    "obstacle_clearance_min": "m", "arm_joint_speed_max": "rad/s", "arm_joint_accel_max": "rad/s^2"}
PROPOSAL_FIELDS = ("schema_version", "object_id", "source_id", "destination_id", "destination_slot_id",
                   "forbidden_zone_ids", "objective_weights", "constraint_bounds")
TEMPLATE_OPENING_M = .06


def _exact(value, keys, label):
    if not isinstance(value, dict) or set(value) != set(keys):
        raise RegistryError("invalid_proposal_shape", f"{label} requires exactly {list(keys)}")


def _finite(value, label):
    try:
        valid = not isinstance(value, bool) and isinstance(value, (int, float)) and math.isfinite(value)
    except OverflowError:
        valid = False
    if not valid:
        raise RegistryError("invalid_number", f"{label} must be finite and non-boolean")


def compile_task_proposal(proposal, registry=None):
    """Expand the implemented skill template; preserve every proposed numeric value.

    Returns a candidate full TaskSpec, not execution admission. Structural checks
    and supported IDs happen here; registry/intent/executor checks happen in
    compile_proposal_response. Unknown IDs are rejected before expansion.
    """
    registry = registry or WorldRegistry.load()
    _exact(proposal, PROPOSAL_FIELDS, "proposal")
    if isinstance(proposal["schema_version"], bool) or proposal["schema_version"] != 1:
        raise RegistryError("unsupported_schema", str(proposal["schema_version"]))
    for field, kind in (("object_id", "object"), ("source_id", "station"),
                        ("destination_id", "station"), ("destination_slot_id", "slot")):
        registry.entity(proposal[field], kind)
    from .command_executor import execution_capabilities
    capabilities = execution_capabilities(registry)
    if (capabilities["source_by_object"].get(proposal["object_id"]) != proposal["source_id"]
            or proposal["destination_id"] != capabilities["destination_id"]
            or proposal["destination_slot_id"] != capabilities["destination_slot_id"]):
        raise RegistryError("unsupported_executor_entity", "Proposal does not match the selected layout's object/source/destination pair")
    _exact(proposal["objective_weights"], TERMS, "objective_weights")
    _exact(proposal["constraint_bounds"], CONSTRAINT_UNITS, "constraint_bounds")
    for group in ("objective_weights", "constraint_bounds"):
        for key, value in proposal[group].items():
            _finite(value, f"{group}.{key}")
    obj = proposal["object_id"]
    skills = [
        {"skill_id": "navigate", "station_id": proposal["source_id"]},
        {"skill_id": "observe", "object_id": obj},
        {"skill_id": "pick", "object_id": obj, "arm": "right", "grasp_face_id": "side_x", "gripper_opening_m": TEMPLATE_OPENING_M},
        {"skill_id": "place_on_tray", "object_id": obj, "slot_id": "tray_slot_1"},
        {"skill_id": "stow_arms"},
        {"skill_id": "navigate", "station_id": proposal["destination_id"]},
        {"skill_id": "observe", "object_id": obj},
        {"skill_id": "pick_from_tray", "object_id": obj, "slot_id": "tray_slot_1", "arm": "right", "grasp_face_id": "side_x", "gripper_opening_m": TEMPLATE_OPENING_M},
        {"skill_id": "place", "object_id": obj, "station_id": proposal["destination_id"], "slot_id": proposal["destination_slot_id"]},
        {"skill_id": "verify_place", "object_id": obj, "station_id": proposal["destination_id"], "slot_id": proposal["destination_slot_id"]}]
    return {"schema_version": 1, "object_id": obj, "source_id": proposal["source_id"],
            "destination_id": proposal["destination_id"], "ordered_skills": skills,
            "objective_terms": [{"term_id": key, "weight": deepcopy(proposal["objective_weights"][key])} for key in TERMS],
            "constraints": [{"constraint_id": key, "unit": unit, "bound": deepcopy(proposal["constraint_bounds"][key])}
                            for key, unit in CONSTRAINT_UNITS.items()],
            "forbidden_zone_ids": deepcopy(proposal["forbidden_zone_ids"])}


def compile_proposal_response(envelope, registry=None, *, requirements=None):
    """Return the standard admitted task/blocked envelope, without repairs.

    No objective mapping, clipping, default filling, or hard-bound relaxation is
    performed. Fixed skills are explicit. Fresh observations remain required by
    the eventual runtime, even when planning-review validation passes.
    """
    registry = registry or WorldRegistry.load()
    try:
        if isinstance(envelope, dict) and envelope.get("status") == "blocked":
            return admit_response(envelope, registry, requirements=requirements)
        _exact(envelope, ("status", "proposal"), "proposal envelope")
        if envelope["status"] != "proposal":
            raise RegistryError("invalid_planner_response", "Expected proposal or blocked envelope")
        task = compile_task_proposal(envelope["proposal"], registry)
        admitted = admit_response({"status": "task", "task": task}, registry, requirements=requirements)
        if admitted["status"] != "task":
            return admitted
        # Delayed import avoids a cycle when the command executor selects this
        # planner. This is the existing restricted execution-admission check.
        from .command_executor import validate_execution_task
        execution_requirements = {
            "object_id": task["object_id"], "source_id": task["source_id"], "destination_id": task["destination_id"],
            "destination_slot_id": envelope["proposal"]["destination_slot_id"], "forbidden_zone_ids": task["forbidden_zone_ids"]}
        if requirements is not None:
            execution_requirements.update(deepcopy(requirements))
        admitted["task"] = validate_execution_task(admitted["task"], execution_requirements, registry)
        return admitted
    except RegistryError as exc:
        return exc.as_feedback()


def proposal_context(registry, now_s=0, *, include_route_guidance=True):
    """Small typed catalog; no world XYZ, raw observations, or skill generation."""
    full = registry.llm_context(now_s)
    catalog = full["static_catalog"]
    from .command_executor import execution_capabilities
    capabilities = execution_capabilities(registry)
    ids = (set(capabilities["source_by_object"]) | set(capabilities["source_by_object"].values())
           | {capabilities["destination_id"], capabilities["destination_slot_id"], "corridor_lower", "corridor_upper"})
    if capabilities["layout"] == "rack-v1":
        ids.add(capabilities["initial_station_id"])
    entities = []
    for entity in catalog["entities"]:
        if entity["id"] in ids:
            row = {key: deepcopy(entity[key]) for key in ("id", "kind", "name", "aliases")}
            if "owner_id" in entity:
                row["owner_id"] = entity["owner_id"]
            if "home_station_id" in entity:
                row["home_station_id"] = entity["home_station_id"]
            entities.append(row)
    constraints = []
    for row in catalog["constraint_types"]:
        item = {key: deepcopy(row[key]) for key in ("id", "unit", "sense", "bound_design_range")}
        if include_route_guidance and "interpretation" in row:
            item["interpretation"] = deepcopy(row["interpretation"])
        constraints.append(item)
    context = {"planning_review_only": True, "entities": entities,
            "objective_terms": [{key: deepcopy(row[key]) for key in ("id", "description")} for row in catalog["objective_terms"]],
            "weight_design_range": deepcopy(catalog["objective_policy"]["weight_design_range"]),
            "constraint_types": constraints,
            "current_map_supported_margin_max_m": .15,
            "observation_status": {key: value for key, value in full["observation_status"].items() if key in ids},
            "coordinate_policy": "No coordinates in proposal; executor retrieves fresh sensor poses.",
            "execution_layout": capabilities["layout"],
            "supported_transports": [{"object_id": obj, "source_id": source,
                "destination_id": capabilities["destination_id"], "destination_slot_id": capabilities["destination_slot_id"]}
                for obj, source in capabilities["source_by_object"].items()],
            "initial_station_id": capabilities["initial_station_id"],
            "source_navigation": capabilities["source_navigation"],
            "normal_motion_design_defaults": deepcopy(capabilities["normal_motion_design_defaults"]),
            "base_speed_command_cap_m_s": capabilities["base_speed_command_cap_m_s"],
            "slow_base_speed_max_m_s": capabilities["slow_base_speed_max_m_s"],
            "fixed_skill_compiler": "Fixed right-arm source navigation, observe/pick/load, destination navigation, observe/tray pick/place/verify. rack-v1 starts at home and really navigates to the selected source; legacy already starts at A. Not LLM-generated ordering.",
            "limits_status": "Project admission/design limits, not robot ratings or safety guarantees."}
    if include_route_guidance:
        context.update(mapped_route_alternatives=[["corridor_lower"], ["corridor_upper"]],
                       route_rule="Forbidding one alternative does not forbid the other; actual path feasibility is checked by Nav2.",
                       acceleration_semantics="QP command slew bounds, not measured physical base/joint acceleration.")
    return context


class CompactLocalPlanner(LocalPlanner):
    """LocalPlanner-compatible messages/plan API, one inference and no execution.

    Response status and response envelope match LocalPlanner (task/blocked).
    candidate_response/raw_proposal remain the actual model proposal/blocked
    envelope. The standard TaskSpec is explicitly compiler-generated from it.
    """
    def __init__(self, registry=None, *, base_url="http://127.0.0.1:8085", timeout_s=300):
        super().__init__(registry=registry, base_url=base_url, timeout_s=timeout_s)
        self.system_prompt = (ROOT / "prompts/task_proposal.txt").read_text()
        self.response_schema = json.loads((ROOT / "configs/task_proposal.schema.json").read_text())
        self.planner_mode = "compact"
        self.default_max_tokens = 1800

    def context(self, now_s=0):
        return proposal_context(self.registry, now_s)

    def extra_request_fields(self):
        return {}

    def messages(self, command, *, now_s=0):
        if not isinstance(command, str) or not command.strip():
            raise ValueError("command must be nonempty text")
        context = json.dumps(self.context(now_s), ensure_ascii=False, separators=(",", ":"))
        return [{"role": "system", "content": self.system_prompt},
                {"role": "user", "content": "PROPOSAL_CATALOG: " + context + "\n\nUSER_COMMAND: " + command}]

    def plan(self, command, *, now_s=0, requirements=None, max_tokens=1800):
        payload = {"model": "local-qwen3-4b", "messages": self.messages(command, now_s=now_s),
                   "temperature": .2, "top_p": .8, "seed": 42, "max_tokens": max_tokens,
                   "response_format": {"type": "json_schema", "json_schema": {
                       "name": "factory_task_proposal", "strict": True, "schema": self.response_schema}}}
        payload.update(self.extra_request_fields())
        report = {"command": command, "requirements": deepcopy(requirements), "planning_review_only": True,
                  "request_count": 1, "automatic_retries": 0, "execute_called": False,
                  "skill_sequence_origin": "deterministic right-arm transport template for the selected catalog layout",
                  "optimization_values_origin": "LLM proposal; copied unchanged or rejected",
                  "raw_api_response": None, "raw_content": None, "request_payload": deepcopy(payload)}
        report["planner_mode"] = self.planner_mode
        started = time.monotonic()
        try:
            req = urllib.request.Request(self.base_url + "/v1/chat/completions", data=json.dumps(payload).encode(),
                                         headers={"Content-Type": "application/json"})
            with urllib.request.urlopen(req, timeout=self.timeout_s) as response:
                api = json.load(response)
            report["raw_api_response"] = api
            report["usage"] = api.get("usage")
            report["server_timings"] = api.get("timings")
            content = api["choices"][0]["message"]["content"]
            report["raw_content"] = content
            candidate = json.loads(content)
            report["candidate_response"] = candidate
            report["raw_proposal"] = candidate
            admitted = compile_proposal_response(candidate, self.registry, requirements=requirements)
            report["response"] = admitted
            report["rejected_after_generation"] = isinstance(candidate, dict) and candidate.get("status") == "proposal" and admitted["status"] == "blocked"
            if admitted["status"] == "task":
                task = admitted["task"]
                p = candidate["proposal"]
                report["numeric_values_copied_unchanged"] = (
                    {x["term_id"]: x["weight"] for x in task["objective_terms"]} == p["objective_weights"]
                    and {x["constraint_id"]: x["bound"] for x in task["constraints"]} == p["constraint_bounds"])
        except urllib.error.HTTPError as exc:
            report["http_status"] = exc.code
            report["raw_http_error"] = exc.read().decode(errors="replace")
            report["response"] = {"status": "blocked", "reason_code": "model_request_failed",
                                  "message": f"Local model returned HTTP {exc.code}; no retry or repairs.", "hard_constraints_relaxed": False}
        except (urllib.error.URLError, TimeoutError, ValueError, KeyError, TypeError, IndexError) as exc:
            report["client_error"] = f"{type(exc).__name__}: {exc}"
            report["response"] = {"status": "blocked", "reason_code": "invalid_model_response",
                                  "message": report["client_error"], "hard_constraints_relaxed": False}
        report["request_seconds"] = time.monotonic() - started
        report["status"] = report["response"]["status"]
        return report


class TypedThinkingLocalPlanner(CompactLocalPlanner):
    """Explicit opt-in matching the measured typed+thinking input configuration.

    Uses separate prompt/schema files and a minimal context. Entity enums are
    filtered to the selected catalog layout. Legacy source/destination enums
    retain both A/B values; intent checks still enforce A->B. Rack/color pairs
    need external semantic checks even when each enum value is known.

    Server must also use --reasoning on --reasoning-format deepseek,
    --reasoning-budget 3072, --chat-template-kwargs '{"enable_thinking":true}'.
    Total reasoning+output budget is 4096, inside the 8192-token context after
    rendering/tokenizing the prompt. The class never starts or owns a server.
    """
    def __init__(self, registry=None, *, base_url="http://127.0.0.1:8085", timeout_s=300):
        super().__init__(registry=registry, base_url=base_url, timeout_s=timeout_s)
        self.system_prompt = (ROOT / "prompts/task_proposal_typed.txt").read_text()
        self.response_schema = json.loads((ROOT / "configs/task_proposal_typed.schema.json").read_text())
        from .command_executor import execution_capabilities
        capabilities = execution_capabilities(self.registry)
        fields = self.response_schema["oneOf"][0]["properties"]["proposal"]["properties"]
        fields["object_id"]["enum"] = list(capabilities["source_by_object"])
        stations = list(capabilities["source_by_object"].values())
        if capabilities["layout"] == "legacy":
            fields["source_id"]["enum"] = ["station_A", "station_B"]
            fields["destination_id"]["enum"] = ["station_A", "station_B"]
        else:
            fields["source_id"]["enum"] = stations
            fields["destination_id"]["enum"] = [capabilities["destination_id"]]
        fields["destination_slot_id"]["enum"] = [capabilities["destination_slot_id"]]
        self.planner_mode = "typed-thinking"
        self.default_max_tokens = 4096

    def context(self, now_s=0):
        return proposal_context(self.registry, now_s, include_route_guidance=False)

    def extra_request_fields(self):
        # Explicit request flags agree with the required server settings. They
        # affect template/parsing only; no proposal IDs or numbers are changed.
        return {"chat_template_kwargs": {"enable_thinking": True}, "reasoning_format": "deepseek"}

    def plan(self, command, *, now_s=0, requirements=None, max_tokens=4096):
        return super().plan(command, now_s=now_s, requirements=requirements, max_tokens=max_tokens)
