"""Static world catalog, dynamic observations and strict task admission.

No model client, robot command, solver or detector is invoked here. Successful
validation checks IDs, typed optimization choices and current observation
preconditions; executor availability, reach, contact and safety remain separate.
"""

from copy import deepcopy
import json
import math
from pathlib import Path
from threading import RLock


ROOT = Path(__file__).resolve().parents[2]


class RegistryError(ValueError):
    """Machine-readable rejection for feedback, without relaxing constraints."""

    def __init__(self, code, message):
        self.code = code
        super().__init__(f"{code}: {message}")

    def as_feedback(self):
        return {"status": "blocked", "reason_code": self.code,
                "message": str(self), "hard_constraints_relaxed": False}


def _reject(code, message):
    raise RegistryError(code, message)


def _number(value, label):
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        _reject("invalid_number", f"{label} must be a finite number")
    try:
        converted = float(value)
    except OverflowError:
        _reject("invalid_number", f"{label} is too large")
    if not math.isfinite(converted):
        _reject("invalid_number", f"{label} must be a finite number")
    return converted


def _keys(value, required, optional=(), label="value"):
    if not isinstance(value, dict):
        _reject("invalid_shape", f"{label} must be an object")
    missing, unknown = set(required) - value.keys(), value.keys() - set(required) - set(optional)
    if missing or unknown:
        _reject("invalid_shape", f"{label}: missing={sorted(missing)}, unknown={sorted(unknown)}")


class WorldRegistry:
    def __init__(self, catalog):
        self._catalog = deepcopy(catalog)
        self._entities = {}
        self._aliases = {}
        self._marker_owners = {}
        self._lock = RLock()
        for entity in self._catalog["entities"]:
            entity_id = entity["id"]
            if entity_id in self._entities:
                _reject("duplicate_entity", entity_id)
            self._entities[entity_id] = entity
            for alias in [entity_id, entity["name"], *entity["aliases"]]:
                key = alias.casefold().strip()
                if key in self._aliases and self._aliases[key] != entity_id:
                    _reject("ambiguous_alias", alias)
                self._aliases[key] = entity_id
            marker_id = entity.get("marker_id")
            if marker_id is not None:
                if isinstance(marker_id, bool) or not isinstance(marker_id, int) or marker_id < 0:
                    _reject("invalid_marker", str(marker_id))
                if marker_id in self._marker_owners:
                    _reject("marker_conflict", f"marker {marker_id} belongs to multiple entities")
                self._marker_owners[marker_id] = entity_id
        self._observations = {entity_id: None for entity_id in self._entities}
        self._skills = {entry["id"]: entry for entry in self._catalog["skills"]}
        self._terms = {entry["id"]: entry for entry in self._catalog["objective_terms"]}
        self._constraints = {entry["id"]: entry for entry in self._catalog["constraint_types"]}

    @classmethod
    def load(cls, path=None):
        return cls(json.loads(Path(path or ROOT / "configs/factory_world.json").read_text()))

    def entity(self, entity_id, kind=None):
        if not isinstance(entity_id, str) or entity_id not in self._entities:
            _reject("unknown_entity", str(entity_id))
        entity = self._entities[entity_id]
        if kind is not None and entity["kind"] != kind:
            _reject("wrong_entity_kind", f"{entity_id} must be {kind}")
        return deepcopy(entity)

    def resolve(self, name_or_alias):
        if not isinstance(name_or_alias, str) or name_or_alias.casefold().strip() not in self._aliases:
            _reject("unknown_entity", str(name_or_alias))
        return self._aliases[name_or_alias.casefold().strip()]

    def record_observation(self, entity_id, observation):
        """Called by a detector/localizer/GT baseline, never by the LLM.

        Pose is position_m plus quaternion_wxyz in frame_id. Invisible reports
        may set pose=None. Robot/map transformations are the bridge's job.
        """
        entity = self.entity(entity_id)
        _keys(observation, ("pose", "frame_id", "stamp_s", "clock_id", "source", "visible"),
              ("marker_id", "reprojection_error_px", "stale", "station_id"), "observation")
        item = deepcopy(observation)
        stamp = _number(item["stamp_s"], "observation.stamp_s")
        if stamp < 0 or not isinstance(item["frame_id"], str) or not item["frame_id"]:
            _reject("invalid_observation", "nonnegative timestamp and a frame are required")
        if item["clock_id"] != self._catalog["world_design"]["clock_id"]:
            _reject("clock_mismatch", str(item["clock_id"]))
        if item["source"] not in self._catalog["observation_policy"]["sources"]:
            _reject("unknown_observation_source", str(item["source"]))
        if not isinstance(item["visible"], bool) or not isinstance(item.get("stale", False), bool):
            _reject("invalid_observation", "visible/stale must be booleans")
        marker = item.get("marker_id")
        if marker is not None and (isinstance(marker, bool) or not isinstance(marker, int)
                                   or self._marker_owners.get(marker) != entity_id):
            _reject("marker_conflict", f"marker {marker} does not identify {entity_id}")
        if item["source"] == "virtual_aruco" and (entity.get("marker_id") is None or marker != entity["marker_id"]):
            _reject("marker_conflict", "ArUco observation must contain the catalog marker")
        error = item.get("reprojection_error_px")
        if error is not None and _number(error, "reprojection_error_px") < 0:
            _reject("invalid_observation", "negative reprojection error")
        if item["source"] == "virtual_aruco" and error is None:
            _reject("invalid_observation", "ArUco quality metric is required")
        if item.get("station_id") is not None:
            self.entity(item["station_id"], "station")
        pose = item["pose"]
        if pose is None:
            if item["visible"]:
                _reject("invalid_observation", "visible observation needs a pose")
        else:
            _keys(pose, ("position_m", "quaternion_wxyz"), label="pose")
            for key, length in (("position_m", 3), ("quaternion_wxyz", 4)):
                if not isinstance(pose[key], list) or len(pose[key]) != length:
                    _reject("invalid_observation", f"pose.{key} must have length {length}")
                for value in pose[key]:
                    _number(value, f"pose.{key}")
            if abs(math.sqrt(sum(x * x for x in pose["quaternion_wxyz"])) - 1) > 1e-3:
                _reject("invalid_observation", "pose quaternion must be normalized")
        item.setdefault("stale", False)
        item.setdefault("marker_id", None)
        item.setdefault("reprojection_error_px", None)
        with self._lock:
            previous = self._observations[entity_id]
            if previous is not None and stamp < previous["stamp_s"]:
                _reject("out_of_order_observation", entity_id)
            self._observations[entity_id] = item

    def observation(self, entity_id):
        """Return a copy of the current observation, or None before observation."""
        self.entity(entity_id)
        with self._lock:
            return deepcopy(self._observations[entity_id])

    def observation_status(self, entity_id, now_s):
        now = _number(now_s, "now_s")
        if now < 0:
            _reject("invalid_number", "now_s must be nonnegative")
        item = self.observation(entity_id)
        if item is None:
            return {"state": "unobserved", "fresh": False}
        policy = self._catalog["observation_policy"]
        age = now - item["stamp_s"]
        state = "fresh"
        if not item["visible"] or item["pose"] is None:
            state = "not_visible"
        elif item["stale"]:
            state = "marked_stale"
        elif age < -policy["max_future_stamp_design_s"]:
            state = "future_timestamp"
        elif age > policy["max_age_design_s"]:
            state = "stale"
        elif item["frame_id"] != self._catalog["world_design"]["frame_id"]:
            state = "transform_required"
        elif item["reprojection_error_px"] is not None and item["reprojection_error_px"] > policy["max_marker_reprojection_error_design_px"]:
            state = "poor_marker_quality"
        return {"state": state, "fresh": state == "fresh", "age_s": age,
                "frame_id": item["frame_id"], "stamp_s": item["stamp_s"],
                "source": item["source"], "visible": item["visible"],
                "reprojection_error_px": item["reprojection_error_px"]}

    def require_fresh_observation(self, entity_id, now_s):
        with self._lock:
            status = self.observation_status(entity_id, now_s)
            if not status["fresh"]:
                _reject("observation_not_ready", f"{entity_id}: {status['state']}")
            return self.observation(entity_id)

    def llm_context(self, now_s):
        """No dynamic XYZ/quaternion: the executor resolves sensor state by ID."""
        with self._lock:
            return {"static_catalog": deepcopy(self._catalog),
                    "observation_status": {key: self.observation_status(key, now_s) for key in self._entities},
                    "coordinate_policy": "Dynamic object/robot poses are retrieved by the executor, not stored in model weights or invented in task output. Catalog world coordinates are unvalidated factory design."}

    def _validate_grasp(self, step, object_entity):
        robot = self.entity("robot_rby1", "robot")
        if not isinstance(step["arm"], str) or step["arm"] not in robot["grippers"]:
            _reject("unknown_arm", str(step["arm"]))
        face_id = step["grasp_face_id"]
        face = object_entity["grasp_faces"].get(face_id) if isinstance(face_id, str) else None
        if face is None:
            _reject("unknown_grasp_face", str(step["grasp_face_id"]))
        opening = _number(step["gripper_opening_m"], "gripper_opening_m")
        gripper = robot["grippers"][step["arm"]]
        low, high = gripper["opening_design_range_m"]
        needed = face["width_m"] + 2 * gripper["approach_clearance_each_side_m"]
        if not low <= opening <= high or opening < needed:
            _reject("impossible_grasp_width", f"need >= {needed} m and design opening in [{low}, {high}] m")

    def validate_task_spec(self, spec, *, now_s=None, require_fresh_observations=True):
        """Validate a task's contract, including observations by default.

        Set require_fresh_observations=False only for planning/review. A return
        value is not permission to skip per-skill freshness, executor readiness,
        geometric reach, physical collision or sensor/braking validation.
        """
        _keys(spec, ("schema_version", "object_id", "source_id", "destination_id",
                     "ordered_skills", "objective_terms", "constraints", "forbidden_zone_ids"), label="task")
        if isinstance(spec["schema_version"], bool) or spec["schema_version"] != 1:
            _reject("unsupported_schema", str(spec["schema_version"]))
        object_entity = self.entity(spec["object_id"], "object")
        source = self.entity(spec["source_id"], "station")
        destination = self.entity(spec["destination_id"], "station")
        if source["id"] == destination["id"]:
            _reject("invalid_task", "source and destination must differ for this transport interface")
        for field in ("ordered_skills", "objective_terms", "constraints", "forbidden_zone_ids"):
            if not isinstance(spec[field], list) or (field != "forbidden_zone_ids" and not spec[field]):
                _reject("invalid_shape", f"{field} must be {'a' if field == 'forbidden_zone_ids' else 'a nonempty'} list")
        if len(spec["ordered_skills"]) > 32:
            _reject("invalid_shape", "at most 32 ordered skills are accepted")
        zones = set()
        for zone_id in spec["forbidden_zone_ids"]:
            self.entity(zone_id, "zone")
            if zone_id in zones:
                _reject("duplicate_zone", zone_id)
            zones.add(zone_id)
        if set(self._catalog["world_design"]["candidate_corridor_zone_ids"]) <= zones:
            _reject("no_allowed_route", "all declared candidate corridors are forbidden")
        weights = {}
        low, high = self._catalog["objective_policy"]["weight_design_range"]
        for term in spec["objective_terms"]:
            _keys(term, ("term_id", "weight"), label="objective term")
            term_id = term["term_id"]
            if not isinstance(term_id, str) or term_id not in self._terms:
                _reject("unknown_objective", str(term_id))
            if term_id in weights:
                _reject("duplicate_objective", term_id)
            weight = _number(term["weight"], "weight")
            if not low <= weight <= high:
                _reject("objective_weight_out_of_range", term_id)
            weights[term_id] = weight
        for required in self._catalog["objective_policy"]["required_positive_terms"]:
            if weights.get(required, 0) <= 0:
                _reject("missing_objective", f"positive {required} term required")
        bounds = {}
        for constraint in spec["constraints"]:
            _keys(constraint, ("constraint_id", "unit", "bound"), label="constraint")
            constraint_id = constraint["constraint_id"]
            if not isinstance(constraint_id, str) or constraint_id not in self._constraints:
                _reject("unknown_constraint", str(constraint_id))
            if constraint_id in bounds:
                _reject("duplicate_constraint", constraint_id)
            definition = self._constraints[constraint_id]
            if constraint["unit"] != definition["unit"]:
                _reject("unit_mismatch", constraint_id)
            bound = _number(constraint["bound"], "constraint.bound")
            if not definition["bound_design_range"][0] <= bound <= definition["bound_design_range"][1]:
                _reject("constraint_out_of_range", constraint_id)
            bounds[constraint_id] = bound
        for constraint_id, definition in self._constraints.items():
            if definition["required"] and constraint_id not in bounds:
                _reject("missing_constraint", constraint_id)
        location, object_state, stowed, observed_at_location = None, "source", False, False
        for step in spec["ordered_skills"]:
            if not isinstance(step, dict) or not isinstance(step.get("skill_id"), str) or step["skill_id"] not in self._skills:
                _reject("unknown_skill", str(step))
            skill = step["skill_id"]
            _keys(step, ("skill_id", *self._skills[skill]["required_arguments"]), label=f"skill {skill}")
            if "object_id" in step and step["object_id"] != object_entity["id"]:
                _reject("object_mismatch", str(step["object_id"]))
            if "station_id" in step:
                self.entity(step["station_id"], "station")
            if "slot_id" in step:
                slot = self.entity(step["slot_id"], "slot")
            if skill in ("pick", "pick_from_tray"):
                self._validate_grasp(step, object_entity)
            if skill == "navigate":
                if step["station_id"] not in (source["id"], destination["id"]):
                    _reject("invalid_skill_order", "navigate target outside the transport task")
                if object_state == "carried" or (object_state == "tray" and not stowed):
                    _reject("invalid_skill_order", "driving requires tray loading and stowed arms")
                location = step["station_id"]
                observed_at_location = False
            elif skill == "observe":
                if location is None:
                    _reject("invalid_skill_order", "observe requires arrival at a station")
                observed_at_location = True
            elif skill == "pick":
                if location != source["id"] or object_state != "source" or not observed_at_location:
                    _reject("invalid_skill_order", "pick requires source arrival and observation")
                object_state, stowed = "carried", False
            elif skill == "place_on_tray":
                if object_state != "carried" or slot["slot_type"] != "tray" or slot["owner_id"] != "robot_rby1":
                    _reject("invalid_skill_order", "load requires held object and robot tray slot")
                object_state = "tray"
            elif skill == "stow_arms":
                if object_state != "tray":
                    _reject("invalid_skill_order", "stow requires object released on tray")
                stowed = True
            elif skill == "pick_from_tray":
                if location != destination["id"] or object_state != "tray" or not observed_at_location or slot["slot_type"] != "tray" or slot["owner_id"] != "robot_rby1":
                    _reject("invalid_skill_order", "unload requires arrival at destination and loaded tray")
                object_state, stowed = "carried", False
            elif skill in ("place", "verify_place"):
                expected_state = "carried" if skill == "place" else "placed"
                if (location != destination["id"] or step["station_id"] != destination["id"]
                        or object_state != expected_state or slot["owner_id"] != destination["id"] or slot["slot_type"] != "table"):
                    _reject("invalid_skill_order", "destination place/verify arguments or state mismatch")
                object_state = "placed" if skill == "place" else "verified"
        if object_state != "verified":
            _reject("incomplete_task", "transport must end with destination verification")
        if require_fresh_observations:
            if now_s is None:
                _reject("observation_not_ready", "now_s in the simulation clock is required")
            with self._lock:
                self.require_fresh_observation("robot_rby1", now_s)
                observation = self.require_fresh_observation(object_entity["id"], now_s)
                if observation.get("station_id") != source["id"]:
                    _reject("source_observation_mismatch", "object's observed station must match task source")
        return deepcopy(spec)
