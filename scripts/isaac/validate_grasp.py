#!/usr/bin/env python3
"""Re-evaluate a saved stationary grasp trace without importing Isaac or running it.

API: validate(samples, scene, dt=.002) -> {passed, metrics, checks, scope}.
CLI: --input RUN_DIR --output NEW_JSON; exit 0 passes, 1 rejects a trace,
2 reports an input/output error. Existing files are never overwritten.

Contact values are the recorded simulation measurements, not calibrated hardware
forces. A trace cannot establish that its producer used no attachment/pose writes;
that requires a separate code/runtime audit. Whole-robot collisions and hardware
safety are outside this evaluator. The table is the scene's static horizontal box.
Without saved object rotations, its half-diagonal bounds every orientation: a
positive edge margin proves containment, while a negative margin is a conservative
rejection and does not by itself prove that an actual corner overhangs.
"""
from __future__ import annotations

import argparse
from collections.abc import Mapping
import hashlib
import json
import math
from pathlib import Path
import sys


PHASE_DURATIONS_S = {
    "settle": 2.0, "approach": 3.0, "close": 2.0, "lift": 3.0,
    "hold": 3.0, "transfer": 3.0, "lower": 3.0, "open": 1.0,
    "retreat": 2.0, "place_settle": 1.0,
}
EXPECTED_DT_S = .002
CONTACT_THRESHOLD_N = .02
MAX_CONTACT_LOSS_S = .10
TABLE_EDGE_MARGIN_M = .005
MAX_RELATIVE_TRANSLATION_M = .01
MAX_RELATIVE_ROTATION_RAD = math.radians(10)


def _scope():
    return {
        "claim": "Saved-trace evaluation of one stationary pick, hold, transfer and place case.",
        "sampling": "Actual timestamp intervals and phase durations are checked against the 2 ms trace contract.",
        "geometry": "Static axis-aligned tabletop; object half-diagonal conservatively bounds all orientations.",
        "not_verified": [
            "Absence of attachment or pose overwrites in the trace producer; requires source/runtime audit.",
            "Whole-robot forbidden contacts, self-collision or complete hand/housing collision geometry.",
            "Calibrated force limits, hardware safety, success rate or real-time performance.",
            "Tray transport, driving, perception, QP or LLM task execution.",
        ],
    }


def _number(value, label, *, nonnegative=False):
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{label} must be a finite number")
    value = float(value)
    if not math.isfinite(value) or (nonnegative and value < 0):
        raise ValueError(f"{label} must be finite" + (" and nonnegative" if nonnegative else ""))
    return value


def _vector(value, length, label, *, positive=False, nonnegative=False):
    if not isinstance(value, (tuple, list)) or len(value) != length:
        raise ValueError(f"{label} must contain {length} numbers")
    result = tuple(_number(x, f"{label}[{i}]", nonnegative=nonnegative) for i, x in enumerate(value))
    if positive and any(x <= 0 for x in result):
        raise ValueError(f"{label} must contain positive dimensions")
    return result


def _norm(value):
    result = math.hypot(*value)
    if not math.isfinite(result):
        raise ValueError("Derived vector norm is non-finite")
    return result


def _maximum_loss(rows, dt):
    """Discrete loss duration, valid only after every actual timestamp step passes."""
    longest = consecutive = 0
    for row in rows:
        consecutive = consecutive + 1 if min(row["finger_normal_force_n"]) <= CONTACT_THRESHOLD_N else 0
        longest = max(longest, consecutive)
    return longest * dt


def validate(samples, scene, dt=.002):
    """Return an explicit rejection for malformed, incomplete or failed traces.

    Existing hold/place thresholds are retained. Added carry checks use the
    producer's hold-relative pose measurements through transfer/lower, check
    airborne table-force absence, and bound continuous bilateral contact loss
    across phase boundaries. Thresholds are evaluation choices, not robot ratings.
    No input object is modified and no simulator/API is imported.
    """
    result = {"passed": False, "metrics": {}, "checks": {"input_fields_valid": False}, "scope": _scope()}
    metrics, checks = result["metrics"], result["checks"]
    try:
        dt = _number(dt, "dt")
        if dt <= 0:
            raise ValueError("dt must be positive")
        if not isinstance(scene, Mapping) or not isinstance(samples, (list, tuple)) or not samples:
            raise ValueError("Nonempty samples and a scene mapping are required")
        size = _vector(scene["object_full_size_m"], 3, "object_full_size_m", positive=True)
        table_size = _vector(scene["table_full_size_m"], 3, "table_full_size_m", positive=True)
        center = _vector(scene["table_center_xy_m"], 2, "table_center_xy_m")
        destination = _vector(scene["place_target_world_center_m"], 3, "place_target_world_center_m")
        _number(scene["table_top_world_z_m"], "table_top_world_z_m")
        nonnegative_scalars = (
            "tcp_error_m", "tcp_rotation_error_rad", "support_force_n",
            "relative_translation_drift_m", "relative_rotation_drift_rad", "base_xy_drift_m",
        )
        for i, row in enumerate(samples):
            if not isinstance(row, Mapping) or row.get("phase") not in PHASE_DURATIONS_S:
                raise ValueError(f"sample {i} has an unknown/missing phase")
            _number(row["time"], f"sample {i}.time", nonnegative=True)
            for name in nonnegative_scalars:
                _number(row[name], f"sample {i}.{name}", nonnegative=True)
            for name in ("clearance_m", "base_up_z"):
                _number(row[name], f"sample {i}.{name}")
            for name in ("object", "tcp", "object_velocity", "object_angular_velocity"):
                _vector(row[name], 3, f"sample {i}.{name}")
            _vector(row["finger_normal_force_n"], 2, f"sample {i}.finger_normal_force_n", nonnegative=True)
        checks["input_fields_valid"] = True
        times = [float(row["time"]) for row in samples]
        intervals = [b - a for a, b in zip(times, times[1:])]
        groups = {phase: [] for phase in PHASE_DURATIONS_S}
        sequence = []
        for row in samples:
            phase = row["phase"]
            groups[phase].append(row)
            if not sequence or sequence[-1] != phase:
                sequence.append(phase)
        counts = {name: len(rows) for name, rows in groups.items()}
        durations = {name: float(rows[-1]["time"] - rows[0]["time"] + dt) if rows else 0.0
                     for name, rows in groups.items()}
        metrics.update(
            samples=len(samples), requested_dt_s=dt,
            minimum_recorded_dt_s=min(intervals) if intervals else None,
            maximum_recorded_dt_s=max(intervals) if intervals else None,
            phase_sequence=sequence, phase_sample_counts=counts, phase_durations_s=durations,
            total_recorded_duration_s=times[-1] - times[0] + dt,
            thresholds={"contact_n": CONTACT_THRESHOLD_N, "maximum_continuous_loss_s": MAX_CONTACT_LOSS_S,
                        "table_edge_margin_m": TABLE_EDGE_MARGIN_M,
                        "relative_translation_m": MAX_RELATIVE_TRANSLATION_M,
                        "relative_rotation_rad": MAX_RELATIVE_ROTATION_RAD},
        )
        checks.update(
            sampling_period_is_2ms=math.isclose(dt, EXPECTED_DT_S, rel_tol=0, abs_tol=1e-12),
            recorded_timestamps_are_2ms=bool(intervals) and all(abs(x - dt) <= max(1e-8, dt * 1e-5) for x in intervals),
            complete_ordered_phases=sequence == list(PHASE_DURATIONS_S),
            complete_phase_sample_counts=all(counts[p] == round(seconds / dt) for p, seconds in PHASE_DURATIONS_S.items()),
            complete_phase_durations=all(abs(durations[p] - seconds) <= max(2e-6, seconds * 1e-5)
                                         for p, seconds in PHASE_DURATIONS_S.items()),
            final_rest_window_is_1s=abs(durations["place_settle"] - 1.0) <= 1e-5,
        )
        if not all(checks.values()):
            return result

        hold, transfer, lower, place = (groups[p] for p in ("hold", "transfer", "lower", "place_settle"))
        carry = hold + transfer + lower
        half_diagonal = .5 * _norm(size)
        xmin, xmax = center[0] - table_size[0] / 2, center[0] + table_size[0] / 2
        ymin, ymax = center[1] - table_size[1] / 2, center[1] + table_size[1] / 2
        edge_margin = min(min(r["object"][0] - xmin, xmax - r["object"][0],
                              r["object"][1] - ymin, ymax - r["object"][1]) - half_diagonal for r in place)
        metrics.update(
            minimum_hold_clearance_m=min(r["clearance_m"] for r in hold),
            minimum_transfer_clearance_m=min(r["clearance_m"] for r in transfer),
            maximum_airborne_table_force_n=max(r["support_force_n"] for r in hold + transfer),
            max_hold_relative_translation_m=max(r["relative_translation_drift_m"] for r in hold),
            max_hold_relative_rotation_rad=max(r["relative_rotation_drift_rad"] for r in hold),
            max_carry_relative_translation_m=max(r["relative_translation_drift_m"] for r in carry),
            max_carry_relative_rotation_rad=max(r["relative_rotation_drift_rad"] for r in carry),
            maximum_continuous_carry_contact_loss_s=_maximum_loss(carry, dt),
            final_xy_error_m=_norm([place[-1]["object"][i] - destination[i] for i in (0, 1)]),
            max_final_clearance_error_m=max(abs(r["clearance_m"]) for r in place),
            minimum_final_support_force_n=min(r["support_force_n"] for r in place),
            max_final_finger_force_n=max(max(r["finger_normal_force_n"]) for r in place),
            max_final_speed_m_s=max(_norm(r["object_velocity"]) for r in place),
            max_final_angular_speed_rad_s=max(_norm(r["object_angular_velocity"]) for r in place),
            object_bounding_radius_m=half_diagonal,
            minimum_final_table_edge_margin_lower_bound_m=edge_margin,
            maximum_base_xy_drift_m=max(r["base_xy_drift_m"] for r in samples),
            minimum_base_up_z=min(r["base_up_z"] for r in samples),
        )
        for phase in ("hold", "transfer", "lower"):
            rows = groups[phase]
            metrics[f"{phase}_bilateral_contact_fraction"] = sum(min(r["finger_normal_force_n"]) > CONTACT_THRESHOLD_N for r in rows) / len(rows)
            metrics[f"{phase}_maximum_continuous_contact_loss_s"] = _maximum_loss(rows, dt)
            checks[f"{phase}_bilateral_contact"] = metrics[f"{phase}_bilateral_contact_fraction"] >= .9
            checks[f"{phase}_continuous_contact_loss"] = metrics[f"{phase}_maximum_continuous_contact_loss_s"] <= MAX_CONTACT_LOSS_S + 1e-10
        checks.update(
            close_bilateral_contact=min(groups["close"][-1]["finger_normal_force_n"]) >= CONTACT_THRESHOLD_N,
            lift=metrics["minimum_hold_clearance_m"] > .05,
            hold_translation=metrics["max_hold_relative_translation_m"] < MAX_RELATIVE_TRANSLATION_M,
            hold_rotation=metrics["max_hold_relative_rotation_rad"] < MAX_RELATIVE_ROTATION_RAD,
            transfer_clearance=metrics["minimum_transfer_clearance_m"] > .05,
            airborne_table_contact_absent=metrics["maximum_airborne_table_force_n"] < CONTACT_THRESHOLD_N,
            carry_translation=metrics["max_carry_relative_translation_m"] < MAX_RELATIVE_TRANSLATION_M,
            carry_rotation=metrics["max_carry_relative_rotation_rad"] < MAX_RELATIVE_ROTATION_RAD,
            carry_continuous_contact_loss=metrics["maximum_continuous_carry_contact_loss_s"] <= MAX_CONTACT_LOSS_S + 1e-10,
            place_xy=metrics["final_xy_error_m"] < .02,
            place_height=metrics["max_final_clearance_error_m"] < .01,
            place_supported=metrics["minimum_final_support_force_n"] > CONTACT_THRESHOLD_N,
            released=metrics["max_final_finger_force_n"] < CONTACT_THRESHOLD_N,
            at_rest=metrics["max_final_speed_m_s"] < .02 and metrics["max_final_angular_speed_rad_s"] < .2,
            fully_on_table_conservative=metrics["minimum_final_table_edge_margin_lower_bound_m"] >= TABLE_EDGE_MARGIN_M,
            stationary_base=metrics["maximum_base_xy_drift_m"] <= .03 and metrics["minimum_base_up_z"] >= .98,
        )
        for phase in ("settle", "approach", "lower"):
            checks[f"{phase}_tcp_reached"] = groups[phase][-1]["tcp_error_m"] <= .004
        result["passed"] = all(checks.values())
        return result
    except (KeyError, ValueError, TypeError, OverflowError) as error:
        checks["input_fields_valid"] = False
        metrics["validation_error"] = f"{type(error).__name__}: {error}"
        return result


def main():
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    parser.add_argument("--input", required=True, type=Path, help="Run directory with state_samples.json and summary.json")
    parser.add_argument("--output", required=True, type=Path, help="New JSON path; never overwrite")
    args = parser.parse_args()
    output = args.output.absolute()
    try:
        if output.exists() or output.is_symlink():
            raise FileExistsError(f"Refusing to overwrite {output}")
        run = args.input.resolve()
        trace_path, summary_path = run / "state_samples.json", run / "summary.json"
        trace_bytes, summary_bytes = trace_path.read_bytes(), summary_path.read_bytes()
        samples = json.loads(trace_bytes)
        original_summary = json.loads(summary_bytes)
        report = validate(samples, original_summary["scene"])
        report["provenance"] = {
            "input_directory": str(run),
            "trace_sha256": hashlib.sha256(trace_bytes).hexdigest(),
            "original_summary_sha256": hashlib.sha256(summary_bytes).hexdigest(),
            "evaluator_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            "original_reported_passed": original_summary.get("passed"),
            "new_physics_execution": False,
        }
        encoded = json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False) + "\n"
        output.parent.mkdir(parents=True, exist_ok=True)
        with output.open("x") as stream:
            stream.write(encoded)
        print(json.dumps({"passed": report["passed"], "output": str(output),
                          "failed_checks": [k for k, v in report["checks"].items() if not v]}))
        return 0 if report["passed"] else 1
    except (OSError, ValueError, KeyError, TypeError) as error:
        print(f"Validation I/O error: {type(error).__name__}: {error}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
