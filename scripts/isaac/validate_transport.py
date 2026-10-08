#!/usr/bin/env python3
"""Independently evaluate saved Isaac load/drive/full transport measurements.

API: validate(samples, scene, transport, *, mode="load", navigation_trace=None,
approach_trace=None, drive_samples=None, phase_durations_s=None, dt=.002, nav_dt=.02).
Pass manipulation_contact_monitor when the summary declares that monitor.
Pass rest_definition_revision and final_rest_evidence forwindowed-rest-v1.
No Isaac, ROS, solver, NumPy or LLM is imported. CLI output must be a new file.
Legacy load traces without saved poses can verify their recorded load gates,
but cannot verify orientation or world-to-tray transforms. Full mode requires
those poses and a complete navigation trace; a producer's passed flag is ignored.
"""
from __future__ import annotations

import argparse
from collections.abc import Mapping
import hashlib
import json
import math
from pathlib import Path
import sys

LOAD_DURATIONS = {
    "settle": 2., "approach": 3., "approach_settle": 2., "close": 2.,
    "lift": 3., "hold": 3., "transfer": 7., "transfer_settle": 2.,
    "lower": 3., "lower_settle": 2., "open": 1., "retreat": 2., "place_settle": 1.,
}
UNLOAD_DURATIONS = {
    "unload_hover": 6., "unload_hover_settle": 3., "unload_approach": 6.,
    "unload_approach_settle": 3., "unload_close": 3., "unload_lift": 6.,
    "unload_hold": 3., "unload_transfer": 10., "unload_transfer_settle": 3.,
    "unload_lower": 6., "unload_lower_settle": 3., "unload_open": 2.,
    "unload_retreat": 4., "unload_place_settle": 2.,
}
DRIVE_DURATIONS = {"reverse": 4., "brake": 2., "turn": 4., "rest": 2.}
POSE_FIELDS = ("object_quaternion_wxyz", "tray_position_m", "tray_quaternion_wxyz")
MANIPULATION_CONTACT_FIELD = "robot_environment_contact_max_n"
BASE_VELOCITY_FIELDS = ("base_linear_velocity_m_s", "base_angular_velocity_rad_s")
_MONITOR_ABSENT = object()
_REST_ABSENT = object()
REST_REVISION = "windowed-rest-v1"
REST_CONFIG = {
    "window_s": .05, "quiet_duration_s": .5,
    "filtered_speed_max_m_s": .006, "filtered_yaw_rate_max_rad_s": .015,
    "raw_speed_max_m_s": .02, "raw_yaw_rate_max_rad_s": .04,
    "position_excursion_max_m": .001, "yaw_excursion_max_rad": .005,
    "max_gap_s": .0041, "averaging": "piecewise-linear-trapezoid",
    "position_excursion": "xy-bounding-box-diagonal",
}
THRESHOLDS = {
    "contact_n": .02, "final_xy_m": .015, "final_z_m": .005,
    "rest_speed_m_s": .01, "rest_angular_speed_rad_s": .2,
    "hold_clearance_m": .05, "tray_x_m": .065, "tray_y_m": .060,
    "tray_rest_z_error_m": .010, "tray_navigation_z_error_m": .012,
    "upright_z": .98, "stationary_base_drift_m": .03,
    "tcp_reach_m": .004, "arm_tracking_rad": .08,
    "hold_translation_drift_m": .01, "hold_rotation_drift_rad": math.radians(10),
    # Loose continuity tolerances accommodate contact acceleration and sampling;
    # they reject gross recorded jumps, rather than proving absence of pose writes.
    "manipulation_position_velocity_residual_m": .001,
    "navigation_position_velocity_residual_m": .004,
    "world_to_tray_residual_m": .0002, "orientation_step_allowance_rad": .02,
    "navigation_contact_max_n": .05, "approach_xy_m": .015,
    "approach_yaw_rad": .02, "navigation_rest_speed_m_s": .006,
    "navigation_rest_yaw_rate_rad_s": .015, "navigation_rest_window_s": .5,
}


def _number(value, label, *, nonnegative=False):
    if isinstance(value, bool) or not isinstance(value, (float, int)):
        raise ValueError(f"{label} must be numeric, not boolean")
    value = float(value)
    if not math.isfinite(value) or (nonnegative and value < 0):
        raise ValueError(f"{label} must be finite" + (" and nonnegative" if nonnegative else ""))
    return value


def _vector(value, count, label):
    if not isinstance(value, (list, tuple)) or len(value) != count:
        raise ValueError(f"{label} must contain {count} numbers")
    return tuple(_number(x, f"{label}[{i}]") for i, x in enumerate(value))


def _finite_tree(value, label):
    if isinstance(value, float) and not math.isfinite(value):
        raise ValueError(f"{label} contains NaN/infinity")
    if isinstance(value, Mapping):
        for key, item in value.items():
            _finite_tree(item, f"{label}.{key}")
    elif isinstance(value, (list, tuple)):
        for i, item in enumerate(value):
            _finite_tree(item, f"{label}[{i}]")


def _quaternion(value, label):
    q = _vector(value, 4, label)
    norm = math.hypot(*q)
    if abs(norm - 1.) > .001:
        raise ValueError(f"{label} quaternion is not normalized")
    return tuple(x / norm for x in q)


def _rotation(q):
    w, x, y, z = q
    return ((1-2*(y*y+z*z), 2*(x*y-w*z), 2*(x*z+w*y)),
            (2*(x*y+w*z), 1-2*(x*x+z*z), 2*(y*z-w*x)),
            (2*(x*z-w*y), 2*(y*z+w*x), 1-2*(x*x+y*y)))


def _in_tray(vector, rotation):
    return tuple(sum(rotation[j][i] * vector[j] for j in range(3)) for i in range(3))


def _validate_pose(row, label, *, required):
    present = [key in row for key in POSE_FIELDS]
    if not any(present) and not required:
        return False
    if not all(present):
        raise ValueError(f"{label} lacks complete object/tray world poses")
    _quaternion(row[POSE_FIELDS[0]], label + ".object_quaternion_wxyz")
    _vector(row[POSE_FIELDS[1]], 3, label + ".tray_position_m")
    _quaternion(row[POSE_FIELDS[2]], label + ".tray_quaternion_wxyz")
    return True


def _rows_valid(rows, *, manipulation, poses_required):
    if not isinstance(rows, (list, tuple)) or not rows:
        raise ValueError("A nonempty measurement list is required")
    pose_count = 0
    for i, row in enumerate(rows):
        if not isinstance(row, Mapping) or not isinstance(row.get("phase"), str):
            raise ValueError(f"row {i} has no phase")
        _finite_tree(row, f"row {i}")
        _number(row["time"], f"row {i}.time", nonnegative=True)
        for key in ("object", "object_velocity", "object_angular_velocity", "object_in_tray"):
            _vector(row[key], 3, f"row {i}.{key}")
        _number(row["tray_support_force_n"], f"row {i}.tray_support_force_n", nonnegative=True)
        _number(row["base_up_z"], f"row {i}.base_up_z")
        if manipulation:
            for key in ("tcp",):
                _vector(row[key], 3, f"row {i}.{key}")
            for key in ("arm_q", "arm_reference"):
                _vector(row[key], 7, f"row {i}.{key}")
            force = _vector(row["finger_normal_force_n"], 2, f"row {i}.finger_normal_force_n")
            if min(force) < 0:
                raise ValueError("Contact force norms cannot be negative")
            for key in ("tcp_error_m", "base_xy_drift_m", "support_force_n",
                        "relative_translation_drift_m", "relative_rotation_drift_rad"):
                _number(row[key], f"row {i}.{key}", nonnegative=True)
            _number(row["clearance_m"], f"row {i}.clearance_m")
        pose_count += _validate_pose(row, f"row {i}", required=poses_required)
    if pose_count not in (0, len(rows)):
        raise ValueError("World-pose fields disappear during the trace")
    return pose_count == len(rows)


def _manipulation_monitor_rows(rows, monitor=_MONITOR_ABSENT):
    """Validate optional complete field families, or all fields when declared.

    A legacy trace cannot gain a contact/rest certificate from absent fields.
    Presence on ANY row makes that family mandatory on EVERY row. Contact and
    velocity families can be optional separately for a new narrow-layout run.
    """
    declared = monitor is not _MONITOR_ABSENT
    force_limit = THRESHOLDS["navigation_contact_max_n"]
    info = {"declared": declared, "contact_saved": False, "velocity_saved": False,
            "force_limit_n": force_limit}
    if declared:
        if not isinstance(monitor, Mapping):
            raise ValueError("Declared manipulation_contact_monitor must be a mapping")
        interval = _number(monitor["interval_s"], "manipulation monitor interval")
        if abs(interval-.002)>1e-7:
            raise ValueError("Manipulation contact monitor must declare2ms sampling")
        limit = _number(monitor["maximum_admitted_force_n"], "manipulation monitor force limit", nonnegative=True)
        if limit>force_limit:
            raise ValueError("Declared manipulation force limit exceeds0.05N")
        filters = monitor["filter_paths"]
        if not isinstance(filters, (list, tuple)) or not filters or any(not isinstance(p, str) or not p.startswith("/") for p in filters):
            raise ValueError("Manipulation monitor requires nonempty absolute contact filter paths")
        if "includes_people" in monitor and not isinstance(monitor["includes_people"], bool):
            raise ValueError("Manipulation includes_people must be boolean")
        info.update(force_limit_n=limit, declared_filter_path_count=len(filters),
                    declared_includes_people=monitor.get("includes_people", False))
    contact_present = declared or any(MANIPULATION_CONTACT_FIELD in row for row in rows)
    velocity_present = declared or any(any(field in row for field in BASE_VELOCITY_FIELDS) for row in rows)
    for i, row in enumerate(rows):
        if contact_present:
            if MANIPULATION_CONTACT_FIELD not in row:
                raise ValueError(f"Manipulation contact field disappears/missing at row{i}")
            _number(row[MANIPULATION_CONTACT_FIELD], f"row{i}.{MANIPULATION_CONTACT_FIELD}", nonnegative=True)
        if velocity_present:
            if not all(field in row for field in BASE_VELOCITY_FIELDS):
                raise ValueError(f"Manipulation base velocity pair disappears/missing at row{i}")
            for field in BASE_VELOCITY_FIELDS:
                _vector(row[field], 3, f"row{i}.{field}")
    info.update(contact_saved=contact_present, velocity_saved=velocity_present)
    return info


def _final_manipulation_base_rest(rows, dt):
    """Require an actual0.5s timestamp span at the end of unload_place_settle.

    The linear velocity is the native base rigid body's COM velocity, NOT the
    navigation helper's COM-to-axle corrected velocity. Compare its XY norm and
    world-Z angular velocity using the same numerical navigation thresholds.
    """
    last = rows[-1]["time"]
    window_s = THRESHOLDS["navigation_rest_window_s"]
    window = [r for r in rows if r["time"]>=last-window_s-1e-7]
    span = window[-1]["time"]-window[0]["time"]
    planar_speed = [math.hypot(*r[BASE_VELOCITY_FIELDS[0]][:2]) for r in window]
    yaw_rate = [abs(r[BASE_VELOCITY_FIELDS[1]][2]) for r in window]
    passed = (len(window)>=round(window_s/dt)+1 and span>=window_s-1e-7
              and all(v<THRESHOLDS["navigation_rest_speed_m_s"] for v in planar_speed)
              and all(w<THRESHOLDS["navigation_rest_yaw_rate_rad_s"] for w in yaw_rate))
    return {"passed": passed, "sample_count": len(window), "first_time_s": window[0]["time"],
            "last_time_s": window[-1]["time"], "observed_timestamp_span_s": span,
            "maximum_base_com_planar_speed_m_s": max(planar_speed),
            "maximum_base_world_z_yaw_rate_rad_s": max(yaw_rate),
            "linear_velocity_point": "native base rigid-body COM; no axle correction is available in these fields"}


def validate_rest_evidence(evidence, *, observed_rows=None, last_time_s=None,
                           observed_period_s=.02, dt=.002):
    """Independently integrate raw GT vectors; never invoke physical_rest.py.

The final0.5s requires all raw/filtered/eligible gates. Its50ms warm-up only
needs finite measurements. XY-box diagonal and unwrapped yaw range bound the
same quiet interval. Producer ready/metrics alone can never admit a record.
"""
    result = {"passed": False, "revision": REST_REVISION, "checks": {}, "metrics": {},
              "scope": "Independent saved2ms windowed-rest evidence; no resimulation or hardware claim."}
    checks, metrics = result["checks"], result["metrics"]
    try:
        if abs(_number(dt,"rest dt")-.002)>1e-12:
            raise ValueError("Windowed-rest raw contract requires2ms sampling")
        if not isinstance(evidence, Mapping) or evidence.get("revision") != REST_REVISION:
            raise ValueError("Missing/unknown windowed-rest evidence revision")
        _finite_tree(evidence, "rest evidence")
        config = evidence.get("config")
        if not isinstance(config, Mapping):
            raise ValueError("Windowed-rest config is absent")
        for name, contract in REST_CONFIG.items():
            value = config.get(name)
            if isinstance(contract, str):
                if value != contract:raise ValueError("Unsupported rest convention: "+name)
            else:
                value = _number(value, "rest config "+name)
                if value<=0:raise ValueError("Rest config must be positive")
                if name in ("window_s", "quiet_duration_s"):
                    if abs(value-contract)>1e-12:raise ValueError("Unexpected rest time-window contract")
                elif value>contract+1e-12:raise ValueError("Rest bound is weaker thanwindowed-rest-v1: "+name)
        rows = evidence.get("raw_records")
        if not isinstance(rows, (list, tuple)) or len(rows)<276:
            raise ValueError("Windowed rest needs at least0.55s of2ms raw records")
        times, positions, velocities, omegas, yaws, eligible = [], [], [], [], [], []
        for row in rows:
            times.append(_number(row["time"], "rest time", nonnegative=True))
            positions.append(_vector(row["axle_xy_m"], 2, "rest axle position"))
            velocities.append(_vector(row["axle_linear_velocity_m_s"], 3, "rest axle velocity"))
            omegas.append(_number(row["measured_yaw_rate_rad_s"], "rest yaw rate"))
            yaw = _number(row["base_yaw_rad"], "rest yaw")
            if yaws:
                yaw = yaws[-1]+math.atan2(math.sin(yaw-yaws[-1]), math.cos(yaw-yaws[-1]))
            yaws.append(yaw)
            if not isinstance(row.get("eligible"), bool):raise ValueError("Rest eligibility must be boolean")
            eligible.append(row["eligible"])
        if any(abs(b-a-dt)>1e-7 for a,b in zip(times,times[1:])):
            raise ValueError("Rest raw record is not complete2ms sampling")
        if any(b-a>config["max_gap_s"]+1e-12 for a,b in zip(times,times[1:])):
            raise ValueError("Rest raw interval exceeds its declared gap bound")
        end = times[-1];quiet_start = end-config["quiet_duration_s"]
        if times[0]>quiet_start-config["window_s"]+1e-7:
            raise ValueError("Rest history does not cover0.5s quiet plus50ms averaging warm-up")
        indices = [i for i,t in enumerate(times) if t>=quiet_start-1e-7]
        if len(indices)<251 or times[indices[0]]>quiet_start+1e-7:
            raise ValueError("Rest history has no complete continuous0.5s quiet window")
        means, mean_omega = [], []
        for i in indices:
            left, right = times[i]-config["window_s"], times[i]
            total = [0.,0.,0.,0.]; covered = 0.
            for j in range(i):
                a,b=times[j],times[j+1];lo,hi=max(left,a),min(right,b)
                if hi<=lo:continue
                fraction_lo,fraction_hi=(lo-a)/(b-a),(hi-a)/(b-a)
                start=(*velocities[j],omegas[j]);finish=(*velocities[j+1],omegas[j+1])
                for k in range(4):
                    va=start[k]+fraction_lo*(finish[k]-start[k])
                    vb=start[k]+fraction_hi*(finish[k]-start[k])
                    total[k]+=.5*(va+vb)*(hi-lo)
                covered+=hi-lo
            if abs(covered-config["window_s"])>1e-7:
                raise ValueError("A50ms rest integration window is incomplete")
            mean=[value/config["window_s"] for value in total]
            means.append(math.hypot(*mean[:2]));mean_omega.append(abs(mean[3]))
        raw_speeds=[math.hypot(*velocities[i][:2]) for i in indices]
        raw_omega=[abs(omegas[i]) for i in indices]
        excursion=math.hypot(*(max(positions[i][k] for i in indices)-min(positions[i][k] for i in indices) for k in (0,1)))
        yaw_excursion=max(yaws[i] for i in indices)-min(yaws[i] for i in indices)
        checks.update(raw_sampling_complete=True, continuous_quiet_duration=True,
                      producer_ready_is_true=evidence.get("ready") is True,
                      producer_last_error_is_none=evidence.get("last_error") is None,
                      eligible_every_quiet_sample=all(eligible[i] for i in indices),
                      filtered_velocity_below_limit=max(means)<config["filtered_speed_max_m_s"],
                      filtered_yaw_rate_below_limit=max(mean_omega)<config["filtered_yaw_rate_max_rad_s"],
                      raw_velocity_below_spike_limit=max(raw_speeds)<config["raw_speed_max_m_s"],
                      raw_yaw_rate_below_spike_limit=max(raw_omega)<config["raw_yaw_rate_max_rad_s"],
                      xy_excursion_within_limit=excursion<=config["position_excursion_max_m"],
                      yaw_excursion_within_limit=yaw_excursion<=config["yaw_excursion_max_rad"])
        metrics.update(raw_sample_count=len(rows),quiet_sample_count=len(indices),
                       first_time_s=times[0],last_time_s=end,raw_timestamp_span_s=end-times[0],
                       quiet_timestamp_span_s=end-times[indices[0]],
                       maximum_filtered_planar_speed_m_s=max(means),maximum_filtered_yaw_rate_rad_s=max(mean_omega),
                       maximum_raw_planar_speed_m_s=max(raw_speeds),maximum_raw_yaw_rate_rad_s=max(raw_omega),
                       xy_bounding_box_diagonal_m=excursion,yaw_unwrapped_range_rad=yaw_excursion)
        if last_time_s is not None:
            gap=end-_number(last_time_s,"observed final timestamp")
            checks["raw_evidence_matches_phase_end"]=-1e-7<=gap<=observed_period_s+dt+1e-7
            metrics["raw_to_last_saved_state_gap_s"]=gap
        if observed_rows is not None:
            if not isinstance(observed_rows,(list,tuple)) or not observed_rows:
                raise ValueError("Nonempty saved phase state records are required")
            checks["raw_evidence_stays_within_its_phase"] = times[0]>=_number(observed_rows[0]["time"],"phase first timestamp")-dt-1e-7
            matched=0;raw_index=0
            for observed in observed_rows:
                now=_number(observed["time"],"observed rest timestamp")
                if now<times[0]-1e-7 or now>end+1e-7:continue
                while raw_index+1<len(times) and times[raw_index]<now-1e-7:raw_index+=1
                if abs(times[raw_index]-now)>1e-7:raise ValueError("Saved state timestamp absent from rest raw history")
                state=observed.get("base_state",observed.get("state"))
                if not isinstance(state,Mapping):raise ValueError("Saved state lacks native axle rest fields")
                xy=_vector(state["axle_xy_m"],2,"observed rest axle")
                velocity=_vector(state["axle_linear_velocity_m_s"],3,"observed rest velocity")
                omega=_number(state["measured_yaw_rate_rad_s"],"observed rest omega")
                yaw=_number(state["base_yaw_rad"],"observed rest yaw")
                if math.dist(xy,positions[raw_index])>1e-6 or math.dist(velocity,velocities[raw_index])>1e-6:
                    raise ValueError("Rest raw position/velocity differs from independently saved state")
                if abs(omega-omegas[raw_index])>1e-7 or abs(math.atan2(math.sin(yaw-yaws[raw_index]),math.cos(yaw-yaws[raw_index])))>1e-7:
                    raise ValueError("Rest raw yaw/rate differs from independently saved state")
                matched+=1
            checks["raw_evidence_cross_matches_saved_states"]=matched>=math.floor(.5/observed_period_s)
            metrics["matched_saved_state_count"]=matched
        result["passed"]=all(checks.values())
    except (KeyError,ValueError,TypeError,OverflowError,IndexError) as error:
        result["error"]=f"{type(error).__name__}: {error}"
        checks["input_valid"]=False
    result["failed_checks"]=[key for key,value in checks.items() if not value]
    return result


def _rest_revision(declared, *, evidence=_REST_ABSENT, trace=None):
    new_fields=(evidence is not _REST_ABSENT or
                isinstance(trace,Mapping) and "rest_windows" in trace)
    if declared is _REST_ABSENT and not new_fields:return None
    revision=evidence.get("revision") if declared is _REST_ABSENT and isinstance(evidence,Mapping) else declared
    if revision!=REST_REVISION:raise ValueError("Declared or inferred rest revision is unknown/missing")
    return revision


def _navigation_rest_windows(trace, groups):
    if not isinstance(trace,Mapping) or not isinstance(trace.get("rest_windows"),Mapping):
        raise ValueError("Declared windowed rest requires per-phase navigation rest_windows")
    windows=trace["rest_windows"]
    if set(windows)!=set(groups):raise ValueError("Navigation rest_windows have missing/unknown phases")
    results={}
    for phase,rows in groups.items():
        results[phase]=validate_rest_evidence(windows[phase],observed_rows=rows,last_time_s=rows[-1]["time"])
    return results


def _sampling(rows, period, *, allow_navigation_gap=False):
    sequence, groups, intervals = [], {}, []
    for i, row in enumerate(rows):
        phase = row["phase"]
        groups.setdefault(phase, []).append(row)
        if not sequence or sequence[-1] != phase:
            sequence.append(phase)
        if i:
            previous = rows[i-1]
            interval = row["time"]-previous["time"]
            if interval <= 0:
                raise ValueError("Physical timestamps must be strictly increasing")
            boundary = previous["phase"] == "place_settle" and phase == "unload_hover"
            if allow_navigation_gap and boundary:
                continue
            if previous["phase"] != phase and period == .02:
                if interval > period + 1e-7 or interval < .002 - 1e-7:
                    raise ValueError("Navigation phase boundary contains a sampling gap")
            elif abs(interval-period) > max(1e-7, period*1e-5):
                raise ValueError(f"Missing/duplicated interval: {interval} s, expected {period} s")
            intervals.append(interval)
    return sequence, groups, {"minimum_interval_s": min(intervals) if intervals else None,
                              "maximum_interval_s": max(intervals) if intervals else None}


def _continuity(rows, *, pose_saved, maximum_residual):
    maximum, rotation_excess, frame_residual = 0., 0., 0.
    for a, b in zip(rows, rows[1:]):
        interval = b["time"]-a["time"]
        if interval <= 0:
            raise ValueError("Continuity timestamps are not increasing")
        residual = math.hypot(*[(b["object"][j]-a["object"][j])
                               -.5*(a["object_velocity"][j]+b["object_velocity"][j])*interval
                               for j in range(3)])
        maximum = max(maximum, residual)
        if pose_saved:
            qa = _quaternion(a[POSE_FIELDS[0]], "object quaternion")
            qb = _quaternion(b[POSE_FIELDS[0]], "object quaternion")
            angle = 2*math.acos(min(1., abs(sum(x*y for x,y in zip(qa,qb)))))
            rate = max(math.hypot(*a["object_angular_velocity"]), math.hypot(*b["object_angular_velocity"]))
            rotation_excess = max(rotation_excess, angle-rate*interval)
    if pose_saved:
        for row in rows:
            rotation = _rotation(_quaternion(row[POSE_FIELDS[2]], "tray quaternion"))
            delta = [row["object"][j]-row[POSE_FIELDS[1]][j] for j in range(3)]
            independent = _in_tray(delta, rotation)
            frame_residual = max(frame_residual, math.hypot(*[independent[j]-row["object_in_tray"][j] for j in range(3)]))
    return {"maximum_position_velocity_residual_m": maximum,
            "maximum_orientation_step_excess_rad": rotation_excess,
            "maximum_world_to_tray_residual_m": frame_residual,
            "passed": maximum <= maximum_residual
            and (not pose_saved or (rotation_excess <= THRESHOLDS["orientation_step_allowance_rad"]
                                   and frame_residual <= THRESHOLDS["world_to_tray_residual_m"]))}


def _tray_gate(rows, expected_z, z_tolerance):
    return all(abs(r["object_in_tray"][0]) < THRESHOLDS["tray_x_m"]
               and abs(r["object_in_tray"][1]) < THRESHOLDS["tray_y_m"]
               and abs(r["object_in_tray"][2]-expected_z) < z_tolerance
               and r["tray_support_force_n"] > THRESHOLDS["contact_n"] for r in rows)


def validate_approach(approach_trace, dock_pose_m_rad, *, nav_dt=.02,
                      rest_definition_revision=_REST_ABSENT):
    """Evaluate saved empty-tray navigate/dock records against the selected rack.

    The target comes from the caller's selected scene, never a fixed station.
    This checks measured motion/rest and recorded contact norms, not the
    producer's passed flag. A 20 ms trace cannot prove absence of contacts
    between samples or establish that the producer never overwrote poses.
    """
    result = {"passed": False, "checks": {"input_valid": False}, "metrics": {},
              "scope": {"claim": "Saved navigate/dock approach and selected-rack arrival gates.",
                        "not_verified": ["No pose writes/attachment: requires producer audit.",
                                         "Unrecorded contacts between 20 ms samples, self-collision or safety.",
                                         "Independent localization, QP optimality or hardware performance."]}}
    checks, metrics = result["checks"], result["metrics"]
    try:
        if abs(_number(nav_dt, "nav_dt")-.02)>1e-12:
            raise ValueError("Approach trace requires 20 ms samples")
        target = _vector(dock_pose_m_rad, 3, "selected rack dock pose")
        revision = _rest_revision(rest_definition_revision, trace=approach_trace)
        rows = approach_trace.get("records") if isinstance(approach_trace, Mapping) else approach_trace
        if isinstance(approach_trace, Mapping) and approach_trace.get("failure"):
            raise ValueError("Approach trace reports a physical/control failure")
        if not isinstance(rows, (list, tuple)) or not rows:
            raise ValueError("Nonempty approach measurements are required")
        _finite_tree(rows, "approach")
        for i, row in enumerate(rows):
            if not isinstance(row, Mapping) or not isinstance(row.get("phase"), str):
                raise ValueError(f"Approach row {i} has no phase")
            _number(row["time"], "approach time", nonnegative=True)
            state = row["state"]
            _vector(state["base_position_m"], 3, "approach base position")
            _number(state["base_yaw_rad"], "approach yaw")
            _number(state["measured_planar_speed_m_s"], "approach measured speed", nonnegative=True)
            _number(state["measured_yaw_rate_rad_s"], "approach measured yaw rate")
            _number(row["base_up_z"], "approach base/tray upright")
            _number(row["robot_environment_contact_max_n"], "approach recorded contact", nonnegative=True)
            if row["qp"].get("accepted") is not True:
                raise ValueError("Approach contains a rejected QP command")
            _vector(row["qp"]["solution"], 2, "approach QP solution")
        sequence, groups, timing = _sampling(rows, nav_dt)
        checks["complete_navigate_dock_sequence"] = sequence == ["navigate", "dock"]
        if not checks["complete_navigate_dock_sequence"]:
            raise ValueError("Approach navigate/dock phases missing, reordered or repeated")
        durations = {phase: group[-1]["time"]-group[0]["time"]+nav_dt for phase, group in groups.items()}
        if min(durations.values()) < THRESHOLDS["navigation_rest_window_s"]-1e-7:
            raise ValueError("Approach phase too short for the 0.5 s rest gate")
        checks["complete_sample_intervals"] = True
        checks["recorded_qps_accepted"] = True
        contact = max(row["robot_environment_contact_max_n"] for row in rows)
        checks["no_recorded_environment_contact"] = contact <= THRESHOLDS["navigation_contact_max_n"]
        checks["base_tray_upright"] = all(row["base_up_z"] >= THRESHOLDS["upright_z"] for row in rows)
        points = [row["state"]["base_position_m"] for row in rows]
        distance = sum(math.hypot(b[0]-a[0], b[1]-a[1]) for a,b in zip(points, points[1:]))
        checks["recorded_nonzero_motion"] = distance > 1e-6
        final = rows[-1]["state"]
        xy_error = math.hypot(final["base_position_m"][0]-target[0], final["base_position_m"][1]-target[1])
        yaw_error = abs(math.atan2(math.sin(final["base_yaw_rad"]-target[2]), math.cos(final["base_yaw_rad"]-target[2])))
        checks["measured_arrival_at_selected_rack"] = xy_error < THRESHOLDS["approach_xy_m"] and yaw_error < THRESHOLDS["approach_yaw_rad"]
        if revision:
            windows = _navigation_rest_windows(approach_trace, groups)
            metrics["windowed_rest_phases"] = windows
            checks["all_navigation_phase_rest_windows"] = all(value["passed"] for value in windows.values())
            checks["measured_final_rest_window"] = windows["dock"]["passed"]
        else:
            rest = [row["state"] for row in groups["dock"]
                    if row["time"] >= rows[-1]["time"]-(THRESHOLDS["navigation_rest_window_s"]-nav_dt)-1e-7]
            checks["measured_final_rest_window"] = len(rest) >= round(THRESHOLDS["navigation_rest_window_s"]/nav_dt) and all(
                state["measured_planar_speed_m_s"] < THRESHOLDS["navigation_rest_speed_m_s"]
                and abs(state["measured_yaw_rate_rad_s"]) < THRESHOLDS["navigation_rest_yaw_rate_rad_s"] for state in rest)
        metrics.update(sample_count=len(rows), phase_sequence=sequence, phase_durations_s=durations,
                       timing=timing, selected_rack_dock_pose_m_rad=list(target),
                       final_xy_error_m=xy_error, final_yaw_error_rad=yaw_error,
                       recorded_distance_m=distance, maximum_recorded_environment_contact_n=contact,
                       first_time_s=rows[0]["time"], last_time_s=rows[-1]["time"])
        metrics["rest_definition_revision"] = revision or "legacy-instantaneous-rest"
        checks["input_valid"] = True
        result["passed"] = all(checks.values())
    except (KeyError, ValueError, TypeError, OverflowError, IndexError, AttributeError) as error:
        result["error"] = f"{type(error).__name__}: {error}"
        checks["input_valid"] = False
    return result


def validate(samples, scene, transport, *, mode="load", navigation_trace=None,
             approach_trace=None, drive_samples=None, phase_durations_s=None,
             manipulation_contact_monitor=_MONITOR_ABSENT, final_rest_evidence=_REST_ABSENT,
             rest_definition_revision=_REST_ABSENT, dt=.002, nav_dt=.02):
    """Evaluate measurements, without using the producer's acceptance flags."""
    result = {"passed": False, "checks": {"input_valid": False}, "metrics": {},
              "mode": mode, "thresholds": THRESHOLDS.copy(),
              "scope": {"claim": "Independent saved physical-trace gates; not a re-simulation.",
                        "not_verified": ["No pose overwrites/attachment: requires producer audit.",
                                         "Whole-robot forbidden contacts, self-collision, braking or dynamic avoidance.",
                                         "LLM correctness, camera estimation accuracy, QP optimality or hardware/real-time performance."]}}
    checks, metrics = result["checks"], result["metrics"]
    try:
        if mode not in ("load", "drive", "full"):
            raise ValueError("Unsupported mode")
        revision = _rest_revision(rest_definition_revision, evidence=final_rest_evidence,
                                  trace=navigation_trace)
        if revision and mode!="full":
            raise ValueError("The new rest revision is declared only for full transport")
        if revision and final_rest_evidence is _REST_ABSENT:
            raise ValueError("Declared full windowed rest requires final_rest_evidence")
        result["rest_definition_revision"] = revision or "legacy-instantaneous-rest"
        if abs(_number(dt,"dt")-.002)>1e-12 or abs(_number(nav_dt,"nav_dt")-.02)>1e-12:
            raise ValueError("Trace contracts require 2 ms manipulation and 20 ms navigation")
        size = _vector(scene["object_full_size_m"],3,"object size")
        if min(size)<=0:
            raise ValueError("Positive object dimensions required")
        table_top = _number(scene["table_top_world_z_m"],"table top")
        tray_center = _vector(transport["tray_center_in_base_m"],3,"tray center")
        tray_floor_offset = _number(transport["tray_top_in_base_m"],"tray top")-tray_center[2]
        expected_tray_z = tray_floor_offset+size[2]/2
        poses = _rows_valid(samples,manipulation=True,poses_required=mode=="full")
        monitor_info = _manipulation_monitor_rows(samples, manipulation_contact_monitor)
        metrics["manipulation_monitor_fields"] = monitor_info
        if monitor_info["contact_saved"] or monitor_info["velocity_saved"]:
            checks["complete_manipulation_monitor_fields"] = True
        if monitor_info["contact_saved"]:
            maximum_contact = max(r[MANIPULATION_CONTACT_FIELD] for r in samples)
            metrics["maximum_recorded_manipulation_environment_contact_n"] = maximum_contact
            checks["manipulation_no_recorded_environment_contact"] = maximum_contact<=monitor_info["force_limit_n"]
            result["scope"]["not_verified"].append("Manipulation contact gate covers saved2ms samples and the producer's declared filter only; omitted rejected-step rows/unmodeled contacts are not certified absent.")
        else:
            result["scope"]["not_verified"].append("Legacy/narrow manipulation trace has no environment contact norm; no manipulation environment-contact certificate.")
        sequence, groups, timing = _sampling(samples,dt,allow_navigation_gap=mode=="full")
        expected = list(LOAD_DURATIONS)+(list(UNLOAD_DURATIONS) if mode=="full" else [])
        checks["complete_phase_sequence"] = sequence==expected
        if not checks["complete_phase_sequence"]:
            raise ValueError("Required phases missing, reordered, repeated or truncated")
        if phase_durations_s is None:
            scale = len(groups["settle"])*dt/LOAD_DURATIONS["settle"]
            if scale<1.-1e-8:
                raise ValueError("Initial settle duration is below the execution contract")
            durations = {name:seconds*scale for name,seconds in LOAD_DURATIONS.items()}
            if mode=="full":durations.update(UNLOAD_DURATIONS)
            metrics["duration_contract_source"] = "Known phase ratios; load scale inferred from initial settle. Exact producer schedule was not supplied."
        else:
            durations = dict(phase_durations_s)
            if set(durations)!=set(expected):
                raise ValueError("Declared phase duration contract has missing/unknown phases")
            metrics["duration_contract_source"] = "Caller-supplied execution schedule"
        for phase, seconds in durations.items():
            seconds = _number(seconds,phase+" duration")
            minimum = (LOAD_DURATIONS if phase in LOAD_DURATIONS else UNLOAD_DURATIONS)[phase]
            if seconds<minimum-1e-8 or len(groups[phase])!=round(seconds/dt):
                raise ValueError(f"Incomplete sample count for {phase}")
        checks["complete_sample_intervals_and_counts"] = True
        metrics.update(samples=len(samples),phase_sequence=sequence,phase_sample_counts={p:len(g) for p,g in groups.items()},
                       phase_durations_s={p:g[-1]["time"]-g[0]["time"]+dt for p,g in groups.items()},timing=timing,
                       object_tray_world_poses_saved=poses)
        if not poses:
            result["scope"]["not_verified"].append("Legacy load trace has no object/tray world poses: orientation and world-to-tray transform were not checked.")
        # A full manipulation trace has a genuine navigation interval, not a missing-data exemption.
        load_rows = [r for r in samples if not r["phase"].startswith("unload_")]
        unload_rows = [r for r in samples if r["phase"].startswith("unload_")]
        for label, rows in (("load",load_rows),("unload",unload_rows)):
            if not rows:continue
            continuity = _continuity(rows,pose_saved=poses,maximum_residual=THRESHOLDS["manipulation_position_velocity_residual_m"])
            metrics[label+"_continuity"] = continuity
            checks[label+"_physical_pose_continuity"] = continuity["passed"]
        checks["stationary_base"] = all(r["base_xy_drift_m"]<=.03 and r["base_up_z"]>=.98 for r in samples)
        checks["arm_reference_tracking"] = all(max(abs(a-b) for a,b in zip(r["arm_q"],r["arm_reference"]))<=.08+1e-9 for r in samples)
        reach_phases = [p for p in expected if p=="settle" or p.endswith(("hover_settle","approach_settle","transfer_settle","lower_settle"))]
        checks["tcp_reached_before_next_action"] = all(groups[p][-1]["tcp_error_m"]<=.004 for p in reach_phases)
        carry_names = ("lift","hold","transfer","transfer_settle","lower","lower_settle")
        checks["load_lift_carry_bilateral"] = all(min(r["finger_normal_force_n"])>=.02 for p in carry_names for r in groups[p])
        checks["load_close_bilateral"] = min(groups["close"][-1]["finger_normal_force_n"])>=.02
        hold = groups["hold"]
        if poses:
            independent_clearance = []
            for row in hold:
                rotation = _rotation(_quaternion(row[POSE_FIELDS[0]], "object quaternion"))
                bottom = row["object"][2]-sum(abs(rotation[2][j])*size[j]/2 for j in range(3))
                independent_clearance.append(bottom-table_top)
            metrics["minimum_load_hold_clearance_m"] = min(independent_clearance)
            checks["load_clearance_world_frame_consistent"] = all(abs(value-row["clearance_m"])<=.0002 for value,row in zip(independent_clearance,hold))
        else:
            metrics["minimum_load_hold_clearance_m"] = min(r["clearance_m"] for r in hold)
        checks["load_lifted_and_held"] = metrics["minimum_load_hold_clearance_m"]>.05 and all(min(r["finger_normal_force_n"])>.02 for r in hold)
        checks["load_hold_drift"] = all(r["relative_translation_drift_m"]<.01 and r["relative_rotation_drift_rad"]<math.radians(10) for r in hold)
        settled = groups["place_settle"]
        checks["released_on_supported_tray"] = _tray_gate(settled,expected_tray_z,.010) and all(max(r["finger_normal_force_n"])<.02 for r in settled)
        checks["tray_load_at_rest"] = all(math.hypot(*r["object_velocity"])<.01 and math.hypot(*r["object_angular_velocity"])<.2 for r in settled)
        metrics["maximum_tray_rest_speed_m_s"] = max(math.hypot(*r["object_velocity"]) for r in settled)
        if mode=="full":
            if scene.get("layout") == "rack-v1" or approach_trace is not None:
                approach = validate_approach(approach_trace, scene["base_dock_pose_m_rad"], nav_dt=nav_dt,
                    **({"rest_definition_revision":revision} if revision else {}))
                metrics["approach_navigation"] = approach
                checks["independent_selected_rack_approach"] = approach["passed"]
                if approach["passed"]:
                    gap = load_rows[0]["time"]-approach["metrics"]["last_time_s"]
                    metrics["approach_to_load_gap_s"] = gap
                    checks["approach_bridges_physical_time"] = -1e-7 <= gap <= nav_dt+dt+1e-7
                else:
                    checks["approach_bridges_physical_time"] = False
            nav = navigation_trace.get("records") if isinstance(navigation_trace,Mapping) else navigation_trace
            nav_poses = _rows_valid(nav,manipulation=False,poses_required=True)
            nav_sequence, nav_groups, nav_timing = _sampling(nav,nav_dt)
            if nav_sequence!=["undock","navigate","dock"]:
                raise ValueError("Navigation undock/navigate/dock phases missing or repeated")
            if any(group[-1]["time"]-group[0]["time"]+nav_dt<.5-1e-7 for group in nav_groups.values()):
                raise ValueError("Navigation phase too short to record the 0.5 s rest gate")
            if isinstance(navigation_trace,Mapping) and navigation_trace.get("failure"):
                raise ValueError("Navigation trace reports a physical/control failure")
            for row in nav:
                state=row["state"]
                _vector(state["base_position_m"],3,"navigation base position")
                _vector(state["axle_xy_m"],2,"navigation axle")
                _number(state["base_yaw_rad"],"navigation yaw")
                _number(state["measured_planar_speed_m_s"],"navigation speed",nonnegative=True)
                _number(state["measured_yaw_rate_rad_s"],"navigation yaw rate")
                _number(row["robot_environment_contact_max_n"],"navigation recorded environment contact",nonnegative=True)
                if row["qp"].get("accepted") is not True:
                    raise ValueError("Navigation contains a rejected QP command")
                _vector(row["qp"]["solution"],2,"navigation QP solution")
            checks["navigation_recorded_load_support"] = _tray_gate(nav,expected_tray_z,.012) and all(r["base_up_z"]>=.98 for r in nav)
            checks["navigation_no_recorded_environment_contact"] = all(r["robot_environment_contact_max_n"]<=THRESHOLDS["navigation_contact_max_n"] for r in nav)
            nav_continuity=_continuity(nav,pose_saved=nav_poses,maximum_residual=THRESHOLDS["navigation_position_velocity_residual_m"])
            metrics.update(navigation_sample_count=len(nav),navigation_timing=nav_timing,navigation_continuity=nav_continuity)
            checks["navigation_physical_pose_continuity"] = nav_continuity["passed"]
            # Mixed-rate streams must actually cover the manipulation gap.
            bridge_pairs=((load_rows[-1],nav[0]),(nav[-1],unload_rows[0]))
            checks["navigation_bridges_physical_time"] = all(-1e-7<=b["time"]-a["time"]<=nav_dt+dt+1e-7 for a,b in bridge_pairs)
            checks["navigation_bridges_world_object_pose"] = all(math.hypot(*[b["object"][j]-a["object"][j] for j in range(3)])<=.004 for a,b in bridge_pairs)
            last=nav[-1]["state"];dock=_vector(transport["destination_base"],3,"B dock")
            metrics["destination_base_pose_m_rad"] = list(dock)
            checks["measured_arrival_at_B"] = math.hypot(last["base_position_m"][0]-dock[0],last["base_position_m"][1]-dock[1])<.015 and abs(math.atan2(math.sin(last["base_yaw_rad"]-dock[2]),math.cos(last["base_yaw_rad"]-dock[2])))<.015 and last["measured_planar_speed_m_s"]<(.02 if revision else .006) and abs(last["measured_yaw_rate_rad_s"])<(.04 if revision else .015)
            if revision:
                windows=_navigation_rest_windows(navigation_trace,nav_groups)
                metrics["windowed_navigation_rest_phases"]=windows
                checks["navigation_all_phase_windowed_rest"]=all(value["passed"] for value in windows.values())
                checks["measured_B_rest_window"]=windows["dock"]["passed"]
            else:
                rest_window=[row["state"] for row in nav_groups["dock"] if row["time"]>=nav[-1]["time"]-(.5-nav_dt)-1e-7]
                checks["measured_B_rest_window"] = len(rest_window)>=25 and all(row["measured_planar_speed_m_s"]<.006 and abs(row["measured_yaw_rate_rad_s"])<.015 for row in rest_window)
            checks["unload_close_bilateral"] = min(groups["unload_close"][-1]["finger_normal_force_n"])>=.02
            checks["unload_lift_carry_bilateral"] = all(min(r["finger_normal_force_n"])>=.02 for p in carry_names for r in groups["unload_"+p])
            unload_hold=groups["unload_hold"]
            clearances=[]
            for row in unload_hold:
                rt=_rotation(_quaternion(row[POSE_FIELDS[2]],"tray quaternion"))
                ro=_rotation(_quaternion(row[POSE_FIELDS[0]],"object quaternion"))
                relative_z_axis=[sum(rt[k][2]*ro[k][j] for k in range(3)) for j in range(3)]
                bottom=row["object_in_tray"][2]-sum(abs(relative_z_axis[j])*size[j]/2 for j in range(3))
                clearances.append(bottom-tray_floor_offset)
            metrics["minimum_unload_hold_above_tray_m"]=min(clearances)
            checks["unload_lifted_and_held_above_tray"]=min(clearances)>.05 and all(min(r["finger_normal_force_n"])>.02 for r in unload_hold)
            checks["unload_hold_drift"] = all(r["relative_translation_drift_m"]<.01 and r["relative_rotation_drift_rad"]<math.radians(10) for r in unload_hold)
            target=_vector(transport["destination_object_center"],3,"B target")
            metrics["destination_object_center_m"] = list(target)
            final=groups["unload_place_settle"]
            for row in final:_number(row["B_support_force_n"],"B support",nonnegative=True)
            checks["final_on_B_target"] = all(math.hypot(r["object"][0]-target[0],r["object"][1]-target[1])<.015 and abs(r["object"][2]-target[2])<.005 for r in final)
            checks["final_B_supported_and_released"] = all(r["B_support_force_n"]>.02 and max(r["finger_normal_force_n"])<.02 for r in final)
            checks["final_B_upright"] = all(_rotation(_quaternion(r[POSE_FIELDS[0]],"object quaternion"))[2][2]>=.98 for r in final)
            checks["final_B_at_rest"] = all(math.hypot(*r["object_velocity"])<.01 and math.hypot(*r["object_angular_velocity"])<.2 for r in final)
            metrics["maximum_final_B_xy_error_m"] = max(math.hypot(r["object"][0]-target[0],r["object"][1]-target[1]) for r in final)
            if revision:
                rest=validate_rest_evidence(final_rest_evidence,observed_rows=final,
                    last_time_s=final[-1]["time"],observed_period_s=dt,dt=dt)
                metrics["final_unload_base_rest"]=rest
                checks["measured_final_unload_base_rest_window"]=rest["passed"]
            elif monitor_info["velocity_saved"]:
                rest = _final_manipulation_base_rest(final, dt)
                metrics["final_unload_base_rest"] = rest
                checks["measured_final_unload_base_rest_window"] = rest["passed"]
            else:
                result["scope"]["not_verified"].append("Legacy full trace lacks manipulation base velocity fields: final unload base rest was not checked.")
        elif mode=="drive":
            if not isinstance(drive_samples,(list,tuple)) or not drive_samples:
                raise ValueError("Drive mode requires complete drive_samples")
            _finite_tree(drive_samples,"drive_samples")
            drive_sequence,drive_groups,drive_timing=_sampling(drive_samples,dt)
            if drive_sequence!=list(DRIVE_DURATIONS) or any(len(drive_groups[p])!=round(s/dt) for p,s in DRIVE_DURATIONS.items()):
                raise ValueError("Drive phases/counts incomplete")
            for row in drive_samples:
                _vector(row["object_in_tray"],3,"drive tray relative position")
                _vector(row["base_position_m"],3,"drive base position")
                _number(row["tray_support_force_n"],"drive tray support",nonnegative=True)
                _number(row["measured_planar_speed_m_s"],"drive measured speed",nonnegative=True)
                _number(row["measured_yaw_rate_rad_s"],"drive yaw rate")
            checks["drive_supported_load"]=_tray_gate(drive_samples,expected_tray_z,.012)
            checks["drive_returns_to_rest"]=all(r["measured_planar_speed_m_s"]<.01 and abs(r["measured_yaw_rate_rad_s"])<.015 for r in drive_groups["rest"][-250:])
            metrics["drive_timing"]=drive_timing
            result["scope"]["not_verified"].append("Legacy drive_samples do not contain complete world object/tray poses; only recorded relative support/rest gates were checked.")
        checks["input_valid"]=True
        result["passed"]=all(checks.values())
    except (KeyError,ValueError,TypeError,OverflowError,IndexError) as error:
        result["error"]=f"{type(error).__name__}: {error}"
        checks["input_valid"]=False
    return result


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__,allow_abbrev=False)
    parser.add_argument("--input",type=Path,required=True,help="Saved run directory")
    parser.add_argument("--output",type=Path,required=True,help="New independent-validation JSON")
    parser.add_argument("--mode",choices=("load","drive","full"))
    args=parser.parse_args(argv)
    try:
        if args.output.exists():raise FileExistsError("Refusing to overwrite validation evidence")
        source=args.input.resolve()
        metadata=json.loads((source/"summary.json").read_text())
        trace=next((source/name for name in ("state_samples.json","rawstate_samples.json","raw_state_samples.json") if (source/name).is_file()),None)
        if trace is None:raise FileNotFoundError("Saved physical state trace is absent")
        samples=json.loads(trace.read_text())
        mode=args.mode or metadata.get("mode","load")
        nav_path=source/"navigation_trace.json";drive_path=source/"drive_samples.json"
        approach_path=source/"approach/navigation_trace.json"
        result=validate(samples,metadata["scene"],metadata["transport"],mode=mode,
                        navigation_trace=json.loads(nav_path.read_text()) if nav_path.is_file() else None,
                        approach_trace=json.loads(approach_path.read_text()) if approach_path.is_file() else None,
                        drive_samples=json.loads(drive_path.read_text()) if drive_path.is_file() else None,
                        phase_durations_s=metadata.get("phase_durations_s"),
                        **({"manipulation_contact_monitor":metadata["manipulation_contact_monitor"]}
                           if "manipulation_contact_monitor" in metadata else {}),
                        **({"rest_definition_revision":metadata["rest_definition_revision"]}
                           if "rest_definition_revision" in metadata else {}),
                        **({"final_rest_evidence":metadata["final_rest_evidence"]}
                           if "final_rest_evidence" in metadata else {}))
        result.update(input=str(source),producer_passed=metadata.get("passed"),producer_mode=metadata.get("mode"),
                      source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                      physical_trace_sha256=hashlib.sha256(trace.read_bytes()).hexdigest())
        args.output.parent.mkdir(parents=True,exist_ok=True)
        with args.output.open("x") as file:file.write(json.dumps(result,ensure_ascii=False,indent=2,allow_nan=False)+"\n")
        print(json.dumps({"passed":result["passed"],"mode":mode,"output":str(args.output.resolve()),"error":result.get("error")},ensure_ascii=False))
        return 0 if result["passed"] else 1
    except (OSError,ValueError,TypeError,KeyError) as error:
        print(f"{type(error).__name__}: {error}",file=sys.stderr)
        return 2


if __name__=="__main__":
    raise SystemExit(main())
