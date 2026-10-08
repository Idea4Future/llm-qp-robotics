#!/usr/bin/env python3
"""Audit SAVED native worker measurements; no Isaac/ROS/GPU imports.

API: validate_people(summary, manipulation_records, collision_snapshot,
                     approach_trace=None, navigation_trace=None,
                     initial_records=None,
                     static_supplement=(), inventory_role='actual_recheck').
Navigation inputs are people_trace.json dictionaries. GT is evaluation ONLY.
The producer's passed flag is not a certificate. This auditor does not test the
robot's LiDAR stop/resume policy, contact safety, or complete transport success.

CLI: python scripts/isaac/validate_people.py --input RUN_DIR --output NEW.json
Prefer saved people_collision_snapshot.json from authoring. For older records,
an explicit --static-supplement file can provide missing workcell footprints;
the different inventory is labelled as a recheck, not the original source.
"""
from __future__ import annotations

import argparse
from collections.abc import Mapping
import hashlib
import json
import math
from pathlib import Path
import sys

LOAD_PHASES = ('settle', 'approach', 'approach_settle', 'close', 'lift', 'hold',
               'transfer', 'transfer_settle', 'lower', 'lower_settle', 'open',
               'retreat', 'place_settle')
UNLOAD_PHASES = ('unload_hover', 'unload_hover_settle', 'unload_approach',
                 'unload_approach_settle', 'unload_close', 'unload_lift',
                 'unload_hold', 'unload_transfer', 'unload_transfer_settle',
                 'unload_lower', 'unload_lower_settle', 'unload_open',
                 'unload_retreat', 'unload_place_settle')
YIELD_MODEL = 'robot-priority-yield-v1'
NATIVE_YIELD_SPEED_TOLERANCE_M_S = .005


def _number(value, label, *, nonnegative=False):
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(label + ' must be numeric, not boolean')
    value = float(value)
    if not math.isfinite(value) or (nonnegative and value < 0):
        raise ValueError(label + ' must be finite' + (' and nonnegative' if nonnegative else ''))
    return value


def _vector(value, length, label):
    if not isinstance(value, (list, tuple)) or len(value) != length:
        raise ValueError(f'{label} must have {length} components')
    return tuple(_number(x, label) for x in value)


def _distance(a, b):
    return math.sqrt(sum((x-y)**2 for x, y in zip(a, b)))


def _point_rectangle(point, rectangle):
    x, y = point[:2]
    lo_x, lo_y, hi_x, hi_y = rectangle
    return math.hypot(max(lo_x-x, 0., x-hi_x), max(lo_y-y, 0., y-hi_y))


def _point_segment(p, a, b):
    vx, vy = b[0]-a[0], b[1]-a[1]
    denominator = vx*vx+vy*vy
    alpha = min(1., max(0., ((p[0]-a[0])*vx+(p[1]-a[1])*vy)/denominator))
    return math.hypot(p[0]-a[0]-alpha*vx, p[1]-a[1]-alpha*vy)


def _segment_rectangle(a, b, rectangle):
    # Slab intersection; otherwise closest endpoints/rectangle corners suffice.
    low, high = 0., 1.
    for axis in (0, 1):
        direction = b[axis]-a[axis]
        if abs(direction) < 1e-15:
            if not rectangle[axis] <= a[axis] <= rectangle[axis+2]:
                low, high = 1., 0.
                break
        else:
            t0 = (rectangle[axis]-a[axis])/direction
            t1 = (rectangle[axis+2]-a[axis])/direction
            low, high = max(low, min(t0, t1)), min(high, max(t0, t1))
    if low <= high:
        return 0.
    corners = ((rectangle[x], rectangle[y]) for x in (0, 2) for y in (1, 3))
    return min(_point_rectangle(a, rectangle), _point_rectangle(b, rectangle),
               *(_point_segment(p, a, b) for p in corners))


def _loop_position(route, elapsed):
    """Independent reconstruction of the saved cubic/endpoint-turn schedule."""
    a = _vector(route['start_position_m'], 3, 'route start')
    b = _vector(route['end_position_m'], 3, 'route end')
    if abs(a[2]) > 1e-9 or abs(b[2]) > 1e-9 or _distance(a[:2], b[:2]) <= 0:
        raise ValueError('Worker route must be a nonzero planar ground-level segment')
    leg = _number(route['leg_duration_s'], 'route leg')
    turn = _number(route['endpoint_turn_duration_s'], 'route turn')
    delay = _number(route['start_delay_s'], 'route delay', nonnegative=True)
    if leg <= 0 or turn <= 0 or _distance(a, b) <= 0:
        raise ValueError('Invalid loop route duration/length')
    period = 2*(leg+turn)
    if abs(_number(route['loop_period_s'], 'route period')-period) > 1e-6:
        raise ValueError('Loop period inconsistent with saved leg/turn durations')
    speed = _number(route['maximum_script_speed_m_s'], 'route peak')
    if speed <= 0 or 1.5*_distance(a, b)/leg > speed+1e-9:
        raise ValueError('Saved cubic route exceeds its command speed design')
    if elapsed < delay:
        return a
    phase = (elapsed-delay) % period
    if phase < leg:
        u, start, end = phase/leg, a, b
    elif phase < leg+turn:
        return b
    elif phase < 2*leg+turn:
        u, start, end = (phase-leg-turn)/leg, b, a
    else:
        return a
    fraction = 3*u*u-2*u*u*u
    return tuple(x+fraction*(y-x) for x, y in zip(start, end))


def _rectangle_inventory(snapshot, supplement, role, metadata):
    if not isinstance(snapshot, Mapping) or snapshot.get('coverage_admitted') is not True:
        raise ValueError('Actual collision snapshot must have admitted coverage')
    height = _vector(snapshot.get('height_range_m'), 2, 'collision height range')
    if height[0] > .06+1e-9 or height[1] < 1.7-1e-9:
        raise ValueError('Static inventory must cover worker body height, not only laser height')
    entries = list(snapshot.get('rectangles', []))+list(supplement)
    if not entries:
        raise ValueError('Empty static inventory')
    rectangles = []
    for item in entries:
        box = _vector(item.get('xy_bounds_m') if isinstance(item, Mapping) else item,
                      4, 'static rectangle')
        if box[0] >= box[2] or box[1] >= box[3]:
            raise ValueError('Invalid static rectangle bounds')
        rectangles.append(box)
    if role not in ('authoring', 'actual_recheck'):
        raise ValueError('inventory_role must be authoring or actual_recheck')
    author_count = metadata.get('static_rectangle_count')
    if isinstance(author_count, bool) or not isinstance(author_count, int) or author_count <= 0:
        raise ValueError('Invalid authored static rectangle count')
    exact = role == 'authoring' and len(entries) == author_count and not supplement
    if role == 'authoring' and not exact:
        raise ValueError('Declared original authoring inventory count differs from saved metadata')
    # Older navigation snapshots exclude destination geometry. A recheck must
    # explicitly carry the coarse destination footprint, not infer its absence.
    paths = {item.get('path') for item in entries if isinstance(item, Mapping)}
    destination = 'workcell_B' in paths or any(
        isinstance(path, str) and path.startswith('/World/TableB/') for path in paths)
    complete = exact or destination
    return rectangles, {'role': role, 'snapshot_rectangle_count': len(snapshot['rectangles']),
                        'supplement_rectangle_count': len(supplement),
                        'recheck_rectangle_count': len(entries),
                        'authored_rectangle_count': author_count,
                        'original_authoring_inventory_reproduced': exact,
                        'destination_footprint_included': destination or exact,
                        'static_coverage_complete_for_this_contract': complete,
                        'height_range_m': list(height)}


def _records(values, label, *, navigation=False):
    if navigation:
        if not isinstance(values, Mapping) or values.get('enabled') is not True:
            raise ValueError(label+' is absent/disabled/invalid')
        values = values.get('records')
    if not isinstance(values, list) or not values:
        raise ValueError(label+' contains no native worker records')
    normalized = []
    previous = None
    for i, row in enumerate(values):
        if not isinstance(row, Mapping) or not isinstance(row.get('phase'), str):
            raise ValueError(f'{label}[{i}] missing phase')
        measure = row.get('actual_external_obstacle_evaluation') if navigation else row
        if not isinstance(measure, Mapping):
            raise ValueError(f'{label}[{i}] missing native worker measure')
        timestamp = _number(measure.get('timestamp_s'), 'worker frame timestamp', nonnegative=True)
        outer = _number(row.get('time') if navigation else row.get('timestamp_s'), 'outer timestamp')
        if abs(timestamp-outer) > 1e-7:
            raise ValueError(f'{label}[{i}] outer/native timestamp mismatch')
        if previous is not None and timestamp <= previous:
            raise ValueError(label+' timestamps are duplicated/reversed')
        previous = timestamp
        normalized.append({'time': timestamp, 'phase': row['phase'],
                           'workers': measure.get('workers'), 'stream': label,
                           'motion_model': measure.get('motion_model'),
                           'robot_state_before_target': measure.get('robot_state_before_target'),
                           'robot_state_at_measurement': measure.get('robot_state_at_measurement')})
    return normalized


def _yield_target(row, route, timestamp, clock, dt, previous):
    """Audit saved coordinates/integration, never replay the producer's FSM."""
    if row.get('motion_model') != YIELD_MODEL:
        raise ValueError('Reactive worker row has missing/changed motion model')
    target = row.get('reactive_target')
    if not isinstance(target, Mapping) or target.get('revision') != 'robot-priority-pedestrian-yield-v1':
        raise ValueError('Reactive worker lacks the declared target revision')
    stamp = _number(target.get('timestamp_s'), 'reactive target time')
    if abs(timestamp-stamp-dt) > 1e-7 or stamp < clock-1e-7:
        raise ValueError('Reactive target timestamp is not native measurement time minus one physics step')
    origin = _number(target.get('origin_time_s'), 'reactive target origin')
    if abs(origin-clock) > 1e-7 or abs(_number(target.get('elapsed_s'), 'reactive elapsed')-(stamp-origin)) > 1e-7:
        raise ValueError('Reactive target origin/elapsed differs from the independent people clock')
    circle = _number(target.get('robot_circle_m'), 'reactive external scenario circle')
    if circle < .95-1e-7:
        raise ValueError('Reactive target weakened its external scenario circle')
    a, b = (_vector(route[key], 3, 'reactive lane endpoint') for key in ('start_position_m', 'end_position_m'))
    length = _distance(a[:2], b[:2])
    if length <= 0 or abs(a[2]) > 1e-9 or abs(b[2]) > 1e-9:
        raise ValueError('Reactive lane must be a nonzero ground-level segment')
    direction = tuple((b[k]-a[k])/length for k in (0, 1))
    s = _number(target.get('s_m'), 'reactive s')
    old_s = _number(target.get('previous_s_m'), 'reactive previous s')
    if not -1e-9 <= s <= length+1e-9 or not -1e-9 <= old_s <= length+1e-9:
        raise ValueError('Reactive target left its admitted lane endpoints')
    xy = _vector(target.get('position_target_xy_m'), 2, 'reactive target XY')
    expected = tuple(a[k]+direction[k]*s for k in (0, 1))
    if _distance(xy, expected) > 1e-7:
        raise ValueError('Reactive target XY differs from its normalized route coordinate')
    vmax = _number(route['maximum_script_speed_m_s'], 'reactive speed design')
    reported_max = _number(target.get('maximum_speed_m_s'), 'reactive target speed cap')
    acceleration = _number(target.get('accel_max_m_s2'), 'reactive acceleration cap')
    if not 0 < vmax <= .30+1e-9 or abs(reported_max-vmax) > 1e-8 or not 0 < acceleration <= .4+1e-9:
        raise ValueError('Reactive target weakened the admitted speed/acceleration design')
    speed, old_speed, actual_acceleration, nominal = (_number(target.get(key), key) for key in
        ('speed_m_s', 'previous_speed_m_s', 'acceleration_m_s2', 'nominal_speed_m_s'))
    if max(abs(speed), abs(old_speed), abs(nominal)) > vmax+1e-8 or abs(actual_acceleration) > acceleration+1e-7:
        raise ValueError('Reactive target exceeds its signed speed/acceleration design')
    step = _number(target.get('dt_s'), 'reactive integration step')
    if abs(stamp-origin) <= 1e-7:
        if abs(step) > 1e-9 or abs(s-old_s) > 1e-9 or abs(speed-old_speed) > 1e-9:
            raise ValueError('First reactive target must preserve its initial state at zero integration time')
    elif abs(step-dt) > 1e-7:
        raise ValueError('Reactive target last integration step differs from2ms physics')
    ramp = _number(target.get('acceleration_ramp_duration_s'), 'reactive acceleration ramp time')
    peak = _number(target.get('peak_acceleration_m_s2'), 'reactive peak acceleration')
    if not -1e-9 <= ramp <= step+1e-9 or not 0 <= peak <= acceleration+1e-7:
        raise ValueError('Reactive ramp duration/peak acceleration exceeds its integration interval/design')
    change = speed-old_speed
    if change != 0.:
        if peak <= 0 or abs(abs(change)-peak*ramp) > 1e-8:
            raise ValueError('Reactive ramp does not explain its signed velocity change')
    elif abs(ramp) > 1e-9 or abs(peak) > 1e-9:
        raise ValueError('Unchanged reactive velocity must have zero acceleration ramp')
    integrated_s = old_s+.5*(old_speed+speed)*ramp+speed*(step-ramp)
    if abs(s-integrated_s) > 1e-8 or abs(change-actual_acceleration*step) > 1e-8:
        raise ValueError('Reactive target violates independent ramp-plus-constant position/signed acceleration integration')
    if 'ramp_duration_s' in target and abs(_number(target['ramp_duration_s'], 'reactive ramp alias')-ramp) > 1e-9:
        raise ValueError('Reactive ramp aliases disagree')
    if 'command_accel_m_s2' in target:
        command = _number(target['command_accel_m_s2'], 'reactive signed ramp acceleration')
        expected_command = math.copysign(peak, change) if change else 0.
        if abs(command-expected_command) > 1e-7:
            raise ValueError('Reactive signed ramp acceleration disagrees with its velocity change')
    if 'velocity_goal_m_s' in target:
        if abs(_number(target['velocity_goal_m_s'], 'reactive velocity goal')) > vmax+1e-8:
            raise ValueError('Reactive requested velocity goal exceeds its speed design')
    mode = target.get('mode')
    if mode not in ('walking', 'yield_brake', 'backoff', 'yield_hold') or not isinstance(target.get('yielding'), bool):
        raise ValueError('Reactive mode/yielding is missing or invalid')
    if target['yielding'] != (mode != 'walking'):
        raise ValueError('Reactive yielding flag conflicts with its mode')
    safe_s = target.get('target_safe_s_m')
    if safe_s is not None and not -1e-9 <= _number(safe_s, 'reactive safe s') <= length+1e-9:
        raise ValueError('Reactive safe target left the admitted lane')
    if mode == 'walking' and safe_s is not None or mode != 'walking' and safe_s is None:
        raise ValueError('Reactive safe target is inconsistent with its mode')
    if mode == 'yield_hold' and abs(speed) > 1e-8:
        raise ValueError('Reactive yield_hold target is still moving')
    counters = tuple(target.get(key) for key in ('yield_count', 'resume_count'))
    if any(isinstance(value, bool) or not isinstance(value, int) or value < 0 for value in counters):
        raise ValueError('Reactive event counters must be nonnegative integers')
    if previous is not None:
        interval = stamp-previous['timestamp_s']
        if interval <= 0 or abs(s-previous['s_m']) > vmax*interval+1e-7:
            raise ValueError('Reactive targets jumped/reversed time between saved observations')
        if abs(speed-previous['speed_m_s']) > acceleration*interval+1e-7:
            raise ValueError('Reactive speed changed faster than its acceleration design between observations')
        if any(value < old for value, old in zip(counters, previous['counters'])):
            raise ValueError('Reactive event counters reversed')
    return tuple((*xy, 0.)), {'timestamp_s':stamp, 's_m':s, 'speed_m_s':speed,
        'acceleration_cap_m_s2':acceleration, 'counters':counters, 'mode':mode,
        'direction':direction, 'robot_circle_m':circle}


def _robot_sample(frame, timestamp, target_time):
    current = frame.get('robot_state_at_measurement')
    before = frame.get('robot_state_before_target')
    if not isinstance(current, Mapping) or not isinstance(before, Mapping):
        raise ValueError('Reactive people frame lacks native robot snapshots')
    now = _number(current.get('timestamp_s'), 'robot measurement time')
    prior = _number(before.get('timestamp_s'), 'robot behavior-context time')
    if abs(now-timestamp) > 1e-7 or not -1e-7 <= target_time-prior <= .02+1e-7:
        raise ValueError('Reactive robot measurement/context timestamps are stale or misaligned')
    for state in (current, before):
        _vector(state.get('axle_xy_m'), 2, 'native robot axle')
        _number(state.get('base_yaw_rad'), 'native robot yaw')
        _vector(state.get('axle_linear_velocity_m_s'), 3, 'native robot velocity')
        _number(state.get('measured_yaw_rate_rad_s'), 'native robot yaw velocity')
        if _number(state.get('robot_circle_radius_m'), 'external scenario robot circle') < .95-1e-7:
            raise ValueError('External people scenario circle is smaller than its .95m design')
    return current, before


def validate_people(summary, manipulation_records, collision_snapshot, *,
                    approach_trace=None, navigation_trace=None, initial_records=None,
                    static_supplement=(),
                    inventory_role='actual_recheck', dt=.002, maximum_record_gap_s=.05):
    """Return a JSON-ready audit; failed/incomplete sources are never skipped.

Native pose at T is compared with the target submitted BEFORE its physics
step, at T-dt. .005m is an audit tolerance, not a hardware accuracy claim.
Full coverage requires both navigation legs and all manipulation phases.
Optional initial_records preserve native measurements during the initial
world settle; they do not count as manipulation or navigation phase evidence.
"""
    result = {'passed': False, 'scope': 'saved native worker motion/spacing/static-geometry audit ONLY',
              'checks': {}, 'metrics': {}, 'transport_success_verified': False,
              'limitations': ['GT is evaluated offline only; this does not verify the robot uses LiDAR correctly.',
                  'Worker bodies are kinematic capsules; gait/limbs are visual approximations.',
                  'Saved sample spacing is not continuous-time contact or human safety certification.',
                  'Script speed caps are design values; native peak velocity is reported, not certified by that cap.',
                  'Raw records cannot prove absence of unlogged pose edits or forces.']}
    checks, metrics = result['checks'], result['metrics']
    try:
        if not isinstance(summary, Mapping):
            raise ValueError('summary must be a mapping')
        metadata = summary.get('walking_workers')
        if not isinstance(metadata, Mapping):
            raise ValueError('summary lacks walking_workers metadata')
        workers = metadata.get('workers')
        if not isinstance(workers, list) or len(workers) not in (1, 3):
            raise ValueError('Expected exactly one or three declared workers')
        reactive = (metadata.get('motion_model') == YIELD_MODEL or
                    summary.get('people_clock_start', {}).get('motion_model') == YIELD_MODEL)
        result['motion_model'] = YIELD_MODEL if reactive else 'legacy-independent-clock-loop'
        if reactive:
            result['limitations'].append('Reactive targets and sampled kinematics are checked; this is not a complete2ms replay of the yielding FSM.')
        expected = {}
        worker_metadata = {}
        for worker in workers:
            path = worker.get('prim_path')
            route = worker.get('route')
            if not isinstance(path, str) or not path.startswith('/World/') or path in expected:
                raise ValueError('Duplicate/invalid declared worker identity')
            if not isinstance(route, Mapping) or route.get('worker_id') not in ('P1', 'P2', 'P3'):
                raise ValueError('Missing/invalid worker route ID')
            if any(route['worker_id'] == old['worker_id'] for old in expected.values()):
                raise ValueError('Duplicate route worker ID')
            if not reactive:
                _loop_position(route, 0.)
            expected[path] = route
            worker_metadata[path] = worker
        if set(metadata.get('prim_paths', [])) != set(expected):
            raise ValueError('Metadata prim_paths do not match declared worker identities')
        clock = _number(summary.get('people_clock_start', {}).get('timestamp_s'),
                        'people clock start', nonnegative=True)
        dt = _number(dt, 'physics interval')
        if dt <= 0 or dt > .01:
            raise ValueError('Invalid worker target-to-physics interval')
        rectangles, inventory = _rectangle_inventory(collision_snapshot, static_supplement,
                                                     inventory_role, metadata)
        result['static_inventory'] = inventory
        checks['static_inventory_coverage'] = inventory['static_coverage_complete_for_this_contract']
        manipulation_frames = _records(manipulation_records, 'manipulation')
        frames = list(manipulation_frames)
        stream_frames = {'manipulation': manipulation_frames}
        if initial_records is not None:
            initial_frames = _records(initial_records, 'initial_settle')
            if any(row['phase'] != 'initial_settle' for row in initial_frames):
                raise ValueError('Initial people records must have phase initial_settle')
            stream_frames['initial_settle'] = initial_frames
            frames += initial_frames
        for label, value in (('approach', approach_trace), ('loaded_navigation', navigation_trace)):
            if value is not None:
                stream_frames[label] = _records(value, label, navigation=True)
                frames += stream_frames[label]
        frames = sorted(frames, key=lambda row: row['time'])
        if 'initial_settle' in stream_frames:
            task_start = min(row['time'] for row in frames
                             if row['stream'] != 'initial_settle')
            checks['initial_settle_recording_precedes_tasks'] = (
                stream_frames['initial_settle'][-1]['time'] < task_start)
        gap_max = 0.
        previous = None
        pairs = {}
        stats = {path: {'worker_id': route['worker_id'], 'records': 0,
                       'script_peak_speed_design_m_s': route['maximum_script_speed_m_s'],
                       'maximum_loop_position_error_m': 0., 'native_path_travel_m': 0.,
                       'native_peak_planar_speed_m_s': 0., 'minimum_static_visual_clearance_m': math.inf}
                 for path, route in expected.items()}
        if reactive:
            for stat in stats.values():
                del stat['maximum_loop_position_error_m']
                stat.update(maximum_native_target_position_error_m=0.,
                            maximum_native_lane_error_m=0., observed_yield_hold_count=0,
                            observed_yield_hold_then_moving_resume_count=0,
                            minimum_robot_circle_physical_surface_separation_m=math.inf,
                            minimum_robot_circle_visual_surface_separation_m=math.inf)
        previous_targets = {}
        previous_native_times = {}
        hold_state = {}
        old_positions = {}
        for frame in frames:
            timestamp = frame['time']
            if timestamp < clock-1e-7:
                raise ValueError('Native worker record precedes its independently started clock')
            if previous is not None:
                gap = timestamp-previous
                if gap <= 0:
                    raise ValueError('Worker streams overlap/duplicate physics timestamps')
                gap_max = max(gap_max, gap)
            previous = timestamp
            rows = frame['workers']
            if not isinstance(rows, list) or len(rows) != len(expected):
                raise ValueError('Missing worker in '+frame['stream']+' at '+str(timestamp))
            actual = {}
            if reactive and frame['motion_model'] != YIELD_MODEL:
                raise ValueError('Reactive frame has missing/changed motion model')
            for row in rows:
                if not isinstance(row, Mapping):
                    raise ValueError('Invalid native worker row')
                path = row.get('prim_path')
                if path not in expected or path in actual:
                    raise ValueError('Unexpected/duplicate native worker identity')
                if abs(_number(row.get('timestamp_s'), 'worker timestamp')-timestamp) > 1e-7:
                    raise ValueError('Per-worker timestamp differs from frame time')
                if row.get('source') != 'native PhysX external-worker GT; evaluation only':
                    raise ValueError('Worker pose is not labelled as a native physics measurement')
                p = _vector(row.get('position_m'), 3, 'native position')
                q = _vector(row.get('quaternion_xyzw'), 4, 'native quaternion')
                if abs(math.sqrt(sum(x*x for x in q))-1.) > .001:
                    raise ValueError('Native worker quaternion is not normalized')
                velocity = _vector(row.get('linear_velocity_m_s'), 3, 'native velocity')
                _vector(row.get('angular_velocity_rad_s'), 3, 'native angular velocity')
                route, stat = expected[path], stats[path]
                if reactive:
                    target, state = _yield_target(row, route, timestamp, clock, dt, previous_targets.get(path))
                    robot, before = _robot_sample(frame, timestamp, state['timestamp_s'])
                    circle = robot['robot_circle_radius_m']
                    if abs(state['robot_circle_m']-before['robot_circle_radius_m']) > 1e-7:
                        raise ValueError('Reactive target robot circle differs from its native context')
                    if abs(q[0]) > .001 or abs(q[1]) > .001:
                        raise ValueError('Native reactive worker is tilted beyond its upright capsule model')
                    native_speed = math.hypot(*velocity[:2])
                    vmax = route['maximum_script_speed_m_s']
                    numerical = NATIVE_YIELD_SPEED_TOLERANCE_M_S
                    if native_speed > vmax+numerical or abs(velocity[2]) > numerical:
                        raise ValueError('Native reactive worker speed exceeds design plus declared numerical tolerance')
                    direction = state['direction']
                    transverse = abs(velocity[0]*direction[1]-velocity[1]*direction[0])
                    along = sum(velocity[k]*direction[k] for k in (0, 1))
                    if transverse > numerical or abs(along-state['speed_m_s']) > numerical+state['acceleration_cap_m_s2']*dt:
                        raise ValueError('Native reactive worker velocity disagrees with its signed lane target')
                    a = route['start_position_m']
                    projection = sum((p[k]-a[k])*direction[k] for k in (0, 1))
                    projected = tuple(a[k]+direction[k]*projection for k in (0, 1))
                    lane_error = _distance(p[:2], projected)
                    length = _distance(route['start_position_m'][:2], route['end_position_m'][:2])
                    if lane_error > .005 or abs(p[2]) > .005 or not -.005 <= projection <= length+.005:
                        raise ValueError('Native reactive worker left its admitted lane/ground height')
                    stat['maximum_native_lane_error_m'] = max(stat['maximum_native_lane_error_m'], lane_error)
                    if path in old_positions and _distance(p[:2], old_positions[path][:2]) > vmax*(timestamp-previous_native_times[path])+.0005:
                        raise ValueError('Native reactive worker jumped between saved measurements')
                    center_distance = _distance(p[:2], robot['axle_xy_m'])
                    physical_radius = _number(worker_metadata[path].get('radius_m', .22), 'worker physical radius')
                    if physical_radius < .22-1e-8:
                        raise ValueError('Reactive worker physical radius is smaller than its .22m design')
                    stat['minimum_robot_circle_physical_surface_separation_m'] = min(
                        stat['minimum_robot_circle_physical_surface_separation_m'], center_distance-circle-physical_radius)
                    stat['minimum_robot_circle_visual_surface_separation_m'] = min(
                        stat['minimum_robot_circle_visual_surface_separation_m'], center_distance-circle-route['visual_sweep_radius_m'])
                    hold = hold_state.setdefault(path, {'start':None, 'low':None, 'high':None, 'confirmed':False, 'awaiting_resume':False})
                    if state['mode'] == 'yield_hold' and native_speed <= numerical:
                        if hold['start'] is None:
                            hold.update(start=timestamp, low=p[:2], high=p[:2], confirmed=False)
                        hold['low'] = tuple(min(hold['low'][k], p[k]) for k in (0, 1))
                        hold['high'] = tuple(max(hold['high'][k], p[k]) for k in (0, 1))
                        excursion = _distance(hold['low'], hold['high'])
                        if excursion > .001:
                            hold.update(start=timestamp, low=p[:2], high=p[:2], confirmed=False)
                        if timestamp-hold['start'] >= .5-1e-7 and excursion <= .001 and not hold['confirmed']:
                            stat['observed_yield_hold_count'] += 1
                            hold.update(confirmed=True, awaiting_resume=True)
                    else:
                        hold.update(start=None, low=None, high=None, confirmed=False)
                        if state['mode'] == 'walking' and native_speed > .02 and hold['awaiting_resume']:
                            stat['observed_yield_hold_then_moving_resume_count'] += 1
                            hold['awaiting_resume'] = False
                    previous_targets[path] = state
                    previous_native_times[path] = timestamp
                else:
                    if row.get('motion_model') == YIELD_MODEL:
                        raise ValueError('Reactive row lacks a matching declared summary motion model')
                    target = _loop_position(route, max(0., timestamp-dt-clock))
                error = _distance(p, target)
                error_field = 'maximum_native_target_position_error_m' if reactive else 'maximum_loop_position_error_m'
                stat[error_field] = max(stat[error_field], error)
                stat['records'] += 1
                stat['native_peak_planar_speed_m_s'] = max(stat['native_peak_planar_speed_m_s'], math.hypot(*velocity[:2]))
                if path in old_positions:
                    stat['native_path_travel_m'] += _distance(p[:2], old_positions[path][:2])
                old_positions[path] = p
                if error > .005:
                    raise ValueError(f'{route["worker_id"]} native/{"target" if reactive else "loop"} position error {error:.8g}m exceeds .005m')
                radius = _number(route.get('visual_sweep_radius_m'), 'visual radius')
                buffer = _number(route.get('sample_buffer_m'), 'route sample buffer')
                if radius < .38-1e-9 or buffer < .025-1e-9:
                    raise ValueError('Recorded static visual sweep assumptions are weaker than .38+.025m')
                clearance = min(_point_rectangle(p, box) for box in rectangles)-radius-buffer
                stat['minimum_static_visual_clearance_m'] = min(stat['minimum_static_visual_clearance_m'], clearance)
                actual[path] = p
            for i, a in enumerate(expected):
                for b in list(expected)[i+1:]:
                    key = expected[a]['worker_id']+'/'+expected[b]['worker_id']
                    pairs[key] = min(pairs.get(key, math.inf), _distance(actual[a][:2], actual[b][:2]))
        checks['native_fields_and_worker_ids'] = True
        checks['native_target_position_error_le_5mm' if reactive else 'loop_position_error_le_5mm'] = True
        if reactive:
            checks['reactive_target_integration_and_observed_step_bounds'] = True
            checks['native_worker_speed_within_design_plus_5mm_s'] = True
            checks['native_robot_snapshot_times_valid'] = True
        checks['sampled_pairwise_center_distance_ge_1_5m'] = all(x >= 1.5-1e-7 for x in pairs.values())
        checks['sampled_static_visual_clearance_ge_0_6m'] = all(
            stat['minimum_static_visual_clearance_m'] >= .60-1e-7 for stat in stats.values())
        checks['combined_record_gap_le_50ms'] = gap_max <= maximum_record_gap_s+1e-6
        route_metrics = {}
        bounds = _vector(metadata.get('map_bounds_xy_m'), 4, 'worker map bounds')
        for path, route in expected.items():
            a, b = route['start_position_m'][:2], route['end_position_m'][:2]
            radius, buffer = route['visual_sweep_radius_m'], route['sample_buffer_m']
            center_distance = min(_segment_rectangle(a, b, box) for box in rectangles)
            visual_clearance = center_distance-radius-buffer
            edge_clearance = min(min(a[k], b[k])-bounds[k] for k in (0, 1))
            edge_clearance = min(edge_clearance, *(bounds[k+2]-max(a[k], b[k]) for k in (0, 1)))
            route_metrics[route['worker_id']] = {'minimum_all_loop_static_visual_clearance_m': visual_clearance,
                                                 'minimum_all_loop_map_edge_margin_m': edge_clearance-radius-buffer-.60}
        checks['whole_loop_static_visual_clearance_ge_0_6m'] = all(
            x['minimum_all_loop_static_visual_clearance_m'] >= .60-1e-7 for x in route_metrics.values())
        checks['whole_loop_inside_map_with_visual_static_margin'] = all(
            x['minimum_all_loop_map_edge_margin_m'] >= -1e-7 for x in route_metrics.values())
        probe = summary.get('probe_only') is True
        mode = summary.get('mode')
        full = mode == 'full' and not probe
        result['requested_scope'] = 'full_people_records' if full else ('probe_only' if probe else str(mode)+'_only')
        phases = {row['phase'] for row in stream_frames['manipulation']}
        expected_phases = ('settle',) if probe else LOAD_PHASES+(UNLOAD_PHASES if full else ())
        checks['requested_manipulation_phases_present'] = set(expected_phases) <= phases
        phase_durations = summary.get('phase_durations_s')
        if not isinstance(phase_durations, Mapping):
            raise ValueError('Summary lacks measured manipulation phase durations')
        phase_coverage = {}
        for phase in expected_phases:
            duration = _number(phase_durations.get(phase), phase+' duration')
            if duration <= 0:
                raise ValueError('Invalid saved phase duration')
            times = [row['time'] for row in manipulation_frames if row['phase'] == phase]
            # Native people are persisted once per ten 2ms steps; allowance
            # covers the last unsaved bin and phase-boundary sample alignment.
            span = times[-1]-times[0] if times else 0.
            phase_coverage[phase] = {'saved_duration_s': duration, 'sampled_span_s': span,
                                     'records': len(times), 'complete_with_sampling_allowance':
                                     bool(times and span+10*dt >= duration-1e-6)}
        checks['manipulation_phase_duration_coverage'] = all(
            value['complete_with_sampling_allowance'] for value in phase_coverage.values())
        if full:
            checks['home_approach_navigation_records_present'] = 'approach' in stream_frames
            checks['loaded_navigation_records_present'] = 'loaded_navigation' in stream_frames
            for label, required in (('approach', {'navigate', 'dock'}),
                                    ('loaded_navigation', {'undock', 'navigate', 'dock'})):
                checks[label+'_phases_present'] = required <= {
                    row['phase'] for row in stream_frames.get(label, [])}
            checks['final_unload_is_last_worker_phase'] = frames[-1]['phase'] == 'unload_place_settle'
            checks['people_recording_starts_at_clock_origin'] = frames[0]['time']-clock <= maximum_record_gap_s+1e-6
            if 'approach' in stream_frames and 'loaded_navigation' in stream_frames:
                approach_end = stream_frames['approach'][-1]['time']
                load_end = max(row['time'] for row in manipulation_frames if row['phase'] in LOAD_PHASES)
                unload_start = min(row['time'] for row in manipulation_frames if row['phase'] in UNLOAD_PHASES)
                loaded_start = stream_frames['loaded_navigation'][0]['time']
                loaded_end = stream_frames['loaded_navigation'][-1]['time']
                checks['home_load_drive_unload_order'] = (approach_end < manipulation_frames[0]['time']
                                                          <= load_end < loaded_start <= loaded_end < unload_start)
        result['full_people_record_coverage'] = bool(full and all(checks.values()))
        result['producer_passed_reported'] = summary.get('passed')
        metrics.update(worker_count=len(expected), native_frames=len(frames),
                       first_timestamp_s=frames[0]['time'], last_timestamp_s=frames[-1]['time'],
                       sampled_span_s=frames[-1]['time']-frames[0]['time'],
                       maximum_combined_record_gap_s=gap_max, loop_clock_start_s=clock,
                       target_submission_offset_s=dt, workers=stats,
                       sampled_pairwise_center_min_m=pairs, whole_route_geometry=route_metrics,
                       manipulation_phase_coverage=phase_coverage,
                       streams={key: {'records': len(value), 'phases': sorted({x['phase'] for x in value})}
                                for key, value in stream_frames.items()})
        if reactive:
            metrics.update(native_speed_numerical_tolerance_m_s=NATIVE_YIELD_SPEED_TOLERANCE_M_S,
                native_step_distance_numerical_tolerance_m=.0005,
                robot_circle_scope='fresh native robot axle with conservative external-people exclusion radius including margin; not robot QP radius',
                sampled_observed_yield_hold_count=sum(s['observed_yield_hold_count'] for s in stats.values()),
                sampled_observed_yield_hold_then_moving_resume_count=sum(s['observed_yield_hold_then_moving_resume_count'] for s in stats.values()))
        result['passed'] = all(checks.values())
    except (ValueError, KeyError, TypeError, IndexError) as error:
        result['error'] = str(error)
        checks['valid_complete_input_contract'] = False
        result['full_people_record_coverage'] = False
    result['failed_checks'] = [key for key, value in checks.items() if not value]
    return result


def validate_people_paths(summary_path, *, manipulation_path=None, collision_path=None,
                          approach_path=None, navigation_path=None, initial_path=None,
                          static_supplement_path=None):
    """Load/hash concrete files and retain missing full-leg files as failures."""
    summary_path = Path(summary_path)
    directory = summary_path.parent
    sources = {}
    def read(path):
        path = Path(path)
        content = path.read_bytes()
        sources[str(path.resolve())] = hashlib.sha256(content).hexdigest()
        return json.loads(content)
    try:
        summary = read(summary_path)
        manipulation = read(manipulation_path or directory/'people_manipulation_samples.json')
        authoring = directory/'people_collision_snapshot.json'
        collision = Path(collision_path) if collision_path else (
            authoring if authoring.is_file() else directory/'warehouse_collision_snapshot.json')
        snapshot = read(collision)
        def optional(path):
            return read(path) if Path(path).is_file() else None
        approach = optional(approach_path or directory/'approach'/'people_trace.json')
        navigation = optional(navigation_path or directory/'people_trace.json')
        # An explicitly supplied path is required; absent legacy defaults remain
        # absent and must still pass the unchanged clock-origin/gap checks.
        initial = read(initial_path) if initial_path is not None else optional(
            directory/'people_initial_samples.json')
        supplement = read(static_supplement_path) if static_supplement_path else []
        if isinstance(supplement, Mapping):
            supplement = supplement.get('rectangles', supplement.get('obstacle_rectangles', []))
        result = validate_people(summary, manipulation, snapshot, approach_trace=approach,
                                 navigation_trace=navigation, initial_records=initial,
                                 static_supplement=supplement,
                                 inventory_role='authoring' if collision.resolve() == authoring.resolve() else 'actual_recheck')
    except (OSError, ValueError, TypeError) as error:
        result = {'passed': False, 'full_people_record_coverage': False,
                  'transport_success_verified': False, 'error': str(error),
                  'failed_checks': ['required_source_files_readable'],
                  'scope': 'saved native worker audit; required inputs incomplete'}
    result['input_sha256'] = sources
    result['auditor_source_sha256'] = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input', required=True, type=Path, help='run directory or summary.json')
    parser.add_argument('--output', required=True, type=Path, help='new JSON file; never overwrite evidence')
    parser.add_argument('--collision-snapshot', type=Path)
    parser.add_argument('--static-supplement', type=Path)
    parser.add_argument('--initial-samples', type=Path,
                        help='native initial-settle people records; default: run/people_initial_samples.json')
    args = parser.parse_args(argv)
    if args.output.exists():
        parser.error('output already exists; use a new audit file')
    summary = args.input/'summary.json' if args.input.is_dir() else args.input
    result = validate_people_paths(summary, collision_path=args.collision_snapshot,
                                  static_supplement_path=args.static_supplement,
                                  initial_path=args.initial_samples)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open('x') as output:
        json.dump(result, output, indent=2, allow_nan=False)
        output.write('\n')
    print(json.dumps({'passed': result['passed'], 'full_people_record_coverage': result['full_people_record_coverage'],
                      'failed_checks': result['failed_checks'], 'output': str(args.output.resolve())}))
    return 0 if result['passed'] else 1


if __name__ == '__main__':
    sys.exit(main())
