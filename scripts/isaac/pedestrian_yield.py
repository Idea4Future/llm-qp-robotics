"""Continuous, CPU-only robot-priority yielding on a worker's fixed lane.

Robot ground truth and planned paths are inputs to this EXTERNAL pedestrian
behavior only. They must never become robot QP/PeopleStopGate observations.
The caller applies returned positions as native kinematic targets; this module
does not touch a simulator, physics poses, or render transforms. Reservations
are design buffers, not a calibrated human perception or safety model.
"""
from __future__ import annotations

import math
import numpy as np


class PedestrianYieldBlocked(RuntimeError):
    """No admissible lane target, invalid clock, or continuous motion impossible."""
    def __init__(self, reason, evidence=None):
        self.reason = reason
        self.evidence = evidence or {}
        super().__init__(reason)


def _vector(value, name):
    array = np.asarray(value, dtype=float)
    if array.shape != (2,) or not np.isfinite(array).all():
        raise ValueError(name + " must be finite XY")
    return array


def _finite(value, name, *, positive=False):
    if isinstance(value, (bool, np.bool_)):
        raise ValueError(name + " must be numeric")
    result = float(value)
    if not math.isfinite(result) or (positive and result <= 0):
        raise ValueError(name + " must be finite" + (" and positive" if positive else ""))
    return result


def _quadratic_interval(offset, direction, radius, lower, upper):
    """Exact interval where ||offset+s*direction|| <= radius."""
    aa = float(direction @ direction)
    bb = 2 * float(offset @ direction)
    cc = float(offset @ offset) - radius * radius
    if aa < 1e-20:
        return [(lower, upper)] if cc <= 1e-12 else []
    disc = bb * bb - 4 * aa * cc
    if disc < -1e-12:
        return []
    root = math.sqrt(max(0., disc))
    left = max(lower, (-bb - root) / (2 * aa))
    right = min(upper, (-bb + root) / (2 * aa))
    return [(left, right)] if left <= right + 1e-12 else []


def _capsule_intervals(origin, direction, length, a, b, radius):
    """Lane intersection with a segment capsule, including endpoint disks."""
    vector = b - a
    squared = float(vector @ vector)
    if squared < 1e-20:
        return _quadratic_interval(origin - a, direction, radius, 0., length)
    p0 = float((origin - a) @ vector) / squared
    p1 = float(direction @ vector) / squared
    cuts = [0., length]
    if abs(p1) > 1e-14:
        cuts += [s for s in (-p0 / p1, (1 - p0) / p1) if 0 < s < length]
    cuts = sorted(set(cuts))
    result = []
    for low, high in zip(cuts[:-1], cuts[1:]):
        projection = p0 + p1 * ((low + high) / 2)
        if projection <= 0:
            offset, slope = origin - a, direction
        elif projection >= 1:
            offset, slope = origin - b, direction
        else:
            offset = origin - a - p0 * vector
            slope = direction - p1 * vector
        result.extend(_quadratic_interval(offset, slope, radius, low, high))
    return result


def _forward_path(path, robot):
    points = np.asarray(path, dtype=float)
    if points.ndim != 2 or points.shape[1] != 2 or not len(points) or not np.isfinite(points).all():
        raise ValueError("remaining_path_xy_m must be finite Nx2")
    points = points[np.r_[True, np.linalg.norm(np.diff(points, axis=0), axis=1) > 1e-9]]
    if len(points) == 1:
        return points
    vectors = np.diff(points, axis=0)
    fraction = np.clip(np.sum((robot - points[:-1]) * vectors, axis=1) / np.sum(vectors**2, axis=1), 0., 1.)
    projected = points[:-1] + fraction[:, None] * vectors
    index = int(np.argmin(np.linalg.norm(projected - robot, axis=1)))
    return np.vstack([projected[index], points[index + 1:]])


class PedestrianYieldPolicy:
    """A worker's scalar lane position/velocity; acceleration-limited FSM.

    Signed nominal_speed_m_s follows start -> end when positive. Safe intervals
    reserve the current robot disk, forward planned polyline, and optional
    active dock disks. Every disk radius means ROBOT circle/work-zone radius;
    the policy adds .90 trigger + .22 worker capsule + .15 extra center buffer.
    Static/worker-pair lane sweeps must already be admitted by the caller.
    A held person resumes only if its next braking horizon is reservation-free.
    The nominal command may change while held; position is never copied from
    an independent loop clock, so resume cannot jump to a later loop target.
    """
    revision = "robot-priority-pedestrian-yield-v1"
    center_buffer_m = .90 + .22 + .15
    parking_inset_m = .025
    preview_extra_m = .15

    def __init__(self, start_xy_m, end_xy_m, maximum_speed_m_s, *, initial_s_m=0.,
                 initial_speed_m_s=0., accel_max_m_s2=.4, maximum_update_gap_s=.05):
        self.start = _vector(start_xy_m, "start_xy_m")
        self.end = _vector(end_xy_m, "end_xy_m")
        self.length = float(np.linalg.norm(self.end - self.start))
        if self.length < 1e-6:
            raise ValueError("Worker lane must have nonzero length")
        self.direction = (self.end - self.start) / self.length
        self.vmax = _finite(maximum_speed_m_s, "maximum_speed_m_s", positive=True)
        self.acceleration = _finite(accel_max_m_s2, "accel_max_m_s2", positive=True)
        if self.vmax > .30 + 1e-12 or self.acceleration > .4 + 1e-12:
            raise ValueError("Worker peak speed .30/acceleration .4 design caps cannot be relaxed")
        self.maximum_gap = _finite(maximum_update_gap_s, "maximum_update_gap_s", positive=True)
        self.s = _finite(initial_s_m, "initial_s_m")
        self.speed = _finite(initial_speed_m_s, "initial_speed_m_s")
        if not 0 <= self.s <= self.length or abs(self.speed) > self.vmax + 1e-12:
            raise ValueError("Initial worker lane state outside limits")
        stopping_distance = self.speed**2 / (2 * self.acceleration)
        available = self.length - self.s if self.speed > 0 else self.s
        if stopping_distance > available + 1e-12:
            raise ValueError("Initial worker speed cannot stop inside its lane")
        self.mode = "walking"
        self.target = None
        self.last_time = self.origin_time = None
        self.yield_count = self.resume_count = 0
        self._geometry_cache_key = self._geometry_cache_value = None
        self._geometry_cache_hit = False

    def _reservations(self, robot, circle, path, disks):
        # The caller refreshes a measured external-person context at 50Hz;
        # unchanged geometry is reusable during its intervening 2ms targets.
        # Exact input equality only: no rounding, approximate caching or stale
        # reservation acceptance. Copies/bytes prevent caller mutation races.
        if path is None:
            path_bytes = None
        else:
            raw_path = np.asarray(path,dtype=float)
            if raw_path.ndim != 2 or raw_path.shape[1] != 2 or not len(raw_path) or not np.isfinite(raw_path).all():
                raise ValueError('remaining_path_xy_m must be finite Nx2')
            path_bytes = (raw_path.shape,raw_path.tobytes())
        disk_key = tuple((tuple(_vector(d["center_xy_m"],"reserved disk center")),
                          _finite(d["radius_m"],"reserved disk radius",positive=True)) for d in disks or [])
        key=(robot.tobytes(),circle,path_bytes,disk_key)
        if key == self._geometry_cache_key:
            self._geometry_cache_hit = True
            return self._geometry_cache_value
        self._geometry_cache_hit = False
        segments = [(robot, robot, circle + self.center_buffer_m)]
        if path is not None:
            forward = _forward_path(path, robot)
            if len(forward) == 1:
                segments.append((forward[0], forward[0], circle + self.center_buffer_m))
            else:
                segments += [(a, b, circle + self.center_buffer_m) for a, b in zip(forward[:-1], forward[1:])]
        for disk in disks or []:
            center = _vector(disk["center_xy_m"], "reserved disk center")
            radius = _finite(disk["radius_m"], "reserved disk radius", positive=True)
            segments.append((center, center, radius + self.center_buffer_m))
        intervals = []
        for a, b, radius in segments:
            intervals.extend(_capsule_intervals(self.start, self.direction, self.length, a, b, radius))
        merged = []
        for low, high in sorted(intervals):
            if merged and low <= merged[-1][1] + 1e-10:
                merged[-1][1] = max(merged[-1][1], high)
            else:
                merged.append([low, high])
        safe = []
        cursor = 0.
        for low, high in merged:
            if low - self.parking_inset_m > cursor:
                safe.append([cursor, low - self.parking_inset_m])
            cursor = min(self.length, high + self.parking_inset_m)
        if cursor < self.length:
            safe.append([cursor, self.length])
        self._geometry_cache_key = key
        self._geometry_cache_value = (merged,safe,segments)
        self._segment_a = np.asarray([a for a,b,r in segments])
        self._segment_vector = np.asarray([b-a for a,b,r in segments])
        self._segment_squared = np.sum(self._segment_vector**2,axis=1)
        self._segment_radius = np.asarray([r for a,b,r in segments])
        return self._geometry_cache_value

    @staticmethod
    def _inside(s, intervals):
        return any(low - 1e-10 <= s <= high + 1e-10 for low, high in intervals)

    def _horizon_clear(self, nominal, safe):
        speed = max(abs(self.speed), abs(nominal))
        horizon = speed**2 / (2 * self.acceleration) + speed * self.maximum_gap + self.preview_extra_m
        end = np.clip(self.s + math.copysign(horizon, nominal), 0., self.length) if nominal else self.s
        return any(low - 1e-10 <= min(self.s, end) and max(self.s, end) <= high + 1e-10 for low, high in safe)

    def _parking_target(self, safe):
        candidates = [float(np.clip(self.s, low, high)) for low, high in safe]
        # Prefer retreating against current travel at an equal-distance tie.
        preferred = self.speed if abs(self.speed) > 1e-10 else self._nominal
        return min(candidates, key=lambda target: (abs(target - self.s), (target - self.s) * preferred))

    def _increment(self, new_speed, dt):
        delta = new_speed - self.speed
        ramp = min(dt, abs(delta) / self.acceleration)
        signed_accel = math.copysign(self.acceleration, delta) if delta else 0.
        return self.speed * ramp + .5 * signed_accel * ramp**2 + new_speed * (dt - ramp)

    def _continuous_next(self, desired, dt):
        """Reach target speed at max acceleration, then coast within this tick.

        Also retain sufficient distance for braking before a fixed lane end.
        This prevents sampling-induced overshoot without clamping position.
        """
        candidate = self.speed + float(np.clip(desired-self.speed, -self.acceleration*dt, self.acceleration*dt))
        direction = math.copysign(1., candidate) if candidate else (math.copysign(1., self.speed) if self.speed else 0.)
        available = self.length-self.s if direction > 0 else self.s
        v0 = direction*self.speed
        if direction and v0 >= 0 and direction*candidate >= 0:
            def required(v):
                return direction*self._increment(direction*v, dt) + v*v/(2*self.acceleration)
            high = direction*candidate
            if required(high) > available:
                low = max(0., v0-self.acceleration*dt)
                if required(low) > available + 1e-10:
                    raise PedestrianYieldBlocked('Continuous lane-end braking became infeasible')
                for _ in range(32):
                    middle=(low+high)/2
                    if required(middle) <= available:low=middle
                    else:high=middle
                candidate=direction*low
        return candidate, self._increment(candidate,dt)

    def update(self, now, nominal_speed_m_s, *, robot_axle_xy_m, robot_circle_m,
               remaining_path_xy_m=None, reserved_disks=None):
        now = _finite(now, "now")
        nominal = _finite(nominal_speed_m_s, "nominal_speed_m_s")
        robot = _vector(robot_axle_xy_m, "robot_axle_xy_m")
        circle = _finite(robot_circle_m, "robot_circle_m", positive=True)
        if now < 0 or abs(nominal) > self.vmax + 1e-9:
            raise ValueError("Worker time/nominal speed outside admitted limits")
        dt = 0. if self.last_time is None else now - self.last_time
        if dt < -1e-12 or dt > self.maximum_gap + 1e-9:
            raise PedestrianYieldBlocked("Worker clock reversed/stale", {"dt_s": dt})
        unsafe, safe, segments = self._reservations(robot, circle, remaining_path_xy_m, reserved_disks)
        if not safe:
            raise PedestrianYieldBlocked("No safe point on the worker's admitted lane", {
                "lane_length_m": self.length, "unsafe_intervals_s_m": unsafe,
                "robot_circle_m": circle, "center_buffer_m": self.center_buffer_m})
        self._nominal = nominal
        risky = self._inside(self.s, unsafe) or (nominal != 0 and not self._horizon_clear(nominal, safe))
        if self.mode == "walking" and risky:
            self.mode = "yield_brake" if abs(self.speed) > 1e-9 else "backoff"
            self.yield_count += 1
            self.target = self._parking_target(safe)
        if self.mode in ("backoff", "yield_hold"):
            self.target = self._parking_target(safe)
            if self.mode == "yield_hold" and (abs(self.target-self.s)>1e-5 or self._inside(self.s,unsafe)):
                self.mode = "backoff"
            if abs(self.target - self.s) <= 1e-5 and abs(self.speed) <= 1e-8 and not self._inside(self.s, unsafe):
                self.mode = "yield_hold"
                if nominal != 0 and self._horizon_clear(nominal, safe):
                    self.mode = "walking"
                    self.target = None
                    self.resume_count += 1
        old_s, old_speed = self.s, self.speed
        if self.mode == "yield_brake":
            desired = 0.
        elif self.mode == "backoff":
            distance = self.target - self.s
            desired = math.copysign(min(self.vmax, max(0., math.sqrt(2 * self.acceleration * abs(distance)) - self.acceleration * dt)), distance) if abs(distance) > 1e-10 else 0.
        elif self.mode == "yield_hold":
            desired = 0.
        else:
            available = max(0., self.length - self.s if nominal > 0 else self.s)
            desired = math.copysign(min(abs(nominal), max(0., math.sqrt(2 * self.acceleration * available) - self.acceleration * dt)), nominal) if nominal else 0.
        if dt > 0:
            # Exact constant-acceleration ramp, followed by constant velocity
            # if target speed is reached before this tick ends. No pose clamp.
            self.speed, increment = self._continuous_next(desired,dt)
            self.s += increment
            if self.s < -1e-9 or self.s > self.length + 1e-9:
                self.s, self.speed = old_s, old_speed
                raise PedestrianYieldBlocked("Continuous target would leave the admitted lane")
        if self.mode == "yield_brake" and abs(self.speed) <= 1e-9:
            self.mode = "backoff"
            self.target = self._parking_target(safe)
        # Reaching a target asymptotically can leave sub-millimetric motion;
        # brake when the next signed motion would cross it, without snapping.
        if self.mode == "backoff" and (self.target - self.s) * self.speed < 0:
            self.mode = "yield_brake"
        self.origin_time = now if self.origin_time is None else self.origin_time
        self.last_time = now
        position = self.start + self.direction * self.s
        projection=np.divide(np.sum((position-self._segment_a)*self._segment_vector,axis=1),
                             self._segment_squared,out=np.zeros(len(segments)),where=self._segment_squared>1e-20)
        closest=self._segment_a+np.clip(projection,0.,1.)[:,None]*self._segment_vector
        clearance=float(np.min(np.linalg.norm(position-closest,axis=1)-self._segment_radius))
        return {"revision": self.revision, "timestamp_s": now, "origin_time_s": self.origin_time,
                "elapsed_s": now - self.origin_time, "dt_s": dt,
                "position_target_xy_m": position.tolist(), "s_m": float(self.s),
                "previous_s_m": float(old_s), "speed_m_s": float(self.speed),
                "previous_speed_m_s": float(old_speed),
                "acceleration_m_s2": float((self.speed - old_speed) / dt) if dt else 0.,
                "peak_acceleration_m_s2": self.acceleration if dt and self.speed != old_speed else 0.,
                "acceleration_ramp_duration_s": abs(self.speed-old_speed)/self.acceleration if dt else 0.,
                "ramp_duration_s": abs(self.speed-old_speed)/self.acceleration if dt else 0.,
                "command_accel_m_s2": math.copysign(self.acceleration,self.speed-old_speed) if dt and self.speed != old_speed else 0.,
                "velocity_goal_m_s": float(desired),
                "nominal_speed_m_s": nominal, "mode": self.mode,
                "yielding": self.mode != "walking", "target_safe_s_m": self.target,
                "minimum_center_reservation_clearance_m": clearance,
                "safe_interval_count": len(safe), "reservation_segment_count": len(segments),
                "reservation_geometry_cache_hit": self._geometry_cache_hit,
                "robot_circle_m": circle, "center_buffer_m": self.center_buffer_m,
                "yield_count": self.yield_count, "resume_count": self.resume_count,
                "maximum_speed_m_s": self.vmax, "accel_max_m_s2": self.acceleration,
                "robot_gt_used_for_external_person_behavior_only": True,
                "position_copied_from_loop_clock": False}
