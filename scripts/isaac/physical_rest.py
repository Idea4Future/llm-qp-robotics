"""Measured stationary-window gate; no engine import or state mutation.

``windowed-rest-v1`` is a NEW stopping criterion, not the old instantaneous
6 mm/s test. Native 2 ms measurements must meet a raw spike cap, an exact
50 ms time-weighted velocity average, and a continuous 0.5 s pose window.
It establishes measured quiescence only, not collision or force safety.
"""
from __future__ import annotations

from collections import deque
from dataclasses import dataclass
import math


REVISION = "windowed-rest-v1"


def _number(value, name):
    if isinstance(value, bool):
        raise ValueError(f"{name} must be a finite number")
    try:
        result = float(value)
    except (TypeError, ValueError, OverflowError) as error:
        raise ValueError(f"{name} must be a finite number") from error
    if not math.isfinite(result):
        raise ValueError(f"{name} must be a finite number")
    return result


def _vector(value, size, name):
    try:
        if len(value) != size:
            raise ValueError(f"{name} must have {size} elements")
        return tuple(_number(v, name) for v in value)
    except TypeError as error:
        raise ValueError(f"{name} must have {size} elements") from error


@dataclass(frozen=True)
class _Sample:
    time: float
    xy: tuple
    yaw: float
    velocity: tuple
    omega: float
    eligible: bool

    def record(self):
        return {"time": self.time, "eligible": self.eligible,
                "axle_xy_m": list(self.xy), "base_yaw_rad": self.yaw,
                "axle_linear_velocity_m_s": list(self.velocity),
                "measured_yaw_rate_rad_s": self.omega}


class MeasuredRestWindow:
    """Process consecutive native physics samples and return rest readiness.

    ``update(now, state, eligible=True)`` returns a boolean. State requires
    ``axle_xy_m`` (2), ``base_yaw_rad``, ``axle_linear_velocity_m_s`` (3),
    and ``measured_yaw_rate_rad_s``. Planar speed is recalculated from XY.
    Missing/nonfinite data, nonincreasing time or a gap greater than max_gap_s
    clears the gate and raises ValueError; the caller must stop or abort.

    The average integrates piecewise-linear measured velocity (trapezoids)
    on [now-window_s, now], interpolating its leading boundary. Excursion
    is the candidate's XY bounding-box diagonal, a conservative upper bound
    on pairwise movement diameter; yaw uses its continuous unwrapped range.
    Pose-window violation restarts the candidate at the current sample;
    raw/filtered/eligibility violation clears it. No thresholds are relaxed
    when reset, and a stationary command alone is never sufficient.
    """
    def __init__(self, *, window_s=.05, quiet_duration_s=.5,
                 filtered_speed_max_m_s=.006,
                 filtered_yaw_rate_max_rad_s=.015,
                 raw_speed_max_m_s=.02, raw_yaw_rate_max_rad_s=.04,
                 position_excursion_max_m=.001, yaw_excursion_max_rad=.005,
                 max_gap_s=.0041):
        self.config = {}
        values = locals().copy()
        for name in ("window_s", "quiet_duration_s", "filtered_speed_max_m_s",
                     "filtered_yaw_rate_max_rad_s", "raw_speed_max_m_s",
                     "raw_yaw_rate_max_rad_s", "position_excursion_max_m",
                     "yaw_excursion_max_rad", "max_gap_s"):
            value = _number(values[name], name)
            if value <= 0:
                raise ValueError(f"{name} must be positive")
            self.config[name] = value
        self.config.update(averaging="piecewise-linear-trapezoid",
                           position_excursion="xy-bounding-box-diagonal")
        self.reset()

    def reset(self):
        self._samples = deque()
        self._history = deque()
        self._last = None
        self._yaw_unwrapped = None
        self._candidate_start = None
        self._xy_low = self._xy_high = None
        self._yaw_low = self._yaw_high = None
        self.ready = False
        self.last_error = None
        self.metrics = {"filtered_window_complete": False,
                        "quiet_duration_s": 0.0, "ready": False}

    def _reject(self, reason):
        self.reset()
        self.last_error = reason
        raise ValueError(reason)

    def _clear_candidate(self):
        self._candidate_start = None
        self._xy_low = self._xy_high = None
        self._yaw_low = self._yaw_high = None
        self.ready = False

    def _start_candidate(self, sample):
        self._candidate_start = sample.time
        self._xy_low = list(sample.xy)
        self._xy_high = list(sample.xy)
        self._yaw_low = self._yaw_high = self._yaw_unwrapped

    def _average(self, now):
        window = self.config["window_s"]
        left = now - window
        # Retain the one sample preceding the exact left interpolation edge.
        while len(self._samples) >= 2 and self._samples[1].time <= left:
            self._samples.popleft()
        if len(self._samples) < 2 or self._samples[0].time > left + 1e-9:
            return None
        total = [0.0, 0.0, 0.0, 0.0]
        samples = list(self._samples)
        for a, b in zip(samples, samples[1:]):
            lo, hi = max(left, a.time), min(now, b.time)
            if hi <= lo:
                continue
            av, bv = (*a.velocity, a.omega), (*b.velocity, b.omega)
            frac_lo = (lo - a.time) / (b.time - a.time)
            frac_hi = (hi - a.time) / (b.time - a.time)
            for k in range(4):
                va = av[k] + frac_lo * (bv[k] - av[k])
                vb = av[k] + frac_hi * (bv[k] - av[k])
                total[k] += .5 * (va + vb) * (hi - lo)
        return [value / window for value in total]

    def update(self, now, state, eligible=True):
        try:
            now = _number(now, "time")
            if not isinstance(eligible, bool):
                raise ValueError("eligible must be a boolean")
            sample = _Sample(now, _vector(state["axle_xy_m"], 2, "axle_xy_m"),
                             _number(state["base_yaw_rad"], "base_yaw_rad"),
                             _vector(state["axle_linear_velocity_m_s"], 3,
                                     "axle_linear_velocity_m_s"),
                             _number(state["measured_yaw_rate_rad_s"],
                                     "measured_yaw_rate_rad_s"), eligible)
            if self._last is not None:
                gap = now - self._last.time
                if gap <= 0:
                    raise ValueError("rest sample time must increase strictly")
                if gap > self.config["max_gap_s"]:
                    raise ValueError("rest sample gap exceeds max_gap_s")
        except (KeyError, TypeError, ValueError) as error:
            self._reject(f"Invalid rest measurement: {error}")
        if self._last is None:
            self._yaw_unwrapped = sample.yaw
        else:
            delta = sample.yaw - self._last.yaw
            self._yaw_unwrapped += math.atan2(math.sin(delta), math.cos(delta))
        self._last = sample
        self._samples.append(sample)
        self._history.append(sample)
        keep_start = now - self.config["window_s"] - self.config["quiet_duration_s"]
        while len(self._history) >= 2 and self._history[1].time <= keep_start:
            self._history.popleft()

        average = self._average(now)
        raw_speed = math.hypot(*sample.velocity[:2])
        filtered_speed = None if average is None else math.hypot(*average[:2])
        raw_good = (raw_speed < self.config["raw_speed_max_m_s"] and
                    abs(sample.omega) < self.config["raw_yaw_rate_max_rad_s"])
        filtered_good = (average is not None and
                         filtered_speed < self.config["filtered_speed_max_m_s"] and
                         abs(average[3]) < self.config["filtered_yaw_rate_max_rad_s"])
        position_excursion = yaw_excursion = 0.0
        pose_reset = False
        if not (eligible and raw_good and filtered_good):
            self._clear_candidate()
        else:
            if self._candidate_start is None:
                self._start_candidate(sample)
            for k in range(2):
                self._xy_low[k] = min(self._xy_low[k], sample.xy[k])
                self._xy_high[k] = max(self._xy_high[k], sample.xy[k])
            self._yaw_low = min(self._yaw_low, self._yaw_unwrapped)
            self._yaw_high = max(self._yaw_high, self._yaw_unwrapped)
            position_excursion = math.hypot(*(hi-lo for lo, hi in zip(self._xy_low, self._xy_high)))
            yaw_excursion = self._yaw_high-self._yaw_low
            if (position_excursion > self.config["position_excursion_max_m"] or
                    yaw_excursion > self.config["yaw_excursion_max_rad"]):
                pose_reset = True
                self._start_candidate(sample)
                self.ready = False
            else:
                self.ready = now-self._candidate_start + 1e-9 >= self.config["quiet_duration_s"]
        duration = 0.0 if self._candidate_start is None else now-self._candidate_start
        self.metrics = {"time": now, "eligible": eligible,
                        "filtered_window_complete": average is not None,
                        "filtered_velocity_m_s": None if average is None else average[:3],
                        "filtered_speed_m_s": filtered_speed,
                        "filtered_yaw_rate_rad_s": None if average is None else average[3],
                        "raw_speed_m_s": raw_speed, "raw_yaw_rate_rad_s": sample.omega,
                        "raw_within_bounds": raw_good, "filtered_within_bounds": filtered_good,
                        "position_excursion_m": position_excursion,
                        "yaw_excursion_rad": yaw_excursion,
                        "pose_window_reset": pose_reset,
                        "candidate_start_time": self._candidate_start,
                        "quiet_duration_s": duration, "ready": self.ready}
        return self.ready

    def evidence(self):
        """Copy raw native samples, configuration and current verdict.

        On readiness the retained trace spans at least window_s+quiet_duration_s
        (0.55 s by default) plus its interpolation bracket. Values are measured
        inputs; no filtered value is substituted for a raw sample.
        """
        return {"revision": REVISION, "config": dict(self.config),
                "ready": self.ready,
                "raw_records": [sample.record() for sample in self._history],
                "metrics": {k: list(v) if isinstance(v, list) else v
                            for k, v in self.metrics.items()},
                "last_error": self.last_error}
