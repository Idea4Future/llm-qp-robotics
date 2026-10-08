"""Native scripted workers with explicit robot-priority yielding.

Robot ground truth is an input to the external people scenario ONLY. Robot
navigation still receives native LiDAR and its own state, never these targets.
No humanoid balance, human vision, or learned social behavior is modeled.
"""
from __future__ import annotations

import copy
import math
import numpy as np

from warehouse_people import WarehousePeople as ClockPeople, _LoopWorker
from pedestrian_yield import PedestrianYieldPolicy


MOTION_MODEL = 'robot-priority-yield-v1'


class YieldingWorker(_LoopWorker):
    def __init__(self,stage,metadata,*,maximum_update_gap_s=.05):
        super().__init__(stage,metadata,maximum_update_gap_s=maximum_update_gap_s)
        # Camera initialization can render after native view initialization
        # but before start_loop/step has produced a gait pose.
        self._pending_gait_angles = None

    def start_loop(self, simulation_time):
        result = super().start_loop(simulation_time)
        route = self._scenario
        self._policy = PedestrianYieldPolicy(route['start_position_m'][:2],
            route['end_position_m'][:2], route['maximum_script_speed_m_s'])
        self._direction = 1.
        self._endpoint_since = None
        self._yaw = float(route['yaw_start_rad'])
        self._gait_distance = 0.
        self._reactive_target = None
        result['motion_model'] = MOTION_MODEL
        return result

    def step(self, simulation_time, *, robot_context, release=False):
        now = float(simulation_time)
        dt = now - self._last_time
        if not math.isfinite(now) or dt < -1e-9 or dt > self.maximum_gap + 1e-9:
            raise RuntimeError('Yielding worker clock missing/reversed')
        route = self._scenario
        start = np.asarray(route['start_position_m'], float)
        end = np.asarray(route['end_position_m'], float)
        length = float(np.linalg.norm(end-start))
        previous_s = 0. if self._reactive_target is None else self._reactive_target['s_m']
        previous_speed = 0. if self._reactive_target is None else self._reactive_target['speed_m_s']
        remaining = length-previous_s if self._direction > 0 else previous_s
        yielding = self._reactive_target is not None and self._reactive_target['yielding']
        if remaining < .001 and abs(previous_speed) < .005 and not yielding:
            if self._endpoint_since is None:
                self._endpoint_since = now
            if now-self._endpoint_since >= route['endpoint_turn_duration_s']:
                self._direction *= -1.
                self._endpoint_since = None
                remaining = length-previous_s if self._direction > 0 else previous_s
        else:
            self._endpoint_since = None
        nominal = self._direction * min(route['maximum_script_speed_m_s'],
                                       math.sqrt(max(0., .8*remaining)))
        if self._endpoint_since is not None or now-self._clock_origin < route['start_delay_s']:
            nominal = 0.
        target = self._policy.update(now, nominal, **robot_context)
        position = np.r_[target['position_target_xy_m'], 0.]
        if np.linalg.norm(position-self._last_position) > route['maximum_script_speed_m_s']*dt+1e-7:
            raise RuntimeError('Yielding target exceeded the admitted worker speed')
        speed = target['speed_m_s']
        direction = math.copysign(1., speed) if abs(speed) > .002 else self._direction
        desired_yaw = route['yaw_start_rad'] + (math.pi if direction < 0 else 0.)
        angle = math.atan2(math.sin(desired_yaw-self._yaw), math.cos(desired_yaw-self._yaw))
        self._yaw += float(np.clip(angle, -math.pi*dt, math.pi*dt))
        self._view.set_kinematic_targets(self._transform(position, self._yaw), np.array([0], dtype=np.uint32))
        self._gait_distance += abs(target['s_m']-previous_s)
        phase = 2*math.pi*self._gait_distance/.65
        pending = {}
        for label, parts in self._animation_attributes.items():
            swing = math.sin(phase)*(1. if label == 'L' else -1.) if abs(speed) > .001 else 0.
            hip = 18.*swing
            knee = 22.*max(0., -swing)
            pending[label] = {'hip':hip, 'knee':knee, 'ankle':-hip-knee,
                              'shoulder':-13.*swing, 'elbow':-7.}
        # Keep the exact last physics tick's angles. USD writes belong to the
        # render hook; native targets and gait-distance integration stay 500Hz.
        self._pending_gait_angles = pending
        self._last_time, self._last_position = now, position
        self._reactive_target = copy.deepcopy(target)
        return {'timestamp_s':now, 'prim_path':self.path, 'state':target['mode'],
                'position_target_m':position.tolist(), 'yaw_target_rad':self._yaw,
                'reactive_target':copy.deepcopy(target), 'native_target_only':True,
                'external_pose_jump':False, 'motion_model':MOTION_MODEL,
                'release_signal_used':False, 'ignored_robot_release_signal':bool(release)}

    def sync_visual_from_physics(self,simulation_time):
        # Both measured Visual-root synchronization and the latest limb pose
        # precede the caller's World.render/Fabric update. Extra renders reuse
        # these exact angles, rather than recomputing phase at render time.
        record = super().sync_visual_from_physics(simulation_time)
        if self._pending_gait_angles is not None:
            for label, angles in self._pending_gait_angles.items():
                for name, value in angles.items():
                    self._animation_attributes[label][name].Set(value)
        return record

    def measure(self, simulation_time):
        measured = super().measure(simulation_time)
        measured.update(motion_independent_of_robot_stop=False, motion_model=MOTION_MODEL,
                        reactive_target=copy.deepcopy(self._reactive_target))
        return measured


class WarehousePeople(ClockPeople):
    def __init__(self, stage, metadata, *, maximum_update_gap_s=.05):
        self.metadata = copy.deepcopy(metadata)
        self.workers = [YieldingWorker(stage, m, maximum_update_gap_s=maximum_update_gap_s)
                        for m in metadata['workers']]
        self._start_time = None
        self._robot_provider = None
        self._robot_context = None
        self._robot_state = None
        self._context_time = None
        self._path = None
        self._path_index = 0
        self._destination = None
        self._circle = .95
        self._last_modes = {}
        self.events = []
        self.navigation_reservations = []
        self._emit = None

    def configure_robot(self, provider, *, emit=None):
        if not callable(provider):
            raise ValueError('People behavior requires a native robot state provider')
        self._robot_provider, self._emit = provider, emit

    def set_navigation(self, path_xy, destination_axle_xy, *, robot_circle_m, timestamp_s):
        points = np.asarray(path_xy, float)
        if points.ndim != 2 or points.shape[1] != 2 or len(points) < 2 or not np.isfinite(points).all():
            raise ValueError('People reservation requires the admitted navigation path')
        self._path = points.copy()
        self._path_index = 0
        self._destination = np.asarray(destination_axle_xy, float).copy()
        self._circle = max(.95, float(robot_circle_m))
        self._context_time = None
        self.navigation_reservations.append({'time':float(timestamp_s), 'path_xy_m':points.tolist(),
            'destination_axle_xy_m':self._destination.tolist(), 'robot_circle_m':self._circle,
            'source':'robot route reserved for external pedestrian behavior only'})

    def clear_navigation(self):
        self._path = None
        self._destination = None
        self._context_time = None

    def _context(self, now):
        # Policy geometry refreshes at50Hz; native motion integrates at500Hz.
        if self._context_time is not None and now-self._context_time < .02-1e-9:
            return self._robot_context
        if self._robot_provider is None:
            raise RuntimeError('Robot-priority people require a native state provider')
        state = self._robot_provider()
        axle = np.asarray(state['axle_xy_m'], float)
        path = None
        if self._path is not None:
            a, b = self._path[:-1], self._path[1:]
            vectors = b-a
            lens = np.sum(vectors*vectors, axis=1)
            alpha = np.clip(np.sum((axle-a)*vectors, axis=1)/np.maximum(lens, 1e-12), 0., 1.)
            projection = a+alpha[:,None]*vectors
            distances = np.linalg.norm(projection-axle, axis=1)
            distances[:self._path_index] = np.inf
            self._path_index = int(np.argmin(distances))
            path = np.vstack([projection[self._path_index], self._path[self._path_index+1:]]).tolist()
        self._robot_state = copy.deepcopy(state)
        self._robot_state.update(timestamp_s=now, robot_circle_radius_m=self._circle,
            circle_scope='conservative external people scenario exclusion, includes margin')
        self._context_time = now
        self._robot_context = {'robot_axle_xy_m':axle.tolist(), 'robot_circle_m':self._circle,
            'remaining_path_xy_m':path,
            'reserved_disks':([] if self._destination is None else
                              [{'center_xy_m':self._destination.tolist(), 'radius_m':self._circle}])}
        return self._robot_context

    def start(self, simulation_time, **kwargs):
        result = super().start(simulation_time, **kwargs)
        result.update(motion_independent_of_robot_stop=False, motion_model=MOTION_MODEL)
        return result

    def step(self, simulation_time, *, release=False):
        if self._start_time is None:
            raise RuntimeError('People motion has not started')
        now = float(simulation_time)
        context = self._context(now)
        targets = []
        for worker in self.workers:
            target = worker.step(now, robot_context=context, release=release)
            targets.append(target)
            mode = target['state']
            if self._last_modes.get(worker.path) != mode:
                self.events.append({'time':now, 'worker':worker.path, 'mode':mode,
                                    'position_target_m':target['position_target_m']})
                self._last_modes[worker.path] = mode
        return {'timestamp_s':now, 'workers':targets, 'release_signal_used':False,
                'motion_independent_of_robot_stop':False, 'motion_model':MOTION_MODEL}

    def measure(self, simulation_time):
        measured = super().measure(simulation_time)
        current = self._robot_provider()
        current.update(timestamp_s=float(simulation_time), robot_circle_radius_m=self._circle,
                       circle_scope='conservative external people scenario exclusion, includes margin')
        measured.update(motion_model=MOTION_MODEL,
                        robot_context_timestamp_s=self._context_time,
                        robot_state_before_target=copy.deepcopy(self._robot_state),
                        robot_state_at_measurement=current)
        return measured
