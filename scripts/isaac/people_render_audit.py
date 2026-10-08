"""Read-only audit of rendered worker Fabric transforms against native poses.

Call WorkerVisualAudit(stage).update(sync_record) AFTER World.render, then save
audit.records and audit.summary(). Never changes a rigid body, USD transform,
Fabric attribute, or simulation setting. This is transform evidence, not an
independent RGB/person detector or full human collision-safety certificate.
"""
from __future__ import annotations

import math


def _number(value, label):
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
        raise ValueError(label+' must be a finite number')
    return float(value)


def _vector(value, length, label):
    if value is None or len(value) != length:
        raise ValueError(label+' has an invalid shape')
    return tuple(_number(x, label) for x in value)


def _rotation_row_xyzw(quaternion):
    x, y, z, w = _vector(quaternion, 4, 'native quaternion xyzw')
    norm = math.sqrt(x*x+y*y+z*z+w*w)
    if norm < 1e-12:
        raise ValueError('Native quaternion has zero norm')
    x, y, z, w = (x/norm, y/norm, z/norm, w/norm)
    # Gf/Fabric use row vectors and row3 for translation.
    return ((1-2*(y*y+z*z), 2*(x*y+z*w), 2*(x*z-y*w)),
            (2*(x*y-z*w), 1-2*(x*x+z*z), 2*(y*z+x*w)),
            (2*(x*z+y*w), 2*(y*z-x*w), 1-2*(x*x+y*y)))


def _decompose(matrix):
    if matrix is None or len(matrix) != 4:
        raise ValueError('Fabric worldMatrix is absent or not4x4')
    rows = [_vector(matrix[i], 4, 'Fabric worldMatrix row') for i in range(4)]
    if max(abs(rows[i][3]) for i in range(3)) > 1e-6 or abs(rows[3][3]-1.) > 1e-6:
        raise ValueError('Fabric worldMatrix is not an affine row-vector transform')
    scale = tuple(math.sqrt(sum(rows[i][j]**2 for j in range(3))) for i in range(3))
    if min(scale) < 1e-12:
        raise ValueError('Fabric worldMatrix has a singular scale')
    rotation = tuple(tuple(rows[i][j]/scale[i] for j in range(3)) for i in range(3))
    if max(abs(sum(rotation[i][k]*rotation[j][k] for k in range(3)))
           for i in range(3) for j in range(i)) > 1e-4:
        raise ValueError('Fabric worldMatrix contains unsupported shear')
    a, b, c = rotation
    determinant = a[0]*(b[1]*c[2]-b[2]*c[1])-a[1]*(b[0]*c[2]-b[2]*c[0])+a[2]*(b[0]*c[1]-b[1]*c[0])
    if abs(determinant-1.) > 1e-4:
        raise ValueError('Fabric worldMatrix has a reflection or invalid orientation')
    return tuple(rows[3][:3]), rotation, scale, [list(row) for row in rows]


def audit_worker_matrices(worker, visual_matrix, vest_matrix, uniform_scale=1., *,
                          position_tolerance_m=.001, rotation_tolerance_rad=.002,
                          scale_tolerance=1e-4):
    """Pure CPU math, also useful for fabricated-matrix negative tests.

PPE vest is a fixed cube at nominal local(0,0,1.20) and nominal scale
(.267,.322,.35); the Visual uniform scale is height_m/1.7.
"""
    position = _vector(worker['native_position_m'], 3, 'native position')
    rotation = _rotation_row_xyzw(worker['native_quaternion_xyzw'])
    offset = _vector(worker.get('visual_local_offset_m', (0., 0., 0.)), 3, 'Visual local offset')
    scale = _number(uniform_scale, 'Visual uniform scale')
    if scale <= 0:
        raise ValueError('Visual scale must be positive')
    root_position = tuple(position[j]+sum(offset[i]*rotation[i][j] for i in range(3)) for j in range(3))
    if 'visual_position_target_m' in worker:
        target = _vector(worker['visual_position_target_m'], 3, 'Visual authored target')
        if math.dist(target, root_position) > position_tolerance_m:
            raise ValueError('Visual target disagrees with native pose plus fixed offset')
    expected = {'Visual': (root_position, (scale,)*3),
                'SafetyVest': (tuple(root_position[j]+1.20*scale*rotation[2][j] for j in range(3)),
                               tuple(s*scale for s in (.267, .322, .35)))}
    result = {'passed': True, 'prim_path': worker['prim_path'],
              'visual_prim_path': worker['visual_prim_path'],
              'native_position_m': list(position),
              'native_quaternion_xyzw': list(worker['native_quaternion_xyzw']),
              'expected_uniform_visual_scale': scale, 'transforms': {}}
    for label, matrix in (('Visual', visual_matrix), ('SafetyVest', vest_matrix)):
        actual_position, actual_rotation, actual_scale, raw = _decompose(matrix)
        dot = sum(actual_rotation[i][j]*rotation[i][j] for i in range(3) for j in range(3))
        angle = math.acos(max(-1., min(1., (dot-1.)/2.)))
        expected_position, expected_scale = expected[label]
        error = math.dist(actual_position, expected_position)
        scale_error = max(abs(a-b) for a, b in zip(actual_scale, expected_scale))
        passed = error <= position_tolerance_m and angle <= rotation_tolerance_rad and scale_error <= scale_tolerance
        result['transforms'][label] = {'fabric_world_matrix': raw,
            'position_m': list(actual_position), 'expected_position_m': list(expected_position),
            'position_error_m': error, 'rotation_error_rad': angle,
            'scale': list(actual_scale), 'expected_scale': list(expected_scale),
            'maximum_scale_error': scale_error, 'passed': passed}
        result['passed'] = result['passed'] and passed
    return result


class WorkerVisualAudit:
    """Audit the active usdrt stage's actual post-render world matrices.

Missing initial Fabric matrices are failures with explicit errors, never skips.
Repeated render calls at one physics timestamp retain the last raw record;
every failed call remains in summary.errors even if a later render is valid.
"""
    def __init__(self, stage, *, position_tolerance_m=.001, rotation_tolerance_rad=.002):
        import omni.usd
        from usdrt import Usd as RtUsd
        context = omni.usd.get_context()
        if context.get_stage() != stage:
            raise RuntimeError('WorkerVisualAudit requires the active omni.usd stage')
        self.stage = stage
        self.stage_id = int(context.get_stage_id())
        self.fabric_stage = RtUsd.Stage.Attach(self.stage_id)
        if not self.fabric_stage:
            raise RuntimeError('Could not attach usdrt to active stage '+str(self.stage_id))
        self.position_tolerance_m = _number(position_tolerance_m, 'position tolerance')
        self.rotation_tolerance_rad = _number(rotation_tolerance_rad, 'rotation tolerance')
        if self.position_tolerance_m <= 0 or self.rotation_tolerance_rad <= 0:
            raise ValueError('Audit tolerances must be positive')
        self.records = []
        self.errors = []
        self._indices = {}
        self._workers = None
        self.render_calls = 0

    def _matrix(self, path):
        prim = self.fabric_stage.GetPrimAtPath(path)
        if not prim or not prim.IsValid():
            raise ValueError('Rendered Fabric prim is absent: '+path)
        attribute = prim.GetAttribute('omni:fabric:worldMatrix')
        if not attribute or not attribute.IsValid():
            raise ValueError('Rendered Fabric worldMatrix attribute absent: '+path+'; call AFTER a completed render')
        matrix = attribute.Get()
        if matrix is None:
            raise ValueError('Rendered Fabric worldMatrix has no value: '+path+'; initial render/Fabric population incomplete')
        return matrix

    def _scale(self, worker):
        visual = self.stage.GetPrimAtPath(worker['visual_prim_path'])
        if not visual or not visual.IsValid():
            raise ValueError('Authored Visual prim is absent: '+worker['visual_prim_path'])
        attr = visual.GetAttribute('xformOp:scale')
        scale = _vector(attr.Get() if attr else None, 3, 'Authored Visual scale')
        if min(scale) <= 0 or max(scale)-min(scale) > 1e-6:
            raise ValueError('Authored Visual scale must be positive and uniform')
        if 'height_m' in worker and abs(scale[0]-_number(worker['height_m'], 'worker height')/1.7) > 1e-6:
            raise ValueError('Authored Visual scale disagrees with worker height')
        return scale[0]

    def update(self, sync_record):
        self.render_calls += 1
        record = {'timestamp_s': None, 'workers': [], 'passed': False,
                  'source': 'post-render active usdrt Fabric worldMatrix vs native PhysX pose; read-only'}
        try:
            timestamp = _number(sync_record['timestamp_s'], 'render sync timestamp')
            if timestamp < 0 or self.records and timestamp < self.records[-1]['timestamp_s']-1e-8:
                raise ValueError('Render audit timestamps are negative or reversed')
            record['timestamp_s'] = timestamp
            workers = sync_record['workers']
            if not isinstance(workers, list) or not workers:
                raise ValueError('Render sync record has no workers')
            identities = [worker['prim_path'] for worker in workers]
            if len(set(identities)) != len(identities):
                raise ValueError('Render sync record has duplicate worker identities')
            if self._workers is None:
                self._workers = set(identities)
            elif self._workers != set(identities):
                raise ValueError('Render sync record dropped/changed a worker identity')
            for worker in workers:
                path = worker['prim_path'];visual = worker['visual_prim_path']
                if not isinstance(path, str) or visual != path+'/Visual':
                    raise ValueError('Worker Visual path does not match its native body path')
                record['workers'].append(audit_worker_matrices(worker, self._matrix(visual),
                    self._matrix(visual+'/SafetyVest'), self._scale(worker),
                    position_tolerance_m=self.position_tolerance_m,
                    rotation_tolerance_rad=self.rotation_tolerance_rad))
            record['passed'] = all(worker['passed'] for worker in record['workers'])
            if not record['passed']:
                raise ValueError('Rendered Visual/SafetyVest transform disagrees with native worker pose')
        except (KeyError, TypeError, ValueError, RuntimeError) as error:
            record['error'] = str(error)
            self.errors.append({'render_call': self.render_calls, 'timestamp_s': record['timestamp_s'],
                                'error': str(error)})
        if record['timestamp_s'] is not None:
            timestamp = record['timestamp_s']
            if timestamp in self._indices:
                self.records[self._indices[timestamp]] = record
            else:
                self._indices[timestamp] = len(self.records)
                self.records.append(record)
        return record

    def summary(self):
        transforms = [value for record in self.records for worker in record['workers']
                      for value in worker['transforms'].values()]
        return {'passed': bool(self.records) and not self.errors and all(r['passed'] for r in self.records),
                'record_count': len(self.records), 'render_call_count': self.render_calls,
                'worker_count': len(self._workers or ()), 'stage_id': self.stage_id,
                'maximum_position_error_m': max((r['position_error_m'] for r in transforms), default=None),
                'maximum_rotation_error_rad': max((r['rotation_error_rad'] for r in transforms), default=None),
                'maximum_scale_error': max((r['maximum_scale_error'] for r in transforms), default=None),
                'position_tolerance_m': self.position_tolerance_m,
                'rotation_tolerance_rad': self.rotation_tolerance_rad, 'errors': list(self.errors),
                'physics_state_modified': False,
                'scope': 'Visual root and fixed SafetyVest Fabric transform agreement only; no RGB recognition or full-body collision claim'}
