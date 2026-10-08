"""Two local convex velocity QPs, solved numerically by CVXPY/OSQP.

Base: min wt||D(u-reference)||² + we||Du||² + wc||D(u-previous)||²,
u=[axle v, yaw omega], D=diag(1/speed_max,1/yaw_rate_max). The three
terms use the same unit normalization. A pure-pursuit reference encodes path
tracking upstream; this objective tracks its velocity, not an entire path.

Arm: min wt||S(J*qdot-tcp_velocity)||² + we||Dq*qdot||² +
wc||Dq(qdot-previous)||². J is the frozen current 6x7 Jacobian, with
linear XYZ then angular XYZ in the SAME frame as tcp_velocity. S uses separate
linear/angular velocity scales, Dq=diag(1/joint_speed_max).

Nonnegative scalar weights make both Hessians positive semidefinite. Bounds
and frozen-heading point-clearance barriers are affine. These are local
velocity problems, not convex robot dynamics, grasping, collision-free global
planning, MPC, or an exact reimplementation of NEO. NEO's QP robot-control
design is a reference only. No model/data state is edited here.

An infeasible/invalid/failed solve returns an explicit ZERO emergency request.
This may violate the normal acceleration or barrier constraints and cannot
make a moving physical robot stop instantaneously. The executor must handle
that exception, actual actuator realization, fresh sensing and contact checks.
Each call builds a small QP; timings do not establish real-time performance.
"""

from numbers import Real
import time
import warnings

import cvxpy as cp
import numpy as np
import osqp


BASE_TERMS = ("base_path_tracking", "base_effort", "base_command_change")
ARM_TERMS = ("arm_tcp_tracking", "arm_joint_velocity", "arm_command_change")
DEFAULT_BASE_WEIGHTS = dict(zip(BASE_TERMS, (1., .01, .1)))
DEFAULT_ARM_WEIGHTS = dict(zip(ARM_TERMS, (1., .001, .01)))
TASK_SPEC_CONSTRAINT_MAPPING = {
    "base_speed_max": ("m/s", "speed_max"),
    "base_accel_max": ("m/s^2", "accel_max"),
    "base_yaw_rate_max": ("rad/s", "yaw_rate_max"),
    "obstacle_clearance_min": ("m", "margin"),
    "arm_joint_speed_max": ("rad/s", "joint_speed_max"),
    "arm_joint_accel_max": ("rad/s^2", "joint_accel_max"),
}
DEFAULT_SOLVER_OPTIONS = {"eps_abs": 1e-7, "eps_rel": 1e-7, "max_iter": 10000,
                          ("polish" if osqp.__version__.split(".")[0] == "0" else "polishing"): False, "warm_start": True, "verbose": False}


def _number(value, name, positive=False, nonnegative=False):
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, Real) or not np.isfinite(value):
        raise ValueError(f"{name} must be a finite numeric scalar")
    value = float(value)
    if positive and value <= 0 or nonnegative and value < 0:
        raise ValueError(f"{name} has an invalid sign")
    return value


def _array(values, shape, name):
    raw = np.asarray(values, dtype=object)
    if raw.shape != shape or any(isinstance(v, (bool, np.bool_)) or not isinstance(v, Real) for v in raw.flat):
        raise ValueError(f"{name} must have numeric shape {shape}")
    result = np.asarray(values, dtype=float)
    if not np.isfinite(result).all():
        raise ValueError(f"{name} must be finite")
    return result.copy()


def _positive_vector(values, name):
    if isinstance(values, Real):
        return np.full(7, _number(values, name, positive=True))
    result = _array(values, (7,), name)
    if np.any(result <= 0):
        raise ValueError(f"{name} must be positive")
    return result


def _weights(values, defaults):
    result = defaults.copy()
    if values is not None:
        if not isinstance(values, dict) or set(values) - set(defaults):
            raise ValueError(f"weights must use only {list(defaults)}")
        result.update(values)
    return {name: _number(value, name, nonnegative=True) for name, value in result.items()}


def task_spec_qp_config(spec):
    """Map a registry-admitted TaskSpec's typed terms, without calling an LLM.

    Missing objective terms have weight zero; both tracking terms must be
    positive as required by the current task catalog. This is a numeric mapping
    and does not replace registry intent/entity/freshness/skill admission.
    Executor/model bounds remain mandatory: task limits must be intersected
    with them, never used to silently widen the current tested envelope.
    obstacle_clearance_min is the extra margin outside the selected circle.
    """
    base_weights = dict.fromkeys(BASE_TERMS, 0.)
    arm_weights = dict.fromkeys(ARM_TERMS, 0.)
    seen = set()
    for term in spec["objective_terms"]:
        if set(term) != {"term_id", "weight"} or term["term_id"] in seen:
            raise ValueError("Invalid or duplicate objective term")
        name = term["term_id"]
        target = base_weights if name in base_weights else arm_weights
        if name not in target:
            raise ValueError("Unknown objective term")
        target[name] = _number(term["weight"], name, nonnegative=True)
        seen.add(name)
    if base_weights[BASE_TERMS[0]] <= 0 or arm_weights[ARM_TERMS[0]] <= 0:
        raise ValueError("Task catalog requires positive base and arm tracking weights")
    base_limits, arm_limits, seen = {}, {}, set()
    for item in spec["constraints"]:
        if set(item) != {"constraint_id", "unit", "bound"} or item["constraint_id"] in seen:
            raise ValueError("Invalid or duplicate constraint")
        name = item["constraint_id"]
        if name not in TASK_SPEC_CONSTRAINT_MAPPING:
            raise ValueError("Unknown constraint")
        unit, key = TASK_SPEC_CONSTRAINT_MAPPING[name]
        if item["unit"] != unit:
            raise ValueError("Constraint unit mismatch")
        target = arm_limits if name.startswith("arm_") else base_limits
        target[key] = _number(item["bound"], name, positive=True)
        seen.add(name)
    return {"base_weights": base_weights, "arm_weights": arm_weights,
            "base_limits": base_limits, "arm_limits": arm_limits}


def _safe_float(value):
    return float(value) if value is not None and np.isfinite(value) else None


def _solver_stats(problem):
    stats = problem.solver_stats
    if stats is None:
        return {}
    info = getattr(getattr(stats, "extra_stats", None), "info", None)
    result = {"solver_name": stats.solver_name, "iterations": stats.num_iters,
              "solver_time_s": _safe_float(stats.solve_time), "setup_time_s": _safe_float(stats.setup_time)}
    if info is not None:
        for name in ("prim_res", "dual_res", "duality_gap", "setup_time", "solve_time", "polish_time", "run_time"):
            result["osqp_" + name] = _safe_float(getattr(info, name, None))
        result["osqp_status"] = getattr(info, "status", None)
    return result


def _rejected(size, status, reason, started, *, previous=None, acceleration=None, dt=None,
              matrix=None, bound=None, extra=None):
    acceleration_excess = None
    if previous is not None and acceleration is not None and dt is not None:
        acceleration_excess = np.maximum(np.abs(previous) / dt - acceleration, 0.).tolist()
    zero_violation = float(np.maximum(-bound, 0.).max()) if bound is not None and len(bound) else None
    result = {"accepted": False, "status": status, "reason": reason, "solution": [0.] * size,
              "emergency_stop_request": True,
              "emergency_zero_violates_acceleration": bool(any(v > 1e-9 for v in acceleration_excess)) if acceleration_excess is not None else None,
              "emergency_zero_acceleration_excess": acceleration_excess,
              "emergency_zero_constraint_violation_max": zero_violation,
              "emergency_request_meaning": "Zero requested velocity; exceptional acceleration/barrier violations are explicit, physical stopping is external.",
              "wall_time_s": time.perf_counter() - started, "solver_time_s": None,
              "versions": {"cvxpy": cp.__version__, "osqp": osqp.__version__}}
    result.update(extra or {})
    return result


def _solve(variable, objective, hessian, linear, matrix, bound, names, *, previous,
           acceleration, dt, started, solver_options):
    hessian = (hessian + hessian.T) / 2.
    eigenvalues = np.linalg.eigvalsh(hessian)
    if not np.isfinite(hessian).all() or not np.isfinite(linear).all() or eigenvalues.min() < -1e-8:
        raise ValueError("Constructed Hessian is nonfinite or not positive semidefinite")
    constraint = matrix @ variable <= bound
    problem = cp.Problem(cp.Minimize(objective), [constraint])
    metadata = {"is_dcp": problem.is_dcp(), "is_qp": problem.is_qp(),
                "hessian_eigenvalues": eigenvalues.tolist(), "hessian_strictly_positive_definite": bool(eigenvalues.min() > 1e-9),
                "constraint_count": len(bound), "warnings": []}
    if not metadata["is_dcp"] or not metadata["is_qp"]:
        raise ValueError("Construction is not a convex quadratic program")
    before_solver = time.perf_counter()
    try:
        with warnings.catch_warnings(record=True) as captured:
            # Preserve warnings in returned diagnostics instead of emitting the
            # OSQP 1.x raise_error-default deprecation on every control tick.
            warnings.simplefilter("always")
            problem.solve(solver=cp.OSQP, **solver_options)
        metadata["warnings"] = sorted({f"{w.category.__name__}: {w.message}" for w in captured})
    except Exception as error:
        metadata["solve_wall_time_s"] = time.perf_counter() - before_solver
        return _rejected(variable.size, "solver_error", f"{type(error).__name__}: {error}", started,
                         previous=previous, acceleration=acceleration, dt=dt, matrix=matrix, bound=bound, extra=metadata)
    metadata["solve_wall_time_s"] = time.perf_counter() - before_solver
    metadata["solver_stats"] = _solver_stats(problem)
    metadata["solver_time_s"] = metadata["solver_stats"].get("solver_time_s")
    if problem.status != cp.OPTIMAL or variable.value is None:
        return _rejected(variable.size, problem.status, "QP did not return an optimal solution", started,
                         previous=previous, acceleration=acceleration, dt=dt, matrix=matrix, bound=bound, extra=metadata)
    solution = np.asarray(variable.value).reshape(variable.size)
    dual = np.asarray(constraint.dual_value).reshape(len(bound))
    residual = matrix @ solution - bound
    maximum_violation = float(np.maximum(residual, 0.).max())
    stationarity = hessian @ solution + linear + matrix.T @ dual
    metadata["kkt"] = {"primal_violation_max": maximum_violation,
                       "worst_constraint": names[int(np.argmax(residual))],
                       "stationarity_infinity_norm": float(np.linalg.norm(stationarity, np.inf)),
                       "dual_nonnegative_violation_max": float(np.maximum(-dual, 0.).max()),
                       "complementarity_infinity_norm": float(np.max(np.abs(dual * residual))),
                       "residual_units": "Original affine row units; maximum combines different units. OSQP stats refer to its canonicalized QP."}
    if not np.isfinite(solution).all() or not np.isfinite(dual).all() or maximum_violation > 1e-6:
        return _rejected(variable.size, "solution_rejected", "Nonfinite or constraint-violating solver output", started,
                         previous=previous, acceleration=acceleration, dt=dt, matrix=matrix, bound=bound, extra=metadata)
    result = {"accepted": True, "status": problem.status, "reason": None, "solution": solution.tolist(),
              "objective_value": _safe_float(problem.value), "emergency_stop_request": False,
              "emergency_zero_violates_acceleration": False, "emergency_zero_acceleration_excess": None,
              "wall_time_s": time.perf_counter() - started,
              "versions": {"cvxpy": cp.__version__, "osqp": osqp.__version__}}
    result.update(metadata)
    return result


class BaseVelocityQP:
    def __init__(self, *, speed_max=.15, yaw_rate_max=.3, accel_max=.1, yaw_accel_max=.2,
                 wheel_radius=.1, wheel_separation=.53, wheel_rate_max=3.14,
                 weights=None, solver_options=None):
        self.limits = {name: _number(value, name, positive=True) for name, value in {
            "speed_max": speed_max, "yaw_rate_max": yaw_rate_max, "accel_max": accel_max,
            "yaw_accel_max": yaw_accel_max, "wheel_radius": wheel_radius,
            "wheel_separation": wheel_separation, "wheel_rate_max": wheel_rate_max}.items()}
        self.weights = _weights(weights, DEFAULT_BASE_WEIGHTS)
        self.solver_options = DEFAULT_SOLVER_OPTIONS | dict(solver_options or {})

    def solve(self, reference, previous_applied, dt, *, weights=None, lidar_points_axle=None,
              heading=None, radius=None, margin=0., barrier_alpha=1.):
        """Solve for axle v/omega with OPTIONAL fixed-heading clearance rows.

        Points are finite obstacle XY positions relative to the axle, expressed
        in world axes. heading is world yaw. For point p, n=p/||p||:
        n·[cos(yaw),sin(yaw)]*v <= alpha*(||p||-(radius+margin)).
        This static-point, instantaneous circle barrier ignores obstacle motion,
        yaw swept volume, unobserved heights, sensor latency and braking.
        previous_applied must be the ACTUAL post-filter command, not an earlier
        desired/solver command. Bounds are design inputs, not robot ratings.
        """
        started = time.perf_counter()
        previous = acceleration = matrix = bound = valid_dt = None
        try:
            valid_dt = _number(dt, "dt", positive=True)
            previous = _array(previous_applied, (2,), "previous_applied")
            reference = _array(reference, (2,), "reference")
            values = _weights(weights, self.weights)
            wt, we, wc = [values[name] for name in BASE_TERMS]
            limits = self.limits
            maximum = np.array([limits["speed_max"], limits["yaw_rate_max"]])
            acceleration = np.array([limits["accel_max"], limits["yaw_accel_max"]])
            scale = 1. / maximum
            wheel = np.array([[1., -limits["wheel_separation"] / 2.], [1., limits["wheel_separation"] / 2.]]) / limits["wheel_radius"]
            matrix = np.vstack([np.eye(2), -np.eye(2), np.eye(2), -np.eye(2), wheel, -wheel])
            bound = np.r_[maximum, maximum, previous + acceleration * valid_dt,
                          acceleration * valid_dt - previous, np.full(4, limits["wheel_rate_max"])]
            names = ["speed_v_upper", "speed_yaw_upper", "speed_v_lower", "speed_yaw_lower",
                     "accel_v_upper", "accel_yaw_upper", "accel_v_lower", "accel_yaw_lower",
                     "wheel_left_upper", "wheel_right_upper", "wheel_left_lower", "wheel_right_lower"]
            if lidar_points_axle is not None:
                raw = np.asarray(lidar_points_axle)
                if raw.ndim != 2 or raw.shape[1] != 2:
                    raise ValueError("lidar_points_axle must have finite shape Nx2")
                points = _array(lidar_points_axle, raw.shape, "lidar_points_axle")
                yaw = _number(heading, "heading")
                circle = _number(radius, "radius", positive=True) + _number(margin, "margin", nonnegative=True)
                alpha = _number(barrier_alpha, "barrier_alpha", positive=True)
                distances = np.linalg.norm(points, axis=1)
                if np.any(distances <= 1e-12) or not np.isfinite(distances).all():
                    raise ValueError("Obstacle normal undefined or nonfinite")
                rows = np.c_[(points / distances[:, None]) @ np.array([np.cos(yaw), np.sin(yaw)]), np.zeros(len(points))]
                matrix = np.vstack([matrix, rows])
                bound = np.r_[bound, alpha * (distances - circle)]
                names += [f"lidar_clearance_{i}" for i in range(len(points))]
            variable = cp.Variable(2)
            objective = wt * cp.sum_squares(cp.multiply(scale, variable - reference)) + we * cp.sum_squares(cp.multiply(scale, variable)) + wc * cp.sum_squares(cp.multiply(scale, variable - previous))
            diagonal = scale**2
            hessian = 2. * (wt + we + wc) * np.diag(diagonal)
            linear = -2. * diagonal * (wt * reference + wc * previous)
            result = _solve(variable, objective, hessian, linear, matrix, bound, names, previous=previous,
                            acceleration=acceleration, dt=valid_dt, started=started, solver_options=self.solver_options)
            result.update(weights=values, limits=limits.copy(), normalization_scales=maximum.tolist(),
                          lidar_point_count=0 if lidar_points_axle is None else len(lidar_points_axle))
            return result
        except (ValueError, TypeError, OverflowError, FloatingPointError) as error:
            return _rejected(2, "invalid_input", str(error), started, previous=previous,
                             acceleration=acceleration, dt=valid_dt, matrix=matrix, bound=bound)


class ArmVelocityQP:
    def __init__(self, joint_lower, joint_upper, *, joint_speed_max=.3, joint_accel_max=None,
                 cartesian_velocity_scales=(.1, .1, .1, .3, .3, .3), weights=None, solver_options=None,
                 joint_position_gain=None):
        self.lower = _array(joint_lower, (7,), "joint_lower")
        self.upper = _array(joint_upper, (7,), "joint_upper")
        if np.any(self.lower >= self.upper):
            raise ValueError("Every joint lower bound must be below its upper bound")
        self.speed = _positive_vector(joint_speed_max, "joint_speed_max")
        self.acceleration = None if joint_accel_max is None else _positive_vector(joint_accel_max, "joint_accel_max")
        self.position_gain = None if joint_position_gain is None else _number(joint_position_gain,'joint_position_gain',positive=True)
        self.task_scales = _array(cartesian_velocity_scales, (6,), "cartesian_velocity_scales")
        if np.any(self.task_scales <= 0):
            raise ValueError("Cartesian velocity scales must be positive")
        self.weights = _weights(weights, DEFAULT_ARM_WEIGHTS)
        self.solver_options = DEFAULT_SOLVER_OPTIONS | dict(solver_options or {})

    def solve(self, q, jacobian, desired_cartesian_velocity, previous_qdot, dt, *, weights=None,
              reference_q=None):
        """Frozen-Jacobian differential IK with Euler joint-box constraints.

        q+dt*qdot lies inside the joint box; this is a kinematic one-step bound,
        not a prediction of the torque-driven physical joint state. Integrating
        qdot into a separate position-servo reference requires independent
        reference bounds, tracking-error guards and contact anti-windup.
        If reference_q is provided, reference_q+dt*qdot is additionally bounded
        by the same joint box, so accumulated servo-reference integration is
        feasible as well. This does not bound the reference-versus-actual gap;
        the executor must separately reject excessive tracking error.
        No joint torque, self-collision, force or grasp constraint is added here.
        """
        started = time.perf_counter()
        previous = matrix = bound = valid_dt = None
        try:
            valid_dt = _number(dt, "dt", positive=True)
            previous = _array(previous_qdot, (7,), "previous_qdot")
            q = _array(q, (7,), "q")
            jacobian = _array(jacobian, (6, 7), "jacobian")
            desired = _array(desired_cartesian_velocity, (6,), "desired_cartesian_velocity")
            values = _weights(weights, self.weights)
            wt, we, wc = [values[name] for name in ARM_TERMS]
            matrix = np.vstack([np.eye(7), -np.eye(7), valid_dt * np.eye(7), -valid_dt * np.eye(7)])
            bound = np.r_[self.speed, self.speed, self.upper - q, q - self.lower]
            names = [f"{group}_{i}" for group in ("speed_upper", "speed_lower", "joint_next_upper", "joint_next_lower") for i in range(7)]
            if reference_q is not None:
                reference = _array(reference_q, (7,), "reference_q")
                matrix = np.vstack([matrix, valid_dt * np.eye(7), -valid_dt * np.eye(7)])
                bound = np.r_[bound, self.upper - reference, reference - self.lower]
                names += [f"{group}_{i}" for group in ("reference_next_upper", "reference_next_lower") for i in range(7)]
            if self.acceleration is not None:
                matrix = np.vstack([matrix, np.eye(7), -np.eye(7)])
                bound = np.r_[bound, previous + self.acceleration * valid_dt, self.acceleration * valid_dt - previous]
                names += [f"{group}_{i}" for group in ("accel_upper", "accel_lower") for i in range(7)]
            if self.position_gain is not None:
                # Frozen-state linear velocity damper starts braking before
                # the one-step joint box is reached. This does not establish
                # recursive feasibility under tracking error/contact dynamics.
                upper_distance=self.upper-q;lower_distance=q-self.lower
                if reference_q is not None:
                    upper_distance=np.minimum(upper_distance,self.upper-reference)
                    lower_distance=np.minimum(lower_distance,reference-self.lower)
                matrix=np.vstack([matrix,np.eye(7),-np.eye(7)])
                bound=np.r_[bound,self.position_gain*upper_distance,self.position_gain*lower_distance]
                names += [f'{group}_{i}' for group in ('joint_damper_upper','joint_damper_lower') for i in range(7)]
            scale = 1. / self.speed
            task_jacobian, task_desired = jacobian / self.task_scales[:, None], desired / self.task_scales
            variable = cp.Variable(7)
            objective = wt * cp.sum_squares(task_jacobian @ variable - task_desired) + we * cp.sum_squares(cp.multiply(scale, variable)) + wc * cp.sum_squares(cp.multiply(scale, variable - previous))
            hessian = 2. * (wt * task_jacobian.T @ task_jacobian + (we + wc) * np.diag(scale**2))
            linear = -2. * (wt * task_jacobian.T @ task_desired + wc * scale**2 * previous)
            result = _solve(variable, objective, hessian, linear, matrix, bound, names, previous=previous,
                            acceleration=self.acceleration, dt=valid_dt, started=started, solver_options=self.solver_options)
            result.update(weights=values, joint_speed_max=self.speed.tolist(),
                          joint_accel_max=None if self.acceleration is None else self.acceleration.tolist(),
                          cartesian_velocity_scales=self.task_scales.tolist(),
                          reference_integration_bounds_applied=reference_q is not None)
            result['joint_position_gain']=self.position_gain
            return result
        except (ValueError, TypeError, OverflowError, FloatingPointError) as error:
            return _rejected(7, "invalid_input", str(error), started, previous=previous,
                             acceleration=self.acceleration, dt=valid_dt, matrix=matrix, bound=bound)
