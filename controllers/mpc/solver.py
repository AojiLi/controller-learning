"""Fixed-structure CasADi/IPOPT solver for the educational Frenet MPC."""

from __future__ import annotations

import math
from dataclasses import dataclass

import casadi as ca
import numpy as np
from numpy.typing import ArrayLike, NDArray

from .helpers import SolverConfig, WeightConfig

STATE_COUNT = 3
CONTROL_COUNT = 2


@dataclass(frozen=True, slots=True)
class MpcLimits:
    """Public vehicle and action bounds embedded in one episode's NLP."""

    wheelbase_m: float
    maximum_speed_mps: float
    maximum_steering_rad: float
    maximum_steering_rate_rad_s: float
    maximum_acceleration_mps2: float
    maximum_deceleration_mps2: float

    def __post_init__(self) -> None:
        values = np.asarray(
            (
                self.wheelbase_m,
                self.maximum_speed_mps,
                self.maximum_steering_rad,
                self.maximum_steering_rate_rad_s,
                self.maximum_acceleration_mps2,
                self.maximum_deceleration_mps2,
            ),
            dtype=np.float64,
        )
        if not np.isfinite(values).all() or np.any(values <= 0.0):
            raise ValueError("all MPC vehicle and action limits must be finite and positive")


@dataclass(frozen=True, slots=True)
class MpcRequest:
    """Numerical parameters for one solve of the fixed NLP graph."""

    initial_state: NDArray[np.float64]
    curvature_1pm: NDArray[np.float64]
    target_speed_mps: NDArray[np.float64]
    effective_half_width_m: NDArray[np.float64]
    previous_action: NDArray[np.float64]

    def __post_init__(self) -> None:
        for name in (
            "initial_state",
            "curvature_1pm",
            "target_speed_mps",
            "effective_half_width_m",
            "previous_action",
        ):
            value = np.array(getattr(self, name), dtype=np.float64, copy=True)
            if not np.isfinite(value).all():
                raise ValueError(f"{name} must contain only finite values")
            value.setflags(write=False)
            object.__setattr__(self, name, value)


@dataclass(frozen=True, slots=True)
class MpcSolveResult:
    """One solver outcome; only successful feasible arrays are retained for control."""

    success: bool
    feasible: bool
    timed_out: bool
    status: str
    used_warm_start: bool
    maximum_violation: float
    states: NDArray[np.float64] | None
    controls: NDArray[np.float64] | None

    def __post_init__(self) -> None:
        if not math.isfinite(self.maximum_violation) and self.maximum_violation != math.inf:
            raise ValueError("maximum_violation must be finite or positive infinity")
        for name in ("states", "controls"):
            source = getattr(self, name)
            if source is None:
                continue
            value = np.array(source, dtype=np.float64, copy=True)
            value.setflags(write=False)
            object.__setattr__(self, name, value)


def shift_primal_warm_start(
    states: ArrayLike,
    controls: ArrayLike,
    new_initial_state: ArrayLike,
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Shift a prior horizon left by one stage and replace its measured initial state."""

    state_values = np.asarray(states, dtype=np.float64)
    control_values = np.asarray(controls, dtype=np.float64)
    initial = np.asarray(new_initial_state, dtype=np.float64)
    if (
        state_values.ndim != 2
        or state_values.shape[0] != STATE_COUNT
        or state_values.shape[1] < 2
        or control_values.shape != (state_values.shape[1] - 1, CONTROL_COUNT)
        or initial.shape != (STATE_COUNT,)
    ):
        raise ValueError("warm-start states, controls, and initial state have incompatible shapes")
    if not (
        np.isfinite(state_values).all()
        and np.isfinite(control_values).all()
        and np.isfinite(initial).all()
    ):
        raise ValueError("warm-start values must contain only finite numbers")

    shifted_states = np.empty_like(state_values)
    shifted_states[:, :-1] = state_values[:, 1:]
    shifted_states[:, -1] = state_values[:, -1]
    shifted_states[:, 0] = initial
    shifted_controls = np.empty_like(control_values)
    shifted_controls[:-1] = control_values[1:]
    shifted_controls[-1] = control_values[-1]
    return shifted_states, shifted_controls


class FrenetMpcSolver:
    """One fixed CasADi graph and mutable per-episode primal warm start."""

    def __init__(
        self,
        *,
        steps: int,
        dt_s: float,
        limits: MpcLimits,
        weights: WeightConfig,
        options: SolverConfig,
    ) -> None:
        if isinstance(steps, bool) or not isinstance(steps, int) or steps < 1:
            raise ValueError("steps must be a positive integer")
        if not math.isfinite(dt_s) or dt_s <= 0.0:
            raise ValueError("dt_s must be finite and positive")
        self.steps = steps
        self.dt_s = float(dt_s)
        self.limits = limits
        self.weights = weights
        self.options = options
        self.build_count = 1
        self._last_states: NDArray[np.float64] | None = None
        self._last_controls: NDArray[np.float64] | None = None
        self._build_graph()

    def _build_graph(self) -> None:
        steps = self.steps
        states = ca.SX.sym("state", STATE_COUNT, steps + 1)
        controls = ca.SX.sym("control", CONTROL_COUNT, steps)
        parameter_count = STATE_COUNT + steps + (steps + 1) + (steps + 1) + CONTROL_COUNT
        parameters = ca.SX.sym("parameter", parameter_count)

        cursor = 0
        initial_state = parameters[cursor : cursor + STATE_COUNT]
        cursor += STATE_COUNT
        curvature = parameters[cursor : cursor + steps]
        cursor += steps
        target_speed = parameters[cursor : cursor + steps + 1]
        cursor += steps + 1
        effective_half_width = parameters[cursor : cursor + steps + 1]
        cursor += steps + 1
        previous_action = parameters[cursor : cursor + CONTROL_COUNT]

        def dynamics(state, control, reference_curvature):
            lateral_error = state[0]
            heading_error = state[1]
            speed = state[2]
            steering = control[0]
            denominator = 1.0 - reference_curvature * lateral_error
            path_rate = speed * ca.cos(heading_error) / denominator
            return ca.vertcat(
                speed * ca.sin(heading_error),
                speed / self.limits.wheelbase_m * ca.tan(steering)
                - reference_curvature * path_rate,
                control[1],
            )

        objective = 0
        constraints = [states[:, 0] - initial_state]
        lower_constraints: list[float] = [0.0] * STATE_COUNT
        upper_constraints: list[float] = [0.0] * STATE_COUNT
        for index in range(steps):
            state = states[:, index]
            control = controls[:, index]
            reference_curvature = curvature[index]
            first = dynamics(state, control, reference_curvature)
            second = dynamics(
                state + 0.5 * self.dt_s * first,
                control,
                reference_curvature,
            )
            third = dynamics(
                state + 0.5 * self.dt_s * second,
                control,
                reference_curvature,
            )
            fourth = dynamics(state + self.dt_s * third, control, reference_curvature)
            integrated = state + self.dt_s * (first + 2 * second + 2 * third + fourth) / 6.0
            constraints.append(states[:, index + 1] - integrated)
            lower_constraints.extend((0.0,) * STATE_COUNT)
            upper_constraints.extend((0.0,) * STATE_COUNT)

            prior_control = previous_action if index == 0 else controls[:, index - 1]
            control_change = control - prior_control
            steering_feedforward = ca.atan(self.limits.wheelbase_m * reference_curvature)
            objective += (
                self.weights.lateral_error * state[0] ** 2
                + self.weights.heading_error * state[1] ** 2
                + self.weights.speed_error * (state[2] - target_speed[index]) ** 2
                + self.weights.steering_feedforward * (control[0] - steering_feedforward) ** 2
                + self.weights.acceleration * control[1] ** 2
                + self.weights.steering_change * control_change[0] ** 2
                + self.weights.acceleration_change * control_change[1] ** 2
            )
            constraints.append(control_change[0])
            steering_step = self.limits.maximum_steering_rate_rad_s * self.dt_s
            lower_constraints.append(-steering_step)
            upper_constraints.append(steering_step)

        terminal = states[:, -1]
        objective += (
            self.weights.terminal_lateral_error * terminal[0] ** 2
            + self.weights.terminal_heading_error * terminal[1] ** 2
            + self.weights.terminal_speed_error * (terminal[2] - target_speed[-1]) ** 2
        )
        for index in range(1, steps + 1):
            constraints.extend(
                (
                    states[0, index] - effective_half_width[index],
                    -states[0, index] - effective_half_width[index],
                )
            )
            lower_constraints.extend((-math.inf, -math.inf))
            upper_constraints.extend((0.0, 0.0))

        decision = ca.vertcat(ca.vec(states), ca.vec(controls))
        constraint = ca.vertcat(*constraints)
        solver_options = {
            "print_time": False,
            "ipopt.print_level": 0,
            "ipopt.sb": "yes",
            "ipopt.option_file_name": "",
            "ipopt.max_iter": self.options.maximum_iterations,
            "ipopt.tol": self.options.tolerance,
            "ipopt.acceptable_tol": self.options.acceptable_tolerance,
            "ipopt.acceptable_iter": self.options.acceptable_iterations,
            "ipopt.max_wall_time": self.options.maximum_wall_time_s,
            "ipopt.bound_relax_factor": 0.0,
        }
        self._solver = ca.nlpsol(
            f"frenet_mpc_{id(self):x}",
            "ipopt",
            {"x": decision, "p": parameters, "f": objective, "g": constraint},
            solver_options,
        )

        variable_count = int(decision.shape[0])
        lower_variables = np.full(variable_count, -math.inf, dtype=np.float64)
        upper_variables = np.full(variable_count, math.inf, dtype=np.float64)
        for index in range(steps + 1):
            speed_index = STATE_COUNT * index + 2
            lower_variables[speed_index] = 0.0
            upper_variables[speed_index] = self.limits.maximum_speed_mps
        control_offset = STATE_COUNT * (steps + 1)
        for index in range(steps):
            steering_index = control_offset + CONTROL_COUNT * index
            acceleration_index = steering_index + 1
            lower_variables[steering_index] = -self.limits.maximum_steering_rad
            upper_variables[steering_index] = self.limits.maximum_steering_rad
            lower_variables[acceleration_index] = -self.limits.maximum_deceleration_mps2
            upper_variables[acceleration_index] = self.limits.maximum_acceleration_mps2

        self._lower_variables = lower_variables
        self._upper_variables = upper_variables
        self._lower_constraints = np.asarray(lower_constraints, dtype=np.float64)
        self._upper_constraints = np.asarray(upper_constraints, dtype=np.float64)
        self._parameter_count = parameter_count

    def _validate_request(self, request: MpcRequest) -> None:
        expected = {
            "initial_state": (STATE_COUNT,),
            "curvature_1pm": (self.steps,),
            "target_speed_mps": (self.steps + 1,),
            "effective_half_width_m": (self.steps + 1,),
            "previous_action": (CONTROL_COUNT,),
        }
        for name, shape in expected.items():
            if getattr(request, name).shape != shape:
                raise ValueError(f"{name} must have shape {shape}")
        if not 0.0 <= request.initial_state[2] <= self.limits.maximum_speed_mps:
            raise ValueError("initial Frenet speed must satisfy the public vehicle speed bounds")
        if np.any(request.target_speed_mps < 0.0) or np.any(
            request.target_speed_mps > self.limits.maximum_speed_mps
        ):
            raise ValueError("target speeds must satisfy the public vehicle speed bounds")
        if np.any(request.effective_half_width_m <= 0.0):
            raise ValueError("effective half-width values must be positive")

    def _parameters(self, request: MpcRequest) -> NDArray[np.float64]:
        values = np.concatenate(
            (
                request.initial_state,
                request.curvature_1pm,
                request.target_speed_mps,
                request.effective_half_width_m,
                request.previous_action,
            )
        )
        if values.shape != (self._parameter_count,):
            raise AssertionError("MPC parameter packing disagrees with the fixed graph")
        return values

    def _dynamics_numeric(
        self,
        state: NDArray[np.float64],
        control: NDArray[np.float64],
        curvature: float,
    ) -> NDArray[np.float64]:
        denominator = 1.0 - curvature * state[0]
        if abs(denominator) < 1.0e-6:
            denominator = math.copysign(1.0e-6, denominator if denominator else 1.0)
        path_rate = state[2] * math.cos(state[1]) / denominator
        return np.asarray(
            (
                state[2] * math.sin(state[1]),
                state[2] / self.limits.wheelbase_m * math.tan(control[0]) - curvature * path_rate,
                control[1],
            ),
            dtype=np.float64,
        )

    def _integrate_numeric(
        self,
        state: NDArray[np.float64],
        control: NDArray[np.float64],
        curvature: float,
    ) -> NDArray[np.float64]:
        first = self._dynamics_numeric(state, control, curvature)
        second = self._dynamics_numeric(state + 0.5 * self.dt_s * first, control, curvature)
        third = self._dynamics_numeric(state + 0.5 * self.dt_s * second, control, curvature)
        fourth = self._dynamics_numeric(state + self.dt_s * third, control, curvature)
        result = state + self.dt_s * (first + 2 * second + 2 * third + fourth) / 6.0
        result[2] = np.clip(result[2], 0.0, self.limits.maximum_speed_mps)
        return result

    def _cold_start(self, request: MpcRequest) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
        states = np.zeros((STATE_COUNT, self.steps + 1), dtype=np.float64)
        controls = np.zeros((self.steps, CONTROL_COUNT), dtype=np.float64)
        states[:, 0] = request.initial_state
        previous_steering = float(request.previous_action[0])
        steering_step = self.limits.maximum_steering_rate_rad_s * self.dt_s
        for index in range(self.steps):
            feedforward = math.atan(self.limits.wheelbase_m * request.curvature_1pm[index])
            steering = float(
                np.clip(
                    feedforward,
                    previous_steering - steering_step,
                    previous_steering + steering_step,
                )
            )
            steering = float(
                np.clip(
                    steering,
                    -self.limits.maximum_steering_rad,
                    self.limits.maximum_steering_rad,
                )
            )
            acceleration = float(
                np.clip(
                    request.target_speed_mps[index] - states[2, index],
                    -self.limits.maximum_deceleration_mps2,
                    self.limits.maximum_acceleration_mps2,
                )
            )
            controls[index] = (steering, acceleration)
            states[:, index + 1] = self._integrate_numeric(
                states[:, index], controls[index], float(request.curvature_1pm[index])
            )
            previous_steering = steering
        return states, controls

    def _initial_guess(
        self, request: MpcRequest
    ) -> tuple[NDArray[np.float64], NDArray[np.float64], bool]:
        if self._last_states is None or self._last_controls is None:
            states, controls = self._cold_start(request)
            return states, controls, False
        states, controls = shift_primal_warm_start(
            self._last_states,
            self._last_controls,
            request.initial_state,
        )
        steering_step = self.limits.maximum_steering_rate_rad_s * self.dt_s
        prior = float(request.previous_action[0])
        for index in range(self.steps):
            controls[index, 0] = np.clip(
                controls[index, 0],
                max(-self.limits.maximum_steering_rad, prior - steering_step),
                min(self.limits.maximum_steering_rad, prior + steering_step),
            )
            controls[index, 1] = np.clip(
                controls[index, 1],
                -self.limits.maximum_deceleration_mps2,
                self.limits.maximum_acceleration_mps2,
            )
            prior = float(controls[index, 0])
        return states, controls, True

    @staticmethod
    def _flatten_guess(
        states: NDArray[np.float64], controls: NDArray[np.float64]
    ) -> NDArray[np.float64]:
        return np.concatenate((states.reshape(-1, order="F"), controls.T.reshape(-1, order="F")))

    def _maximum_violation(
        self, decision: NDArray[np.float64], constraint: NDArray[np.float64]
    ) -> float:
        violations = [0.0]
        finite_lower_variables = np.isfinite(self._lower_variables)
        finite_upper_variables = np.isfinite(self._upper_variables)
        finite_lower_constraints = np.isfinite(self._lower_constraints)
        finite_upper_constraints = np.isfinite(self._upper_constraints)
        if np.any(finite_lower_variables):
            violations.append(
                float(
                    np.max(
                        self._lower_variables[finite_lower_variables]
                        - decision[finite_lower_variables]
                    )
                )
            )
        if np.any(finite_upper_variables):
            violations.append(
                float(
                    np.max(
                        decision[finite_upper_variables]
                        - self._upper_variables[finite_upper_variables]
                    )
                )
            )
        if np.any(finite_lower_constraints):
            violations.append(
                float(
                    np.max(
                        self._lower_constraints[finite_lower_constraints]
                        - constraint[finite_lower_constraints]
                    )
                )
            )
        if np.any(finite_upper_constraints):
            violations.append(
                float(
                    np.max(
                        constraint[finite_upper_constraints]
                        - self._upper_constraints[finite_upper_constraints]
                    )
                )
            )
        return max(0.0, *violations)

    def solve(self, request: MpcRequest) -> MpcSolveResult:
        """Solve once and retain only a successful independently feasible primal solution."""

        if not isinstance(request, MpcRequest):
            raise TypeError("request must be an MpcRequest")
        self._validate_request(request)
        initial_states, initial_controls, used_warm_start = self._initial_guess(request)
        initial_guess = self._flatten_guess(initial_states, initial_controls)
        try:
            solution = self._solver(
                x0=initial_guess,
                p=self._parameters(request),
                lbx=self._lower_variables,
                ubx=self._upper_variables,
                lbg=self._lower_constraints,
                ubg=self._upper_constraints,
            )
            statistics = self._solver.stats()
        except (RuntimeError, ValueError):
            return MpcSolveResult(
                success=False,
                feasible=False,
                timed_out=False,
                status="solver_exception",
                used_warm_start=used_warm_start,
                maximum_violation=math.inf,
                states=None,
                controls=None,
            )

        decision = np.asarray(solution["x"], dtype=np.float64).reshape(-1)
        constraint = np.asarray(solution["g"], dtype=np.float64).reshape(-1)
        finite = np.isfinite(decision).all() and np.isfinite(constraint).all()
        maximum_violation = self._maximum_violation(decision, constraint) if finite else math.inf
        feasible = bool(finite and maximum_violation <= self.options.feasibility_tolerance)
        status = str(statistics.get("return_status", "unknown"))
        timed_out = "WallTime" in status or "CpuTime" in status
        success = bool(statistics.get("success", False)) and feasible and not timed_out

        state_values: NDArray[np.float64] | None = None
        control_values: NDArray[np.float64] | None = None
        if finite:
            state_size = STATE_COUNT * (self.steps + 1)
            state_values = decision[:state_size].reshape((STATE_COUNT, self.steps + 1), order="F")
            control_values = decision[state_size:].reshape((CONTROL_COUNT, self.steps), order="F").T
        # A wall-time exit is never used as the current action, but an independently feasible
        # primal remains a better numerical starting point than rebuilding the same cold rollout.
        # Controller fallback policy stays separate and still treats the timed-out call as failed.
        if feasible and state_values is not None and control_values is not None:
            self._last_states = np.array(state_values, copy=True)
            self._last_controls = np.array(control_values, copy=True)

        return MpcSolveResult(
            success=success,
            feasible=feasible,
            timed_out=timed_out,
            status=status,
            used_warm_start=used_warm_start,
            maximum_violation=maximum_violation,
            states=state_values,
            controls=control_values,
        )


__all__ = [
    "CONTROL_COUNT",
    "STATE_COUNT",
    "FrenetMpcSolver",
    "MpcLimits",
    "MpcRequest",
    "MpcSolveResult",
    "shift_primal_warm_start",
]
