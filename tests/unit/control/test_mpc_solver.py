"""Focused numerical contracts for the fixed-structure Frenet MPC solver."""

from __future__ import annotations

import importlib
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

from controller_learning.config import load_project_config
from controller_learning.control import (
    build_public_controller_config,
    load_controller,
    load_controller_config,
)

PROJECT_ROOT = Path(__file__).parents[3]
MPC_DIRECTORY = PROJECT_ROOT / "controllers" / "mpc"


def _mpc_modules():
    controller_class = load_controller(MPC_DIRECTORY)
    package = controller_class.__module__.rsplit(".", 1)[0]
    return (
        importlib.import_module(f"{package}.helpers"),
        importlib.import_module(f"{package}.solver"),
    )


def _solver():
    helpers, solver_module = _mpc_modules()
    project = load_project_config(PROJECT_ROOT)
    public = build_public_controller_config(
        project,
        0,
        load_controller_config(MPC_DIRECTORY),
    )
    parameters = helpers.MpcControllerConfig.from_public_config(public)
    vehicle = project.vehicle
    limits = solver_module.MpcLimits(
        wheelbase_m=vehicle.vehicle.wheelbase_m,
        maximum_speed_mps=vehicle.vehicle.max_speed_mps,
        maximum_steering_rad=vehicle.actuator.max_steering_angle_rad,
        maximum_steering_rate_rad_s=vehicle.actuator.max_steering_rate_rad_s,
        maximum_acceleration_mps2=vehicle.actuator.max_acceleration_mps2,
        maximum_deceleration_mps2=vehicle.actuator.max_deceleration_mps2,
    )
    solver = solver_module.FrenetMpcSolver(
        steps=parameters.horizon.steps,
        dt_s=vehicle.simulation.control_dt_s,
        limits=limits,
        weights=parameters.weights,
        options=replace(parameters.solver, maximum_wall_time_s=1.0),
        feedback=parameters.feedback,
    )
    return solver_module, solver


def _request(
    solver_module,
    *,
    lateral_error_m: float = 0.1,
    heading_error_rad: float = 0.04,
    speed_mps: float = 3.0,
    curvature_1pm: float = 0.0,
    target_speed_mps: float = 4.0,
    previous_steering_rad: float = 0.0,
):
    steps = 20
    return solver_module.MpcRequest(
        initial_state=np.asarray((lateral_error_m, heading_error_rad, speed_mps), dtype=np.float64),
        curvature_1pm=np.full(steps, curvature_1pm, dtype=np.float64),
        target_speed_mps=np.full(steps + 1, target_speed_mps, dtype=np.float64),
        effective_half_width_m=np.full(steps + 1, 2.35, dtype=np.float64),
        previous_action=np.asarray((previous_steering_rad, 0.0), dtype=np.float64),
    )


class _SolverOutcome:
    def __init__(self, decision, constraint, *, status: str, success: bool) -> None:
        self._decision = np.array(decision, copy=True)
        self._constraint = np.array(constraint, copy=True)
        self._status = status
        self._success = success

    def __call__(self, **_kwargs):
        return {"x": self._decision, "g": self._constraint}

    def stats(self):
        return {"return_status": self._status, "success": self._success}


def _feasible_raw_solution(solver, request):
    states, controls = solver._cold_start(request)
    return solver._solver(
        x0=solver._flatten_guess(states, controls),
        p=solver._parameters(request),
        lbx=solver._lower_variables,
        ubx=solver._upper_variables,
        lbg=solver._lower_constraints,
        ubg=solver._upper_constraints,
    )


def test_primal_warm_shift_moves_both_horizons_and_replaces_initial_state() -> None:
    solver_module, _ = _solver()
    states = np.arange(3 * 21, dtype=np.float64).reshape((3, 21))
    controls = np.arange(20 * 2, dtype=np.float64).reshape((20, 2))
    initial = np.asarray((0.25, -0.1, 3.5), dtype=np.float64)

    shifted_states, shifted_controls = solver_module.shift_primal_warm_start(
        states, controls, initial
    )

    np.testing.assert_array_equal(shifted_states[:, 0], initial)
    np.testing.assert_array_equal(shifted_states[:, 1:-1], states[:, 2:])
    np.testing.assert_array_equal(shifted_states[:, -1], states[:, -1])
    np.testing.assert_array_equal(shifted_controls[:-1], controls[1:])
    np.testing.assert_array_equal(shifted_controls[-1], controls[-1])


def test_tiny_straight_problem_returns_one_finite_feasible_plan() -> None:
    solver_module, solver = _solver()

    result = solver.solve(_request(solver_module))

    assert result.success is True, result.status
    assert result.feasible is True
    assert result.timed_out is False
    assert result.maximum_violation <= solver.options.feasibility_tolerance
    assert result.states is not None and result.states.shape == (3, 21)
    assert result.controls is not None and result.controls.shape == (20, 2)
    assert np.isfinite(result.states).all()
    assert np.isfinite(result.controls).all()


def test_bounded_iteration_primal_is_accepted_only_after_feasibility_check() -> None:
    solver_module, solver = _solver()

    result = solver.solve(_request(solver_module))

    assert result.status == "Maximum_Iterations_Exceeded"
    assert result.success is True
    assert result.feasible is True
    assert result.timed_out is False
    assert result.maximum_violation <= solver.options.feasibility_tolerance


def test_iteration_limited_primal_is_rejected_when_a_hard_bound_is_violated() -> None:
    solver_module, solver = _solver()
    request = _request(solver_module)
    raw = _feasible_raw_solution(solver, request)
    decision = np.asarray(raw["x"], dtype=np.float64).reshape(-1)
    decision[2] = -2.0 * solver.options.feasibility_tolerance
    solver._solver = _SolverOutcome(
        decision,
        raw["g"],
        status="Maximum_Iterations_Exceeded",
        success=False,
    )

    result = solver.solve(request)

    assert result.status == "Maximum_Iterations_Exceeded"
    assert result.success is False
    assert result.feasible is False
    assert result.maximum_violation > solver.options.feasibility_tolerance


def test_feasible_wall_time_primal_is_not_accepted_for_the_current_action() -> None:
    solver_module, solver = _solver()
    request = _request(solver_module)
    raw = _feasible_raw_solution(solver, request)
    solver._solver = _SolverOutcome(
        raw["x"],
        raw["g"],
        status="Maximum_WallTime_Exceeded",
        success=False,
    )

    result = solver.solve(request)

    assert result.feasible is True
    assert result.timed_out is True
    assert result.success is False


def test_constant_curve_solution_respects_track_action_speed_and_rate_constraints() -> None:
    solver_module, solver = _solver()
    previous_steering = 0.2
    request = _request(
        solver_module,
        lateral_error_m=0.0,
        heading_error_rad=0.0,
        speed_mps=4.0,
        curvature_1pm=0.02,
        target_speed_mps=4.0,
        previous_steering_rad=previous_steering,
    )

    result = solver.solve(request)

    assert result.success is True, result.status
    assert result.states is not None and result.controls is not None
    states = result.states
    controls = result.controls
    assert np.all(np.abs(states[0, 1:]) <= request.effective_half_width_m[1:] + 1.0e-6)
    assert np.all((states[2] >= -1.0e-8) & (states[2] <= solver.limits.maximum_speed_mps))
    assert np.all(np.abs(controls[:, 0]) <= solver.limits.maximum_steering_rad + 1.0e-8)
    assert np.all(
        (controls[:, 1] >= -solver.limits.maximum_deceleration_mps2 - 1.0e-8)
        & (controls[:, 1] <= solver.limits.maximum_acceleration_mps2 + 1.0e-8)
    )
    steering = np.concatenate(((previous_steering,), controls[:, 0]))
    maximum_step = solver.limits.maximum_steering_rate_rad_s * solver.dt_s
    assert np.all(np.abs(np.diff(steering)) <= maximum_step + 1.0e-6)
    assert controls[0, 0] > 0.0


def test_second_solve_uses_shifted_primal_without_rebuilding_the_graph() -> None:
    solver_module, solver = _solver()
    first = solver.solve(_request(solver_module))
    assert first.success is True

    second = solver.solve(
        _request(
            solver_module,
            lateral_error_m=0.08,
            heading_error_rad=0.02,
            speed_mps=3.1,
        )
    )

    assert second.success is True, second.status
    assert second.used_warm_start is True
    assert solver.build_count == 1


def test_warm_start_is_rerolled_from_the_new_measured_state() -> None:
    solver_module, solver = _solver()
    first = solver.solve(_request(solver_module))
    assert first.success is True
    next_request = _request(
        solver_module,
        lateral_error_m=-0.12,
        heading_error_rad=0.06,
        speed_mps=solver.limits.maximum_speed_mps,
        curvature_1pm=0.015,
        target_speed_mps=solver.limits.maximum_speed_mps,
        previous_steering_rad=0.04,
    )

    states, controls, used_warm_start = solver._initial_guess(next_request)

    assert used_warm_start is True
    np.testing.assert_array_equal(states[:, 0], next_request.initial_state)
    assert np.all(states[2] >= 0.0)
    assert np.all(states[2] <= solver.limits.maximum_speed_mps)
    for index in range(solver.steps):
        expected = solver._integrate_numeric(
            states[:, index],
            controls[index],
            float(next_request.curvature_1pm[index]),
        )
        np.testing.assert_allclose(states[:, index + 1], expected, atol=1.0e-12, rtol=0.0)


def test_request_rejects_shapes_or_targets_outside_public_speed_bounds() -> None:
    solver_module, solver = _solver()
    bad_shape = solver_module.MpcRequest(
        initial_state=np.zeros(3),
        curvature_1pm=np.zeros(19),
        target_speed_mps=np.ones(21),
        effective_half_width_m=np.ones(21),
        previous_action=np.zeros(2),
    )
    with pytest.raises(ValueError, match="curvature_1pm must have shape"):
        solver.solve(bad_shape)

    too_fast = _request(solver_module, target_speed_mps=solver.limits.maximum_speed_mps + 1.0)
    with pytest.raises(ValueError, match="target speeds"):
        solver.solve(too_fast)
