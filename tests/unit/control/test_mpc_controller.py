"""Public-boundary and fallback tests for the educational MPC Controller plugin."""

from __future__ import annotations

import importlib
import inspect
import tomllib
from pathlib import Path
from typing import ClassVar

import numpy as np
import pytest

from controller_learning.config import load_project_config
from controller_learning.control import (
    Controller,
    build_public_controller_config,
    load_controller,
    load_controller_config,
)
from controller_learning.control.debug_draw import PointsCommand, TextCommand, _DebugDrawBuffer

PROJECT_ROOT = Path(__file__).parents[3]
MPC_DIRECTORY = PROJECT_ROOT / "controllers" / "mpc"


def _circle_observation(
    *,
    radius_m: float = 50.0,
    position_m: tuple[float, float] = (0.0, 0.0),
    yaw_rad: float = 0.0,
    forward_speed_mps: float = 0.0,
    steering_angle_rad: float = 0.0,
) -> dict[str, np.ndarray]:
    capacity = 640
    segment_count = 320
    angle = np.linspace(-0.5 * np.pi, 1.5 * np.pi, segment_count + 1)
    radial = np.stack((np.cos(angle), np.sin(angle)), axis=1)
    valid_center = radial * radius_m + np.asarray((0.0, radius_m))
    valid_center[-1] = valid_center[0]
    valid_left = np.stack(
        (
            radial[:, 0] * (radius_m - 3.5),
            radial[:, 1] * (radius_m - 3.5) + radius_m,
        ),
        axis=1,
    )
    valid_right = np.stack(
        (
            radial[:, 0] * (radius_m + 3.5),
            radial[:, 1] * (radius_m + 3.5) + radius_m,
        ),
        axis=1,
    )
    valid_left[-1] = valid_left[0]
    valid_right[-1] = valid_right[0]

    center = np.zeros((capacity, 2), dtype=np.float32)
    left = np.zeros_like(center)
    right = np.zeros_like(center)
    mask = np.zeros(capacity, dtype=np.int8)
    center[: segment_count + 1] = valid_center
    left[: segment_count + 1] = valid_left
    right[: segment_count + 1] = valid_right
    mask[: segment_count + 1] = 1
    lengths = np.linalg.norm(
        np.diff(center[: segment_count + 1].astype(np.float64), axis=0), axis=1
    )
    return {
        "position": np.asarray(position_m, dtype=np.float32),
        "yaw": np.asarray(yaw_rad, dtype=np.float32),
        "velocity_body": np.asarray((forward_speed_mps, 0.0), dtype=np.float32),
        "yaw_rate": np.asarray(0.0, dtype=np.float32),
        "steering_angle": np.asarray(steering_angle_rad, dtype=np.float32),
        "track_progress": np.asarray(0.0, dtype=np.float32),
        "centerline": center,
        "left_boundary": left,
        "right_boundary": right,
        "track_mask": mask,
        "track_length": np.asarray(np.sum(lengths), dtype=np.float32),
    }


def _controller_components():
    project = load_project_config(PROJECT_ROOT)
    plugin_config = load_controller_config(MPC_DIRECTORY)
    public = build_public_controller_config(project, 0, plugin_config)
    controller_class = load_controller(MPC_DIRECTORY)
    package = controller_class.__module__.rsplit(".", 1)[0]
    controller_module = importlib.import_module(controller_class.__module__)
    helpers = importlib.import_module(f"{package}.helpers")
    solver = importlib.import_module(f"{package}.solver")
    return project, public, controller_class, controller_module, helpers, solver


def _successful_result(solver_module, *, first: tuple[float, float] = (0.05, 1.0)):
    states = np.zeros((3, 21), dtype=np.float64)
    controls = np.zeros((20, 2), dtype=np.float64)
    controls[:, 0] = np.linspace(first[0], 0.12, 20)
    controls[:, 1] = first[1]
    return solver_module.MpcSolveResult(
        success=True,
        feasible=True,
        timed_out=False,
        status="Solve_Succeeded",
        used_warm_start=False,
        maximum_violation=0.0,
        states=states,
        controls=controls,
    )


def _failed_result(solver_module, *, timed_out: bool = True):
    return solver_module.MpcSolveResult(
        success=False,
        feasible=False,
        timed_out=timed_out,
        status="Maximum_WallTime_Exceeded" if timed_out else "Infeasible_Problem_Detected",
        used_warm_start=True,
        maximum_violation=np.inf,
        states=None,
        controls=None,
    )


class _FakeSolver:
    outcomes: ClassVar[list[object]] = []

    def __init__(self, **kwargs) -> None:
        del kwargs
        self.build_count = 1
        self._outcomes = list(type(self).outcomes)

    def solve(self, request):
        del request
        return self._outcomes.pop(0)


def test_mpc_directory_loads_one_controller_and_strict_configuration() -> None:
    project, public, controller_class, _, helpers, _ = _controller_components()

    assert issubclass(controller_class, Controller)
    assert controller_class.__name__ == "MpcController"
    parsed = helpers.MpcControllerConfig.from_public_config(public)
    assert parsed.horizon.steps == 20
    assert public["control_dt_s"] == project.vehicle.simulation.control_dt_s == 0.05

    raw = tomllib.loads((MPC_DIRECTORY / "config.toml").read_text(encoding="utf-8"))
    raw["unexpected"] = True
    with pytest.raises(helpers.MpcConfigurationError, match=r"extra=.+unexpected"):
        helpers.MpcControllerConfig.from_public_config({"controller": raw})


def test_real_controller_returns_finite_float32_bounded_action_without_rebuilding() -> None:
    observation = _circle_observation(steering_angle_rad=0.2)
    project, public, controller_class, _, _, _ = _controller_components()
    controller = controller_class(observation, {"controller_seed": 1}, public)

    first = controller.compute_control(observation)
    second = controller.compute_control(observation)

    actuator = project.vehicle.actuator
    steering_step = actuator.max_steering_rate_rad_s * project.vehicle.simulation.control_dt_s
    for action in (first, second):
        assert action.shape == (2,)
        assert action.dtype == np.float32
        assert np.isfinite(action).all()
        assert 0.2 - steering_step <= action[0] <= 0.2 + steering_step
        assert -actuator.max_deceleration_mps2 <= action[1] <= actuator.max_acceleration_mps2
    assert controller._solver.build_count == 1


def test_timeout_consumes_last_feasible_shifted_plan_before_feedback(monkeypatch) -> None:
    observation = _circle_observation()
    _, public, controller_class, controller_module, _, solver_module = _controller_components()
    _FakeSolver.outcomes = [
        _successful_result(solver_module),
        _failed_result(solver_module),
    ]
    monkeypatch.setattr(controller_module, "FrenetMpcSolver", _FakeSolver)
    controller = controller_class(observation, {"controller_seed": 1}, public)

    first = controller.compute_control(observation)
    next_observation = dict(observation)
    next_observation["steering_angle"] = np.asarray(first[0], dtype=np.float32)
    second = controller.compute_control(next_observation)

    assert first[0] == pytest.approx(0.05)
    assert second[0] == pytest.approx(np.linspace(0.05, 0.12, 20)[1])
    assert controller._debug.mode == "shifted-plan"
    assert controller._debug.solver_status == "Maximum_WallTime_Exceeded"


def test_timeout_without_prior_plan_uses_deterministic_public_geometry_fallback(
    monkeypatch,
) -> None:
    observation = _circle_observation(position_m=(0.0, 1.0), forward_speed_mps=0.0)
    _, public, controller_class, controller_module, _, solver_module = _controller_components()
    _FakeSolver.outcomes = [_failed_result(solver_module), _failed_result(solver_module)]
    monkeypatch.setattr(controller_module, "FrenetMpcSolver", _FakeSolver)
    first_controller = controller_class(observation, {"controller_seed": 1}, public)
    second_controller = controller_class(observation, {"controller_seed": 1}, public)

    first = first_controller.compute_control(observation)
    second = second_controller.compute_control(observation)

    np.testing.assert_array_equal(first, second)
    assert first[0] < 0.0
    assert first[1] > 0.0
    assert first_controller._debug.mode == "feedback-fallback"


def test_debug_draw_contains_reference_prediction_and_solver_mode(monkeypatch) -> None:
    observation = _circle_observation(forward_speed_mps=3.0)
    _, public, controller_class, controller_module, _, solver_module = _controller_components()
    _FakeSolver.outcomes = [_successful_result(solver_module)]
    monkeypatch.setattr(controller_module, "FrenetMpcSolver", _FakeSolver)
    controller = controller_class(observation, {"controller_seed": 1}, public)
    controller.compute_control(observation)
    buffer = _DebugDrawBuffer()

    controller.render_callback(buffer.writer)

    commands = buffer.snapshot()
    assert len(commands) == 4
    assert sum(isinstance(command, PointsCommand) for command in commands) == 2
    text = next(command for command in commands if isinstance(command, TextCommand))
    assert "mpc (Solve_Succeeded)" in text.text


def test_mpc_source_does_not_import_challenge_or_simulator_internals() -> None:
    _, _, controller_class, _, _, _ = _controller_components()
    package = controller_class.__module__.rsplit(".", 1)[0]
    source = "".join(
        inspect.getsource(importlib.import_module(name))
        for name in (
            controller_class.__module__,
            f"{package}.helpers",
            f"{package}.solver",
        )
    )

    for forbidden in (
        "controller_learning.envs",
        "race_core",
        "TrackBatch",
        "TrackPool",
        "controller_learning.physics",
        "import jax",
        "import mujoco",
        "import warp",
    ):
        assert forbidden not in source
