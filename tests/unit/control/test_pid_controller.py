"""Tests for the observation-only educational PID Controller plugin."""

from __future__ import annotations

import importlib
import inspect
from pathlib import Path

import numpy as np
import pytest

from controller_learning.config import load_project_config
from controller_learning.control import (
    Controller,
    build_public_controller_config,
    curvature_speed_profile,
    load_controller,
    load_controller_config,
)
from controller_learning.control.debug_draw import PointsCommand, TextCommand, _DebugDrawBuffer

PROJECT_ROOT = Path(__file__).parents[3]
PID_DIRECTORY = PROJECT_ROOT / "controllers" / "pid"


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
    length = np.linalg.norm(np.diff(center[: segment_count + 1].astype(np.float64), axis=0), axis=1)
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
        "track_length": np.asarray(np.sum(length), dtype=np.float32),
    }


def _controller_inputs(level_id: int = 0):
    project = load_project_config(PROJECT_ROOT)
    plugin_config = load_controller_config(PID_DIRECTORY)
    public_config = build_public_controller_config(project, level_id, plugin_config)
    controller_class = load_controller(PID_DIRECTORY)
    return project, public_config, controller_class


def _controller(obs: dict[str, np.ndarray]):
    _, public_config, controller_class = _controller_inputs()
    return controller_class(obs, {"controller_seed": 1}, public_config)


def test_pid_directory_loads_one_controller_and_strict_config() -> None:
    project, public_config, controller_class = _controller_inputs()

    assert issubclass(controller_class, Controller)
    assert controller_class.__name__ == "PidController"
    assert public_config["controller"]["name"] == "pid"
    assert public_config["control_dt_s"] == project.vehicle.simulation.control_dt_s


def test_pid_action_is_finite_float32_bounded_and_rate_limited() -> None:
    observation = _circle_observation(steering_angle_rad=0.2)
    project, public_config, controller_class = _controller_inputs()
    controller = controller_class(observation, {"controller_seed": 1}, public_config)

    action = controller.compute_control(observation)

    assert action.shape == (2,)
    assert action.dtype == np.float32
    assert np.isfinite(action).all()
    actuator = project.vehicle.actuator
    assert -actuator.max_steering_angle_rad <= action[0] <= actuator.max_steering_angle_rad
    assert -actuator.max_deceleration_mps2 <= action[1] <= actuator.max_acceleration_mps2
    steering_step = actuator.max_steering_rate_rad_s * project.vehicle.simulation.control_dt_s
    assert 0.2 - steering_step <= action[0] <= 0.2 + steering_step


def test_lateral_and_heading_feedback_have_the_expected_sign() -> None:
    left_of_path = _circle_observation(position_m=(0.0, 1.0))
    left_action = _controller(left_of_path).compute_control(left_of_path)
    assert left_action[0] < 0.0

    heading_left = _circle_observation(yaw_rad=0.2)
    heading_action = _controller(heading_left).compute_control(heading_left)
    assert heading_action[0] < 0.0


def test_acceleration_loop_drives_from_rest_and_reduces_above_target() -> None:
    stopped = _circle_observation(forward_speed_mps=0.0)
    stopped_action = _controller(stopped).compute_control(stopped)
    assert stopped_action[1] > 0.0

    fast = _circle_observation(forward_speed_mps=8.0)
    fast_action = _controller(fast).compute_control(fast)
    assert fast_action[1] < 0.0


def test_pid_instances_do_not_share_integrators_or_projection_hints() -> None:
    observation = _circle_observation(position_m=(0.0, 0.8), forward_speed_mps=2.0)
    first = _controller(observation)
    for _ in range(20):
        first.compute_control(observation)

    second = _controller(observation)
    third = _controller(observation)
    np.testing.assert_array_equal(
        second.compute_control(observation),
        third.compute_control(observation),
    )


def test_pid_debug_draw_contains_only_derived_reference_commands() -> None:
    observation = _circle_observation(forward_speed_mps=3.0)
    controller = _controller(observation)
    controller.compute_control(observation)
    buffer = _DebugDrawBuffer()

    controller.render_callback(buffer.writer)

    commands = buffer.snapshot()
    assert len(commands) == 4
    assert sum(isinstance(command, PointsCommand) for command in commands) == 2
    assert sum(isinstance(command, TextCommand) for command in commands) == 1


def test_pid_helpers_apply_conditional_anti_windup_and_curvature_braking() -> None:
    _, public_config, controller_class = _controller_inputs()
    package_name = controller_class.__module__.rsplit(".", 1)[0]
    helpers = importlib.import_module(f"{package_name}.helpers")

    pid = helpers.SaturatingPid(kp=10.0, ki=1.0, kd=0.0, integral_limit=2.0)
    for _ in range(50):
        assert pid.step(
            error=1.0,
            error_derivative=0.0,
            dt_s=0.05,
            lower=-0.5,
            upper=0.5,
        ) == pytest.approx(0.5)
    assert pid.integral == 0.0

    parameters = helpers.PidControllerConfig.from_public_config(public_config).longitudinal
    straight = curvature_speed_profile(
        (0.0, 0.0),
        (0.0, 1.0),
        minimum_speed_mps=parameters.minimum_corner_speed_mps,
        maximum_speed_mps=parameters.cruise_speed_mps,
        maximum_lateral_acceleration_mps2=parameters.maximum_lateral_acceleration_mps2,
        braking_deceleration_mps2=parameters.braking_deceleration_mps2,
    )
    approaching_curve = curvature_speed_profile(
        (0.0, 0.2),
        (0.0, 1.0),
        minimum_speed_mps=parameters.minimum_corner_speed_mps,
        maximum_speed_mps=parameters.cruise_speed_mps,
        maximum_lateral_acceleration_mps2=parameters.maximum_lateral_acceleration_mps2,
        braking_deceleration_mps2=parameters.braking_deceleration_mps2,
    )
    assert straight[0] == pytest.approx(parameters.cruise_speed_mps)
    assert parameters.minimum_corner_speed_mps < approaching_curve[0] < straight[0]


def test_pid_source_does_not_import_challenge_or_simulator_internals() -> None:
    controller_class = load_controller(PID_DIRECTORY)
    source = inspect.getsource(importlib.import_module(controller_class.__module__))
    helpers_source = inspect.getsource(
        importlib.import_module(f"{controller_class.__module__.rsplit('.', 1)[0]}.helpers")
    )
    combined = source + helpers_source

    for forbidden in (
        "controller_learning.envs",
        "race_core",
        "TrackBatch",
        "controller_learning.physics",
        "import jax",
        "import mujoco",
        "import warp",
    ):
        assert forbidden not in combined
