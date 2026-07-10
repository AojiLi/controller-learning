"""Tests for the narrow CPU/MJX environment vehicle-driver boundary."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from controller_learning.config import load_project_config
from controller_learning.envs._vehicle_driver import (
    VehicleDriverError,
    VehicleDriverShapeError,
    create_vehicle_driver,
)

PROJECT_ROOT = Path(__file__).parents[3]


@pytest.fixture
def project_config():
    return load_project_config(PROJECT_ROOT)


def test_cpu_reference_is_explicit_and_batch_one_only(project_config) -> None:
    driver = create_vehicle_driver(
        "cpu_reference",
        project_config.vehicle,
        num_worlds=1,
    )
    assert driver.backend == "cpu_reference"
    assert driver.num_worlds == 1

    with pytest.raises(VehicleDriverError, match="num_worlds=1"):
        create_vehicle_driver("cpu_reference", project_config.vehicle, num_worlds=2)
    with pytest.raises(VehicleDriverError, match="unsupported"):
        create_vehicle_driver("unknown", project_config.vehicle, num_worlds=1)  # type: ignore[arg-type]


def test_cpu_initial_state_uses_requested_rear_axle_pose(project_config) -> None:
    driver = create_vehicle_driver("cpu_reference", project_config.vehicle, num_worlds=1)
    pose = np.asarray(((4.0, -2.0, 0.5),), dtype=np.float32)
    state = driver.initial_state(pose)
    view = driver.read_state(state)

    np.testing.assert_allclose(view.position_world_m[0, :2], pose[0, :2], atol=1e-6)
    assert view.yaw_rad[0] == pytest.approx(pose[0, 2], abs=1e-6)
    np.testing.assert_array_equal(state.control_step_count, (0,))


def test_cpu_step_matches_action_clipping_and_invalid_contract(project_config) -> None:
    driver = create_vehicle_driver("cpu_reference", project_config.vehicle, num_worlds=1)
    pose = np.zeros((1, 3), dtype=np.float32)
    state = driver.initial_state(pose)

    clipped = driver.step(state, ((10.0, 10.0),))
    assert clipped.applied.steering_angle_rad[0] == pytest.approx(0.6)
    assert clipped.applied.longitudinal_acceleration_mps2[0] == pytest.approx(4.0)
    assert clipped.applied.saturation_count[0] == 2
    assert not clipped.applied.invalid_action[0]

    invalid = driver.step(clipped.state, ((np.nan, 1.0),))
    assert invalid.applied.invalid_action[0]
    assert invalid.applied.steering_angle_rad[0] == 0.0
    assert invalid.applied.longitudinal_acceleration_mps2[0] == 0.0
    assert invalid.state.control_step_count[0] == 2


def test_cpu_masked_reset_preserves_or_resets_the_only_world(project_config) -> None:
    driver = create_vehicle_driver("cpu_reference", project_config.vehicle, num_worlds=1)
    origin = np.zeros((1, 3), dtype=np.float32)
    state = driver.initial_state(origin)
    moved = driver.step(state, ((0.0, 2.0),)).state
    moved_view = driver.read_state(moved)

    preserved = driver.masked_reset(moved, (False,), ((8.0, 3.0, 1.0),))
    np.testing.assert_array_equal(preserved.control_step_count, moved.control_step_count)
    np.testing.assert_array_equal(
        driver.read_state(preserved).position_world_m,
        moved_view.position_world_m,
    )

    reset = driver.masked_reset(moved, (True,), ((8.0, 3.0, 1.0),))
    reset_view = driver.read_state(reset)
    np.testing.assert_array_equal(reset.control_step_count, (0,))
    np.testing.assert_allclose(reset_view.position_world_m[0, :2], (8.0, 3.0), atol=1e-6)
    assert reset_view.yaw_rad[0] == pytest.approx(1.0, abs=1e-6)


@pytest.mark.parametrize(
    ("operation", "value", "message"),
    (
        ("initial", (0.0, 0.0, 0.0), "rear-axle pose"),
        ("step", (0.0, 0.0), "batched actions"),
        ("reset", (True, False), "reset mask"),
    ),
)
def test_cpu_driver_rejects_wrong_batch_shapes(
    project_config,
    operation: str,
    value,
    message: str,
) -> None:
    driver = create_vehicle_driver("cpu_reference", project_config.vehicle, num_worlds=1)
    state = driver.initial_state(np.zeros((1, 3), dtype=np.float32))

    with pytest.raises(VehicleDriverShapeError, match=message):
        if operation == "initial":
            driver.initial_state(value)
        elif operation == "step":
            driver.step(state, value)
        else:
            driver.masked_reset(state, value, np.zeros((1, 3), dtype=np.float32))
