"""Tests for the shared physical action-to-actuator mapping."""

from pathlib import Path

import numpy as np
import pytest

from controller_learning.config import load_vehicle_config
from controller_learning.physics.actuation import (
    VehicleActionError,
    map_vehicle_action,
    wheel_torques_for_acceleration,
)

PROJECT_ROOT = Path(__file__).parents[3]


@pytest.fixture
def vehicle_config():
    return load_vehicle_config(PROJECT_ROOT / "configs" / "vehicle.toml")


def test_action_is_clipped_and_steering_target_is_rate_limited(vehicle_config) -> None:
    applied = map_vehicle_action(
        vehicle_config,
        (10.0, 10.0),
        previous_steering_target_rad=0.0,
        wheel_angular_velocity_rad_s=(0.0, 0.0, 0.0, 0.0),
        longitudinal_velocity_mps=0.0,
    )

    assert applied.steering_angle_rad == 0.6
    assert applied.longitudinal_acceleration_mps2 == 4.0
    assert np.isclose(applied.steering_target_rad, 0.06)
    assert applied.wheel_torque_nm == pytest.approx((408.0, 408.0, 408.0, 408.0))
    assert applied.saturation_count == 2


def test_braking_torque_opposes_each_rotating_wheel(vehicle_config) -> None:
    torque = wheel_torques_for_acceleration(
        vehicle_config,
        -8.0,
        (10.0, 12.0, 9.0, 11.0),
        3.0,
    )

    assert torque == pytest.approx((-816.0, -816.0, -816.0, -816.0))


def test_braking_does_not_create_reverse_torque_at_rest(vehicle_config) -> None:
    torque = wheel_torques_for_acceleration(
        vehicle_config,
        -8.0,
        (0.0, 0.0, 0.0, 0.0),
        0.0,
    )

    assert torque == (0.0, 0.0, 0.0, 0.0)


@pytest.mark.parametrize(
    "action",
    [(0.0,), (0.0, 0.0, 0.0), (np.nan, 0.0), (0.0, np.inf), ("bad", 0.0)],
)
def test_invalid_actions_raise_domain_error(vehicle_config, action) -> None:
    with pytest.raises(VehicleActionError):
        map_vehicle_action(
            vehicle_config,
            action,
            previous_steering_target_rad=0.0,
            wheel_angular_velocity_rad_s=(0.0, 0.0, 0.0, 0.0),
            longitudinal_velocity_mps=0.0,
        )
