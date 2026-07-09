"""Integration tests for the CPU four-wheel vehicle reference."""

from math import cos, pi, sin
from pathlib import Path

import mujoco
import numpy as np
import pytest

from controller_learning.config import load_vehicle_config
from controller_learning.physics import CpuVehicle

PROJECT_ROOT = Path(__file__).parents[3]


@pytest.fixture
def vehicle_config():
    return load_vehicle_config(PROJECT_ROOT / "configs" / "vehicle.toml")


def test_reset_uses_rear_axle_reference_and_project_coordinates(vehicle_config) -> None:
    vehicle = CpuVehicle(vehicle_config)
    state = vehicle.reset()

    assert state.time_s == 0.0
    assert state.position_world_m == pytest.approx((-1.35, 0.0, 0.34))
    assert state.chassis_position_world_m == pytest.approx((0.0, 0.0, 0.56))
    assert state.quaternion_wxyz == pytest.approx((1.0, 0.0, 0.0, 0.0))
    assert state.roll_rad == pytest.approx(0.0)
    assert state.pitch_rad == pytest.approx(0.0)
    assert state.yaw_rad == pytest.approx(0.0)
    assert state.velocity_body_mps == pytest.approx((0.0, 0.0, 0.0))


def test_zero_action_settles_on_all_four_wheels_without_unexpected_contact(vehicle_config) -> None:
    vehicle = CpuVehicle(vehicle_config)

    for _ in range(20):
        state = vehicle.step((0.0, 0.0))

    contacts = vehicle.contact_metrics()
    assert np.isfinite(state.position_world_m).all()
    assert all(count >= 1 for count in contacts.wheel_ground_contact_count)
    assert contacts.unexpected_contact_count == 0
    assert contacts.maximum_penetration_m < 0.005
    assert vehicle.warning_count == 0


def test_control_step_has_exact_duration_and_rate_limited_target(vehicle_config) -> None:
    vehicle = CpuVehicle(vehicle_config)

    state = vehicle.step((1.0, 10.0))

    assert state.time_s == pytest.approx(0.05)
    assert vehicle.last_applied_action.steering_target_rad == pytest.approx(0.06)
    assert vehicle.last_applied_action.longitudinal_acceleration_mps2 == 4.0
    assert vehicle.last_applied_action.saturation_count == 2
    diagnostics = vehicle.last_step_diagnostics
    assert diagnostics.physics_step_count == 10
    assert all(0.0 <= fraction <= 1.0 for fraction in diagnostics.wheel_ground_contact_fraction)
    assert diagnostics.maximum_penetration_m >= 0.0
    assert diagnostics.maximum_wheel_contact_gap_s <= 0.05


def test_reset_is_deterministic_after_motion(vehicle_config) -> None:
    vehicle = CpuVehicle(vehicle_config)
    initial = vehicle.reset()
    for _ in range(10):
        vehicle.step((0.2, 2.0))

    reset = vehicle.reset()

    assert reset == initial


def test_body_velocity_is_expressed_in_the_rotated_vehicle_frame(vehicle_config) -> None:
    vehicle = CpuVehicle(vehicle_config)
    yaw = pi / 2.0
    vehicle.data.qpos[:3] = (0.0, 0.0, 2.0)
    vehicle.data.qpos[3:7] = (cos(yaw / 2.0), 0.0, 0.0, sin(yaw / 2.0))
    vehicle.data.qvel[:6] = (0.0, 3.0, 0.0, 0.0, 0.0, 0.0)
    mujoco.mj_forward(vehicle.model, vehicle.data)

    state = vehicle.state()

    assert state.yaw_rad == pytest.approx(yaw)
    assert state.velocity_body_mps == pytest.approx((3.0, 0.0, 0.0), abs=1e-12)
