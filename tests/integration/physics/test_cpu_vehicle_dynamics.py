"""Behavior regressions for the physical CPU four-wheel vehicle."""

from pathlib import Path

import numpy as np
import pytest

from controller_learning.config import load_vehicle_config
from controller_learning.physics import CpuVehicle

PROJECT_ROOT = Path(__file__).parents[3]


@pytest.fixture
def vehicle_config():
    return load_vehicle_config(PROJECT_ROOT / "configs" / "vehicle.toml")


def _advance(vehicle: CpuVehicle, action: tuple[float, float], steps: int):
    states = []
    for _ in range(steps):
        states.append(vehicle.step(action))
    return states


def test_vehicle_remains_stable_at_rest_for_ten_seconds(vehicle_config) -> None:
    vehicle = CpuVehicle(vehicle_config)
    initial = vehicle.state()
    states = _advance(vehicle, (0.0, 0.0), 200)
    final = states[-1]
    contact = vehicle.contact_metrics()

    xy_drift = np.linalg.norm(
        np.asarray(final.position_world_m[:2]) - np.asarray(initial.position_world_m[:2])
    )
    assert xy_drift < 0.01
    assert abs(final.yaw_rad) < 0.005
    assert abs(final.roll_rad) < 0.01
    assert abs(final.pitch_rad) < 0.01
    assert abs(final.longitudinal_velocity_mps) < 0.02
    assert all(count >= 1 for count in contact.wheel_ground_contact_count)
    assert sum(contact.wheel_normal_force_n) == pytest.approx(1200.0 * 9.81, rel=0.02)
    assert contact.unexpected_contact_count == 0
    assert contact.maximum_penetration_m < 0.005
    assert vehicle.warning_count == 0


def test_positive_torque_drives_straight_along_world_x(vehicle_config) -> None:
    vehicle = CpuVehicle(vehicle_config)
    initial = vehicle.state()
    _advance(vehicle, (0.0, 0.0), 20)
    acceleration_states = _advance(vehicle, (0.0, 2.0), 60)
    final = _advance(vehicle, (0.0, 0.0), 40)[-1]

    displacement_x = final.position_world_m[0] - initial.position_world_m[0]
    assert displacement_x > 5.0
    assert 4.5 <= acceleration_states[-1].longitudinal_velocity_mps <= 7.5
    assert abs(final.position_world_m[1]) < 0.02
    assert abs(final.yaw_rad) < 0.01
    assert all(rate > 0.0 for rate in final.wheel_angular_velocity_rad_s)


def _steering_run(vehicle_config, steering_angle_rad: float):
    vehicle = CpuVehicle(vehicle_config)
    _advance(vehicle, (0.0, 0.0), 20)
    _advance(vehicle, (0.0, 1.5), 60)
    _advance(vehicle, (steering_angle_rad, 0.0), 40)
    return _advance(vehicle, (0.0, 0.0), 20)[-1]


def test_steering_sign_and_left_right_symmetry(vehicle_config) -> None:
    left = _steering_run(vehicle_config, 0.2)
    right = _steering_run(vehicle_config, -0.2)

    assert left.position_world_m[1] > 0.2
    assert left.yaw_rad > 0.1
    assert right.position_world_m[1] < -0.2
    assert right.yaw_rad < -0.1
    assert left.position_world_m[1] == pytest.approx(-right.position_world_m[1], rel=0.01)
    assert left.yaw_rad == pytest.approx(-right.yaw_rad, rel=0.01)


def test_negative_acceleration_brakes_without_reversing(vehicle_config) -> None:
    vehicle = CpuVehicle(vehicle_config)
    _advance(vehicle, (0.0, 0.0), 20)
    speed_before_braking = _advance(vehicle, (0.0, 2.0), 80)[-1].longitudinal_velocity_mps
    braking_states = _advance(vehicle, (0.0, -4.0), 40)
    final = _advance(vehicle, (0.0, 0.0), 20)[-1]
    braking_speeds = np.asarray([state.longitudinal_velocity_mps for state in braking_states])

    assert speed_before_braking > 5.0
    assert final.longitudinal_velocity_mps <= 0.5
    assert braking_speeds.min() >= -0.2
    assert np.mean(np.diff(braking_speeds) <= 1e-6) >= 0.95


@pytest.mark.parametrize("physics_dt_s", [0.01, 0.005, 0.002])
def test_timestep_candidates_keep_four_wheel_rest_contact(vehicle_config, physics_dt_s) -> None:
    vehicle = CpuVehicle(vehicle_config, physics_dt_s=physics_dt_s)
    _advance(vehicle, (0.0, 0.0), 40)
    contact = vehicle.contact_metrics()

    assert all(count >= 1 for count in contact.wheel_ground_contact_count)
    assert contact.unexpected_contact_count == 0
    assert contact.maximum_penetration_m < 0.005
    assert vehicle.warning_count == 0
