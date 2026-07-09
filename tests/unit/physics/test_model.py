"""Tests for the packaged physical four-wheel vehicle model."""

from importlib.resources import files
from pathlib import Path

import mujoco
import numpy as np
import pytest

from controller_learning.config import load_vehicle_config
from controller_learning.physics.model import (
    DRIVE_ACTUATOR_NAMES,
    STEERING_JOINT_NAMES,
    WHEEL_JOINT_NAMES,
    VehicleModelError,
    load_vehicle_model,
)

PROJECT_ROOT = Path(__file__).parents[3]


@pytest.fixture
def vehicle_config():
    return load_vehicle_config(PROJECT_ROOT / "configs" / "vehicle.toml")


def test_packaged_mjcf_exists() -> None:
    resource = files("controller_learning").joinpath("assets", "vehicle", "car.xml")

    assert resource.is_file()


def test_vehicle_model_has_required_four_wheel_structure(vehicle_config) -> None:
    model, indices = load_vehicle_model(vehicle_config)

    assert model.nq == 13
    assert model.nv == 12
    assert model.nu == 6
    assert len(indices.steering_joints) == len(STEERING_JOINT_NAMES) == 2
    assert len(indices.wheel_joints) == len(WHEEL_JOINT_NAMES) == 4
    assert len(indices.drive_actuators) == len(DRIVE_ACTUATOR_NAMES) == 4
    assert model.jnt_type[0] == mujoco.mjtJoint.mjJNT_FREE
    assert np.isclose(model.body_subtreemass[indices.chassis_body], 1200.0)


@pytest.mark.parametrize("physics_dt_s", [0.01, 0.005, 0.002])
def test_vehicle_model_accepts_confirmed_timestep_candidates(vehicle_config, physics_dt_s) -> None:
    model, _ = load_vehicle_model(vehicle_config, physics_dt_s=physics_dt_s)

    assert model.opt.timestep == physics_dt_s


def test_vehicle_model_rejects_non_integral_control_ratio(vehicle_config) -> None:
    with pytest.raises(VehicleModelError, match="must be an integer"):
        load_vehicle_model(vehicle_config, physics_dt_s=0.007)
