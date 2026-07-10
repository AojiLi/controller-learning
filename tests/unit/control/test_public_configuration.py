"""Tests for the exact immutable Controller configuration boundary."""

from __future__ import annotations

from pathlib import Path
from types import MappingProxyType

import pytest

from controller_learning.config import load_project_config
from controller_learning.control.configuration import (
    PUBLIC_CONTROLLER_CONFIG_KEYS,
    build_public_controller_config,
)

PROJECT_ROOT = Path(__file__).parents[3]


def _all_mapping_keys(value: object) -> set[str]:
    if not isinstance(value, dict | MappingProxyType):
        return set()
    keys = set(value)
    for item in value.values():
        keys.update(_all_mapping_keys(item))
    return keys


def test_public_config_contains_only_the_documented_challenge_values() -> None:
    project = load_project_config(PROJECT_ROOT)

    config = build_public_controller_config(
        project,
        level_id=1,
        controller_parameters={"gains": {"speed": [1.0, 2.0]}},
    )

    assert tuple(config) == PUBLIC_CONTROLLER_CONFIG_KEYS
    assert config["benchmark_version"] == project.benchmark.version
    assert config["level_id"] == 1
    assert config["level_name"] == project.levels[1].name
    assert config["control_dt_s"] == project.vehicle.simulation.control_dt_s
    assert config["vehicle"] == {
        "mass_kg": project.vehicle.vehicle.mass_kg,
        "wheelbase_m": project.vehicle.vehicle.wheelbase_m,
        "track_width_m": project.vehicle.vehicle.track_width_m,
        "vehicle_width_m": project.vehicle.vehicle.vehicle_width_m,
        "wheel_radius_m": project.vehicle.vehicle.wheel_radius_m,
        "max_speed_mps": project.vehicle.vehicle.max_speed_mps,
    }
    assert config["action_limits"] == {
        "max_steering_angle_rad": project.vehicle.actuator.max_steering_angle_rad,
        "max_steering_rate_rad_s": project.vehicle.actuator.max_steering_rate_rad_s,
        "max_acceleration_mps2": project.vehicle.actuator.max_acceleration_mps2,
        "max_deceleration_mps2": project.vehicle.actuator.max_deceleration_mps2,
    }
    assert config["track"] == {
        "width_m": project.levels[1].track_width_m,
        "max_track_points": project.track.representation.max_track_points,
        "max_checkpoints": project.track.representation.max_checkpoints,
    }

    forbidden = {
        "physics_dt_s",
        "generator_version",
        "validation",
        "projection_backward_segments",
        "projection_forward_segments",
        "track_source",
        "test_track_count",
        "ranking",
        "backend",
        "simulator",
    }
    assert forbidden.isdisjoint(_all_mapping_keys(config))


def test_public_config_and_plugin_parameters_are_recursively_immutable() -> None:
    project = load_project_config(PROJECT_ROOT)
    source = {"nested": {"sequence": [1, {"gain": 2.0}]}}

    config = build_public_controller_config(project, 0, source)

    assert isinstance(config, MappingProxyType)
    assert isinstance(config["vehicle"], MappingProxyType)
    assert isinstance(config["controller"], MappingProxyType)
    assert config["controller"]["nested"]["sequence"] == (1, {"gain": 2.0})
    source["nested"]["sequence"].append(3)  # type: ignore[union-attr]
    assert len(config["controller"]["nested"]["sequence"]) == 2
    with pytest.raises(TypeError):
        config["level_id"] = 7  # type: ignore[index]
    with pytest.raises(TypeError):
        config["vehicle"]["wheelbase_m"] = 99.0
    with pytest.raises(TypeError):
        config["controller"]["nested"]["sequence"][1]["gain"] = 4.0


def test_public_config_rejects_an_unknown_or_noninteger_level() -> None:
    project = load_project_config(PROJECT_ROOT)

    with pytest.raises(ValueError, match="unknown level_id"):
        build_public_controller_config(project, 9, {})
    with pytest.raises(TypeError, match="level_id"):
        build_public_controller_config(project, True, {})  # type: ignore[arg-type]
