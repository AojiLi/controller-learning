"""Compose the immutable configuration exposed to trusted Controller plugins."""

from __future__ import annotations

from collections.abc import Mapping
from types import MappingProxyType
from typing import Any, TypeAlias

from controller_learning.config.models import ProjectConfig

PublicControllerConfig: TypeAlias = Mapping[str, Any]

PUBLIC_CONTROLLER_CONFIG_KEYS = (
    "benchmark_version",
    "level_id",
    "level_name",
    "control_dt_s",
    "vehicle",
    "action_limits",
    "track",
    "controller",
)
"""The complete top-level public Controller configuration whitelist."""


def _freeze(value: Any) -> Any:
    if isinstance(value, Mapping):
        return MappingProxyType({str(key): _freeze(item) for key, item in value.items()})
    if isinstance(value, (list, tuple)):
        return tuple(_freeze(item) for item in value)
    return value


def build_public_controller_config(
    project_config: ProjectConfig,
    level_id: int,
    controller_parameters: Mapping[str, Any],
) -> PublicControllerConfig:
    """Return the exact read-only Challenge and plugin values visible to a Controller.

    Generator, validator, projection, pool, evaluation-order, simulator, and backend settings are
    deliberately absent. Controller-specific TOML values live under ``controller`` so they cannot
    shadow Challenge-owned values.
    """

    if not isinstance(project_config, ProjectConfig):
        raise TypeError("project_config must be a ProjectConfig")
    if isinstance(level_id, bool) or not isinstance(level_id, int):
        raise TypeError("level_id must be an integer")
    if not isinstance(controller_parameters, Mapping):
        raise TypeError("controller_parameters must be a mapping")

    levels_by_id = {level.level_id: level for level in project_config.levels}
    try:
        level = levels_by_id[level_id]
    except KeyError as error:
        available = ", ".join(str(identifier) for identifier in sorted(levels_by_id))
        raise ValueError(f"unknown level_id {level_id}; available levels: {available}") from error

    geometry = project_config.vehicle.vehicle
    actuator = project_config.vehicle.actuator
    representation = project_config.track.representation
    values = {
        "benchmark_version": project_config.benchmark.version,
        "level_id": level.level_id,
        "level_name": level.name,
        "control_dt_s": project_config.vehicle.simulation.control_dt_s,
        "vehicle": {
            "mass_kg": geometry.mass_kg,
            "wheelbase_m": geometry.wheelbase_m,
            "track_width_m": geometry.track_width_m,
            "vehicle_width_m": geometry.vehicle_width_m,
            "wheel_radius_m": geometry.wheel_radius_m,
            "max_speed_mps": geometry.max_speed_mps,
        },
        "action_limits": {
            "max_steering_angle_rad": actuator.max_steering_angle_rad,
            "max_steering_rate_rad_s": actuator.max_steering_rate_rad_s,
            "max_acceleration_mps2": actuator.max_acceleration_mps2,
            "max_deceleration_mps2": actuator.max_deceleration_mps2,
        },
        "track": {
            "width_m": level.track_width_m,
            "max_track_points": representation.max_track_points,
            "max_checkpoints": representation.max_checkpoints,
        },
        "controller": controller_parameters,
    }
    frozen = _freeze(values)
    if not isinstance(frozen, Mapping):  # pragma: no cover - fixed construction invariant
        raise AssertionError("public Controller config must be a mapping")
    return frozen


__all__ = [
    "PUBLIC_CONTROLLER_CONFIG_KEYS",
    "PublicControllerConfig",
    "build_public_controller_config",
]
