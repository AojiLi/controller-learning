"""Strict TOML loaders for project configuration."""

from __future__ import annotations

import tomllib
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from controller_learning.config.models import (
    ActuatorConfig,
    BenchmarkConfig,
    ConfigError,
    ControllerTimingConfig,
    EpisodeTimingConfig,
    LevelConfig,
    ProjectConfig,
    RankingConfig,
    SimulationConfig,
    VehicleConfig,
    VehicleGeometryConfig,
)


def _read_toml(path: Path) -> Mapping[str, Any]:
    if path.suffix != ".toml":
        raise ConfigError(f"configuration file must use the .toml suffix: {path}")
    try:
        with path.open("rb") as file:
            data = tomllib.load(file)
    except FileNotFoundError as error:
        raise ConfigError(f"configuration file does not exist: {path}") from error
    except tomllib.TOMLDecodeError as error:
        raise ConfigError(f"invalid TOML in {path}: {error}") from error
    return data


def _exact_keys(data: Mapping[str, Any], expected: set[str], context: str) -> None:
    actual = set(data)
    missing = expected - actual
    extra = actual - expected
    if missing or extra:
        details = []
        if missing:
            details.append(f"missing keys: {', '.join(sorted(missing))}")
        if extra:
            details.append(f"unexpected keys: {', '.join(sorted(extra))}")
        raise ConfigError(f"{context} has {'; '.join(details)}")


def _section(data: Mapping[str, Any], key: str, context: str) -> Mapping[str, Any]:
    value = data[key]
    if not isinstance(value, Mapping):
        raise ConfigError(f"{context}.{key} must be a TOML table")
    return value


def _integer(data: Mapping[str, Any], key: str, context: str) -> int:
    value = data[key]
    if type(value) is not int:
        raise ConfigError(f"{context}.{key} must be an integer")
    return value


def _number(data: Mapping[str, Any], key: str, context: str) -> float:
    value = data[key]
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ConfigError(f"{context}.{key} must be a number")
    return float(value)


def _boolean(data: Mapping[str, Any], key: str, context: str) -> bool:
    value = data[key]
    if type(value) is not bool:
        raise ConfigError(f"{context}.{key} must be a boolean")
    return value


def _string(data: Mapping[str, Any], key: str, context: str) -> str:
    value = data[key]
    if not isinstance(value, str):
        raise ConfigError(f"{context}.{key} must be a string")
    return value


def load_vehicle_config(path: str | Path) -> VehicleConfig:
    """Load and validate the candidate four-wheel vehicle configuration."""

    config_path = Path(path)
    data = _read_toml(config_path)
    _exact_keys(data, {"schema_version", "vehicle", "actuator", "simulation"}, "vehicle config")

    vehicle = _section(data, "vehicle", "vehicle config")
    _exact_keys(
        vehicle,
        {
            "mass_kg",
            "wheelbase_m",
            "track_width_m",
            "vehicle_width_m",
            "wheel_radius_m",
            "max_speed_mps",
        },
        "vehicle",
    )
    actuator = _section(data, "actuator", "vehicle config")
    _exact_keys(
        actuator,
        {
            "max_steering_angle_rad",
            "max_steering_rate_rad_s",
            "max_acceleration_mps2",
            "max_deceleration_mps2",
        },
        "actuator",
    )
    simulation = _section(data, "simulation", "vehicle config")
    _exact_keys(simulation, {"physics_dt_s", "control_dt_s"}, "simulation")

    return VehicleConfig(
        schema_version=_integer(data, "schema_version", "vehicle config"),
        vehicle=VehicleGeometryConfig(
            mass_kg=_number(vehicle, "mass_kg", "vehicle"),
            wheelbase_m=_number(vehicle, "wheelbase_m", "vehicle"),
            track_width_m=_number(vehicle, "track_width_m", "vehicle"),
            vehicle_width_m=_number(vehicle, "vehicle_width_m", "vehicle"),
            wheel_radius_m=_number(vehicle, "wheel_radius_m", "vehicle"),
            max_speed_mps=_number(vehicle, "max_speed_mps", "vehicle"),
        ),
        actuator=ActuatorConfig(
            max_steering_angle_rad=_number(actuator, "max_steering_angle_rad", "actuator"),
            max_steering_rate_rad_s=_number(actuator, "max_steering_rate_rad_s", "actuator"),
            max_acceleration_mps2=_number(actuator, "max_acceleration_mps2", "actuator"),
            max_deceleration_mps2=_number(actuator, "max_deceleration_mps2", "actuator"),
        ),
        simulation=SimulationConfig(
            physics_dt_s=_number(simulation, "physics_dt_s", "simulation"),
            control_dt_s=_number(simulation, "control_dt_s", "simulation"),
        ),
    )


def load_level_config(path: str | Path) -> LevelConfig:
    """Load and validate one public Challenge Level."""

    config_path = Path(path)
    data = _read_toml(config_path)
    _exact_keys(data, {"schema_version", "level", "track"}, "level config")
    level = _section(data, "level", "level config")
    _exact_keys(
        level,
        {
            "id",
            "name",
            "random_track_geometry",
            "fixed_vehicle_parameters",
            "fixed_start_state",
            "full_state_observation",
        },
        "level",
    )
    track = _section(data, "track", "level config")
    _exact_keys(track, {"source", "width_m"}, "track")

    return LevelConfig(
        schema_version=_integer(data, "schema_version", "level config"),
        level_id=_integer(level, "id", "level"),
        name=_string(level, "name", "level"),
        random_track_geometry=_boolean(level, "random_track_geometry", "level"),
        fixed_vehicle_parameters=_boolean(level, "fixed_vehicle_parameters", "level"),
        fixed_start_state=_boolean(level, "fixed_start_state", "level"),
        full_state_observation=_boolean(level, "full_state_observation", "level"),
        track_source=_string(track, "source", "track"),
        track_width_m=_number(track, "width_m", "track"),
    )


def load_benchmark_config(path: str | Path) -> BenchmarkConfig:
    """Load and validate the public benchmark protocol."""

    config_path = Path(path)
    data = _read_toml(config_path)
    _exact_keys(
        data,
        {"schema_version", "benchmark", "controller", "episode", "ranking"},
        "benchmark config",
    )
    benchmark = _section(data, "benchmark", "benchmark config")
    _exact_keys(benchmark, {"version", "official_level", "test_track_count"}, "benchmark")
    controller = _section(data, "controller", "benchmark config")
    _exact_keys(
        controller,
        {"frequency_hz", "init_timeout_s", "compute_deadline_s"},
        "controller",
    )
    episode = _section(data, "episode", "benchmark config")
    _exact_keys(episode, {"minimum_timeout_s", "timeout_reference_speed_mps"}, "episode")
    ranking = _section(data, "ranking", "benchmark config")
    _exact_keys(ranking, {"primary", "tiebreak"}, "ranking")

    return BenchmarkConfig(
        schema_version=_integer(data, "schema_version", "benchmark config"),
        version=_string(benchmark, "version", "benchmark"),
        official_level=_integer(benchmark, "official_level", "benchmark"),
        test_track_count=_integer(benchmark, "test_track_count", "benchmark"),
        controller=ControllerTimingConfig(
            frequency_hz=_number(controller, "frequency_hz", "controller"),
            init_timeout_s=_number(controller, "init_timeout_s", "controller"),
            compute_deadline_s=_number(controller, "compute_deadline_s", "controller"),
        ),
        episode=EpisodeTimingConfig(
            minimum_timeout_s=_number(episode, "minimum_timeout_s", "episode"),
            timeout_reference_speed_mps=_number(
                episode,
                "timeout_reference_speed_mps",
                "episode",
            ),
        ),
        ranking=RankingConfig(
            primary=_string(ranking, "primary", "ranking"),
            tiebreak=_string(ranking, "tiebreak", "ranking"),
        ),
    )


def load_project_config(root: str | Path) -> ProjectConfig:
    """Load the complete v0.1 configuration set rooted at a repository path."""

    root_path = Path(root)
    vehicle = load_vehicle_config(root_path / "configs" / "vehicle.toml")
    levels = (
        load_level_config(root_path / "configs" / "levels" / "level0.toml"),
        load_level_config(root_path / "configs" / "levels" / "level1.toml"),
    )
    benchmark = load_benchmark_config(root_path / "configs" / "benchmark.toml")
    return ProjectConfig(vehicle=vehicle, levels=levels, benchmark=benchmark)
