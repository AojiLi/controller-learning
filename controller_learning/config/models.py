"""Immutable configuration models and cross-field validation."""

from __future__ import annotations

from dataclasses import dataclass
from math import isclose, isfinite

SCHEMA_VERSION = 1


class ConfigError(ValueError):
    """Raised when project configuration violates its public schema."""


def _require_positive(value: float, field: str) -> None:
    if not isfinite(value) or value <= 0.0:
        raise ConfigError(f"{field} must be a finite positive number, got {value!r}")


def _validate_schema_version(value: int, config_name: str) -> None:
    if value != SCHEMA_VERSION:
        raise ConfigError(f"{config_name}.schema_version must be {SCHEMA_VERSION}, got {value!r}")


@dataclass(frozen=True, slots=True)
class VehicleGeometryConfig:
    """Physical dimensions shared by the vehicle model and public configuration."""

    mass_kg: float
    wheelbase_m: float
    track_width_m: float
    vehicle_width_m: float
    wheel_radius_m: float
    max_speed_mps: float

    def __post_init__(self) -> None:
        for field in (
            "mass_kg",
            "wheelbase_m",
            "track_width_m",
            "vehicle_width_m",
            "wheel_radius_m",
            "max_speed_mps",
        ):
            _require_positive(getattr(self, field), f"vehicle.{field}")
        if self.track_width_m >= self.vehicle_width_m:
            raise ConfigError("vehicle.track_width_m must be smaller than vehicle.vehicle_width_m")
        if 2.0 * self.wheel_radius_m >= self.wheelbase_m:
            raise ConfigError("vehicle wheels are too large for the configured wheelbase")


@dataclass(frozen=True, slots=True)
class ActuatorConfig:
    """Physical action limits shared by every Controller."""

    max_steering_angle_rad: float
    max_steering_rate_rad_s: float
    max_acceleration_mps2: float
    max_deceleration_mps2: float

    def __post_init__(self) -> None:
        for field in (
            "max_steering_angle_rad",
            "max_steering_rate_rad_s",
            "max_acceleration_mps2",
            "max_deceleration_mps2",
        ):
            _require_positive(getattr(self, field), f"actuator.{field}")
        if self.max_steering_angle_rad >= 1.5708:
            raise ConfigError("actuator.max_steering_angle_rad must be less than pi/2")


@dataclass(frozen=True, slots=True)
class SimulationConfig:
    """Simulation and Controller timing, subject to the current milestone gates."""

    physics_dt_s: float
    control_dt_s: float

    def __post_init__(self) -> None:
        _require_positive(self.physics_dt_s, "simulation.physics_dt_s")
        _require_positive(self.control_dt_s, "simulation.control_dt_s")
        if self.physics_dt_s > self.control_dt_s:
            raise ConfigError("simulation.physics_dt_s cannot exceed simulation.control_dt_s")
        ratio = self.control_dt_s / self.physics_dt_s
        if not isclose(ratio, round(ratio), rel_tol=0.0, abs_tol=1e-9):
            raise ConfigError("simulation.control_dt_s / physics_dt_s must be an integer")

    @property
    def physics_steps_per_control(self) -> int:
        """Return the number of physics substeps in one Controller period."""

        return round(self.control_dt_s / self.physics_dt_s)


@dataclass(frozen=True, slots=True)
class VehicleConfig:
    """Complete vehicle configuration."""

    schema_version: int
    vehicle: VehicleGeometryConfig
    actuator: ActuatorConfig
    simulation: SimulationConfig

    def __post_init__(self) -> None:
        _validate_schema_version(self.schema_version, "vehicle")


@dataclass(frozen=True, slots=True)
class LevelConfig:
    """Public Level rules that cannot be changed by a Controller."""

    schema_version: int
    level_id: int
    name: str
    random_track_geometry: bool
    fixed_vehicle_parameters: bool
    fixed_start_state: bool
    full_state_observation: bool
    track_source: str
    track_width_m: float

    def __post_init__(self) -> None:
        _validate_schema_version(self.schema_version, self.name or "level")
        if self.level_id < 0:
            raise ConfigError("level.id must be non-negative")
        if not self.name:
            raise ConfigError("level.name cannot be empty")
        if self.track_source not in {"fixed", "procedural_pool"}:
            raise ConfigError("track.source must be 'fixed' or 'procedural_pool'")
        _require_positive(self.track_width_m, "track.width_m")


@dataclass(frozen=True, slots=True)
class ControllerTimingConfig:
    """Controller frequency and soft timing limits."""

    frequency_hz: float
    init_timeout_s: float
    compute_deadline_s: float

    def __post_init__(self) -> None:
        _require_positive(self.frequency_hz, "controller.frequency_hz")
        _require_positive(self.init_timeout_s, "controller.init_timeout_s")
        _require_positive(self.compute_deadline_s, "controller.compute_deadline_s")
        if self.compute_deadline_s > self.period_s:
            raise ConfigError("controller.compute_deadline_s cannot exceed one Controller period")

    @property
    def period_s(self) -> float:
        """Return one Controller period in seconds."""

        return 1.0 / self.frequency_hz


@dataclass(frozen=True, slots=True)
class EpisodeTimingConfig:
    """Length-aware episode timeout parameters."""

    minimum_timeout_s: float
    timeout_reference_speed_mps: float

    def __post_init__(self) -> None:
        _require_positive(self.minimum_timeout_s, "episode.minimum_timeout_s")
        _require_positive(self.timeout_reference_speed_mps, "episode.timeout_reference_speed_mps")


@dataclass(frozen=True, slots=True)
class RankingConfig:
    """Official lexicographic benchmark ordering."""

    primary: str
    tiebreak: str

    def __post_init__(self) -> None:
        if self.primary != "success_rate":
            raise ConfigError("ranking.primary must be 'success_rate'")
        if self.tiebreak != "mean_successful_lap_time":
            raise ConfigError("ranking.tiebreak must be 'mean_successful_lap_time'")


@dataclass(frozen=True, slots=True)
class BenchmarkConfig:
    """Versioned public benchmark protocol settings."""

    schema_version: int
    version: str
    official_level: int
    test_track_count: int
    controller: ControllerTimingConfig
    episode: EpisodeTimingConfig
    ranking: RankingConfig

    def __post_init__(self) -> None:
        _validate_schema_version(self.schema_version, "benchmark")
        if not self.version:
            raise ConfigError("benchmark.version cannot be empty")
        if self.official_level < 0:
            raise ConfigError("benchmark.official_level must be non-negative")
        if self.test_track_count < 20:
            raise ConfigError("benchmark.test_track_count must be at least 20")


@dataclass(frozen=True, slots=True)
class ProjectConfig:
    """Cross-validated vehicle, Levels, and benchmark protocol."""

    vehicle: VehicleConfig
    levels: tuple[LevelConfig, ...]
    benchmark: BenchmarkConfig

    def __post_init__(self) -> None:
        levels_by_id = {level.level_id: level for level in self.levels}
        if len(levels_by_id) != len(self.levels):
            raise ConfigError("level ids must be unique")
        if set(levels_by_id) != {0, 1}:
            raise ConfigError("v0.1 must define exactly Level 0 and Level 1")
        if self.benchmark.official_level != 1:
            raise ConfigError("Level 1 must be the official v0.1 benchmark")

        level0 = levels_by_id[0]
        level1 = levels_by_id[1]
        if level0.random_track_geometry or level0.track_source != "fixed":
            raise ConfigError("Level 0 must use fixed track geometry")
        if not level1.random_track_geometry or level1.track_source != "procedural_pool":
            raise ConfigError("Level 1 must use the procedural track pool")
        for level in self.levels:
            if not (
                level.fixed_vehicle_parameters
                and level.fixed_start_state
                and level.full_state_observation
            ):
                raise ConfigError(
                    "v0.1 Levels require fixed vehicle/start and full-state observation"
                )
        if not isclose(level0.track_width_m, level1.track_width_m):
            raise ConfigError("Level 0 and Level 1 must use the same fixed track width")
        if not isclose(
            self.vehicle.simulation.control_dt_s,
            self.benchmark.controller.period_s,
            rel_tol=0.0,
            abs_tol=1e-12,
        ):
            raise ConfigError("vehicle control_dt must match benchmark Controller frequency")
