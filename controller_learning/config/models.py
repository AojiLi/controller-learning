"""Immutable configuration models and cross-field validation."""

from __future__ import annotations

from dataclasses import dataclass
from math import ceil, floor, isclose, isfinite

SCHEMA_VERSION = 1


class ConfigError(ValueError):
    """Raised when project configuration violates its public schema."""


def _require_positive(value: float, field: str) -> None:
    if not isfinite(value) or value <= 0.0:
        raise ConfigError(f"{field} must be a finite positive number, got {value!r}")


def _require_integer_at_least(value: int, minimum: int, field: str) -> None:
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise ConfigError(f"{field} must be an integer greater than or equal to {minimum}")


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
class TrackRepresentationConfig:
    """Fixed-shape Track storage and arc-length sampling parameters."""

    arc_spacing_m: float
    max_track_points: int
    checkpoint_spacing_m: float
    max_checkpoints: int

    def __post_init__(self) -> None:
        _require_positive(self.arc_spacing_m, "track.representation.arc_spacing_m")
        _require_integer_at_least(
            self.max_track_points,
            4,
            "track.representation.max_track_points",
        )
        _require_positive(
            self.checkpoint_spacing_m,
            "track.representation.checkpoint_spacing_m",
        )
        _require_integer_at_least(
            self.max_checkpoints,
            1,
            "track.representation.max_checkpoints",
        )


@dataclass(frozen=True, slots=True)
class TrackGeneratorConfig:
    """Versioned parameters for deterministic periodic-spline generation."""

    generator_version: str
    min_control_points: int
    max_control_points: int
    min_radius_m: float
    max_radius_m: float
    angular_gap_jitter: float
    radial_perturbation: float
    dense_samples_per_control_point: int
    arc_length_convergence_m: float
    tail_merge_fraction: float

    def __post_init__(self) -> None:
        if not self.generator_version:
            raise ConfigError("track.generator.generator_version cannot be empty")
        _require_integer_at_least(
            self.min_control_points,
            4,
            "track.generator.min_control_points",
        )
        _require_integer_at_least(
            self.max_control_points,
            self.min_control_points,
            "track.generator.max_control_points",
        )
        _require_positive(self.min_radius_m, "track.generator.min_radius_m")
        _require_positive(self.max_radius_m, "track.generator.max_radius_m")
        if self.min_radius_m >= self.max_radius_m:
            raise ConfigError("track generator radius limits must be strictly ordered")
        for value, field in (
            (self.angular_gap_jitter, "angular_gap_jitter"),
            (self.radial_perturbation, "radial_perturbation"),
        ):
            if not isfinite(value) or not 0.0 <= value < 1.0:
                raise ConfigError(f"track.generator.{field} must be in [0, 1)")
        _require_integer_at_least(
            self.dense_samples_per_control_point,
            64,
            "track.generator.dense_samples_per_control_point",
        )
        _require_positive(
            self.arc_length_convergence_m,
            "track.generator.arc_length_convergence_m",
        )
        if not isfinite(self.tail_merge_fraction) or not 0.0 < self.tail_merge_fraction <= 1.0:
            raise ConfigError("track.generator.tail_merge_fraction must be in (0, 1]")


@dataclass(frozen=True, slots=True)
class TrackValidationConfig:
    """Geometric validity limits for generated Level 1 tracks."""

    min_length_m: float
    max_length_m: float
    min_turn_radius_m: float
    min_nonlocal_centerline_clearance_m: float
    local_arc_exclusion_m: float
    start_window_m: float
    start_max_curvature_1pm: float

    def __post_init__(self) -> None:
        for field in (
            "min_length_m",
            "max_length_m",
            "min_turn_radius_m",
            "min_nonlocal_centerline_clearance_m",
            "local_arc_exclusion_m",
            "start_window_m",
            "start_max_curvature_1pm",
        ):
            _require_positive(getattr(self, field), f"track.validation.{field}")
        if self.min_length_m >= self.max_length_m:
            raise ConfigError("track validation length limits must be strictly ordered")
        if self.start_window_m >= self.min_length_m:
            raise ConfigError("track.validation.start_window_m must be shorter than min_length_m")
        if self.local_arc_exclusion_m < self.min_nonlocal_centerline_clearance_m:
            raise ConfigError(
                "track.validation.local_arc_exclusion_m must be at least the nonlocal clearance"
            )

    @property
    def max_abs_curvature_1pm(self) -> float:
        """Return the curvature limit corresponding to ``min_turn_radius_m``."""

        return 1.0 / self.min_turn_radius_m


@dataclass(frozen=True, slots=True)
class TrackRaceConfig:
    """Topology-local projection and effective-boundary rules."""

    safety_margin_m: float
    projection_backward_segments: int
    projection_forward_segments: int

    def __post_init__(self) -> None:
        if not isfinite(self.safety_margin_m) or self.safety_margin_m < 0.0:
            raise ConfigError("track.race.safety_margin_m must be finite and non-negative")
        _require_integer_at_least(
            self.projection_backward_segments,
            0,
            "track.race.projection_backward_segments",
        )
        _require_integer_at_least(
            self.projection_forward_segments,
            1,
            "track.race.projection_forward_segments",
        )


@dataclass(frozen=True, slots=True)
class TrackConfig:
    """Complete Track generation, validation, representation, and race configuration."""

    schema_version: int
    representation: TrackRepresentationConfig
    generator: TrackGeneratorConfig
    validation: TrackValidationConfig
    race: TrackRaceConfig

    def __post_init__(self) -> None:
        _validate_schema_version(self.schema_version, "track")
        if self.generator.arc_length_convergence_m >= self.representation.arc_spacing_m:
            raise ConfigError(
                "track generator arc-length convergence must be smaller than arc spacing"
            )
        if self.generator.min_radius_m < self.validation.min_turn_radius_m:
            raise ConfigError(
                "track generator minimum radius cannot be smaller than the validation turn radius"
            )
        if self.representation.max_track_points < self.required_track_points:
            raise ConfigError(
                "track.representation.max_track_points cannot cover the maximum valid track "
                f"length: requires at least {self.required_track_points}"
            )
        if self.representation.max_checkpoints < self.required_checkpoints:
            raise ConfigError(
                "track.representation.max_checkpoints cannot cover the maximum valid track "
                f"length: requires at least {self.required_checkpoints}"
            )

    @property
    def required_track_points(self) -> int:
        """Return the worst-case point count, including the explicit closing point."""

        spacing = self.representation.arc_spacing_m
        full_steps = floor(self.validation.max_length_m / spacing)
        remainder_m = self.validation.max_length_m - full_steps * spacing
        if remainder_m <= 1.0e-10 or remainder_m < spacing * self.generator.tail_merge_fraction:
            return full_steps + 1
        return full_steps + 2

    @property
    def required_checkpoints(self) -> int:
        """Return the worst-case ordered checkpoint count, including the finish line."""

        return ceil(self.validation.max_length_m / self.representation.checkpoint_spacing_m)


@dataclass(frozen=True, slots=True)
class ProjectConfig:
    """Cross-validated vehicle, Levels, and benchmark protocol."""

    vehicle: VehicleConfig
    levels: tuple[LevelConfig, ...]
    benchmark: BenchmarkConfig
    track: TrackConfig

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
        track_width_m = level1.track_width_m
        if (
            track_width_m
            <= self.vehicle.vehicle.vehicle_width_m + 2.0 * self.track.race.safety_margin_m
        ):
            raise ConfigError(
                "track effective half-width must remain positive after vehicle width and safety "
                "margin"
            )
        if self.track.validation.min_turn_radius_m <= 0.5 * track_width_m:
            raise ConfigError(
                "track validation turn radius must exceed half the fixed Level track width"
            )
        if self.track.validation.min_nonlocal_centerline_clearance_m <= track_width_m:
            raise ConfigError(
                "track nonlocal centerline clearance must exceed the fixed Level track width"
            )
        if not isclose(
            self.vehicle.simulation.control_dt_s,
            self.benchmark.controller.period_s,
            rel_tol=0.0,
            abs_tol=1e-12,
        ):
            raise ConfigError("vehicle control_dt must match benchmark Controller frequency")
