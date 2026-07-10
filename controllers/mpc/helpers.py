"""Strict configuration and observation-derived planning for the MPC example."""

from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

import numpy as np
from numpy.typing import NDArray

from controller_learning.control import CenterlineReference, curvature_speed_profile


class MpcConfigurationError(ValueError):
    """Raised when ``controllers/mpc/config.toml`` violates its public schema."""


def _mapping(value: object, name: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise MpcConfigurationError(f"{name} must be a table")
    return value


def _exact_keys(value: Mapping[str, Any], expected: set[str], name: str) -> None:
    actual = set(value)
    missing = sorted(expected - actual)
    extra = sorted(actual - expected)
    if missing or extra:
        raise MpcConfigurationError(f"{name} keys differ; missing={missing}, extra={extra}")


def _number(value: Mapping[str, Any], key: str, name: str, *, minimum: float = 0.0) -> float:
    item = value[key]
    if isinstance(item, bool) or not isinstance(item, (int, float)):
        raise MpcConfigurationError(f"{name}.{key} must be a number")
    result = float(item)
    if not math.isfinite(result) or result < minimum:
        raise MpcConfigurationError(f"{name}.{key} must be finite and at least {minimum}")
    return result


def _positive(value: Mapping[str, Any], key: str, name: str) -> float:
    result = _number(value, key, name)
    if result <= 0.0:
        raise MpcConfigurationError(f"{name}.{key} must be positive")
    return result


def _positive_integer(value: Mapping[str, Any], key: str, name: str) -> int:
    item = value[key]
    if isinstance(item, bool) or not isinstance(item, int) or item < 1:
        raise MpcConfigurationError(f"{name}.{key} must be a positive integer")
    return item


@dataclass(frozen=True, slots=True)
class HorizonConfig:
    """Fixed v0.1 prediction-horizon shape."""

    steps: int


@dataclass(frozen=True, slots=True)
class PlanningConfig:
    """Observation-derived reference and speed-profile parameters."""

    minimum_speed_mps: float
    maximum_speed_mps: float
    maximum_lateral_acceleration_mps2: float
    braking_deceleration_mps2: float
    speed_preview_distance_m: float
    speed_preview_step_m: float
    track_margin_m: float
    projection_backward_segments: int
    projection_forward_segments: int


@dataclass(frozen=True, slots=True)
class WeightConfig:
    """Non-negative stage and terminal objective weights."""

    lateral_error: float
    heading_error: float
    speed_error: float
    steering_reference: float
    acceleration: float
    steering_change: float
    acceleration_change: float
    terminal_lateral_error: float
    terminal_heading_error: float
    terminal_speed_error: float


@dataclass(frozen=True, slots=True)
class SolverConfig:
    """Bounded IPOPT settings and independent primal-feasibility tolerance."""

    maximum_iterations: int
    tolerance: float
    acceptable_tolerance: float
    acceptable_iterations: int
    maximum_wall_time_s: float
    feasibility_tolerance: float


@dataclass(frozen=True, slots=True)
class FeedbackConfig:
    """Shared public-geometry feedback gains for references, guesses, and fallback."""

    lateral_error_gain: float
    heading_error_gain: float
    speed_error_gain: float


@dataclass(frozen=True, slots=True)
class DebugConfig:
    """Renderer-only prediction sampling."""

    prediction_stride: int


@dataclass(frozen=True, slots=True)
class MpcControllerConfig:
    """Strict immutable configuration for one MPC Controller episode."""

    horizon: HorizonConfig
    planning: PlanningConfig
    weights: WeightConfig
    solver: SolverConfig
    feedback: FeedbackConfig
    debug: DebugConfig

    @classmethod
    def from_public_config(cls, config: Mapping[str, Any]) -> MpcControllerConfig:
        """Parse only the plugin-owned subtree of immutable public configuration."""

        public = _mapping(config, "public config")
        plugin = _mapping(public.get("controller"), "controller")
        _exact_keys(
            plugin,
            {
                "name",
                "description",
                "schema_version",
                "horizon",
                "planning",
                "weights",
                "solver",
                "feedback",
                "debug",
            },
            "controller",
        )
        if plugin["name"] != "mpc":
            raise MpcConfigurationError("controller.name must be 'mpc'")
        if not isinstance(plugin["description"], str) or not plugin["description"]:
            raise MpcConfigurationError("controller.description must be a non-empty string")
        if type(plugin["schema_version"]) is not int or plugin["schema_version"] != 1:
            raise MpcConfigurationError("controller.schema_version must be 1")

        horizon = _mapping(plugin["horizon"], "controller.horizon")
        _exact_keys(horizon, {"steps"}, "controller.horizon")
        horizon_config = HorizonConfig(steps=_positive_integer(horizon, "steps", "horizon"))
        if horizon_config.steps != 20:
            raise MpcConfigurationError("horizon.steps must be 20 for the v0.1 MPC")

        planning = _mapping(plugin["planning"], "controller.planning")
        _exact_keys(
            planning,
            {
                "minimum_speed_mps",
                "maximum_speed_mps",
                "maximum_lateral_acceleration_mps2",
                "braking_deceleration_mps2",
                "speed_preview_distance_m",
                "speed_preview_step_m",
                "track_margin_m",
                "projection_backward_segments",
                "projection_forward_segments",
            },
            "controller.planning",
        )
        planning_config = PlanningConfig(
            minimum_speed_mps=_positive(planning, "minimum_speed_mps", "planning"),
            maximum_speed_mps=_positive(planning, "maximum_speed_mps", "planning"),
            maximum_lateral_acceleration_mps2=_positive(
                planning, "maximum_lateral_acceleration_mps2", "planning"
            ),
            braking_deceleration_mps2=_positive(planning, "braking_deceleration_mps2", "planning"),
            speed_preview_distance_m=_positive(planning, "speed_preview_distance_m", "planning"),
            speed_preview_step_m=_positive(planning, "speed_preview_step_m", "planning"),
            track_margin_m=_number(planning, "track_margin_m", "planning"),
            projection_backward_segments=_positive_integer(
                planning, "projection_backward_segments", "planning"
            ),
            projection_forward_segments=_positive_integer(
                planning, "projection_forward_segments", "planning"
            ),
        )
        if planning_config.minimum_speed_mps > planning_config.maximum_speed_mps:
            raise MpcConfigurationError("minimum_speed_mps cannot exceed maximum_speed_mps")
        if planning_config.speed_preview_step_m > planning_config.speed_preview_distance_m:
            raise MpcConfigurationError(
                "speed_preview_step_m cannot exceed speed_preview_distance_m"
            )

        weights = _mapping(plugin["weights"], "controller.weights")
        weight_keys = {
            "lateral_error",
            "heading_error",
            "speed_error",
            "steering_reference",
            "acceleration",
            "steering_change",
            "acceleration_change",
            "terminal_lateral_error",
            "terminal_heading_error",
            "terminal_speed_error",
        }
        _exact_keys(weights, weight_keys, "controller.weights")
        weight_config = WeightConfig(
            **{key: _number(weights, key, "weights") for key in weight_keys}
        )
        if not any(getattr(weight_config, key) > 0.0 for key in weight_keys):
            raise MpcConfigurationError("at least one MPC objective weight must be positive")

        solver = _mapping(plugin["solver"], "controller.solver")
        _exact_keys(
            solver,
            {
                "maximum_iterations",
                "tolerance",
                "acceptable_tolerance",
                "acceptable_iterations",
                "maximum_wall_time_s",
                "feasibility_tolerance",
            },
            "controller.solver",
        )
        solver_config = SolverConfig(
            maximum_iterations=_positive_integer(solver, "maximum_iterations", "solver"),
            tolerance=_positive(solver, "tolerance", "solver"),
            acceptable_tolerance=_positive(solver, "acceptable_tolerance", "solver"),
            acceptable_iterations=_positive_integer(solver, "acceptable_iterations", "solver"),
            maximum_wall_time_s=_positive(solver, "maximum_wall_time_s", "solver"),
            feasibility_tolerance=_positive(solver, "feasibility_tolerance", "solver"),
        )
        if solver_config.tolerance > solver_config.acceptable_tolerance:
            raise MpcConfigurationError("solver.tolerance cannot exceed acceptable_tolerance")

        feedback = _mapping(plugin["feedback"], "controller.feedback")
        feedback_keys = {"lateral_error_gain", "heading_error_gain", "speed_error_gain"}
        _exact_keys(feedback, feedback_keys, "controller.feedback")
        feedback_config = FeedbackConfig(
            **{key: _number(feedback, key, "feedback") for key in feedback_keys}
        )

        debug = _mapping(plugin["debug"], "controller.debug")
        _exact_keys(debug, {"prediction_stride"}, "controller.debug")
        return cls(
            horizon=horizon_config,
            planning=planning_config,
            weights=weight_config,
            solver=solver_config,
            feedback=feedback_config,
            debug=DebugConfig(
                prediction_stride=_positive_integer(debug, "prediction_stride", "debug")
            ),
        )


@dataclass(frozen=True, slots=True)
class HorizonReference:
    """Read-only horizon parameters and world-frame debug geometry."""

    offsets_m: NDArray[np.float64]
    curvature_1pm: NDArray[np.float64]
    target_speed_mps: NDArray[np.float64]
    effective_half_width_m: NDArray[np.float64]
    center_m: NDArray[np.float64]
    tangent: NDArray[np.float64]

    def __post_init__(self) -> None:
        for name in (
            "offsets_m",
            "curvature_1pm",
            "target_speed_mps",
            "effective_half_width_m",
            "center_m",
            "tangent",
        ):
            value = np.array(getattr(self, name), dtype=np.float64, copy=True)
            if not np.isfinite(value).all():
                raise ValueError(f"{name} must contain only finite values")
            value.setflags(write=False)
            object.__setattr__(self, name, value)


def build_horizon_reference(
    reference: CenterlineReference,
    *,
    start_s_m: float,
    dt_s: float,
    steps: int,
    vehicle_width_m: float,
    config: PlanningConfig,
) -> HorizonReference:
    """Build fixed-shape Frenet parameters from public geometry only."""

    scalars = np.asarray((start_s_m, dt_s, vehicle_width_m), dtype=np.float64)
    if not np.isfinite(scalars).all() or dt_s <= 0.0 or vehicle_width_m <= 0.0:
        raise ValueError("horizon start, positive dt, and vehicle width must be finite")
    if isinstance(steps, bool) or not isinstance(steps, int) or steps < 1:
        raise ValueError("steps must be a positive integer")

    preview_count = math.ceil(config.speed_preview_distance_m / config.speed_preview_step_m)
    speed_offsets = np.linspace(
        0.0,
        config.speed_preview_distance_m,
        preview_count + 1,
        dtype=np.float64,
    )
    speed_sample = reference.preview(start_s_m, speed_offsets)
    speed_profile = curvature_speed_profile(
        speed_sample.curvature_1pm,
        speed_offsets,
        minimum_speed_mps=config.minimum_speed_mps,
        maximum_speed_mps=config.maximum_speed_mps,
        maximum_lateral_acceleration_mps2=config.maximum_lateral_acceleration_mps2,
        braking_deceleration_mps2=config.braking_deceleration_mps2,
    )

    offsets = np.arange(steps + 1, dtype=np.float64) * config.maximum_speed_mps * dt_s
    for _ in range(2):
        target_speed = np.interp(offsets, speed_offsets, speed_profile)
        increments = 0.5 * (target_speed[:-1] + target_speed[1:]) * dt_s
        offsets[0] = 0.0
        offsets[1:] = np.cumsum(increments, dtype=np.float64)

    sample = reference.preview(start_s_m, offsets)
    target_speed = np.interp(offsets, speed_offsets, speed_profile)
    left_half_width = np.linalg.norm(sample.left_boundary_m - sample.center_m, axis=1)
    right_half_width = np.linalg.norm(sample.right_boundary_m - sample.center_m, axis=1)
    effective_half_width = (
        np.minimum(left_half_width, right_half_width)
        - 0.5 * vehicle_width_m
        - config.track_margin_m
    )
    if np.any(effective_half_width <= 0.0):
        raise ValueError("public Track boundaries leave no positive MPC effective half-width")

    return HorizonReference(
        offsets_m=offsets,
        curvature_1pm=np.asarray(sample.curvature_1pm, dtype=np.float64),
        target_speed_mps=target_speed,
        effective_half_width_m=effective_half_width,
        center_m=sample.center_m,
        tangent=sample.tangent,
    )


def deterministic_fallback_action(
    *,
    lateral_error_m: float,
    heading_error_rad: float,
    speed_mps: float,
    curvature_1pm: float,
    target_speed_mps: float,
    wheelbase_m: float,
    config: FeedbackConfig,
) -> NDArray[np.float64]:
    """Return an unconstrained kinematic feedforward/feedback fallback request."""

    values = np.asarray(
        (
            lateral_error_m,
            heading_error_rad,
            speed_mps,
            curvature_1pm,
            target_speed_mps,
            wheelbase_m,
        ),
        dtype=np.float64,
    )
    if not np.isfinite(values).all() or wheelbase_m <= 0.0:
        raise ValueError("fallback inputs must be finite and wheelbase must be positive")
    steering = (
        math.atan(wheelbase_m * curvature_1pm)
        - config.lateral_error_gain * lateral_error_m
        - config.heading_error_gain * heading_error_rad
    )
    acceleration = config.speed_error_gain * (target_speed_mps - speed_mps)
    return np.asarray((steering, acceleration), dtype=np.float64)


__all__ = [
    "DebugConfig",
    "FeedbackConfig",
    "HorizonConfig",
    "HorizonReference",
    "MpcConfigurationError",
    "MpcControllerConfig",
    "PlanningConfig",
    "SolverConfig",
    "WeightConfig",
    "build_horizon_reference",
    "deterministic_fallback_action",
]
