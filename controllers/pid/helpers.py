"""Configuration, PID, and speed-planning helpers for the example Controller."""

from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

import numpy as np


class PidConfigurationError(ValueError):
    """Raised when ``controllers/pid/config.toml`` violates its public schema."""


def _mapping(value: object, name: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise PidConfigurationError(f"{name} must be a table")
    return value


def _exact_keys(value: Mapping[str, Any], expected: set[str], name: str) -> None:
    actual = set(value)
    missing = sorted(expected - actual)
    extra = sorted(actual - expected)
    if missing or extra:
        raise PidConfigurationError(f"{name} keys differ; missing={missing}, extra={extra}")


def _number(value: Mapping[str, Any], key: str, name: str, *, minimum: float = 0.0) -> float:
    item = value[key]
    if isinstance(item, bool) or not isinstance(item, (int, float)):
        raise PidConfigurationError(f"{name}.{key} must be a number")
    result = float(item)
    if not math.isfinite(result) or result < minimum:
        raise PidConfigurationError(f"{name}.{key} must be finite and at least {minimum}")
    return result


def _positive(value: Mapping[str, Any], key: str, name: str) -> float:
    result = _number(value, key, name)
    if result <= 0.0:
        raise PidConfigurationError(f"{name}.{key} must be positive")
    return result


def _positive_integer(value: Mapping[str, Any], key: str, name: str) -> int:
    item = value[key]
    if isinstance(item, bool) or not isinstance(item, int) or item < 1:
        raise PidConfigurationError(f"{name}.{key} must be a positive integer")
    return item


@dataclass(frozen=True, slots=True)
class LongitudinalConfig:
    """Curvature speed planner and acceleration-loop parameters."""

    cruise_speed_mps: float
    minimum_corner_speed_mps: float
    maximum_lateral_acceleration_mps2: float
    preview_distance_m: float
    preview_step_m: float
    braking_deceleration_mps2: float
    proportional_gain: float
    integral_gain: float
    derivative_gain: float
    integral_limit: float
    derivative_filter_time_s: float


@dataclass(frozen=True, slots=True)
class LateralConfig:
    """Projection, outer lateral PID, and inner heading-PD parameters."""

    projection_backward_segments: int
    projection_forward_segments: int
    curvature_lookahead_m: float
    lateral_proportional_gain: float
    lateral_integral_gain: float
    lateral_derivative_gain: float
    lateral_integral_limit: float
    maximum_heading_correction_rad: float
    heading_proportional_gain: float
    heading_derivative_gain: float
    curvature_feedforward_gain: float


@dataclass(frozen=True, slots=True)
class DebugConfig:
    """Small bounded set of renderer-only preview parameters."""

    preview_points: int


@dataclass(frozen=True, slots=True)
class PidControllerConfig:
    """Strict immutable configuration for one PID Controller episode."""

    longitudinal: LongitudinalConfig
    lateral: LateralConfig
    debug: DebugConfig

    @classmethod
    def from_public_config(cls, config: Mapping[str, Any]) -> PidControllerConfig:
        """Parse only the plugin-owned subtree of the immutable public config."""

        public = _mapping(config, "public config")
        plugin = _mapping(public.get("controller"), "controller")
        _exact_keys(
            plugin,
            {"name", "description", "schema_version", "longitudinal", "lateral", "debug"},
            "controller",
        )
        if plugin["name"] != "pid":
            raise PidConfigurationError("controller.name must be 'pid'")
        if not isinstance(plugin["description"], str) or not plugin["description"]:
            raise PidConfigurationError("controller.description must be a non-empty string")
        if type(plugin["schema_version"]) is not int or plugin["schema_version"] != 1:
            raise PidConfigurationError("controller.schema_version must be 1")

        longitudinal = _mapping(plugin["longitudinal"], "controller.longitudinal")
        _exact_keys(
            longitudinal,
            {
                "cruise_speed_mps",
                "minimum_corner_speed_mps",
                "maximum_lateral_acceleration_mps2",
                "preview_distance_m",
                "preview_step_m",
                "braking_deceleration_mps2",
                "proportional_gain",
                "integral_gain",
                "derivative_gain",
                "integral_limit",
                "derivative_filter_time_s",
            },
            "controller.longitudinal",
        )
        longitudinal_config = LongitudinalConfig(
            cruise_speed_mps=_positive(longitudinal, "cruise_speed_mps", "longitudinal"),
            minimum_corner_speed_mps=_positive(
                longitudinal, "minimum_corner_speed_mps", "longitudinal"
            ),
            maximum_lateral_acceleration_mps2=_positive(
                longitudinal, "maximum_lateral_acceleration_mps2", "longitudinal"
            ),
            preview_distance_m=_positive(longitudinal, "preview_distance_m", "longitudinal"),
            preview_step_m=_positive(longitudinal, "preview_step_m", "longitudinal"),
            braking_deceleration_mps2=_positive(
                longitudinal, "braking_deceleration_mps2", "longitudinal"
            ),
            proportional_gain=_number(longitudinal, "proportional_gain", "longitudinal"),
            integral_gain=_number(longitudinal, "integral_gain", "longitudinal"),
            derivative_gain=_number(longitudinal, "derivative_gain", "longitudinal"),
            integral_limit=_positive(longitudinal, "integral_limit", "longitudinal"),
            derivative_filter_time_s=_number(
                longitudinal, "derivative_filter_time_s", "longitudinal"
            ),
        )
        if longitudinal_config.minimum_corner_speed_mps > longitudinal_config.cruise_speed_mps:
            raise PidConfigurationError("minimum_corner_speed_mps cannot exceed cruise_speed_mps")
        if longitudinal_config.preview_step_m > longitudinal_config.preview_distance_m:
            raise PidConfigurationError("preview_step_m cannot exceed preview_distance_m")

        lateral = _mapping(plugin["lateral"], "controller.lateral")
        _exact_keys(
            lateral,
            {
                "projection_backward_segments",
                "projection_forward_segments",
                "curvature_lookahead_m",
                "lateral_proportional_gain",
                "lateral_integral_gain",
                "lateral_derivative_gain",
                "lateral_integral_limit",
                "maximum_heading_correction_rad",
                "heading_proportional_gain",
                "heading_derivative_gain",
                "curvature_feedforward_gain",
            },
            "controller.lateral",
        )
        lateral_config = LateralConfig(
            projection_backward_segments=_positive_integer(
                lateral, "projection_backward_segments", "lateral"
            ),
            projection_forward_segments=_positive_integer(
                lateral, "projection_forward_segments", "lateral"
            ),
            curvature_lookahead_m=_number(lateral, "curvature_lookahead_m", "lateral"),
            lateral_proportional_gain=_number(lateral, "lateral_proportional_gain", "lateral"),
            lateral_integral_gain=_number(lateral, "lateral_integral_gain", "lateral"),
            lateral_derivative_gain=_number(lateral, "lateral_derivative_gain", "lateral"),
            lateral_integral_limit=_positive(lateral, "lateral_integral_limit", "lateral"),
            maximum_heading_correction_rad=_positive(
                lateral, "maximum_heading_correction_rad", "lateral"
            ),
            heading_proportional_gain=_number(lateral, "heading_proportional_gain", "lateral"),
            heading_derivative_gain=_number(lateral, "heading_derivative_gain", "lateral"),
            curvature_feedforward_gain=_number(lateral, "curvature_feedforward_gain", "lateral"),
        )
        if lateral_config.maximum_heading_correction_rad >= 0.5 * math.pi:
            raise PidConfigurationError("maximum_heading_correction_rad must be less than pi/2")

        debug = _mapping(plugin["debug"], "controller.debug")
        _exact_keys(debug, {"preview_points"}, "controller.debug")
        return cls(
            longitudinal=longitudinal_config,
            lateral=lateral_config,
            debug=DebugConfig(preview_points=_positive_integer(debug, "preview_points", "debug")),
        )


class SaturatingPid:
    """Stateful PID with conditional integration and an explicit derivative input."""

    __slots__ = ("integral", "integral_limit", "kd", "ki", "kp")

    def __init__(self, *, kp: float, ki: float, kd: float, integral_limit: float) -> None:
        values = np.asarray((kp, ki, kd, integral_limit), dtype=np.float64)
        if not np.isfinite(values).all() or np.any(values < 0.0) or integral_limit <= 0.0:
            raise ValueError(
                "PID gains must be finite and non-negative; integral_limit is positive"
            )
        self.kp = float(kp)
        self.ki = float(ki)
        self.kd = float(kd)
        self.integral_limit = float(integral_limit)
        self.integral = 0.0

    def step(
        self,
        *,
        error: float,
        error_derivative: float,
        dt_s: float,
        lower: float,
        upper: float,
    ) -> float:
        """Return one saturated output and integrate only when it cannot worsen saturation."""

        values = np.asarray((error, error_derivative, dt_s, lower, upper), dtype=np.float64)
        if not np.isfinite(values).all() or dt_s <= 0.0 or lower >= upper:
            raise ValueError("PID step inputs must be finite with dt_s > 0 and lower < upper")
        candidate_integral = float(
            np.clip(self.integral + error * dt_s, -self.integral_limit, self.integral_limit)
        )
        candidate = self.kp * error + self.ki * candidate_integral + self.kd * error_derivative
        worsens_upper = candidate > upper and error > 0.0
        worsens_lower = candidate < lower and error < 0.0
        if not (worsens_upper or worsens_lower):
            self.integral = candidate_integral
        output = self.kp * error + self.ki * self.integral + self.kd * error_derivative
        return float(np.clip(output, lower, upper))


class FilteredDerivative:
    """First-order low-pass estimate of a scalar measurement derivative."""

    __slots__ = ("_derivative", "_previous", "time_constant_s")

    def __init__(self, time_constant_s: float) -> None:
        if not math.isfinite(time_constant_s) or time_constant_s < 0.0:
            raise ValueError("time_constant_s must be finite and non-negative")
        self.time_constant_s = float(time_constant_s)
        self._previous: float | None = None
        self._derivative = 0.0

    def update(self, measurement: float, dt_s: float) -> float:
        """Update the estimate, returning zero on the first sample."""

        if not math.isfinite(measurement) or not math.isfinite(dt_s) or dt_s <= 0.0:
            raise ValueError("measurement and positive dt_s must be finite")
        if self._previous is None:
            self._previous = float(measurement)
            return 0.0
        raw = (measurement - self._previous) / dt_s
        alpha = dt_s / (self.time_constant_s + dt_s)
        self._derivative += alpha * (raw - self._derivative)
        self._previous = float(measurement)
        return self._derivative


__all__ = [
    "FilteredDerivative",
    "LateralConfig",
    "LongitudinalConfig",
    "PidConfigurationError",
    "PidControllerConfig",
    "SaturatingPid",
]
