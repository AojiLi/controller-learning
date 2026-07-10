"""Educational curvature-aware PID Controller using only public Challenge inputs."""

from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

import numpy as np
from numpy.typing import NDArray

from controller_learning.control import (
    CenterlineReference,
    Controller,
    DebugDraw,
    body_to_world,
    curvature_speed_profile,
    wrap_angle,
)

from .helpers import (
    FilteredDerivative,
    PidControllerConfig,
    SaturatingPid,
)


def _public_table(config: Mapping[str, Any], key: str) -> Mapping[str, Any]:
    value = config.get(key)
    if not isinstance(value, Mapping):
        raise ValueError(f"public config field {key!r} must be a table")
    return value


def _finite_positive(value: object, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{name} must be a number")
    result = float(value)
    if not math.isfinite(result) or result <= 0.0:
        raise ValueError(f"{name} must be finite and positive")
    return result


@dataclass(frozen=True, slots=True)
class _DebugState:
    position_m: NDArray[np.float64]
    projected_point_m: NDArray[np.float64]
    reference_points_m: NDArray[np.float64]
    target_point_m: NDArray[np.float64]
    target_speed_mps: float
    lateral_error_m: float
    heading_error_rad: float


class PidController(Controller):
    """Track speed and centerline with longitudinal PID and cascaded lateral/heading loops."""

    def __init__(
        self,
        obs: Mapping[str, Any],
        info: Mapping[str, Any],
        config: Mapping[str, Any],
    ) -> None:
        del info
        self._reference = CenterlineReference.from_observation(obs)
        self._parameters = PidControllerConfig.from_public_config(config)
        self._dt_s = _finite_positive(config.get("control_dt_s"), "control_dt_s")

        vehicle = _public_table(config, "vehicle")
        action_limits = _public_table(config, "action_limits")
        self._wheelbase_m = _finite_positive(vehicle.get("wheelbase_m"), "wheelbase_m")
        self._maximum_speed_mps = _finite_positive(vehicle.get("max_speed_mps"), "max_speed_mps")
        self._maximum_steering_rad = _finite_positive(
            action_limits.get("max_steering_angle_rad"), "max_steering_angle_rad"
        )
        self._maximum_steering_rate_rad_s = _finite_positive(
            action_limits.get("max_steering_rate_rad_s"), "max_steering_rate_rad_s"
        )
        self._maximum_acceleration_mps2 = _finite_positive(
            action_limits.get("max_acceleration_mps2"), "max_acceleration_mps2"
        )
        self._maximum_deceleration_mps2 = _finite_positive(
            action_limits.get("max_deceleration_mps2"), "max_deceleration_mps2"
        )

        longitudinal = self._parameters.longitudinal
        if longitudinal.cruise_speed_mps > self._maximum_speed_mps:
            raise ValueError("PID cruise speed exceeds the public vehicle speed limit")
        self._speed_pid = SaturatingPid(
            kp=longitudinal.proportional_gain,
            ki=longitudinal.integral_gain,
            kd=longitudinal.derivative_gain,
            integral_limit=longitudinal.integral_limit,
        )
        self._speed_derivative = FilteredDerivative(longitudinal.derivative_filter_time_s)

        lateral = self._parameters.lateral
        self._lateral_pid = SaturatingPid(
            kp=lateral.lateral_proportional_gain,
            ki=lateral.lateral_integral_gain,
            kd=lateral.lateral_derivative_gain,
            integral_limit=lateral.lateral_integral_limit,
        )
        self._segment_hint: int | None = None
        self._debug: _DebugState | None = None

        preview_count = math.floor(longitudinal.preview_distance_m / longitudinal.preview_step_m)
        self._speed_preview_offsets_m = np.linspace(
            0.0,
            preview_count * longitudinal.preview_step_m,
            preview_count + 1,
            dtype=np.float64,
        )
        self._debug_preview_offsets_m = np.linspace(
            0.0,
            longitudinal.preview_distance_m,
            self._parameters.debug.preview_points,
            dtype=np.float64,
        )

    def compute_control(
        self,
        obs: Mapping[str, Any],
        info: Mapping[str, Any] | None = None,
    ) -> NDArray[np.float32]:
        """Return bounded steering angle and longitudinal acceleration in physical units."""

        del info
        position = np.asarray(obs["position"], dtype=np.float64)
        velocity_body = np.asarray(obs["velocity_body"], dtype=np.float64)
        yaw = float(np.asarray(obs["yaw"], dtype=np.float64))
        yaw_rate = float(np.asarray(obs["yaw_rate"], dtype=np.float64))
        measured_steering = float(np.asarray(obs["steering_angle"], dtype=np.float64))
        if (
            position.shape != (2,)
            or velocity_body.shape != (2,)
            or not np.isfinite(position).all()
            or not np.isfinite(velocity_body).all()
            or not np.isfinite((yaw, yaw_rate, measured_steering)).all()
        ):
            raise ValueError("PID Controller received a non-finite or malformed vehicle state")

        lateral = self._parameters.lateral
        projection = self._reference.project(
            position,
            hint_segment=self._segment_hint,
            backward_segments=lateral.projection_backward_segments,
            forward_segments=lateral.projection_forward_segments,
        )
        self._segment_hint = projection.segment_index

        longitudinal = self._parameters.longitudinal
        speed_preview = self._reference.sample(projection.s_m + self._speed_preview_offsets_m)
        target_speed_profile = curvature_speed_profile(
            speed_preview.curvature_1pm,
            self._speed_preview_offsets_m,
            minimum_speed_mps=longitudinal.minimum_corner_speed_mps,
            maximum_speed_mps=longitudinal.cruise_speed_mps,
            maximum_lateral_acceleration_mps2=(longitudinal.maximum_lateral_acceleration_mps2),
            braking_deceleration_mps2=longitudinal.braking_deceleration_mps2,
        )
        target_speed = float(target_speed_profile[0])
        forward_speed = float(velocity_body[0])
        speed_error = target_speed - forward_speed
        speed_measurement_derivative = self._speed_derivative.update(forward_speed, self._dt_s)
        acceleration = self._speed_pid.step(
            error=speed_error,
            error_derivative=-speed_measurement_derivative,
            dt_s=self._dt_s,
            lower=-self._maximum_deceleration_mps2,
            upper=self._maximum_acceleration_mps2,
        )

        velocity_world = np.asarray(body_to_world(velocity_body, yaw), dtype=np.float64)
        normal = np.asarray((-projection.tangent[1], projection.tangent[0]), dtype=np.float64)
        lateral_velocity = float(np.dot(velocity_world, normal))
        heading_correction = self._lateral_pid.step(
            error=-projection.lateral_error_m,
            error_derivative=-lateral_velocity,
            dt_s=self._dt_s,
            lower=-lateral.maximum_heading_correction_rad,
            upper=lateral.maximum_heading_correction_rad,
        )

        curvature_sample = self._reference.sample(projection.s_m + lateral.curvature_lookahead_m)
        reference_curvature = float(curvature_sample.curvature_1pm)
        reference_heading = math.atan2(float(projection.tangent[1]), float(projection.tangent[0]))
        desired_heading = reference_heading + heading_correction
        heading_error = float(wrap_angle(desired_heading - yaw))
        desired_yaw_rate = max(forward_speed, 0.0) * reference_curvature
        steering_feedforward = lateral.curvature_feedforward_gain * math.atan(
            self._wheelbase_m * reference_curvature
        )
        requested_steering = (
            steering_feedforward
            + lateral.heading_proportional_gain * heading_error
            + lateral.heading_derivative_gain * (desired_yaw_rate - yaw_rate)
        )
        steering_step = self._maximum_steering_rate_rad_s * self._dt_s
        steering = float(
            np.clip(
                requested_steering,
                max(-self._maximum_steering_rad, measured_steering - steering_step),
                min(self._maximum_steering_rad, measured_steering + steering_step),
            )
        )

        debug_sample = self._reference.sample(projection.s_m + self._debug_preview_offsets_m)
        self._debug = _DebugState(
            position_m=np.array(position, copy=True),
            projected_point_m=np.array(projection.point_m, copy=True),
            reference_points_m=np.array(debug_sample.center_m, copy=True),
            target_point_m=np.array(curvature_sample.center_m, copy=True),
            target_speed_mps=target_speed,
            lateral_error_m=projection.lateral_error_m,
            heading_error_rad=heading_error,
        )

        action = np.asarray((steering, acceleration), dtype=np.float32)
        if action.shape != (2,) or not np.isfinite(action).all():
            raise RuntimeError("PID Controller produced an invalid action")
        return action

    def render_callback(self, debug_draw: DebugDraw) -> None:
        """Draw only Controller-derived reference and error information."""

        if self._debug is None:
            return
        state = self._debug
        debug_draw.line(
            state.position_m,
            state.projected_point_m,
            color=(1.0, 0.35, 0.2, 0.9),
            width=1.5,
        )
        debug_draw.points(
            state.reference_points_m,
            color=(0.15, 0.85, 0.35, 0.8),
            size=2.5,
        )
        debug_draw.points((state.target_point_m,), color=(1.0, 0.85, 0.1), size=5.0)
        label_position = state.position_m + np.asarray((2.0, 2.0), dtype=np.float64)
        debug_draw.text(
            label_position,
            (
                f"v*={state.target_speed_mps:.2f} m/s  "
                f"ey={state.lateral_error_m:+.2f} m  "
                f"epsi={state.heading_error_rad:+.2f} rad"
            ),
            color=(0.05, 0.05, 0.05),
        )
