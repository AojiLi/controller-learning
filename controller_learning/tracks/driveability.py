"""Internal low-speed driveability policy for offline track admission.

This module is deliberately not a public Controller or a ranking entry.  It provides a small,
deterministic, pure-JAX reference policy that can exercise generated geometry against the formal
four-wheel plant before a track is admitted to a pool.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import jax
import jax.numpy as jnp

from controller_learning.envs.race_core import TrackProjection, wrap_angle
from controller_learning.physics.mjx_warp import MjxWarpVehicleStateView
from controller_learning.tracks.types import TrackBatch


@dataclass(frozen=True, slots=True)
class ConservativeDriveabilityPolicyConfig:
    """Static tuning for the internal low-speed geometric reference policy."""

    target_speed_mps: float = 4.0
    minimum_corner_speed_mps: float = 2.5
    maximum_lateral_acceleration_mps2: float = 1.5
    speed_proportional_gain: float = 1.5
    wheelbase_m: float = 2.7
    lookahead_base_m: float = 3.0
    lookahead_speed_gain_s: float = 0.7
    maximum_lookahead_m: float = 8.0
    lookahead_search_points: int = 16
    curvature_preview_points: int = 32
    heading_error_gain: float = 0.15
    yaw_rate_damping_s: float = 0.08
    maximum_steering_angle_rad: float = 0.6
    maximum_acceleration_mps2: float = 4.0
    maximum_deceleration_mps2: float = 8.0

    def __post_init__(self) -> None:
        positive = (
            "target_speed_mps",
            "minimum_corner_speed_mps",
            "maximum_lateral_acceleration_mps2",
            "speed_proportional_gain",
            "wheelbase_m",
            "lookahead_base_m",
            "lookahead_speed_gain_s",
            "maximum_lookahead_m",
            "maximum_steering_angle_rad",
            "maximum_acceleration_mps2",
            "maximum_deceleration_mps2",
        )
        for name in positive:
            value = getattr(self, name)
            if not math.isfinite(value) or value <= 0.0:
                raise ValueError(f"{name} must be finite and positive")
        if self.minimum_corner_speed_mps > self.target_speed_mps:
            raise ValueError("minimum_corner_speed_mps cannot exceed target_speed_mps")
        if self.lookahead_base_m > self.maximum_lookahead_m:
            raise ValueError("lookahead_base_m cannot exceed maximum_lookahead_m")
        for name in ("lookahead_search_points", "curvature_preview_points"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value < 1:
                raise ValueError(f"{name} must be a positive integer")
        for name in ("heading_error_gain", "yaw_rate_damping_s"):
            value = getattr(self, name)
            if not math.isfinite(value) or value < 0.0:
                raise ValueError(f"{name} must be finite and non-negative")


def _gather_world(values: jax.Array, indices: jax.Array) -> jax.Array:
    worlds = jnp.arange(values.shape[0], dtype=jnp.int32)[:, None]
    return values[worlds, indices]


def conservative_driveability_action(
    track_batch: TrackBatch,
    projection: TrackProjection,
    vehicle_state: MjxWarpVehicleStateView,
    config: ConservativeDriveabilityPolicyConfig,
) -> jax.Array:
    """Return bounded ``[steering, acceleration]`` actions for every track world.

    The policy uses only fixed-capacity public track geometry, the public Race Core projection, and
    the public physical vehicle state.  Local pure-pursuit steering is combined with a conservative
    curvature-based speed cap.  No simulator internals, generator control points, or hidden retry
    state are used.
    """

    centerline = jnp.asarray(track_batch.centerline_m)
    curvature = jnp.asarray(track_batch.curvature_1pm)
    segment_count = jnp.asarray(track_batch.point_count, dtype=jnp.int32) - 1
    current_segment = jnp.asarray(projection.segment_index, dtype=jnp.int32)
    position = jnp.asarray(vehicle_state.position_world_m)[..., :2]
    yaw = jnp.asarray(vehicle_state.yaw_rad)
    forward_speed = jnp.maximum(jnp.asarray(vehicle_state.velocity_body_mps)[..., 0], 0.0)

    lookahead_offsets = jnp.arange(1, config.lookahead_search_points + 1, dtype=jnp.int32)
    lookahead_indices = jnp.mod(
        current_segment[:, None] + lookahead_offsets[None, :],
        segment_count[:, None],
    )
    lookahead_points = _gather_world(centerline, lookahead_indices)
    point_distance = jnp.linalg.norm(lookahead_points - position[:, None, :], axis=-1)
    desired_lookahead = jnp.clip(
        config.lookahead_base_m + config.lookahead_speed_gain_s * forward_speed,
        config.lookahead_base_m,
        config.maximum_lookahead_m,
    )
    selected_offset = jnp.argmin(
        jnp.abs(point_distance - desired_lookahead[:, None]),
        axis=1,
    )
    worlds = jnp.arange(centerline.shape[0], dtype=jnp.int32)
    target_point = lookahead_points[worlds, selected_offset]
    target_distance = jnp.maximum(point_distance[worlds, selected_offset], 0.1)
    target_vector = target_point - position
    target_heading = jnp.arctan2(target_vector[:, 1], target_vector[:, 0])
    target_index = lookahead_indices[worlds, selected_offset]
    target_tangent = _gather_world(
        jnp.asarray(track_batch.tangent),
        target_index[:, None],
    )[:, 0]
    tangent_heading = jnp.arctan2(target_tangent[:, 1], target_tangent[:, 0])

    alpha = wrap_angle(target_heading - yaw)
    heading_error = wrap_angle(tangent_heading - yaw)
    pure_pursuit = jnp.arctan2(
        2.0 * config.wheelbase_m * jnp.sin(alpha),
        target_distance,
    )
    yaw_rate = jnp.asarray(vehicle_state.angular_velocity_body_rad_s)[..., 2]
    steering = (
        pure_pursuit
        + config.heading_error_gain * heading_error
        - config.yaw_rate_damping_s * yaw_rate
    )
    steering = jnp.clip(
        steering,
        -config.maximum_steering_angle_rad,
        config.maximum_steering_angle_rad,
    )

    preview_offsets = jnp.arange(config.curvature_preview_points, dtype=jnp.int32)
    preview_indices = jnp.mod(
        current_segment[:, None] + preview_offsets[None, :],
        segment_count[:, None],
    )
    preview_curvature = jnp.abs(_gather_world(curvature, preview_indices))
    maximum_preview_curvature = jnp.max(preview_curvature, axis=1)
    curvature_speed = jnp.sqrt(
        config.maximum_lateral_acceleration_mps2
        / jnp.maximum(maximum_preview_curvature, jnp.asarray(1.0e-4, curvature.dtype))
    )
    desired_speed = jnp.clip(
        curvature_speed,
        config.minimum_corner_speed_mps,
        config.target_speed_mps,
    )
    acceleration = config.speed_proportional_gain * (desired_speed - forward_speed)
    acceleration = jnp.clip(
        acceleration,
        -config.maximum_deceleration_mps2,
        config.maximum_acceleration_mps2,
    )
    return jnp.stack((steering, acceleration), axis=1)


__all__ = [
    "ConservativeDriveabilityPolicyConfig",
    "conservative_driveability_action",
]
