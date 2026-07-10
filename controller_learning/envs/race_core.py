"""Pure-JAX batched race progress, checkpoint, reward, and termination logic."""

from __future__ import annotations

import math
from dataclasses import dataclass
from enum import IntEnum
from typing import NamedTuple

import jax
import jax.numpy as jnp

from controller_learning.tracks.types import TrackBatch


class RaceTermination(IntEnum):
    """Numerical termination reasons suitable for JAX arrays."""

    NONE = 0
    SUCCESS = 1
    OFF_TRACK = 2
    INVALID_ACTION = 3
    TIMEOUT = 4


@dataclass(frozen=True, slots=True)
class RaceCoreConfig:
    """Static rules used by the batched Race Core."""

    control_dt_s: float
    vehicle_width_m: float
    safety_margin_m: float
    projection_backward_segments: int
    projection_forward_segments: int
    min_timeout_s: float = 60.0
    timeout_reference_speed_mps: float = 3.0

    def __post_init__(self) -> None:
        positive_floats = {
            "control_dt_s": self.control_dt_s,
            "vehicle_width_m": self.vehicle_width_m,
            "min_timeout_s": self.min_timeout_s,
            "timeout_reference_speed_mps": self.timeout_reference_speed_mps,
        }
        for name, value in positive_floats.items():
            if not math.isfinite(value) or value <= 0.0:
                raise ValueError(f"{name} must be finite and positive")
        if not math.isfinite(self.safety_margin_m) or self.safety_margin_m < 0.0:
            raise ValueError("safety_margin_m must be finite and non-negative")
        for name, value in (
            ("projection_backward_segments", self.projection_backward_segments),
            ("projection_forward_segments", self.projection_forward_segments),
        ):
            if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                raise ValueError(f"{name} must be a non-negative integer")
        if self.projection_forward_segments < 1:
            raise ValueError("projection_forward_segments must be positive")


class TrackProjection(NamedTuple):
    """Nearest point within the topology-local segment window for every world."""

    segment_index: jax.Array
    segment_fraction: jax.Array
    projected_s_m: jax.Array
    closest_point_m: jax.Array
    tangent: jax.Array
    lateral_error_m: jax.Array
    distance_m: jax.Array


class RaceState(NamedTuple):
    """Per-world race state independent from the physics backend."""

    previous_position_m: jax.Array
    segment_index: jax.Array
    projected_s_m: jax.Array
    unwrapped_s_m: jax.Array
    legal_progress_m: jax.Array
    next_checkpoint_index: jax.Array
    elapsed_steps: jax.Array


class RaceStep(NamedTuple):
    """Batched result of one Race Core transition."""

    state: RaceState
    projection: TrackProjection
    reward: jax.Array
    terminated: jax.Array
    truncated: jax.Array
    termination_reason: jax.Array
    success: jax.Array
    off_track: jax.Array
    invalid_action: jax.Array
    timeout: jax.Array
    checkpoint_crossed: jax.Array
    forward_progress_m: jax.Array
    effective_half_width_m: jax.Array


def wrap_angle(angle_rad: jax.Array) -> jax.Array:
    """Wrap angles to the half-open interval ``[-pi, pi)``."""

    angle = jnp.asarray(angle_rad)
    return jnp.mod(angle + jnp.pi, 2.0 * jnp.pi) - jnp.pi


def world_to_body(vector_world: jax.Array, yaw_rad: jax.Array) -> jax.Array:
    """Rotate planar vectors from world coordinates into body coordinates."""

    vector = jnp.asarray(vector_world)
    yaw = jnp.asarray(yaw_rad)
    cosine = jnp.cos(yaw)
    sine = jnp.sin(yaw)
    return jnp.stack(
        (
            cosine * vector[..., 0] + sine * vector[..., 1],
            -sine * vector[..., 0] + cosine * vector[..., 1],
        ),
        axis=-1,
    )


def body_to_world(vector_body: jax.Array, yaw_rad: jax.Array) -> jax.Array:
    """Rotate planar vectors from body coordinates into world coordinates."""

    vector = jnp.asarray(vector_body)
    yaw = jnp.asarray(yaw_rad)
    cosine = jnp.cos(yaw)
    sine = jnp.sin(yaw)
    return jnp.stack(
        (
            cosine * vector[..., 0] - sine * vector[..., 1],
            sine * vector[..., 0] + cosine * vector[..., 1],
        ),
        axis=-1,
    )


def _gather_world(values: jax.Array, indices: jax.Array) -> jax.Array:
    world_indices = jnp.arange(values.shape[0], dtype=jnp.int32)[:, None]
    return values[world_indices, indices]


def project_to_track(
    track_batch: TrackBatch,
    positions_m: jax.Array,
    prior_segment: jax.Array,
    config: RaceCoreConfig,
) -> TrackProjection:
    """Project rear-axle positions onto a fixed-size, topology-local segment window.

    Only segments near each world's previous segment are considered. This deliberately prevents a
    spatially close hairpin or parallel part of the track from creating a progress shortcut.
    """

    centerline = jnp.asarray(track_batch.centerline_m)
    cumulative_s = jnp.asarray(track_batch.cumulative_s_m)
    segment_count = jnp.asarray(track_batch.point_count, dtype=jnp.int32) - 1
    positions = jnp.asarray(positions_m)
    prior = jnp.mod(jnp.asarray(prior_segment, dtype=jnp.int32), segment_count)

    offsets = jnp.arange(
        -config.projection_backward_segments,
        config.projection_forward_segments + 1,
        dtype=jnp.int32,
    )
    candidate_indices = jnp.mod(prior[:, None] + offsets[None, :], segment_count[:, None])
    starts = _gather_world(centerline, candidate_indices)
    ends = _gather_world(centerline, candidate_indices + 1)
    segment_vectors = ends - starts
    squared_lengths = jnp.sum(segment_vectors * segment_vectors, axis=-1)
    safe_squared_lengths = jnp.maximum(squared_lengths, jnp.finfo(centerline.dtype).tiny)
    fractions = jnp.clip(
        jnp.sum((positions[:, None, :] - starts) * segment_vectors, axis=-1) / safe_squared_lengths,
        0.0,
        1.0,
    )
    closest_points = starts + fractions[..., None] * segment_vectors
    residuals = positions[:, None, :] - closest_points
    squared_distances = jnp.sum(residuals * residuals, axis=-1)

    # Shared endpoints and the explicit closure can produce exact distance ties. Prefer the current
    # segment, then a forward neighbor, then a backward neighbor without perturbing real distances.
    minimum_squared_distance = jnp.min(squared_distances, axis=1, keepdims=True)
    tied = squared_distances <= minimum_squared_distance + 1e-7
    topology_rank = 2 * jnp.abs(offsets) + (offsets < 0).astype(jnp.int32)
    tied_rank = jnp.where(tied, topology_rank[None, :], jnp.iinfo(jnp.int32).max)
    selected = jnp.argmin(tied_rank, axis=1)
    worlds = jnp.arange(centerline.shape[0], dtype=jnp.int32)

    selected_indices = candidate_indices[worlds, selected]
    selected_fractions = fractions[worlds, selected]
    selected_points = closest_points[worlds, selected]
    selected_vectors = segment_vectors[worlds, selected]
    selected_lengths = jnp.sqrt(safe_squared_lengths[worlds, selected])
    selected_tangent = selected_vectors / selected_lengths[:, None]
    selected_residual = positions - selected_points
    lateral_error = (
        selected_tangent[:, 0] * selected_residual[:, 1]
        - selected_tangent[:, 1] * selected_residual[:, 0]
    )
    selected_s = _gather_world(cumulative_s, selected_indices[:, None])[:, 0]
    projected_s = selected_s + selected_fractions * selected_lengths

    return TrackProjection(
        segment_index=selected_indices,
        segment_fraction=selected_fractions,
        projected_s_m=projected_s,
        closest_point_m=selected_points,
        tangent=selected_tangent,
        lateral_error_m=lateral_error,
        distance_m=jnp.linalg.norm(selected_residual, axis=1),
    )


def reset_race_state(track_batch: TrackBatch) -> RaceState:
    """Create the standard-at-rest Race Core state for a batch of tracks."""

    start_positions = jnp.asarray(track_batch.start_pose)[..., :2]
    num_worlds = start_positions.shape[0]
    zeros_float = jnp.zeros(num_worlds, dtype=start_positions.dtype)
    zeros_int = jnp.zeros(num_worlds, dtype=jnp.int32)
    return RaceState(
        previous_position_m=start_positions,
        segment_index=zeros_int,
        projected_s_m=zeros_float,
        unwrapped_s_m=zeros_float,
        legal_progress_m=zeros_float,
        next_checkpoint_index=zeros_int,
        elapsed_steps=zeros_int,
    )


def masked_reset_race_state(
    current: RaceState,
    reset: RaceState,
    mask: jax.Array,
) -> RaceState:
    """Replace every Race Core field only in worlds selected by ``mask``."""

    reset_mask = jnp.asarray(mask, dtype=bool)
    return RaceState(
        previous_position_m=jnp.where(
            reset_mask[:, None],
            reset.previous_position_m,
            current.previous_position_m,
        ),
        segment_index=jnp.where(reset_mask, reset.segment_index, current.segment_index),
        projected_s_m=jnp.where(reset_mask, reset.projected_s_m, current.projected_s_m),
        unwrapped_s_m=jnp.where(reset_mask, reset.unwrapped_s_m, current.unwrapped_s_m),
        legal_progress_m=jnp.where(
            reset_mask,
            reset.legal_progress_m,
            current.legal_progress_m,
        ),
        next_checkpoint_index=jnp.where(
            reset_mask,
            reset.next_checkpoint_index,
            current.next_checkpoint_index,
        ),
        elapsed_steps=jnp.where(reset_mask, reset.elapsed_steps, current.elapsed_steps),
    )


def _checkpoint_crossing(
    track_batch: TrackBatch,
    state: RaceState,
    positions_m: jax.Array,
    candidate_unwrapped_s_m: jax.Array,
    effective_half_width_m: jax.Array,
) -> jax.Array:
    checkpoint_count = jnp.asarray(track_batch.checkpoint_count, dtype=jnp.int32)
    has_checkpoint = state.next_checkpoint_index < checkpoint_count
    checkpoint_index = jnp.minimum(state.next_checkpoint_index, checkpoint_count - 1)
    worlds = jnp.arange(checkpoint_count.shape[0], dtype=jnp.int32)
    centers = jnp.asarray(track_batch.checkpoint_center_m)[worlds, checkpoint_index]
    tangents = jnp.asarray(track_batch.checkpoint_tangent)[worlds, checkpoint_index]
    target_s = jnp.asarray(track_batch.checkpoint_s_m)[worlds, checkpoint_index]

    previous_relative = state.previous_position_m - centers
    current_relative = positions_m - centers
    previous_longitudinal = jnp.sum(previous_relative * tangents, axis=1)
    current_longitudinal = jnp.sum(current_relative * tangents, axis=1)
    denominator = current_longitudinal - previous_longitudinal
    crosses_forward = (previous_longitudinal < 0.0) & (current_longitudinal >= 0.0)
    fraction = jnp.clip(
        -previous_longitudinal / jnp.where(denominator > 0.0, denominator, 1.0),
        0.0,
        1.0,
    )
    crossing_point = state.previous_position_m + fraction[:, None] * (
        positions_m - state.previous_position_m
    )
    normals = jnp.stack((-tangents[:, 1], tangents[:, 0]), axis=1)
    crossing_lateral = jnp.abs(jnp.sum((crossing_point - centers) * normals, axis=1))
    reached_topologically = candidate_unwrapped_s_m >= target_s - 1e-5
    return (
        has_checkpoint
        & crosses_forward
        & (denominator > 0.0)
        & (crossing_lateral <= effective_half_width_m)
        & reached_topologically
    )


def step_race_core(
    track_batch: TrackBatch,
    state: RaceState,
    position_m: jax.Array,
    invalid_action: jax.Array,
    config: RaceCoreConfig,
) -> RaceStep:
    """Advance progress and episode rules for a batch of rear-axle positions."""

    positions = jnp.asarray(position_m)
    invalid = jnp.asarray(invalid_action, dtype=bool)
    track_length = jnp.asarray(track_batch.length_m)
    projection = project_to_track(track_batch, positions, state.segment_index, config)

    raw_delta_s = projection.projected_s_m - state.projected_s_m
    wrapped_delta_s = jnp.mod(raw_delta_s + 0.5 * track_length, track_length) - 0.5 * track_length
    candidate_unwrapped_s = state.unwrapped_s_m + wrapped_delta_s
    candidate_legal_progress = jnp.clip(candidate_unwrapped_s, 0.0, track_length)
    legal_progress = jnp.maximum(state.legal_progress_m, candidate_legal_progress)
    forward_increment = legal_progress - state.legal_progress_m

    effective_half_width = (
        0.5 * jnp.asarray(track_batch.width_m)
        - 0.5 * config.vehicle_width_m
        - config.safety_margin_m
    )
    off_track = (effective_half_width <= 0.0) | (projection.distance_m > effective_half_width)

    checkpoint_crossed = _checkpoint_crossing(
        track_batch,
        state,
        positions,
        candidate_unwrapped_s,
        effective_half_width,
    )
    accepted_checkpoint = checkpoint_crossed & ~invalid & ~off_track
    next_checkpoint_index = state.next_checkpoint_index + accepted_checkpoint.astype(jnp.int32)
    success = accepted_checkpoint & (
        next_checkpoint_index == jnp.asarray(track_batch.checkpoint_count, dtype=jnp.int32)
    )

    elapsed_steps = state.elapsed_steps + jnp.int32(1)
    maximum_time_s = jnp.maximum(
        config.min_timeout_s,
        track_length / config.timeout_reference_speed_mps,
    )
    timeout_steps = jnp.ceil(maximum_time_s / config.control_dt_s).astype(jnp.int32)
    timeout_reached = elapsed_steps >= timeout_steps

    terminated = invalid | off_track | success
    truncated = timeout_reached & ~terminated
    termination = jnp.where(
        invalid,
        jnp.int32(RaceTermination.INVALID_ACTION),
        jnp.where(
            off_track,
            jnp.int32(RaceTermination.OFF_TRACK),
            jnp.where(
                success,
                jnp.int32(RaceTermination.SUCCESS),
                jnp.where(
                    truncated,
                    jnp.int32(RaceTermination.TIMEOUT),
                    jnp.int32(RaceTermination.NONE),
                ),
            ),
        ),
    )
    reward = (
        forward_increment / track_length
        + success.astype(track_length.dtype)
        - (invalid | off_track).astype(track_length.dtype)
    )

    next_state = RaceState(
        previous_position_m=positions,
        segment_index=projection.segment_index,
        projected_s_m=projection.projected_s_m,
        unwrapped_s_m=candidate_unwrapped_s,
        legal_progress_m=legal_progress,
        next_checkpoint_index=next_checkpoint_index,
        elapsed_steps=elapsed_steps,
    )
    return RaceStep(
        state=next_state,
        projection=projection,
        reward=reward,
        terminated=terminated,
        truncated=truncated,
        termination_reason=termination,
        success=success,
        off_track=off_track,
        invalid_action=invalid,
        timeout=truncated,
        checkpoint_crossed=accepted_checkpoint,
        forward_progress_m=forward_increment,
        effective_half_width_m=effective_half_width,
    )


__all__ = [
    "RaceCoreConfig",
    "RaceState",
    "RaceStep",
    "RaceTermination",
    "TrackProjection",
    "body_to_world",
    "masked_reset_race_state",
    "project_to_track",
    "reset_race_state",
    "step_race_core",
    "world_to_body",
    "wrap_angle",
]
