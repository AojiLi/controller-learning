"""GPU integration tests for fixed-shape Track batches and the pure-JAX Race Core."""

from __future__ import annotations

from typing import NamedTuple

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from controller_learning.envs.race_core import (
    RaceCoreConfig,
    RaceState,
    masked_reset_race_state,
    project_to_track,
    reset_race_state,
    step_race_core,
)
from controller_learning.tracks.generator import (
    TrackGenerationError,
    generate_track_candidate,
    pack_track,
)
from controller_learning.tracks.types import TrackBatch, TrackCapacity, stack_tracks
from controller_learning.tracks.validator import validate_track_candidate

NUM_WORLDS = 1024
CAPACITY = TrackCapacity(max_track_points=640, max_checkpoints=48)
CONFIG = RaceCoreConfig(
    control_dt_s=0.05,
    vehicle_width_m=1.8,
    safety_margin_m=0.2,
    projection_backward_segments=4,
    projection_forward_segments=12,
)
pytestmark = pytest.mark.gpu


class _GpuTrackBatches(NamedTuple):
    first: TrackBatch
    second: TrackBatch


def _collect_accepted_tracks() -> TrackBatch:
    tracks = []
    for seed in range(10_000):
        try:
            candidate = generate_track_candidate(seed)
        except TrackGenerationError:
            continue
        if validate_track_candidate(candidate).valid:
            tracks.append(pack_track(candidate, CAPACITY))
        if len(tracks) == NUM_WORLDS:
            break

    assert len(tracks) == NUM_WORLDS, "fewer than 1024 valid tracks in the deterministic seed scan"
    batch = stack_tracks(tracks)
    assert np.unique(batch.seed).size == NUM_WORLDS
    assert batch.centerline_m.shape == (NUM_WORLDS, CAPACITY.max_track_points, 2)
    assert batch.checkpoint_center_m.shape == (NUM_WORLDS, CAPACITY.max_checkpoints, 2)
    return batch


@pytest.fixture(scope="module")
def gpu_track_batches() -> _GpuTrackBatches:
    devices = jax.devices()
    assert jax.default_backend() == "gpu"
    assert any("nvidia" in device.device_kind.lower() for device in devices)

    host_first = _collect_accepted_tracks()
    permutation = np.roll(np.arange(NUM_WORLDS), 137)
    host_second = jax.tree.map(
        lambda value: np.ascontiguousarray(np.asarray(value)[permutation]),
        host_first,
    )
    first = jax.device_put(host_first)
    second = jax.device_put(host_second)
    jax.block_until_ready((first, second))
    return _GpuTrackBatches(first=first, second=second)


def _segment_samples(
    tracks: TrackBatch,
    segment_index: jax.Array,
    fraction: jax.Array,
    lateral_offset_m: jax.Array | float = 0.0,
) -> jax.Array:
    worlds = jnp.arange(NUM_WORLDS, dtype=jnp.int32)
    starts = tracks.centerline_m[worlds, segment_index]
    ends = tracks.centerline_m[worlds, segment_index + 1]
    vectors = ends - starts
    lengths = jnp.linalg.norm(vectors, axis=1)
    tangent = vectors / lengths[:, None]
    normal = jnp.stack((-tangent[:, 1], tangent[:, 0]), axis=1)
    return (
        starts + jnp.asarray(fraction)[:, None] * vectors + jnp.asarray(lateral_offset_m) * normal
    )


def _state_on_segments(
    tracks: TrackBatch,
    segment_index: jax.Array,
    fraction: jax.Array,
) -> RaceState:
    worlds = jnp.arange(NUM_WORLDS, dtype=jnp.int32)
    positions = _segment_samples(tracks, segment_index, fraction)
    starts = tracks.centerline_m[worlds, segment_index]
    ends = tracks.centerline_m[worlds, segment_index + 1]
    segment_length = jnp.linalg.norm(ends - starts, axis=1)
    projected_s = tracks.cumulative_s_m[worlds, segment_index] + fraction * segment_length
    return RaceState(
        previous_position_m=positions,
        segment_index=segment_index,
        projected_s_m=projected_s,
        unwrapped_s_m=projected_s,
        legal_progress_m=projected_s,
        next_checkpoint_index=jnp.zeros(NUM_WORLDS, dtype=jnp.int32),
        elapsed_steps=jnp.arange(NUM_WORLDS, dtype=jnp.int32) % 100,
    )


def _assert_finite(tree: object) -> None:
    for leaf in jax.tree.leaves(tree):
        array = np.asarray(jax.device_get(leaf))
        if np.issubdtype(array.dtype, np.inexact):
            assert np.isfinite(array).all()


def _replace_worlds(
    current: TrackBatch,
    replacement: TrackBatch,
    mask: jax.Array,
) -> TrackBatch:
    def select(current_value: jax.Array, replacement_value: jax.Array) -> jax.Array:
        selection = mask.reshape((mask.shape[0],) + (1,) * (current_value.ndim - 1))
        return jnp.where(selection, replacement_value, current_value)

    return jax.tree.map(select, current, replacement)


def _translated_tracks(tracks: TrackBatch) -> TrackBatch:
    worlds = jnp.arange(NUM_WORLDS, dtype=jnp.float32)
    offset = jnp.stack((25.0 + 0.01 * worlds, -12.0 + 0.005 * worlds), axis=1)

    def translate_points(values: jax.Array, valid: jax.Array) -> jax.Array:
        translated = values + offset[:, None, :]
        return jnp.where(valid[..., None], translated, jnp.zeros_like(values))

    start_pose = tracks.start_pose.at[:, :2].add(offset)
    return tracks._replace(
        centerline_m=translate_points(tracks.centerline_m, tracks.track_mask),
        left_boundary_m=translate_points(tracks.left_boundary_m, tracks.track_mask),
        right_boundary_m=translate_points(tracks.right_boundary_m, tracks.track_mask),
        checkpoint_center_m=translate_points(
            tracks.checkpoint_center_m,
            tracks.checkpoint_mask,
        ),
        start_pose=start_pose,
    )


def test_1024_different_tracks_reuse_compiled_projection_and_step(
    gpu_track_batches: _GpuTrackBatches,
) -> None:
    first, second = gpu_track_batches
    worlds = jnp.arange(NUM_WORLDS, dtype=jnp.int32)

    def inputs(tracks: TrackBatch):
        segment_count = tracks.point_count - 1
        segment_index = (11 + 37 * worlds) % segment_count
        fraction = 0.2 + 0.6 * ((17 * worlds) % 101).astype(jnp.float32) / 100.0
        positions = _segment_samples(tracks, segment_index, fraction)
        state = _state_on_segments(tracks, segment_index, fraction * 0.25)
        return segment_index, fraction, positions, state

    first_segment, first_fraction, first_position, first_state = inputs(first)
    second_segment, second_fraction, second_position, second_state = inputs(second)

    projection_executable = (
        jax.jit(
            lambda tracks, position, segment: project_to_track(tracks, position, segment, CONFIG)
        )
        .lower(first, first_position, first_segment)
        .compile()
    )
    first_projection = projection_executable(first, first_position, first_segment)
    second_projection = projection_executable(second, second_position, second_segment)

    step_executable = (
        jax.jit(
            lambda tracks, state, position: step_race_core(
                tracks,
                state,
                position,
                jnp.zeros(NUM_WORLDS, dtype=bool),
                CONFIG,
            )
        )
        .lower(first, first_state, first_position)
        .compile()
    )
    first_step = step_executable(first, first_state, first_position)
    second_step = step_executable(second, second_state, second_position)
    jax.block_until_ready((first_projection, second_projection, first_step, second_step))

    for projection, expected_segment in (
        (first_projection, first_segment),
        (second_projection, second_segment),
    ):
        _assert_finite(projection)
        np.testing.assert_array_equal(
            np.asarray(projection.segment_index),
            np.asarray(expected_segment),
        )
        assert float(jnp.max(projection.distance_m)) < 2.0e-5

    for result, expected_segment in (
        (first_step, first_segment),
        (second_step, second_segment),
    ):
        _assert_finite(result)
        np.testing.assert_array_equal(
            np.asarray(result.projection.segment_index),
            np.asarray(expected_segment),
        )
        assert np.all(np.asarray(result.forward_progress_m) > 0.0)
        assert np.unique(np.asarray(result.state.legal_progress_m)).size > NUM_WORLDS // 2
        assert not np.any(np.asarray(result.terminated))
        assert not np.any(np.asarray(result.truncated))

    assert not np.array_equal(np.asarray(first.seed), np.asarray(second.seed))
    assert float(jnp.std(first_fraction)) > 0.1
    assert float(jnp.std(second_fraction)) > 0.1


def test_masked_track_replacement_and_race_reset_preserve_unselected_worlds(
    gpu_track_batches: _GpuTrackBatches,
) -> None:
    first, second = gpu_track_batches
    replacement = _translated_tracks(second)
    worlds = jnp.arange(NUM_WORLDS, dtype=jnp.int32)
    mask = (worlds % 7 == 2) | (worlds % 29 == 5)
    current = RaceState(
        previous_position_m=jnp.stack((0.1 * worlds, -0.05 * worlds), axis=1),
        segment_index=worlds % (first.point_count - 1),
        projected_s_m=0.25 * worlds,
        unwrapped_s_m=0.5 * worlds,
        legal_progress_m=0.125 * worlds,
        next_checkpoint_index=worlds % first.checkpoint_count,
        elapsed_steps=worlds + 3,
    )

    def replace_and_reset(
        current_tracks: TrackBatch,
        new_tracks: TrackBatch,
        current_state: RaceState,
        reset_mask: jax.Array,
    ):
        merged = _replace_worlds(current_tracks, new_tracks, reset_mask)
        reset = reset_race_state(merged)
        return merged, reset, masked_reset_race_state(current_state, reset, reset_mask)

    executable = jax.jit(replace_and_reset).lower(first, replacement, current, mask).compile()
    merged, reset, result = executable(first, replacement, current, mask)
    jax.block_until_ready((merged, reset, result))
    host_mask = np.asarray(mask)

    for merged_leaf, current_leaf, replacement_leaf in zip(
        jax.tree.leaves(merged),
        jax.tree.leaves(first),
        jax.tree.leaves(replacement),
        strict=True,
    ):
        merged_value = np.asarray(merged_leaf)
        np.testing.assert_array_equal(
            merged_value[~host_mask], np.asarray(current_leaf)[~host_mask]
        )
        np.testing.assert_array_equal(
            merged_value[host_mask],
            np.asarray(replacement_leaf)[host_mask],
        )

    for field in RaceState._fields:
        result_value = np.asarray(getattr(result, field))
        current_value = np.asarray(getattr(current, field))
        reset_value = np.asarray(getattr(reset, field))
        np.testing.assert_array_equal(result_value[~host_mask], current_value[~host_mask])
        np.testing.assert_array_equal(result_value[host_mask], reset_value[host_mask])

    selected_start = np.asarray(result.previous_position_m)[host_mask]
    np.testing.assert_array_equal(selected_start, np.asarray(merged.start_pose)[host_mask, :2])
    assert np.all(np.linalg.norm(selected_start, axis=1) > 20.0)


def _synthetic_positions(tracks: TrackBatch, *, seed: int, steps: int) -> jax.Array:
    rng = np.random.default_rng(seed)
    speed = rng.integers(1, 4, size=NUM_WORLDS, dtype=np.int32)
    segments = np.arange(1, steps + 1, dtype=np.int32)[:, None] * speed[None, :]
    fraction = rng.uniform(0.2, 0.8, size=(steps, NUM_WORLDS)).astype(np.float32)
    lateral = rng.uniform(-0.08, 0.08, size=(steps, NUM_WORLDS)).astype(np.float32)

    worlds = jnp.arange(NUM_WORLDS, dtype=jnp.int32)[None, :]
    segment_index = jnp.asarray(segments)
    starts = tracks.centerline_m[worlds, segment_index]
    ends = tracks.centerline_m[worlds, segment_index + 1]
    vectors = ends - starts
    tangent = vectors / jnp.linalg.norm(vectors, axis=2, keepdims=True)
    normal = jnp.stack((-tangent[..., 1], tangent[..., 0]), axis=2)
    return (
        starts
        + jnp.asarray(fraction)[..., None] * vectors
        + jnp.asarray(lateral)[..., None] * normal
    )


def test_short_randomized_rollout_has_no_cross_world_contamination(
    gpu_track_batches: _GpuTrackBatches,
) -> None:
    first, second = gpu_track_batches
    steps = 16
    positions = _synthetic_positions(first, seed=20260710, steps=steps)
    replacement = _translated_tracks(second)
    alternate_positions = _synthetic_positions(replacement, seed=20260711, steps=steps)
    target_world = 731
    mask = jnp.arange(NUM_WORLDS, dtype=jnp.int32) == target_world
    perturbed_tracks = _replace_worlds(first, replacement, mask)
    perturbed_positions = positions.at[:, target_world].set(alternate_positions[:, target_world])

    def rollout(tracks: TrackBatch, position_sequence: jax.Array):
        def advance(state: RaceState, position: jax.Array):
            result = step_race_core(
                tracks,
                state,
                position,
                jnp.zeros(NUM_WORLDS, dtype=bool),
                CONFIG,
            )
            history = (
                result.reward,
                result.projection.segment_index,
                result.projection.distance_m,
                result.termination_reason,
            )
            return result.state, history

        return jax.lax.scan(advance, reset_race_state(tracks), position_sequence)

    executable = jax.jit(rollout).lower(first, positions).compile()
    baseline_state, baseline_history = executable(first, positions)
    perturbed_state, perturbed_history = executable(perturbed_tracks, perturbed_positions)
    jax.block_until_ready((baseline_state, baseline_history, perturbed_state, perturbed_history))
    keep = np.arange(NUM_WORLDS) != target_world

    _assert_finite((baseline_state, baseline_history, perturbed_state, perturbed_history))
    for baseline_leaf, perturbed_leaf in zip(
        jax.tree.leaves(baseline_state),
        jax.tree.leaves(perturbed_state),
        strict=True,
    ):
        np.testing.assert_array_equal(
            np.asarray(baseline_leaf)[keep],
            np.asarray(perturbed_leaf)[keep],
        )
    for baseline_leaf, perturbed_leaf in zip(
        jax.tree.leaves(baseline_history),
        jax.tree.leaves(perturbed_history),
        strict=True,
    ):
        np.testing.assert_array_equal(
            np.asarray(baseline_leaf)[:, keep],
            np.asarray(perturbed_leaf)[:, keep],
        )

    assert not np.array_equal(
        np.asarray(baseline_state.previous_position_m)[target_world],
        np.asarray(perturbed_state.previous_position_m)[target_world],
    )
    assert float(jnp.max(baseline_history[2])) < 0.081
    assert not np.any(np.asarray(baseline_history[3]))
