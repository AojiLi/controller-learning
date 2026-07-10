"""Tests for immutable Track pools and device-native selection helpers."""

from __future__ import annotations

from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from controller_learning.config import load_project_config
from controller_learning.tracks.generator import generate_track_candidate, pack_track
from controller_learning.tracks.pool import (
    TrackPool,
    gather_track_batch,
    masked_replace_track_batch,
    track_pool_indices,
)
from controller_learning.tracks.specs import (
    generation_spec_from_project,
    track_capacity_from_project,
)
from controller_learning.tracks.types import TrackSchemaError

PROJECT_ROOT = Path(__file__).parents[3]


@pytest.fixture(scope="module")
def tracks():
    project = load_project_config(PROJECT_ROOT)
    generation = generation_spec_from_project(project)
    capacity = track_capacity_from_project(project)
    return tuple(
        pack_track(generate_track_candidate(seed, generation), capacity) for seed in (3, 7, 11)
    )


def test_track_pool_owns_validated_immutable_arrays(tracks) -> None:
    pool = TrackPool.from_tracks(tracks, benchmark_version="0.1", split="train")

    assert pool.size == 3
    assert pool.capacity == tracks[0].capacity
    assert pool.generator_version == "v0.1"
    assert pool.batch.seed.tolist() == [3, 7, 11]
    assert not pool.batch.centerline_m.flags.writeable
    with pytest.raises(ValueError, match="read-only"):
        pool.batch.centerline_m[0, 0, 0] = 1.0


def test_track_pool_rejects_bad_metadata_and_numerical_schema(tracks) -> None:
    valid = TrackPool.from_tracks(tracks, benchmark_version="0.1", split="train")
    with pytest.raises(ValueError, match="split"):
        TrackPool(
            benchmark_version="0.1",
            generator_version="v0.1",
            split="unknown",  # type: ignore[arg-type]
            batch=valid.batch,
        )
    with pytest.raises(TrackSchemaError, match=r"seed.*dtype"):
        TrackPool(
            benchmark_version="0.1",
            generator_version="v0.1",
            split="train",
            batch=valid.batch._replace(seed=valid.batch.seed.astype(np.int64)),
        )
    with pytest.raises(TrackSchemaError, match="seeds must be unique"):
        TrackPool(
            benchmark_version="0.1",
            generator_version="v0.1",
            split="train",
            batch=valid.batch._replace(seed=np.asarray((3, 3, 11), dtype=np.uint32)),
        )


def test_device_selection_gathers_and_masked_replaces_every_leaf(tracks) -> None:
    pool = TrackPool.from_tracks(tracks, benchmark_version="0.1", split="train")
    device_pool = jax.tree.map(jnp.asarray, pool.batch)
    seeds = jnp.asarray((0, 4, 8), dtype=jnp.uint32)
    indices = jax.jit(lambda value: track_pool_indices(value, pool.size))(seeds)
    np.testing.assert_array_equal(indices, (0, 1, 2))

    selected = jax.jit(gather_track_batch)(device_pool, indices)
    np.testing.assert_array_equal(selected.seed, (3, 7, 11))

    current = gather_track_batch(device_pool, jnp.asarray((2, 2, 2), dtype=jnp.int32))
    mask = jnp.asarray((False, True, False), dtype=bool)
    replaced = jax.jit(masked_replace_track_batch)(current, selected, mask)
    host_mask = np.asarray(mask)
    for old, new, result in zip(
        jax.tree.leaves(current),
        jax.tree.leaves(selected),
        jax.tree.leaves(replaced),
        strict=True,
    ):
        np.testing.assert_array_equal(
            np.asarray(result)[~host_mask],
            np.asarray(old)[~host_mask],
        )
        np.testing.assert_array_equal(
            np.asarray(result)[host_mask],
            np.asarray(new)[host_mask],
        )
