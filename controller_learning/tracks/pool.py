"""Immutable host Track pools and pure-JAX device selection helpers."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import jax
import jax.numpy as jnp
import numpy as np

from controller_learning.tracks.types import (
    Track,
    TrackBatch,
    TrackCapacity,
    TrackSchemaError,
    stack_tracks,
)

TrackPoolSplit = Literal["train", "validation", "test"]

_FIELD_DTYPES = {
    "seed": np.dtype(np.uint32),
    "centerline_m": np.dtype(np.float32),
    "left_boundary_m": np.dtype(np.float32),
    "right_boundary_m": np.dtype(np.float32),
    "tangent": np.dtype(np.float32),
    "curvature_1pm": np.dtype(np.float32),
    "cumulative_s_m": np.dtype(np.float32),
    "track_mask": np.dtype(np.bool_),
    "checkpoint_center_m": np.dtype(np.float32),
    "checkpoint_tangent": np.dtype(np.float32),
    "checkpoint_s_m": np.dtype(np.float32),
    "checkpoint_mask": np.dtype(np.bool_),
    "start_pose": np.dtype(np.float32),
    "point_count": np.dtype(np.int32),
    "checkpoint_count": np.dtype(np.int32),
    "length_m": np.dtype(np.float32),
    "width_m": np.dtype(np.float32),
}


def _validated_host_batch(value: object) -> TrackBatch:
    if not isinstance(value, TrackBatch):
        raise TypeError("TrackPool.batch must be a TrackBatch")
    source = {name: np.asarray(getattr(value, name)) for name in TrackBatch._fields}
    seeds = source["seed"]
    if seeds.ndim != 1 or seeds.size < 1:
        raise TrackSchemaError("TrackPool seed must be a non-empty one-dimensional array")
    size = int(seeds.size)

    centerline = source["centerline_m"]
    checkpoints = source["checkpoint_center_m"]
    if centerline.ndim != 3 or centerline.shape[0] != size or centerline.shape[2] != 2:
        raise TrackSchemaError("TrackPool centerline_m must have shape (size, points, 2)")
    if checkpoints.ndim != 3 or checkpoints.shape[0] != size or checkpoints.shape[2] != 2:
        raise TrackSchemaError(
            "TrackPool checkpoint_center_m must have shape (size, checkpoints, 2)"
        )
    points = int(centerline.shape[1])
    checkpoint_capacity = int(checkpoints.shape[1])
    capacity = TrackCapacity(points, checkpoint_capacity)
    del capacity

    shapes = {
        "seed": (size,),
        "centerline_m": (size, points, 2),
        "left_boundary_m": (size, points, 2),
        "right_boundary_m": (size, points, 2),
        "tangent": (size, points, 2),
        "curvature_1pm": (size, points),
        "cumulative_s_m": (size, points),
        "track_mask": (size, points),
        "checkpoint_center_m": (size, checkpoint_capacity, 2),
        "checkpoint_tangent": (size, checkpoint_capacity, 2),
        "checkpoint_s_m": (size, checkpoint_capacity),
        "checkpoint_mask": (size, checkpoint_capacity),
        "start_pose": (size, 3),
        "point_count": (size,),
        "checkpoint_count": (size,),
        "length_m": (size,),
        "width_m": (size,),
    }
    arrays: dict[str, np.ndarray] = {}
    for name in TrackBatch._fields:
        array = source[name]
        if array.shape != shapes[name]:
            raise TrackSchemaError(
                f"TrackPool {name} must have shape {shapes[name]}, got {array.shape}"
            )
        if array.dtype != _FIELD_DTYPES[name]:
            raise TrackSchemaError(
                f"TrackPool {name} must have dtype {_FIELD_DTYPES[name]}, got {array.dtype}"
            )
        if np.issubdtype(array.dtype, np.floating) and not np.isfinite(array).all():
            raise TrackSchemaError(f"TrackPool {name} must contain only finite values")
        owned = np.array(array, copy=True)
        owned.setflags(write=False)
        arrays[name] = owned

    if np.unique(arrays["seed"]).size != size:
        raise TrackSchemaError("TrackPool seeds must be unique")
    point_count = arrays["point_count"]
    checkpoint_count = arrays["checkpoint_count"]
    if np.any((point_count < 4) | (point_count > points)):
        raise TrackSchemaError("TrackPool point_count must fit the Track capacity")
    if np.any((checkpoint_count < 1) | (checkpoint_count > checkpoint_capacity)):
        raise TrackSchemaError("TrackPool checkpoint_count must fit the checkpoint capacity")
    if np.any(arrays["length_m"] <= 0.0) or np.any(arrays["width_m"] <= 0.0):
        raise TrackSchemaError("TrackPool length_m and width_m must be positive")
    if not np.all(arrays["width_m"] == arrays["width_m"][0]):
        raise TrackSchemaError("TrackPool width_m must be identical for every Track")

    point_prefix = np.arange(points)[None, :] < point_count[:, None]
    checkpoint_prefix = np.arange(checkpoint_capacity)[None, :] < checkpoint_count[:, None]
    if not np.array_equal(arrays["track_mask"], point_prefix):
        raise TrackSchemaError("TrackPool track_mask must match point_count")
    if not np.array_equal(arrays["checkpoint_mask"], checkpoint_prefix):
        raise TrackSchemaError("TrackPool checkpoint_mask must match checkpoint_count")

    for name in (
        "centerline_m",
        "left_boundary_m",
        "right_boundary_m",
        "tangent",
        "curvature_1pm",
        "cumulative_s_m",
    ):
        mask = point_prefix[..., None] if arrays[name].ndim == 3 else point_prefix
        if np.any(np.where(mask, 0, arrays[name]) != 0):
            raise TrackSchemaError(f"TrackPool {name} padding must be zero")
    for name in ("checkpoint_center_m", "checkpoint_tangent", "checkpoint_s_m"):
        mask = checkpoint_prefix[..., None] if arrays[name].ndim == 3 else checkpoint_prefix
        if np.any(np.where(mask, 0, arrays[name]) != 0):
            raise TrackSchemaError(f"TrackPool {name} padding must be zero")

    rows = np.arange(size)
    closing = point_count - 1
    if not np.array_equal(arrays["centerline_m"][:, 0], arrays["centerline_m"][rows, closing]):
        raise TrackSchemaError("every TrackPool centerline must be closed")
    cumulative = arrays["cumulative_s_m"]
    if np.any(cumulative[:, 0] != 0.0):
        raise TrackSchemaError("every TrackPool cumulative_s_m must start at zero")
    valid_differences = np.arange(points - 1)[None, :] < closing[:, None]
    if np.any(valid_differences & (np.diff(cumulative, axis=1) <= 0.0)):
        raise TrackSchemaError("valid TrackPool cumulative_s_m values must increase")
    if not np.allclose(
        cumulative[rows, closing],
        arrays["length_m"],
        rtol=0.0,
        atol=1.0e-4,
    ):
        raise TrackSchemaError("TrackPool closing distance must equal length_m")

    return TrackBatch(**arrays)


@dataclass(frozen=True, slots=True)
class TrackPool:
    """One versioned immutable host pool copied once to the device by the Challenge."""

    benchmark_version: str
    generator_version: str
    split: TrackPoolSplit
    batch: TrackBatch

    def __post_init__(self) -> None:
        if not isinstance(self.benchmark_version, str) or not self.benchmark_version:
            raise ValueError("TrackPool benchmark_version must be a non-empty string")
        if not isinstance(self.generator_version, str) or not self.generator_version:
            raise ValueError("TrackPool generator_version must be a non-empty string")
        if self.split not in ("train", "validation", "test"):
            raise ValueError("TrackPool split must be 'train', 'validation', or 'test'")
        object.__setattr__(self, "batch", _validated_host_batch(self.batch))

    @classmethod
    def from_tracks(
        cls,
        tracks: tuple[Track, ...] | list[Track],
        *,
        benchmark_version: str,
        split: TrackPoolSplit,
    ) -> TrackPool:
        """Build a pool from already validated immutable host Tracks."""

        if not tracks:
            raise TrackSchemaError("TrackPool requires at least one Track")
        generator_version = tracks[0].generator_version
        return cls(
            benchmark_version=benchmark_version,
            generator_version=generator_version,
            split=split,
            batch=stack_tracks(tracks),
        )

    @property
    def size(self) -> int:
        """Return the number of selectable Tracks."""

        return int(self.batch.seed.shape[0])

    @property
    def capacity(self) -> TrackCapacity:
        """Return the common fixed representation capacity."""

        return TrackCapacity(
            max_track_points=int(self.batch.centerline_m.shape[1]),
            max_checkpoints=int(self.batch.checkpoint_center_m.shape[1]),
        )


def track_pool_indices(selection_seeds: jax.Array, pool_size: int) -> jax.Array:
    """Map uint32 selection seeds to pool rows with deterministic sampling with replacement."""

    if isinstance(pool_size, bool) or not isinstance(pool_size, int) or pool_size < 1:
        raise ValueError("pool_size must be a positive integer")
    if pool_size > np.iinfo(np.int32).max:
        raise ValueError("pool_size must fit in int32")
    seeds = jnp.asarray(selection_seeds, dtype=jnp.uint32)
    if seeds.ndim != 1:
        raise ValueError("selection_seeds must be one-dimensional")
    return jnp.remainder(seeds, jnp.uint32(pool_size)).astype(jnp.int32)


def gather_track_batch(pool: TrackBatch, indices: jax.Array) -> TrackBatch:
    """Gather one fixed-capacity Track row per world using only device operations."""

    selected = jnp.asarray(indices, dtype=jnp.int32)
    if selected.ndim != 1:
        raise ValueError("indices must be one-dimensional")
    return jax.tree.map(lambda value: jnp.take(value, selected, axis=0), pool)


def masked_replace_track_batch(
    current: TrackBatch,
    replacement: TrackBatch,
    mask: jax.Array,
) -> TrackBatch:
    """Replace selected world rows while preserving every unselected leaf bit-exactly."""

    reset_mask = jnp.asarray(mask, dtype=bool)
    if reset_mask.ndim != 1:
        raise ValueError("mask must be one-dimensional")

    def select(old: jax.Array, new: jax.Array) -> jax.Array:
        if old.shape != new.shape or old.shape[0] != reset_mask.shape[0]:
            raise ValueError("TrackBatch replacement leaves must match the mask and each other")
        expanded = reset_mask.reshape((reset_mask.shape[0],) + (1,) * (old.ndim - 1))
        return jnp.where(expanded, new, old)

    return jax.tree.map(select, current, replacement)


__all__ = [
    "TrackPool",
    "TrackPoolSplit",
    "gather_track_batch",
    "masked_replace_track_batch",
    "track_pool_indices",
]
