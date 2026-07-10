"""Immutable host tracks and fixed-shape runtime track batches."""

from __future__ import annotations

from dataclasses import dataclass
from typing import NamedTuple

import numpy as np
from numpy.typing import NDArray


class TrackSchemaError(ValueError):
    """Raised when fixed-capacity track data violates its structural contract."""


@dataclass(frozen=True, slots=True)
class TrackCapacity:
    """Static point and checkpoint capacities shared by one benchmark version."""

    max_track_points: int
    max_checkpoints: int

    def __post_init__(self) -> None:
        if self.max_track_points < 4:
            raise TrackSchemaError("max_track_points must be at least four")
        if self.max_checkpoints < 1:
            raise TrackSchemaError("max_checkpoints must be positive")


def _readonly_array(
    value: object,
    *,
    dtype: np.dtype,
    shape: tuple[int, ...],
    field: str,
) -> NDArray:
    array = np.array(value, dtype=dtype, copy=True)
    if array.shape != shape:
        raise TrackSchemaError(f"{field} must have shape {shape}, got {array.shape}")
    if np.issubdtype(array.dtype, np.floating) and not np.isfinite(array).all():
        raise TrackSchemaError(f"{field} must contain only finite values")
    array.setflags(write=False)
    return array


def _require_zero_padding(array: NDArray, count: int, field: str) -> None:
    if np.any(array[count:] != 0):
        raise TrackSchemaError(f"{field} padding must be zero")


@dataclass(frozen=True, slots=True)
class Track:
    """One validated closed track stored in a fixed-capacity host representation.

    The last valid centerline point duplicates the first point. A track with ``point_count`` valid
    points therefore has ``point_count - 1`` directed segments and never connects through padding.
    """

    seed: int
    generator_version: str
    centerline_m: NDArray[np.float32]
    left_boundary_m: NDArray[np.float32]
    right_boundary_m: NDArray[np.float32]
    tangent: NDArray[np.float32]
    curvature_1pm: NDArray[np.float32]
    cumulative_s_m: NDArray[np.float32]
    track_mask: NDArray[np.bool_]
    checkpoint_center_m: NDArray[np.float32]
    checkpoint_tangent: NDArray[np.float32]
    checkpoint_s_m: NDArray[np.float32]
    checkpoint_mask: NDArray[np.bool_]
    start_pose: NDArray[np.float32]
    point_count: int
    checkpoint_count: int
    length_m: float
    width_m: float

    def __post_init__(self) -> None:
        if not 0 <= self.seed <= np.iinfo(np.uint32).max:
            raise TrackSchemaError("seed must fit in uint32")
        if not self.generator_version:
            raise TrackSchemaError("generator_version cannot be empty")
        if not np.isfinite(self.length_m) or self.length_m <= 0.0:
            raise TrackSchemaError("length_m must be finite and positive")
        if not np.isfinite(self.width_m) or self.width_m <= 0.0:
            raise TrackSchemaError("width_m must be finite and positive")

        centerline = np.asarray(self.centerline_m)
        checkpoints = np.asarray(self.checkpoint_center_m)
        if centerline.ndim != 2 or centerline.shape[1:] != (2,):
            raise TrackSchemaError("centerline_m must have shape (max_track_points, 2)")
        if checkpoints.ndim != 2 or checkpoints.shape[1:] != (2,):
            raise TrackSchemaError("checkpoint_center_m must have shape (max_checkpoints, 2)")
        max_track_points = centerline.shape[0]
        max_checkpoints = checkpoints.shape[0]
        if not 4 <= self.point_count <= max_track_points:
            raise TrackSchemaError("point_count must fit the track capacity and include closure")
        if not 1 <= self.checkpoint_count <= max_checkpoints:
            raise TrackSchemaError("checkpoint_count must fit the checkpoint capacity")

        float_track_fields = (
            ("centerline_m", (max_track_points, 2)),
            ("left_boundary_m", (max_track_points, 2)),
            ("right_boundary_m", (max_track_points, 2)),
            ("tangent", (max_track_points, 2)),
            ("curvature_1pm", (max_track_points,)),
            ("cumulative_s_m", (max_track_points,)),
        )
        float_checkpoint_fields = (
            ("checkpoint_center_m", (max_checkpoints, 2)),
            ("checkpoint_tangent", (max_checkpoints, 2)),
            ("checkpoint_s_m", (max_checkpoints,)),
        )
        for field, shape in (*float_track_fields, *float_checkpoint_fields):
            array = _readonly_array(
                getattr(self, field),
                dtype=np.dtype(np.float32),
                shape=shape,
                field=field,
            )
            _require_zero_padding(
                array,
                self.point_count if field in dict(float_track_fields) else self.checkpoint_count,
                field,
            )
            object.__setattr__(self, field, array)

        track_mask = _readonly_array(
            self.track_mask,
            dtype=np.dtype(np.bool_),
            shape=(max_track_points,),
            field="track_mask",
        )
        checkpoint_mask = _readonly_array(
            self.checkpoint_mask,
            dtype=np.dtype(np.bool_),
            shape=(max_checkpoints,),
            field="checkpoint_mask",
        )
        expected_track_mask = np.arange(max_track_points) < self.point_count
        expected_checkpoint_mask = np.arange(max_checkpoints) < self.checkpoint_count
        if not np.array_equal(track_mask, expected_track_mask):
            raise TrackSchemaError("track_mask must be one contiguous valid prefix")
        if not np.array_equal(checkpoint_mask, expected_checkpoint_mask):
            raise TrackSchemaError("checkpoint_mask must be one contiguous valid prefix")
        object.__setattr__(self, "track_mask", track_mask)
        object.__setattr__(self, "checkpoint_mask", checkpoint_mask)

        start_pose = _readonly_array(
            self.start_pose,
            dtype=np.dtype(np.float32),
            shape=(3,),
            field="start_pose",
        )
        object.__setattr__(self, "start_pose", start_pose)

        if not np.array_equal(self.centerline_m[0], self.centerline_m[self.point_count - 1]):
            raise TrackSchemaError("the last valid centerline point must duplicate the first")
        if self.cumulative_s_m[0] != 0.0:
            raise TrackSchemaError("cumulative_s_m must start at zero")
        if not np.all(np.diff(self.cumulative_s_m[: self.point_count]) > 0.0):
            raise TrackSchemaError("valid cumulative_s_m values must increase strictly")
        if not np.isclose(
            self.cumulative_s_m[self.point_count - 1],
            self.length_m,
            rtol=0.0,
            atol=1e-4,
        ):
            raise TrackSchemaError("the closing cumulative distance must equal length_m")

    @property
    def capacity(self) -> TrackCapacity:
        """Return the static representation capacity."""

        return TrackCapacity(self.centerline_m.shape[0], self.checkpoint_center_m.shape[0])

    @property
    def segment_count(self) -> int:
        """Return the number of valid directed centerline segments."""

        return self.point_count - 1


class TrackBatch(NamedTuple):
    """Fixed-shape numerical track values with one leading world dimension."""

    seed: NDArray[np.uint32]
    centerline_m: NDArray[np.float32]
    left_boundary_m: NDArray[np.float32]
    right_boundary_m: NDArray[np.float32]
    tangent: NDArray[np.float32]
    curvature_1pm: NDArray[np.float32]
    cumulative_s_m: NDArray[np.float32]
    track_mask: NDArray[np.bool_]
    checkpoint_center_m: NDArray[np.float32]
    checkpoint_tangent: NDArray[np.float32]
    checkpoint_s_m: NDArray[np.float32]
    checkpoint_mask: NDArray[np.bool_]
    start_pose: NDArray[np.float32]
    point_count: NDArray[np.int32]
    checkpoint_count: NDArray[np.int32]
    length_m: NDArray[np.float32]
    width_m: NDArray[np.float32]


def stack_tracks(tracks: tuple[Track, ...] | list[Track]) -> TrackBatch:
    """Stack compatible host tracks without importing or initializing JAX."""

    if not tracks:
        raise TrackSchemaError("at least one track is required")
    capacity = tracks[0].capacity
    generator_version = tracks[0].generator_version
    for track in tracks[1:]:
        if track.capacity != capacity:
            raise TrackSchemaError("all tracks in a batch must use the same capacity")
        if track.generator_version != generator_version:
            raise TrackSchemaError("all tracks in a batch must use one generator version")

    def stack(field: str) -> NDArray:
        return np.stack([getattr(track, field) for track in tracks], axis=0)

    return TrackBatch(
        seed=np.asarray([track.seed for track in tracks], dtype=np.uint32),
        centerline_m=stack("centerline_m"),
        left_boundary_m=stack("left_boundary_m"),
        right_boundary_m=stack("right_boundary_m"),
        tangent=stack("tangent"),
        curvature_1pm=stack("curvature_1pm"),
        cumulative_s_m=stack("cumulative_s_m"),
        track_mask=stack("track_mask"),
        checkpoint_center_m=stack("checkpoint_center_m"),
        checkpoint_tangent=stack("checkpoint_tangent"),
        checkpoint_s_m=stack("checkpoint_s_m"),
        checkpoint_mask=stack("checkpoint_mask"),
        start_pose=stack("start_pose"),
        point_count=np.asarray([track.point_count for track in tracks], dtype=np.int32),
        checkpoint_count=np.asarray(
            [track.checkpoint_count for track in tracks],
            dtype=np.int32,
        ),
        length_m=np.asarray([track.length_m for track in tracks], dtype=np.float32),
        width_m=np.asarray([track.width_m for track in tracks], dtype=np.float32),
    )


def track_array_bytes(track: Track) -> int:
    """Return bytes used by the fixed numerical arrays of one host track."""

    return sum(
        array.nbytes
        for array in (
            track.centerline_m,
            track.left_boundary_m,
            track.right_boundary_m,
            track.tangent,
            track.curvature_1pm,
            track.cumulative_s_m,
            track.track_mask,
            track.checkpoint_center_m,
            track.checkpoint_tangent,
            track.checkpoint_s_m,
            track.checkpoint_mask,
            track.start_pose,
        )
    )


__all__ = [
    "Track",
    "TrackBatch",
    "TrackCapacity",
    "TrackSchemaError",
    "stack_tracks",
    "track_array_bytes",
]
