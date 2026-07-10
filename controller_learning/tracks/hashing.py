"""Canonical hashes for packed Track geometry."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping

import numpy as np

from controller_learning.tracks.types import Track, TrackBatch, TrackSchemaError

_HASH_DOMAIN = b"controller-learning/packed-track-geometry/v1\x00"

_GEOMETRY_DTYPES: tuple[tuple[str, np.dtype], ...] = (
    ("centerline_m", np.dtype(np.float32)),
    ("left_boundary_m", np.dtype(np.float32)),
    ("right_boundary_m", np.dtype(np.float32)),
    ("tangent", np.dtype(np.float32)),
    ("curvature_1pm", np.dtype(np.float32)),
    ("cumulative_s_m", np.dtype(np.float32)),
    ("track_mask", np.dtype(np.bool_)),
    ("checkpoint_center_m", np.dtype(np.float32)),
    ("checkpoint_tangent", np.dtype(np.float32)),
    ("checkpoint_s_m", np.dtype(np.float32)),
    ("checkpoint_mask", np.dtype(np.bool_)),
    ("start_pose", np.dtype(np.float32)),
    ("point_count", np.dtype(np.int32)),
    ("checkpoint_count", np.dtype(np.int32)),
    ("length_m", np.dtype(np.float32)),
    ("width_m", np.dtype(np.float32)),
)


def _canonical_array(value: object, expected_dtype: np.dtype, field: str) -> np.ndarray:
    array = np.asarray(value)
    if array.dtype.kind != expected_dtype.kind or array.dtype.itemsize != expected_dtype.itemsize:
        raise TrackSchemaError(f"{field} must use {expected_dtype.name}, got {array.dtype.name}")
    canonical_dtype = expected_dtype.newbyteorder("<")
    return np.ascontiguousarray(array, dtype=canonical_dtype)


def _geometry_digest(values: Mapping[str, object]) -> str:
    digest = hashlib.sha256(_HASH_DOMAIN)
    for field, dtype in _GEOMETRY_DTYPES:
        array = _canonical_array(values[field], dtype, field)
        header = json.dumps(
            {"dtype": array.dtype.str, "field": field, "shape": list(array.shape)},
            allow_nan=False,
            separators=(",", ":"),
            sort_keys=True,
        ).encode("utf-8")
        digest.update(len(header).to_bytes(4, byteorder="little", signed=False))
        digest.update(header)
        digest.update(array.tobytes(order="C"))
    return digest.hexdigest()


def track_geometry_sha256(track: Track) -> str:
    """Hash one packed Track's exact geometry, excluding seed and generator version.

    Field names, shapes, canonical little-endian dtypes, values, masks, padding, capacity, and
    scalar geometry are part of the digest. Identity metadata is deliberately excluded so equal
    geometry generated under different seeds or source labels can be detected.
    """

    values = {field: getattr(track, field) for field, _ in _GEOMETRY_DTYPES}
    values["point_count"] = np.asarray(track.point_count, dtype=np.int32)
    values["checkpoint_count"] = np.asarray(track.checkpoint_count, dtype=np.int32)
    values["length_m"] = np.asarray(track.length_m, dtype=np.float32)
    values["width_m"] = np.asarray(track.width_m, dtype=np.float32)
    return _geometry_digest(values)


def track_batch_geometry_sha256(batch: TrackBatch) -> tuple[str, ...]:
    """Return canonical per-row geometry hashes for one fixed-shape TrackBatch."""

    seed = np.asarray(batch.seed)
    if seed.ndim != 1 or seed.shape[0] < 1:
        raise TrackSchemaError("TrackBatch.seed must have one non-empty leading dimension")
    track_count = seed.shape[0]
    arrays = {field: np.asarray(getattr(batch, field)) for field, _ in _GEOMETRY_DTYPES}
    if any(array.ndim < 1 or array.shape[0] != track_count for array in arrays.values()):
        raise TrackSchemaError("all TrackBatch geometry fields must share the leading dimension")
    return tuple(
        _geometry_digest({field: array[index] for field, array in arrays.items()})
        for index in range(track_count)
    )


__all__ = ["track_batch_geometry_sha256", "track_geometry_sha256"]
