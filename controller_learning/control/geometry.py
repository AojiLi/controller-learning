"""Observation-only planar path geometry for Controller implementations.

This module deliberately depends only on NumPy and public observation values.  It is safe for a
Controller plugin to import: no Environment, Race Core, physics, or Track implementation details
cross the Controller boundary.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

import numpy as np
from numpy.typing import ArrayLike, NDArray

_CLOSURE_ATOL_M = 0.0
_LENGTH_ATOL_M = 1.0e-4
# Public geometry is float32, while official lengths originate before the final float32 packing.
# Summing hundreds of unpacked segments in float64 can therefore differ by a few millimetres.
_LENGTH_RTOL = 3.0e-5
_MIN_NORM = 1.0e-12


def _numeric_array(value: Any, *, name: str) -> NDArray[Any]:
    try:
        array = np.asarray(value)
    except (TypeError, ValueError) as error:
        raise ValueError(f"{name} must be a numeric array") from error
    if not np.issubdtype(array.dtype, np.number) and array.dtype != np.dtype(np.bool_):
        raise ValueError(f"{name} must be a numeric array")
    return array


def _readonly(value: ArrayLike) -> NDArray[np.float64]:
    array = np.array(value, dtype=np.float64, copy=True)
    array.setflags(write=False)
    return array


def _finite_vector(value: ArrayLike, *, name: str) -> NDArray[np.float64]:
    array = _numeric_array(value, name=name)
    if array.shape[-1:] != (2,):
        raise ValueError(f"{name} must have shape (..., 2)")
    result = np.asarray(array, dtype=np.float64)
    if not np.isfinite(result).all():
        raise ValueError(f"{name} must contain only finite values")
    return result


def _scalar_or_array(value: NDArray[np.float64]) -> float | NDArray[np.float64]:
    if value.ndim == 0:
        return float(value)
    value.setflags(write=False)
    return value


def wrap_angle(angle_rad: ArrayLike) -> float | NDArray[np.float64]:
    """Wrap finite angles to the half-open interval ``[-pi, pi)``."""

    angle = _numeric_array(angle_rad, name="angle_rad")
    values = np.asarray(angle, dtype=np.float64)
    if not np.isfinite(values).all():
        raise ValueError("angle_rad must contain only finite values")
    wrapped = np.asarray((values + np.pi) % (2.0 * np.pi) - np.pi, dtype=np.float64)
    return _scalar_or_array(wrapped)


def world_to_body(vector_world: ArrayLike, yaw_rad: ArrayLike) -> NDArray[np.float64]:
    """Rotate world-frame planar vectors into a body frame with the given yaw."""

    vector = _finite_vector(vector_world, name="vector_world")
    yaw = np.asarray(_numeric_array(yaw_rad, name="yaw_rad"), dtype=np.float64)
    if not np.isfinite(yaw).all():
        raise ValueError("yaw_rad must contain only finite values")
    try:
        x, y, cosine, sine = np.broadcast_arrays(
            vector[..., 0], vector[..., 1], np.cos(yaw), np.sin(yaw)
        )
    except ValueError as error:
        raise ValueError("vector_world and yaw_rad cannot be broadcast together") from error
    return np.stack((cosine * x + sine * y, -sine * x + cosine * y), axis=-1)


def body_to_world(vector_body: ArrayLike, yaw_rad: ArrayLike) -> NDArray[np.float64]:
    """Rotate body-frame planar vectors into the world frame with the given yaw."""

    vector = _finite_vector(vector_body, name="vector_body")
    yaw = np.asarray(_numeric_array(yaw_rad, name="yaw_rad"), dtype=np.float64)
    if not np.isfinite(yaw).all():
        raise ValueError("yaw_rad must contain only finite values")
    try:
        x, y, cosine, sine = np.broadcast_arrays(
            vector[..., 0], vector[..., 1], np.cos(yaw), np.sin(yaw)
        )
    except ValueError as error:
        raise ValueError("vector_body and yaw_rad cannot be broadcast together") from error
    return np.stack((cosine * x - sine * y, sine * x + cosine * y), axis=-1)


@dataclass(frozen=True, slots=True)
class PathProjection:
    """Immutable closest-point result on one directed centerline segment."""

    segment_index: int
    segment_fraction: float
    s_m: float
    point_m: NDArray[np.float64]
    tangent: NDArray[np.float64]
    lateral_error_m: float
    distance_m: float

    def __post_init__(self) -> None:
        object.__setattr__(self, "point_m", _readonly(self.point_m))
        object.__setattr__(self, "tangent", _readonly(self.tangent))


@dataclass(frozen=True, slots=True)
class PathSample:
    """Immutable periodic path sample at one or more arc-length coordinates."""

    s_m: float | NDArray[np.float64]
    center_m: NDArray[np.float64]
    tangent: NDArray[np.float64]
    curvature_1pm: float | NDArray[np.float64]
    left_boundary_m: NDArray[np.float64]
    right_boundary_m: NDArray[np.float64]

    def __post_init__(self) -> None:
        if isinstance(self.s_m, np.ndarray):
            object.__setattr__(self, "s_m", _readonly(self.s_m))
        if isinstance(self.curvature_1pm, np.ndarray):
            object.__setattr__(self, "curvature_1pm", _readonly(self.curvature_1pm))
        for name in ("center_m", "tangent", "left_boundary_m", "right_boundary_m"):
            object.__setattr__(self, name, _readonly(getattr(self, name)))


@dataclass(frozen=True, slots=True)
class CenterlineReference:
    """Validated periodic centerline derived solely from a public observation.

    The valid geometry and all cached arrays are owned, float64, and read-only. Padding outside the
    contiguous public mask is intentionally neither copied nor inspected.
    """

    centerline_m: NDArray[np.float64]
    left_boundary_m: NDArray[np.float64]
    right_boundary_m: NDArray[np.float64]
    segment_delta_m: NDArray[np.float64]
    segment_length_m: NDArray[np.float64]
    segment_tangent: NDArray[np.float64]
    cumulative_s_m: NDArray[np.float64]
    tangent: NDArray[np.float64]
    curvature_1pm: NDArray[np.float64]
    track_length_m: float

    @classmethod
    def from_observation(cls, observation: Mapping[str, Any]) -> CenterlineReference:
        """Validate and copy the valid closed geometry in one public observation."""

        if not isinstance(observation, Mapping):
            raise TypeError("observation must be a mapping")
        required = {
            "centerline",
            "left_boundary",
            "right_boundary",
            "track_mask",
            "track_length",
        }
        missing = sorted(required - set(observation))
        if missing:
            raise ValueError(f"observation is missing geometry field(s): {missing}")

        center_source = _numeric_array(observation["centerline"], name="centerline")
        if center_source.ndim != 2 or center_source.shape[1] != 2:
            raise ValueError("centerline must have shape (capacity, 2)")
        capacity = center_source.shape[0]
        if capacity < 4:
            raise ValueError("centerline capacity must contain at least four points")

        boundary_sources: dict[str, NDArray[Any]] = {}
        for name in ("left_boundary", "right_boundary"):
            value = _numeric_array(observation[name], name=name)
            if value.shape != (capacity, 2):
                raise ValueError(f"{name} must have shape ({capacity}, 2)")
            boundary_sources[name] = value

        mask = _numeric_array(observation["track_mask"], name="track_mask")
        if mask.shape != (capacity,):
            raise ValueError(f"track_mask must have shape ({capacity},)")
        if not np.all((mask == 0) | (mask == 1)):
            raise ValueError("track_mask must contain only zero or one")
        valid_count = int(np.count_nonzero(mask))
        if valid_count < 4:
            raise ValueError("track_mask must select at least four points including closure")
        expected_mask = np.arange(capacity) < valid_count
        if not np.array_equal(mask.astype(np.bool_), expected_mask):
            raise ValueError("track_mask must be one contiguous valid prefix")

        centerline = np.array(center_source[:valid_count], dtype=np.float64, copy=True)
        left_boundary = np.array(
            boundary_sources["left_boundary"][:valid_count], dtype=np.float64, copy=True
        )
        right_boundary = np.array(
            boundary_sources["right_boundary"][:valid_count], dtype=np.float64, copy=True
        )
        for name, geometry in (
            ("centerline", centerline),
            ("left_boundary", left_boundary),
            ("right_boundary", right_boundary),
        ):
            if not np.isfinite(geometry).all():
                raise ValueError(f"valid {name} points must contain only finite values")
            if not np.allclose(geometry[0], geometry[-1], rtol=0.0, atol=_CLOSURE_ATOL_M):
                raise ValueError(f"valid {name} points must be explicitly closed")

        segment_delta = np.diff(centerline, axis=0)
        segment_length = np.linalg.norm(segment_delta, axis=1)
        if not np.all(segment_length > 0.0):
            raise ValueError("all valid centerline segments must have positive length")
        computed_length = float(np.sum(segment_length, dtype=np.float64))

        track_length_source = _numeric_array(observation["track_length"], name="track_length")
        if track_length_source.shape != ():
            raise ValueError("track_length must be a scalar")
        track_length = float(track_length_source)
        if not np.isfinite(track_length) or track_length <= 0.0:
            raise ValueError("track_length must be finite and positive")
        if not np.isclose(
            track_length,
            computed_length,
            rtol=_LENGTH_RTOL,
            atol=_LENGTH_ATOL_M,
        ):
            raise ValueError(
                "track_length is inconsistent with the valid centerline segment lengths"
            )

        segment_tangent = segment_delta / segment_length[:, None]
        previous_tangent = np.roll(segment_tangent, 1, axis=0)
        vertex_tangent = previous_tangent + segment_tangent
        vertex_norm = np.linalg.norm(vertex_tangent, axis=1)
        degenerate = vertex_norm <= _MIN_NORM
        vertex_tangent[~degenerate] /= vertex_norm[~degenerate, None]
        vertex_tangent[degenerate] = segment_tangent[degenerate]

        turn_angle = np.arctan2(
            previous_tangent[:, 0] * segment_tangent[:, 1]
            - previous_tangent[:, 1] * segment_tangent[:, 0],
            np.sum(previous_tangent * segment_tangent, axis=1),
        )
        previous_length = np.roll(segment_length, 1)
        curvature = turn_angle / (0.5 * (previous_length + segment_length))
        tangent = np.concatenate((vertex_tangent, vertex_tangent[:1]), axis=0)
        closed_curvature = np.concatenate((curvature, curvature[:1]), axis=0)
        cumulative = np.concatenate(
            (np.zeros(1, dtype=np.float64), np.cumsum(segment_length, dtype=np.float64))
        )

        values = (
            centerline,
            left_boundary,
            right_boundary,
            segment_delta,
            segment_length,
            segment_tangent,
            cumulative,
            tangent,
            closed_curvature,
        )
        for value in values:
            value.setflags(write=False)
        return cls(*values, track_length_m=computed_length)

    @property
    def point_count(self) -> int:
        """Number of valid points, including the explicit closure point."""

        return int(self.centerline_m.shape[0])

    @property
    def segment_count(self) -> int:
        """Number of directed centerline segments."""

        return int(self.segment_length_m.shape[0])

    def project(
        self,
        position_m: ArrayLike,
        hint_segment: int | None = None,
        backward_segments: int = 8,
        forward_segments: int = 32,
    ) -> PathProjection:
        """Project a point globally, or into a modulo-local window around ``hint_segment``."""

        position = _finite_vector(position_m, name="position_m")
        if position.shape != (2,):
            raise ValueError("position_m must have shape (2,)")
        for name, count in (
            ("backward_segments", backward_segments),
            ("forward_segments", forward_segments),
        ):
            if isinstance(count, bool) or not isinstance(count, int) or count < 0:
                raise ValueError(f"{name} must be a non-negative integer")

        if hint_segment is None:
            candidates = np.arange(self.segment_count, dtype=np.int64)
            tie_rank = candidates
        else:
            if isinstance(hint_segment, bool) or not isinstance(hint_segment, int):
                raise TypeError("hint_segment must be an integer or None")
            offsets = np.arange(-backward_segments, forward_segments + 1, dtype=np.int64)
            candidates = np.mod(hint_segment + offsets, self.segment_count)
            # Shared vertices (especially the explicit closure) create exact ties. Prefer the
            # hinted segment, then a forward neighbor, then a backward neighbor so a stationary
            # Controller does not jump across the seam.
            tie_rank = 2 * np.abs(offsets) + (offsets < 0)

        starts = self.centerline_m[candidates]
        deltas = self.segment_delta_m[candidates]
        squared_lengths = self.segment_length_m[candidates] ** 2
        fractions = np.clip(
            np.sum((position - starts) * deltas, axis=1) / squared_lengths,
            0.0,
            1.0,
        )
        points = starts + fractions[:, None] * deltas
        squared_distances = np.sum((position - points) ** 2, axis=1)
        minimum_distance = float(np.min(squared_distances))
        tied = squared_distances <= minimum_distance + 1.0e-12
        selected = int(np.argmin(np.where(tied, tie_rank, np.iinfo(np.int64).max)))
        segment_index = int(candidates[selected])
        fraction = float(fractions[selected])
        point = points[selected]
        tangent = self.segment_tangent[segment_index]
        offset = position - point
        lateral_error = float(tangent[0] * offset[1] - tangent[1] * offset[0])
        s_m = float(
            np.mod(
                self.cumulative_s_m[segment_index]
                + fraction * self.segment_length_m[segment_index],
                self.track_length_m,
            )
        )
        return PathProjection(
            segment_index=segment_index,
            segment_fraction=fraction,
            s_m=s_m,
            point_m=point,
            tangent=tangent,
            lateral_error_m=lateral_error,
            distance_m=float(np.sqrt(squared_distances[selected])),
        )

    def sample(self, s_m: ArrayLike) -> PathSample:
        """Periodically interpolate center, tangent, curvature, and boundaries at ``s_m``."""

        source = _numeric_array(s_m, name="s_m")
        values = np.asarray(source, dtype=np.float64)
        if not np.isfinite(values).all():
            raise ValueError("s_m must contain only finite values")
        wrapped = np.asarray(np.mod(values, self.track_length_m), dtype=np.float64)
        flat = wrapped.reshape(-1)
        segment = np.searchsorted(self.cumulative_s_m[1:], flat, side="right")
        segment = np.minimum(segment, self.segment_count - 1)
        fraction = (flat - self.cumulative_s_m[segment]) / self.segment_length_m[segment]

        def interpolate(points: NDArray[np.float64]) -> NDArray[np.float64]:
            result = points[segment] + fraction[:, None] * (points[segment + 1] - points[segment])
            return result.reshape((*wrapped.shape, 2))

        center = interpolate(self.centerline_m)
        left = interpolate(self.left_boundary_m)
        right = interpolate(self.right_boundary_m)
        tangent = interpolate(self.tangent)
        tangent_norm = np.linalg.norm(tangent, axis=-1, keepdims=True)
        fallback = self.segment_tangent[segment].reshape((*wrapped.shape, 2))
        tangent = np.divide(
            tangent,
            tangent_norm,
            out=fallback.copy(),
            where=tangent_norm > _MIN_NORM,
        )
        curvature_flat = self.curvature_1pm[segment] + fraction * (
            self.curvature_1pm[segment + 1] - self.curvature_1pm[segment]
        )
        curvature = curvature_flat.reshape(wrapped.shape)

        sample_s = _scalar_or_array(wrapped.copy())
        sample_curvature = _scalar_or_array(np.asarray(curvature, dtype=np.float64).copy())
        return PathSample(
            s_m=sample_s,
            center_m=center,
            tangent=tangent,
            curvature_1pm=sample_curvature,
            left_boundary_m=left,
            right_boundary_m=right,
        )

    def preview(self, start_s_m: float, offsets_m: ArrayLike) -> PathSample:
        """Sample periodic lookahead offsets from one scalar arc-length coordinate."""

        if isinstance(start_s_m, bool) or not np.isscalar(start_s_m):
            raise TypeError("start_s_m must be a scalar")
        start = float(start_s_m)
        if not np.isfinite(start):
            raise ValueError("start_s_m must be finite")
        offsets = _numeric_array(offsets_m, name="offsets_m")
        offset_values = np.asarray(offsets, dtype=np.float64)
        if not np.isfinite(offset_values).all():
            raise ValueError("offsets_m must contain only finite values")
        return self.sample(start + offset_values)


__all__ = [
    "CenterlineReference",
    "PathProjection",
    "PathSample",
    "body_to_world",
    "world_to_body",
    "wrap_angle",
]
