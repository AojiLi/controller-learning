"""Dependency-free planar geometry helpers for offline track validation."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
from numpy.typing import ArrayLike, NDArray


def cross_2d(left: ArrayLike, right: ArrayLike) -> NDArray[np.float64] | np.float64:
    """Return the scalar 2-D cross product, preserving broadcast dimensions."""

    left_array = np.asarray(left, dtype=np.float64)
    right_array = np.asarray(right, dtype=np.float64)
    return left_array[..., 0] * right_array[..., 1] - left_array[..., 1] * right_array[..., 0]


def signed_area(points: ArrayLike) -> float:
    """Return the signed area of a polygon, accepting an optional repeated endpoint."""

    vertices = _points(points)
    if len(vertices) < 3:
        return 0.0
    if np.array_equal(vertices[0], vertices[-1]):
        vertices = vertices[:-1]
    if len(vertices) < 3:
        return 0.0
    following = np.roll(vertices, -1, axis=0)
    return float(0.5 * np.sum(cross_2d(vertices, following)))


def segments_intersect(
    start_a: ArrayLike,
    end_a: ArrayLike,
    start_b: ArrayLike,
    end_b: ArrayLike,
    *,
    atol: float = 1e-10,
) -> bool:
    """Return whether two closed 2-D segments intersect, including touching/overlap."""

    a = _point(start_a)
    b = _point(end_a)
    c = _point(start_b)
    d = _point(end_b)
    if not _aabbs_overlap(a, b, c, d, atol):
        return False

    ab = b - a
    cd = d - c
    scale = max(float(np.linalg.norm(ab)), float(np.linalg.norm(cd)), 1.0)
    tolerance = atol * scale * scale
    o1 = float(cross_2d(ab, c - a))
    o2 = float(cross_2d(ab, d - a))
    o3 = float(cross_2d(cd, a - c))
    o4 = float(cross_2d(cd, b - c))

    if ((o1 > tolerance and o2 < -tolerance) or (o1 < -tolerance and o2 > tolerance)) and (
        (o3 > tolerance and o4 < -tolerance) or (o3 < -tolerance and o4 > tolerance)
    ):
        return True
    return (
        (abs(o1) <= tolerance and _point_on_segment(c, a, b, atol))
        or (abs(o2) <= tolerance and _point_on_segment(d, a, b, atol))
        or (abs(o3) <= tolerance and _point_on_segment(a, c, d, atol))
        or (abs(o4) <= tolerance and _point_on_segment(b, c, d, atol))
    )


def point_segment_distance(point: ArrayLike, start: ArrayLike, end: ArrayLike) -> float:
    """Return the Euclidean distance from a point to a closed segment."""

    value = _point(point)
    segment_start = _point(start)
    segment = _point(end) - segment_start
    squared_length = float(np.dot(segment, segment))
    if squared_length == 0.0:
        return float(np.linalg.norm(value - segment_start))
    fraction = float(np.clip(np.dot(value - segment_start, segment) / squared_length, 0.0, 1.0))
    return float(np.linalg.norm(value - (segment_start + fraction * segment)))


def segment_distance(
    start_a: ArrayLike,
    end_a: ArrayLike,
    start_b: ArrayLike,
    end_b: ArrayLike,
    *,
    atol: float = 1e-10,
) -> float:
    """Return the exact minimum Euclidean distance between two closed segments."""

    if segments_intersect(start_a, end_a, start_b, end_b, atol=atol):
        return 0.0
    return min(
        point_segment_distance(start_a, start_b, end_b),
        point_segment_distance(end_a, start_b, end_b),
        point_segment_distance(start_b, start_a, end_a),
        point_segment_distance(end_b, start_a, end_a),
    )


def closed_polyline_self_intersections(
    points: ArrayLike,
    *,
    atol: float = 1e-10,
) -> tuple[tuple[int, int], ...]:
    """Return non-adjacent intersecting segment-index pairs of a closed polyline.

    The input must explicitly repeat its first point at the end. Adjacent segments and the
    first/last seam pair share a legal endpoint and are excluded.
    """

    vertices = _closed_points(points, atol)
    segment_count = len(vertices) - 1
    intersections: list[tuple[int, int]] = []
    for first in range(segment_count):
        for second in range(first + 1, segment_count):
            if second == first + 1 or (first == 0 and second == segment_count - 1):
                continue
            if segments_intersect(
                vertices[first],
                vertices[first + 1],
                vertices[second],
                vertices[second + 1],
                atol=atol,
            ):
                intersections.append((first, second))
    return tuple(intersections)


def minimum_nonlocal_clearance(
    points: ArrayLike,
    cumulative_s_m: ArrayLike,
    *,
    local_arc_exclusion_m: float,
    atol: float = 1e-10,
) -> tuple[float, tuple[int, int] | None]:
    """Return minimum segment clearance outside a fixed circular arc-length neighborhood.

    Segment midpoints define arc separation. This makes the exclusion independent of point count
    and is appropriate for fixed-arc-length resampled tracks.
    """

    if local_arc_exclusion_m < 0.0:
        raise ValueError("local_arc_exclusion_m cannot be negative")
    vertices = _closed_points(points, atol)
    cumulative = np.asarray(cumulative_s_m, dtype=np.float64)
    if cumulative.shape != (len(vertices),):
        raise ValueError("cumulative_s_m must have one value per point")
    if cumulative[0] != 0.0 or np.any(np.diff(cumulative) <= 0.0):
        raise ValueError("cumulative_s_m must start at zero and increase strictly")
    length = float(cumulative[-1])
    if length <= 0.0:
        raise ValueError("closed polyline length must be positive")

    midpoint_s = 0.5 * (cumulative[:-1] + cumulative[1:])
    best_distance = np.inf
    best_pair: tuple[int, int] | None = None
    segment_count = len(vertices) - 1
    for first in range(segment_count):
        for second in range(first + 1, segment_count):
            if second == first + 1 or (first == 0 and second == segment_count - 1):
                continue
            arc_delta = abs(float(midpoint_s[first] - midpoint_s[second]))
            circular_arc_delta = min(arc_delta, length - arc_delta)
            if circular_arc_delta <= local_arc_exclusion_m + atol:
                continue
            distance = segment_distance(
                vertices[first],
                vertices[first + 1],
                vertices[second],
                vertices[second + 1],
                atol=atol,
            )
            if distance < best_distance:
                best_distance = distance
                best_pair = (first, second)
    return float(best_distance), best_pair


def _point(value: ArrayLike) -> NDArray[np.float64]:
    point = np.asarray(value, dtype=np.float64)
    if point.shape != (2,) or not np.isfinite(point).all():
        raise ValueError("point must be a finite shape-(2,) value")
    return point


def _points(value: ArrayLike) -> NDArray[np.float64]:
    points = np.asarray(value, dtype=np.float64)
    if points.ndim != 2 or points.shape[1] != 2 or not np.isfinite(points).all():
        raise ValueError("points must be a finite shape-(n, 2) array")
    return points


def _closed_points(value: ArrayLike, atol: float) -> NDArray[np.float64]:
    points = _points(value)
    if len(points) < 4 or not np.allclose(points[0], points[-1], rtol=0.0, atol=atol):
        raise ValueError(
            "closed polyline must contain at least three segments and repeat its start"
        )
    return points


def _aabbs_overlap(
    a: NDArray[np.float64],
    b: NDArray[np.float64],
    c: NDArray[np.float64],
    d: NDArray[np.float64],
    atol: float,
) -> bool:
    return bool(
        np.all(
            np.maximum(np.minimum(a, b), np.minimum(c, d))
            <= np.minimum(np.maximum(a, b), np.maximum(c, d)) + atol
        )
    )


def _point_on_segment(
    point: NDArray[np.float64],
    start: NDArray[np.float64],
    end: NDArray[np.float64],
    atol: float,
) -> bool:
    return bool(
        np.all(point >= np.minimum(start, end) - atol)
        and np.all(point <= np.maximum(start, end) + atol)
    )


__all__: Sequence[str] = (
    "closed_polyline_self_intersections",
    "cross_2d",
    "minimum_nonlocal_clearance",
    "point_segment_distance",
    "segment_distance",
    "segments_intersect",
    "signed_area",
)
