"""Deterministic offline validation for generated track candidates."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from types import MappingProxyType

import numpy as np
from numpy.typing import NDArray
from scipy.spatial import cKDTree

from controller_learning.tracks.generator import TrackCandidate
from controller_learning.tracks.geometry import segment_distance, segments_intersect, signed_area

MetricValue = bool | int | float


@dataclass(frozen=True, slots=True)
class TrackValidationSpec:
    """Geometry limits for the initial v0.1 Level 1 generator spike."""

    min_length_m: float = 300.0
    max_length_m: float = 600.0
    expected_width_m: float = 7.0
    max_abs_curvature_1pm: float = 1.0 / 15.0
    start_window_m: float = 25.0
    start_max_abs_curvature_1pm: float = 1.0 / 40.0
    min_nonlocal_centerline_clearance_m: float = 9.0
    local_arc_exclusion_m: float = 25.0
    closure_tolerance_m: float = 1.0e-8
    cumulative_tolerance_m: float = 1.0e-6
    width_tolerance_m: float = 1.0e-6
    tangent_norm_tolerance: float = 1.0e-6
    boundary_offset_tolerance_m: float = 1.0e-5
    start_pose_tolerance: float = 1.0e-8
    checkpoint_center_tolerance_m: float = 2.0e-2
    checkpoint_tangent_tolerance: float = 1.0e-3
    intersection_tolerance_m: float = 1.0e-9

    def __post_init__(self) -> None:
        positive = (
            self.min_length_m,
            self.max_length_m,
            self.expected_width_m,
            self.max_abs_curvature_1pm,
            self.start_window_m,
            self.start_max_abs_curvature_1pm,
            self.min_nonlocal_centerline_clearance_m,
        )
        if not all(np.isfinite(value) and value > 0.0 for value in positive):
            raise ValueError("track validation limits must be finite and positive")
        if self.min_length_m >= self.max_length_m:
            raise ValueError("length limits must be strictly ordered")
        if not np.isfinite(self.local_arc_exclusion_m) or self.local_arc_exclusion_m < 0.0:
            raise ValueError("local_arc_exclusion_m must be finite and nonnegative")
        tolerances = (
            self.closure_tolerance_m,
            self.cumulative_tolerance_m,
            self.width_tolerance_m,
            self.tangent_norm_tolerance,
            self.boundary_offset_tolerance_m,
            self.start_pose_tolerance,
            self.checkpoint_center_tolerance_m,
            self.checkpoint_tangent_tolerance,
            self.intersection_tolerance_m,
        )
        if not all(np.isfinite(value) and value >= 0.0 for value in tolerances):
            raise ValueError("track validation tolerances must be finite and nonnegative")


@dataclass(frozen=True, slots=True)
class ValidationResult:
    """Immutable result with every failure reported in a stable priority order."""

    valid: bool
    reasons: tuple[str, ...]
    primary_reason: str | None
    metrics: Mapping[str, MetricValue]

    def __post_init__(self) -> None:
        reasons = tuple(self.reasons)
        if self.valid != (not reasons):
            raise ValueError("valid must agree with reasons")
        expected_primary = reasons[0] if reasons else None
        if self.primary_reason != expected_primary:
            raise ValueError("primary_reason must be the first reason, or None when valid")
        object.__setattr__(self, "reasons", reasons)
        object.__setattr__(self, "metrics", MappingProxyType(dict(self.metrics)))


_REASON_ORDER: tuple[str, ...] = (
    "invalid_shapes",
    "non_finite",
    "centerline_not_closed",
    "left_boundary_not_closed",
    "right_boundary_not_closed",
    "tangent_not_closed",
    "cumulative_s_invalid",
    "tangent_not_unit",
    "start_pose_invalid",
    "centerline_not_ccw",
    "length_out_of_range",
    "width_invalid",
    "boundary_offset_invalid",
    "curvature_exceeded",
    "start_not_straight",
    "centerline_self_intersection",
    "left_boundary_self_intersection",
    "right_boundary_self_intersection",
    "boundaries_intersect",
    "nonlocal_clearance",
    "checkpoint_s_invalid",
    "checkpoint_finish_invalid",
    "checkpoint_center_invalid",
    "checkpoint_tangent_invalid",
)


def validate_track_candidate(
    candidate: TrackCandidate,
    spec: TrackValidationSpec | None = None,
) -> ValidationResult:
    """Validate one candidate without modifying it or attempting another seed."""

    spec = TrackValidationSpec() if spec is None else spec
    failures: set[str] = set()
    metrics: dict[str, MetricValue] = {}

    arrays, shape_is_valid, values_are_finite = _candidate_arrays(candidate)
    if not shape_is_valid:
        failures.add("invalid_shapes")
    if not values_are_finite:
        failures.add("non_finite")
    if not shape_is_valid or not values_are_finite:
        return _make_result(failures, metrics)

    centerline = arrays["centerline_m"]
    left_boundary = arrays["left_boundary_m"]
    right_boundary = arrays["right_boundary_m"]
    tangent = arrays["tangent"]
    curvature = arrays["curvature_1pm"]
    cumulative = arrays["cumulative_s_m"]
    checkpoint_center = arrays["checkpoint_center_m"]
    checkpoint_tangent = arrays["checkpoint_tangent"]
    checkpoint_s = arrays["checkpoint_s_m"]
    start_pose = arrays["start_pose"]

    metrics["point_count"] = int(centerline.shape[0])
    metrics["checkpoint_count"] = int(checkpoint_s.shape[0])
    metrics["length_m"] = float(candidate.length_m)
    metrics["width_m"] = float(candidate.width_m)

    closure_by_field = (
        ("centerline_not_closed", centerline),
        ("left_boundary_not_closed", left_boundary),
        ("right_boundary_not_closed", right_boundary),
        ("tangent_not_closed", tangent),
    )
    closed: dict[str, bool] = {}
    for reason, values in closure_by_field:
        is_closed = bool(
            np.allclose(
                values[0],
                values[-1],
                rtol=0.0,
                atol=spec.closure_tolerance_m,
            )
        )
        closed[reason] = is_closed
        if not is_closed:
            failures.add(reason)

    cumulative_is_valid = bool(
        abs(float(cumulative[0])) <= spec.cumulative_tolerance_m
        and np.all(np.diff(cumulative) > 0.0)
        and abs(float(cumulative[-1] - candidate.length_m)) <= spec.cumulative_tolerance_m
    )
    if not cumulative_is_valid:
        failures.add("cumulative_s_invalid")

    tangent_norm = np.linalg.norm(tangent, axis=1)
    max_tangent_norm_error = float(np.max(np.abs(tangent_norm - 1.0)))
    metrics["max_tangent_norm_error"] = max_tangent_norm_error
    tangent_is_unit = max_tangent_norm_error <= spec.tangent_norm_tolerance
    if not tangent_is_unit:
        failures.add("tangent_not_unit")

    start_heading = float(np.arctan2(tangent[0, 1], tangent[0, 0]))
    start_heading_error = abs(_wrap_angle(start_heading - float(start_pose[2])))
    start_position_error = float(np.linalg.norm(centerline[0] - start_pose[:2]))
    canonical_pose_error = float(np.max(np.abs(start_pose)))
    metrics["start_position_error_m"] = start_position_error
    metrics["start_heading_error_rad"] = start_heading_error
    if (
        start_position_error > spec.start_pose_tolerance
        or start_heading_error > spec.start_pose_tolerance
        or canonical_pose_error > spec.start_pose_tolerance
    ):
        failures.add("start_pose_invalid")

    area = signed_area(centerline)
    metrics["signed_area_m2"] = area
    if area <= spec.intersection_tolerance_m:
        failures.add("centerline_not_ccw")

    if not spec.min_length_m <= candidate.length_m <= spec.max_length_m:
        failures.add("length_out_of_range")
    if (
        candidate.width_m <= 0.0
        or abs(candidate.width_m - spec.expected_width_m) > spec.width_tolerance_m
    ):
        failures.add("width_invalid")

    if np.all(tangent_norm > 0.0):
        unit_tangent = tangent / tangent_norm[:, None]
        left_normal = np.column_stack((-unit_tangent[:, 1], unit_tangent[:, 0]))
        half_width = 0.5 * float(candidate.width_m)
        expected_left = centerline + half_width * left_normal
        expected_right = centerline - half_width * left_normal
        boundary_error = max(
            float(np.max(np.linalg.norm(left_boundary - expected_left, axis=1))),
            float(np.max(np.linalg.norm(right_boundary - expected_right, axis=1))),
        )
        metrics["max_boundary_offset_error_m"] = boundary_error
        if boundary_error > spec.boundary_offset_tolerance_m:
            failures.add("boundary_offset_invalid")
    else:
        failures.add("boundary_offset_invalid")

    max_abs_curvature = float(np.max(np.abs(curvature)))
    metrics["max_abs_curvature_1pm"] = max_abs_curvature
    if max_abs_curvature > spec.max_abs_curvature_1pm:
        failures.add("curvature_exceeded")

    if cumulative_is_valid:
        start_window_mask = cumulative < spec.start_window_m - spec.cumulative_tolerance_m
        start_window_mask[0] = True
        start_max_curvature = float(np.max(np.abs(curvature[start_window_mask])))
        metrics["start_max_abs_curvature_1pm"] = start_max_curvature
        if start_max_curvature > spec.start_max_abs_curvature_1pm:
            failures.add("start_not_straight")

    intersection_inputs = (
        ("centerline_self_intersection", "centerline_not_closed", centerline),
        ("left_boundary_self_intersection", "left_boundary_not_closed", left_boundary),
        ("right_boundary_self_intersection", "right_boundary_not_closed", right_boundary),
    )
    for reason, closure_reason, points in intersection_inputs:
        if not closed[closure_reason]:
            continue
        intersections = _self_intersections(points, spec.intersection_tolerance_m)
        metrics[f"{reason}_count"] = len(intersections)
        if intersections:
            failures.add(reason)

    if closed["left_boundary_not_closed"] and closed["right_boundary_not_closed"]:
        boundary_intersections = _mutual_intersections(
            left_boundary,
            right_boundary,
            spec.intersection_tolerance_m,
        )
        metrics["boundary_mutual_intersection_count"] = len(boundary_intersections)
        if boundary_intersections:
            failures.add("boundaries_intersect")

    if closed["centerline_not_closed"] and cumulative_is_valid:
        clearance_floor, clearance_is_censored = _nonlocal_clearance_floor(
            centerline,
            cumulative,
            threshold_m=spec.min_nonlocal_centerline_clearance_m,
            local_arc_exclusion_m=spec.local_arc_exclusion_m,
            atol=spec.intersection_tolerance_m,
        )
        metrics["nonlocal_centerline_clearance_floor_m"] = clearance_floor
        metrics["nonlocal_clearance_censored"] = clearance_is_censored
        if (
            clearance_floor
            < spec.min_nonlocal_centerline_clearance_m - spec.intersection_tolerance_m
        ):
            failures.add("nonlocal_clearance")

    checkpoint_s_is_valid = bool(
        checkpoint_s[0] > 0.0
        and np.all(np.diff(checkpoint_s) > 0.0)
        and np.all(checkpoint_s <= candidate.length_m + spec.cumulative_tolerance_m)
    )
    if not checkpoint_s_is_valid:
        failures.add("checkpoint_s_invalid")
    if abs(float(checkpoint_s[-1] - candidate.length_m)) > spec.cumulative_tolerance_m:
        failures.add("checkpoint_finish_invalid")

    checkpoint_tangent_norm = np.linalg.norm(checkpoint_tangent, axis=1)
    checkpoint_tangent_norm_error = float(np.max(np.abs(checkpoint_tangent_norm - 1.0)))
    metrics["max_checkpoint_tangent_norm_error"] = checkpoint_tangent_norm_error

    if cumulative_is_valid and checkpoint_s_is_valid:
        expected_center, expected_tangent = _interpolate_track(
            centerline,
            tangent,
            cumulative,
            checkpoint_s,
        )
        checkpoint_center_error = float(
            np.max(np.linalg.norm(checkpoint_center - expected_center, axis=1))
        )
        checkpoint_tangent_error = float(
            np.max(np.linalg.norm(checkpoint_tangent - expected_tangent, axis=1))
        )
        metrics["max_checkpoint_center_error_m"] = checkpoint_center_error
        metrics["max_checkpoint_tangent_error"] = checkpoint_tangent_error
        if checkpoint_center_error > spec.checkpoint_center_tolerance_m:
            failures.add("checkpoint_center_invalid")
        if (
            checkpoint_tangent_norm_error > spec.tangent_norm_tolerance
            or checkpoint_tangent_error > spec.checkpoint_tangent_tolerance
        ):
            failures.add("checkpoint_tangent_invalid")
    elif checkpoint_tangent_norm_error > spec.tangent_norm_tolerance:
        failures.add("checkpoint_tangent_invalid")

    return _make_result(failures, metrics)


def _candidate_arrays(
    candidate: TrackCandidate,
) -> tuple[dict[str, NDArray[np.float64]], bool, bool]:
    field_names = (
        "control_points_m",
        "centerline_m",
        "left_boundary_m",
        "right_boundary_m",
        "tangent",
        "curvature_1pm",
        "cumulative_s_m",
        "checkpoint_center_m",
        "checkpoint_tangent",
        "checkpoint_s_m",
        "start_pose",
    )
    arrays: dict[str, NDArray[np.float64]] = {}
    try:
        for name in field_names:
            arrays[name] = np.asarray(getattr(candidate, name), dtype=np.float64)
        scalar_values = np.asarray((candidate.length_m, candidate.width_m), dtype=np.float64)
    except (AttributeError, TypeError, ValueError):
        return arrays, False, False

    centerline = arrays["centerline_m"]
    checkpoint_s = arrays["checkpoint_s_m"]
    point_count = centerline.shape[0] if centerline.ndim == 2 else -1
    checkpoint_count = checkpoint_s.shape[0] if checkpoint_s.ndim == 1 else -1
    shape_is_valid = bool(
        arrays["control_points_m"].ndim == 2
        and arrays["control_points_m"].shape[0] >= 4
        and arrays["control_points_m"].shape[1:] == (2,)
        and point_count >= 4
        and centerline.shape[1:] == (2,)
        and all(
            arrays[name].shape == (point_count, 2)
            for name in ("left_boundary_m", "right_boundary_m", "tangent")
        )
        and all(
            arrays[name].shape == (point_count,) for name in ("curvature_1pm", "cumulative_s_m")
        )
        and checkpoint_count >= 1
        and arrays["checkpoint_center_m"].shape == (checkpoint_count, 2)
        and arrays["checkpoint_tangent"].shape == (checkpoint_count, 2)
        and arrays["start_pose"].shape == (3,)
    )
    values_are_finite = bool(
        np.isfinite(scalar_values).all()
        and all(np.isfinite(values).all() for values in arrays.values())
    )
    return arrays, shape_is_valid, values_are_finite


def _make_result(
    failures: set[str],
    metrics: Mapping[str, MetricValue],
) -> ValidationResult:
    unknown = failures.difference(_REASON_ORDER)
    if unknown:
        raise AssertionError(f"unknown validation reasons: {sorted(unknown)}")
    reasons = tuple(reason for reason in _REASON_ORDER if reason in failures)
    return ValidationResult(
        valid=not reasons,
        reasons=reasons,
        primary_reason=reasons[0] if reasons else None,
        metrics=metrics,
    )


def _wrap_angle(angle: float) -> float:
    return float((angle + np.pi) % (2.0 * np.pi) - np.pi)


def _self_intersections(
    points: NDArray[np.float64],
    atol: float,
) -> tuple[tuple[int, int], ...]:
    pairs = _aabb_pairs(points, points, atol=atol, same_polyline=True)
    intersections = [
        (first, second)
        for first, second in pairs
        if segments_intersect(
            points[first],
            points[first + 1],
            points[second],
            points[second + 1],
            atol=atol,
        )
    ]
    return tuple(intersections)


def _mutual_intersections(
    left: NDArray[np.float64],
    right: NDArray[np.float64],
    atol: float,
) -> tuple[tuple[int, int], ...]:
    pairs = _aabb_pairs(left, right, atol=atol, same_polyline=False)
    intersections = [
        (first, second)
        for first, second in pairs
        if segments_intersect(
            left[first],
            left[first + 1],
            right[second],
            right[second + 1],
            atol=atol,
        )
    ]
    return tuple(intersections)


def _aabb_pairs(
    first_points: NDArray[np.float64],
    second_points: NDArray[np.float64],
    *,
    atol: float,
    same_polyline: bool,
) -> tuple[tuple[int, int], ...]:
    """Use a sweep-axis AABB broad phase before exact segment predicates."""

    first_start, first_end = first_points[:-1], first_points[1:]
    second_start, second_end = second_points[:-1], second_points[1:]
    first_min = np.minimum(first_start, first_end)
    first_max = np.maximum(first_start, first_end)
    second_min = np.minimum(second_start, second_end)
    second_max = np.maximum(second_start, second_end)
    second_order = np.argsort(second_min[:, 0], kind="stable")
    sorted_second_min_x = second_min[second_order, 0]
    second_segment_count = second_start.shape[0]

    pairs: list[tuple[int, int]] = []
    for first in range(first_start.shape[0]):
        upper = int(np.searchsorted(sorted_second_min_x, first_max[first, 0] + atol, side="right"))
        candidates = second_order[:upper]
        overlap = (
            (second_max[candidates, 0] >= first_min[first, 0] - atol)
            & (second_max[candidates, 1] >= first_min[first, 1] - atol)
            & (second_min[candidates, 1] <= first_max[first, 1] + atol)
        )
        for second in candidates[overlap].tolist():
            if same_polyline:
                if second <= first:
                    continue
                if second == first + 1 or (first == 0 and second == second_segment_count - 1):
                    continue
            pairs.append((first, int(second)))
    return tuple(pairs)


def _nonlocal_clearance_floor(
    points: NDArray[np.float64],
    cumulative_s_m: NDArray[np.float64],
    *,
    threshold_m: float,
    local_arc_exclusion_m: float,
    atol: float,
) -> tuple[float, bool]:
    """Return exact violating clearance, or the validated threshold as a censored floor."""

    segment_start = points[:-1]
    segment_end = points[1:]
    segment_length = np.linalg.norm(segment_end - segment_start, axis=1)
    midpoint = 0.5 * (segment_start + segment_end)
    midpoint_s = 0.5 * (cumulative_s_m[:-1] + cumulative_s_m[1:])
    length_m = float(cumulative_s_m[-1])
    query_radius = threshold_m + float(np.max(segment_length)) + atol
    pairs = cKDTree(midpoint).query_pairs(query_radius, output_type="ndarray")
    if pairs.size == 0:
        return threshold_m, True
    pairs = np.asarray(pairs, dtype=np.int64).reshape((-1, 2))
    arc_delta = np.abs(midpoint_s[pairs[:, 0]] - midpoint_s[pairs[:, 1]])
    circular_arc_delta = np.minimum(arc_delta, length_m - arc_delta)
    nonlocal_pairs = pairs[circular_arc_delta > local_arc_exclusion_m + atol]
    if nonlocal_pairs.size == 0:
        return threshold_m, True

    best = np.inf
    for first, second in nonlocal_pairs.tolist():
        distance = segment_distance(
            segment_start[first],
            segment_end[first],
            segment_start[second],
            segment_end[second],
            atol=atol,
        )
        best = min(best, distance)
    if best >= threshold_m - atol:
        return threshold_m, True
    return float(best), False


def _interpolate_track(
    centerline: NDArray[np.float64],
    tangent: NDArray[np.float64],
    cumulative_s_m: NDArray[np.float64],
    query_s_m: NDArray[np.float64],
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    segment = np.searchsorted(cumulative_s_m, query_s_m, side="right") - 1
    segment = np.clip(segment, 0, centerline.shape[0] - 2)
    segment_length = cumulative_s_m[segment + 1] - cumulative_s_m[segment]
    fraction = (query_s_m - cumulative_s_m[segment]) / segment_length
    fraction = np.clip(fraction, 0.0, 1.0)
    position = centerline[segment] + fraction[:, None] * (
        centerline[segment + 1] - centerline[segment]
    )
    direction = tangent[segment] + fraction[:, None] * (tangent[segment + 1] - tangent[segment])
    direction_norm = np.linalg.norm(direction, axis=1)
    safe_norm = np.where(direction_norm > 0.0, direction_norm, 1.0)
    direction = direction / safe_norm[:, None]
    return position, direction


__all__: Sequence[str] = (
    "TrackValidationSpec",
    "ValidationResult",
    "validate_track_candidate",
)
