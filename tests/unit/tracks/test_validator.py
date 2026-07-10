"""Tests for deterministic offline track candidate validation."""

from __future__ import annotations

from dataclasses import replace

import numpy as np
import pytest

from controller_learning.tracks.generator import generate_track_candidate
from controller_learning.tracks.validator import TrackValidationSpec, validate_track_candidate


@pytest.fixture(scope="module")
def candidate():
    return generate_track_candidate(42)


def test_default_spec_records_v01_geometry_limits() -> None:
    spec = TrackValidationSpec()
    assert (spec.min_length_m, spec.max_length_m) == (300.0, 600.0)
    assert spec.expected_width_m == 7.0
    assert spec.max_abs_curvature_1pm == pytest.approx(1.0 / 15.0)
    assert spec.start_window_m == 25.0
    assert spec.start_max_abs_curvature_1pm == pytest.approx(1.0 / 40.0)
    assert spec.min_nonlocal_centerline_clearance_m == 9.0
    assert spec.local_arc_exclusion_m == 25.0


def test_generated_candidate_passes_with_immutable_metrics(candidate) -> None:
    result = validate_track_candidate(candidate)
    assert result.valid
    assert result.reasons == ()
    assert result.primary_reason is None
    assert result.metrics["point_count"] == candidate.point_count
    assert result.metrics["max_abs_curvature_1pm"] == pytest.approx(
        np.max(np.abs(candidate.curvature_1pm))
    )
    with pytest.raises(TypeError):
        result.metrics["point_count"] = 0  # type: ignore[index]


def test_structural_defense_reports_shape_and_finite_failures(candidate) -> None:
    bad_shape = replace(candidate)
    object.__setattr__(bad_shape, "tangent", bad_shape.tangent[:-1])
    shape_result = validate_track_candidate(bad_shape)
    assert shape_result.reasons == ("invalid_shapes",)

    non_finite = replace(candidate)
    curvature = non_finite.curvature_1pm.copy()
    curvature[10] = np.nan
    object.__setattr__(non_finite, "curvature_1pm", curvature)
    finite_result = validate_track_candidate(non_finite)
    assert finite_result.reasons == ("non_finite",)


def test_closure_checks_cover_centerline_boundaries_and_tangent(candidate) -> None:
    centerline = candidate.centerline_m.copy()
    centerline[-1] += (0.1, 0.0)
    bad_centerline = replace(candidate)
    object.__setattr__(bad_centerline, "centerline_m", centerline)
    assert validate_track_candidate(bad_centerline).primary_reason == "centerline_not_closed"

    left = candidate.left_boundary_m.copy()
    left[-1] += (0.1, 0.0)
    left_result = validate_track_candidate(replace(candidate, left_boundary_m=left))
    assert left_result.primary_reason == "left_boundary_not_closed"

    right = candidate.right_boundary_m.copy()
    right[-1] += (0.1, 0.0)
    right_result = validate_track_candidate(replace(candidate, right_boundary_m=right))
    assert right_result.primary_reason == "right_boundary_not_closed"

    tangent = candidate.tangent.copy()
    tangent[-1] = (0.0, 1.0)
    tangent_result = validate_track_candidate(replace(candidate, tangent=tangent))
    assert tangent_result.primary_reason == "tangent_not_closed"


def test_cumulative_distance_and_tangent_norm_are_checked(candidate) -> None:
    cumulative = candidate.cumulative_s_m.copy()
    cumulative[10] = cumulative[9]
    cumulative_result = validate_track_candidate(replace(candidate, cumulative_s_m=cumulative))
    assert cumulative_result.primary_reason == "cumulative_s_invalid"

    tangent_result = validate_track_candidate(replace(candidate, tangent=2.0 * candidate.tangent))
    assert tangent_result.primary_reason == "tangent_not_unit"
    assert tangent_result.metrics["max_tangent_norm_error"] == pytest.approx(1.0)


def test_start_pose_orientation_length_and_width_are_checked(candidate) -> None:
    pose_result = validate_track_candidate(replace(candidate, start_pose=np.array((1.0, 0.0, 0.0))))
    assert pose_result.primary_reason == "start_pose_invalid"

    clockwise = candidate.centerline_m[::-1]
    orientation_result = validate_track_candidate(replace(candidate, centerline_m=clockwise))
    assert "centerline_not_ccw" in orientation_result.reasons

    length_spec = replace(TrackValidationSpec(), min_length_m=candidate.length_m + 1.0)
    length_result = validate_track_candidate(candidate, length_spec)
    assert length_result.reasons == ("length_out_of_range",)

    width_result = validate_track_candidate(
        candidate, replace(TrackValidationSpec(), expected_width_m=8.0)
    )
    assert width_result.reasons == ("width_invalid",)


def test_boundary_offsets_must_keep_left_right_orientation(candidate) -> None:
    result = validate_track_candidate(
        replace(
            candidate,
            left_boundary_m=candidate.right_boundary_m,
            right_boundary_m=candidate.left_boundary_m,
        )
    )
    assert "boundary_offset_invalid" in result.reasons
    assert result.metrics["max_boundary_offset_error_m"] == pytest.approx(candidate.width_m)


def test_curvature_and_start_window_failures_have_stable_order(candidate) -> None:
    curvature = np.full_like(candidate.curvature_1pm, 0.1)
    result = validate_track_candidate(replace(candidate, curvature_1pm=curvature))
    assert result.reasons == ("curvature_exceeded", "start_not_straight")
    assert result.primary_reason == "curvature_exceeded"


def test_self_and_mutual_intersections_are_detected(candidate) -> None:
    centerline = candidate.centerline_m.copy()
    centerline[:4] = ((0.0, 0.0), (2.0, 2.0), (0.0, 2.0), (2.0, 0.0))
    centerline_result = validate_track_candidate(replace(candidate, centerline_m=centerline))
    assert "centerline_self_intersection" in centerline_result.reasons

    left = candidate.left_boundary_m.copy()
    origin = left[0].copy()
    left[:4] = origin + np.array(((0.0, 0.0), (2.0, 2.0), (0.0, 2.0), (2.0, 0.0)))
    left_result = validate_track_candidate(replace(candidate, left_boundary_m=left))
    assert "left_boundary_self_intersection" in left_result.reasons

    right = candidate.right_boundary_m.copy()
    origin = right[0].copy()
    right[:4] = origin + np.array(((0.0, 0.0), (2.0, 2.0), (0.0, 2.0), (2.0, 0.0)))
    right_result = validate_track_candidate(replace(candidate, right_boundary_m=right))
    assert "right_boundary_self_intersection" in right_result.reasons

    mutual_result = validate_track_candidate(
        replace(candidate, right_boundary_m=candidate.left_boundary_m)
    )
    assert "boundaries_intersect" in mutual_result.reasons
    assert mutual_result.metrics["boundary_mutual_intersection_count"] > 0


def test_nonlocal_clearance_uses_threshold_censored_metric(candidate) -> None:
    valid_result = validate_track_candidate(candidate)
    assert valid_result.metrics["nonlocal_centerline_clearance_floor_m"] == 9.0
    assert valid_result.metrics["nonlocal_clearance_censored"] is True

    strict_spec = replace(
        TrackValidationSpec(),
        min_nonlocal_centerline_clearance_m=100.0,
    )
    strict_result = validate_track_candidate(candidate, strict_spec)
    assert "nonlocal_clearance" in strict_result.reasons
    assert strict_result.metrics["nonlocal_centerline_clearance_floor_m"] < 100.0
    assert strict_result.metrics["nonlocal_clearance_censored"] is False


def test_checkpoint_order_finish_centers_and_tangents_are_checked(candidate) -> None:
    unordered = candidate.checkpoint_s_m.copy()
    unordered[1] = unordered[0]
    unordered_result = validate_track_candidate(replace(candidate, checkpoint_s_m=unordered))
    assert "checkpoint_s_invalid" in unordered_result.reasons

    early_finish = candidate.checkpoint_s_m.copy()
    early_finish[-1] = 0.5 * (early_finish[-2] + candidate.length_m)
    finish_result = validate_track_candidate(replace(candidate, checkpoint_s_m=early_finish))
    assert finish_result.reasons == (
        "checkpoint_finish_invalid",
        "checkpoint_center_invalid",
        "checkpoint_tangent_invalid",
    )

    checkpoint_center = candidate.checkpoint_center_m.copy()
    checkpoint_center[0] += (1.0, 0.0)
    center_result = validate_track_candidate(
        replace(candidate, checkpoint_center_m=checkpoint_center)
    )
    assert center_result.reasons == ("checkpoint_center_invalid",)

    tangent_result = validate_track_candidate(
        replace(candidate, checkpoint_tangent=-candidate.checkpoint_tangent)
    )
    assert tangent_result.reasons == ("checkpoint_tangent_invalid",)


def test_reason_priority_is_independent_of_check_execution_order(candidate) -> None:
    curvature = np.full_like(candidate.curvature_1pm, 0.1)
    result = validate_track_candidate(replace(candidate, width_m=8.0, curvature_1pm=curvature))
    assert result.reasons == (
        "width_invalid",
        "boundary_offset_invalid",
        "curvature_exceeded",
        "start_not_straight",
    )
    assert result.primary_reason == "width_invalid"
