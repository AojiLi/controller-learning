"""Tests for deterministic offline track candidate generation."""

from __future__ import annotations

import numpy as np
import pytest

from controller_learning.tracks.generator import (
    TrackGenerationError,
    TrackGenerationSpec,
    generate_track_candidate,
    pack_track,
)
from controller_learning.tracks.types import TrackCapacity


@pytest.fixture(scope="module")
def candidate():
    return generate_track_candidate(42)


def test_default_spec_records_the_spike_distribution() -> None:
    spec = TrackGenerationSpec()
    assert (spec.min_control_points, spec.max_control_points) == (8, 16)
    assert (spec.min_radius_m, spec.max_radius_m) == (52.0, 88.0)
    assert spec.angular_gap_jitter == 0.18
    assert spec.radial_perturbation == 0.16
    assert spec.width_m == 7.0
    assert spec.arc_spacing_m == 1.0
    assert spec.checkpoint_spacing_m == 15.0
    assert (spec.min_length_m, spec.max_length_m) == (300.0, 600.0)
    assert spec.generator_version == "v0.1"


def test_generation_is_exactly_deterministic_and_float64(candidate) -> None:
    repeated = generate_track_candidate(42)
    assert candidate.seed == repeated.seed == 42
    for name in (
        "control_points_m",
        "centerline_m",
        "tangent",
        "curvature_1pm",
        "checkpoint_center_m",
    ):
        actual = getattr(candidate, name)
        assert actual.dtype == np.float64
        assert np.array_equal(actual, getattr(repeated, name))
        assert not actual.flags.writeable


def test_candidate_has_closed_ccw_normalized_geometry(candidate) -> None:
    assert np.array_equal(candidate.centerline_m[0], candidate.centerline_m[-1])
    assert candidate.centerline_m[0] == pytest.approx((0.0, 0.0), abs=1e-12)
    assert candidate.tangent[0] == pytest.approx((1.0, 0.0), abs=1e-12)
    assert candidate.start_pose == pytest.approx((0.0, 0.0, 0.0))
    xy = candidate.centerline_m[:-1]
    area_twice = np.sum(xy[:, 0] * np.roll(xy[:, 1], -1) - np.roll(xy[:, 0], -1) * xy[:, 1])
    assert area_twice > 0.0
    assert np.linalg.norm(candidate.tangent, axis=1) == pytest.approx(1.0, abs=1e-12)


def test_spacing_boundaries_and_checkpoints(candidate) -> None:
    segment_lengths = np.diff(candidate.cumulative_s_m)
    assert segment_lengths[:-1] == pytest.approx(1.0, abs=1e-12)
    assert 0.5 <= segment_lengths[-1] <= 1.5
    assert candidate.cumulative_s_m[-1] == candidate.length_m
    assert np.linalg.norm(
        candidate.left_boundary_m - candidate.centerline_m, axis=1
    ) == pytest.approx(3.5)
    assert np.linalg.norm(
        candidate.right_boundary_m - candidate.centerline_m, axis=1
    ) == pytest.approx(3.5)
    assert candidate.checkpoint_s_m[-1] == candidate.length_m
    assert np.diff(candidate.checkpoint_s_m)[:-1] == pytest.approx(15.0)
    assert candidate.checkpoint_center_m[-1] == pytest.approx((0.0, 0.0), abs=1e-12)


def test_selected_start_window_obeys_curvature_limit(candidate) -> None:
    spec = TrackGenerationSpec()
    window_points = int(np.ceil(spec.start_window_m / spec.arc_spacing_m))
    assert np.max(np.abs(candidate.curvature_1pm[:window_points])) <= spec.start_max_curvature_1pm


def test_pack_uses_fixed_shapes_and_zero_padding(candidate) -> None:
    capacity = TrackCapacity(candidate.point_count + 3, candidate.checkpoint_count + 2)
    track = pack_track(candidate, capacity)
    assert track.centerline_m.shape == (capacity.max_track_points, 2)
    assert track.checkpoint_s_m.shape == (capacity.max_checkpoints,)
    assert np.all(track.centerline_m[candidate.point_count :] == 0.0)
    assert np.all(track.checkpoint_s_m[candidate.checkpoint_count :] == 0.0)
    assert track.point_count == candidate.point_count
    assert track.checkpoint_count == candidate.checkpoint_count


def test_pack_reports_structured_capacity_overflow(candidate) -> None:
    with pytest.raises(TrackGenerationError) as caught:
        pack_track(candidate, TrackCapacity(candidate.point_count - 1, candidate.checkpoint_count))
    assert caught.value.reason == "track_capacity_overflow"
    assert caught.value.context["required"] == candidate.point_count

    with pytest.raises(TrackGenerationError) as caught:
        pack_track(candidate, TrackCapacity(candidate.point_count, candidate.checkpoint_count - 1))
    assert caught.value.reason == "checkpoint_capacity_overflow"
