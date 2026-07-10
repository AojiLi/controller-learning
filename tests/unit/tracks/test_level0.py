"""Tests for the fixed Level 0 teaching Track."""

from __future__ import annotations

import numpy as np
import pytest

from controller_learning.tracks.level0 import (
    DEFAULT_LEVEL0_CAPACITY,
    LEVEL0_TRACK_SEED,
    Level0TrackSpec,
    build_level0_candidate,
    build_level0_track,
)
from controller_learning.tracks.validator import validate_track_candidate


def test_level0_candidate_is_deterministic_and_passes_v01_geometry_validation() -> None:
    candidate = build_level0_candidate()
    repeated = build_level0_candidate()
    result = validate_track_candidate(candidate)

    assert result.valid, result.reasons
    assert candidate.seed == LEVEL0_TRACK_SEED
    assert candidate.generator_version == "v0.1"
    assert candidate.width_m == 7.0
    assert 300.0 <= candidate.length_m <= 600.0
    assert np.array_equal(candidate.centerline_m, repeated.centerline_m)
    assert np.ptp(candidate.curvature_1pm) > 0.01
    assert candidate.start_pose == pytest.approx((0.0, 0.0, 0.0))
    assert candidate.tangent[0] == pytest.approx((1.0, 0.0), abs=1.0e-12)


def test_level0_track_uses_the_locked_capacity_and_zero_padding() -> None:
    track = build_level0_track()

    assert track.capacity == DEFAULT_LEVEL0_CAPACITY
    assert track.seed == np.iinfo(np.uint32).max
    assert track.point_count < DEFAULT_LEVEL0_CAPACITY.max_track_points
    assert track.checkpoint_count < DEFAULT_LEVEL0_CAPACITY.max_checkpoints
    assert np.all(track.centerline_m[track.point_count :] == 0.0)
    assert np.all(track.checkpoint_center_m[track.checkpoint_count :] == 0.0)


def test_level0_spec_rejects_reversed_axes() -> None:
    with pytest.raises(ValueError, match="semi_major_axis_m"):
        Level0TrackSpec(semi_major_axis_m=40.0, semi_minor_axis_m=50.0)
