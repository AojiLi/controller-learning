"""Tests for immutable fixed-capacity track values."""

from __future__ import annotations

from dataclasses import replace

import numpy as np
import pytest

from controller_learning.tracks.types import (
    Track,
    TrackCapacity,
    TrackSchemaError,
    stack_tracks,
    track_array_bytes,
)


def _track(*, seed: int = 7, version: str = "test-v1") -> Track:
    max_points = 8
    max_checkpoints = 3
    point_count = 5
    checkpoint_count = 2
    centerline = np.zeros((max_points, 2), dtype=np.float32)
    centerline[:point_count] = ((0, 0), (1, 0), (1, 1), (0, 1), (0, 0))
    tangent = np.zeros_like(centerline)
    tangent[:point_count] = ((1, 0), (0, 1), (-1, 0), (0, -1), (1, 0))
    cumulative = np.zeros(max_points, dtype=np.float32)
    cumulative[:point_count] = (0, 1, 2, 3, 4)
    track_mask = np.arange(max_points) < point_count
    checkpoint_center = np.zeros((max_checkpoints, 2), dtype=np.float32)
    checkpoint_center[:checkpoint_count] = ((1, 0), (0, 0))
    checkpoint_tangent = np.zeros_like(checkpoint_center)
    checkpoint_tangent[:checkpoint_count] = ((0, 1), (1, 0))
    checkpoint_s = np.zeros(max_checkpoints, dtype=np.float32)
    checkpoint_s[:checkpoint_count] = (1, 4)
    checkpoint_mask = np.arange(max_checkpoints) < checkpoint_count
    return Track(
        seed=seed,
        generator_version=version,
        centerline_m=centerline,
        left_boundary_m=centerline,
        right_boundary_m=centerline,
        tangent=tangent,
        curvature_1pm=np.zeros(max_points, dtype=np.float32),
        cumulative_s_m=cumulative,
        track_mask=track_mask,
        checkpoint_center_m=checkpoint_center,
        checkpoint_tangent=checkpoint_tangent,
        checkpoint_s_m=checkpoint_s,
        checkpoint_mask=checkpoint_mask,
        start_pose=np.zeros(3, dtype=np.float32),
        point_count=point_count,
        checkpoint_count=checkpoint_count,
        length_m=4.0,
        width_m=7.0,
    )


def test_track_copies_arrays_and_makes_them_read_only() -> None:
    source = np.zeros((8, 2), dtype=np.float32)
    source[:5] = ((0, 0), (1, 0), (1, 1), (0, 1), (0, 0))
    template = _track()
    track = replace(template, centerline_m=source)
    source[0] = (99, 99)

    assert track.centerline_m[0] == pytest.approx((0, 0))
    with pytest.raises(ValueError, match="read-only"):
        track.centerline_m[0, 0] = 1.0


def test_track_exposes_capacity_segment_count_and_array_bytes() -> None:
    track = _track()

    assert track.capacity == TrackCapacity(max_track_points=8, max_checkpoints=3)
    assert track.segment_count == 4
    assert track_array_bytes(track) == sum(
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


def test_padding_must_be_zero() -> None:
    track = _track()
    curvature = track.curvature_1pm.copy()
    curvature[track.point_count] = 1.0

    with pytest.raises(TrackSchemaError, match="padding must be zero"):
        replace(track, curvature_1pm=curvature)


def test_masks_must_be_contiguous_prefixes() -> None:
    track = _track()
    mask = track.track_mask.copy()
    mask[1] = False

    with pytest.raises(TrackSchemaError, match="contiguous valid prefix"):
        replace(track, track_mask=mask)


def test_centerline_must_have_explicit_closure() -> None:
    track = _track()
    centerline = track.centerline_m.copy()
    centerline[track.point_count - 1] = (0.1, 0.0)

    with pytest.raises(TrackSchemaError, match="duplicate the first"):
        replace(track, centerline_m=centerline)


def test_stacked_batch_preserves_leading_world_dimension() -> None:
    batch = stack_tracks([_track(seed=1), _track(seed=2)])

    assert batch.seed.tolist() == [1, 2]
    assert batch.centerline_m.shape == (2, 8, 2)
    assert batch.checkpoint_center_m.shape == (2, 3, 2)
    assert batch.point_count.tolist() == [5, 5]
    assert batch.length_m.dtype == np.float32


def test_stack_rejects_mixed_versions_or_capacities() -> None:
    with pytest.raises(TrackSchemaError, match="one generator version"):
        stack_tracks([_track(version="a"), _track(version="b")])
    with pytest.raises(TrackSchemaError, match="same capacity"):
        track = _track()
        smaller = replace(
            track,
            checkpoint_center_m=track.checkpoint_center_m[:2],
            checkpoint_tangent=track.checkpoint_tangent[:2],
            checkpoint_s_m=track.checkpoint_s_m[:2],
            checkpoint_mask=track.checkpoint_mask[:2],
        )
        stack_tracks([track, smaller])


@pytest.mark.parametrize("capacity", [(3, 1), (4, 0)])
def test_capacity_rejects_invalid_values(capacity) -> None:
    with pytest.raises(TrackSchemaError):
        TrackCapacity(*capacity)
