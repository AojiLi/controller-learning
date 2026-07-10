"""Tests for dependency-free planar track geometry."""

from __future__ import annotations

import numpy as np
import pytest

from controller_learning.tracks.geometry import (
    closed_polyline_self_intersections,
    cross_2d,
    minimum_nonlocal_clearance,
    segment_distance,
    segments_intersect,
    signed_area,
)


def test_cross_product_broadcasts_and_signed_area_tracks_orientation() -> None:
    assert cross_2d((1, 0), (0, 1)) == pytest.approx(1.0)
    assert cross_2d(np.array(((1, 0), (0, 1))), (0, 1)).tolist() == [1.0, 0.0]
    square = np.array(((0, 0), (2, 0), (2, 1), (0, 1), (0, 0)), dtype=float)
    assert signed_area(square) == pytest.approx(2.0)
    assert signed_area(square[::-1]) == pytest.approx(-2.0)


@pytest.mark.parametrize(
    ("first", "second"),
    [
        (((0, 0), (2, 2)), ((0, 2), (2, 0))),
        (((0, 0), (1, 0)), ((1, 0), (2, 1))),
        (((0, 0), (3, 0)), ((1, 0), (2, 0))),
    ],
)
def test_segment_intersection_includes_crossing_touching_and_overlap(first, second) -> None:
    assert segments_intersect(*first, *second)
    assert segment_distance(*first, *second) == 0.0


def test_segment_distance_handles_parallel_and_degenerate_segments() -> None:
    assert not segments_intersect((0, 0), (2, 0), (0, 3), (2, 3))
    assert segment_distance((0, 0), (2, 0), (0, 3), (2, 3)) == pytest.approx(3.0)
    assert segment_distance((0, 0), (0, 0), (3, 4), (3, 4)) == pytest.approx(5.0)


def test_closed_polyline_self_intersection_ignores_legal_seam_neighbors() -> None:
    square = np.array(((0, 0), (2, 0), (2, 2), (0, 2), (0, 0)), dtype=float)
    bow_tie = np.array(((0, 0), (2, 2), (0, 2), (2, 0), (0, 0)), dtype=float)

    assert closed_polyline_self_intersections(square) == ()
    assert closed_polyline_self_intersections(bow_tie) == ((0, 2),)


def test_nonlocal_clearance_uses_arc_length_not_point_index() -> None:
    rectangle = np.array(((0, 0), (10, 0), (10, 1), (0, 1), (0, 0)), dtype=float)
    cumulative = np.array((0, 10, 11, 21, 22), dtype=float)

    clearance, pair = minimum_nonlocal_clearance(
        rectangle,
        cumulative,
        local_arc_exclusion_m=2.0,
    )
    assert clearance == pytest.approx(1.0)
    assert pair == (0, 2)

    clearance, pair = minimum_nonlocal_clearance(
        rectangle,
        cumulative,
        local_arc_exclusion_m=11.0,
    )
    assert clearance == np.inf
    assert pair is None


def test_closed_polyline_helpers_require_explicit_closure() -> None:
    with pytest.raises(ValueError, match="repeat its start"):
        closed_polyline_self_intersections(((0, 0), (1, 0), (0, 1)))
