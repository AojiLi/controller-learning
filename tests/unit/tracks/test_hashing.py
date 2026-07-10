"""Tests for canonical packed Track geometry hashes."""

from __future__ import annotations

from dataclasses import replace

from controller_learning.tracks.hashing import (
    track_batch_geometry_sha256,
    track_geometry_sha256,
)
from controller_learning.tracks.level0 import build_level0_track
from controller_learning.tracks.types import stack_tracks


def test_hash_excludes_identity_but_includes_exact_packed_geometry() -> None:
    track = build_level0_track()
    different_identity = replace(track, seed=17, generator_version="another-source")
    different_geometry = replace(track, width_m=track.width_m + 1.0)

    assert track_geometry_sha256(track) == track_geometry_sha256(different_identity)
    assert track_geometry_sha256(track) != track_geometry_sha256(different_geometry)


def test_batch_hashes_match_individual_tracks_in_order() -> None:
    first = build_level0_track()
    second = replace(first, seed=17, width_m=8.0)

    assert track_batch_geometry_sha256(stack_tracks([first, second])) == (
        track_geometry_sha256(first),
        track_geometry_sha256(second),
    )
