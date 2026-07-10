"""Deterministic fixed geometry for the Level 0 teaching Track."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from controller_learning.tracks.generator import TrackCandidate, pack_track
from controller_learning.tracks.types import Track, TrackCapacity

LEVEL0_TRACK_SEED = int(np.iinfo(np.uint32).max)
DEFAULT_LEVEL0_CAPACITY = TrackCapacity(max_track_points=640, max_checkpoints=48)


@dataclass(frozen=True, slots=True)
class Level0TrackSpec:
    """Parameters of the simple fixed ellipse used by Level 0."""

    semi_major_axis_m: float = 70.0
    semi_minor_axis_m: float = 50.0
    width_m: float = 7.0
    arc_spacing_m: float = 1.0
    checkpoint_spacing_m: float = 15.0
    generator_version: str = "v0.1"
    dense_sample_count: int = 131_072

    def __post_init__(self) -> None:
        values = (
            self.semi_major_axis_m,
            self.semi_minor_axis_m,
            self.width_m,
            self.arc_spacing_m,
            self.checkpoint_spacing_m,
        )
        if not all(np.isfinite(value) and value > 0.0 for value in values):
            raise ValueError("Level 0 distances must be finite and positive")
        if self.semi_major_axis_m < self.semi_minor_axis_m:
            raise ValueError("semi_major_axis_m cannot be smaller than semi_minor_axis_m")
        if type(self.dense_sample_count) is not int or self.dense_sample_count < 4096:
            raise ValueError("dense_sample_count must be at least 4096")
        if not self.generator_version:
            raise ValueError("generator_version cannot be empty")


def _spaced_distances(length_m: float, spacing_m: float) -> np.ndarray:
    full_steps = int(np.floor(length_m / spacing_m))
    remainder = length_m - full_steps * spacing_m
    if remainder <= 1.0e-10:
        distances = np.arange(full_steps + 1, dtype=np.float64) * spacing_m
        distances[-1] = length_m
        return distances
    if remainder < 0.5 * spacing_m:
        prefix = np.arange(full_steps, dtype=np.float64) * spacing_m
    else:
        prefix = np.arange(full_steps + 1, dtype=np.float64) * spacing_m
    return np.concatenate((prefix, np.asarray([length_m], dtype=np.float64)))


def _ellipse_values(theta: np.ndarray, spec: Level0TrackSpec) -> tuple[np.ndarray, ...]:
    a = spec.semi_major_axis_m
    b = spec.semi_minor_axis_m
    sine = np.sin(theta)
    cosine = np.cos(theta)
    position = np.column_stack((a * sine, b * (1.0 - cosine)))
    derivative = np.column_stack((a * cosine, b * sine))
    speed = np.linalg.norm(derivative, axis=1)
    tangent = derivative / speed[:, None]
    curvature = np.full(theta.shape, a * b, dtype=np.float64) / np.power(speed, 3)
    return position, tangent, curvature


def build_level0_candidate(spec: Level0TrackSpec | None = None) -> TrackCandidate:
    """Build the deterministic float64 Level 0 candidate for validation and packing."""

    spec = Level0TrackSpec() if spec is None else spec
    dense_theta = np.linspace(
        0.0,
        2.0 * np.pi,
        spec.dense_sample_count + 1,
        dtype=np.float64,
    )
    dense_position, _, _ = _ellipse_values(dense_theta, spec)
    dense_s = np.concatenate(
        (
            np.zeros(1, dtype=np.float64),
            np.cumsum(np.linalg.norm(np.diff(dense_position, axis=0), axis=1)),
        )
    )
    length_m = float(dense_s[-1])
    cumulative_s = _spaced_distances(length_m, spec.arc_spacing_m)
    theta = np.interp(cumulative_s, dense_s, dense_theta)
    centerline, tangent, curvature = _ellipse_values(theta, spec)
    centerline[-1] = centerline[0]
    tangent[-1] = tangent[0]
    curvature[-1] = curvature[0]

    normal = np.column_stack((-tangent[:, 1], tangent[:, 0]))
    left_boundary = centerline + 0.5 * spec.width_m * normal
    right_boundary = centerline - 0.5 * spec.width_m * normal
    left_boundary[-1] = left_boundary[0]
    right_boundary[-1] = right_boundary[0]

    checkpoint_s = np.arange(
        spec.checkpoint_spacing_m,
        length_m,
        spec.checkpoint_spacing_m,
        dtype=np.float64,
    )
    checkpoint_s = np.concatenate((checkpoint_s, np.asarray([length_m], dtype=np.float64)))
    checkpoint_theta = np.interp(checkpoint_s, dense_s, dense_theta)
    checkpoint_center, checkpoint_tangent, _ = _ellipse_values(checkpoint_theta, spec)
    checkpoint_center[-1] = centerline[0]
    checkpoint_tangent[-1] = tangent[0]

    control_theta = np.linspace(0.0, 2.0 * np.pi, 8, endpoint=False, dtype=np.float64)
    control_points, _, _ = _ellipse_values(control_theta, spec)
    return TrackCandidate(
        seed=LEVEL0_TRACK_SEED,
        generator_version=spec.generator_version,
        control_points_m=control_points,
        centerline_m=centerline,
        left_boundary_m=left_boundary,
        right_boundary_m=right_boundary,
        tangent=tangent,
        curvature_1pm=curvature,
        cumulative_s_m=cumulative_s,
        checkpoint_center_m=checkpoint_center,
        checkpoint_tangent=checkpoint_tangent,
        checkpoint_s_m=checkpoint_s,
        start_pose=np.zeros(3, dtype=np.float64),
        length_m=length_m,
        width_m=spec.width_m,
    )


def build_level0_track(
    capacity: TrackCapacity = DEFAULT_LEVEL0_CAPACITY,
    spec: Level0TrackSpec | None = None,
) -> Track:
    """Build the immutable fixed Level 0 Track in the benchmark representation.

    The start is the ellipse's low-curvature bottom point, normalized to the origin with a +x
    tangent. The reserved maximum uint32 seed keeps its public numeric Track ID disjoint from
    ordinary Level 1 generator seeds.
    """

    return pack_track(build_level0_candidate(spec), capacity)


__all__ = [
    "DEFAULT_LEVEL0_CAPACITY",
    "LEVEL0_TRACK_SEED",
    "Level0TrackSpec",
    "build_level0_candidate",
    "build_level0_track",
]
