"""Deterministic offline candidates for closed planar racing tracks."""

from __future__ import annotations

from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Any

import numpy as np
from numpy.typing import NDArray
from scipy.interpolate import CubicSpline

from controller_learning.tracks.types import Track, TrackCapacity


class TrackGenerationError(RuntimeError):
    """A structured, deterministic failure for one generated candidate."""

    def __init__(self, reason: str, message: str, **context: Any) -> None:
        super().__init__(message)
        self.reason = reason
        self.context = MappingProxyType(dict(context))


@dataclass(frozen=True, slots=True)
class TrackGenerationSpec:
    """Versioned parameters for the initial Level 1 generator spike."""

    min_control_points: int = 8
    max_control_points: int = 16
    min_radius_m: float = 52.0
    max_radius_m: float = 88.0
    angular_gap_jitter: float = 0.18
    radial_perturbation: float = 0.16
    width_m: float = 7.0
    arc_spacing_m: float = 1.0
    checkpoint_spacing_m: float = 15.0
    min_length_m: float = 300.0
    max_length_m: float = 600.0
    start_window_m: float = 25.0
    start_max_curvature_1pm: float = 1.0 / 40.0
    generator_version: str = "v0.1"
    dense_samples_per_control_point: int = 1024
    arc_length_convergence_m: float = 2.0e-3
    tail_merge_fraction: float = 0.5

    def __post_init__(self) -> None:
        if not 4 <= self.min_control_points <= self.max_control_points:
            raise ValueError("control-point bounds must be ordered and at least four")
        if not 0.0 < self.min_radius_m < self.max_radius_m:
            raise ValueError("radius bounds must be finite, positive, and ordered")
        if not 0.0 <= self.angular_gap_jitter < 1.0:
            raise ValueError("angular_gap_jitter must be in [0, 1)")
        if not 0.0 <= self.radial_perturbation < 1.0:
            raise ValueError("radial_perturbation must be in [0, 1)")
        positive = (
            self.width_m,
            self.arc_spacing_m,
            self.checkpoint_spacing_m,
            self.start_window_m,
            self.start_max_curvature_1pm,
            self.arc_length_convergence_m,
        )
        if not all(np.isfinite(value) and value > 0.0 for value in positive):
            raise ValueError("distance and curvature parameters must be finite and positive")
        if not 0.0 < self.min_length_m < self.max_length_m:
            raise ValueError("length bounds must be finite, positive, and ordered")
        if self.dense_samples_per_control_point < 64:
            raise ValueError("dense_samples_per_control_point must be at least 64")
        if not 0.0 < self.tail_merge_fraction <= 1.0:
            raise ValueError("tail_merge_fraction must be in (0, 1]")
        if not self.generator_version:
            raise ValueError("generator_version cannot be empty")


def _immutable_float64(
    value: object, shape_tail: tuple[int, ...], field_name: str
) -> NDArray[np.float64]:
    array = np.array(value, dtype=np.float64, copy=True)
    if array.ndim != 1 + len(shape_tail) or array.shape[1:] != shape_tail:
        raise ValueError(f"{field_name} has an invalid shape")
    if not np.isfinite(array).all():
        raise ValueError(f"{field_name} must contain only finite values")
    array.setflags(write=False)
    return array


@dataclass(frozen=True, slots=True)
class TrackCandidate:
    """One immutable, dynamically sized, float64 generator result."""

    seed: int
    generator_version: str
    control_points_m: NDArray[np.float64]
    centerline_m: NDArray[np.float64]
    left_boundary_m: NDArray[np.float64]
    right_boundary_m: NDArray[np.float64]
    tangent: NDArray[np.float64]
    curvature_1pm: NDArray[np.float64]
    cumulative_s_m: NDArray[np.float64]
    checkpoint_center_m: NDArray[np.float64]
    checkpoint_tangent: NDArray[np.float64]
    checkpoint_s_m: NDArray[np.float64]
    start_pose: NDArray[np.float64]
    length_m: float
    width_m: float

    def __post_init__(self) -> None:
        if not 0 <= self.seed <= np.iinfo(np.uint32).max:
            raise ValueError("seed must fit in uint32")
        point_fields = (
            "centerline_m",
            "left_boundary_m",
            "right_boundary_m",
            "tangent",
        )
        for name in (
            "control_points_m",
            *point_fields,
            "checkpoint_center_m",
            "checkpoint_tangent",
        ):
            object.__setattr__(self, name, _immutable_float64(getattr(self, name), (2,), name))
        for name in ("curvature_1pm", "cumulative_s_m", "checkpoint_s_m"):
            object.__setattr__(self, name, _immutable_float64(getattr(self, name), (), name))
        start_pose = np.array(self.start_pose, dtype=np.float64, copy=True)
        if start_pose.shape != (3,) or not np.isfinite(start_pose).all():
            raise ValueError("start_pose must be a finite (3,) array")
        start_pose.setflags(write=False)
        object.__setattr__(self, "start_pose", start_pose)
        point_count = self.centerline_m.shape[0]
        if point_count < 4 or any(
            getattr(self, name).shape[0] != point_count for name in point_fields[1:]
        ):
            raise ValueError("all sampled track arrays must share one point count")
        if (
            self.curvature_1pm.shape[0] != point_count
            or self.cumulative_s_m.shape[0] != point_count
        ):
            raise ValueError("scalar track arrays must share the centerline point count")
        checkpoint_count = self.checkpoint_s_m.shape[0]
        if (
            checkpoint_count < 1
            or self.checkpoint_center_m.shape[0] != checkpoint_count
            or self.checkpoint_tangent.shape[0] != checkpoint_count
        ):
            raise ValueError("checkpoint arrays must share one nonzero count")
        if not np.array_equal(self.centerline_m[0], self.centerline_m[-1]):
            raise ValueError("centerline must include explicit closure")

    @property
    def point_count(self) -> int:
        return self.centerline_m.shape[0]

    @property
    def checkpoint_count(self) -> int:
        return self.checkpoint_s_m.shape[0]


@dataclass(frozen=True, slots=True)
class _ArcTable:
    parameter: NDArray[np.float64]
    cumulative_s_m: NDArray[np.float64]
    length_m: float
    spline: CubicSpline = field(repr=False)


def _control_points(rng: np.random.Generator, spec: TrackGenerationSpec) -> NDArray[np.float64]:
    count = int(rng.integers(spec.min_control_points, spec.max_control_points + 1))
    gap_weights = 1.0 + rng.uniform(-spec.angular_gap_jitter, spec.angular_gap_jitter, count)
    gaps = 2.0 * np.pi * gap_weights / np.sum(gap_weights)
    angles = np.cumsum(np.concatenate(([0.0], gaps[:-1])))
    angles += rng.uniform(0.0, 2.0 * np.pi)

    base_radius = rng.uniform(spec.min_radius_m, spec.max_radius_m)
    noise = rng.normal(size=count)
    smooth = (np.roll(noise, 1) + 2.0 * noise + np.roll(noise, -1)) / 4.0
    scale = np.max(np.abs(smooth))
    if scale > 0.0:
        smooth /= scale
    # ``min_radius_m`` and ``max_radius_m`` bound the sampled base scale.  The
    # perturbation is deliberately applied around that scale; clipping it back to
    # the base bounds would collapse the tails of the intended 300--600 m
    # distribution and make the radial-variation parameter seed-dependent in
    # effect.
    radii = base_radius * (1.0 + spec.radial_perturbation * smooth)
    return np.column_stack((radii * np.cos(angles), radii * np.sin(angles)))


def _arc_table(control_points: NDArray[np.float64], spec: TrackGenerationSpec) -> _ArcTable:
    closed = np.concatenate((control_points, control_points[:1]), axis=0)
    chords = np.linalg.norm(np.diff(closed, axis=0), axis=1)
    if np.any(chords <= 1.0e-9):
        raise TrackGenerationError(
            "degenerate_control_points", "control points contain a zero chord"
        )
    chord_s = np.concatenate(([0.0], np.cumsum(chords)))
    spline = CubicSpline(chord_s, closed, bc_type="periodic", axis=0)
    dense_count = control_points.shape[0] * spec.dense_samples_per_control_point
    parameter = np.linspace(chord_s[0], chord_s[-1], dense_count + 1, dtype=np.float64)
    positions = spline(parameter)
    segment_length = np.linalg.norm(np.diff(positions, axis=0), axis=1)
    cumulative = np.concatenate(([0.0], np.cumsum(segment_length)))
    # Check that halving the dense table changes its arc estimate by less than the contract.
    coarse_length = np.sum(np.linalg.norm(np.diff(positions[::2], axis=0), axis=1))
    residual = float(cumulative[-1] - coarse_length)
    if residual > spec.arc_length_convergence_m:
        raise TrackGenerationError(
            "arc_length_not_converged",
            "dense arc-length table did not meet its convergence tolerance",
            residual_m=residual,
        )
    return _ArcTable(parameter, cumulative, float(cumulative[-1]), spline)


def _sample_distances(length_m: float, spec: TrackGenerationSpec) -> NDArray[np.float64]:
    spacing = spec.arc_spacing_m
    full_steps = int(np.floor(length_m / spacing))
    remainder = length_m - full_steps * spacing
    if remainder <= 1.0e-10:
        distances = np.arange(full_steps + 1, dtype=np.float64) * spacing
    elif remainder < spacing * spec.tail_merge_fraction:
        distances = np.concatenate((np.arange(full_steps, dtype=np.float64) * spacing, [length_m]))
    else:
        distances = np.concatenate(
            (np.arange(full_steps + 1, dtype=np.float64) * spacing, [length_m])
        )
    distances[-1] = length_m
    return distances


def _evaluate(
    table: _ArcTable, source_s_m: NDArray[np.float64]
) -> tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64]]:
    wrapped = np.mod(source_s_m, table.length_m)
    parameter = np.interp(wrapped, table.cumulative_s_m, table.parameter)
    position = np.asarray(table.spline(parameter), dtype=np.float64)
    first = np.asarray(table.spline(parameter, 1), dtype=np.float64)
    second = np.asarray(table.spline(parameter, 2), dtype=np.float64)
    speed = np.linalg.norm(first, axis=1)
    if np.any(speed <= 1.0e-10):
        raise TrackGenerationError("degenerate_tangent", "spline contains a degenerate tangent")
    tangent = first / speed[:, None]
    curvature = (first[:, 0] * second[:, 1] - first[:, 1] * second[:, 0]) / speed**3
    return position, tangent, curvature


def generate_track_candidate(seed: int, spec: TrackGenerationSpec | None = None) -> TrackCandidate:
    """Generate exactly one candidate from ``PCG64(seed)`` without hidden retries."""

    spec = TrackGenerationSpec() if spec is None else spec
    if not 0 <= seed <= np.iinfo(np.uint32).max:
        raise TrackGenerationError("invalid_seed", "seed must fit in uint32", seed=seed)
    rng = np.random.Generator(np.random.PCG64(seed))
    control_points = _control_points(rng, spec)
    signed_area = 0.5 * np.sum(
        control_points[:, 0] * np.roll(control_points[:, 1], -1)
        - np.roll(control_points[:, 0], -1) * control_points[:, 1]
    )
    if signed_area < 0.0:
        control_points = control_points[::-1].copy()
    table = _arc_table(control_points, spec)
    if not spec.min_length_m <= table.length_m <= spec.max_length_m:
        raise TrackGenerationError(
            "length_out_of_range",
            "candidate length is outside the generator bounds",
            length_m=table.length_m,
        )

    distances = _sample_distances(table.length_m, spec)
    _, _, initial_curvature = _evaluate(table, distances[:-1])
    window_points = max(1, int(np.ceil(spec.start_window_m / spec.arc_spacing_m)))
    scores = np.array(
        [
            np.max(
                np.abs(
                    np.take(initial_curvature, np.arange(index, index + window_points), mode="wrap")
                )
            )
            for index in range(initial_curvature.size)
        ]
    )
    start_index = int(np.argmin(scores))
    if scores[start_index] > spec.start_max_curvature_1pm:
        raise TrackGenerationError(
            "no_straight_start",
            "candidate has no start window below the curvature limit",
            best_max_curvature_1pm=float(scores[start_index]),
        )
    start_s = distances[start_index]
    position, tangent, curvature = _evaluate(table, start_s + distances)
    position[-1] = position[0]
    tangent[-1] = tangent[0]
    curvature[-1] = curvature[0]

    origin = position[0].copy()
    heading = float(np.arctan2(tangent[0, 1], tangent[0, 0]))
    cosine, sine = np.cos(-heading), np.sin(-heading)
    rotation = np.array(((cosine, -sine), (sine, cosine)), dtype=np.float64)
    position = (position - origin) @ rotation.T
    tangent = tangent @ rotation.T
    position[0] = position[-1] = 0.0
    tangent[0] = tangent[-1] = (1.0, 0.0)
    normal = np.column_stack((-tangent[:, 1], tangent[:, 0]))
    left_boundary = position + 0.5 * spec.width_m * normal
    right_boundary = position - 0.5 * spec.width_m * normal
    left_boundary[-1] = left_boundary[0]
    right_boundary[-1] = right_boundary[0]

    checkpoint_s = np.arange(
        spec.checkpoint_spacing_m,
        table.length_m,
        spec.checkpoint_spacing_m,
        dtype=np.float64,
    )
    checkpoint_s = np.concatenate((checkpoint_s, [table.length_m]))
    checkpoint_position, checkpoint_tangent, _ = _evaluate(table, start_s + checkpoint_s)
    checkpoint_position = (checkpoint_position - origin) @ rotation.T
    checkpoint_tangent = checkpoint_tangent @ rotation.T
    checkpoint_position[-1] = 0.0
    checkpoint_tangent[-1] = (1.0, 0.0)

    transformed_controls = (control_points - origin) @ rotation.T
    return TrackCandidate(
        seed=seed,
        generator_version=spec.generator_version,
        control_points_m=transformed_controls,
        centerline_m=position,
        left_boundary_m=left_boundary,
        right_boundary_m=right_boundary,
        tangent=tangent,
        curvature_1pm=curvature,
        cumulative_s_m=distances,
        checkpoint_center_m=checkpoint_position,
        checkpoint_tangent=checkpoint_tangent,
        checkpoint_s_m=checkpoint_s,
        start_pose=np.zeros(3, dtype=np.float64),
        length_m=table.length_m,
        width_m=spec.width_m,
    )


def pack_track(candidate: TrackCandidate, capacity: TrackCapacity) -> Track:
    """Pack a dynamic candidate into fixed float32 arrays, rejecting overflow."""

    if candidate.point_count > capacity.max_track_points:
        raise TrackGenerationError(
            "track_capacity_overflow",
            "candidate exceeds max_track_points",
            required=candidate.point_count,
            capacity=capacity.max_track_points,
        )
    if candidate.checkpoint_count > capacity.max_checkpoints:
        raise TrackGenerationError(
            "checkpoint_capacity_overflow",
            "candidate exceeds max_checkpoints",
            required=candidate.checkpoint_count,
            capacity=capacity.max_checkpoints,
        )

    def padded(source: NDArray[np.float64], shape: tuple[int, ...]) -> NDArray[np.float32]:
        target = np.zeros(shape, dtype=np.float32)
        target[: source.shape[0]] = source
        return target

    point_shape = (capacity.max_track_points,)
    checkpoint_shape = (capacity.max_checkpoints,)
    track_mask = np.arange(capacity.max_track_points) < candidate.point_count
    checkpoint_mask = np.arange(capacity.max_checkpoints) < candidate.checkpoint_count
    return Track(
        seed=candidate.seed,
        generator_version=candidate.generator_version,
        centerline_m=padded(candidate.centerline_m, (*point_shape, 2)),
        left_boundary_m=padded(candidate.left_boundary_m, (*point_shape, 2)),
        right_boundary_m=padded(candidate.right_boundary_m, (*point_shape, 2)),
        tangent=padded(candidate.tangent, (*point_shape, 2)),
        curvature_1pm=padded(candidate.curvature_1pm, point_shape),
        cumulative_s_m=padded(candidate.cumulative_s_m, point_shape),
        track_mask=track_mask,
        checkpoint_center_m=padded(candidate.checkpoint_center_m, (*checkpoint_shape, 2)),
        checkpoint_tangent=padded(candidate.checkpoint_tangent, (*checkpoint_shape, 2)),
        checkpoint_s_m=padded(candidate.checkpoint_s_m, checkpoint_shape),
        checkpoint_mask=checkpoint_mask,
        start_pose=candidate.start_pose.astype(np.float32),
        point_count=candidate.point_count,
        checkpoint_count=candidate.checkpoint_count,
        length_m=candidate.length_m,
        width_m=candidate.width_m,
    )


__all__ = [
    "TrackCandidate",
    "TrackGenerationError",
    "TrackGenerationSpec",
    "generate_track_candidate",
    "pack_track",
]
