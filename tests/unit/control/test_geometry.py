"""Tests for public observation-only Controller path geometry."""

from __future__ import annotations

import inspect
from dataclasses import FrozenInstanceError

import numpy as np
import pytest

from controller_learning.control import geometry
from controller_learning.control.geometry import (
    CenterlineReference,
    PathProjection,
    body_to_world,
    world_to_body,
    wrap_angle,
)


def _observation(
    centerline: np.ndarray | None = None,
    *,
    capacity: int = 8,
) -> dict[str, np.ndarray]:
    if centerline is None:
        centerline = np.asarray(
            ((0.0, 0.0), (2.0, 0.0), (2.0, 2.0), (0.0, 2.0), (0.0, 0.0)),
            dtype=np.float32,
        )
    valid_count = len(centerline)
    if capacity < valid_count:
        raise ValueError("capacity is too small")
    center = np.zeros((capacity, 2), dtype=np.float32)
    left = np.zeros_like(center)
    right = np.zeros_like(center)
    center[:valid_count] = centerline
    left[:valid_count] = centerline + np.asarray((0.0, 1.0), dtype=np.float32)
    right[:valid_count] = centerline - np.asarray((0.0, 1.0), dtype=np.float32)
    mask = np.zeros(capacity, dtype=np.int8)
    mask[:valid_count] = 1
    length = np.linalg.norm(np.diff(centerline.astype(np.float64), axis=0), axis=1).sum()
    return {
        "centerline": center,
        "left_boundary": left,
        "right_boundary": right,
        "track_mask": mask,
        "track_length": np.asarray(length, dtype=np.float32),
    }


def _circle_observation(radius_m: float = 10.0, segments: int = 64) -> dict[str, np.ndarray]:
    angle = np.linspace(0.0, 2.0 * np.pi, segments + 1)
    center = np.stack((radius_m * np.cos(angle), radius_m * np.sin(angle)), axis=1)
    center[-1] = center[0]
    radial = center / radius_m
    observation = _observation(center.astype(np.float32), capacity=segments + 5)
    observation["left_boundary"][: segments + 1] = center - radial
    observation["right_boundary"][: segments + 1] = center + radial
    return observation


def test_frame_transforms_broadcast_and_are_inverse() -> None:
    vectors = np.asarray(((1.0, 0.0), (0.0, 2.0)))
    yaw = np.asarray((np.pi / 2.0, -np.pi / 2.0))
    body = world_to_body(vectors, yaw)
    np.testing.assert_allclose(body, ((0.0, -1.0), (-2.0, 0.0)), atol=1e-12)
    np.testing.assert_allclose(body_to_world(body, yaw), vectors, atol=1e-12)
    assert wrap_angle(3.0 * np.pi) == pytest.approx(-np.pi)
    np.testing.assert_allclose(wrap_angle(np.asarray((-3.0 * np.pi, 2.0 * np.pi))), (-np.pi, 0.0))


def test_square_projection_has_left_positive_sign_and_readonly_results() -> None:
    reference = CenterlineReference.from_observation(_observation())
    projection = reference.project((0.75, 0.5))
    assert projection.segment_index == 0
    assert projection.segment_fraction == pytest.approx(0.375)
    assert projection.s_m == pytest.approx(0.75)
    assert projection.lateral_error_m == pytest.approx(0.5)
    assert projection.distance_m == pytest.approx(0.5)
    np.testing.assert_allclose(projection.point_m, (0.75, 0.0))
    np.testing.assert_allclose(projection.tangent, (1.0, 0.0))
    assert not projection.point_m.flags.writeable
    with pytest.raises(ValueError):
        projection.point_m[0] = 2.0
    with pytest.raises(FrozenInstanceError):
        projection.distance_m = 2.0  # type: ignore[misc]

    right = reference.project((0.75, -0.25))
    assert right.lateral_error_m == pytest.approx(-0.25)


def test_periodic_sample_preserves_scalar_and_array_shapes_across_seam() -> None:
    reference = CenterlineReference.from_observation(_observation())
    scalar = reference.sample(0.5)
    assert isinstance(scalar.s_m, float)
    assert isinstance(scalar.curvature_1pm, float)
    assert scalar.center_m.shape == (2,)
    np.testing.assert_allclose(scalar.center_m, (0.5, 0.0))

    sampled = reference.sample(np.asarray((0.0, 8.0, 8.5, -0.5)))
    assert isinstance(sampled.s_m, np.ndarray)
    assert sampled.s_m.shape == (4,)
    assert sampled.center_m.shape == (4, 2)
    np.testing.assert_allclose(sampled.s_m, (0.0, 0.0, 0.5, 7.5))
    np.testing.assert_allclose(sampled.center_m[0], sampled.center_m[1])
    np.testing.assert_allclose(sampled.center_m[2], (0.5, 0.0))
    np.testing.assert_allclose(sampled.center_m[3], (0.0, 0.5))
    assert not sampled.center_m.flags.writeable

    preview = reference.preview(7.5, np.asarray((0.0, 0.5, 1.0)))
    np.testing.assert_allclose(preview.s_m, (7.5, 0.0, 0.5))


def test_circle_caches_positive_signed_curvature_and_periodic_tangent() -> None:
    reference = CenterlineReference.from_observation(_circle_observation())
    assert reference.segment_count == 64
    assert np.all(reference.curvature_1pm > 0.0)
    assert np.median(reference.curvature_1pm) == pytest.approx(0.1, rel=2e-3)
    np.testing.assert_allclose(reference.tangent[0], reference.tangent[-1])
    np.testing.assert_allclose(reference.curvature_1pm[0], reference.curvature_1pm[-1])
    sampled = reference.sample(np.linspace(0.0, reference.track_length_m, 9))
    np.testing.assert_allclose(np.linalg.norm(sampled.tangent, axis=1), 1.0, atol=1e-12)
    np.testing.assert_allclose(sampled.center_m[0], sampled.center_m[-1])


def test_hint_search_is_local_and_wraps_modulo_seam() -> None:
    angle = np.linspace(0.0, 2.0 * np.pi, 12, endpoint=False)
    points = np.stack((10.0 * np.cos(angle), 10.0 * np.sin(angle)), axis=1)
    points = np.concatenate((points, points[:1]), axis=0)
    reference = CenterlineReference.from_observation(_observation(points, capacity=16))

    global_projection = reference.project(points[6])
    assert global_projection.distance_m == pytest.approx(0.0)
    local_projection = reference.project(
        points[6], hint_segment=0, backward_segments=0, forward_segments=0
    )
    assert local_projection.segment_index == 0
    assert local_projection.distance_m > 10.0
    seam_projection = reference.project(
        points[-2], hint_segment=0, backward_segments=1, forward_segments=0
    )
    assert seam_projection.segment_index == reference.segment_count - 1
    assert seam_projection.distance_m == pytest.approx(0.0, abs=1e-6)


def test_hint_tie_prefers_current_segment_at_explicit_closure() -> None:
    reference = CenterlineReference.from_observation(_observation())

    projection = reference.project(
        reference.centerline_m[0],
        hint_segment=0,
        backward_segments=2,
        forward_segments=2,
    )

    assert projection.segment_index == 0


def test_valid_prefix_is_owned_readonly_and_padding_is_ignored() -> None:
    observation = _observation()
    observation["centerline"][5:] = np.nan
    observation["left_boundary"][5:] = np.inf
    observation["right_boundary"][5:] = -np.inf
    reference = CenterlineReference.from_observation(observation)
    original = reference.centerline_m.copy()
    observation["centerline"][:5] = 999.0
    np.testing.assert_array_equal(reference.centerline_m, original)
    assert reference.point_count == 5
    for value in (
        reference.centerline_m,
        reference.left_boundary_m,
        reference.right_boundary_m,
        reference.segment_delta_m,
        reference.segment_length_m,
        reference.segment_tangent,
        reference.cumulative_s_m,
        reference.tangent,
        reference.curvature_1pm,
    ):
        assert not value.flags.writeable


@pytest.mark.parametrize(
    ("mutation", "match"),
    [
        (lambda obs: obs.pop("centerline"), "missing geometry"),
        (
            lambda obs: obs["track_mask"].__setitem__(slice(None), (1, 1, 0, 1, 1, 0, 0, 0)),
            "contiguous",
        ),
        (lambda obs: obs["centerline"].__setitem__((4, 0), 0.1), "explicitly closed"),
        (lambda obs: obs["left_boundary"].__setitem__((4, 0), 0.1), "explicitly closed"),
        (lambda obs: obs["centerline"].__setitem__(1, obs["centerline"][0]), "positive length"),
        (lambda obs: obs.__setitem__("track_length", np.asarray(12.0)), "inconsistent"),
        (lambda obs: obs["centerline"].__setitem__((1, 0), np.nan), "finite"),
    ],
)
def test_invalid_public_geometry_is_rejected(mutation, match: str) -> None:
    observation = _observation()
    mutation(observation)
    with pytest.raises(ValueError, match=match):
        CenterlineReference.from_observation(observation)


def test_transform_and_projection_input_boundaries_are_rejected() -> None:
    reference = CenterlineReference.from_observation(_observation())
    with pytest.raises(ValueError, match="shape"):
        reference.project((1.0, 2.0, 3.0))
    with pytest.raises(ValueError, match="non-negative"):
        reference.project((0.0, 0.0), hint_segment=0, backward_segments=-1)
    with pytest.raises(TypeError, match="integer"):
        reference.project((0.0, 0.0), hint_segment=1.5)  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="finite"):
        reference.sample((0.0, np.inf))
    with pytest.raises(ValueError, match="finite"):
        world_to_body((1.0, np.nan), 0.0)


def test_projection_value_object_copies_constructor_arrays() -> None:
    point = np.asarray((1.0, 2.0))
    projection = PathProjection(0, 0.5, 0.5, point, np.asarray((1.0, 0.0)), 2.0, 2.0)
    point[0] = 9.0
    np.testing.assert_array_equal(projection.point_m, (1.0, 2.0))


def test_public_geometry_source_has_no_challenge_or_backend_imports() -> None:
    source = inspect.getsource(geometry)

    for forbidden in (
        "controller_learning.envs",
        "race_core",
        "controller_learning.tracks",
        "TrackBatch",
        "controller_learning.physics",
        "import jax",
        "import mujoco",
        "import warp",
    ):
        assert forbidden not in source
