"""Tests for the exact public observation and action contract."""

from __future__ import annotations

from pathlib import Path
from typing import NamedTuple

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from controller_learning.config import load_project_config
from controller_learning.envs.observation import (
    OBSERVATION_KEYS,
    action_space,
    batched_action_space,
    batched_observation_space,
    encode_batched_observation,
    observation_space,
    observation_to_host,
    unbatch_observation,
)
from controller_learning.envs.race_core import RaceState
from controller_learning.tracks.types import TrackBatch

PROJECT_ROOT = Path(__file__).parents[3]


class _VehicleView(NamedTuple):
    position_world_m: jax.Array
    yaw_rad: jax.Array
    velocity_body_mps: jax.Array
    angular_velocity_body_rad_s: jax.Array
    steering_angle_rad: jax.Array


def _track_batch(*, num_worlds: int, coordinate_offset: float = 0.0) -> TrackBatch:
    project = load_project_config(PROJECT_ROOT)
    points = project.track.representation.max_track_points
    checkpoints = project.track.representation.max_checkpoints
    centerline = np.zeros((num_worlds, points, 2), dtype=np.float32)
    centerline[..., 0] = coordinate_offset
    left = centerline.copy()
    left[..., 1] = 3.5
    right = centerline.copy()
    right[..., 1] = -3.5
    tangent = np.zeros_like(centerline)
    tangent[..., 0] = 1.0
    track_mask = np.zeros((num_worlds, points), dtype=bool)
    track_mask[:, :401] = True
    checkpoint_mask = np.zeros((num_worlds, checkpoints), dtype=bool)
    checkpoint_mask[:, :27] = True
    return TrackBatch(
        seed=np.arange(num_worlds, dtype=np.uint32),
        centerline_m=centerline,
        left_boundary_m=left,
        right_boundary_m=right,
        tangent=tangent,
        curvature_1pm=np.zeros((num_worlds, points), dtype=np.float32),
        cumulative_s_m=np.zeros((num_worlds, points), dtype=np.float32),
        track_mask=track_mask,
        checkpoint_center_m=np.zeros((num_worlds, checkpoints, 2), dtype=np.float32),
        checkpoint_tangent=np.zeros((num_worlds, checkpoints, 2), dtype=np.float32),
        checkpoint_s_m=np.zeros((num_worlds, checkpoints), dtype=np.float32),
        checkpoint_mask=checkpoint_mask,
        start_pose=np.zeros((num_worlds, 3), dtype=np.float32),
        point_count=np.full(num_worlds, 401, dtype=np.int32),
        checkpoint_count=np.full(num_worlds, 27, dtype=np.int32),
        length_m=np.full(num_worlds, 400.0, dtype=np.float32),
        width_m=np.full(num_worlds, 7.0, dtype=np.float32),
    )


def _race_state(progress: np.ndarray) -> RaceState:
    worlds = progress.shape[0]
    return RaceState(
        previous_position_m=jnp.zeros((worlds, 2), dtype=jnp.float32),
        segment_index=jnp.zeros(worlds, dtype=jnp.int32),
        projected_s_m=jnp.asarray(progress, dtype=jnp.float32),
        unwrapped_s_m=jnp.asarray(progress, dtype=jnp.float32),
        legal_progress_m=jnp.asarray(progress, dtype=jnp.float32),
        next_checkpoint_index=jnp.zeros(worlds, dtype=jnp.int32),
        elapsed_steps=jnp.zeros(worlds, dtype=jnp.int32),
    )


def _vehicle_view(num_worlds: int) -> _VehicleView:
    return _VehicleView(
        position_world_m=jnp.tile(
            jnp.asarray(((2.0, -3.0, 0.4),), dtype=jnp.float32),
            (num_worlds, 1),
        ),
        yaw_rad=jnp.full(num_worlds, 0.25, dtype=jnp.float32),
        velocity_body_mps=jnp.tile(
            jnp.asarray(((4.0, -0.5, 0.1),), dtype=jnp.float32),
            (num_worlds, 1),
        ),
        angular_velocity_body_rad_s=jnp.tile(
            jnp.asarray(((0.1, -0.2, 0.3),), dtype=jnp.float32),
            (num_worlds, 1),
        ),
        steering_angle_rad=jnp.full(num_worlds, -0.15, dtype=jnp.float32),
    )


def test_single_observation_space_has_exact_public_schema() -> None:
    project = load_project_config(PROJECT_ROOT)
    space = observation_space(project)
    points = project.track.representation.max_track_points

    assert tuple(space.spaces) == OBSERVATION_KEYS
    expected_shapes = {
        "position": (2,),
        "yaw": (),
        "velocity_body": (2,),
        "yaw_rate": (),
        "steering_angle": (),
        "track_progress": (),
        "centerline": (points, 2),
        "left_boundary": (points, 2),
        "right_boundary": (points, 2),
        "track_mask": (points,),
        "track_length": (),
    }
    for key, shape in expected_shapes.items():
        assert space[key].shape == shape
        assert space[key].dtype == (np.dtype(np.int8) if key == "track_mask" else np.float32)

    assert float(space["track_progress"].low) == 0.0
    assert float(space["track_progress"].high) == 1.0
    assert float(space["yaw"].low) == pytest.approx(-np.pi)
    assert float(space["yaw"].high) == pytest.approx(np.pi)
    assert float(space["steering_angle"].low) == pytest.approx(
        -project.vehicle.actuator.max_steering_angle_rad
    )
    assert float(space["steering_angle"].high) == pytest.approx(
        project.vehicle.actuator.max_steering_angle_rad
    )
    assert float(space["track_length"].low) == project.track.validation.min_length_m
    assert float(space["track_length"].high) == project.track.validation.max_length_m


def test_encoded_batch_has_exact_shapes_dtypes_and_no_privileged_fields() -> None:
    project = load_project_config(PROJECT_ROOT)
    points = project.track.representation.max_track_points
    tracks = _track_batch(num_worlds=2)
    encoded = encode_batched_observation(
        tracks,
        _race_state(np.asarray((100.0, 200.0), dtype=np.float32)),
        _vehicle_view(2),
    )
    host = observation_to_host(encoded)

    assert tuple(host) == OBSERVATION_KEYS
    assert {"lateral_error", "heading_error", "target_speed", "segment_index"}.isdisjoint(host)
    assert {"checkpoint", "future_state", "simulator", "data"}.isdisjoint(host)
    assert host["position"].shape == (2, 2)
    assert host["yaw"].shape == (2,)
    assert host["velocity_body"].shape == (2, 2)
    assert host["yaw_rate"].shape == (2,)
    assert host["steering_angle"].shape == (2,)
    assert host["track_progress"].shape == (2,)
    assert host["centerline"].shape == (2, points, 2)
    assert host["left_boundary"].shape == (2, points, 2)
    assert host["right_boundary"].shape == (2, points, 2)
    assert host["track_mask"].shape == (2, points)
    assert host["track_length"].shape == (2,)
    assert all(value.dtype == np.float32 for key, value in host.items() if key != "track_mask")
    assert host["track_mask"].dtype == np.int8
    np.testing.assert_array_equal(host["position"], ((2.0, -3.0), (2.0, -3.0)))
    np.testing.assert_array_equal(host["velocity_body"], ((4.0, -0.5), (4.0, -0.5)))
    np.testing.assert_allclose(host["yaw_rate"], 0.3)
    np.testing.assert_allclose(host["track_progress"], (0.25, 0.5))
    assert batched_observation_space(project, 2).contains(host)


def test_batch_one_unbatches_to_the_exact_single_observation() -> None:
    project = load_project_config(PROJECT_ROOT)
    encoded = encode_batched_observation(
        _track_batch(num_worlds=1),
        _race_state(np.asarray((80.0,), dtype=np.float32)),
        _vehicle_view(1),
    )
    host_batch = observation_to_host(encoded)
    single = unbatch_observation(encoded)

    for key in OBSERVATION_KEYS:
        np.testing.assert_array_equal(single[key], host_batch[key][0])
        assert single[key].dtype == host_batch[key].dtype
    assert observation_space(project).contains(single)
    assert batched_observation_space(project, 1).contains(host_batch)


def test_progress_is_clipped_to_public_unit_interval() -> None:
    encoded = encode_batched_observation(
        _track_batch(num_worlds=2),
        _race_state(np.asarray((-10.0, 800.0), dtype=np.float32)),
        _vehicle_view(2),
    )
    np.testing.assert_array_equal(np.asarray(encoded["track_progress"]), (0.0, 1.0))


def test_one_jit_executable_accepts_different_same_shape_track_values() -> None:
    first_tracks = jax.tree.map(jnp.asarray, _track_batch(num_worlds=1, coordinate_offset=0.0))
    second_tracks = jax.tree.map(jnp.asarray, _track_batch(num_worlds=1, coordinate_offset=17.0))
    state = _race_state(np.asarray((100.0,), dtype=np.float32))
    vehicle = _vehicle_view(1)
    compiled = jax.jit(encode_batched_observation).lower(first_tracks, state, vehicle).compile()

    first = compiled(first_tracks, state, vehicle)
    second = compiled(second_tracks, state, vehicle)

    assert float(first["centerline"][0, 0, 0]) == 0.0
    assert float(second["centerline"][0, 0, 0]) == 17.0
    assert compiled.runtime_executable() is not None


def test_action_spaces_use_exact_physical_limits_and_float32() -> None:
    project = load_project_config(PROJECT_ROOT)
    actuator = project.vehicle.actuator
    single = action_space(project)
    vector = batched_action_space(project, 3)

    np.testing.assert_array_equal(
        single.low,
        np.asarray(
            (-actuator.max_steering_angle_rad, -actuator.max_deceleration_mps2),
            dtype=np.float32,
        ),
    )
    np.testing.assert_array_equal(
        single.high,
        np.asarray(
            (actuator.max_steering_angle_rad, actuator.max_acceleration_mps2),
            dtype=np.float32,
        ),
    )
    assert single.shape == (2,)
    assert single.dtype == np.float32
    assert vector.shape == (3, 2)
    assert vector.dtype == np.float32
    np.testing.assert_array_equal(vector.low, np.broadcast_to(single.low, (3, 2)))
    np.testing.assert_array_equal(vector.high, np.broadcast_to(single.high, (3, 2)))


@pytest.mark.parametrize("num_envs", (0, -1, True, 1.5))
def test_batched_spaces_reject_invalid_world_counts(num_envs: object) -> None:
    project = load_project_config(PROJECT_ROOT)
    with pytest.raises(ValueError, match="positive integer"):
        batched_observation_space(project, num_envs)  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="positive integer"):
        batched_action_space(project, num_envs)  # type: ignore[arg-type]
