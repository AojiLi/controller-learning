"""Unit tests for the internal conservative track-driveability policy."""

from __future__ import annotations

from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from controller_learning.config import load_project_config
from controller_learning.envs.configuration import race_core_config_from_project
from controller_learning.envs.race_core import project_to_track
from controller_learning.physics.mjx_warp import MjxWarpVehicleStateView
from controller_learning.tracks.driveability import (
    ConservativeDriveabilityPolicyConfig,
    conservative_driveability_action,
)
from controller_learning.tracks.generator import generate_track_candidate, pack_track
from controller_learning.tracks.specs import (
    generation_spec_from_project,
    track_capacity_from_project,
)
from controller_learning.tracks.types import stack_tracks

PROJECT_ROOT = Path(__file__).parents[3]


def _batch_and_projection():
    project_config = load_project_config(PROJECT_ROOT)
    track = pack_track(
        generate_track_candidate(42, generation_spec_from_project(project_config)),
        track_capacity_from_project(project_config),
    )
    batch = stack_tracks([track, track])
    positions = jnp.asarray(((0.0, 0.0), (0.25, -0.1)), dtype=jnp.float32)
    race_config = race_core_config_from_project(project_config)
    assert race_config.projection_backward_segments == 4
    assert race_config.projection_forward_segments == 12
    projection = project_to_track(
        batch,
        positions,
        jnp.zeros(2, dtype=jnp.int32),
        race_config,
    )
    return batch, positions, projection


def _vehicle_view(positions: jax.Array) -> MjxWarpVehicleStateView:
    count = positions.shape[0]
    zeros = jnp.zeros(count, dtype=jnp.float32)
    zeros3 = jnp.zeros((count, 3), dtype=jnp.float32)
    return MjxWarpVehicleStateView(
        time_s=zeros,
        position_world_m=jnp.column_stack((positions, zeros)),
        chassis_position_world_m=jnp.column_stack((positions, jnp.full(count, 0.56))),
        quaternion_wxyz=jnp.tile(jnp.asarray((1.0, 0.0, 0.0, 0.0)), (count, 1)),
        roll_rad=zeros,
        pitch_rad=zeros,
        yaw_rad=jnp.asarray((0.0, 2.5), dtype=jnp.float32),
        velocity_body_mps=zeros3.at[:, 0].set(jnp.asarray((0.0, 30.0))),
        angular_velocity_body_rad_s=zeros3.at[:, 2].set(jnp.asarray((0.0, -20.0))),
        steering_angle_rad=zeros,
        front_steering_angles_rad=jnp.zeros((count, 2), dtype=jnp.float32),
        wheel_angular_velocity_rad_s=jnp.zeros((count, 4), dtype=jnp.float32),
    )


def test_policy_is_deterministic_batched_and_action_bounded() -> None:
    batch, positions, projection = _batch_and_projection()
    view = _vehicle_view(positions)
    config = ConservativeDriveabilityPolicyConfig()
    compiled = jax.jit(
        lambda tracks, track_projection, state: conservative_driveability_action(
            tracks,
            track_projection,
            state,
            config,
        )
    )

    first = np.asarray(compiled(batch, projection, view))
    second = np.asarray(compiled(batch, projection, view))

    assert first.shape == (2, 2)
    assert first.dtype == np.float32
    assert np.array_equal(first, second)
    assert np.isfinite(first).all()
    assert np.all(np.abs(first[:, 0]) <= config.maximum_steering_angle_rad)
    assert np.all(first[:, 1] <= config.maximum_acceleration_mps2)
    assert np.all(first[:, 1] >= -config.maximum_deceleration_mps2)
    assert first[0, 1] > 0.0
    assert first[1, 1] == pytest.approx(-config.maximum_deceleration_mps2)


@pytest.mark.parametrize(
    ("field", "value"),
    (
        ("target_speed_mps", 0.0),
        ("minimum_corner_speed_mps", 5.0),
        ("maximum_lateral_acceleration_mps2", np.inf),
        ("lookahead_search_points", 0),
        ("curvature_preview_points", True),
        ("heading_error_gain", -0.1),
    ),
)
def test_policy_config_rejects_invalid_values(field: str, value: float | int) -> None:
    values: dict[str, float | int] = {
        "target_speed_mps": 4.0,
        "minimum_corner_speed_mps": 2.5,
        "maximum_lateral_acceleration_mps2": 1.5,
        "lookahead_search_points": 16,
        "curvature_preview_points": 32,
        "heading_error_gain": 0.15,
    }
    values[field] = value
    with pytest.raises(ValueError):
        ConservativeDriveabilityPolicyConfig(**values)  # type: ignore[arg-type]
