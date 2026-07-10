"""GPU smoke test for generated-track driveability on the formal four-wheel backend."""

from __future__ import annotations

from pathlib import Path

import jax
import numpy as np
import pytest

from controller_learning.config import load_project_config
from controller_learning.envs.configuration import race_core_config_from_project
from controller_learning.envs.race_core import (
    project_to_track,
    reset_race_state,
    step_race_core,
)
from controller_learning.physics.mjx_warp import MjxWarpVehicle
from controller_learning.tracks.driveability import (
    ConservativeDriveabilityPolicyConfig,
    conservative_driveability_action,
)
from controller_learning.tracks.generator import generate_track_candidate, pack_track
from controller_learning.tracks.specs import (
    generation_spec_from_project,
    track_capacity_from_project,
    validation_spec_from_project,
)
from controller_learning.tracks.types import stack_tracks
from controller_learning.tracks.validator import validate_track_candidate

PROJECT_ROOT = Path(__file__).parents[2]
pytestmark = pytest.mark.gpu


def test_reference_policy_advances_independent_generated_tracks() -> None:
    project_config = load_project_config(PROJECT_ROOT)
    generation_spec = generation_spec_from_project(project_config)
    validation_spec = validation_spec_from_project(project_config)
    candidates = [generate_track_candidate(seed, generation_spec) for seed in (0, 1)]
    assert all(
        validate_track_candidate(candidate, validation_spec).valid for candidate in candidates
    )
    capacity = track_capacity_from_project(project_config)
    tracks = [pack_track(candidate, capacity) for candidate in candidates]
    track_batch = stack_tracks(tracks)
    vehicle_config = project_config.vehicle
    vehicle = MjxWarpVehicle.create(vehicle_config, num_worlds=len(tracks))
    physics_state = vehicle.initial_state(track_batch.start_pose)
    race_state = reset_race_state(track_batch)
    race_config = race_core_config_from_project(project_config)
    policy_config = ConservativeDriveabilityPolicyConfig(
        wheelbase_m=vehicle_config.vehicle.wheelbase_m,
        maximum_steering_angle_rad=vehicle_config.actuator.max_steering_angle_rad,
        maximum_acceleration_mps2=vehicle_config.actuator.max_acceleration_mps2,
        maximum_deceleration_mps2=vehicle_config.actuator.max_deceleration_mps2,
    )
    policy = jax.jit(
        lambda projection, view: conservative_driveability_action(
            track_batch,
            projection,
            view,
            policy_config,
        )
    )
    race_step = jax.jit(
        lambda state, positions, invalid: step_race_core(
            track_batch,
            state,
            positions,
            invalid,
            race_config,
        )
    )
    view = vehicle.read_state(physics_state)
    projection = project_to_track(
        track_batch,
        view.position_world_m[:, :2],
        race_state.segment_index,
        race_config,
    )

    for _ in range(100):
        actions = policy(projection, view)
        physics_state, applied, diagnostics = vehicle.step(physics_state, actions)
        view = vehicle.read_state(physics_state)
        race = race_step(
            race_state,
            view.position_world_m[:, :2],
            applied.invalid_action,
        )
        race_state = race.state
        projection = race.projection
    jax.block_until_ready((physics_state.data.qpos, race_state.legal_progress_m))

    assert np.all(np.asarray(diagnostics.finite_per_world))
    assert not bool(diagnostics.contact_overflow)
    assert not bool(diagnostics.constraint_overflow)
    assert not bool(diagnostics.unexpected_contact)
    assert np.all(np.asarray(race_state.legal_progress_m) > 5.0)
    assert not np.any(np.asarray(race.off_track))
    assert np.isfinite(np.asarray(actions)).all()
