"""Contract tests for the official vector Challenge environment."""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import jax
import numpy as np
import pytest
from gymnasium import error
from gymnasium.vector import AutoresetMode

from controller_learning.config import load_project_config
from controller_learning.envs.episode import PUBLIC_INFO_KEYS
from controller_learning.envs.race_core import RaceTermination
from controller_learning.envs.vector_racing import VecCarRacingEnv
from controller_learning.tracks.generator import generate_track_candidate, pack_track
from controller_learning.tracks.specs import (
    generation_spec_from_project,
    track_capacity_from_project,
)

PROJECT_ROOT = Path(__file__).parents[3]


@pytest.fixture(scope="module")
def project_config():
    return load_project_config(PROJECT_ROOT)


@pytest.fixture(scope="module")
def track(project_config):
    return pack_track(
        generate_track_candidate(42, generation_spec_from_project(project_config)),
        track_capacity_from_project(project_config),
    )


def _environment(project_config, track) -> VecCarRacingEnv:
    return VecCarRacingEnv(
        num_envs=1,
        project_config=project_config,
        level_id=1,
        tracks=(track,),
        backend="cpu_reference",
    )


def test_constructor_requires_explicit_compatible_level_and_tracks(project_config, track) -> None:
    with pytest.raises(ValueError, match="one Track per world"):
        VecCarRacingEnv(
            num_envs=1,
            project_config=project_config,
            level_id=1,
            tracks=(),
            backend="cpu_reference",
        )
    with pytest.raises(ValueError, match="unknown level_id"):
        VecCarRacingEnv(
            num_envs=1,
            project_config=project_config,
            level_id=99,
            tracks=(track,),
            backend="cpu_reference",
        )
    with pytest.raises(ValueError, match="generator version"):
        VecCarRacingEnv(
            num_envs=1,
            project_config=project_config,
            level_id=1,
            tracks=(replace(track, generator_version="other"),),
            backend="cpu_reference",
        )
    with pytest.raises(ValueError, match="width"):
        VecCarRacingEnv(
            num_envs=1,
            project_config=project_config,
            level_id=1,
            tracks=(replace(track, width_m=track.width_m + 1.0),),
            backend="cpu_reference",
        )
    with pytest.raises(ValueError, match="render_mode=None"):
        VecCarRacingEnv(
            num_envs=1,
            project_config=project_config,
            level_id=1,
            tracks=(track,),
            backend="cpu_reference",
            render_mode="human",
        )


def test_reset_returns_jax_batch_and_restricted_reproducible_info(project_config, track) -> None:
    env = _environment(project_config, track)
    try:
        first_observation, first_info = env.reset(seed=123)
        repeated_observation, repeated_info = env.reset(seed=123)

        assert env.metadata["autoreset_mode"] is AutoresetMode.NEXT_STEP
        assert env.observation_space.contains(
            {key: np.asarray(value) for key, value in first_observation.items()}
        )
        assert all(isinstance(value, jax.Array) for value in first_observation.values())
        assert all(
            isinstance(first_info[key], jax.Array)
            for key in (
                "episode_seed",
                "controller_seed",
                "termination_reason",
                "lap_completed",
                "lap_time_s",
            )
        )
        assert isinstance(env._pending_reset, jax.Array)
        assert tuple(first_info) == PUBLIC_INFO_KEYS
        assert {"physics", "projection", "saturation", "diagnostics"}.isdisjoint(first_info)
        for key in first_observation:
            np.testing.assert_array_equal(first_observation[key], repeated_observation[key])
        for key in first_info:
            np.testing.assert_array_equal(first_info[key], repeated_info[key])
        assert first_info["track_id"].tolist() == [f"{track.generator_version}:{track.seed}"]
    finally:
        env.close()


def test_step_clips_finite_actions_and_invalidates_bad_rows(project_config, track) -> None:
    env = _environment(project_config, track)
    try:
        env.reset(seed=7)
        _, _, terminated, truncated, info = env.step(np.asarray(((99.0, 99.0),), dtype=np.float32))
        assert not bool(terminated[0])
        assert not bool(truncated[0])
        assert env._last_applied_action is not None
        assert float(env._last_applied_action.steering_angle_rad[0]) == pytest.approx(
            project_config.vehicle.actuator.max_steering_angle_rad
        )
        assert float(env._last_applied_action.longitudinal_acceleration_mps2[0]) == pytest.approx(
            project_config.vehicle.actuator.max_acceleration_mps2
        )
        assert int(env._last_applied_action.saturation_count[0]) == 2
        assert int(info["termination_reason"][0]) == RaceTermination.NONE

        env.reset(seed=7)
        _, reward, terminated, truncated, info = env.step(
            np.asarray(((np.nan, 1.0),), dtype=np.float32)
        )
        assert bool(terminated[0])
        assert not bool(truncated[0])
        assert float(reward[0]) == pytest.approx(-1.0, abs=1e-4)
        assert int(info["termination_reason"][0]) == RaceTermination.INVALID_ACTION
        assert env._last_applied_action is not None
        assert float(env._last_applied_action.steering_angle_rad[0]) == 0.0
        assert float(env._last_applied_action.longitudinal_acceleration_mps2[0]) == 0.0

        env.reset(seed=7)
        _, _, terminated, _, info = env.step(np.zeros(2, dtype=np.float32))
        assert bool(terminated[0])
        assert int(info["termination_reason"][0]) == RaceTermination.INVALID_ACTION

        env.reset(seed=7)
        _, _, terminated, _, info = env.step((("bad", "action"),))
        assert bool(terminated[0])
        assert int(info["termination_reason"][0]) == RaceTermination.INVALID_ACTION

        env.reset(seed=7)
        _, _, terminated, _, info = env.step(
            jax.numpy.asarray(((np.nan, 0.0),), dtype=jax.numpy.float32)
        )
        assert bool(terminated[0])
        assert int(info["termination_reason"][0]) == RaceTermination.INVALID_ACTION
    finally:
        env.close()


def test_next_step_autoreset_ignores_action_and_advances_only_episode(
    project_config, track
) -> None:
    env = _environment(project_config, track)
    try:
        initial_observation, initial_info = env.reset(seed=991)
        _, _, terminated, _, terminal_info = env.step(
            np.asarray(((np.nan, 0.0),), dtype=np.float32)
        )
        assert bool(terminated[0])
        assert int(terminal_info["termination_reason"][0]) == RaceTermination.INVALID_ACTION

        observation, reward, terminated, truncated, reset_info = env.step(
            np.asarray(((np.nan, np.inf),), dtype=np.float32)
        )
        for key in initial_observation:
            np.testing.assert_allclose(observation[key], initial_observation[key], atol=1e-6)
        np.testing.assert_array_equal(reward, (0.0,))
        np.testing.assert_array_equal(terminated, (False,))
        np.testing.assert_array_equal(truncated, (False,))
        assert env._identity is not None
        np.testing.assert_array_equal(env._identity.episode_counter, (1,))
        assert reset_info["episode_seed"][0] != initial_info["episode_seed"][0]
        assert int(reset_info["termination_reason"][0]) == RaceTermination.NONE
        assert reset_info["track_id"][0] == initial_info["track_id"][0]
        assert env._race_state is not None
        assert int(env._race_state.elapsed_steps[0]) == 0
    finally:
        env.close()


def test_step_requires_reset_and_close_is_idempotent(project_config, track) -> None:
    env = _environment(project_config, track)
    with pytest.raises(error.ResetNeeded, match="call reset"):
        env.step(np.zeros((1, 2), dtype=np.float32))
    env.close()
    env.close()
