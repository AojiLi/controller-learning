"""GPU integration tests for the official native vector Challenge."""

from __future__ import annotations

from pathlib import Path

import jax
import numpy as np
import pytest

from controller_learning.config import load_project_config
from controller_learning.envs.race_core import RaceTermination
from controller_learning.envs.vector_racing import VecCarRacingEnv
from controller_learning.tracks.generator import generate_track_candidate, pack_track
from controller_learning.tracks.specs import (
    generation_spec_from_project,
    track_capacity_from_project,
)

PROJECT_ROOT = Path(__file__).parents[2]
pytestmark = pytest.mark.gpu


def _tracks(project_config, count: int):
    generation = generation_spec_from_project(project_config)
    capacity = track_capacity_from_project(project_config)
    return tuple(
        pack_track(generate_track_candidate(seed, generation), capacity) for seed in range(count)
    )


@pytest.mark.parametrize("num_envs", (1, 4))
def test_mjx_warp_environment_smoke(num_envs: int) -> None:
    project = load_project_config(PROJECT_ROOT)
    tracks = _tracks(project, num_envs)
    env = VecCarRacingEnv(
        num_envs=num_envs,
        project_config=project,
        level_id=1,
        tracks=tracks,
        backend="mjx_warp",
    )
    try:
        observation, info = env.reset(seed=123)
        assert all(isinstance(value, jax.Array) for value in observation.values())
        assert len(set(info["track_id"].tolist())) == num_envs
        for _ in range(3):
            observation, reward, terminated, truncated, info = env.step(
                jax.numpy.zeros((num_envs, 2), dtype=jax.numpy.float32)
            )
        jax.block_until_ready((observation, reward, terminated, truncated))
        assert np.isfinite(np.asarray(reward)).all()
        assert not np.any(np.asarray(terminated))
        assert not np.any(np.asarray(truncated))
        assert np.all(np.asarray(info["termination_reason"]) == RaceTermination.NONE)
    finally:
        env.close()


def test_mixed_next_step_reset_isolates_one_gpu_world() -> None:
    project = load_project_config(PROJECT_ROOT)
    tracks = _tracks(project, 4)
    env = VecCarRacingEnv(
        num_envs=4,
        project_config=project,
        level_id=1,
        tracks=tracks,
        backend="mjx_warp",
    )
    try:
        initial_observation, _ = env.reset(seed=77)
        invalid = np.zeros((4, 2), dtype=np.float32)
        invalid[1, 0] = np.nan
        _, _, terminated, _, terminal_info = env.step(invalid)
        np.testing.assert_array_equal(terminated, (False, True, False, False))
        assert int(terminal_info["termination_reason"][1]) == RaceTermination.INVALID_ACTION

        prior_episode_seed = np.array(terminal_info["episode_seed"], copy=True)
        observation, reward, terminated, truncated, info = env.step(
            np.full((4, 2), (0.0, 0.5), dtype=np.float32)
        )
        jax.block_until_ready((observation, reward, terminated, truncated))

        np.testing.assert_array_equal(reward[1], 0.0)
        assert not bool(terminated[1])
        assert not bool(truncated[1])
        for key in initial_observation:
            np.testing.assert_allclose(observation[key][1], initial_observation[key][1], atol=1e-6)
        next_episode_seed = np.asarray(info["episode_seed"])
        assert next_episode_seed[1] != prior_episode_seed[1]
        np.testing.assert_array_equal(next_episode_seed[[0, 2, 3]], prior_episode_seed[[0, 2, 3]])
        assert env._race_state is not None
        np.testing.assert_array_equal(env._race_state.elapsed_steps, (2, 0, 2, 2))
    finally:
        env.close()


def test_warm_gpu_steps_disallow_all_host_device_transfers() -> None:
    project = load_project_config(PROJECT_ROOT)
    tracks = _tracks(project, 4)
    env = VecCarRacingEnv(
        num_envs=4,
        project_config=project,
        level_id=1,
        tracks=tracks,
        backend="mjx_warp",
    )
    try:
        action = jax.numpy.zeros((4, 2), dtype=jax.numpy.float32)
        env.reset(seed=104)
        warm = env.step(action)
        jax.block_until_ready(warm[:4])

        with jax.transfer_guard("disallow"):
            active = env.step(action)
            jax.block_until_ready(active[:4])

        invalid = action.at[1, 0].set(jax.numpy.nan)
        terminal = env.step(invalid)
        jax.block_until_ready(terminal[:4])
        with jax.transfer_guard("disallow"):
            autoreset = env.step(action)
            jax.block_until_ready(autoreset[:4])

        np.testing.assert_array_equal(autoreset[1][1], 0.0)
        assert not bool(autoreset[2][1])
        assert not bool(autoreset[3][1])
        assert all(
            isinstance(autoreset[4][key], jax.Array)
            for key in (
                "episode_seed",
                "controller_seed",
                "termination_reason",
                "lap_completed",
                "lap_time_s",
            )
        )
    finally:
        env.close()
