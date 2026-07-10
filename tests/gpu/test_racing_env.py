"""GPU integration tests for the official native vector Challenge."""

from __future__ import annotations

from pathlib import Path

import jax
import numpy as np
import pytest

from controller_learning.config import load_project_config
from controller_learning.envs.car_racing import CarRacingEnv
from controller_learning.envs.episode import (
    initialize_episode_identities,
    masked_next_episode,
    track_pool_seeds,
)
from controller_learning.envs.race_core import RaceTermination
from controller_learning.envs.vector_racing import VecCarRacingEnv
from controller_learning.tracks.assets import load_manifest_track_batch
from controller_learning.tracks.generator import generate_track_candidate, pack_track
from controller_learning.tracks.official_assets import (
    official_track_asset_directory,
    official_track_split_spec,
    validate_official_manifest,
)
from controller_learning.tracks.pool import TrackPool
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


def test_pool_autoreset_selects_and_replaces_tracks_without_transfers() -> None:
    project = load_project_config(PROJECT_ROOT)
    tracks = _tracks(project, 8)
    pool = TrackPool.from_tracks(
        tracks,
        benchmark_version=project.benchmark.version,
        split="train",
    )
    env = VecCarRacingEnv(
        num_envs=4,
        project_config=project,
        level_id=1,
        track_pool=pool,
        backend="mjx_warp",
    )
    try:
        identity = initialize_episode_identities(77, 4)
        initial_indices = np.asarray(track_pool_seeds(identity) % pool.size, dtype=np.int32)
        initial_observation, initial_info = env.reset(seed=77)
        jax.block_until_ready((initial_observation, initial_info["track_id"]))
        np.testing.assert_array_equal(initial_info["track_id"], pool.batch.seed[initial_indices])
        initial_tracks = jax.tree.map(lambda value: np.array(value, copy=True), env._track_batch)

        invalid = jax.numpy.zeros((4, 2), dtype=jax.numpy.float32).at[1, 0].set(jax.numpy.nan)
        terminal = env.step(invalid)
        jax.block_until_ready((terminal[:4], terminal[4]["track_id"]))
        assert bool(terminal[2][1])
        np.testing.assert_array_equal(terminal[4]["track_id"], initial_info["track_id"])

        action = jax.numpy.zeros((4, 2), dtype=jax.numpy.float32)
        with jax.transfer_guard("disallow"):
            autoreset = env.step(action)
            jax.block_until_ready((autoreset[:4], autoreset[4]["track_id"]))

        next_identity = masked_next_episode(
            identity,
            np.asarray((False, True, False, False), dtype=np.bool_),
        )
        next_indices = np.asarray(track_pool_seeds(next_identity) % pool.size, dtype=np.int32)
        expected_ids = np.array(pool.batch.seed[initial_indices], copy=True)
        expected_ids[1] = pool.batch.seed[next_indices[1]]
        np.testing.assert_array_equal(autoreset[4]["track_id"], expected_ids)
        np.testing.assert_array_equal(
            autoreset[0]["centerline"][1],
            pool.batch.centerline_m[next_indices[1]],
        )
        for before, after in zip(
            jax.tree.leaves(initial_tracks),
            jax.tree.leaves(env._track_batch),
            strict=True,
        ):
            np.testing.assert_array_equal(np.asarray(after)[[0, 2, 3]], before[[0, 2, 3]])
        np.testing.assert_array_equal(autoreset[1][1], 0.0)
        assert not bool(autoreset[2][1])
        assert not bool(autoreset[3][1])
    finally:
        env.close()


def test_pool_explicit_rows_reuse_one_gpu_backend_with_fresh_identity_reset() -> None:
    project = load_project_config(PROJECT_ROOT)
    validation_spec = official_track_split_spec("validation")
    manifest, batch = load_manifest_track_batch(
        official_track_asset_directory() / validation_spec.manifest_file
    )
    validate_official_manifest(project, manifest)
    pool = TrackPool(
        benchmark_version=project.benchmark.version,
        generator_version=manifest.generator_version,
        split="validation",
        batch=batch,
    )
    env = CarRacingEnv(
        project_config=project,
        level_id=1,
        track_pool=pool,
        backend="mjx_warp",
    )
    try:
        identities = []
        first_cache_sizes = None
        action = np.zeros(2, dtype=np.float32)
        for index in range(pool.size):
            observation, info = env.reset(
                seed=19,
                options={"track_index": index},
            )
            step = env.step(action)

            assert info["track_id"] == int(pool.batch.seed[index])
            np.testing.assert_array_equal(observation["centerline"], pool.batch.centerline_m[index])
            np.testing.assert_array_equal(observation["track_mask"], pool.batch.track_mask[index])
            np.testing.assert_allclose(
                observation["position"],
                pool.batch.start_pose[index, :2],
                rtol=0.0,
                atol=1.0e-6,
            )
            assert np.isfinite(np.asarray(step[1])).all()
            assert not step[2]
            assert not step[3]
            assert step[4]["track_id"] == int(pool.batch.seed[index])
            identities.append(
                (
                    info["episode_seed"],
                    info["controller_seed"],
                )
            )
            if index == 0:
                vector = env._vector_env
                jit_functions = {
                    "gather": vector._gather_pool_tracks,
                    "reset": vector._reset_race,
                    "race_step": vector._step_race,
                    "encode": vector._encode_observation,
                    "normalize": vector._normalize_actions,
                    "read": vector._read_vehicle_state,
                    "finalize": vector._finalize_gpu_step,
                    "vehicle_step": vector._vehicle_driver._vehicle._step_function,
                }
                first_cache_sizes = {
                    name: function._cache_size() for name, function in jit_functions.items()
                }
                assert all(size >= 1 for size in first_cache_sizes.values())
        assert len(set(identities)) == 1
        assert first_cache_sizes is not None
        assert {
            name: function._cache_size() for name, function in jit_functions.items()
        } == first_cache_sizes
    finally:
        env.close()
