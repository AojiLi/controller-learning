"""Tests for the host single-world adapter."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
from gymnasium import error
from gymnasium.utils.env_checker import check_env

from controller_learning.config import load_project_config
from controller_learning.control.debug_draw import _DebugDrawBuffer
from controller_learning.envs.car_racing import CarRacingEnv
from controller_learning.envs.episode import PUBLIC_INFO_KEYS
from controller_learning.envs.vector_racing import VecCarRacingEnv
from controller_learning.tracks.generator import generate_track_candidate, pack_track
from controller_learning.tracks.pool import TrackPool
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


@pytest.fixture(scope="module")
def track_pool(project_config):
    generation = generation_spec_from_project(project_config)
    capacity = track_capacity_from_project(project_config)
    tracks = tuple(
        pack_track(generate_track_candidate(seed, generation), capacity) for seed in (101, 202, 303)
    )
    return TrackPool.from_tracks(
        tracks,
        benchmark_version=project_config.benchmark.version,
        split="validation",
    )


def _single(project_config, track) -> CarRacingEnv:
    return CarRacingEnv(
        project_config=project_config,
        level_id=1,
        track=track,
        backend="cpu_reference",
    )


def test_single_environment_passes_gymnasium_checker(project_config, track) -> None:
    env = _single(project_config, track)
    try:
        check_env(env, skip_render_check=True)
    finally:
        env.close()


def test_single_is_host_unbatch_of_batch_one_vector(project_config, track) -> None:
    single = _single(project_config, track)
    vector = VecCarRacingEnv(
        num_envs=1,
        project_config=project_config,
        level_id=1,
        tracks=(track,),
        backend="cpu_reference",
    )
    try:
        single_observation, single_info = single.reset(seed=314)
        vector_observation, vector_info = vector.reset(seed=314)
        for key, value in single_observation.items():
            np.testing.assert_array_equal(value, np.asarray(vector_observation[key])[0])
        assert tuple(single_info) == PUBLIC_INFO_KEYS
        for key, value in single_info.items():
            assert value == np.asarray(vector_info[key])[0]

        action = np.asarray((0.1, 1.0), dtype=np.float32)
        single_step = single.step(action)
        vector_step = vector.step(action[None, :])
        for key, value in single_step[0].items():
            np.testing.assert_allclose(value, np.asarray(vector_step[0][key])[0], atol=1e-6)
        assert single_step[1] == pytest.approx(float(vector_step[1][0]))
        assert single_step[2] is bool(vector_step[2][0])
        assert single_step[3] is bool(vector_step[3][0])
    finally:
        single.close()
        vector.close()


def test_single_requires_explicit_reset_before_and_after_terminal(project_config, track) -> None:
    env = _single(project_config, track)
    try:
        with pytest.raises(error.ResetNeeded):
            env.step(np.zeros(2, dtype=np.float32))
        env.reset(seed=1)
        _, _, terminated, truncated, _ = env.step(np.asarray((np.nan, 0.0), dtype=np.float32))
        assert terminated
        assert not truncated
        with pytest.raises(error.ResetNeeded):
            env.step(np.zeros(2, dtype=np.float32))
        observation, _ = env.reset()
        assert env.observation_space.contains(observation)
    finally:
        env.close()


def test_single_rejects_unsupported_reset_options(project_config, track) -> None:
    env = _single(project_config, track)
    try:
        with pytest.raises(ValueError, match="does not define reset options"):
            env.reset(options={"track_seed": 4})
    finally:
        env.close()


def test_single_pool_reset_translates_one_explicit_track_index(project_config, track_pool) -> None:
    env = CarRacingEnv(
        project_config=project_config,
        level_id=1,
        track_pool=track_pool,
        backend="cpu_reference",
    )
    try:
        first_observation, first_info = env.reset(seed=8, options={"track_index": 0})
        last_observation, last_info = env.reset(
            seed=8,
            options={"track_index": np.int64(track_pool.size - 1)},
        )

        assert first_info["track_id"] == int(track_pool.batch.seed[0])
        assert last_info["track_id"] == int(track_pool.batch.seed[-1])
        np.testing.assert_array_equal(first_info["episode_seed"], last_info["episode_seed"])
        np.testing.assert_array_equal(first_info["controller_seed"], last_info["controller_seed"])
        np.testing.assert_array_equal(
            first_observation["centerline"], track_pool.batch.centerline_m[0]
        )
        np.testing.assert_array_equal(
            last_observation["centerline"], track_pool.batch.centerline_m[-1]
        )
    finally:
        env.close()


def test_reused_pool_episode_matches_fresh_fixed_environment(project_config, track) -> None:
    pool = TrackPool.from_tracks(
        (track,),
        benchmark_version=project_config.benchmark.version,
        split="validation",
    )
    fixed = CarRacingEnv(
        project_config=project_config,
        level_id=1,
        track=track,
        backend="cpu_reference",
    )
    reused = CarRacingEnv(
        project_config=project_config,
        level_id=1,
        track_pool=pool,
        backend="cpu_reference",
    )
    try:
        fixed_output = fixed.reset(seed=37)
        reused_output = reused.reset(seed=37, options={"track_index": 0})
        for key in fixed_output[0]:
            np.testing.assert_array_equal(fixed_output[0][key], reused_output[0][key])
        assert fixed_output[1] == reused_output[1]

        for action in (
            np.asarray((0.0, 1.0), dtype=np.float32),
            np.asarray((0.1, 0.5), dtype=np.float32),
            np.asarray((-0.1, -0.25), dtype=np.float32),
        ):
            fixed_step = fixed.step(action)
            reused_step = reused.step(action)
            for key in fixed_step[0]:
                np.testing.assert_allclose(fixed_step[0][key], reused_step[0][key], atol=1.0e-6)
            assert fixed_step[1] == pytest.approx(reused_step[1])
            assert fixed_step[2] is reused_step[2]
            assert fixed_step[3] is reused_step[3]
            assert fixed_step[4] == reused_step[4]
    finally:
        fixed.close()
        reused.close()


@pytest.mark.parametrize(
    ("options", "error"),
    [
        ({"track_index": True}, TypeError),
        ({"track_index": 1.0}, TypeError),
        ({"track_index": -1}, ValueError),
        ({"track_index": 3}, ValueError),
        ({"track_indices": (0,)}, ValueError),
        ({"track_index": 0, "extra": 1}, ValueError),
    ],
)
def test_single_pool_reset_rejects_invalid_track_index(
    project_config,
    track_pool,
    options,
    error,
) -> None:
    env = CarRacingEnv(
        project_config=project_config,
        level_id=1,
        track_pool=track_pool,
        backend="cpu_reference",
    )
    try:
        with pytest.raises(error):
            env.reset(seed=8, options=options)
    finally:
        env.close()


def test_single_constructor_requires_exactly_one_track_source(
    project_config, track, track_pool
) -> None:
    with pytest.raises(ValueError, match="exactly one"):
        CarRacingEnv(
            project_config=project_config,
            level_id=1,
            backend="cpu_reference",
        )
    with pytest.raises(ValueError, match="exactly one"):
        CarRacingEnv(
            project_config=project_config,
            level_id=1,
            backend="cpu_reference",
            track=track,
            track_pool=track_pool,
        )


def test_single_exposes_the_actual_challenge_configuration(project_config, track) -> None:
    env = _single(project_config, track)
    try:
        assert env.project_config is project_config
        assert env.level_id == 1
        assert env.backend == "cpu_reference"
    finally:
        env.close()


def test_rgb_render_uses_latest_public_observation_and_debug_frame(
    project_config,
    track,
) -> None:
    env = CarRacingEnv(
        project_config=project_config,
        level_id=1,
        track=track,
        backend="cpu_reference",
        render_mode="rgb_array",
    )
    try:
        with pytest.raises(error.ResetNeeded, match="reset"):
            env.render()

        env.reset(seed=5)
        buffer = _DebugDrawBuffer()
        buffer.writer.line((0.0, 0.0), (2.0, 1.0), color=(1.0, 0.0, 0.0))
        env.render_debug_frame(buffer.snapshot())
        image = env.render()

        assert image is not None
        assert image.dtype == np.uint8
        assert image.ndim == 3
        assert image.shape[2] == 3
    finally:
        env.close()


def test_render_contract_rejects_invalid_mode_and_headless_debug_frame(
    project_config,
    track,
) -> None:
    with pytest.raises(ValueError, match="render_mode"):
        CarRacingEnv(
            project_config=project_config,
            level_id=1,
            track=track,
            backend="cpu_reference",
            render_mode="ansi",  # type: ignore[arg-type]
        )

    env = _single(project_config, track)
    try:
        with pytest.raises(error.Error, match="render_mode"):
            env.render_debug_frame(())
        assert env.render() is None
    finally:
        env.close()
