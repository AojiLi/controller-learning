"""Tests for public PPO reward shaping and NEXT_STEP state."""

from __future__ import annotations

import gc
import weakref
from pathlib import Path

import gymnasium as gym
import jax
import jax.numpy as jnp
import numpy as np
import pytest

from controller_learning.config import load_project_config
from controller_learning.envs.episode import PUBLIC_INFO_KEYS
from controller_learning.envs.vector_racing import VecCarRacingEnv
from controller_learning.rl.configuration import PpoObservationConfig, PpoRewardConfig
from controller_learning.rl.features import LocalTrackObservationVecEnv
from controller_learning.rl.reward import (
    PublicRewardShapingVecEnv,
    normalize_physical_actions_jax,
    shape_public_reward_jax,
)
from controller_learning.tracks.generator import generate_track_candidate, pack_track
from controller_learning.tracks.specs import (
    generation_spec_from_project,
    track_capacity_from_project,
)

PROJECT_ROOT = Path(__file__).parents[3]


def _reward_config(*, penalties: bool) -> PpoRewardConfig:
    return PpoRewardConfig(
        progress_scale=100.0,
        success_bonus=1.0,
        offtrack_invalid_penalty=1.0,
        lateral_error_weight=0.05 if penalties else 0.0,
        heading_error_weight=0.02 if penalties else 0.0,
        reverse_speed_weight=0.02 if penalties else 0.0,
        action_change_weight=0.005 if penalties else 0.0,
    )


def _square_observation(num_envs: int) -> dict[str, jax.Array]:
    centerline = np.zeros((num_envs, 6, 2), dtype=np.float32)
    centerline[:, :5] = np.asarray(
        ((0.0, 0.0), (10.0, 0.0), (10.0, 10.0), (0.0, 10.0), (0.0, 0.0)),
        dtype=np.float32,
    )
    left = np.array(centerline, copy=True)
    right = np.array(centerline, copy=True)
    left[:, :5, 1] += 3.0
    right[:, :5, 1] -= 3.0
    mask = np.zeros((num_envs, 6), dtype=np.int8)
    mask[:, :5] = 1
    return {
        "position": jnp.zeros((num_envs, 2), dtype=jnp.float32),
        "yaw": jnp.zeros(num_envs, dtype=jnp.float32),
        "velocity_body": jnp.zeros((num_envs, 2), dtype=jnp.float32),
        "yaw_rate": jnp.zeros(num_envs, dtype=jnp.float32),
        "steering_angle": jnp.zeros(num_envs, dtype=jnp.float32),
        "track_progress": jnp.zeros(num_envs, dtype=jnp.float32),
        "centerline": jnp.asarray(centerline),
        "left_boundary": jnp.asarray(left),
        "right_boundary": jnp.asarray(right),
        "track_mask": jnp.asarray(mask),
        "track_length": jnp.full((num_envs,), 40.0, dtype=jnp.float32),
    }


def test_reward_reconstructs_public_progress_and_terminal_terms() -> None:
    observation = _square_observation(5)
    base_reward = jnp.asarray((0.01, 1.02, -0.97, -1.0, 0.04), dtype=jnp.float32)
    reason = jnp.asarray((0, 1, 2, 3, 4), dtype=jnp.int32)
    zeros = jnp.zeros((5, 2), dtype=jnp.float32)

    result = shape_public_reward_jax(
        observation,
        base_reward,
        reason,
        zeros,
        zeros,
        jnp.zeros(5, dtype=bool),
        config=_reward_config(penalties=False),
        max_steering_angle_rad=0.6,
        max_acceleration_mps2=4.0,
        max_deceleration_mps2=8.0,
    )

    np.testing.assert_allclose(
        np.asarray(result.reward),
        (1.0, 3.0, 2.0, -1.0, 4.0),
        rtol=0.0,
        atol=1.0e-5,
    )
    assert result.reward.dtype == jnp.float32


def test_reward_penalties_are_squared_and_reset_only_is_exact_zero() -> None:
    observation = _square_observation(2)
    observation["position"] = jnp.asarray(((0.0, 2.0), (0.0, 2.0)), dtype=jnp.float32)
    observation["yaw"] = jnp.asarray((0.5, 0.5), dtype=jnp.float32)
    observation["velocity_body"] = jnp.asarray(((-3.0, 0.0), (-3.0, 0.0)))
    action = jnp.asarray(((0.3, 2.0), (0.3, 2.0)), dtype=jnp.float32)

    result = shape_public_reward_jax(
        observation,
        jnp.zeros(2, dtype=jnp.float32),
        jnp.zeros(2, dtype=jnp.int32),
        action,
        jnp.zeros((2, 2), dtype=jnp.float32),
        jnp.asarray((False, True)),
        config=_reward_config(penalties=True),
        max_steering_angle_rad=0.6,
        max_acceleration_mps2=4.0,
        max_deceleration_mps2=8.0,
    )

    assert float(result.reward[0]) == pytest.approx(-0.3875, abs=1.0e-6)
    assert float(result.reward[1]) == 0.0
    np.testing.assert_array_equal(np.asarray(result.normalized_action[0]), (0.5, 0.5))
    np.testing.assert_array_equal(np.asarray(result.normalized_action[1]), (0.0, 0.0))


def test_action_normalization_is_asymmetric_and_nonfinite_safe() -> None:
    normalized = normalize_physical_actions_jax(
        jnp.asarray(
            ((-0.6, -8.0), (0.6, 4.0), (jnp.nan, jnp.inf)),
            dtype=jnp.float32,
        ),
        max_steering_angle_rad=0.6,
        max_acceleration_mps2=4.0,
        max_deceleration_mps2=8.0,
    )

    np.testing.assert_allclose(
        np.asarray(normalized),
        ((-1.0, -1.0), (1.0, 1.0), (0.0, 0.0)),
        rtol=0.0,
        atol=1.0e-7,
    )


def test_reward_wrapper_masks_actual_next_step_autoreset_slot() -> None:
    project = load_project_config(PROJECT_ROOT)
    track = pack_track(
        generate_track_candidate(123, generation_spec_from_project(project)),
        track_capacity_from_project(project),
    )
    base = VecCarRacingEnv(
        num_envs=1,
        project_config=project,
        level_id=1,
        tracks=(track,),
        backend="cpu_reference",
    )
    env = PublicRewardShapingVecEnv(base, _reward_config(penalties=True))
    try:
        _, reset_info = env.reset(seed=91)
        assert tuple(reset_info) == PUBLIC_INFO_KEYS
        env.step(np.zeros((1, 2), dtype=np.float32))
        terminal = env.step(np.asarray(((np.nan, 0.0),), dtype=np.float32))
        assert bool(np.asarray(terminal[2])[0])
        assert int(np.asarray(terminal[4]["termination_reason"])[0]) == 3
        assert np.isfinite(np.asarray(terminal[1])).all()

        autoreset = env.step(np.zeros((1, 2), dtype=np.float32))
        assert float(np.asarray(autoreset[1])[0]) == 0.0
        assert not bool(np.asarray(autoreset[2])[0])
        assert not bool(np.asarray(autoreset[3])[0])
        np.testing.assert_array_equal(
            np.asarray(env._previous_normalized_action),
            ((0.0, 0.0),),
        )
        assert tuple(autoreset[4]) == PUBLIC_INFO_KEYS
    finally:
        env.close()


def test_reward_wrapper_must_be_the_first_layer_over_the_challenge() -> None:
    project = load_project_config(PROJECT_ROOT)
    track = pack_track(
        generate_track_candidate(124, generation_spec_from_project(project)),
        track_capacity_from_project(project),
    )
    base = VecCarRacingEnv(
        num_envs=1,
        project_config=project,
        level_id=1,
        tracks=(track,),
        backend="cpu_reference",
    )
    try:
        with pytest.raises(TypeError, match="directly wrap"):
            PublicRewardShapingVecEnv(
                gym.vector.VectorWrapper(base), _reward_config(penalties=True)
            )
    finally:
        base.close()


def test_public_wrapper_stack_releases_jit_and_environment_references() -> None:
    project = load_project_config(PROJECT_ROOT)
    track = pack_track(
        generate_track_candidate(125, generation_spec_from_project(project)),
        track_capacity_from_project(project),
    )
    base = VecCarRacingEnv(
        num_envs=1,
        project_config=project,
        level_id=1,
        tracks=(track,),
        backend="cpu_reference",
    )
    shaped = PublicRewardShapingVecEnv(base, _reward_config(penalties=True))
    featured = LocalTrackObservationVecEnv(
        shaped,
        config=PpoObservationConfig(
            preview_points=16,
            preview_distance_m=40.0,
            max_speed_mps=15.0,
        ),
    )
    featured.reset(seed=93)
    featured.step(jnp.zeros((1, 2), dtype=jnp.float32))
    references = tuple(weakref.ref(value) for value in (featured, shaped, base))

    featured.close()
    assert featured._encode_observation is None
    assert shaped._shape_reward is None
    del featured, shaped, base
    gc.collect()

    assert all(reference() is None for reference in references)
