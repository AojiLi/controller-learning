"""Gymnasium registration tests for public single and vector entry points."""

from __future__ import annotations

from pathlib import Path

import gymnasium as gym
import numpy as np
import pytest

from controller_learning.config import load_project_config
from controller_learning.envs.car_racing import CarRacingEnv
from controller_learning.envs.registration import ENV_ID, register_environments
from controller_learning.envs.vector_racing import VecCarRacingEnv
from controller_learning.tracks.generator import generate_track_candidate, pack_track
from controller_learning.tracks.specs import (
    generation_spec_from_project,
    track_capacity_from_project,
)

PROJECT_ROOT = Path(__file__).parents[3]


@pytest.fixture(scope="module")
def configured_track():
    project = load_project_config(PROJECT_ROOT)
    track = pack_track(
        generate_track_candidate(42, generation_spec_from_project(project)),
        track_capacity_from_project(project),
    )
    return project, track


def test_registration_has_native_single_and_vector_entries() -> None:
    register_environments()
    register_environments()
    spec = gym.spec(ENV_ID)

    assert spec.entry_point == "controller_learning.envs.car_racing:CarRacingEnv"
    assert spec.vector_entry_point == "controller_learning.envs.vector_racing:VecCarRacingEnv"
    assert spec.max_episode_steps is None
    assert spec.disable_env_checker is False


def test_registration_rejects_a_conflicting_existing_id() -> None:
    existing = gym.registry[ENV_ID]
    gym.registry[ENV_ID] = gym.envs.registration.EnvSpec(
        id=ENV_ID,
        entry_point="somewhere_else:Environment",
    )
    try:
        with pytest.raises(RuntimeError, match="already registered"):
            register_environments()
    finally:
        gym.registry[ENV_ID] = existing


def test_make_requires_and_forwards_explicit_m4_arguments(configured_track) -> None:
    project, track = configured_track
    env = gym.make(
        ENV_ID,
        project_config=project,
        level_id=1,
        track=track,
        backend="cpu_reference",
    )
    try:
        assert isinstance(env.unwrapped, CarRacingEnv)
        observation, _ = env.reset(seed=8)
        assert env.observation_space.contains(observation)
    finally:
        env.close()

    vector = gym.make_vec(
        ENV_ID,
        num_envs=1,
        project_config=project,
        level_id=1,
        tracks=(track,),
        backend="cpu_reference",
    )
    try:
        assert isinstance(vector.unwrapped, VecCarRacingEnv)
        observation, _ = vector.reset(seed=8)
        assert vector.observation_space.contains(
            {key: np.asarray(value) for key, value in observation.items()}
        )
    finally:
        vector.close()

    with pytest.raises(TypeError):
        gym.make(ENV_ID)
