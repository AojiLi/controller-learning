"""GPU parity tests for lazy PPO-to-NumPy actor conversion."""

from __future__ import annotations

import importlib
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from controller_learning.config import load_project_config
from controller_learning.envs.observation import action_space

PROJECT_ROOT = Path(__file__).parents[2]
OBSERVATION_DIM = 100
pytestmark = pytest.mark.gpu


def _torch() -> Any:
    return importlib.import_module("torch")


def _policy_module() -> Any:
    return importlib.import_module("controller_learning.rl.policy")


def _numpy_actor_module() -> Any:
    return importlib.import_module("controller_learning.rl.numpy_actor")


def _device() -> Any:
    torch = _torch()
    return torch.device("cuda", torch.cuda.current_device())


def _policy(seed: int = 401) -> Any:
    physical = action_space(load_project_config(PROJECT_ROOT))
    return _policy_module().PpoActorCritic(
        OBSERVATION_DIM,
        action_low=physical.low,
        action_high=physical.high,
        policy_seed=seed,
        initial_log_std=-0.5,
        device=_device(),
    )


def test_converted_numpy_actor_matches_cuda_deterministic_policy_and_round_trip(
    tmp_path: Path,
) -> None:
    torch = _torch()
    numpy_actor = _numpy_actor_module()
    policy = _policy()
    actor = numpy_actor.numpy_actor_from_ppo_state_dict(policy.state_dict())
    observations = (
        np.random.default_rng(409)
        .normal(
            0.0,
            1.0,
            size=(257, OBSERVATION_DIM),
        )
        .astype(np.float32)
    )

    with torch.no_grad():
        expected = policy.deterministic(torch.as_tensor(observations, device=_device()))
    actual = actor.deterministic(observations)

    np.testing.assert_allclose(
        actual.pre_tanh,
        expected.pre_tanh.cpu().numpy(),
        rtol=1.0e-5,
        atol=2.0e-6,
    )
    np.testing.assert_allclose(
        actual.action,
        expected.action.cpu().numpy(),
        rtol=1.0e-5,
        atol=2.0e-6,
    )
    artifact_path = tmp_path / "actor.npz"
    evidence = numpy_actor.save_numpy_actor_npz(actor, artifact_path)
    loaded = numpy_actor.load_numpy_actor_npz(
        artifact_path,
        expected_sha256=evidence.sha256,
        expected_size_bytes=evidence.size_bytes,
    )
    np.testing.assert_array_equal(loaded.actor(observations), actual.action)


def test_converter_rejects_missing_redundant_or_wrong_dtype_policy_state() -> None:
    torch = _torch()
    numpy_actor = _numpy_actor_module()
    policy = _policy(seed=419)
    state = policy.state_dict()

    missing = dict(state)
    del missing["actor_mean.bias"]
    with pytest.raises(ValueError, match="keys differ"):
        numpy_actor.numpy_actor_from_ppo_state_dict(missing)

    wrong_scale = dict(state)
    wrong_scale["action_scale"] = state["action_scale"].clone()
    wrong_scale["action_scale"][0] += 0.25
    with pytest.raises(ValueError, match="action_scale"):
        numpy_actor.numpy_actor_from_ppo_state_dict(wrong_scale)

    wrong_dtype = dict(state)
    wrong_dtype["actor_mean.weight"] = state["actor_mean.weight"].to(dtype=torch.float64)
    with pytest.raises(TypeError, match=r"torch\.float32"):
        numpy_actor.numpy_actor_from_ppo_state_dict(wrong_dtype)
