"""GPU integration tests for the same-device JAX-to-Torch vector bridge."""

from __future__ import annotations

import importlib
from pathlib import Path
from typing import Any

import gymnasium as gym
import jax
import numpy as np
import pytest

from controller_learning.config import load_project_config
from controller_learning.envs.episode import PUBLIC_INFO_KEYS
from controller_learning.envs.vector_racing import VecCarRacingEnv
from controller_learning.rl.configuration import load_ppo_config
from controller_learning.rl.features import LOCAL_TRACK_FEATURE_DIM, LocalTrackObservationVecEnv
from controller_learning.rl.reward import PublicRewardShapingVecEnv
from controller_learning.rl.torch_bridge import (
    JaxToTorchVecEnv,
    _jax_array_to_torch,
    _torch_array_to_jax,
)
from controller_learning.tracks.generator import generate_track_candidate, pack_track
from controller_learning.tracks.specs import (
    generation_spec_from_project,
    track_capacity_from_project,
)

PROJECT_ROOT = Path(__file__).parents[2]
pytestmark = pytest.mark.gpu


def _torch() -> Any:
    return importlib.import_module("torch")


def _tracks(project_config, count: int):
    generation = generation_spec_from_project(project_config)
    capacity = track_capacity_from_project(project_config)
    return tuple(
        pack_track(generate_track_candidate(seed, generation), capacity)
        for seed in range(100, 100 + count)
    )


def _base_env(num_envs: int) -> VecCarRacingEnv:
    project = load_project_config(PROJECT_ROOT)
    return VecCarRacingEnv(
        num_envs=num_envs,
        project_config=project,
        level_id=1,
        tracks=_tracks(project, num_envs),
        backend="mjx_warp",
    )


def _repeated_track_env(num_envs: int) -> VecCarRacingEnv:
    project = load_project_config(PROJECT_ROOT)
    track = _tracks(project, 1)[0]
    return VecCarRacingEnv(
        num_envs=num_envs,
        project_config=project,
        level_id=1,
        tracks=(track,) * num_envs,
        backend="mjx_warp",
    )


def _assert_cuda_tensor_tree(value: Any, torch: Any) -> None:
    if isinstance(value, torch.Tensor):
        assert value.device.type == "cuda"
        return
    if isinstance(value, dict):
        for leaf in value.values():
            _assert_cuda_tensor_tree(leaf, torch)
        return
    if isinstance(value, (tuple, list)):
        for leaf in value:
            _assert_cuda_tensor_tree(leaf, torch)
        return
    raise AssertionError(f"unexpected converted leaf {type(value).__name__}")


def test_one_world_bridge_preserves_devices_schema_strings_and_dlpack_buffers() -> None:
    torch = _torch()
    base = _base_env(1)
    env = JaxToTorchVecEnv(base, device="cuda")
    try:
        observation, info = env.reset(seed=71)
        _assert_cuda_tensor_tree(observation, torch)
        assert observation["position"].shape == (1, 2)
        assert observation["centerline"].shape == (1, 640, 2)
        assert observation["track_mask"].shape == (1, 640)

        assert tuple(info) == PUBLIC_INFO_KEYS
        benchmark_versions = info["benchmark_version"]
        assert benchmark_versions is base._benchmark_versions
        assert benchmark_versions.shape == (1,)
        assert benchmark_versions.dtype.kind == "U"
        assert not benchmark_versions.flags.writeable
        for name in PUBLIC_INFO_KEYS:
            if name != "benchmark_version":
                assert isinstance(info[name], torch.Tensor)
                assert info[name].shape == (1,)
                assert info[name].device.type == "cuda"

        action = torch.zeros((1, 2), dtype=torch.float32, device="cuda")
        next_observation, reward, terminated, truncated, step_info = env.step(action)
        _assert_cuda_tensor_tree(next_observation, torch)
        assert reward.shape == terminated.shape == truncated.shape == (1,)
        assert reward.dtype == torch.float32
        assert terminated.dtype == torch.bool
        assert truncated.dtype == torch.bool
        assert reward.device.type == "cuda"
        assert step_info["benchmark_version"] is benchmark_versions

        jax_source = jax.numpy.arange(8, dtype=jax.numpy.float32)
        torch_view = _jax_array_to_torch(jax_source, torch=torch, device=env.device)
        assert jax_source.unsafe_buffer_pointer() == torch_view.data_ptr()

        torch_source = torch.arange(8, dtype=torch.float32, device="cuda")
        jax_view = _torch_array_to_jax(torch_source, torch=torch, device=env.device)
        assert torch_source.data_ptr() == jax_view.unsafe_buffer_pointer()

        with pytest.raises(ValueError, match="actions are on cpu"):
            env.step(torch.zeros((1, 2), dtype=torch.float32, device="cpu"))
    finally:
        env.close()


def test_mixed_next_step_reset_stays_on_device_through_torch_bridge() -> None:
    torch = _torch()
    base = _base_env(2)
    env = JaxToTorchVecEnv(base, device="cuda")
    try:
        initial_observation, initial_info = env.reset(seed=79)
        initial_episode_seed = initial_info["episode_seed"].clone()

        invalid = torch.zeros((2, 2), dtype=torch.float32, device="cuda")
        invalid[1, 0] = torch.nan
        zero_action = torch.zeros((2, 2), dtype=torch.float32, device="cuda")
        env.step(zero_action)
        torch.cuda.synchronize()
        with jax.transfer_guard("disallow"):
            terminal = env.step(invalid)
            autoreset = env.step(zero_action)
        torch.cuda.synchronize()

        assert torch.equal(
            terminal[2],
            torch.tensor((False, True), dtype=torch.bool, device="cuda"),
        )
        assert terminal[4]["termination_reason"][1] == 3
        assert not torch.any(autoreset[2])
        assert not torch.any(autoreset[3])
        assert autoreset[1][1] == 0.0
        assert autoreset[4]["episode_seed"][0] == initial_episode_seed[0]
        assert autoreset[4]["episode_seed"][1] != initial_episode_seed[1]
        assert torch.allclose(
            autoreset[0]["position"][1],
            initial_observation["position"][1],
            rtol=0.0,
            atol=1.0e-6,
        )
        assert not autoreset[4]["benchmark_version"].flags.writeable
    finally:
        env.close()


class _UnexpectedInfo(gym.vector.VectorWrapper):
    def reset(self, **kwargs):
        observation, info = self.env.reset(**kwargs)
        return observation, {**info, "private_backend_state": jax.numpy.zeros(self.num_envs)}


class _WrongBenchmarkInfo(gym.vector.VectorWrapper):
    def reset(self, **kwargs):
        observation, info = self.env.reset(**kwargs)
        versions = np.asarray(["wrong-version"] * self.num_envs, dtype=np.str_)
        versions.setflags(write=False)
        return observation, {**info, "benchmark_version": versions}


def test_bridge_rejects_non_public_info_fields() -> None:
    base = _base_env(1)
    env = JaxToTorchVecEnv(_UnexpectedInfo(base), device="cuda")
    try:
        with pytest.raises(ValueError, match="extra=\\['private_backend_state'\\]"):
            env.reset(seed=83)
    finally:
        env.close()


def test_bridge_requires_an_explicit_cuda_device() -> None:
    base = _base_env(1)
    try:
        with pytest.raises(TypeError, match="device"):
            JaxToTorchVecEnv(base)  # type: ignore[call-arg]
        with pytest.raises(ValueError, match="explicit CUDA device"):
            JaxToTorchVecEnv(base, device="cpu")
        with pytest.raises(ValueError, match="explicit CUDA device"):
            JaxToTorchVecEnv(base, device=None)
    finally:
        base.close()


def test_bridge_rejects_wrong_public_benchmark_version_contents() -> None:
    base = _base_env(1)
    env = JaxToTorchVecEnv(_WrongBenchmarkInfo(base), device="cuda")
    try:
        with pytest.raises(ValueError, match="does not match the official environment"):
            env.reset(seed=89)
    finally:
        env.close()


def test_formal_1024_world_public_wrapper_stack_bridges_only_compact_features() -> None:
    torch = _torch()
    config = load_ppo_config(PROJECT_ROOT / "configs" / "ppo.toml")
    base = _repeated_track_env(config.environment.num_envs)
    shaped = PublicRewardShapingVecEnv(base, config.reward)
    featured = LocalTrackObservationVecEnv(shaped, config=config.observation)
    env = JaxToTorchVecEnv(featured, device="cuda")
    try:
        observation, info = env.reset(seed=config.environment.environment_seed)
        assert isinstance(observation, torch.Tensor)
        assert observation.shape == (
            config.environment.num_envs,
            LOCAL_TRACK_FEATURE_DIM,
        )
        assert observation.dtype == torch.float32
        assert observation.device == env.device
        assert torch.all(torch.isfinite(observation))
        assert tuple(info) == PUBLIC_INFO_KEYS

        action = torch.zeros(
            (config.environment.num_envs, 2),
            dtype=torch.float32,
            device=env.device,
        )
        # The first call compiles MJX-Warp and the two public wrappers. Transfer guarding applies
        # to the steady-state path after all shape-specialized executables exist.
        env.step(action)
        torch.cuda.synchronize(env.device)
        with jax.transfer_guard("disallow"):
            next_observation, reward, terminated, truncated, step_info = env.step(action)
        torch.cuda.synchronize(env.device)

        assert next_observation.shape == observation.shape
        assert reward.shape == terminated.shape == truncated.shape == (config.environment.num_envs,)
        assert torch.all(torch.isfinite(next_observation))
        assert torch.all(torch.isfinite(reward))
        assert not torch.any(terminated)
        assert not torch.any(truncated)
        assert tuple(step_info) == PUBLIC_INFO_KEYS
    finally:
        env.close()
