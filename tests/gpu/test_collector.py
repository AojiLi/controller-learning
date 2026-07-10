"""GPU tests for CUDA rollout ownership and exact ``NEXT_STEP`` PPO accounting."""

from __future__ import annotations

import importlib
from dataclasses import replace
from pathlib import Path
from typing import Any

import gymnasium as gym
import jax
import jax.numpy as jnp
import numpy as np
import pytest

from controller_learning.config import load_project_config
from controller_learning.envs.vector_racing import VecCarRacingEnv
from controller_learning.rl.configuration import load_ppo_config
from controller_learning.rl.features import LOCAL_TRACK_FEATURE_DIM, LocalTrackObservationVecEnv
from controller_learning.rl.reward import PublicRewardShapingVecEnv
from controller_learning.rl.rollout import (
    TransitionCounts,
    build_rollout_transition_masks,
    generalized_advantage_estimate,
)
from controller_learning.rl.torch_bridge import JaxToTorchVecEnv
from controller_learning.tracks.generator import generate_track_candidate, pack_track
from controller_learning.tracks.specs import (
    generation_spec_from_project,
    track_capacity_from_project,
)

PROJECT_ROOT = Path(__file__).parents[2]
pytestmark = pytest.mark.gpu


def _torch() -> Any:
    return importlib.import_module("torch")


def _collector_module() -> Any:
    return importlib.import_module("controller_learning.rl.collector")


def _policy_module() -> Any:
    return importlib.import_module("controller_learning.rl.policy")


def _device() -> Any:
    torch = _torch()
    return torch.device("cuda", torch.cuda.current_device())


def _hand_rollout() -> tuple[np.ndarray, np.ndarray, Any]:
    terminated = np.asarray(
        (
            (False, False),
            (True, False),
            (False, False),
            (False, False),
        ),
        dtype=np.bool_,
    )
    truncated = np.asarray(
        (
            (False, False),
            (False, True),
            (False, False),
            (False, False),
        ),
        dtype=np.bool_,
    )
    masks = build_rollout_transition_masks(
        np.zeros(2, dtype=np.bool_),
        terminated,
        truncated,
    )
    rewards = np.asarray(
        (
            (1.0, 1.0),
            (2.0, 2.0),
            (0.0, 0.0),
            (3.0, 3.0),
        ),
        dtype=np.float64,
    )
    values = np.asarray(
        (
            (10.0, 5.0),
            (20.0, 6.0),
            (99.0, 7.0),
            (30.0, 8.0),
            (40.0, 9.0),
        ),
        dtype=np.float64,
    )
    return rewards, values, masks


def test_torch_gae_matches_numpy_reference_for_termination_truncation_and_dummy_rows() -> None:
    torch = _torch()
    rewards, values, masks = _hand_rollout()
    reference = generalized_advantage_estimate(
        rewards,
        values,
        masks,
        gamma=0.9,
        gae_lambda=0.8,
    )
    torch_masks = _collector_module().build_torch_rollout_transition_masks(
        torch.zeros(2, dtype=torch.bool, device=_device()),
        torch.as_tensor(np.array(masks.terminated), device=_device()),
        torch.as_tensor(np.array(masks.truncated), device=_device()),
    )

    assert torch.equal(
        torch_masks.valid_transition.cpu(),
        torch.from_numpy(np.array(masks.valid_transition)),
    )
    assert torch.equal(
        torch_masks.reset_only.cpu(),
        torch.from_numpy(np.array(masks.reset_only)),
    )
    assert torch.equal(
        torch_masks.terminal_event.cpu(),
        torch.from_numpy(np.array(masks.terminal_event)),
    )
    assert torch.equal(
        torch_masks.final_pending_reset.cpu(),
        torch.from_numpy(np.array(masks.final_pending_reset)),
    )

    result = _collector_module().torch_generalized_advantage_estimate(
        torch.as_tensor(rewards, device=_device()),
        torch.as_tensor(values, device=_device()),
        valid_transition=torch_masks.valid_transition,
        reset_only=torch_masks.reset_only,
        terminated=torch_masks.terminated,
        truncated=torch_masks.truncated,
        gamma=0.9,
        gae_lambda=0.8,
    )

    torch.testing.assert_close(
        result.temporal_difference.cpu(),
        torch.from_numpy(np.array(reference.temporal_difference)),
        rtol=0.0,
        atol=1.0e-12,
    )
    torch.testing.assert_close(
        result.advantages.cpu(),
        torch.from_numpy(np.array(reference.advantages)),
        rtol=0.0,
        atol=1.0e-12,
    )
    torch.testing.assert_close(
        result.returns.cpu(),
        torch.from_numpy(np.array(reference.returns)),
        rtol=0.0,
        atol=1.0e-12,
    )
    assert result.returns[1, 0].item() == pytest.approx(2.0)
    assert result.returns[1, 1].item() == pytest.approx(2.0 + 0.9 * 7.0)
    assert torch.count_nonzero(result.advantages[2]).item() == 0
    assert torch.count_nonzero(result.returns[2]).item() == 0


def test_torch_gae_rejects_reset_only_reward_or_terminal_leakage() -> None:
    torch = _torch()
    device = _device()
    rewards = torch.tensor(((0.0,), (1.0,)), device=device)
    values = torch.zeros((3, 1), device=device)
    valid = torch.tensor(((False,), (True,)), dtype=torch.bool, device=device)
    reset = torch.logical_not(valid)
    terminal = torch.zeros_like(valid)

    bad_reward = rewards.clone()
    bad_reward[0, 0] = 1.0
    with pytest.raises(ValueError, match="reset-only rewards"):
        _collector_module().torch_generalized_advantage_estimate(
            bad_reward,
            values,
            valid_transition=valid,
            reset_only=reset,
            terminated=terminal,
            truncated=terminal,
            gamma=0.99,
            gae_lambda=0.95,
        )

    bad_terminal = terminal.clone()
    bad_terminal[0, 0] = True
    with pytest.raises(ValueError, match="reset-only row"):
        _collector_module().torch_generalized_advantage_estimate(
            rewards,
            values,
            valid_transition=valid,
            reset_only=reset,
            terminated=bad_terminal,
            truncated=terminal,
            gamma=0.99,
            gae_lambda=0.95,
        )


class _InjectInvalidActionOnCall(gym.vector.VectorWrapper):
    """Test-only wrapper that triggers one official invalid-action termination."""

    def __init__(self, env: gym.vector.VectorEnv, *, call: int, world: int) -> None:
        super().__init__(env)
        self._target_call = call
        self._target_world = world
        self._calls = 0

    def step(self, actions: Any) -> tuple[Any, Any, Any, Any, dict[str, Any]]:
        self._calls += 1
        if self._calls == self._target_call:
            actions = jnp.asarray(actions).at[self._target_world, 0].set(jnp.nan)
        return self.env.step(actions)


def _small_formal_stack(
    *,
    inject_invalid: bool = True,
    short_timeout: bool = False,
) -> tuple[JaxToTorchVecEnv, Any]:
    project = load_project_config(PROJECT_ROOT)
    if short_timeout:
        project = replace(
            project,
            benchmark=replace(
                project.benchmark,
                episode=replace(
                    project.benchmark.episode,
                    minimum_timeout_s=0.1,
                    timeout_reference_speed_mps=1.0e6,
                ),
            ),
        )
    ppo = load_ppo_config(PROJECT_ROOT / "configs" / "ppo.toml")
    generation = generation_spec_from_project(project)
    capacity = track_capacity_from_project(project)
    track = pack_track(generate_track_candidate(401, generation), capacity)
    base = VecCarRacingEnv(
        num_envs=2,
        project_config=project,
        level_id=1,
        tracks=(track, track),
        backend="mjx_warp",
    )
    shaped = PublicRewardShapingVecEnv(base, ppo.reward)
    featured = LocalTrackObservationVecEnv(shaped, config=ppo.observation)
    outer = _InjectInvalidActionOnCall(featured, call=2, world=1) if inject_invalid else featured
    env = JaxToTorchVecEnv(outer, device=_device())
    policy = _policy_module().PpoActorCritic(
        LOCAL_TRACK_FEATURE_DIM,
        action_low=base.single_action_space.low,
        action_high=base.single_action_space.high,
        policy_seed=ppo.ppo.policy_seed,
        initial_log_std=-5.0,
        device=env.device,
    )
    torch = _torch()
    with torch.no_grad():
        policy.actor_mean.weight.zero_()
        policy.actor_mean.bias.copy_(torch.tensor((0.0, 2.0), device=env.device))
    return env, policy


def test_collector_owns_cuda_rollout_and_preserves_pending_reset_between_updates() -> None:
    torch = _torch()
    env, policy = _small_formal_stack()
    collector = _collector_module().TorchRolloutCollector(env, policy, rollout_steps=2)
    generator = torch.Generator(device=env.device).manual_seed(409)
    try:
        state = collector.initialize(seed=419)
        rollout_a = collector.collect(state, generator=generator)
        saved_a = rollout_a.observations.clone()

        assert rollout_a.shape == (2, 2)
        expected_shapes = {
            "observations": (2, 2, LOCAL_TRACK_FEATURE_DIM),
            "pre_tanh_actions": (2, 2, 2),
            "actions": (2, 2, 2),
            "old_log_prob": (2, 2),
            "values": (3, 2),
            "rewards": (2, 2),
            "terminated": (2, 2),
            "truncated": (2, 2),
            "termination_reason": (2, 2),
            "lap_completed": (2, 2),
            "lap_time_s": (2, 2),
            "episode_seed": (2, 2),
            "controller_seed": (2, 2),
            "track_id": (2, 2),
            "valid_transition": (2, 2),
            "reset_only": (2, 2),
        }
        for name, shape in expected_shapes.items():
            tensor = getattr(rollout_a, name)
            assert tensor.shape == shape
            assert tensor.device == env.device
            assert tensor.is_contiguous()
            assert not tensor.requires_grad
        assert rollout_a.counts == TransitionCounts(
            num_envs=2,
            environment_step_calls=2,
            raw_transitions=4,
            valid_transitions=4,
            dummy_reset_transitions=0,
            autoreset_slots=0,
            terminal_events=1,
            terminated_events=1,
            truncated_events=0,
        )
        assert torch.equal(
            rollout_a.final_state.pending_reset,
            torch.tensor((False, True), dtype=torch.bool, device=env.device),
        )
        assert rollout_a.terminated[1, 1]
        assert rollout_a.termination_reason[1, 1] == 3
        assert not torch.any(rollout_a.lap_completed)
        assert torch.all(rollout_a.lap_time_s == 0.0)
        assert not torch.any(rollout_a.truncated)
        assert torch.all(rollout_a.actions >= policy.action_low)
        assert torch.all(rollout_a.actions <= policy.action_high)
        assert not torch.equal(rollout_a.observations[0, 0], rollout_a.observations[1, 0])

        gae_a = rollout_a.generalized_advantage_estimate(gamma=0.99, gae_lambda=0.95)
        torch.testing.assert_close(
            gae_a.returns[1, 1],
            rollout_a.rewards[1, 1],
            rtol=1.0e-6,
            atol=1.0e-6,
        )

        rollout_b = collector.collect(rollout_a.final_state, generator=generator)
        torch.testing.assert_close(rollout_a.observations, saved_a, rtol=0.0, atol=0.0)
        assert rollout_b.counts == TransitionCounts(
            num_envs=2,
            environment_step_calls=2,
            raw_transitions=4,
            valid_transitions=3,
            dummy_reset_transitions=1,
            autoreset_slots=1,
            terminal_events=0,
            terminated_events=0,
            truncated_events=0,
        )
        assert torch.equal(
            rollout_b.reset_only,
            torch.tensor(((False, True), (False, False)), device=env.device),
        )
        assert torch.equal(rollout_b.valid_transition, torch.logical_not(rollout_b.reset_only))
        assert rollout_b.rewards[0, 1].item() == 0.0
        assert rollout_b.termination_reason[0, 1].item() == 0
        assert not rollout_b.lap_completed[0, 1]
        assert rollout_b.lap_time_s[0, 1].item() == 0.0
        assert not rollout_b.terminated[0, 1]
        assert not rollout_b.truncated[0, 1]
        torch.testing.assert_close(
            rollout_b.observations[0, 1],
            rollout_a.final_state.observation[1],
            rtol=0.0,
            atol=0.0,
        )
        torch.testing.assert_close(
            rollout_b.values[0, 1],
            rollout_a.values[-1, 1],
            rtol=0.0,
            atol=0.0,
        )
        gae_b = rollout_b.generalized_advantage_estimate(gamma=0.99, gae_lambda=0.95)
        assert gae_b.temporal_difference[0, 1].item() == 0.0
        assert gae_b.advantages[0, 1].item() == 0.0
        assert gae_b.returns[0, 1].item() == 0.0
        assert not torch.any(rollout_b.final_state.pending_reset)
    finally:
        env.close()


def test_collector_requires_explicit_compatible_state_and_policy_generator() -> None:
    torch = _torch()
    env, policy = _small_formal_stack()
    collector = _collector_module().TorchRolloutCollector(env, policy, rollout_steps=1)
    try:
        state = collector.initialize(seed=421)
        with pytest.raises(TypeError, match=r"explicit torch\.Generator"):
            collector.collect(state, generator=None)  # type: ignore[arg-type]
        with pytest.raises(ValueError, match=r"state\.pending_reset is on cpu"):
            collector.collect(
                type(state)(state.observation, state.pending_reset.cpu()),
                generator=torch.Generator(device=env.device),
            )
    finally:
        env.close()


@pytest.mark.parametrize("rollout_steps", (2, 3))
def test_actual_timeout_bootstraps_terminal_observation_at_end_or_mid_rollout(
    rollout_steps: int,
) -> None:
    torch = _torch()
    env, policy = _small_formal_stack(inject_invalid=False, short_timeout=True)
    collector = _collector_module().TorchRolloutCollector(
        env,
        policy,
        rollout_steps=rollout_steps,
    )
    generator = torch.Generator(device=env.device).manual_seed(431)
    try:
        rollout = collector.collect(collector.initialize(seed=433), generator=generator)
        assert torch.all(rollout.truncated[1])
        assert torch.all(rollout.termination_reason[1] == 4)
        assert not torch.any(rollout.terminated)
        assert torch.all(rollout.final_state.pending_reset == (rollout_steps == 2))
        terminal_observation = (
            rollout.final_state.observation if rollout_steps == 2 else rollout.observations[2]
        )
        with torch.no_grad():
            terminal_value = policy.value(terminal_observation)
        torch.testing.assert_close(
            rollout.values[2],
            terminal_value,
            rtol=0.0,
            atol=0.0,
        )

        gae = rollout.generalized_advantage_estimate(gamma=0.9, gae_lambda=0.8)
        torch.testing.assert_close(
            gae.returns[1],
            rollout.rewards[1] + 0.9 * rollout.values[2],
            rtol=1.0e-6,
            atol=1.0e-6,
        )
        if rollout_steps == 3:
            assert torch.all(rollout.reset_only[2])
            assert torch.all(rollout.rewards[2] == 0.0)
            assert torch.all(gae.advantages[2] == 0.0)
            assert torch.all(gae.returns[2] == 0.0)
    finally:
        env.close()


def test_collector_module_does_not_copy_device_data_through_numpy() -> None:
    source = (PROJECT_ROOT / "controller_learning" / "rl" / "collector.py").read_text(
        encoding="utf-8"
    )
    assert "numpy" not in source
    assert ".cpu()" not in source
    assert ".item()" not in source
    assert source.count('.to(device="cpu")') == 3
    assert "next_observation_buffer.copy_(validated_next)" in source
    assert jax.default_backend() == "gpu"
