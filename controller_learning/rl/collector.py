"""CUDA-native PPO rollout collection with exact ``NEXT_STEP`` accounting.

The official vector environment returns Torch views over JAX buffers through DLPack.  Those views
are deliberately copied into collector-owned, preallocated CUDA storage before another environment
step can reuse simulator buffers.  Deferred autoreset rows remain in the fixed-width rollout for
accounting, but masks exclude them from GAE and later PPO losses.
"""

from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import dataclass
from numbers import Real
from typing import Any

import torch
from torch import Tensor

from controller_learning.rl.policy import PpoActorCritic
from controller_learning.rl.rollout import TransitionCounts
from controller_learning.rl.torch_bridge import JaxToTorchVecEnv


def _positive_integer(value: object, *, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise ValueError(f"{name} must be a positive integer")
    return value


def _uint32_seed(value: object, *, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or not 0 <= value < 2**32:
        raise ValueError(f"{name} must be an integer in the uint32 range")
    return value


def _probability(value: Real, *, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, Real):
        raise TypeError(f"{name} must be a real number")
    result = float(value)
    if not math.isfinite(result) or not 0.0 <= result <= 1.0:
        raise ValueError(f"{name} must be finite and in [0, 1]")
    return result


def _validate_tensor(
    value: Any,
    *,
    name: str,
    shape: tuple[int, ...],
    dtype: torch.dtype,
    device: torch.device,
) -> Tensor:
    if not isinstance(value, Tensor):
        raise TypeError(f"{name} must be a torch.Tensor")
    if value.shape != shape:
        raise ValueError(f"{name} must have shape {shape}, got {tuple(value.shape)}")
    if value.dtype is not dtype:
        raise TypeError(f"{name} must use {dtype}, got {value.dtype}")
    if value.device != device:
        raise ValueError(f"{name} is on {value.device}, but the collector is on {device}")
    return value


@dataclass(frozen=True, slots=True)
class CollectorState:
    """Observation and deferred-reset carry required to continue the next rollout."""

    observation: Tensor
    pending_reset: Tensor


@dataclass(frozen=True, slots=True)
class TorchGaeResult:
    """Device-native temporal differences, advantages, and value targets."""

    temporal_difference: Tensor
    advantages: Tensor
    returns: Tensor


@dataclass(frozen=True, slots=True)
class TorchRolloutMasks:
    """Time-major Torch masks derived from one explicit pending-reset carry."""

    initial_pending_reset: Tensor
    valid_transition: Tensor
    reset_only: Tensor
    terminated: Tensor
    truncated: Tensor
    terminal_event: Tensor
    final_pending_reset: Tensor


@dataclass(frozen=True, slots=True)
class CollectedRollout:
    """One time-major rollout whose tensors are owned by the collector."""

    observations: Tensor
    pre_tanh_actions: Tensor
    actions: Tensor
    old_log_prob: Tensor
    values: Tensor
    rewards: Tensor
    terminated: Tensor
    truncated: Tensor
    termination_reason: Tensor
    lap_completed: Tensor
    lap_time_s: Tensor
    episode_seed: Tensor
    controller_seed: Tensor
    track_id: Tensor
    valid_transition: Tensor
    reset_only: Tensor
    initial_pending_reset: Tensor
    final_state: CollectorState
    counts: TransitionCounts

    @property
    def shape(self) -> tuple[int, int]:
        """Return ``(rollout_steps, num_envs)``."""

        return self.rewards.shape

    def generalized_advantage_estimate(
        self,
        *,
        gamma: Real,
        gae_lambda: Real,
    ) -> TorchGaeResult:
        """Compute masked Torch GAE directly from this rollout."""

        return torch_generalized_advantage_estimate(
            self.rewards,
            self.values,
            valid_transition=self.valid_transition,
            reset_only=self.reset_only,
            terminated=self.terminated,
            truncated=self.truncated,
            gamma=gamma,
            gae_lambda=gae_lambda,
        )


def build_torch_rollout_transition_masks(
    initial_pending_reset: Any,
    terminated: Any,
    truncated: Any,
) -> TorchRolloutMasks:
    """Derive fixed-width ``NEXT_STEP`` masks without leaving Torch."""

    if not isinstance(initial_pending_reset, Tensor):
        raise TypeError("initial_pending_reset must be a torch.Tensor")
    if initial_pending_reset.ndim != 1 or initial_pending_reset.shape[0] < 1:
        raise ValueError("initial_pending_reset must be a non-empty one-dimensional tensor")
    initial = _validate_tensor(
        initial_pending_reset,
        name="initial_pending_reset",
        shape=tuple(initial_pending_reset.shape),
        dtype=torch.bool,
        device=initial_pending_reset.device,
    )
    if not isinstance(terminated, Tensor) or terminated.ndim != 2:
        raise ValueError("terminated must be a two-dimensional torch.Tensor")
    shape = tuple(terminated.shape)
    if shape[0] < 1 or shape[1] != initial.shape[0]:
        raise ValueError("terminal flag shape must be non-empty and match initial_pending_reset")
    term = _validate_tensor(
        terminated,
        name="terminated",
        shape=shape,
        dtype=torch.bool,
        device=initial.device,
    )
    trunc = _validate_tensor(
        truncated,
        name="truncated",
        shape=shape,
        dtype=torch.bool,
        device=initial.device,
    )

    valid = torch.empty_like(term)
    reset = torch.empty_like(term)
    terminal = torch.empty_like(term)
    pending = initial.clone()
    for step in range(shape[0]):
        reset[step].copy_(pending)
        torch.logical_not(pending, out=valid[step])
        torch.logical_or(term[step], trunc[step], out=terminal[step])
        pending.copy_(terminal[step])

    # Validate the value-dependent NEXT_STEP invariants once after constructing the full masks.
    checks = torch.stack(
        (
            torch.logical_not(torch.any(term & trunc)),
            torch.logical_not(torch.any(reset & terminal)),
        )
    ).to(device="cpu")
    valid_checks = checks.tolist()
    if not valid_checks[0]:
        raise ValueError("a transition cannot be both terminated and truncated")
    if not valid_checks[1]:
        raise ValueError("a NEXT_STEP reset-only row must return false terminal flags")
    return TorchRolloutMasks(
        initial_pending_reset=initial.clone(),
        valid_transition=valid,
        reset_only=reset,
        terminated=term,
        truncated=trunc,
        terminal_event=terminal,
        final_pending_reset=pending,
    )


def _validate_gae_inputs(
    rewards: Any,
    values: Any,
    *,
    valid_transition: Any,
    reset_only: Any,
    terminated: Any,
    truncated: Any,
) -> tuple[Tensor, Tensor, Tensor, Tensor, Tensor, Tensor]:
    if not isinstance(rewards, Tensor):
        raise TypeError("rewards must be a torch.Tensor")
    if rewards.ndim != 2 or any(size < 1 for size in rewards.shape):
        raise ValueError("rewards must be a non-empty two-dimensional tensor")
    if not rewards.dtype.is_floating_point:
        raise TypeError("rewards must have a floating dtype")
    shape = tuple(rewards.shape)
    value_shape = (shape[0] + 1, shape[1])
    reward = _validate_tensor(
        rewards,
        name="rewards",
        shape=shape,
        dtype=rewards.dtype,
        device=rewards.device,
    )
    value = _validate_tensor(
        values,
        name="values",
        shape=value_shape,
        dtype=rewards.dtype,
        device=rewards.device,
    )
    masks = tuple(
        _validate_tensor(
            candidate,
            name=name,
            shape=shape,
            dtype=torch.bool,
            device=rewards.device,
        )
        for name, candidate in (
            ("valid_transition", valid_transition),
            ("reset_only", reset_only),
            ("terminated", terminated),
            ("truncated", truncated),
        )
    )
    return reward, value, *masks


def torch_generalized_advantage_estimate(
    rewards: Any,
    values: Any,
    *,
    valid_transition: Any,
    reset_only: Any,
    terminated: Any,
    truncated: Any,
    gamma: Real,
    gae_lambda: Real,
) -> TorchGaeResult:
    """Compute GAE without leaving Torch or allowing reset-only rows into learning.

    Terminations neither bootstrap nor recurse.  Truncations bootstrap from the returned terminal
    observation in ``values[t + 1]`` but stop recursion at the episode boundary.  A deferred reset
    row has exactly zero delta, advantage, and return.
    """

    reward, value, valid, reset, term, trunc = _validate_gae_inputs(
        rewards,
        values,
        valid_transition=valid_transition,
        reset_only=reset_only,
        terminated=terminated,
        truncated=truncated,
    )
    discount = _probability(gamma, name="gamma")
    trace_decay = _probability(gae_lambda, name="gae_lambda")

    temporal_difference = torch.zeros_like(reward)
    advantages = torch.zeros_like(reward)
    last_advantage = torch.zeros(reward.shape[1], dtype=reward.dtype, device=reward.device)
    terminal = torch.logical_or(term, trunc)

    for step in range(reward.shape[0] - 1, -1, -1):
        bootstrap = torch.logical_not(term[step]).to(dtype=reward.dtype)
        continuation = torch.logical_not(terminal[step]).to(dtype=reward.dtype)
        delta = reward[step] + discount * bootstrap * value[step + 1] - value[step]
        candidate = delta + discount * trace_decay * continuation * last_advantage
        temporal_difference[step] = torch.where(valid[step], delta, 0.0)
        last_advantage = torch.where(valid[step], candidate, 0.0)
        advantages[step] = last_advantage

    returns = torch.where(valid, advantages + value[:-1], 0.0)

    # All value-dependent checks share one synchronization at this public function boundary.
    checks = torch.stack(
        (
            torch.all(valid == torch.logical_not(reset)),
            torch.logical_not(torch.any(term & trunc)),
            torch.logical_not(torch.any(reset & terminal)),
            torch.logical_not(torch.any(reset & (reward != 0.0))),
            torch.all(torch.isfinite(reward)),
            torch.all(torch.isfinite(value)),
            torch.all(torch.isfinite(temporal_difference)),
            torch.all(torch.isfinite(advantages)),
            torch.all(torch.isfinite(returns)),
        )
    ).to(device="cpu")
    valid_checks = checks.tolist()
    if not valid_checks[0]:
        raise ValueError("valid_transition must be the exact complement of reset_only")
    if not valid_checks[1]:
        raise ValueError("a transition cannot be both terminated and truncated")
    if not valid_checks[2]:
        raise ValueError("a NEXT_STEP reset-only row must return false terminal flags")
    if not valid_checks[3]:
        raise ValueError("NEXT_STEP reset-only rewards must be exactly zero")
    if not all(valid_checks[4:]):
        raise FloatingPointError("GAE inputs or outputs contain a non-finite value")
    return TorchGaeResult(
        temporal_difference=temporal_difference,
        advantages=advantages,
        returns=returns,
    )


class TorchRolloutCollector:
    """Collect fixed-length PPO rollouts from the official CUDA wrapper stack."""

    def __init__(
        self,
        env: JaxToTorchVecEnv,
        policy: PpoActorCritic,
        *,
        rollout_steps: int,
    ) -> None:
        if not isinstance(env, JaxToTorchVecEnv):
            raise TypeError("env must be a JaxToTorchVecEnv")
        if not isinstance(policy, PpoActorCritic):
            raise TypeError("policy must be a PpoActorCritic")
        if env.device != policy.device:
            raise ValueError(
                f"environment is on {env.device}, but the policy is on {policy.device}"
            )
        if policy.device.type != "cuda":
            raise ValueError("TorchRolloutCollector requires CUDA")
        observation_shape = env.single_observation_space.shape
        if observation_shape != (policy.observation_dim,):
            raise ValueError(
                "environment and policy observation dimensions differ: "
                f"{observation_shape} != {(policy.observation_dim,)}"
            )

        self.env = env
        self.policy = policy
        self.rollout_steps = _positive_integer(rollout_steps, name="rollout_steps")
        self.num_envs = env.num_envs
        self.observation_dim = policy.observation_dim
        self.action_dim = policy.action_dim
        self.device = policy.device
        self.dtype = policy.dtype

    def initialize(self, *, seed: int) -> CollectorState:
        """Reset the environment and create an owned zero-pending collector state."""

        environment_seed = _uint32_seed(seed, name="seed")
        observation, _ = self.env.reset(seed=environment_seed)
        current = _validate_tensor(
            observation,
            name="reset observation",
            shape=(self.num_envs, self.observation_dim),
            dtype=self.dtype,
            device=self.device,
        ).clone()
        if not bool(torch.all(torch.isfinite(current))):
            raise FloatingPointError("reset observation contains a non-finite value")
        return CollectorState(
            observation=current,
            pending_reset=torch.zeros(self.num_envs, dtype=torch.bool, device=self.device),
        )

    def _validate_state(self, state: Any) -> CollectorState:
        if not isinstance(state, CollectorState):
            raise TypeError("state must be a CollectorState")
        observation = _validate_tensor(
            state.observation,
            name="state.observation",
            shape=(self.num_envs, self.observation_dim),
            dtype=self.dtype,
            device=self.device,
        )
        pending_reset = _validate_tensor(
            state.pending_reset,
            name="state.pending_reset",
            shape=(self.num_envs,),
            dtype=torch.bool,
            device=self.device,
        )
        return CollectorState(observation=observation, pending_reset=pending_reset)

    def collect(
        self,
        state: CollectorState,
        *,
        generator: torch.Generator,
    ) -> CollectedRollout:
        """Collect one rollout using only the caller-owned policy generator."""

        checked_state = self._validate_state(state)
        # Validate the explicit RNG before allocating the relatively large rollout buffers.
        self.policy._validate_generator(generator)

        time_world = (self.rollout_steps, self.num_envs)
        observations = torch.empty(
            (*time_world, self.observation_dim),
            dtype=self.dtype,
            device=self.device,
        )
        pre_tanh_actions = torch.empty(
            (*time_world, self.action_dim),
            dtype=self.dtype,
            device=self.device,
        )
        actions = torch.empty_like(pre_tanh_actions)
        old_log_prob = torch.empty(time_world, dtype=self.dtype, device=self.device)
        values = torch.empty(
            (self.rollout_steps + 1, self.num_envs),
            dtype=self.dtype,
            device=self.device,
        )
        rewards = torch.empty(time_world, dtype=self.dtype, device=self.device)
        terminated = torch.empty(time_world, dtype=torch.bool, device=self.device)
        truncated = torch.empty_like(terminated)
        termination_reason = torch.empty(time_world, dtype=torch.int32, device=self.device)
        lap_completed = torch.empty_like(terminated)
        lap_time_s = torch.empty(time_world, dtype=self.dtype, device=self.device)
        episode_seed = torch.empty(time_world, dtype=torch.uint32, device=self.device)
        controller_seed = torch.empty_like(episode_seed)
        track_id = torch.empty_like(episode_seed)
        valid_transition = torch.empty_like(terminated)
        reset_only = torch.empty_like(terminated)

        initial_pending_reset = checked_state.pending_reset.clone()
        pending_reset = checked_state.pending_reset.clone()
        # The two owned observation buffers alternate.  This makes every DLPack result independent
        # from simulator storage before the following environment call.
        current_observation = checked_state.observation.clone()
        next_observation_buffer = torch.empty_like(current_observation)

        with torch.no_grad():
            for step in range(self.rollout_steps):
                observations[step].copy_(current_observation)
                reset_only[step].copy_(pending_reset)
                torch.logical_not(pending_reset, out=valid_transition[step])

                sample = self.policy.sample(current_observation, generator=generator)
                pre_tanh_actions[step].copy_(sample.pre_tanh)
                actions[step].copy_(sample.action)
                old_log_prob[step].copy_(sample.log_prob)
                values[step].copy_(sample.value)

                next_observation, reward, term, trunc, info = self.env.step(sample.action)
                if not isinstance(info, Mapping):
                    raise TypeError("public vector info must be a mapping")
                validated_next = _validate_tensor(
                    next_observation,
                    name="step observation",
                    shape=(self.num_envs, self.observation_dim),
                    dtype=self.dtype,
                    device=self.device,
                )
                next_observation_buffer.copy_(validated_next)
                rewards[step].copy_(
                    _validate_tensor(
                        reward,
                        name="reward",
                        shape=(self.num_envs,),
                        dtype=self.dtype,
                        device=self.device,
                    )
                )
                terminated[step].copy_(
                    _validate_tensor(
                        term,
                        name="terminated",
                        shape=(self.num_envs,),
                        dtype=torch.bool,
                        device=self.device,
                    )
                )
                truncated[step].copy_(
                    _validate_tensor(
                        trunc,
                        name="truncated",
                        shape=(self.num_envs,),
                        dtype=torch.bool,
                        device=self.device,
                    )
                )
                termination_reason[step].copy_(
                    _validate_tensor(
                        info["termination_reason"],
                        name="info['termination_reason']",
                        shape=(self.num_envs,),
                        dtype=torch.int32,
                        device=self.device,
                    )
                )
                lap_completed[step].copy_(
                    _validate_tensor(
                        info["lap_completed"],
                        name="info['lap_completed']",
                        shape=(self.num_envs,),
                        dtype=torch.bool,
                        device=self.device,
                    )
                )
                lap_time_s[step].copy_(
                    _validate_tensor(
                        info["lap_time_s"],
                        name="info['lap_time_s']",
                        shape=(self.num_envs,),
                        dtype=self.dtype,
                        device=self.device,
                    )
                )
                episode_seed[step].copy_(
                    _validate_tensor(
                        info["episode_seed"],
                        name="info['episode_seed']",
                        shape=(self.num_envs,),
                        dtype=torch.uint32,
                        device=self.device,
                    )
                )
                controller_seed[step].copy_(
                    _validate_tensor(
                        info["controller_seed"],
                        name="info['controller_seed']",
                        shape=(self.num_envs,),
                        dtype=torch.uint32,
                        device=self.device,
                    )
                )
                track_id[step].copy_(
                    _validate_tensor(
                        info["track_id"],
                        name="info['track_id']",
                        shape=(self.num_envs,),
                        dtype=torch.uint32,
                        device=self.device,
                    )
                )
                torch.logical_or(terminated[step], truncated[step], out=pending_reset)
                current_observation, next_observation_buffer = (
                    next_observation_buffer,
                    current_observation,
                )

            values[self.rollout_steps].copy_(self.policy.value(current_observation))

        final_state = CollectorState(
            observation=current_observation,
            pending_reset=pending_reset.clone(),
        )
        counts = self._validate_boundary(
            observations=observations,
            pre_tanh_actions=pre_tanh_actions,
            actions=actions,
            old_log_prob=old_log_prob,
            values=values,
            rewards=rewards,
            terminated=terminated,
            truncated=truncated,
            termination_reason=termination_reason,
            lap_completed=lap_completed,
            lap_time_s=lap_time_s,
            episode_seed=episode_seed,
            controller_seed=controller_seed,
            track_id=track_id,
            valid_transition=valid_transition,
            reset_only=reset_only,
            initial_pending_reset=initial_pending_reset,
            final_state=final_state,
        )
        return CollectedRollout(
            observations=observations,
            pre_tanh_actions=pre_tanh_actions,
            actions=actions,
            old_log_prob=old_log_prob,
            values=values,
            rewards=rewards,
            terminated=terminated,
            truncated=truncated,
            termination_reason=termination_reason,
            lap_completed=lap_completed,
            lap_time_s=lap_time_s,
            episode_seed=episode_seed,
            controller_seed=controller_seed,
            track_id=track_id,
            valid_transition=valid_transition,
            reset_only=reset_only,
            initial_pending_reset=initial_pending_reset,
            final_state=final_state,
            counts=counts,
        )

    def _validate_boundary(
        self,
        *,
        observations: Tensor,
        pre_tanh_actions: Tensor,
        actions: Tensor,
        old_log_prob: Tensor,
        values: Tensor,
        rewards: Tensor,
        terminated: Tensor,
        truncated: Tensor,
        termination_reason: Tensor,
        lap_completed: Tensor,
        lap_time_s: Tensor,
        episode_seed: Tensor,
        controller_seed: Tensor,
        track_id: Tensor,
        valid_transition: Tensor,
        reset_only: Tensor,
        initial_pending_reset: Tensor,
        final_state: CollectorState,
    ) -> TransitionCounts:
        """Validate all device values and transfer one compact summary to the host."""

        terminal = torch.logical_or(terminated, truncated)
        numerical_failure = torch.logical_not(
            torch.stack(
                (
                    torch.all(torch.isfinite(observations)),
                    torch.all(torch.isfinite(pre_tanh_actions)),
                    torch.all(torch.isfinite(actions)),
                    torch.all(torch.isfinite(old_log_prob)),
                    torch.all(torch.isfinite(values)),
                    torch.all(torch.isfinite(rewards)),
                    torch.all(torch.isfinite(lap_time_s)),
                    torch.all(torch.isfinite(final_state.observation)),
                )
            ).all()
        )
        summary = torch.stack(
            (
                valid_transition.sum(dtype=torch.int64),
                reset_only.sum(dtype=torch.int64),
                terminated.sum(dtype=torch.int64),
                truncated.sum(dtype=torch.int64),
                initial_pending_reset.sum(dtype=torch.int64),
                final_state.pending_reset.sum(dtype=torch.int64),
                (terminated & truncated).sum(dtype=torch.int64),
                (reset_only & terminal).sum(dtype=torch.int64),
                (reset_only & (rewards != 0.0)).sum(dtype=torch.int64),
                (reset_only & (termination_reason != 0)).sum(dtype=torch.int64),
                (reset_only & lap_completed).sum(dtype=torch.int64),
                (lap_completed & torch.logical_not(terminated)).sum(dtype=torch.int64),
                ((termination_reason < 0) | (termination_reason > 4)).sum(dtype=torch.int64),
                ((termination_reason == 0) != torch.logical_not(terminal)).sum(dtype=torch.int64),
                ((termination_reason == 4) != truncated).sum(dtype=torch.int64),
                ((termination_reason == 1) != lap_completed).sum(dtype=torch.int64),
                (lap_completed & (lap_time_s <= 0.0)).sum(dtype=torch.int64),
                (torch.logical_not(lap_completed) & (lap_time_s != 0.0)).sum(dtype=torch.int64),
                numerical_failure.to(dtype=torch.int64),
            )
        ).to(device="cpu")
        (
            valid_count,
            reset_count,
            terminated_count,
            truncated_count,
            initial_pending_count,
            final_pending_count,
            overlapping_terminal_count,
            reset_terminal_count,
            reset_reward_count,
            reset_reason_count,
            reset_lap_count,
            lap_without_termination_count,
            invalid_reason_count,
            reason_terminal_mismatch_count,
            timeout_mismatch_count,
            success_mismatch_count,
            success_time_count,
            nonsuccess_time_count,
            numerical_failure_count,
        ) = summary.tolist()

        if overlapping_terminal_count:
            raise ValueError("a transition cannot be both terminated and truncated")
        if reset_terminal_count:
            raise ValueError("a NEXT_STEP reset-only row must return false terminal flags")
        if reset_reward_count:
            raise ValueError("NEXT_STEP reset-only rewards must be exactly zero")
        if reset_reason_count or reset_lap_count:
            raise ValueError("NEXT_STEP reset-only public episode info must be neutral")
        if lap_without_termination_count:
            raise ValueError("lap_completed requires a terminated transition")
        if invalid_reason_count:
            raise ValueError("termination_reason must use the public range [0, 4]")
        if reason_terminal_mismatch_count:
            raise ValueError("termination_reason NONE must match nonterminal transitions")
        if timeout_mismatch_count:
            raise ValueError("termination_reason TIMEOUT must match truncated transitions")
        if success_mismatch_count:
            raise ValueError("termination_reason SUCCESS must match lap_completed")
        if success_time_count or nonsuccess_time_count:
            raise ValueError("lap_time_s must be positive only for completed laps")
        if numerical_failure_count:
            raise FloatingPointError("rollout tensors contain a non-finite value")

        terminal_count = terminated_count + truncated_count
        counts = TransitionCounts(
            num_envs=self.num_envs,
            environment_step_calls=self.rollout_steps,
            raw_transitions=self.rollout_steps * self.num_envs,
            valid_transitions=valid_count,
            dummy_reset_transitions=reset_count,
            autoreset_slots=reset_count,
            terminal_events=terminal_count,
            terminated_events=terminated_count,
            truncated_events=truncated_count,
        )
        if terminal_count - reset_count != final_pending_count - initial_pending_count:
            raise ValueError(
                "terminal_events - autoreset_slots must equal the pending-reset carry change"
            )
        return counts


__all__ = [
    "CollectedRollout",
    "CollectorState",
    "TorchGaeResult",
    "TorchRolloutCollector",
    "TorchRolloutMasks",
    "build_torch_rollout_transition_masks",
    "torch_generalized_advantage_estimate",
]
