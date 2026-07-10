"""Seed-reproducible orchestration for the CUDA-native M7 PPO training loop.

The trainer owns the two stochastic streams that remain after the environment seed is fixed:
policy sampling and PPO minibatch order.  Environment state is intentionally absent from
checkpoint requests because M7 resume semantics start a fresh official vector environment.

All per-world tensors stay on the accelerator until a rollout boundary.  The trainer transfers
one compact public-metric summary per rollout; it never synchronizes inside an environment step.
"""

from __future__ import annotations

import copy
import csv
import math
import os
import time
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from pathlib import Path
from types import TracebackType
from typing import Any, Protocol

import torch
from torch import Tensor

from controller_learning.envs.race_core import RaceTermination
from controller_learning.rl.collector import CollectedRollout, CollectorState, TorchRolloutCollector
from controller_learning.rl.configuration import PpoTrainingConfig
from controller_learning.rl.ppo import PpoUpdater, UpdateMetrics
from controller_learning.rl.rollout import TransitionCounts

TRAINING_METRICS_COLUMNS = (
    "update_index",
    "vector_steps",
    "update_environment_step_calls",
    "update_raw_transitions",
    "update_valid_transitions",
    "update_dummy_reset_transitions",
    "update_autoreset_slots",
    "update_terminal_events",
    "update_terminated_events",
    "update_truncated_events",
    "cumulative_raw_transitions",
    "cumulative_valid_transitions",
    "cumulative_dummy_reset_transitions",
    "cumulative_autoreset_slots",
    "cumulative_discarded_pending_reset_slots",
    "cumulative_terminal_events",
    "cumulative_terminated_events",
    "cumulative_truncated_events",
    "update_successful_episodes",
    "update_offtrack_episodes",
    "update_invalid_action_episodes",
    "update_timeout_episodes",
    "update_success_rate",
    "update_successful_lap_time_sum_s",
    "update_mean_successful_lap_time_s",
    "update_episode_length_sum_steps",
    "update_mean_episode_length_steps",
    "cumulative_successful_episodes",
    "cumulative_offtrack_episodes",
    "cumulative_invalid_action_episodes",
    "cumulative_timeout_episodes",
    "cumulative_success_rate",
    "cumulative_successful_lap_time_sum_s",
    "cumulative_mean_successful_lap_time_s",
    "cumulative_episode_length_sum_steps",
    "cumulative_mean_episode_length_steps",
    "update_reward_sum",
    "cumulative_reward_sum",
    "update_mean_valid_reward",
    "cumulative_mean_valid_reward",
    "learning_rate",
    "ppo_valid_samples",
    "ppo_samples_processed",
    "ppo_epochs_run",
    "ppo_epochs_completed",
    "ppo_minibatches_processed",
    "ppo_early_stopped_for_kl",
    "policy_loss",
    "value_loss",
    "latent_entropy",
    "total_loss",
    "optimization_mean_kl",
    "post_epoch_kl",
    "clip_fraction",
    "mean_gradient_norm_before_clip",
    "max_gradient_norm_before_clip",
    "explained_variance",
    "compute_update_wall_seconds",
    "cumulative_compute_update_seconds",
    "compute_valid_transitions_per_second",
    "cumulative_compute_valid_transitions_per_second",
    "wall_elapsed_before_persistence_seconds",
    "torch_cuda_memory_allocated_bytes",
    "torch_cuda_memory_reserved_bytes",
    "torch_cuda_max_memory_allocated_bytes",
)


def _plain_positive_integer(value: object, *, name: str) -> int:
    if type(value) is not int or value < 1:
        raise ValueError(f"{name} must be a positive integer")
    return value


def _finite_nonnegative(value: object, *, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"{name} must be a real number")
    result = float(value)
    if not math.isfinite(result) or result < 0.0:
        raise ValueError(f"{name} must be finite and non-negative")
    return result


def _finite_positive(value: object, *, name: str) -> float:
    result = _finite_nonnegative(value, name=name)
    if result == 0.0:
        raise ValueError(f"{name} must be positive")
    return result


def _zero_transition_counts(num_envs: int) -> TransitionCounts:
    return TransitionCounts(
        num_envs=num_envs,
        environment_step_calls=0,
        raw_transitions=0,
        valid_transitions=0,
        dummy_reset_transitions=0,
        autoreset_slots=0,
        terminal_events=0,
        terminated_events=0,
        truncated_events=0,
    )


def _pending_reset_slots(
    counts: TransitionCounts,
    discarded_pending_reset_slots: object,
    *,
    name: str,
) -> int:
    """Validate cumulative fresh-reset compensation and return the live pending carry."""

    if not isinstance(counts, TransitionCounts):
        raise TypeError(f"{name}.counts must be TransitionCounts")
    if type(discarded_pending_reset_slots) is not int or discarded_pending_reset_slots < 0:
        raise ValueError(f"{name}.discarded_pending_reset_slots must be a non-negative integer")
    pending = counts.terminal_events - counts.autoreset_slots - discarded_pending_reset_slots
    if not 0 <= pending <= counts.num_envs:
        raise ValueError(f"{name} uncompensated pending-reset slots must be in [0, num_envs]")
    return pending


@dataclass(frozen=True, slots=True)
class EpisodeMetrics:
    """Public terminal-reason and successful-lap totals for one training prefix."""

    episodes: int
    successful_episodes: int
    offtrack_episodes: int
    invalid_action_episodes: int
    timeout_episodes: int
    successful_lap_time_sum_s: float
    episode_length_sum_steps: int

    def __post_init__(self) -> None:
        for field in (
            "episodes",
            "successful_episodes",
            "offtrack_episodes",
            "invalid_action_episodes",
            "timeout_episodes",
        ):
            value = getattr(self, field)
            if type(value) is not int or value < 0:
                raise ValueError(f"{field} must be a non-negative integer")
        categorized = (
            self.successful_episodes
            + self.offtrack_episodes
            + self.invalid_action_episodes
            + self.timeout_episodes
        )
        if self.episodes != categorized:
            raise ValueError("episodes must equal the sum of the four public terminal reasons")
        if type(self.episode_length_sum_steps) is not int or self.episode_length_sum_steps < 0:
            raise ValueError("episode_length_sum_steps must be a non-negative integer")
        if (self.episodes == 0) != (self.episode_length_sum_steps == 0):
            raise ValueError(
                "episode_length_sum_steps must be positive exactly when episodes exist"
            )
        lap_sum = _finite_nonnegative(
            self.successful_lap_time_sum_s,
            name="successful_lap_time_sum_s",
        )
        if (self.successful_episodes == 0) != (lap_sum == 0.0):
            raise ValueError(
                "successful_lap_time_sum_s must be positive exactly when successes exist"
            )
        object.__setattr__(self, "successful_lap_time_sum_s", lap_sum)

    @property
    def success_rate(self) -> float:
        """Return completed laps divided by all terminal events, or zero before any event."""

        return self.successful_episodes / self.episodes if self.episodes else 0.0

    @property
    def mean_successful_lap_time_s(self) -> float:
        """Return the mean successful lap time, or zero before the first success."""

        if self.successful_episodes == 0:
            return 0.0
        return self.successful_lap_time_sum_s / self.successful_episodes

    @property
    def mean_episode_length_steps(self) -> float:
        """Return mean valid transitions per completed episode, or zero before completion."""

        if self.episodes == 0:
            return 0.0
        return self.episode_length_sum_steps / self.episodes

    def __add__(self, other: object) -> EpisodeMetrics:
        if not isinstance(other, EpisodeMetrics):
            return NotImplemented
        return EpisodeMetrics(
            episodes=self.episodes + other.episodes,
            successful_episodes=self.successful_episodes + other.successful_episodes,
            offtrack_episodes=self.offtrack_episodes + other.offtrack_episodes,
            invalid_action_episodes=self.invalid_action_episodes + other.invalid_action_episodes,
            timeout_episodes=self.timeout_episodes + other.timeout_episodes,
            successful_lap_time_sum_s=self.successful_lap_time_sum_s
            + other.successful_lap_time_sum_s,
            episode_length_sum_steps=self.episode_length_sum_steps + other.episode_length_sum_steps,
        )


def _zero_episode_metrics() -> EpisodeMetrics:
    return EpisodeMetrics(
        episodes=0,
        successful_episodes=0,
        offtrack_episodes=0,
        invalid_action_episodes=0,
        timeout_episodes=0,
        successful_lap_time_sum_s=0.0,
        episode_length_sum_steps=0,
    )


@dataclass(frozen=True, slots=True)
class TorchCudaMemoryMetrics:
    """Allocator counters sampled without synchronizing accelerator work."""

    allocated_bytes: int
    reserved_bytes: int
    max_allocated_bytes: int

    def __post_init__(self) -> None:
        for field in ("allocated_bytes", "reserved_bytes", "max_allocated_bytes"):
            value = getattr(self, field)
            if type(value) is not int or value < 0:
                raise ValueError(f"{field} must be a non-negative integer")
        if self.max_allocated_bytes < self.allocated_bytes:
            raise ValueError("max_allocated_bytes cannot be below allocated_bytes")


@dataclass(frozen=True, slots=True)
class UpdateRecord:
    """Immutable host snapshot for one completed collect/GAE/PPO update."""

    update_index: int
    vector_steps: int
    rollout_counts: TransitionCounts
    cumulative_counts: TransitionCounts
    discarded_pending_reset_slots: int
    rollout_episodes: EpisodeMetrics
    cumulative_episodes: EpisodeMetrics
    rollout_reward_sum: float
    cumulative_reward_sum: float
    optimization: UpdateMetrics
    compute_update_wall_seconds: float
    cumulative_compute_update_seconds: float
    compute_valid_transitions_per_second: float
    cumulative_compute_valid_transitions_per_second: float
    wall_elapsed_before_persistence_seconds: float
    torch_cuda_memory: TorchCudaMemoryMetrics | None

    def __post_init__(self) -> None:
        _plain_positive_integer(self.update_index, name="update_index")
        _plain_positive_integer(self.vector_steps, name="vector_steps")
        if not isinstance(self.rollout_counts, TransitionCounts) or not isinstance(
            self.cumulative_counts, TransitionCounts
        ):
            raise TypeError("rollout_counts and cumulative_counts must be TransitionCounts")
        if self.rollout_counts.num_envs != self.cumulative_counts.num_envs:
            raise ValueError("rollout and cumulative transition widths differ")
        if self.vector_steps != self.cumulative_counts.environment_step_calls:
            raise ValueError("vector_steps must equal cumulative environment step calls")
        _pending_reset_slots(
            self.cumulative_counts,
            self.discarded_pending_reset_slots,
            name="update record",
        )
        if not isinstance(self.rollout_episodes, EpisodeMetrics) or not isinstance(
            self.cumulative_episodes, EpisodeMetrics
        ):
            raise TypeError("rollout_episodes and cumulative_episodes must be EpisodeMetrics")
        if self.rollout_episodes.episodes != self.rollout_counts.terminal_events:
            raise ValueError("rollout public episodes must equal rollout terminal events")
        if self.cumulative_episodes.episodes != self.cumulative_counts.terminal_events:
            raise ValueError("cumulative public episodes must equal cumulative terminal events")
        object.__setattr__(
            self,
            "rollout_reward_sum",
            _finite_nonnegative_or_signed(self.rollout_reward_sum, name="rollout_reward_sum"),
        )
        object.__setattr__(
            self,
            "cumulative_reward_sum",
            _finite_nonnegative_or_signed(
                self.cumulative_reward_sum,
                name="cumulative_reward_sum",
            ),
        )
        if not isinstance(self.optimization, UpdateMetrics):
            raise TypeError("optimization must be UpdateMetrics")
        if self.optimization.valid_samples != self.rollout_counts.valid_transitions:
            raise ValueError("PPO valid samples must equal rollout valid transitions")
        for field in (
            "compute_update_wall_seconds",
            "cumulative_compute_update_seconds",
            "compute_valid_transitions_per_second",
            "cumulative_compute_valid_transitions_per_second",
            "wall_elapsed_before_persistence_seconds",
        ):
            object.__setattr__(
                self,
                field,
                _finite_positive(getattr(self, field), name=field),
            )
        if self.cumulative_compute_update_seconds < self.compute_update_wall_seconds:
            raise ValueError(
                "cumulative_compute_update_seconds cannot be below compute_update_wall_seconds"
            )
        if self.wall_elapsed_before_persistence_seconds < self.cumulative_compute_update_seconds:
            raise ValueError("wall_elapsed_before_persistence_seconds cannot be below compute time")
        if self.torch_cuda_memory is not None and not isinstance(
            self.torch_cuda_memory,
            TorchCudaMemoryMetrics,
        ):
            raise TypeError("torch_cuda_memory must be TorchCudaMemoryMetrics or None")

    @property
    def rollout_mean_valid_reward(self) -> float:
        return self.rollout_reward_sum / self.rollout_counts.valid_transitions

    @property
    def cumulative_mean_valid_reward(self) -> float:
        return self.cumulative_reward_sum / self.cumulative_counts.valid_transitions

    def to_csv_row(self) -> dict[str, int | float | str]:
        """Flatten this record into the one supported CSV schema."""

        update = self.rollout_counts
        cumulative = self.cumulative_counts
        update_episode = self.rollout_episodes
        cumulative_episode = self.cumulative_episodes
        optimization = self.optimization
        memory = self.torch_cuda_memory
        row: dict[str, int | float | str] = {
            "update_index": self.update_index,
            "vector_steps": self.vector_steps,
            "update_environment_step_calls": update.environment_step_calls,
            "update_raw_transitions": update.raw_transitions,
            "update_valid_transitions": update.valid_transitions,
            "update_dummy_reset_transitions": update.dummy_reset_transitions,
            "update_autoreset_slots": update.autoreset_slots,
            "update_terminal_events": update.terminal_events,
            "update_terminated_events": update.terminated_events,
            "update_truncated_events": update.truncated_events,
            "cumulative_raw_transitions": cumulative.raw_transitions,
            "cumulative_valid_transitions": cumulative.valid_transitions,
            "cumulative_dummy_reset_transitions": cumulative.dummy_reset_transitions,
            "cumulative_autoreset_slots": cumulative.autoreset_slots,
            "cumulative_discarded_pending_reset_slots": (self.discarded_pending_reset_slots),
            "cumulative_terminal_events": cumulative.terminal_events,
            "cumulative_terminated_events": cumulative.terminated_events,
            "cumulative_truncated_events": cumulative.truncated_events,
            "update_successful_episodes": update_episode.successful_episodes,
            "update_offtrack_episodes": update_episode.offtrack_episodes,
            "update_invalid_action_episodes": update_episode.invalid_action_episodes,
            "update_timeout_episodes": update_episode.timeout_episodes,
            "update_success_rate": update_episode.success_rate,
            "update_successful_lap_time_sum_s": (update_episode.successful_lap_time_sum_s),
            "update_mean_successful_lap_time_s": update_episode.mean_successful_lap_time_s,
            "update_episode_length_sum_steps": update_episode.episode_length_sum_steps,
            "update_mean_episode_length_steps": update_episode.mean_episode_length_steps,
            "cumulative_successful_episodes": cumulative_episode.successful_episodes,
            "cumulative_offtrack_episodes": cumulative_episode.offtrack_episodes,
            "cumulative_invalid_action_episodes": (cumulative_episode.invalid_action_episodes),
            "cumulative_timeout_episodes": cumulative_episode.timeout_episodes,
            "cumulative_success_rate": cumulative_episode.success_rate,
            "cumulative_successful_lap_time_sum_s": (cumulative_episode.successful_lap_time_sum_s),
            "cumulative_mean_successful_lap_time_s": (
                cumulative_episode.mean_successful_lap_time_s
            ),
            "cumulative_episode_length_sum_steps": (cumulative_episode.episode_length_sum_steps),
            "cumulative_mean_episode_length_steps": (cumulative_episode.mean_episode_length_steps),
            "update_reward_sum": self.rollout_reward_sum,
            "cumulative_reward_sum": self.cumulative_reward_sum,
            "update_mean_valid_reward": self.rollout_mean_valid_reward,
            "cumulative_mean_valid_reward": self.cumulative_mean_valid_reward,
            "learning_rate": optimization.learning_rate,
            "ppo_valid_samples": optimization.valid_samples,
            "ppo_samples_processed": optimization.samples_processed,
            "ppo_epochs_run": optimization.epochs_run,
            "ppo_epochs_completed": optimization.epochs_completed,
            "ppo_minibatches_processed": optimization.minibatches_processed,
            "ppo_early_stopped_for_kl": int(optimization.early_stopped_for_kl),
            "policy_loss": optimization.policy_loss,
            "value_loss": optimization.value_loss,
            "latent_entropy": optimization.latent_entropy,
            "total_loss": optimization.total_loss,
            "optimization_mean_kl": optimization.optimization_mean_kl,
            "post_epoch_kl": optimization.post_epoch_kl,
            "clip_fraction": optimization.clip_fraction,
            "mean_gradient_norm_before_clip": (optimization.mean_gradient_norm_before_clip),
            "max_gradient_norm_before_clip": optimization.max_gradient_norm_before_clip,
            "explained_variance": optimization.explained_variance,
            "compute_update_wall_seconds": self.compute_update_wall_seconds,
            "cumulative_compute_update_seconds": self.cumulative_compute_update_seconds,
            "compute_valid_transitions_per_second": (self.compute_valid_transitions_per_second),
            "cumulative_compute_valid_transitions_per_second": (
                self.cumulative_compute_valid_transitions_per_second
            ),
            "wall_elapsed_before_persistence_seconds": (
                self.wall_elapsed_before_persistence_seconds
            ),
            "torch_cuda_memory_allocated_bytes": "" if memory is None else memory.allocated_bytes,
            "torch_cuda_memory_reserved_bytes": "" if memory is None else memory.reserved_bytes,
            "torch_cuda_max_memory_allocated_bytes": (
                "" if memory is None else memory.max_allocated_bytes
            ),
        }
        if tuple(row) != TRAINING_METRICS_COLUMNS:
            raise RuntimeError("UpdateRecord CSV fields differ from TRAINING_METRICS_COLUMNS")
        return row


def _finite_nonnegative_or_signed(value: object, *, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"{name} must be a real number")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{name} must be finite")
    return result


def _owned_generator_state(value: object, *, name: str) -> Tensor:
    if (
        not isinstance(value, Tensor)
        or value.dtype is not torch.uint8
        or value.ndim != 1
        or value.device.type != "cpu"
        or value.numel() < 1
    ):
        raise TypeError(f"{name} must be a non-empty one-dimensional CPU uint8 tensor")
    return value.detach().clone()


@dataclass(frozen=True, slots=True, eq=False)
class TrainingResumeState:
    """Optimizer-continuation counters and RNG streams for a fresh environment reset.

    The caller restores model and optimizer state before invoking :func:`train_ppo`.  Simulator,
    collector, and partial per-world episode state are deliberately not restored; the resumed
    invocation calls the official environment reset and starts fresh active episodes.
    """

    starting_update: int
    counts: TransitionCounts
    discarded_pending_reset_slots: int
    episodes: EpisodeMetrics
    cumulative_reward_sum: float
    cumulative_compute_update_seconds: float
    wall_elapsed_before_persistence_seconds: float
    policy_rng_state: Tensor
    minibatch_rng_state: Tensor

    def __post_init__(self) -> None:
        _plain_positive_integer(self.starting_update, name="starting_update")
        if not isinstance(self.counts, TransitionCounts):
            raise TypeError("counts must be TransitionCounts")
        if not isinstance(self.episodes, EpisodeMetrics):
            raise TypeError("episodes must be EpisodeMetrics")
        _pending_reset_slots(
            self.counts,
            self.discarded_pending_reset_slots,
            name="resume state",
        )
        if self.episodes.episodes != self.counts.terminal_events:
            raise ValueError("resume episodes must equal resume terminal events")
        object.__setattr__(
            self,
            "cumulative_reward_sum",
            _finite_nonnegative_or_signed(
                self.cumulative_reward_sum,
                name="cumulative_reward_sum",
            ),
        )
        compute = _finite_positive(
            self.cumulative_compute_update_seconds,
            name="cumulative_compute_update_seconds",
        )
        elapsed = _finite_positive(
            self.wall_elapsed_before_persistence_seconds,
            name="wall_elapsed_before_persistence_seconds",
        )
        if elapsed < compute:
            raise ValueError(
                "wall_elapsed_before_persistence_seconds cannot be below cumulative compute time"
            )
        object.__setattr__(self, "cumulative_compute_update_seconds", compute)
        object.__setattr__(self, "wall_elapsed_before_persistence_seconds", elapsed)
        object.__setattr__(
            self,
            "policy_rng_state",
            _owned_generator_state(self.policy_rng_state, name="policy_rng_state"),
        )
        object.__setattr__(
            self,
            "minibatch_rng_state",
            _owned_generator_state(self.minibatch_rng_state, name="minibatch_rng_state"),
        )

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, TrainingResumeState):
            return NotImplemented
        return (
            self.starting_update == other.starting_update
            and self.counts == other.counts
            and self.discarded_pending_reset_slots == other.discarded_pending_reset_slots
            and self.episodes == other.episodes
            and self.cumulative_reward_sum == other.cumulative_reward_sum
            and self.cumulative_compute_update_seconds == other.cumulative_compute_update_seconds
            and self.wall_elapsed_before_persistence_seconds
            == other.wall_elapsed_before_persistence_seconds
            and torch.equal(self.policy_rng_state, other.policy_rng_state)
            and torch.equal(self.minibatch_rng_state, other.minibatch_rng_state)
        )


@dataclass(frozen=True, slots=True)
class TrainingCheckpointRequest:
    """Snapshot passed to an atomic persistence callback at configured boundaries.

    ``elapsed_seconds`` is sampled immediately before invoking the callback.  It cannot include
    the duration of persisting the checkpoint that contains it.
    """

    update_index: int
    vector_steps: int
    elapsed_seconds: float
    counts: TransitionCounts
    discarded_pending_reset_slots: int
    episodes: EpisodeMetrics
    record: UpdateRecord
    model_state_dict: Mapping[str, Any]
    optimizer_state_dict: Mapping[str, Any] | None
    policy_rng_state: Tensor
    minibatch_rng_state: Tensor
    is_scheduled: bool
    is_final: bool

    def __post_init__(self) -> None:
        if self.update_index != self.record.update_index:
            raise ValueError("checkpoint update_index differs from its update record")
        if self.vector_steps != self.record.vector_steps:
            raise ValueError("checkpoint vector_steps differs from its update record")
        if self.counts != self.record.cumulative_counts:
            raise ValueError("checkpoint counts differ from its update record")
        if self.discarded_pending_reset_slots != self.record.discarded_pending_reset_slots:
            raise ValueError("checkpoint discarded pending slots differ from its update record")
        if self.episodes != self.record.cumulative_episodes:
            raise ValueError("checkpoint episodes differ from its update record")
        object.__setattr__(
            self,
            "elapsed_seconds",
            _finite_positive(self.elapsed_seconds, name="elapsed_seconds"),
        )
        if not isinstance(self.model_state_dict, Mapping) or not self.model_state_dict:
            raise TypeError("model_state_dict must be a non-empty mapping")
        if self.optimizer_state_dict is not None and (
            not isinstance(self.optimizer_state_dict, Mapping) or not self.optimizer_state_dict
        ):
            raise TypeError("optimizer_state_dict must be a non-empty mapping or None")
        object.__setattr__(
            self,
            "policy_rng_state",
            _owned_generator_state(self.policy_rng_state, name="policy_rng_state"),
        )
        object.__setattr__(
            self,
            "minibatch_rng_state",
            _owned_generator_state(self.minibatch_rng_state, name="minibatch_rng_state"),
        )
        if type(self.is_scheduled) is not bool or type(self.is_final) is not bool:
            raise TypeError("checkpoint reason flags must be booleans")
        if not (self.is_scheduled or self.is_final):
            raise ValueError("a checkpoint request must be scheduled, final, or both")

    def to_resume_state(self) -> TrainingResumeState:
        """Return the non-environment state needed for a fresh-reset continuation."""

        return TrainingResumeState(
            starting_update=self.update_index,
            counts=self.counts,
            discarded_pending_reset_slots=self.discarded_pending_reset_slots,
            episodes=self.episodes,
            cumulative_reward_sum=self.record.cumulative_reward_sum,
            cumulative_compute_update_seconds=(self.record.cumulative_compute_update_seconds),
            wall_elapsed_before_persistence_seconds=self.elapsed_seconds,
            policy_rng_state=self.policy_rng_state,
            minibatch_rng_state=self.minibatch_rng_state,
        )

    @property
    def resume_state(self) -> TrainingResumeState:
        """Expose the complete strict continuation state for artifact persistence."""

        return self.to_resume_state()


class AtomicCheckpointCallback(Protocol):
    """Persist one checkpoint request atomically before returning to the training loop."""

    def __call__(self, request: TrainingCheckpointRequest) -> object: ...


@dataclass(frozen=True, slots=True)
class TrainingSummary:
    """Immutable result for a complete configured run or an explicit smoke prefix."""

    configured_updates: int
    starting_update: int
    completed_updates: int
    configured_budget_completed: bool
    vector_steps: int
    counts: TransitionCounts
    discarded_pending_reset_slots: int
    episodes: EpisodeMetrics
    cumulative_reward_sum: float
    compute_update_seconds: float
    compute_valid_transitions_per_second: float
    end_to_end_elapsed_seconds: float
    end_to_end_valid_transitions_per_second: float
    policy_seed: int
    minibatch_seed: int
    environment_seed: int
    records: tuple[UpdateRecord, ...]
    metrics_path: Path

    def __post_init__(self) -> None:
        _plain_positive_integer(self.configured_updates, name="configured_updates")
        if type(self.starting_update) is not int or self.starting_update < 0:
            raise ValueError("starting_update must be a non-negative integer")
        _plain_positive_integer(self.completed_updates, name="completed_updates")
        if not self.starting_update < self.completed_updates <= self.configured_updates:
            raise ValueError(
                "completed_updates must be above starting_update and within configured_updates"
            )
        if self.configured_budget_completed != (self.completed_updates == self.configured_updates):
            raise ValueError("configured_budget_completed differs from completed_updates")
        if len(self.records) != self.completed_updates - self.starting_update:
            raise ValueError(
                "records must contain exactly one record per update in this invocation"
            )
        if self.records[-1].cumulative_counts != self.counts:
            raise ValueError("summary transition counts differ from the final record")
        if self.records[-1].cumulative_episodes != self.episodes:
            raise ValueError("summary episode metrics differ from the final record")
        if self.vector_steps != self.counts.environment_step_calls:
            raise ValueError("summary vector_steps differ from transition counts")
        if self.records[-1].discarded_pending_reset_slots != self.discarded_pending_reset_slots:
            raise ValueError("summary discarded pending slots differ from the final record")
        _pending_reset_slots(
            self.counts,
            self.discarded_pending_reset_slots,
            name="training summary",
        )
        object.__setattr__(
            self,
            "cumulative_reward_sum",
            _finite_nonnegative_or_signed(
                self.cumulative_reward_sum,
                name="cumulative_reward_sum",
            ),
        )
        object.__setattr__(
            self,
            "compute_update_seconds",
            _finite_positive(self.compute_update_seconds, name="compute_update_seconds"),
        )
        object.__setattr__(
            self,
            "compute_valid_transitions_per_second",
            _finite_positive(
                self.compute_valid_transitions_per_second,
                name="compute_valid_transitions_per_second",
            ),
        )
        end_to_end = _finite_positive(
            self.end_to_end_elapsed_seconds,
            name="end_to_end_elapsed_seconds",
        )
        if end_to_end < self.compute_update_seconds:
            raise ValueError("end-to-end elapsed time cannot be below compute/update time")
        object.__setattr__(self, "end_to_end_elapsed_seconds", end_to_end)
        object.__setattr__(
            self,
            "end_to_end_valid_transitions_per_second",
            _finite_positive(
                self.end_to_end_valid_transitions_per_second,
                name="end_to_end_valid_transitions_per_second",
            ),
        )
        if not isinstance(self.metrics_path, Path):
            raise TypeError("metrics_path must be pathlib.Path")


def _csv_integer(row: Mapping[str | None, str | None], field: str) -> int:
    value = row.get(field)
    if value is None or not value or value.strip() != value:
        raise ValueError(f"resume metrics CSV field {field} is not a canonical integer")
    try:
        parsed = int(value)
    except ValueError as error:
        raise ValueError(f"resume metrics CSV field {field} is not an integer") from error
    if str(parsed) != value:
        raise ValueError(f"resume metrics CSV field {field} is not a canonical integer")
    return parsed


def _csv_float(row: Mapping[str | None, str | None], field: str) -> float:
    value = row.get(field)
    if value is None or not value or value.strip() != value:
        raise ValueError(f"resume metrics CSV field {field} is not a finite float")
    try:
        parsed = float(value)
    except ValueError as error:
        raise ValueError(f"resume metrics CSV field {field} is not a float") from error
    if not math.isfinite(parsed):
        raise ValueError(f"resume metrics CSV field {field} is not a finite float")
    return parsed


class FixedColumnCsvWriter:
    """Create one append-only metrics CSV and durably flush it at update cadence."""

    def __init__(
        self,
        path: str | Path,
        *,
        flush_interval_updates: int,
        resume_state: TrainingResumeState | None = None,
    ) -> None:
        self.path = Path(path)
        self.flush_interval_updates = _plain_positive_integer(
            flush_interval_updates,
            name="flush_interval_updates",
        )
        self.path.parent.mkdir(parents=True, exist_ok=True)
        if resume_state is not None and not isinstance(resume_state, TrainingResumeState):
            raise TypeError("resume_state must be TrainingResumeState or None")
        if resume_state is None:
            self._file = self.path.open("x", encoding="utf-8", newline="")
        else:
            if self.path.is_symlink():
                raise ValueError("refusing to resume through a symbolic-link metrics path")
            self._file = self.path.open("r+", encoding="utf-8", newline="")
            try:
                self._validate_resume_file(resume_state)
            except BaseException:
                self._file.close()
                raise
        self._writer = csv.DictWriter(
            self._file,
            fieldnames=TRAINING_METRICS_COLUMNS,
            extrasaction="raise",
            lineterminator="\n",
        )
        if resume_state is None:
            self._writer.writeheader()
            self._dirty = True
        else:
            self._file.seek(0, os.SEEK_END)
            self._dirty = False
        self._closed = False

    def _validate_resume_file(self, state: TrainingResumeState) -> None:
        self._file.seek(0)
        content = self._file.read()
        if not content or not content.endswith("\n"):
            raise ValueError("resume metrics CSV must be non-empty and newline-terminated")
        self._file.seek(0)
        reader = csv.DictReader(self._file)
        if tuple(reader.fieldnames or ()) != TRAINING_METRICS_COLUMNS:
            raise ValueError("resume metrics CSV header differs from the fixed schema")
        rows = list(reader)
        if not rows:
            raise ValueError("resume metrics CSV must contain at least one update row")
        if any(None in row or any(value is None for value in row.values()) for row in rows):
            raise ValueError("resume metrics CSV contains a malformed row")
        indices = tuple(_csv_integer(row, "update_index") for row in rows)
        if indices != tuple(range(1, state.starting_update + 1)):
            raise ValueError("resume metrics CSV update rows are not a complete ordered prefix")
        last = rows[-1]
        expected_integers = {
            "update_index": state.starting_update,
            "vector_steps": state.counts.environment_step_calls,
            "cumulative_raw_transitions": state.counts.raw_transitions,
            "cumulative_valid_transitions": state.counts.valid_transitions,
            "cumulative_dummy_reset_transitions": state.counts.dummy_reset_transitions,
            "cumulative_autoreset_slots": state.counts.autoreset_slots,
            "cumulative_discarded_pending_reset_slots": (state.discarded_pending_reset_slots),
            "cumulative_terminal_events": state.counts.terminal_events,
            "cumulative_terminated_events": state.counts.terminated_events,
            "cumulative_truncated_events": state.counts.truncated_events,
            "cumulative_successful_episodes": state.episodes.successful_episodes,
            "cumulative_offtrack_episodes": state.episodes.offtrack_episodes,
            "cumulative_invalid_action_episodes": state.episodes.invalid_action_episodes,
            "cumulative_timeout_episodes": state.episodes.timeout_episodes,
            "cumulative_episode_length_sum_steps": state.episodes.episode_length_sum_steps,
        }
        for field, expected in expected_integers.items():
            if _csv_integer(last, field) != expected:
                raise ValueError(f"resume metrics CSV last-row {field} differs from resume state")
        expected_floats = {
            "cumulative_successful_lap_time_sum_s": (state.episodes.successful_lap_time_sum_s),
            "cumulative_reward_sum": state.cumulative_reward_sum,
            "cumulative_compute_update_seconds": (state.cumulative_compute_update_seconds),
        }
        for field, expected in expected_floats.items():
            if _csv_float(last, field) != expected:
                raise ValueError(f"resume metrics CSV last-row {field} differs from resume state")

    def write(self, record: UpdateRecord, *, final: bool = False) -> None:
        if self._closed:
            raise RuntimeError("cannot write to a closed metrics CSV")
        if not isinstance(record, UpdateRecord):
            raise TypeError("record must be an UpdateRecord")
        if type(final) is not bool:
            raise TypeError("final must be a boolean")
        self._writer.writerow(record.to_csv_row())
        self._dirty = True
        if final or record.update_index % self.flush_interval_updates == 0:
            self.flush()

    def flush(self) -> None:
        if self._closed:
            raise RuntimeError("cannot flush a closed metrics CSV")
        if not self._dirty:
            return
        self._file.flush()
        os.fsync(self._file.fileno())
        self._dirty = False

    def close(self) -> None:
        if self._closed:
            return
        try:
            self.flush()
        finally:
            self._file.close()
            self._closed = True

    def __enter__(self) -> FixedColumnCsvWriter:
        return self

    def __exit__(
        self,
        exception_type: type[BaseException] | None,
        exception: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        self.close()


class _SummaryWriter(Protocol):
    def add_scalar(self, tag: str, scalar_value: float | int, global_step: int) -> None: ...

    def flush(self) -> None: ...

    def close(self) -> None: ...


def _create_summary_writer(path: Path) -> _SummaryWriter:
    try:
        from torch.utils.tensorboard import SummaryWriter
    except ImportError as error:  # pragma: no cover - depends on the selected Pixi environment.
        raise RuntimeError(
            "TensorBoard logging is enabled, but torch.utils.tensorboard is unavailable"
        ) from error
    return SummaryWriter(log_dir=os.fspath(path))


def _log_tensorboard(writer: _SummaryWriter, record: UpdateRecord) -> None:
    row = record.to_csv_row()
    for name, value in row.items():
        if name in {"update_index", "vector_steps"} or value == "":
            continue
        writer.add_scalar(name, value, record.vector_steps)


def _torch_cuda_memory(device: torch.device) -> TorchCudaMemoryMetrics:
    return TorchCudaMemoryMetrics(
        allocated_bytes=torch.cuda.memory_allocated(device),
        reserved_bytes=torch.cuda.memory_reserved(device),
        max_allocated_bytes=torch.cuda.max_memory_allocated(device),
    )


def _validate_update_limit(
    config: PpoTrainingConfig,
    update_limit: int | None,
    *,
    starting_update: int,
) -> int:
    if update_limit is None:
        limit = config.update_count
    else:
        limit = _plain_positive_integer(update_limit, name="update_limit")
    if not starting_update < limit <= config.update_count:
        raise ValueError(f"update_limit cannot exceed the configured {config.update_count} updates")
    return limit


def _validate_resume_state(
    state: TrainingResumeState | None,
    *,
    config: PpoTrainingConfig,
) -> int:
    if state is None:
        return 0
    if not isinstance(state, TrainingResumeState):
        raise TypeError("resume_state must be TrainingResumeState or None")
    if state.starting_update >= config.update_count:
        raise ValueError("resume starting_update must leave at least one configured update")
    expected_vector_steps = state.starting_update * config.rollout.steps_per_update
    expected_raw = state.starting_update * config.nominal_world_slots_per_update
    if state.counts.num_envs != config.environment.num_envs:
        raise ValueError("resume transition width differs from config.environment.num_envs")
    if state.counts.environment_step_calls != expected_vector_steps:
        raise ValueError("resume vector-step count differs from starting_update")
    if state.counts.raw_transitions != expected_raw:
        raise ValueError("resume raw transition count differs from starting_update")
    _pending_reset_slots(
        state.counts,
        state.discarded_pending_reset_slots,
        name="resume state",
    )
    return state.starting_update


def _validate_training_stack(
    collector: TorchRolloutCollector,
    updater: PpoUpdater,
    config: PpoTrainingConfig,
) -> torch.device:
    if not isinstance(config, PpoTrainingConfig):
        raise TypeError("config must be a PpoTrainingConfig")
    required_collector = ("policy", "rollout_steps", "num_envs", "initialize", "collect")
    if any(not hasattr(collector, name) for name in required_collector):
        raise TypeError("collector does not provide the TorchRolloutCollector interface")
    required_updater = ("policy", "config", "optimizer", "update")
    if any(not hasattr(updater, name) for name in required_updater):
        raise TypeError("updater does not provide the PpoUpdater interface")
    if collector.policy is not updater.policy:
        raise ValueError("collector and updater must share the exact policy instance")
    if updater.config != config.ppo:
        raise ValueError("updater config differs from config.ppo")
    if collector.rollout_steps != config.rollout.steps_per_update:
        raise ValueError("collector rollout_steps differs from the configured rollout")
    if collector.num_envs != config.environment.num_envs:
        raise ValueError("collector num_envs differs from the formal configured vector width")
    if getattr(updater.policy, "policy_seed", None) != config.ppo.policy_seed:
        raise ValueError("policy initialization seed differs from config.ppo.policy_seed")
    device = getattr(updater.policy, "device", None)
    if not isinstance(device, torch.device) or device.type != "cuda":
        raise ValueError("the training policy must be on CUDA")
    return device


def _learning_rate(config: PpoTrainingConfig, update_index: int) -> float:
    if not config.ppo.anneal_learning_rate:
        return config.ppo.learning_rate
    fraction = 1.0 - (update_index - 1) / config.update_count
    return config.ppo.learning_rate * fraction


class _GpuEpisodeTracker:
    """Track active public episode identity and valid-only length across rollout boundaries."""

    def __init__(self, *, num_envs: int, device: torch.device) -> None:
        self.num_envs = _plain_positive_integer(num_envs, name="num_envs")
        self.device = device
        self.initialized = torch.zeros(self.num_envs, dtype=torch.bool, device=device)
        self.awaiting_reset = torch.zeros_like(self.initialized)
        self.length_steps = torch.zeros(self.num_envs, dtype=torch.int64, device=device)
        # Torch CUDA has limited uint32 elementwise kernels, so identities are widened internally.
        self.episode_seed = torch.zeros(self.num_envs, dtype=torch.int64, device=device)
        self.controller_seed = torch.zeros_like(self.episode_seed)
        self.track_id = torch.zeros_like(self.episode_seed)

    def consume(self, rollout: CollectedRollout) -> tuple[Tensor, Tensor]:
        """Update tracker state and return device scalars for length sum and violations."""

        rollout_steps, num_envs = rollout.shape
        if num_envs != self.num_envs:
            raise ValueError("rollout width differs from the episode tracker")
        time_world = (rollout_steps, num_envs)
        for name in (
            "valid_transition",
            "reset_only",
            "terminated",
            "truncated",
        ):
            value = getattr(rollout, name)
            if (
                value.shape != time_world
                or value.dtype is not torch.bool
                or value.device != self.device
            ):
                raise ValueError(f"rollout.{name} differs from the episode tracker shape/device")
        for name in ("episode_seed", "controller_seed", "track_id"):
            value = getattr(rollout, name)
            if (
                value.shape != time_world
                or value.dtype is not torch.uint32
                or value.device != self.device
            ):
                raise ValueError(f"rollout.{name} must be time-major CUDA uint32 identity")

        violations = (rollout.initial_pending_reset != self.awaiting_reset).sum(dtype=torch.int64)
        episode_length_sum = torch.zeros((), dtype=torch.int64, device=self.device)
        stored_identities = (self.episode_seed, self.controller_seed, self.track_id)
        rollout_identities = (rollout.episode_seed, rollout.controller_seed, rollout.track_id)

        for step in range(rollout_steps):
            reset = rollout.reset_only[step]
            valid = rollout.valid_transition[step]
            terminal = rollout.terminated[step] | rollout.truncated[step]
            violations += (reset != self.awaiting_reset).sum(dtype=torch.int64)
            violations += (valid != torch.logical_not(reset)).sum(dtype=torch.int64)

            first_active_observation = valid & torch.logical_not(self.initialized)
            load_identity = reset | first_active_observation
            compare_identity = valid & self.initialized
            for stored, time_major in zip(
                stored_identities,
                rollout_identities,
                strict=True,
            ):
                current = time_major[step].to(dtype=torch.int64)
                violations += (compare_identity & (current != stored)).sum(dtype=torch.int64)
                stored.copy_(torch.where(load_identity, current, stored))

            violations += (valid & self.awaiting_reset).sum(dtype=torch.int64)
            self.initialized.logical_or_(load_identity)
            self.length_steps.masked_fill_(reset, 0)
            self.awaiting_reset.masked_fill_(reset, False)
            self.length_steps.add_(valid.to(dtype=torch.int64))
            completed = valid & terminal
            episode_length_sum += torch.where(
                completed,
                self.length_steps,
                0,
            ).sum(dtype=torch.int64)
            self.awaiting_reset.logical_or_(completed)

        violations += (rollout.final_state.pending_reset != self.awaiting_reset).sum(
            dtype=torch.int64
        )
        return episode_length_sum, violations


def _extract_public_metrics(
    rollout: CollectedRollout,
    tracker: _GpuEpisodeTracker,
) -> tuple[EpisodeMetrics, float]:
    """Validate public terminal fields and transfer one compact summary to the host."""

    if not isinstance(rollout, CollectedRollout):
        raise TypeError("collector must return a CollectedRollout")
    terminal = torch.logical_or(rollout.terminated, rollout.truncated)
    reason = rollout.termination_reason
    lap_completed = rollout.lap_completed
    lap_time = rollout.lap_time_s
    valid = rollout.valid_transition
    success = reason == int(RaceTermination.SUCCESS)
    offtrack = reason == int(RaceTermination.OFF_TRACK)
    invalid_action = reason == int(RaceTermination.INVALID_ACTION)
    timeout = reason == int(RaceTermination.TIMEOUT)
    recognized = success | offtrack | invalid_action | timeout
    successful_lap_times = torch.where(success, lap_time, 0.0)
    episode_length_sum, episode_tracker_violations = tracker.consume(rollout)

    summary = torch.stack(
        (
            terminal.sum(dtype=torch.int64).to(dtype=torch.float64),
            success.sum(dtype=torch.int64).to(dtype=torch.float64),
            offtrack.sum(dtype=torch.int64).to(dtype=torch.float64),
            invalid_action.sum(dtype=torch.int64).to(dtype=torch.float64),
            timeout.sum(dtype=torch.int64).to(dtype=torch.float64),
            successful_lap_times.sum(dtype=torch.float64),
            episode_length_sum.to(dtype=torch.float64),
            torch.where(valid, rollout.rewards, 0.0).sum(dtype=torch.float64),
            episode_tracker_violations.to(dtype=torch.float64),
            (terminal & torch.logical_not(recognized))
            .sum(dtype=torch.int64)
            .to(dtype=torch.float64),
            (torch.logical_not(terminal) & (reason != int(RaceTermination.NONE)))
            .sum(dtype=torch.int64)
            .to(dtype=torch.float64),
            torch.logical_xor(success, lap_completed)
            .sum(dtype=torch.int64)
            .to(dtype=torch.float64),
            (success & torch.logical_not(rollout.terminated))
            .sum(dtype=torch.int64)
            .to(dtype=torch.float64),
            (success & (lap_time <= 0.0)).sum(dtype=torch.int64).to(dtype=torch.float64),
            (offtrack & torch.logical_not(rollout.terminated))
            .sum(dtype=torch.int64)
            .to(dtype=torch.float64),
            (invalid_action & torch.logical_not(rollout.terminated))
            .sum(dtype=torch.int64)
            .to(dtype=torch.float64),
            (timeout & torch.logical_not(rollout.truncated))
            .sum(dtype=torch.int64)
            .to(dtype=torch.float64),
        )
    ).to(device="cpu")
    (
        episode_count,
        success_count,
        offtrack_count,
        invalid_action_count,
        timeout_count,
        successful_lap_time_sum,
        episode_length_sum_steps,
        valid_reward_sum,
        tracker_violation_count,
        unknown_terminal_count,
        reason_without_terminal_count,
        success_lap_mismatch_count,
        success_without_termination_count,
        nonpositive_success_lap_time_count,
        offtrack_without_termination_count,
        invalid_action_without_termination_count,
        timeout_without_truncation_count,
    ) = summary.tolist()

    if tracker_violation_count:
        raise ValueError(
            "public episode identity/length state changed outside an official reset-only row"
        )
    if unknown_terminal_count or reason_without_terminal_count:
        raise ValueError("public termination_reason does not match terminal flags")
    if success_lap_mismatch_count or success_without_termination_count:
        raise ValueError("public SUCCESS reason, lap_completed, and termination flags differ")
    if nonpositive_success_lap_time_count:
        raise ValueError("successful public lap times must be positive")
    if offtrack_without_termination_count or invalid_action_without_termination_count:
        raise ValueError("OFF_TRACK and INVALID_ACTION must be terminated events")
    if timeout_without_truncation_count:
        raise ValueError("TIMEOUT must be a truncated event")

    metrics = EpisodeMetrics(
        episodes=int(episode_count),
        successful_episodes=int(success_count),
        offtrack_episodes=int(offtrack_count),
        invalid_action_episodes=int(invalid_action_count),
        timeout_episodes=int(timeout_count),
        successful_lap_time_sum_s=float(successful_lap_time_sum),
        episode_length_sum_steps=int(episode_length_sum_steps),
    )
    if metrics.episodes != rollout.counts.terminal_events:
        raise ValueError("public episode metrics differ from rollout transition counts")
    return metrics, _finite_nonnegative_or_signed(valid_reward_sum, name="valid_reward_sum")


def _validate_rollout_budget(
    rollout: CollectedRollout,
    *,
    config: PpoTrainingConfig,
) -> None:
    counts = rollout.counts
    if counts.num_envs != config.environment.num_envs:
        raise ValueError("rollout transition width differs from the configured num_envs")
    if counts.environment_step_calls != config.rollout.steps_per_update:
        raise ValueError("rollout environment calls differ from steps_per_update")
    if counts.raw_transitions != config.nominal_world_slots_per_update:
        raise ValueError("rollout raw transition count differs from the configured budget")
    if counts.valid_transitions < config.ppo.num_minibatches:
        raise ValueError("rollout has fewer valid transitions than PPO minibatches")


def _checkpoint_request(
    *,
    record: UpdateRecord,
    updater: PpoUpdater,
    policy_generator: torch.Generator,
    minibatch_generator: torch.Generator,
    save_optimizer_state: bool,
    is_scheduled: bool,
    is_final: bool,
) -> TrainingCheckpointRequest:
    # Deep copies prevent a callback that queues serialization from observing later optimizer or
    # policy mutations.  The callback remains responsible for atomic durable persistence.
    model_state = copy.deepcopy(updater.policy.state_dict())
    optimizer_state = (
        copy.deepcopy(updater.optimizer.state_dict()) if save_optimizer_state else None
    )
    return TrainingCheckpointRequest(
        update_index=record.update_index,
        vector_steps=record.vector_steps,
        elapsed_seconds=record.wall_elapsed_before_persistence_seconds,
        counts=record.cumulative_counts,
        discarded_pending_reset_slots=record.discarded_pending_reset_slots,
        episodes=record.cumulative_episodes,
        record=record,
        model_state_dict=model_state,
        optimizer_state_dict=optimizer_state,
        policy_rng_state=policy_generator.get_state().clone(),
        minibatch_rng_state=minibatch_generator.get_state().clone(),
        is_scheduled=is_scheduled,
        is_final=is_final,
    )


def train_ppo(
    collector: TorchRolloutCollector,
    updater: PpoUpdater,
    config: PpoTrainingConfig,
    *,
    run_directory: str | Path,
    update_limit: int | None = None,
    resume_state: TrainingResumeState | None = None,
    checkpoint_callback: AtomicCheckpointCallback | None = None,
    clock: Callable[[], float] = time.perf_counter,
    summary_writer_factory: Callable[[Path], _SummaryWriter] = _create_summary_writer,
    memory_sampler: Callable[[torch.device], TorchCudaMemoryMetrics] = _torch_cuda_memory,
) -> TrainingSummary:
    """Run a seeded training budget against one long-lived official vector environment.

    ``update_limit`` is an absolute configured update index, including during resume.  Omitting it
    enforces the exact full budget.  Resume restores optimizer-continuation counters and the two
    Torch RNG streams, then resets a fresh environment; no in-flight simulator state is claimed.
    Checkpoint callbacks run at configured absolute indices and once at the final requested update.
    """

    device = _validate_training_stack(collector, updater, config)
    starting_update = _validate_resume_state(resume_state, config=config)
    target_updates = _validate_update_limit(
        config,
        update_limit,
        starting_update=starting_update,
    )
    if checkpoint_callback is not None and not callable(checkpoint_callback):
        raise TypeError("checkpoint_callback must be callable or None")
    if not callable(clock):
        raise TypeError("clock must be callable")
    if not callable(summary_writer_factory):
        raise TypeError("summary_writer_factory must be callable")
    if not callable(memory_sampler):
        raise TypeError("memory_sampler must be callable")

    output = Path(run_directory)
    output.mkdir(parents=True, exist_ok=True)
    metrics_path = output / "metrics.csv"
    policy_generator = torch.Generator(device=device).manual_seed(config.ppo.policy_seed)
    minibatch_generator = torch.Generator(device=device).manual_seed(config.ppo.minibatch_seed)
    if policy_generator is minibatch_generator:
        raise RuntimeError("policy and minibatch generators must be distinct objects")

    if resume_state is not None:
        policy_generator.set_state(resume_state.policy_rng_state)
        minibatch_generator.set_state(resume_state.minibatch_rng_state)
        cumulative_counts = resume_state.counts
        outstanding_pending_reset_slots = _pending_reset_slots(
            cumulative_counts,
            resume_state.discarded_pending_reset_slots,
            name="resume state",
        )
        cumulative_discarded_pending_reset_slots = (
            resume_state.discarded_pending_reset_slots + outstanding_pending_reset_slots
        )
        cumulative_episodes = resume_state.episodes
        cumulative_reward_sum = resume_state.cumulative_reward_sum
        cumulative_compute_seconds = resume_state.cumulative_compute_update_seconds
        prior_wall_before_persistence_seconds = resume_state.wall_elapsed_before_persistence_seconds
    else:
        cumulative_counts = _zero_transition_counts(config.environment.num_envs)
        cumulative_discarded_pending_reset_slots = 0
        cumulative_episodes = _zero_episode_metrics()
        cumulative_reward_sum = 0.0
        cumulative_compute_seconds = 0.0
        prior_wall_before_persistence_seconds = 0.0

    records: list[UpdateRecord] = []
    starting_valid_transitions = cumulative_counts.valid_transitions
    invocation_compute_seconds = 0.0
    invocation_start = _finite_nonnegative(clock(), name="clock()")
    csv_writer: FixedColumnCsvWriter | None = None
    tensorboard: _SummaryWriter | None = None
    body_error: BaseException | None = None
    try:
        csv_writer = FixedColumnCsvWriter(
            metrics_path,
            flush_interval_updates=config.logging.csv_flush_interval_updates,
            resume_state=resume_state,
        )
        tensorboard = summary_writer_factory(output) if config.logging.tensorboard_enabled else None
        state: CollectorState = collector.initialize(seed=config.environment.environment_seed)
        episode_tracker = _GpuEpisodeTracker(
            num_envs=config.environment.num_envs,
            device=device,
        )
        for update_index in range(starting_update + 1, target_updates + 1):
            compute_start = _finite_nonnegative(clock(), name="clock()")
            rollout = collector.collect(state, generator=policy_generator)
            _validate_rollout_budget(rollout, config=config)
            state = rollout.final_state
            rollout_episodes, rollout_reward_sum = _extract_public_metrics(
                rollout,
                episode_tracker,
            )
            learning_rate = _learning_rate(config, update_index)
            optimization = updater.update(
                rollout,
                learning_rate=learning_rate,
                minibatch_generator=minibatch_generator,
            )
            cumulative_counts = cumulative_counts + rollout.counts
            cumulative_episodes = cumulative_episodes + rollout_episodes
            cumulative_reward_sum += rollout_reward_sum

            compute_end = _finite_nonnegative(clock(), name="clock()")
            compute_update_wall_seconds = compute_end - compute_start
            if compute_update_wall_seconds <= 0.0:
                raise RuntimeError("clock must advance across every compute/update interval")
            cumulative_compute_seconds += compute_update_wall_seconds
            invocation_compute_seconds += compute_update_wall_seconds
            wall_elapsed_before_persistence = prior_wall_before_persistence_seconds + (
                compute_end - invocation_start
            )
            sample_memory = (
                update_index % config.logging.memory_sample_interval_updates == 0
                or update_index == target_updates
            )
            torch_cuda_memory = memory_sampler(device) if sample_memory else None
            if torch_cuda_memory is not None and not isinstance(
                torch_cuda_memory,
                TorchCudaMemoryMetrics,
            ):
                raise TypeError("memory_sampler must return TorchCudaMemoryMetrics")

            record = UpdateRecord(
                update_index=update_index,
                vector_steps=cumulative_counts.environment_step_calls,
                rollout_counts=rollout.counts,
                cumulative_counts=cumulative_counts,
                discarded_pending_reset_slots=cumulative_discarded_pending_reset_slots,
                rollout_episodes=rollout_episodes,
                cumulative_episodes=cumulative_episodes,
                rollout_reward_sum=rollout_reward_sum,
                cumulative_reward_sum=cumulative_reward_sum,
                optimization=optimization,
                compute_update_wall_seconds=compute_update_wall_seconds,
                cumulative_compute_update_seconds=cumulative_compute_seconds,
                compute_valid_transitions_per_second=(
                    rollout.counts.valid_transitions / compute_update_wall_seconds
                ),
                cumulative_compute_valid_transitions_per_second=(
                    cumulative_counts.valid_transitions / cumulative_compute_seconds
                ),
                wall_elapsed_before_persistence_seconds=(wall_elapsed_before_persistence),
                torch_cuda_memory=torch_cuda_memory,
            )
            records.append(record)

            is_final = update_index == target_updates
            # CSV is the complete audit trail, so every update receives a row.  Its durability
            # cadence is independent from the lower-frequency presentation logger.
            csv_writer.write(record, final=is_final)
            should_log = update_index % config.logging.log_interval_updates == 0 or is_final
            if should_log and tensorboard is not None:
                _log_tensorboard(tensorboard, record)
                if update_index % config.logging.csv_flush_interval_updates == 0 or is_final:
                    tensorboard.flush()

            is_scheduled_checkpoint = update_index % config.checkpoint.interval_updates == 0
            if checkpoint_callback is not None and (is_scheduled_checkpoint or is_final):
                checkpoint_callback(
                    _checkpoint_request(
                        record=record,
                        updater=updater,
                        policy_generator=policy_generator,
                        minibatch_generator=minibatch_generator,
                        save_optimizer_state=config.checkpoint.save_optimizer_state,
                        is_scheduled=is_scheduled_checkpoint,
                        is_final=is_final,
                    )
                )
    except BaseException as error:
        body_error = error

    cleanup_errors: list[BaseException] = []
    if tensorboard is not None:
        try:
            tensorboard.close()
        except BaseException as error:
            cleanup_errors.append(error)
    if csv_writer is not None:
        try:
            csv_writer.close()
        except BaseException as error:
            cleanup_errors.append(error)
    if body_error is not None:
        if cleanup_errors:
            raise BaseExceptionGroup(
                "training failed and cleanup also failed",
                [body_error, *cleanup_errors],
            )
        raise body_error.with_traceback(body_error.__traceback__)
    if cleanup_errors:
        if len(cleanup_errors) == 1:
            error = cleanup_errors[0]
            raise error.with_traceback(error.__traceback__)
        raise BaseExceptionGroup("multiple training logger cleanup failures", cleanup_errors)

    durable_end = _finite_nonnegative(clock(), name="clock()")
    invocation_end_to_end_seconds = durable_end - invocation_start
    if invocation_end_to_end_seconds <= 0.0:
        raise RuntimeError("clock must advance across the durable training invocation")
    end_to_end_elapsed_seconds = invocation_end_to_end_seconds
    invocation_valid_transitions = cumulative_counts.valid_transitions - starting_valid_transitions

    vector_steps = cumulative_counts.environment_step_calls
    expected_vector_steps = target_updates * config.rollout.steps_per_update
    expected_raw_transitions = target_updates * config.nominal_world_slots_per_update
    if vector_steps != expected_vector_steps:
        raise RuntimeError("training did not consume the exact requested vector-step prefix")
    if cumulative_counts.raw_transitions != expected_raw_transitions:
        raise RuntimeError("training did not consume the exact requested world-slot prefix")
    configured_budget_completed = target_updates == config.update_count
    if configured_budget_completed:
        if vector_steps != config.rollout.total_vector_steps:
            raise RuntimeError("full training did not consume total_vector_steps")
        if cumulative_counts.raw_transitions != config.world_step_slot_budget:
            raise RuntimeError("full training did not consume world_step_slot_budget")

    return TrainingSummary(
        configured_updates=config.update_count,
        starting_update=starting_update,
        completed_updates=target_updates,
        configured_budget_completed=configured_budget_completed,
        vector_steps=vector_steps,
        counts=cumulative_counts,
        discarded_pending_reset_slots=cumulative_discarded_pending_reset_slots,
        episodes=cumulative_episodes,
        cumulative_reward_sum=cumulative_reward_sum,
        compute_update_seconds=invocation_compute_seconds,
        compute_valid_transitions_per_second=(
            invocation_valid_transitions / invocation_compute_seconds
        ),
        end_to_end_elapsed_seconds=end_to_end_elapsed_seconds,
        end_to_end_valid_transitions_per_second=(
            invocation_valid_transitions / end_to_end_elapsed_seconds
        ),
        policy_seed=config.ppo.policy_seed,
        minibatch_seed=config.ppo.minibatch_seed,
        environment_seed=config.environment.environment_seed,
        records=tuple(records),
        metrics_path=metrics_path,
    )


__all__ = [
    "TRAINING_METRICS_COLUMNS",
    "AtomicCheckpointCallback",
    "EpisodeMetrics",
    "FixedColumnCsvWriter",
    "TorchCudaMemoryMetrics",
    "TrainingCheckpointRequest",
    "TrainingResumeState",
    "TrainingSummary",
    "UpdateRecord",
    "train_ppo",
]
