"""NEXT_STEP-aware rollout accounting and NumPy GAE reference equations.

This module contains no simulator or PyTorch dependency.  It defines the exact transition masks
that a batched PPO implementation must use with Gymnasium ``NEXT_STEP`` autoreset semantics and a
small NumPy reference for tests against later device implementations.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, fields
from numbers import Real
from typing import Any

import numpy as np
from numpy.typing import NDArray

BoolArray = NDArray[np.bool_]
FloatArray = NDArray[np.float64]
IndexArray = NDArray[np.int64]


def _readonly_bool(value: Any, *, name: str, ndim: int) -> BoolArray:
    array = np.asarray(value)
    if array.dtype != np.dtype(np.bool_):
        raise TypeError(f"{name} must have boolean dtype")
    if array.ndim != ndim:
        raise ValueError(f"{name} must be {ndim}-dimensional")
    if any(size < 1 for size in array.shape):
        raise ValueError(f"{name} dimensions must be non-empty")
    result = np.array(array, dtype=np.bool_, copy=True)
    result.setflags(write=False)
    return result


def _readonly_float(value: Any, *, name: str, ndim: int) -> FloatArray:
    array = np.asarray(value)
    if array.dtype.kind != "f":
        raise TypeError(f"{name} must have a floating dtype")
    if array.ndim != ndim:
        raise ValueError(f"{name} must be {ndim}-dimensional")
    if any(size < 1 for size in array.shape):
        raise ValueError(f"{name} dimensions must be non-empty")
    result = np.array(array, dtype=np.float64, copy=True)
    if not np.isfinite(result).all():
        raise ValueError(f"{name} must contain only finite values")
    result.setflags(write=False)
    return result


def _readonly_copy(value: NDArray[Any]) -> NDArray[Any]:
    result = np.array(value, copy=True)
    result.setflags(write=False)
    return result


def _probability(value: Real, *, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, Real):
        raise TypeError(f"{name} must be a real number")
    result = float(value)
    if not math.isfinite(result) or not 0.0 <= result <= 1.0:
        raise ValueError(f"{name} must be finite and in [0, 1]")
    return result


@dataclass(frozen=True, slots=True)
class StepTransitionMasks:
    """Masks for one vector-environment call and the carry into the next call."""

    valid_transition: BoolArray
    reset_only: BoolArray
    terminated: BoolArray
    truncated: BoolArray
    terminal_event: BoolArray
    next_pending_reset: BoolArray


@dataclass(frozen=True, slots=True)
class TransitionCounts:
    """Exact cumulative transition accounting for one fixed-width vector environment."""

    num_envs: int
    environment_step_calls: int
    raw_transitions: int
    valid_transitions: int
    dummy_reset_transitions: int
    autoreset_slots: int
    terminal_events: int
    terminated_events: int
    truncated_events: int

    def __post_init__(self) -> None:
        if type(self.num_envs) is not int or self.num_envs < 1:
            raise ValueError("num_envs must be a positive integer")
        for field in fields(self):
            if field.name == "num_envs":
                continue
            value = getattr(self, field.name)
            if type(value) is not int or value < 0:
                raise ValueError(f"{field.name} must be a non-negative integer")
        if self.raw_transitions != self.num_envs * self.environment_step_calls:
            raise ValueError("raw_transitions must equal num_envs * environment_step_calls")
        if self.raw_transitions != self.valid_transitions + self.dummy_reset_transitions:
            raise ValueError(
                "raw_transitions must equal valid_transitions + dummy_reset_transitions"
            )
        if self.autoreset_slots != self.dummy_reset_transitions:
            raise ValueError("autoreset_slots must equal dummy_reset_transitions")
        if self.terminal_events != self.terminated_events + self.truncated_events:
            raise ValueError("terminal_events must equal terminated_events + truncated_events")
        if self.terminal_events > self.valid_transitions:
            raise ValueError("terminal_events cannot exceed valid_transitions")

    def __add__(self, other: object) -> TransitionCounts:
        """Combine adjacent rollouts that use the same vector width."""

        if not isinstance(other, TransitionCounts):
            return NotImplemented
        if self.num_envs != other.num_envs:
            raise ValueError("cannot combine transition counts with different num_envs")
        return TransitionCounts(
            num_envs=self.num_envs,
            environment_step_calls=self.environment_step_calls + other.environment_step_calls,
            raw_transitions=self.raw_transitions + other.raw_transitions,
            valid_transitions=self.valid_transitions + other.valid_transitions,
            dummy_reset_transitions=self.dummy_reset_transitions + other.dummy_reset_transitions,
            autoreset_slots=self.autoreset_slots + other.autoreset_slots,
            terminal_events=self.terminal_events + other.terminal_events,
            terminated_events=self.terminated_events + other.terminated_events,
            truncated_events=self.truncated_events + other.truncated_events,
        )


@dataclass(frozen=True, slots=True)
class RolloutTransitionMasks:
    """Time-major masks derived from one explicit pending-reset carry."""

    initial_pending_reset: BoolArray
    valid_transition: BoolArray
    reset_only: BoolArray
    terminated: BoolArray
    truncated: BoolArray
    terminal_event: BoolArray
    final_pending_reset: BoolArray
    counts: TransitionCounts

    def __post_init__(self) -> None:
        if not isinstance(self.counts, TransitionCounts):
            raise TypeError("counts must be TransitionCounts")
        initial_pending_count = int(np.count_nonzero(self.initial_pending_reset))
        final_pending_count = int(np.count_nonzero(self.final_pending_reset))
        if (
            self.counts.terminal_events - self.counts.autoreset_slots
            != final_pending_count - initial_pending_count
        ):
            raise ValueError(
                "terminal_events - autoreset_slots must equal the pending-reset carry change"
            )

    @property
    def shape(self) -> tuple[int, int]:
        """Return ``(rollout_steps, num_envs)``."""

        return self.valid_transition.shape


@dataclass(frozen=True, slots=True)
class GaeResult:
    """Reference temporal differences, advantages, and value targets."""

    temporal_difference: FloatArray
    advantages: FloatArray
    returns: FloatArray


def build_step_transition_masks(
    pending_reset: Any,
    terminated: Any,
    truncated: Any,
) -> StepTransitionMasks:
    """Derive the learning mask and next carry for one NEXT_STEP environment call.

    A pending world performs only its deferred reset.  Gymnasium requires that row to return false
    termination flags, so a reset-only row carrying a terminal flag is rejected rather than hidden.
    """

    pending = _readonly_bool(pending_reset, name="pending_reset", ndim=1)
    term = _readonly_bool(terminated, name="terminated", ndim=1)
    trunc = _readonly_bool(truncated, name="truncated", ndim=1)
    if term.shape != pending.shape or trunc.shape != pending.shape:
        raise ValueError("pending_reset, terminated, and truncated must have identical shapes")
    if np.any(term & trunc):
        raise ValueError("a transition cannot be both terminated and truncated")
    if np.any(pending & (term | trunc)):
        raise ValueError("a NEXT_STEP reset-only row must return false terminal flags")

    valid = ~pending
    terminal = term | trunc
    return StepTransitionMasks(
        valid_transition=_readonly_copy(valid),
        reset_only=_readonly_copy(pending),
        terminated=_readonly_copy(term),
        truncated=_readonly_copy(trunc),
        terminal_event=_readonly_copy(terminal),
        next_pending_reset=_readonly_copy(terminal),
    )


def build_rollout_transition_masks(
    initial_pending_reset: Any,
    terminated: Any,
    truncated: Any,
) -> RolloutTransitionMasks:
    """Derive every rollout mask while preserving pending reset across rollout boundaries."""

    initial = _readonly_bool(
        initial_pending_reset,
        name="initial_pending_reset",
        ndim=1,
    )
    term = _readonly_bool(terminated, name="terminated", ndim=2)
    trunc = _readonly_bool(truncated, name="truncated", ndim=2)
    if term.shape != trunc.shape:
        raise ValueError("terminated and truncated must have identical shapes")
    if term.shape[1] != initial.shape[0]:
        raise ValueError("terminal flag width must match initial_pending_reset")

    valid = np.empty_like(term)
    reset_only = np.empty_like(term)
    terminal = np.empty_like(term)
    pending = np.array(initial, copy=True)
    for step in range(term.shape[0]):
        masks = build_step_transition_masks(pending, term[step], trunc[step])
        valid[step] = masks.valid_transition
        reset_only[step] = masks.reset_only
        terminal[step] = masks.terminal_event
        pending = np.array(masks.next_pending_reset, copy=True)

    valid_count = int(np.count_nonzero(valid))
    dummy_count = int(np.count_nonzero(reset_only))
    terminated_count = int(np.count_nonzero(term))
    truncated_count = int(np.count_nonzero(trunc))
    counts = TransitionCounts(
        num_envs=term.shape[1],
        environment_step_calls=term.shape[0],
        raw_transitions=term.size,
        valid_transitions=valid_count,
        dummy_reset_transitions=dummy_count,
        autoreset_slots=dummy_count,
        terminal_events=terminated_count + truncated_count,
        terminated_events=terminated_count,
        truncated_events=truncated_count,
    )
    return RolloutTransitionMasks(
        initial_pending_reset=initial,
        valid_transition=_readonly_copy(valid),
        reset_only=_readonly_copy(reset_only),
        terminated=term,
        truncated=trunc,
        terminal_event=_readonly_copy(terminal),
        final_pending_reset=_readonly_copy(pending),
        counts=counts,
    )


def generalized_advantage_estimate(
    rewards: Any,
    values: Any,
    masks: RolloutTransitionMasks,
    *,
    gamma: Real,
    gae_lambda: Real,
) -> GaeResult:
    """Compute NEXT_STEP-aware GAE with correct time-limit bootstrapping.

    True termination neither bootstraps nor recurses.  Truncation bootstraps from the returned
    terminal observation but stops recursion at the episode boundary.  A reset-only row contributes
    exactly zero temporal difference, advantage, and return.
    """

    if not isinstance(masks, RolloutTransitionMasks):
        raise TypeError("masks must be RolloutTransitionMasks")
    reward = _readonly_float(rewards, name="rewards", ndim=2)
    value = _readonly_float(values, name="values", ndim=2)
    if reward.shape != masks.shape:
        raise ValueError("rewards shape must match rollout masks")
    expected_value_shape = (masks.shape[0] + 1, masks.shape[1])
    if value.shape != expected_value_shape:
        raise ValueError(f"values must have shape {expected_value_shape}")
    if np.any(reward[masks.reset_only] != 0.0):
        raise ValueError("NEXT_STEP reset-only rewards must be exactly zero")

    discount = _probability(gamma, name="gamma")
    trace_decay = _probability(gae_lambda, name="gae_lambda")
    temporal_difference = np.zeros(masks.shape, dtype=np.float64)
    advantages = np.zeros(masks.shape, dtype=np.float64)
    last_advantage = np.zeros(masks.shape[1], dtype=np.float64)

    for step in range(masks.shape[0] - 1, -1, -1):
        bootstrap = (~masks.terminated[step]).astype(np.float64)
        continuation = (~masks.terminal_event[step]).astype(np.float64)
        delta = reward[step] + discount * bootstrap * value[step + 1] - value[step]
        candidate = delta + discount * trace_decay * continuation * last_advantage
        valid = masks.valid_transition[step]
        temporal_difference[step] = np.where(valid, delta, 0.0)
        last_advantage = np.where(valid, candidate, 0.0)
        advantages[step] = last_advantage

    returns = np.where(
        masks.valid_transition,
        advantages + value[:-1],
        0.0,
    )
    if not (
        np.isfinite(temporal_difference).all()
        and np.isfinite(advantages).all()
        and np.isfinite(returns).all()
    ):
        raise FloatingPointError("GAE produced a non-finite result")
    return GaeResult(
        temporal_difference=_readonly_copy(temporal_difference),
        advantages=_readonly_copy(advantages),
        returns=_readonly_copy(returns),
    )


def valid_flat_indices(valid_transition: Any) -> IndexArray:
    """Return time-major flattened indices for every valid learning transition."""

    valid = _readonly_bool(valid_transition, name="valid_transition", ndim=2)
    indices = np.flatnonzero(valid.reshape(-1)).astype(np.int64, copy=False)
    indices.setflags(write=False)
    return indices


def split_valid_indices(
    valid_transition: Any,
    num_minibatches: int,
    *,
    ordered_indices: Any | None = None,
) -> tuple[IndexArray, ...]:
    """Split every valid flat index exactly once without padding, dropping, or duplication.

    ``ordered_indices`` can be a caller-generated shuffle, but it must be an exact permutation of
    the valid flat indices.  Minibatch sizes differ by at most one.
    """

    if type(num_minibatches) is not int or num_minibatches < 1:
        raise ValueError("num_minibatches must be a positive integer")
    expected = valid_flat_indices(valid_transition)
    if expected.size < num_minibatches:
        raise ValueError("num_minibatches cannot exceed the number of valid transitions")

    if ordered_indices is None:
        ordered = np.array(expected, copy=True)
    else:
        source = np.asarray(ordered_indices)
        if source.ndim != 1 or source.dtype.kind not in {"i", "u"}:
            raise TypeError("ordered_indices must be a one-dimensional integer array")
        if source.size != expected.size:
            raise ValueError("ordered_indices must contain every valid index exactly once")
        ordered = np.asarray(source, dtype=np.int64)
        if not np.array_equal(np.sort(ordered), expected):
            raise ValueError("ordered_indices must be an exact permutation of valid indices")

    batches = []
    for batch in np.array_split(ordered, num_minibatches):
        value = np.array(batch, dtype=np.int64, copy=True)
        value.setflags(write=False)
        batches.append(value)
    return tuple(batches)


__all__ = [
    "GaeResult",
    "RolloutTransitionMasks",
    "StepTransitionMasks",
    "TransitionCounts",
    "build_rollout_transition_masks",
    "build_step_transition_masks",
    "generalized_advantage_estimate",
    "split_valid_indices",
    "valid_flat_indices",
]
