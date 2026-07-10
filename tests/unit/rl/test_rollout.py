"""Tests for NEXT_STEP rollout masking and reference GAE equations."""

from __future__ import annotations

from dataclasses import replace

import numpy as np
import pytest

from controller_learning.rl.rollout import (
    TransitionCounts,
    build_rollout_transition_masks,
    build_step_transition_masks,
    generalized_advantage_estimate,
    split_valid_indices,
    valid_flat_indices,
)


def _hand_computed_rollout():
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
    return masks, rewards, values


def test_hand_computed_termination_truncation_and_reset_only_gae() -> None:
    masks, rewards, values = _hand_computed_rollout()

    result = generalized_advantage_estimate(
        rewards,
        values,
        masks,
        gamma=0.9,
        gae_lambda=0.8,
    )

    np.testing.assert_array_equal(
        masks.valid_transition,
        (
            (True, True),
            (True, True),
            (False, False),
            (True, True),
        ),
    )
    np.testing.assert_allclose(
        result.advantages,
        (
            (-3.960, 3.056),
            (-18.000, 2.300),
            (0.000, 0.000),
            (9.000, 3.100),
        ),
        rtol=0.0,
        atol=1.0e-12,
    )
    np.testing.assert_allclose(
        result.returns,
        (
            (6.040, 8.056),
            (2.000, 8.300),
            (0.000, 0.000),
            (39.000, 11.100),
        ),
        rtol=0.0,
        atol=1.0e-12,
    )
    # World 0 terminates and therefore ignores V=99.  World 1 truncates and bootstraps V=7,
    # while both stop recursion before the new episodes begin.
    assert result.returns[1, 0] == pytest.approx(2.0)
    assert result.returns[1, 1] == pytest.approx(2.0 + 0.9 * 7.0)
    np.testing.assert_array_equal(result.temporal_difference[2], (0.0, 0.0))
    for array in (
        masks.valid_transition,
        masks.final_pending_reset,
        result.temporal_difference,
        result.advantages,
        result.returns,
    ):
        assert not array.flags.writeable

    assert masks.counts == TransitionCounts(
        num_envs=2,
        environment_step_calls=4,
        raw_transitions=8,
        valid_transitions=6,
        dummy_reset_transitions=2,
        autoreset_slots=2,
        terminal_events=2,
        terminated_events=1,
        truncated_events=1,
    )


def test_pending_reset_carry_survives_a_rollout_boundary() -> None:
    rollout_a = build_rollout_transition_masks(
        np.asarray((False, False), dtype=np.bool_),
        np.asarray(((True, False),), dtype=np.bool_),
        np.asarray(((False, True),), dtype=np.bool_),
    )
    np.testing.assert_array_equal(rollout_a.final_pending_reset, (True, True))
    assert rollout_a.counts.terminal_events - rollout_a.counts.autoreset_slots == 2

    rollout_b = build_rollout_transition_masks(
        rollout_a.final_pending_reset,
        np.zeros((2, 2), dtype=np.bool_),
        np.zeros((2, 2), dtype=np.bool_),
    )

    np.testing.assert_array_equal(
        rollout_b.valid_transition,
        ((False, False), (True, True)),
    )
    np.testing.assert_array_equal(
        rollout_b.reset_only,
        ((True, True), (False, False)),
    )
    np.testing.assert_array_equal(rollout_b.final_pending_reset, (False, False))
    assert rollout_b.counts.terminal_events - rollout_b.counts.autoreset_slots == -2

    boundary_result = generalized_advantage_estimate(
        np.asarray(((0.0, 0.0), (2.0, 3.0)), dtype=np.float64),
        np.asarray(((9.0, 8.0), (3.0, 4.0), (5.0, 6.0)), dtype=np.float64),
        rollout_b,
        gamma=0.9,
        gae_lambda=0.8,
    )
    np.testing.assert_array_equal(boundary_result.advantages[0], (0.0, 0.0))
    np.testing.assert_allclose(
        boundary_result.advantages[1],
        (3.5, 4.4),
        rtol=0.0,
        atol=1.0e-12,
    )
    np.testing.assert_allclose(
        boundary_result.returns[1],
        (6.5, 8.4),
        rtol=0.0,
        atol=1.0e-12,
    )
    assert rollout_a.counts + rollout_b.counts == TransitionCounts(
        num_envs=2,
        environment_step_calls=3,
        raw_transitions=6,
        valid_transitions=4,
        dummy_reset_transitions=2,
        autoreset_slots=2,
        terminal_events=2,
        terminated_events=1,
        truncated_events=1,
    )


def test_rollout_masks_reject_transition_count_conservation_drift() -> None:
    masks, _, _ = _hand_computed_rollout()
    contradictory_counts = TransitionCounts(
        num_envs=2,
        environment_step_calls=4,
        raw_transitions=8,
        valid_transitions=6,
        dummy_reset_transitions=2,
        autoreset_slots=2,
        terminal_events=3,
        terminated_events=2,
        truncated_events=1,
    )

    with pytest.raises(ValueError, match="pending-reset carry change"):
        replace(masks, counts=contradictory_counts)


def test_valid_indices_and_minibatches_cover_each_sample_exactly_once() -> None:
    masks, _, _ = _hand_computed_rollout()
    expected = np.asarray((0, 1, 2, 3, 6, 7), dtype=np.int64)
    np.testing.assert_array_equal(valid_flat_indices(masks.valid_transition), expected)

    shuffled = np.asarray((7, 0, 6, 2, 1, 3), dtype=np.int64)
    batches = split_valid_indices(
        masks.valid_transition,
        4,
        ordered_indices=shuffled,
    )

    assert tuple(batch.size for batch in batches) == (2, 2, 1, 1)
    np.testing.assert_array_equal(np.concatenate(batches), shuffled)
    np.testing.assert_array_equal(np.sort(np.concatenate(batches)), expected)
    assert all(not batch.flags.writeable for batch in batches)

    with pytest.raises(ValueError, match="exact permutation"):
        split_valid_indices(
            masks.valid_transition,
            2,
            ordered_indices=np.asarray((0, 1, 2, 3, 6, 6)),
        )
    with pytest.raises(ValueError, match="cannot exceed"):
        split_valid_indices(masks.valid_transition, 7)


def test_step_masks_reject_impossible_or_untyped_next_step_rows() -> None:
    pending = np.asarray((True, False), dtype=np.bool_)
    false = np.asarray((False, False), dtype=np.bool_)
    masks = build_step_transition_masks(pending, false, false)
    np.testing.assert_array_equal(masks.valid_transition, (False, True))
    np.testing.assert_array_equal(masks.next_pending_reset, (False, False))

    with pytest.raises(ValueError, match="reset-only"):
        build_step_transition_masks(
            pending,
            np.asarray((True, False), dtype=np.bool_),
            false,
        )
    with pytest.raises(ValueError, match="both"):
        build_step_transition_masks(
            false,
            np.asarray((True, False), dtype=np.bool_),
            np.asarray((True, False), dtype=np.bool_),
        )
    with pytest.raises(TypeError, match="boolean dtype"):
        build_step_transition_masks(np.asarray((0, 1), dtype=np.int8), false, false)


def test_gae_rejects_nonzero_dummy_reward_and_bad_shapes() -> None:
    masks, rewards, values = _hand_computed_rollout()
    bad_reward = np.array(rewards, copy=True)
    bad_reward[2, 0] = 1.0

    with pytest.raises(ValueError, match="exactly zero"):
        generalized_advantage_estimate(
            bad_reward,
            values,
            masks,
            gamma=0.9,
            gae_lambda=0.8,
        )
    with pytest.raises(ValueError, match="values must have shape"):
        generalized_advantage_estimate(
            rewards,
            values[:-1],
            masks,
            gamma=0.9,
            gae_lambda=0.8,
        )
    with pytest.raises(ValueError, match=r"\[0, 1\]"):
        generalized_advantage_estimate(
            rewards,
            values,
            masks,
            gamma=1.1,
            gae_lambda=0.8,
        )
