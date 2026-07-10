"""GPU tests for valid-only, explicitly shuffled PPO optimization."""

from __future__ import annotations

import dataclasses
import importlib
import math
from pathlib import Path
from typing import Any

import pytest

from controller_learning.config import load_project_config
from controller_learning.envs.observation import action_space
from controller_learning.rl.configuration import PpoAlgorithmConfig, load_ppo_config
from controller_learning.rl.rollout import TransitionCounts

PROJECT_ROOT = Path(__file__).parents[2]
OBSERVATION_DIM = 100
pytestmark = pytest.mark.gpu


def _torch() -> Any:
    return importlib.import_module("torch")


def _collector_module() -> Any:
    return importlib.import_module("controller_learning.rl.collector")


def _policy_module() -> Any:
    return importlib.import_module("controller_learning.rl.policy")


def _ppo_module() -> Any:
    return importlib.import_module("controller_learning.rl.ppo")


def _device() -> Any:
    torch = _torch()
    return torch.device("cuda", torch.cuda.current_device())


def _policy(*, seed: int = 11) -> Any:
    physical = action_space(load_project_config(PROJECT_ROOT))
    return _policy_module().PpoActorCritic(
        OBSERVATION_DIM,
        action_low=physical.low,
        action_high=physical.high,
        policy_seed=seed,
        initial_log_std=-0.5,
        device=_device(),
    )


def _config(**changes: Any) -> PpoAlgorithmConfig:
    base = load_ppo_config(PROJECT_ROOT / "configs" / "ppo.toml").ppo
    return dataclasses.replace(base, **changes)


def _generator(seed: int) -> Any:
    torch = _torch()
    return torch.Generator(device=_device()).manual_seed(seed)


def _synthetic_rollout(policy: Any) -> Any:
    torch = _torch()
    collector_module = _collector_module()
    device = policy.device
    rollout_steps = 4
    num_envs = 3
    observations = torch.linspace(
        -1.0,
        1.0,
        rollout_steps * num_envs * OBSERVATION_DIM,
        device=device,
    ).reshape(rollout_steps, num_envs, OBSERVATION_DIM)
    pre_tanh = torch.stack(
        (0.3 * observations[..., 0], -0.2 * observations[..., 1]),
        dim=-1,
    )
    with torch.no_grad():
        evaluated = policy.evaluate(observations, pre_tanh)

    terminated = torch.zeros((rollout_steps, num_envs), dtype=torch.bool, device=device)
    truncated = torch.zeros_like(terminated)
    terminated[1, 2] = True
    reset_only = torch.zeros_like(terminated)
    reset_only[0, 1] = True
    reset_only[2, 2] = True
    valid_transition = torch.logical_not(reset_only)
    values = torch.cat(
        (evaluated.value, torch.zeros((1, num_envs), device=device)),
        dim=0,
    )
    rewards = torch.zeros((rollout_steps, num_envs), device=device)
    initial_pending = torch.tensor((False, True, False), dtype=torch.bool, device=device)
    final_pending = torch.zeros(num_envs, dtype=torch.bool, device=device)
    counts = TransitionCounts(
        num_envs=num_envs,
        environment_step_calls=rollout_steps,
        raw_transitions=rollout_steps * num_envs,
        valid_transitions=10,
        dummy_reset_transitions=2,
        autoreset_slots=2,
        terminal_events=1,
        terminated_events=1,
        truncated_events=0,
    )
    return collector_module.CollectedRollout(
        observations=observations,
        pre_tanh_actions=pre_tanh,
        actions=evaluated.action,
        old_log_prob=evaluated.log_prob,
        values=values,
        rewards=rewards,
        terminated=terminated,
        truncated=truncated,
        termination_reason=torch.zeros((rollout_steps, num_envs), dtype=torch.int32, device=device),
        lap_completed=torch.zeros((rollout_steps, num_envs), dtype=torch.bool, device=device),
        lap_time_s=torch.zeros((rollout_steps, num_envs), device=device),
        episode_seed=torch.zeros((rollout_steps, num_envs), dtype=torch.uint32, device=device),
        controller_seed=torch.zeros((rollout_steps, num_envs), dtype=torch.uint32, device=device),
        track_id=torch.zeros((rollout_steps, num_envs), dtype=torch.uint32, device=device),
        valid_transition=valid_transition,
        reset_only=reset_only,
        initial_pending_reset=initial_pending,
        final_state=collector_module.CollectorState(
            observation=observations[-1].clone(),
            pending_reset=final_pending,
        ),
        counts=counts,
    )


def _perturb_dummy_rows(
    rollout: Any,
) -> Any:
    torch = _torch()
    dummy = rollout.reset_only
    observations = rollout.observations.clone()
    observations[dummy] = torch.nan
    pre_tanh = rollout.pre_tanh_actions.clone()
    pre_tanh[dummy] = torch.inf
    old_log_prob = rollout.old_log_prob.clone()
    old_log_prob[dummy] = -torch.inf
    return dataclasses.replace(
        rollout,
        observations=observations,
        pre_tanh_actions=pre_tanh,
        old_log_prob=old_log_prob,
    )


def test_valid_minibatches_cover_every_sample_once_without_padding() -> None:
    torch = _torch()
    ppo_module = _ppo_module()
    valid = torch.tensor(
        (
            (True, False, True, True),
            (False, True, True, True),
            (True, True, False, True),
            (True, False, False, True),
        ),
        dtype=torch.bool,
        device=_device(),
    )

    batches = ppo_module.build_valid_minibatches(
        valid, num_minibatches=4, generator=_generator(101)
    )
    repeated = ppo_module.build_valid_minibatches(
        valid, num_minibatches=4, generator=_generator(101)
    )
    flat = torch.cat(batches)
    expected = torch.nonzero(valid.reshape(-1), as_tuple=False).squeeze(-1)

    assert tuple(batch.shape[0] for batch in batches) == (3, 3, 3, 2)
    assert all(batch.device == _device() and batch.dtype == torch.int64 for batch in batches)
    assert torch.equal(torch.sort(flat).values, expected)
    assert torch.unique(flat).shape[0] == expected.shape[0]
    assert all(torch.equal(first, second) for first, second in zip(batches, repeated, strict=True))
    assert torch.all(valid.reshape(-1).index_select(0, flat))

    with pytest.raises(ValueError, match="CUDA generator"):
        ppo_module.build_valid_minibatches(
            valid,
            num_minibatches=4,
            generator=torch.Generator(device="cpu"),
        )


def test_update_is_finite_changes_parameters_and_uses_explicit_learning_rate() -> None:
    torch = _torch()
    ppo_module = _ppo_module()
    policy = _policy(seed=107)
    rollout = _synthetic_rollout(policy)
    config = _config(
        num_minibatches=3,
        update_epochs=2,
        target_kl=1.0e6,
        max_gradient_norm=1.0e-4,
    )
    updater = ppo_module.PpoUpdater(policy, config)
    before = {name: tensor.clone() for name, tensor in policy.state_dict().items()}
    learning_rate = config.learning_rate * 0.5

    metrics = updater.update(
        rollout,
        learning_rate=learning_rate,
        minibatch_generator=_generator(109),
    )

    assert isinstance(metrics, ppo_module.UpdateMetrics)
    assert metrics.learning_rate == learning_rate
    assert metrics.valid_samples == 10
    assert metrics.samples_processed == 20
    assert metrics.epochs_run == metrics.epochs_completed == 2
    assert metrics.minibatches_processed == 6
    assert not metrics.early_stopped_for_kl
    assert metrics.max_gradient_norm_before_clip > config.max_gradient_norm
    assert isinstance(updater.optimizer, torch.optim.Adam)
    assert updater.optimizer.defaults["eps"] == config.adam_epsilon
    assert all(group["lr"] == learning_rate for group in updater.optimizer.param_groups)
    assert any(
        not torch.equal(before[name], current) for name, current in policy.state_dict().items()
    )
    assert torch.all(policy.log_std >= -5.0)
    assert torch.all(policy.log_std <= 2.0)
    for field in dataclasses.fields(metrics):
        value = getattr(metrics, field.name)
        assert not isinstance(value, torch.Tensor)
        if isinstance(value, float):
            assert math.isfinite(value)


def test_dummy_row_perturbations_and_ambient_rng_do_not_change_update() -> None:
    torch = _torch()
    ppo_module = _ppo_module()
    first_policy = _policy(seed=127)
    second_policy = _policy(seed=127)
    rollout = _synthetic_rollout(first_policy)
    perturbed_rollout = _perturb_dummy_rows(rollout)
    config = _config(num_minibatches=4, update_epochs=2, target_kl=1.0e6)
    first_updater = ppo_module.PpoUpdater(first_policy, config)
    second_updater = ppo_module.PpoUpdater(second_policy, config)

    first_metrics = first_updater.update(
        rollout,
        learning_rate=config.learning_rate,
        minibatch_generator=_generator(131),
    )
    torch.rand((4096,), device=_device())
    second_metrics = second_updater.update(
        perturbed_rollout,
        learning_rate=config.learning_rate,
        minibatch_generator=_generator(131),
    )

    assert first_metrics == second_metrics
    for name, first in first_policy.state_dict().items():
        torch.testing.assert_close(
            second_policy.state_dict()[name],
            first,
            rtol=0.0,
            atol=0.0,
        )


def test_target_kl_stops_after_a_fresh_post_epoch_full_batch_measurement() -> None:
    torch = _torch()
    ppo_module = _ppo_module()
    policy = _policy(seed=137)
    rollout = _synthetic_rollout(policy)
    shifted_log_prob = rollout.old_log_prob.clone()
    shifted_log_prob[rollout.valid_transition] += 1.0
    rollout = dataclasses.replace(rollout, old_log_prob=shifted_log_prob)
    config = _config(num_minibatches=4, update_epochs=3, target_kl=1.0e-5)
    updater = ppo_module.PpoUpdater(policy, config)

    metrics = updater.update(
        rollout,
        learning_rate=config.learning_rate,
        minibatch_generator=_generator(139),
    )

    assert metrics.early_stopped_for_kl
    assert metrics.epochs_run == 1
    assert metrics.epochs_completed == 1
    assert metrics.minibatches_processed == 4
    assert metrics.samples_processed == 10
    assert metrics.post_epoch_kl > config.target_kl
    assert metrics.clip_fraction == pytest.approx(1.0)
    valid = rollout.valid_transition
    with torch.no_grad():
        current = policy.evaluate(
            rollout.observations[valid],
            rollout.pre_tanh_actions[valid],
        )
        log_ratio = current.log_prob - rollout.old_log_prob[valid]
        ratio = torch.exp(log_ratio)
        expected_post_epoch_kl = ((ratio - 1.0) - log_ratio).mean().item()
    assert metrics.post_epoch_kl == pytest.approx(expected_post_epoch_kl, rel=1.0e-6)


def test_nonfinite_valid_input_is_rejected_before_optimizer_mutation() -> None:
    torch = _torch()
    ppo_module = _ppo_module()
    policy = _policy(seed=149)
    rollout = _synthetic_rollout(policy)
    observations = rollout.observations.clone()
    observations[0, 0, 0] = torch.nan
    bad_rollout = dataclasses.replace(rollout, observations=observations)
    updater = ppo_module.PpoUpdater(policy, _config(num_minibatches=2, update_epochs=1))
    before = {name: tensor.clone() for name, tensor in policy.state_dict().items()}

    with pytest.raises(FloatingPointError, match="valid PPO rollout"):
        updater.update(
            bad_rollout,
            learning_rate=updater.config.learning_rate,
            minibatch_generator=_generator(151),
        )
    for name, current in policy.state_dict().items():
        torch.testing.assert_close(current, before[name], rtol=0.0, atol=0.0)

    overflow_log_prob = rollout.old_log_prob.clone()
    overflow_log_prob[rollout.valid_transition] = -1.0e30
    overflow_rollout = dataclasses.replace(rollout, old_log_prob=overflow_log_prob)
    with pytest.raises(FloatingPointError, match="gradient norm"):
        updater.update(
            overflow_rollout,
            learning_rate=updater.config.learning_rate,
            minibatch_generator=_generator(153),
        )
    for name, current in policy.state_dict().items():
        torch.testing.assert_close(current, before[name], rtol=0.0, atol=0.0)

    with pytest.raises(ValueError, match="finite and positive"):
        updater.update(
            rollout,
            learning_rate=0.0,
            minibatch_generator=_generator(157),
        )
