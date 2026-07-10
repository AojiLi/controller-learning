"""GPU tests for the CleanRL-style squashed-Gaussian PPO policy."""

from __future__ import annotations

import importlib
import math
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


def _device() -> Any:
    torch = _torch()
    return torch.device("cuda", torch.cuda.current_device())


def _bounds() -> tuple[np.ndarray, np.ndarray]:
    physical = action_space(load_project_config(PROJECT_ROOT))
    return physical.low, physical.high


def _policy(*, seed: int = 11) -> Any:
    policy_module = _policy_module()
    low, high = _bounds()
    return policy_module.PpoActorCritic(
        OBSERVATION_DIM,
        action_low=low,
        action_high=high,
        policy_seed=seed,
        initial_log_std=-0.5,
        hidden_sizes=policy_module.PPO_HIDDEN_SIZES,
        device=_device(),
    )


def _generator(seed: int) -> Any:
    torch = _torch()
    return torch.Generator(device=_device()).manual_seed(seed)


def test_policy_samples_finite_actions_inside_exact_physical_bounds() -> None:
    torch = _torch()
    policy = _policy()
    observations = torch.randn((4096, OBSERVATION_DIM), device=_device())

    sampled = policy.sample(observations, generator=_generator(101))

    assert sampled.action.shape == sampled.pre_tanh.shape == sampled.mean.shape == (4096, 2)
    assert sampled.log_prob.shape == sampled.latent_entropy.shape == sampled.value.shape == (4096,)
    for tensor in (
        sampled.action,
        sampled.pre_tanh,
        sampled.mean,
        sampled.log_prob,
        sampled.latent_entropy,
        sampled.value,
    ):
        assert tensor.dtype == torch.float32
        assert tensor.device == _device()
        assert torch.all(torch.isfinite(tensor))
    assert torch.all(sampled.action >= policy.action_low)
    assert torch.all(sampled.action <= policy.action_high)

    low, high = _bounds()
    torch.testing.assert_close(policy.action_low.cpu(), torch.from_numpy(low), rtol=0.0, atol=0.0)
    torch.testing.assert_close(policy.action_high.cpu(), torch.from_numpy(high), rtol=0.0, atol=0.0)
    assert policy.action_bias[1].item() != 0.0
    torch.testing.assert_close(
        policy.log_std,
        torch.full((2,), -0.5, device=_device()),
        rtol=0.0,
        atol=0.0,
    )


def test_stored_latent_evaluation_exactly_matches_sampling() -> None:
    torch = _torch()
    policy = _policy()
    observations = torch.randn((64, OBSERVATION_DIM), device=_device())
    sampled = policy.sample(observations, generator=_generator(103))

    evaluated = policy.evaluate(observations, sampled.pre_tanh)

    torch.testing.assert_close(evaluated.action, sampled.action, rtol=0.0, atol=0.0)
    torch.testing.assert_close(evaluated.pre_tanh, sampled.pre_tanh, rtol=0.0, atol=0.0)
    torch.testing.assert_close(evaluated.mean, sampled.mean, rtol=0.0, atol=0.0)
    torch.testing.assert_close(evaluated.log_prob, sampled.log_prob, rtol=0.0, atol=0.0)
    torch.testing.assert_close(
        evaluated.latent_entropy,
        sampled.latent_entropy,
        rtol=0.0,
        atol=0.0,
    )
    torch.testing.assert_close(evaluated.value, sampled.value, rtol=0.0, atol=0.0)
    torch.testing.assert_close(policy.value(observations), sampled.value, rtol=0.0, atol=0.0)


def test_log_probability_uses_stable_tanh_and_action_scale_jacobians() -> None:
    torch = _torch()
    policy = _policy()
    observations = torch.zeros((3, OBSERVATION_DIM), device=_device())
    latent = torch.tensor(((20.0, -20.0), (0.0, 0.0), (-3.0, 4.0)), device=_device())

    evaluated = policy.evaluate(observations, latent)
    log_std = policy._effective_log_std(evaluated.mean)
    normal = (
        -0.5 * ((latent - evaluated.mean) * torch.exp(-log_std)).square()
        - log_std
        - 0.5 * math.log(2.0 * math.pi)
    )
    log_tanh_jacobian = 2.0 * (math.log(2.0) - latent - torch.nn.functional.softplus(-2.0 * latent))
    expected = torch.sum(normal - log_tanh_jacobian - torch.log(policy.action_scale), dim=-1)

    assert torch.all(torch.isfinite(evaluated.log_prob))
    torch.testing.assert_close(evaluated.log_prob, expected, rtol=1.0e-6, atol=1.0e-6)
    assert torch.all(evaluated.action >= policy.action_low)
    assert torch.all(evaluated.action <= policy.action_high)


def test_policy_initialization_and_explicit_sampling_ignore_ambient_rngs() -> None:
    torch = _torch()
    torch.manual_seed(211)
    torch.cuda.manual_seed_all(223)
    expected_cpu = torch.rand(8)
    expected_cuda = torch.rand(8, device=_device())

    torch.manual_seed(211)
    torch.cuda.manual_seed_all(223)
    first = _policy(seed=227)
    actual_cpu = torch.rand(8)
    actual_cuda = torch.rand(8, device=_device())
    torch.testing.assert_close(actual_cpu, expected_cpu, rtol=0.0, atol=0.0)
    torch.testing.assert_close(actual_cuda, expected_cuda, rtol=0.0, atol=0.0)

    torch.manual_seed(229)
    torch.cuda.manual_seed_all(233)
    torch.rand(37)
    torch.rand(41, device=_device())
    second = _policy(seed=227)
    for name, value in first.state_dict().items():
        torch.testing.assert_close(value, second.state_dict()[name], rtol=0.0, atol=0.0)

    observations = torch.ones((16, OBSERVATION_DIM), device=_device())
    first_sample = first.sample(observations, generator=_generator(239))
    torch.randn((1024,), device=_device())
    second_sample = second.sample(observations, generator=_generator(239))
    torch.testing.assert_close(first_sample.action, second_sample.action, rtol=0.0, atol=0.0)
    torch.testing.assert_close(
        first_sample.pre_tanh,
        second_sample.pre_tanh,
        rtol=0.0,
        atol=0.0,
    )


def test_deterministic_action_is_the_affine_mapped_tanh_mean() -> None:
    torch = _torch()
    policy = _policy()
    observations = torch.randn((32, OBSERVATION_DIM), device=_device())

    first = policy.deterministic(observations)
    torch.randn((1000,), device=_device())
    second = policy.deterministic(observations)
    expected = policy.action_bias + policy.action_scale * torch.tanh(first.pre_tanh)

    torch.testing.assert_close(first.action, expected, rtol=0.0, atol=0.0)
    torch.testing.assert_close(second.action, first.action, rtol=0.0, atol=0.0)
    torch.testing.assert_close(second.pre_tanh, first.pre_tanh, rtol=0.0, atol=0.0)
    torch.testing.assert_close(second.value, first.value, rtol=0.0, atol=0.0)


def test_policy_gradients_and_state_dict_reload() -> None:
    torch = _torch()
    policy = _policy(seed=241)
    observations = torch.randn((128, OBSERVATION_DIM), device=_device())
    with torch.no_grad():
        stored = policy.sample(observations, generator=_generator(251)).pre_tanh

    evaluated = policy.evaluate(observations, stored)
    loss = (
        -evaluated.log_prob.mean()
        - 0.01 * evaluated.latent_entropy.mean()
        + 0.5 * (evaluated.value - 1.0).square().mean()
    )
    loss.backward()

    gradient_groups = {
        "trunk": tuple(policy.trunk.parameters()),
        "actor": tuple(policy.actor_mean.parameters()),
        "critic": tuple(policy.critic.parameters()),
        "log_std": (policy.log_std,),
    }
    for parameters in gradient_groups.values():
        assert all(parameter.grad is not None for parameter in parameters)
        assert all(torch.all(torch.isfinite(parameter.grad)) for parameter in parameters)
        assert any(torch.count_nonzero(parameter.grad) > 0 for parameter in parameters)

    restored = _policy(seed=257)
    restored.load_state_dict(policy.state_dict())
    reference = policy.deterministic(observations)
    reloaded = restored.deterministic(observations)
    torch.testing.assert_close(reloaded.action, reference.action, rtol=0.0, atol=0.0)
    torch.testing.assert_close(reloaded.value, reference.value, rtol=0.0, atol=0.0)


def test_policy_clamps_trainable_log_std_to_a_finite_operating_range() -> None:
    torch = _torch()
    policy = _policy()
    observations = torch.zeros((32, OBSERVATION_DIM), device=_device())

    for parameter_value, expected in ((1.0e20, 2.0), (-1.0e20, -5.0)):
        with torch.no_grad():
            policy.log_std.fill_(parameter_value)
        sample = policy.sample(observations, generator=_generator(263))
        assert torch.all(torch.isfinite(sample.action))
        assert torch.all(torch.isfinite(sample.pre_tanh))
        assert torch.all(torch.isfinite(sample.log_prob))
        torch.testing.assert_close(
            policy._effective_log_std(sample.mean),
            torch.full_like(sample.mean, expected),
            rtol=0.0,
            atol=0.0,
        )

    with torch.no_grad():
        policy.log_std.fill_(100.0)
    policy.project_log_std_()
    torch.testing.assert_close(
        policy.log_std,
        torch.full_like(policy.log_std, 2.0),
        rtol=0.0,
        atol=0.0,
    )
    policy.zero_grad(set_to_none=True)
    evaluation = policy.evaluate(
        observations,
        torch.zeros((32, 2), device=_device()),
    )
    evaluation.latent_entropy.mean().backward()
    assert torch.all(policy.log_std.grad > 0.0)
    optimizer = torch.optim.SGD((policy.log_std,), lr=0.1)
    optimizer.step()
    assert torch.all(policy.log_std < 2.0)


def test_policy_rejects_invalid_shapes_devices_dtypes_bounds_and_generators() -> None:
    torch = _torch()
    policy_class = _policy_module().PpoActorCritic
    low, high = _bounds()
    with pytest.raises(ValueError, match="positive integer"):
        policy_class(
            0,
            action_low=low,
            action_high=high,
            policy_seed=1,
            initial_log_std=-0.5,
            device=_device(),
        )
    with pytest.raises(ValueError, match="exactly"):
        policy_class(
            OBSERVATION_DIM,
            action_low=low,
            action_high=high,
            policy_seed=1,
            initial_log_std=-0.5,
            hidden_sizes=(64, 64),
            device=_device(),
        )
    with pytest.raises(ValueError, match="uint32"):
        policy_class(
            OBSERVATION_DIM,
            action_low=low,
            action_high=high,
            policy_seed=-1,
            initial_log_std=-0.5,
            device=_device(),
        )
    with pytest.raises(ValueError, match="greater"):
        policy_class(
            OBSERVATION_DIM,
            action_low=high,
            action_high=low,
            policy_seed=1,
            initial_log_std=-0.5,
            device=_device(),
        )
    with pytest.raises(ValueError, match=r"remain finite in torch\.float32"):
        policy_class(
            OBSERVATION_DIM,
            action_low=(-1.0e300, -1.0),
            action_high=(1.0e300, 1.0),
            policy_seed=1,
            initial_log_std=-0.5,
            device=_device(),
        )
    with pytest.raises(ValueError, match="float32"):
        policy_class(
            OBSERVATION_DIM,
            action_low=low,
            action_high=high,
            policy_seed=1,
            initial_log_std=-0.5,
            device=_device(),
            dtype=torch.float64,
        )
    with pytest.raises(ValueError, match="initial_log_std must be finite"):
        policy_class(
            OBSERVATION_DIM,
            action_low=low,
            action_high=high,
            policy_seed=1,
            initial_log_std=float("nan"),
            device=_device(),
        )
    with pytest.raises(ValueError, match="initial_log_std must be in"):
        policy_class(
            OBSERVATION_DIM,
            action_low=low,
            action_high=high,
            policy_seed=1,
            initial_log_std=3.0,
            device=_device(),
        )

    policy = _policy()
    valid = torch.zeros((2, OBSERVATION_DIM), device=_device())
    with pytest.raises(ValueError, match="fixed observation dimension"):
        policy.value(torch.zeros((2, OBSERVATION_DIM - 1), device=_device()))
    with pytest.raises(TypeError, match=r"torch\.float32"):
        policy.value(valid.to(dtype=torch.float64))
    with pytest.raises(ValueError, match="policy is on"):
        policy.value(valid.cpu())
    with pytest.raises(TypeError, match=r"explicit torch\.Generator"):
        policy.sample(valid, generator=None)  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="generator is on cpu"):
        policy.sample(valid, generator=torch.Generator(device="cpu"))
    with pytest.raises(ValueError, match="pre_tanh must have shape"):
        policy.evaluate(valid, torch.zeros((2, 3), device=_device()))
