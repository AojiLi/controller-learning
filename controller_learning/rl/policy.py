"""PyTorch actor/critic used by the M7 PPO training pipeline.

The policy samples in an unconstrained latent space and applies a tanh transform followed by the
exact affine map into the physical action bounds.  Rollouts retain the pre-tanh latent so PPO can
re-evaluate the same action without an unstable inverse tanh at a saturated action.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

import torch
import torch.nn.functional as functional
from torch import Tensor, nn

from controller_learning.rl.configuration import (
    PPO_FORMAL_HIDDEN_SIZES,
    PPO_MAX_LOG_STD,
    PPO_MIN_LOG_STD,
)

PPO_ACTION_DIM = 2
PPO_HIDDEN_SIZES = PPO_FORMAL_HIDDEN_SIZES

_LOG_TWO = math.log(2.0)
_LOG_TWO_PI = math.log(2.0 * math.pi)
_NORMAL_ENTROPY_CONSTANT = 0.5 * math.log(2.0 * math.pi * math.e)


@dataclass(frozen=True, slots=True)
class PolicySample:
    """One reparameterized stochastic policy result."""

    action: Tensor
    pre_tanh: Tensor
    mean: Tensor
    log_prob: Tensor
    latent_entropy: Tensor
    value: Tensor


@dataclass(frozen=True, slots=True)
class PolicyEvaluation:
    """Current-policy statistics for a previously stored pre-tanh action."""

    action: Tensor
    pre_tanh: Tensor
    mean: Tensor
    log_prob: Tensor
    latent_entropy: Tensor
    value: Tensor


@dataclass(frozen=True, slots=True)
class DeterministicPolicyAction:
    """Mean action and value estimate used by deterministic inference."""

    action: Tensor
    pre_tanh: Tensor
    value: Tensor


def _positive_dimension(value: object, *, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise ValueError(f"{name} must be a positive integer")
    return value


def _policy_seed(value: object) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or not 0 <= value < 2**32:
        raise ValueError("policy_seed must be an integer in the uint32 range")
    return value


def _initial_log_std(value: object) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError("initial_log_std must be a real number")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError("initial_log_std must be finite")
    if not PPO_MIN_LOG_STD <= result <= PPO_MAX_LOG_STD:
        raise ValueError(f"initial_log_std must be in [{PPO_MIN_LOG_STD}, {PPO_MAX_LOG_STD}]")
    return result


def _hidden_sizes(value: object) -> tuple[int, int]:
    if isinstance(value, (str, bytes)) or not isinstance(value, Sequence):
        raise TypeError("hidden_sizes must be a sequence")
    sizes = tuple(value)
    if any(isinstance(size, bool) or not isinstance(size, int) for size in sizes):
        raise TypeError("hidden_sizes must contain integers")
    if sizes != PPO_HIDDEN_SIZES:
        raise ValueError(f"hidden_sizes must be exactly {PPO_HIDDEN_SIZES}")
    return PPO_HIDDEN_SIZES


def _torch_device(value: Any) -> torch.device:
    if value is None:
        raise ValueError("device must be explicit")
    try:
        device = torch.device(value)
    except (RuntimeError, TypeError, ValueError) as error:
        raise ValueError(f"invalid Torch device {value!r}") from error
    if device.type == "cuda" and not torch.cuda.is_available():
        raise ValueError("CUDA was requested but is not available")
    return device


def _bound_tensor(value: Any, *, name: str) -> Tensor:
    try:
        result = torch.as_tensor(value, dtype=torch.float64, device="cpu")
    except (RuntimeError, TypeError, ValueError) as error:
        raise TypeError(f"{name} must be convertible to a numerical tensor") from error
    if result.shape != (PPO_ACTION_DIM,):
        raise ValueError(f"{name} must have shape {(PPO_ACTION_DIM,)}, got {tuple(result.shape)}")
    if not bool(torch.all(torch.isfinite(result))):
        raise ValueError(f"{name} must contain only finite values")
    result = result.to(dtype=torch.float32)
    if not bool(torch.all(torch.isfinite(result))):
        raise ValueError(f"{name} must remain finite in torch.float32")
    return result


def _initialize_linear(layer: nn.Linear, *, standard_deviation: float) -> None:
    nn.init.orthogonal_(layer.weight, standard_deviation)
    nn.init.constant_(layer.bias, 0.0)


class PpoActorCritic(nn.Module):
    """Shared-trunk MLP actor/critic for two physical race-car actions.

    Construction is reproducible from ``policy_seed`` and restores the ambient CPU RNG state.
    Modules are initialized on CPU and only then transferred to ``device``, so initialization does
    not consume an ambient CUDA generator either.
    """

    def __init__(
        self,
        observation_dim: int,
        *,
        action_low: Any,
        action_high: Any,
        policy_seed: int,
        initial_log_std: float,
        hidden_sizes: Sequence[int] = PPO_HIDDEN_SIZES,
        device: Any,
        dtype: torch.dtype = torch.float32,
    ) -> None:
        super().__init__()
        self.observation_dim = _positive_dimension(observation_dim, name="observation_dim")
        self.hidden_sizes = _hidden_sizes(hidden_sizes)
        self.action_dim = PPO_ACTION_DIM
        self.policy_seed = _policy_seed(policy_seed)
        self.initial_log_std = _initial_log_std(initial_log_std)
        selected_device = _torch_device(device)
        if dtype is not torch.float32:
            raise ValueError("PpoActorCritic requires torch.float32 parameters")

        low = _bound_tensor(action_low, name="action_low")
        high = _bound_tensor(action_high, name="action_high")
        if not bool(torch.all(high > low)):
            raise ValueError("action_high must be greater than action_low in every dimension")
        scale = (high - low) * 0.5
        bias = (high + low) * 0.5
        if not bool(torch.all(scale > 0.0)):
            raise ValueError("physical action scale must remain positive in torch.float32")

        # Linear constructors and orthogonal initialization both consume the default CPU generator.
        # fork_rng restores that generator after constructing this model.  Seeding the CPU default
        # generator directly avoids modifying any CUDA generator.
        with torch.random.fork_rng(devices=[], enabled=True):
            torch.random.default_generator.manual_seed(self.policy_seed)
            self.trunk = nn.Sequential(
                nn.Linear(self.observation_dim, self.hidden_sizes[0]),
                nn.Tanh(),
                nn.Linear(self.hidden_sizes[0], self.hidden_sizes[1]),
                nn.Tanh(),
            )
            self.actor_mean = nn.Linear(self.hidden_sizes[1], self.action_dim)
            self.critic = nn.Linear(self.hidden_sizes[1], 1)
            self.log_std = nn.Parameter(
                torch.full(
                    (self.action_dim,),
                    self.initial_log_std,
                    dtype=torch.float32,
                )
            )

            _initialize_linear(self.trunk[0], standard_deviation=math.sqrt(2.0))
            _initialize_linear(self.trunk[2], standard_deviation=math.sqrt(2.0))
            _initialize_linear(self.actor_mean, standard_deviation=0.01)
            _initialize_linear(self.critic, standard_deviation=1.0)

        self.register_buffer("action_low", low)
        self.register_buffer("action_high", high)
        self.register_buffer("action_scale", scale)
        self.register_buffer("action_bias", bias)
        self.register_buffer("log_action_scale", torch.log(scale))
        self.to(device=selected_device, dtype=dtype)

    @property
    def device(self) -> torch.device:
        """Return the device shared by all parameters and buffers."""

        return self.log_std.device

    @property
    def dtype(self) -> torch.dtype:
        """Return the floating-point dtype used by the policy."""

        return self.log_std.dtype

    def _validate_observations(self, observations: Tensor) -> None:
        if not isinstance(observations, Tensor):
            raise TypeError("observations must be a torch.Tensor")
        if observations.ndim < 1 or observations.shape[-1] != self.observation_dim:
            raise ValueError(
                "observations must end with the fixed observation dimension "
                f"{self.observation_dim}, got {tuple(observations.shape)}"
            )
        if observations.device != self.device:
            raise ValueError(
                f"observations are on {observations.device}, but the policy is on {self.device}"
            )
        if observations.dtype is not self.dtype:
            raise TypeError(f"observations must use {self.dtype}, got {observations.dtype}")

    def _validate_pre_tanh(self, pre_tanh: Tensor, observations: Tensor) -> None:
        if not isinstance(pre_tanh, Tensor):
            raise TypeError("pre_tanh must be a torch.Tensor")
        expected_shape = (*observations.shape[:-1], self.action_dim)
        if pre_tanh.shape != expected_shape:
            raise ValueError(
                f"pre_tanh must have shape {expected_shape}, got {tuple(pre_tanh.shape)}"
            )
        if pre_tanh.device != self.device:
            raise ValueError(
                f"pre_tanh is on {pre_tanh.device}, but the policy is on {self.device}"
            )
        if pre_tanh.dtype is not self.dtype:
            raise TypeError(f"pre_tanh must use {self.dtype}, got {pre_tanh.dtype}")

    def _validate_generator(self, generator: torch.Generator) -> None:
        if not isinstance(generator, torch.Generator):
            raise TypeError("generator must be an explicit torch.Generator")
        generator_device = torch.device(generator.device)
        model_device = self.device
        if generator_device.type != model_device.type:
            raise ValueError(
                f"generator is on {generator_device}, but the policy is on {model_device}"
            )
        if model_device.type == "cuda":
            generator_index = generator_device.index
            if generator_index is None:
                generator_index = torch.cuda.current_device()
            if generator_index != model_device.index:
                raise ValueError(
                    f"generator is on cuda:{generator_index}, but the policy is on {model_device}"
                )

    def _network(self, observations: Tensor) -> tuple[Tensor, Tensor]:
        self._validate_observations(observations)
        features = self.trunk(observations)
        return self.actor_mean(features), self.critic(features).squeeze(-1)

    def _effective_log_std(self, mean: Tensor) -> Tensor:
        """Return the finite exploration range used by sampling and PPO evaluation."""

        return torch.clamp(
            self.log_std,
            min=PPO_MIN_LOG_STD,
            max=PPO_MAX_LOG_STD,
        ).expand_as(mean)

    @torch.no_grad()
    def project_log_std_(self) -> None:
        """Project the trainable raw parameter after every optimizer step.

        The forward clamp remains a checkpoint/input safety boundary. Keeping the raw parameter
        inside the same interval prevents an optimizer overshoot from leaving it in the clamp's
        zero-gradient region.
        """

        self.log_std.clamp_(min=PPO_MIN_LOG_STD, max=PPO_MAX_LOG_STD)

    def _action(self, pre_tanh: Tensor) -> Tensor:
        return self.action_bias + self.action_scale * torch.tanh(pre_tanh)

    def _entropy(self, log_std: Tensor) -> Tensor:
        # This is the exact entropy of the latent Gaussian.  It is the conventional, low-variance
        # entropy regularizer used by CleanRL-style continuous-action PPO.
        return torch.sum(log_std + _NORMAL_ENTROPY_CONSTANT, dim=-1)

    def _log_prob(self, mean: Tensor, log_std: Tensor, pre_tanh: Tensor) -> Tensor:
        inverse_std = torch.exp(-log_std)
        standardized = (pre_tanh - mean) * inverse_std
        normal_log_prob = -0.5 * standardized.square() - log_std - 0.5 * _LOG_TWO_PI
        # Stable log(1 - tanh(z)^2), including saturated finite z where direct subtraction rounds
        # to zero.  The affine action-scale Jacobian is required for physical-unit log density.
        log_tanh_jacobian = 2.0 * (_LOG_TWO - pre_tanh - functional.softplus(-2.0 * pre_tanh))
        return torch.sum(
            normal_log_prob - log_tanh_jacobian - self.log_action_scale,
            dim=-1,
        )

    def sample(self, observations: Tensor, *, generator: torch.Generator) -> PolicySample:
        """Sample actions using only the caller-owned explicit random generator."""

        self._validate_generator(generator)
        mean, value = self._network(observations)
        log_std = self._effective_log_std(mean)
        noise = torch.randn(
            mean.shape,
            dtype=mean.dtype,
            device=mean.device,
            generator=generator,
        )
        pre_tanh = mean + torch.exp(log_std) * noise
        return PolicySample(
            action=self._action(pre_tanh),
            pre_tanh=pre_tanh,
            mean=mean,
            log_prob=self._log_prob(mean, log_std, pre_tanh),
            latent_entropy=self._entropy(log_std),
            value=value,
        )

    def evaluate(self, observations: Tensor, pre_tanh: Tensor) -> PolicyEvaluation:
        """Evaluate a stored latent under the current actor and critic parameters."""

        self._validate_observations(observations)
        self._validate_pre_tanh(pre_tanh, observations)
        mean, value = self._network(observations)
        log_std = self._effective_log_std(mean)
        return PolicyEvaluation(
            action=self._action(pre_tanh),
            pre_tanh=pre_tanh,
            mean=mean,
            log_prob=self._log_prob(mean, log_std, pre_tanh),
            latent_entropy=self._entropy(log_std),
            value=value,
        )

    def deterministic(self, observations: Tensor) -> DeterministicPolicyAction:
        """Return the affine-mapped tanh of the latent Gaussian mean."""

        mean, value = self._network(observations)
        return DeterministicPolicyAction(
            action=self._action(mean),
            pre_tanh=mean,
            value=value,
        )

    def value(self, observations: Tensor) -> Tensor:
        """Return only the scalar critic estimate for each leading observation index."""

        self._validate_observations(observations)
        return self.critic(self.trunk(observations)).squeeze(-1)


__all__ = [
    "PPO_ACTION_DIM",
    "PPO_HIDDEN_SIZES",
    "DeterministicPolicyAction",
    "PolicyEvaluation",
    "PolicySample",
    "PpoActorCritic",
]
