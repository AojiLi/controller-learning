"""CUDA-native CleanRL-style PPO updates for the M7 training pipeline.

Rollouts retain fixed-width ``NEXT_STEP`` reset-only rows for exact environment accounting.  This
module first selects the valid transition indices and then performs every normalization, loss, and
metric calculation exclusively on those indices.  Minibatch order is controlled only by a
caller-owned CUDA ``torch.Generator``.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from numbers import Real
from typing import Any

import torch
from torch import Tensor

from controller_learning.rl.collector import CollectedRollout, TorchGaeResult
from controller_learning.rl.configuration import PpoAlgorithmConfig
from controller_learning.rl.policy import PpoActorCritic

_ADVANTAGE_EPSILON = 1.0e-8


def _positive_integer(value: object, *, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise ValueError(f"{name} must be a positive integer")
    return value


def _positive_learning_rate(value: Real) -> float:
    if isinstance(value, bool) or not isinstance(value, Real):
        raise TypeError("learning_rate must be a real number")
    result = float(value)
    if not math.isfinite(result) or result <= 0.0:
        raise ValueError("learning_rate must be finite and positive")
    return result


def _validate_cuda_generator(generator: Any, *, device: torch.device) -> torch.Generator:
    if not isinstance(generator, torch.Generator):
        raise TypeError("minibatch_generator must be an explicit torch.Generator")
    generator_device = torch.device(generator.device)
    if generator_device.type != "cuda":
        raise ValueError("minibatch_generator must be a CUDA generator")
    generator_index = generator_device.index
    if generator_index is None:
        generator_index = torch.cuda.current_device()
    if generator_index != device.index:
        raise ValueError(
            f"minibatch_generator is on cuda:{generator_index}, but the policy is on {device}"
        )
    return generator


def build_valid_minibatches(
    valid_transition: Tensor,
    *,
    num_minibatches: int,
    generator: torch.Generator,
) -> tuple[Tensor, ...]:
    """Shuffle valid flat indices once and partition them without padding or duplication.

    The returned batches cover every true entry in ``valid_transition`` exactly once.  When the
    valid count is not divisible by ``num_minibatches``, the first batches contain one extra index.
    """

    if not isinstance(valid_transition, Tensor):
        raise TypeError("valid_transition must be a torch.Tensor")
    if valid_transition.ndim != 2 or any(size < 1 for size in valid_transition.shape):
        raise ValueError("valid_transition must be a non-empty two-dimensional tensor")
    if valid_transition.dtype is not torch.bool:
        raise TypeError("valid_transition must use torch.bool")
    if valid_transition.device.type != "cuda":
        raise ValueError("valid_transition must be on CUDA")
    minibatch_count = _positive_integer(num_minibatches, name="num_minibatches")
    checked_generator = _validate_cuda_generator(generator, device=valid_transition.device)

    valid_indices = torch.nonzero(valid_transition.reshape(-1), as_tuple=False).squeeze(-1)
    valid_count = valid_indices.shape[0]
    if valid_count < minibatch_count:
        raise ValueError("num_minibatches cannot exceed the number of valid transitions")
    order = torch.randperm(
        valid_count,
        dtype=torch.int64,
        device=valid_transition.device,
        generator=checked_generator,
    )
    shuffled = valid_indices.index_select(0, order)
    base_size, larger_batches = divmod(valid_count, minibatch_count)
    batches: list[Tensor] = []
    offset = 0
    for batch_index in range(minibatch_count):
        size = base_size + int(batch_index < larger_batches)
        batches.append(shuffled[offset : offset + size])
        offset += size
    return tuple(batches)


def _build_compact_minibatches(
    valid_count: int,
    *,
    num_minibatches: int,
    generator: torch.Generator,
    device: torch.device,
) -> tuple[Tensor, ...]:
    """Shuffle indices into already compact valid-only tensors."""

    if valid_count < num_minibatches:
        raise ValueError("num_minibatches cannot exceed the number of valid transitions")
    order = torch.randperm(
        valid_count,
        dtype=torch.int64,
        device=device,
        generator=generator,
    )
    base_size, larger_batches = divmod(valid_count, num_minibatches)
    batches: list[Tensor] = []
    offset = 0
    for batch_index in range(num_minibatches):
        size = base_size + int(batch_index < larger_batches)
        batches.append(order[offset : offset + size])
        offset += size
    return tuple(batches)


@dataclass(frozen=True, slots=True)
class UpdateMetrics:
    """Immutable host scalars produced after one PPO update boundary."""

    learning_rate: float
    valid_samples: int
    samples_processed: int
    epochs_run: int
    epochs_completed: int
    minibatches_processed: int
    early_stopped_for_kl: bool
    policy_loss: float
    value_loss: float
    latent_entropy: float
    total_loss: float
    optimization_mean_kl: float
    post_epoch_kl: float
    clip_fraction: float
    mean_gradient_norm_before_clip: float
    max_gradient_norm_before_clip: float
    explained_variance: float


@dataclass(frozen=True, slots=True)
class _ValidPpoBatch:
    observations: Tensor
    pre_tanh_actions: Tensor
    old_log_prob: Tensor
    old_values: Tensor
    advantages: Tensor
    returns: Tensor

    @property
    def valid_samples(self) -> int:
        return self.observations.shape[0]


def _require_tensor(
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
        raise ValueError(f"{name} is on {value.device}, but the policy is on {device}")
    return value


def _select_valid_batch(
    rollout: CollectedRollout,
    gae: TorchGaeResult,
    *,
    policy: PpoActorCritic,
) -> _ValidPpoBatch:
    if not isinstance(rollout, CollectedRollout):
        raise TypeError("rollout must be a CollectedRollout")
    if not isinstance(gae, TorchGaeResult):
        raise TypeError("gae must be a TorchGaeResult")

    rollout_steps, num_envs = rollout.shape
    time_world = (rollout_steps, num_envs)
    device = policy.device
    dtype = policy.dtype
    observations = _require_tensor(
        rollout.observations,
        name="rollout.observations",
        shape=(*time_world, policy.observation_dim),
        dtype=dtype,
        device=device,
    )
    pre_tanh_actions = _require_tensor(
        rollout.pre_tanh_actions,
        name="rollout.pre_tanh_actions",
        shape=(*time_world, policy.action_dim),
        dtype=dtype,
        device=device,
    )
    old_log_prob = _require_tensor(
        rollout.old_log_prob,
        name="rollout.old_log_prob",
        shape=time_world,
        dtype=dtype,
        device=device,
    )
    values = _require_tensor(
        rollout.values,
        name="rollout.values",
        shape=(rollout_steps + 1, num_envs),
        dtype=dtype,
        device=device,
    )
    valid_transition = _require_tensor(
        rollout.valid_transition,
        name="rollout.valid_transition",
        shape=time_world,
        dtype=torch.bool,
        device=device,
    )
    reset_only = _require_tensor(
        rollout.reset_only,
        name="rollout.reset_only",
        shape=time_world,
        dtype=torch.bool,
        device=device,
    )
    advantages = _require_tensor(
        gae.advantages,
        name="gae.advantages",
        shape=time_world,
        dtype=dtype,
        device=device,
    )
    returns = _require_tensor(
        gae.returns,
        name="gae.returns",
        shape=time_world,
        dtype=dtype,
        device=device,
    )
    _require_tensor(
        gae.temporal_difference,
        name="gae.temporal_difference",
        shape=time_world,
        dtype=dtype,
        device=device,
    )
    valid_indices = torch.nonzero(valid_transition.reshape(-1), as_tuple=False).squeeze(-1)
    valid_count = valid_indices.shape[0]
    if valid_count < 1:
        raise ValueError("rollout must contain at least one valid transition")
    if rollout.counts.valid_transitions != valid_count:
        raise ValueError("rollout counts do not match valid_transition")

    selected = _ValidPpoBatch(
        observations=observations.reshape(-1, policy.observation_dim)
        .index_select(0, valid_indices)
        .detach(),
        pre_tanh_actions=pre_tanh_actions.reshape(-1, policy.action_dim)
        .index_select(0, valid_indices)
        .detach(),
        old_log_prob=old_log_prob.reshape(-1).index_select(0, valid_indices).detach(),
        old_values=values[:-1].reshape(-1).index_select(0, valid_indices).detach(),
        advantages=advantages.reshape(-1).index_select(0, valid_indices).detach(),
        returns=returns.reshape(-1).index_select(0, valid_indices).detach(),
    )
    checks = torch.stack(
        (
            torch.all(valid_transition == torch.logical_not(reset_only)),
            *(
                torch.all(torch.isfinite(tensor))
                for tensor in (
                    selected.observations,
                    selected.pre_tanh_actions,
                    selected.old_log_prob,
                    selected.old_values,
                    selected.advantages,
                    selected.returns,
                )
            ),
        )
    ).to(device="cpu")
    valid_checks = checks.tolist()
    if not valid_checks[0]:
        raise ValueError("valid_transition must be the exact complement of reset_only")
    if not all(valid_checks[1:]):
        raise FloatingPointError("valid PPO rollout or GAE values contain a non-finite value")
    return selected


def _parameters_are_finite(policy: PpoActorCritic) -> bool:
    return bool(torch.stack(tuple(torch.all(torch.isfinite(p)) for p in policy.parameters())).all())


class PpoUpdater:
    """Stateful Adam optimizer with explicitly seeded minibatch ordering."""

    def __init__(self, policy: PpoActorCritic, config: PpoAlgorithmConfig) -> None:
        if not isinstance(policy, PpoActorCritic):
            raise TypeError("policy must be a PpoActorCritic")
        if not isinstance(config, PpoAlgorithmConfig):
            raise TypeError("config must be a PpoAlgorithmConfig")
        if policy.device.type != "cuda":
            raise ValueError("PpoUpdater requires a CUDA policy")
        if not _parameters_are_finite(policy):
            raise FloatingPointError("policy parameters contain a non-finite value")
        self.policy = policy
        self.config = config
        self.optimizer = torch.optim.Adam(
            policy.parameters(),
            lr=config.learning_rate,
            eps=config.adam_epsilon,
        )

    def update(
        self,
        rollout: CollectedRollout,
        *,
        learning_rate: Real,
        minibatch_generator: torch.Generator,
    ) -> UpdateMetrics:
        """Apply configured PPO epochs and return one host-safe metric snapshot.

        ``learning_rate`` is supplied by the training loop so schedule state is explicit and can be
        checkpointed independently. GAE is derived internally from this exact rollout, and the KL
        stop is checked from a fresh full-valid-batch evaluation after each completed epoch.
        """

        rate = _positive_learning_rate(learning_rate)
        checked_generator = _validate_cuda_generator(
            minibatch_generator,
            device=self.policy.device,
        )
        gae = rollout.generalized_advantage_estimate(
            gamma=self.config.discount_factor,
            gae_lambda=self.config.gae_lambda,
        )
        batch = _select_valid_batch(rollout, gae, policy=self.policy)
        if batch.valid_samples < self.config.num_minibatches:
            raise ValueError("num_minibatches cannot exceed the number of valid transitions")
        if not _parameters_are_finite(self.policy):
            raise FloatingPointError("policy parameters contain a non-finite value")
        for parameter_group in self.optimizer.param_groups:
            parameter_group["lr"] = rate

        device = self.policy.device
        statistic_sums = torch.zeros(8, dtype=self.policy.dtype, device=device)
        maximum_gradient_norm = torch.zeros((), dtype=self.policy.dtype, device=device)
        samples_processed = 0
        minibatches_processed = 0
        epochs_run = 0
        epochs_completed = 0
        early_stopped = False
        post_epoch_kl = torch.zeros((), dtype=self.policy.dtype, device=device)

        for epoch_index in range(self.config.update_epochs):
            epochs_run += 1
            compact_minibatches = _build_compact_minibatches(
                batch.valid_samples,
                num_minibatches=self.config.num_minibatches,
                generator=checked_generator,
                device=device,
            )

            for compact_indices in compact_minibatches:
                minibatch_size = compact_indices.shape[0]
                observations = batch.observations.index_select(0, compact_indices)
                pre_tanh_actions = batch.pre_tanh_actions.index_select(0, compact_indices)
                old_log_prob = batch.old_log_prob.index_select(0, compact_indices)
                old_values = batch.old_values.index_select(0, compact_indices)
                advantages = batch.advantages.index_select(0, compact_indices)
                returns = batch.returns.index_select(0, compact_indices)

                if self.config.normalize_advantages:
                    advantage_mean = advantages.mean()
                    advantage_variance = advantages.var(unbiased=False)
                    advantages = (advantages - advantage_mean) * torch.rsqrt(
                        advantage_variance + _ADVANTAGE_EPSILON
                    )

                evaluated = self.policy.evaluate(observations, pre_tanh_actions)
                log_ratio = evaluated.log_prob - old_log_prob
                ratio = torch.exp(log_ratio)
                approximate_kl = ((ratio - 1.0) - log_ratio).mean()
                clip_fraction = (
                    (torch.abs(ratio - 1.0) > self.config.clip_coefficient)
                    .to(dtype=self.policy.dtype)
                    .mean()
                )

                unclipped_policy_loss = -advantages * ratio
                clipped_policy_loss = -advantages * torch.clamp(
                    ratio,
                    1.0 - self.config.clip_coefficient,
                    1.0 + self.config.clip_coefficient,
                )
                policy_loss = torch.maximum(
                    unclipped_policy_loss,
                    clipped_policy_loss,
                ).mean()

                if self.config.clip_value_loss:
                    value_delta = evaluated.value - old_values
                    clipped_value = old_values + torch.clamp(
                        value_delta,
                        -self.config.clip_coefficient,
                        self.config.clip_coefficient,
                    )
                    value_loss = (
                        0.5
                        * torch.maximum(
                            (evaluated.value - returns).square(),
                            (clipped_value - returns).square(),
                        ).mean()
                    )
                else:
                    value_loss = 0.5 * (evaluated.value - returns).square().mean()
                latent_entropy = evaluated.latent_entropy.mean()
                total_loss = (
                    policy_loss
                    - self.config.entropy_coefficient * latent_entropy
                    + self.config.value_coefficient * value_loss
                )

                pre_step_statistics = torch.stack(
                    (
                        policy_loss,
                        value_loss,
                        latent_entropy,
                        total_loss,
                        approximate_kl,
                        clip_fraction,
                    )
                )
                self.optimizer.zero_grad(set_to_none=True)
                total_loss.backward()
                try:
                    gradient_norm = torch.nn.utils.clip_grad_norm_(
                        self.policy.parameters(),
                        self.config.max_gradient_norm,
                        error_if_nonfinite=True,
                    )
                except RuntimeError as error:
                    self.optimizer.zero_grad(set_to_none=True)
                    raise FloatingPointError("PPO gradient norm is non-finite") from error
                self.optimizer.step()
                self.policy.project_log_std_()

                weight = float(minibatch_size)
                statistic_sums[:6] += pre_step_statistics.detach() * weight
                statistic_sums[6] += gradient_norm.detach() * weight
                statistic_sums[7] += weight
                maximum_gradient_norm = torch.maximum(
                    maximum_gradient_norm,
                    gradient_norm.detach(),
                )
                samples_processed += minibatch_size
                minibatches_processed += 1

            epochs_completed += 1
            with torch.no_grad():
                post_epoch_evaluation = self.policy.evaluate(
                    batch.observations,
                    batch.pre_tanh_actions,
                )
                post_epoch_log_ratio = post_epoch_evaluation.log_prob - batch.old_log_prob
                post_epoch_ratio = torch.exp(post_epoch_log_ratio)
                post_epoch_kl = ((post_epoch_ratio - 1.0) - post_epoch_log_ratio).mean()
            if epoch_index + 1 < self.config.update_epochs and bool(
                post_epoch_kl > self.config.target_kl
            ):
                early_stopped = True
                break

        valid_returns = batch.returns
        return_variance = valid_returns.var(unbiased=False)
        residual_variance = (valid_returns - batch.old_values).var(unbiased=False)
        explained_variance = torch.where(
            return_variance > torch.finfo(self.policy.dtype).eps,
            1.0 - residual_variance / return_variance,
            torch.zeros_like(return_variance),
        )
        denominator = statistic_sums[7]
        parameter_finite = torch.stack(
            tuple(torch.all(torch.isfinite(parameter)) for parameter in self.policy.parameters())
        ).all()
        metric_statistics = torch.cat(
            (
                statistic_sums[:7] / denominator,
                maximum_gradient_norm.reshape(1),
                explained_variance.reshape(1),
                post_epoch_kl.reshape(1),
                parameter_finite.to(dtype=self.policy.dtype).reshape(1),
            )
        )
        host_statistics = metric_statistics.to(device="cpu")
        (
            policy_loss_value,
            value_loss_value,
            latent_entropy_value,
            total_loss_value,
            optimization_mean_kl_value,
            clip_fraction_value,
            mean_gradient_norm_value,
            max_gradient_norm_value,
            explained_variance_value,
            post_epoch_kl_value,
            parameter_finite_value,
        ) = host_statistics.tolist()
        finite_statistics = (
            policy_loss_value,
            value_loss_value,
            latent_entropy_value,
            total_loss_value,
            optimization_mean_kl_value,
            clip_fraction_value,
            mean_gradient_norm_value,
            max_gradient_norm_value,
            explained_variance_value,
            post_epoch_kl_value,
        )
        if parameter_finite_value != 1.0 or not all(map(math.isfinite, finite_statistics)):
            raise FloatingPointError(
                "PPO update produced a non-finite loss, gradient, or parameter"
            )
        return UpdateMetrics(
            learning_rate=rate,
            valid_samples=batch.valid_samples,
            samples_processed=samples_processed,
            epochs_run=epochs_run,
            epochs_completed=epochs_completed,
            minibatches_processed=minibatches_processed,
            early_stopped_for_kl=early_stopped,
            policy_loss=float(policy_loss_value),
            value_loss=float(value_loss_value),
            latent_entropy=float(latent_entropy_value),
            total_loss=float(total_loss_value),
            optimization_mean_kl=float(optimization_mean_kl_value),
            post_epoch_kl=float(post_epoch_kl_value),
            clip_fraction=float(clip_fraction_value),
            mean_gradient_norm_before_clip=float(mean_gradient_norm_value),
            max_gradient_norm_before_clip=float(max_gradient_norm_value),
            explained_variance=float(explained_variance_value),
        )


__all__ = [
    "PpoUpdater",
    "UpdateMetrics",
    "build_valid_minibatches",
]
