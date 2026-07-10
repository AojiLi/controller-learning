"""Strict, immutable configuration for the v0.1 PPO training pipeline."""

from __future__ import annotations

import math
import tomllib
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Any, Literal

from controller_learning.config import ConfigError

PPO_CONFIG_SCHEMA_VERSION = 1
PPO_FORMAL_BENCHMARK_VERSION = "0.1"
PPO_FORMAL_LEVEL_ID = 1
PPO_FORMAL_BACKEND = "mjx_warp"
PPO_FORMAL_NUM_ENVS = 1024
PPO_FORMAL_TRAIN_CACHE = ".track-cache/v0.1/train_pool.npz"
PPO_FORMAL_PREVIEW_POINTS = 16
PPO_FORMAL_MAX_SPEED_MPS = 15.0
PPO_FORMAL_HIDDEN_SIZES = (128, 128)
PPO_MIN_LOG_STD = -5.0
PPO_MAX_LOG_STD = 2.0
_FLOAT32_MAX = 3.4028234663852886e38
_FLOAT32_MIN_NORMAL = 1.1754943508222875e-38


def _positive_integer(value: object, *, field: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise ConfigError(f"{field} must be a positive integer")
    return value


def _uint32_integer(value: object, *, field: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or not 0 <= value < 2**32:
        raise ConfigError(f"{field} must be an integer in the uint32 range")
    return value


def _finite_positive(value: object, *, field: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ConfigError(f"{field} must be a number")
    result = float(value)
    if not math.isfinite(result) or result <= 0.0:
        raise ConfigError(f"{field} must be finite and positive")
    if result > _FLOAT32_MAX:
        raise ConfigError(f"{field} must fit in float32")
    if result < _FLOAT32_MIN_NORMAL:
        raise ConfigError(f"{field} must remain positive in normal float32")
    return result


def _finite_nonnegative(value: object, *, field: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ConfigError(f"{field} must be a number")
    result = float(value)
    if not math.isfinite(result) or result < 0.0:
        raise ConfigError(f"{field} must be finite and non-negative")
    if result > _FLOAT32_MAX:
        raise ConfigError(f"{field} must fit in float32")
    if 0.0 < result < _FLOAT32_MIN_NORMAL:
        raise ConfigError(f"{field} must remain positive in normal float32")
    return result


def _finite(value: object, *, field: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ConfigError(f"{field} must be a number")
    result = float(value)
    if not math.isfinite(result):
        raise ConfigError(f"{field} must be finite")
    if abs(result) > _FLOAT32_MAX:
        raise ConfigError(f"{field} must fit in float32")
    if 0.0 < abs(result) < _FLOAT32_MIN_NORMAL:
        raise ConfigError(f"{field} must remain nonzero in normal float32")
    return result


def _probability(value: object, *, field: str, include_one: bool = True) -> float:
    result = _finite_positive(value, field=field)
    valid = result <= 1.0 if include_one else result < 1.0
    if not valid:
        interval = "(0, 1]" if include_one else "(0, 1)"
        raise ConfigError(f"{field} must be in {interval}")
    return result


def _boolean(value: object, *, field: str) -> bool:
    if type(value) is not bool:
        raise ConfigError(f"{field} must be a boolean")
    return value


def _nonempty_string(value: object, *, field: str) -> str:
    if not isinstance(value, str) or not value:
        raise ConfigError(f"{field} must be a non-empty string")
    return value


def _safe_relative_path(value: object, *, field: str, suffix: str | None = None) -> str:
    text = _nonempty_string(value, field=field)
    path = PurePosixPath(text)
    if "\\" in text or path.is_absolute() or path == PurePosixPath(".") or ".." in path.parts:
        raise ConfigError(f"{field} must be a safe relative POSIX path")
    if suffix is not None and path.suffix != suffix:
        raise ConfigError(f"{field} must use the {suffix} suffix")
    return path.as_posix()


@dataclass(frozen=True, slots=True)
class PpoEnvironmentConfig:
    """Official Challenge and Train-pool identity used by PPO."""

    benchmark_version: str
    level_id: int
    backend: Literal["mjx_warp"]
    num_envs: int
    environment_seed: int
    train_cache: str

    def __post_init__(self) -> None:
        benchmark_version = _nonempty_string(
            self.benchmark_version,
            field="ppo.environment.benchmark_version",
        )
        if benchmark_version != PPO_FORMAL_BENCHMARK_VERSION:
            raise ConfigError(
                "ppo.environment.benchmark_version must be '0.1' for the v0.1 pipeline"
            )
        if type(self.level_id) is not int or self.level_id != PPO_FORMAL_LEVEL_ID:
            raise ConfigError("ppo.environment.level_id must be 1")
        if self.backend != PPO_FORMAL_BACKEND:
            raise ConfigError("ppo.environment.backend must be 'mjx_warp'")
        num_envs = _positive_integer(self.num_envs, field="ppo.environment.num_envs")
        if num_envs != PPO_FORMAL_NUM_ENVS:
            raise ConfigError("ppo.environment.num_envs must be 1024")
        environment_seed = _uint32_integer(
            self.environment_seed,
            field="ppo.environment.environment_seed",
        )
        train_cache = _safe_relative_path(
            self.train_cache,
            field="ppo.environment.train_cache",
            suffix=".npz",
        )
        if train_cache != PPO_FORMAL_TRAIN_CACHE:
            raise ConfigError(
                "ppo.environment.train_cache must identify the official Train-only cache "
                f"{PPO_FORMAL_TRAIN_CACHE!r}"
            )
        object.__setattr__(self, "benchmark_version", benchmark_version)
        object.__setattr__(self, "num_envs", num_envs)
        object.__setattr__(self, "environment_seed", environment_seed)
        object.__setattr__(self, "train_cache", train_cache)


@dataclass(frozen=True, slots=True)
class PpoObservationConfig:
    """Public-observation compression parameters; no Challenge internals are allowed."""

    preview_points: int
    preview_distance_m: float
    max_speed_mps: float

    def __post_init__(self) -> None:
        preview_points = _positive_integer(
            self.preview_points,
            field="ppo.observation.preview_points",
        )
        if preview_points < 2:
            raise ConfigError("ppo.observation.preview_points must be at least two")
        if preview_points != PPO_FORMAL_PREVIEW_POINTS:
            raise ConfigError(
                f"ppo.observation.preview_points must be {PPO_FORMAL_PREVIEW_POINTS} "
                "for feature schema v1"
            )
        object.__setattr__(self, "preview_points", preview_points)
        object.__setattr__(
            self,
            "preview_distance_m",
            _finite_positive(
                self.preview_distance_m,
                field="ppo.observation.preview_distance_m",
            ),
        )
        max_speed_mps = _finite_positive(
            self.max_speed_mps,
            field="ppo.observation.max_speed_mps",
        )
        if max_speed_mps != PPO_FORMAL_MAX_SPEED_MPS:
            raise ConfigError(
                f"ppo.observation.max_speed_mps must be {PPO_FORMAL_MAX_SPEED_MPS} "
                "for the formal vehicle"
            )
        object.__setattr__(
            self,
            "max_speed_mps",
            max_speed_mps,
        )


@dataclass(frozen=True, slots=True)
class PpoRewardConfig:
    """Public reward-shaping weights layered over the official environment."""

    progress_scale: float
    success_bonus: float
    offtrack_invalid_penalty: float
    lateral_error_weight: float
    heading_error_weight: float
    reverse_speed_weight: float
    action_change_weight: float

    def __post_init__(self) -> None:
        for field in ("progress_scale", "success_bonus", "offtrack_invalid_penalty"):
            object.__setattr__(
                self,
                field,
                _finite_positive(getattr(self, field), field=f"ppo.reward.{field}"),
            )
        for field in (
            "lateral_error_weight",
            "heading_error_weight",
            "reverse_speed_weight",
            "action_change_weight",
        ):
            object.__setattr__(
                self,
                field,
                _finite_nonnegative(getattr(self, field), field=f"ppo.reward.{field}"),
            )


@dataclass(frozen=True, slots=True)
class PpoRolloutConfig:
    """Fixed vector-step budget and rollout length."""

    steps_per_update: int
    total_vector_steps: int

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "steps_per_update",
            _positive_integer(
                self.steps_per_update,
                field="ppo.rollout.steps_per_update",
            ),
        )
        object.__setattr__(
            self,
            "total_vector_steps",
            _positive_integer(
                self.total_vector_steps,
                field="ppo.rollout.total_vector_steps",
            ),
        )


@dataclass(frozen=True, slots=True)
class PpoAlgorithmConfig:
    """CleanRL-style PPO optimizer and actor/critic architecture settings."""

    hidden_sizes: tuple[int, int]
    policy_seed: int
    minibatch_seed: int
    initial_log_std: float
    learning_rate: float
    adam_epsilon: float
    anneal_learning_rate: bool
    discount_factor: float
    gae_lambda: float
    num_minibatches: int
    update_epochs: int
    normalize_advantages: bool
    clip_coefficient: float
    clip_value_loss: bool
    entropy_coefficient: float
    value_coefficient: float
    max_gradient_norm: float
    target_kl: float

    def __post_init__(self) -> None:
        if (
            not isinstance(self.hidden_sizes, tuple)
            or len(self.hidden_sizes) != 2
            or any(
                isinstance(size, bool) or not isinstance(size, int) or size < 1
                for size in self.hidden_sizes
            )
        ):
            raise ConfigError("ppo.ppo.hidden_sizes must contain exactly two positive integers")
        if self.hidden_sizes != PPO_FORMAL_HIDDEN_SIZES:
            raise ConfigError(
                f"ppo.ppo.hidden_sizes must be {PPO_FORMAL_HIDDEN_SIZES} for policy schema v1"
            )
        object.__setattr__(
            self,
            "policy_seed",
            _uint32_integer(self.policy_seed, field="ppo.ppo.policy_seed"),
        )
        object.__setattr__(
            self,
            "minibatch_seed",
            _uint32_integer(self.minibatch_seed, field="ppo.ppo.minibatch_seed"),
        )
        object.__setattr__(
            self,
            "initial_log_std",
            _finite(self.initial_log_std, field="ppo.ppo.initial_log_std"),
        )
        if not PPO_MIN_LOG_STD <= self.initial_log_std <= PPO_MAX_LOG_STD:
            raise ConfigError(
                f"ppo.ppo.initial_log_std must be in [{PPO_MIN_LOG_STD}, {PPO_MAX_LOG_STD}]"
            )
        object.__setattr__(
            self,
            "learning_rate",
            _finite_positive(self.learning_rate, field="ppo.ppo.learning_rate"),
        )
        object.__setattr__(
            self,
            "adam_epsilon",
            _finite_positive(self.adam_epsilon, field="ppo.ppo.adam_epsilon"),
        )
        object.__setattr__(
            self,
            "anneal_learning_rate",
            _boolean(self.anneal_learning_rate, field="ppo.ppo.anneal_learning_rate"),
        )
        object.__setattr__(
            self,
            "discount_factor",
            _probability(self.discount_factor, field="ppo.ppo.discount_factor"),
        )
        object.__setattr__(
            self,
            "gae_lambda",
            _probability(self.gae_lambda, field="ppo.ppo.gae_lambda"),
        )
        object.__setattr__(
            self,
            "num_minibatches",
            _positive_integer(self.num_minibatches, field="ppo.ppo.num_minibatches"),
        )
        object.__setattr__(
            self,
            "update_epochs",
            _positive_integer(self.update_epochs, field="ppo.ppo.update_epochs"),
        )
        object.__setattr__(
            self,
            "normalize_advantages",
            _boolean(self.normalize_advantages, field="ppo.ppo.normalize_advantages"),
        )
        object.__setattr__(
            self,
            "clip_coefficient",
            _probability(
                self.clip_coefficient,
                field="ppo.ppo.clip_coefficient",
                include_one=False,
            ),
        )
        object.__setattr__(
            self,
            "clip_value_loss",
            _boolean(self.clip_value_loss, field="ppo.ppo.clip_value_loss"),
        )
        object.__setattr__(
            self,
            "entropy_coefficient",
            _finite_nonnegative(
                self.entropy_coefficient,
                field="ppo.ppo.entropy_coefficient",
            ),
        )
        object.__setattr__(
            self,
            "value_coefficient",
            _finite_nonnegative(
                self.value_coefficient,
                field="ppo.ppo.value_coefficient",
            ),
        )
        object.__setattr__(
            self,
            "max_gradient_norm",
            _finite_positive(
                self.max_gradient_norm,
                field="ppo.ppo.max_gradient_norm",
            ),
        )
        object.__setattr__(
            self,
            "target_kl",
            _finite_positive(self.target_kl, field="ppo.ppo.target_kl"),
        )


@dataclass(frozen=True, slots=True)
class PpoLoggingConfig:
    """Local, account-free CSV/TensorBoard and memory sampling settings."""

    run_directory: str
    log_interval_updates: int
    csv_flush_interval_updates: int
    tensorboard_enabled: bool
    memory_sample_interval_updates: int

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "run_directory",
            _safe_relative_path(
                self.run_directory,
                field="ppo.logging.run_directory",
            ),
        )
        for field in (
            "log_interval_updates",
            "csv_flush_interval_updates",
            "memory_sample_interval_updates",
        ):
            object.__setattr__(
                self,
                field,
                _positive_integer(getattr(self, field), field=f"ppo.logging.{field}"),
            )
        object.__setattr__(
            self,
            "tensorboard_enabled",
            _boolean(self.tensorboard_enabled, field="ppo.logging.tensorboard_enabled"),
        )


@dataclass(frozen=True, slots=True)
class PpoCheckpointConfig:
    """Local resume cadence and public inference-artifact size policy."""

    interval_updates: int
    keep_last: int
    save_optimizer_state: bool
    public_checkpoint_max_bytes: int

    def __post_init__(self) -> None:
        for field in ("interval_updates", "keep_last", "public_checkpoint_max_bytes"):
            object.__setattr__(
                self,
                field,
                _positive_integer(getattr(self, field), field=f"ppo.checkpoint.{field}"),
            )
        object.__setattr__(
            self,
            "save_optimizer_state",
            _boolean(
                self.save_optimizer_state,
                field="ppo.checkpoint.save_optimizer_state",
            ),
        )
        if not self.save_optimizer_state:
            raise ConfigError(
                "ppo.checkpoint.save_optimizer_state must be true for optimizer continuation"
            )


@dataclass(frozen=True, slots=True)
class PpoTrainingConfig:
    """Complete cross-validated M7 PPO configuration."""

    schema_version: int
    environment: PpoEnvironmentConfig
    observation: PpoObservationConfig
    reward: PpoRewardConfig
    rollout: PpoRolloutConfig
    ppo: PpoAlgorithmConfig
    logging: PpoLoggingConfig
    checkpoint: PpoCheckpointConfig

    def __post_init__(self) -> None:
        if type(self.schema_version) is not int or self.schema_version != PPO_CONFIG_SCHEMA_VERSION:
            raise ConfigError(
                f"ppo.schema_version must be {PPO_CONFIG_SCHEMA_VERSION}, "
                f"got {self.schema_version!r}"
            )
        expected_types = (
            ("environment", PpoEnvironmentConfig),
            ("observation", PpoObservationConfig),
            ("reward", PpoRewardConfig),
            ("rollout", PpoRolloutConfig),
            ("ppo", PpoAlgorithmConfig),
            ("logging", PpoLoggingConfig),
            ("checkpoint", PpoCheckpointConfig),
        )
        for field, expected in expected_types:
            if not isinstance(getattr(self, field), expected):
                raise ConfigError(f"ppo.{field} must be a {expected.__name__}")

        if self.rollout.total_vector_steps < self.rollout.steps_per_update:
            raise ConfigError(
                "ppo.rollout.total_vector_steps must cover at least one complete update"
            )
        if self.rollout.total_vector_steps % self.rollout.steps_per_update != 0:
            raise ConfigError(
                "ppo.rollout.total_vector_steps must be divisible by steps_per_update"
            )
        if self.nominal_world_slots_per_update % self.ppo.num_minibatches != 0:
            raise ConfigError(
                "num_envs * steps_per_update must be divisible by ppo.num_minibatches"
            )
        if self.minimum_valid_world_slots_per_update < self.ppo.num_minibatches:
            raise ConfigError(
                "ppo.ppo.num_minibatches cannot exceed the minimum valid transitions under "
                "NEXT_STEP masking"
            )
        seeds = (
            self.environment.environment_seed,
            self.ppo.policy_seed,
            self.ppo.minibatch_seed,
        )
        if len(set(seeds)) != len(seeds):
            raise ConfigError(
                "environment_seed, policy_seed, and minibatch_seed must be distinct RNG domains"
            )
        if self.checkpoint.interval_updates > self.update_count:
            raise ConfigError("ppo.checkpoint.interval_updates cannot exceed the update count")
        for field in (
            "log_interval_updates",
            "csv_flush_interval_updates",
            "memory_sample_interval_updates",
        ):
            if getattr(self.logging, field) > self.update_count:
                raise ConfigError(f"ppo.logging.{field} cannot exceed the update count")

    @property
    def update_count(self) -> int:
        """Return the exact number of complete rollout/update iterations."""

        return self.rollout.total_vector_steps // self.rollout.steps_per_update

    @property
    def nominal_world_slots_per_update(self) -> int:
        """Return vector worlds times calls before NEXT_STEP reset-only masking."""

        return self.environment.num_envs * self.rollout.steps_per_update

    @property
    def nominal_minibatch_size(self) -> int:
        """Return the pre-mask nominal minibatch size used for shape planning."""

        return self.nominal_world_slots_per_update // self.ppo.num_minibatches

    @property
    def minimum_valid_world_slots_per_update(self) -> int:
        """Return the worst-case valid count under alternating terminal/reset-only rows."""

        return self.environment.num_envs * (self.rollout.steps_per_update // 2)

    @property
    def world_step_slot_budget(self) -> int:
        """Return the total world slots; valid learning transitions are reported separately."""

        return self.environment.num_envs * self.rollout.total_vector_steps


def _exact_keys(data: Mapping[str, Any], expected: set[str], *, context: str) -> None:
    actual = set(data)
    missing = expected - actual
    extra = actual - expected
    details: list[str] = []
    if missing:
        details.append(f"missing keys: {', '.join(sorted(missing))}")
    if extra:
        details.append(f"unexpected keys: {', '.join(sorted(extra))}")
    if details:
        raise ConfigError(f"{context} has {'; '.join(details)}")


def _table(data: Mapping[str, Any], key: str) -> Mapping[str, Any]:
    value = data[key]
    if not isinstance(value, Mapping):
        raise ConfigError(f"ppo.{key} must be a TOML table")
    return value


def _hidden_sizes(value: object) -> tuple[int, int]:
    if isinstance(value, (str, bytes)) or not isinstance(value, Sequence):
        raise ConfigError("ppo.ppo.hidden_sizes must be an array")
    values = tuple(value)
    if len(values) != 2 or any(
        isinstance(item, bool) or not isinstance(item, int) for item in values
    ):
        raise ConfigError("ppo.ppo.hidden_sizes must contain exactly two integers")
    return values  # type: ignore[return-value]


def load_ppo_config(path: str | Path) -> PpoTrainingConfig:
    """Load one strict seven-table PPO TOML document."""

    source = Path(path)
    if source.suffix != ".toml":
        raise ConfigError(f"PPO configuration file must use the .toml suffix: {source}")
    try:
        with source.open("rb") as file:
            data = tomllib.load(file)
    except FileNotFoundError as error:
        raise ConfigError(f"PPO configuration file does not exist: {source}") from error
    except tomllib.TOMLDecodeError as error:
        raise ConfigError(f"invalid TOML in {source}: {error}") from error

    _exact_keys(
        data,
        {
            "schema_version",
            "environment",
            "observation",
            "reward",
            "rollout",
            "ppo",
            "logging",
            "checkpoint",
        },
        context="PPO config",
    )
    environment = _table(data, "environment")
    observation = _table(data, "observation")
    reward = _table(data, "reward")
    rollout = _table(data, "rollout")
    algorithm = _table(data, "ppo")
    logging = _table(data, "logging")
    checkpoint = _table(data, "checkpoint")

    _exact_keys(
        environment,
        {
            "benchmark_version",
            "level_id",
            "backend",
            "num_envs",
            "environment_seed",
            "train_cache",
        },
        context="ppo.environment",
    )
    _exact_keys(
        observation,
        {"preview_points", "preview_distance_m", "max_speed_mps"},
        context="ppo.observation",
    )
    _exact_keys(
        reward,
        {
            "progress_scale",
            "success_bonus",
            "offtrack_invalid_penalty",
            "lateral_error_weight",
            "heading_error_weight",
            "reverse_speed_weight",
            "action_change_weight",
        },
        context="ppo.reward",
    )
    _exact_keys(
        rollout,
        {"steps_per_update", "total_vector_steps"},
        context="ppo.rollout",
    )
    _exact_keys(
        algorithm,
        {
            "hidden_sizes",
            "policy_seed",
            "minibatch_seed",
            "initial_log_std",
            "learning_rate",
            "adam_epsilon",
            "anneal_learning_rate",
            "discount_factor",
            "gae_lambda",
            "num_minibatches",
            "update_epochs",
            "normalize_advantages",
            "clip_coefficient",
            "clip_value_loss",
            "entropy_coefficient",
            "value_coefficient",
            "max_gradient_norm",
            "target_kl",
        },
        context="ppo.ppo",
    )
    _exact_keys(
        logging,
        {
            "run_directory",
            "log_interval_updates",
            "csv_flush_interval_updates",
            "tensorboard_enabled",
            "memory_sample_interval_updates",
        },
        context="ppo.logging",
    )
    _exact_keys(
        checkpoint,
        {
            "interval_updates",
            "keep_last",
            "save_optimizer_state",
            "public_checkpoint_max_bytes",
        },
        context="ppo.checkpoint",
    )

    return PpoTrainingConfig(
        schema_version=data["schema_version"],
        environment=PpoEnvironmentConfig(**environment),
        observation=PpoObservationConfig(**observation),
        reward=PpoRewardConfig(**reward),
        rollout=PpoRolloutConfig(**rollout),
        ppo=PpoAlgorithmConfig(
            hidden_sizes=_hidden_sizes(algorithm["hidden_sizes"]),
            policy_seed=algorithm["policy_seed"],
            minibatch_seed=algorithm["minibatch_seed"],
            initial_log_std=algorithm["initial_log_std"],
            learning_rate=algorithm["learning_rate"],
            adam_epsilon=algorithm["adam_epsilon"],
            anneal_learning_rate=algorithm["anneal_learning_rate"],
            discount_factor=algorithm["discount_factor"],
            gae_lambda=algorithm["gae_lambda"],
            num_minibatches=algorithm["num_minibatches"],
            update_epochs=algorithm["update_epochs"],
            normalize_advantages=algorithm["normalize_advantages"],
            clip_coefficient=algorithm["clip_coefficient"],
            clip_value_loss=algorithm["clip_value_loss"],
            entropy_coefficient=algorithm["entropy_coefficient"],
            value_coefficient=algorithm["value_coefficient"],
            max_gradient_norm=algorithm["max_gradient_norm"],
            target_kl=algorithm["target_kl"],
        ),
        logging=PpoLoggingConfig(**logging),
        checkpoint=PpoCheckpointConfig(**checkpoint),
    )


__all__ = [
    "PPO_CONFIG_SCHEMA_VERSION",
    "PPO_FORMAL_HIDDEN_SIZES",
    "PPO_FORMAL_MAX_SPEED_MPS",
    "PPO_FORMAL_PREVIEW_POINTS",
    "PPO_FORMAL_TRAIN_CACHE",
    "PPO_MAX_LOG_STD",
    "PPO_MIN_LOG_STD",
    "PpoAlgorithmConfig",
    "PpoCheckpointConfig",
    "PpoEnvironmentConfig",
    "PpoLoggingConfig",
    "PpoObservationConfig",
    "PpoRewardConfig",
    "PpoRolloutConfig",
    "PpoTrainingConfig",
    "load_ppo_config",
]
