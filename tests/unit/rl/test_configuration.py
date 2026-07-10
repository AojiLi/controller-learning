"""Tests for the strict M7 PPO configuration."""

from __future__ import annotations

import subprocess
import sys
from dataclasses import FrozenInstanceError
from pathlib import Path

import pytest

from controller_learning.config import ConfigError
from controller_learning.rl.configuration import load_ppo_config

PROJECT_ROOT = Path(__file__).parents[3]
PPO_CONFIG = PROJECT_ROOT / "configs" / "ppo.toml"


def _candidate(tmp_path: Path, old: str, new: str) -> Path:
    path = tmp_path / "ppo.toml"
    text = PPO_CONFIG.read_text(encoding="utf-8")
    assert old in text
    path.write_text(text.replace(old, new), encoding="utf-8")
    return path


def test_repository_ppo_config_is_strict_cross_validated_and_immutable() -> None:
    config = load_ppo_config(PPO_CONFIG)

    assert config.environment.benchmark_version == "0.1"
    assert config.environment.level_id == 1
    assert config.environment.backend == "mjx_warp"
    assert config.environment.num_envs == 1024
    assert config.environment.environment_seed == 7
    assert config.environment.train_cache == ".track-cache/v0.1/train_pool.npz"
    assert config.observation.preview_points == 16
    assert config.observation.preview_distance_m == 40.0
    assert config.observation.max_speed_mps == 15.0
    assert config.reward.progress_scale == 100.0
    assert config.rollout.steps_per_update == 128
    assert config.rollout.total_vector_steps == 10_240
    assert config.ppo.hidden_sizes == (128, 128)
    assert config.ppo.policy_seed == 11
    assert config.ppo.minibatch_seed == 13
    assert config.ppo.initial_log_std == -0.5
    assert config.ppo.adam_epsilon == 1.0e-5
    assert config.ppo.num_minibatches == 32
    assert config.ppo.update_epochs == 4
    assert config.update_count == 80
    assert config.nominal_world_slots_per_update == 131_072
    assert config.nominal_minibatch_size == 4_096
    assert config.minimum_valid_world_slots_per_update == 65_536
    assert config.world_step_slot_budget == 10_485_760
    assert config.checkpoint.public_checkpoint_max_bytes == 5 * 1024 * 1024

    with pytest.raises(FrozenInstanceError):
        config.rollout.steps_per_update = 1  # type: ignore[misc]


@pytest.mark.parametrize(
    ("old", "new", "message"),
    (
        (
            "total_vector_steps = 10240",
            "total_vector_steps = 10241",
            "must be divisible by steps_per_update",
        ),
        (
            "num_minibatches = 32",
            "num_minibatches = 31",
            "must be divisible by ppo.num_minibatches",
        ),
        (
            'train_cache = ".track-cache/v0.1/train_pool.npz"',
            'train_cache = "../test.npz"',
            "safe relative POSIX path",
        ),
        (
            'train_cache = ".track-cache/v0.1/train_pool.npz"',
            'train_cache = "controller_learning/assets/tracks/v0.1/test.npz"',
            "official Train-only cache",
        ),
        (
            "lateral_error_weight = 0.05",
            "lateral_error_weight = -0.05",
            "finite and non-negative",
        ),
        (
            "preview_points = 16",
            "preview_points = 8",
            "must be 16 for feature schema v1",
        ),
        (
            "max_speed_mps = 15.0",
            "max_speed_mps = 14.0",
            "must be 15.0 for the formal vehicle",
        ),
        (
            "hidden_sizes = [128, 128]",
            "hidden_sizes = [64, 64]",
            r"must be \(128, 128\) for policy schema v1",
        ),
        (
            "initial_log_std = -0.5",
            "initial_log_std = 3.0",
            r"must be in \[-5.0, 2.0\]",
        ),
        (
            "num_envs = 1024",
            "num_envs = 256",
            "must be 1024",
        ),
        (
            "num_minibatches = 32",
            "num_minibatches = 131072",
            "minimum valid transitions under NEXT_STEP masking",
        ),
        (
            "minibatch_seed = 13",
            "minibatch_seed = 11",
            "must be distinct RNG domains",
        ),
        (
            "progress_scale = 100.0",
            "progress_scale = 1e300",
            "must fit in float32",
        ),
        (
            "progress_scale = 100.0",
            "progress_scale = 1e-300",
            "must remain positive in normal float32",
        ),
        (
            "save_optimizer_state = true",
            "save_optimizer_state = false",
            "must be true for optimizer continuation",
        ),
    ),
)
def test_ppo_config_rejects_invalid_values(
    tmp_path: Path,
    old: str,
    new: str,
    message: str,
) -> None:
    path = _candidate(tmp_path, old, new)

    with pytest.raises(ConfigError, match=message):
        load_ppo_config(path)


def test_ppo_config_rejects_unknown_and_missing_keys(tmp_path: Path) -> None:
    unknown = _candidate(
        tmp_path,
        "preview_points = 16",
        "preview_points = 16\nprivate_projection_index = true",
    )
    with pytest.raises(ConfigError, match="unexpected keys: private_projection_index"):
        load_ppo_config(unknown)

    missing = _candidate(tmp_path, "target_kl = 0.03\n", "")
    with pytest.raises(ConfigError, match="missing keys: target_kl"):
        load_ppo_config(missing)


def test_ppo_config_rejects_wrong_scalar_and_array_types(tmp_path: Path) -> None:
    boolean_integer = _candidate(tmp_path, "steps_per_update = 128", "steps_per_update = true")
    with pytest.raises(ConfigError, match="must be a positive integer"):
        load_ppo_config(boolean_integer)

    hidden_sizes = _candidate(tmp_path, "hidden_sizes = [128, 128]", "hidden_sizes = [128]")
    with pytest.raises(ConfigError, match="exactly two integers"):
        load_ppo_config(hidden_sizes)


def test_ppo_modules_do_not_import_torch() -> None:
    command = (
        "import sys; import controller_learning.rl.configuration; "
        "import controller_learning.rl.assets; assert 'torch' not in sys.modules"
    )
    subprocess.run([sys.executable, "-c", command], check=True)


def test_configuration_import_does_not_eagerly_import_jax() -> None:
    command = (
        "import sys; import controller_learning.rl.configuration; assert 'jax' not in sys.modules"
    )
    subprocess.run([sys.executable, "-c", command], check=True)
