"""Public M7 reinforcement-learning interfaces with lazy optional imports."""

from __future__ import annotations

from importlib import import_module
from typing import Any

_EXPORT_MODULES = {
    "PPO_CONFIG_SCHEMA_VERSION": "controller_learning.rl.configuration",
    "PPO_FORMAL_TRAIN_CACHE": "controller_learning.rl.configuration",
    "PpoAlgorithmConfig": "controller_learning.rl.configuration",
    "PpoCheckpointConfig": "controller_learning.rl.configuration",
    "PpoEnvironmentConfig": "controller_learning.rl.configuration",
    "PpoLoggingConfig": "controller_learning.rl.configuration",
    "PpoObservationConfig": "controller_learning.rl.configuration",
    "PpoRewardConfig": "controller_learning.rl.configuration",
    "PpoRolloutConfig": "controller_learning.rl.configuration",
    "PpoTrainingConfig": "controller_learning.rl.configuration",
    "load_ppo_config": "controller_learning.rl.configuration",
    "TRAIN_POOL_ACCESS_SCHEMA_VERSION": "controller_learning.rl.assets",
    "TrainPoolAccessEvidence": "controller_learning.rl.assets",
    "VerifiedTrainPool": "controller_learning.rl.assets",
    "load_verified_train_pool": "controller_learning.rl.assets",
}

__all__ = sorted(_EXPORT_MODULES)


def __getattr__(name: str) -> Any:
    """Resolve heavier Track/JAX-backed exports only when callers request them."""

    try:
        module_name = _EXPORT_MODULES[name]
    except KeyError as error:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}") from error
    value = getattr(import_module(module_name), name)
    globals()[name] = value
    return value
