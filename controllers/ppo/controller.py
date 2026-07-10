"""Torch-free PPO Controller over the versioned public local-track feature schema."""

from __future__ import annotations

import math
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import NDArray

from controller_learning.control import Controller
from controller_learning.rl.controller_export import load_ppo_controller_runtime
from controller_learning.rl.features import (
    LOCAL_TRACK_FEATURE_DIM,
    encode_local_track_features_numpy,
)


def _table(config: Mapping[str, Any], key: str) -> Mapping[str, Any]:
    value = config.get(key)
    if not isinstance(value, Mapping):
        raise ValueError(f"public config field {key!r} must be a table")
    return value


def _finite_positive(value: object, *, field: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{field} must be a number")
    result = float(value)
    if not math.isfinite(result) or result <= 0.0:
        raise ValueError(f"{field} must be finite and positive")
    return result


class PpoController(Controller):
    """Evaluate the selected deterministic NumPy actor for one public observation."""

    def __init__(
        self,
        obs: Mapping[str, Any],
        info: Mapping[str, Any],
        config: Mapping[str, Any],
    ) -> None:
        benchmark_version = config.get("benchmark_version")
        if benchmark_version != "0.1" or config.get("level_id") != 1:
            raise ValueError("PPO Controller requires the benchmark-0.1 Level-1 Challenge")
        if info.get("benchmark_version") != benchmark_version:
            raise ValueError("public info and Challenge benchmark versions differ")
        runtime = load_ppo_controller_runtime(
            config,
            plugin_directory=Path(__file__).resolve().parent,
        )
        action_limits = _table(config, "action_limits")
        vehicle = _table(config, "vehicle")
        self._control_dt_s = _finite_positive(config.get("control_dt_s"), field="control_dt_s")
        self._maximum_steering_rad = _finite_positive(
            action_limits.get("max_steering_angle_rad"),
            field="max_steering_angle_rad",
        )
        maximum_acceleration = _finite_positive(
            action_limits.get("max_acceleration_mps2"),
            field="max_acceleration_mps2",
        )
        maximum_deceleration = _finite_positive(
            action_limits.get("max_deceleration_mps2"),
            field="max_deceleration_mps2",
        )
        maximum_speed = _finite_positive(vehicle.get("max_speed_mps"), field="max_speed_mps")
        if not math.isclose(
            runtime.observation.max_speed_mps,
            maximum_speed,
            rel_tol=0.0,
            abs_tol=1.0e-6,
        ):
            raise ValueError("PPO feature max_speed_mps differs from the public vehicle")
        expected_low = np.asarray(
            (-self._maximum_steering_rad, -maximum_deceleration),
            dtype=np.float32,
        )
        expected_high = np.asarray(
            (self._maximum_steering_rad, maximum_acceleration),
            dtype=np.float32,
        )
        if not np.array_equal(runtime.actor.action_low, expected_low) or not np.array_equal(
            runtime.actor.action_high,
            expected_high,
        ):
            raise ValueError("PPO policy bounds differ from the public physical action limits")
        self._runtime = runtime
        self._encode(obs)

    def _encode(self, obs: Mapping[str, Any]) -> NDArray[np.float32]:
        observation = self._runtime.observation
        features = encode_local_track_features_numpy(
            obs,
            preview_points=observation.preview_points,
            preview_distance_m=observation.preview_distance_m,
            max_speed_mps=observation.max_speed_mps,
            control_dt_s=self._control_dt_s,
            max_steering_angle_rad=self._maximum_steering_rad,
        )
        if features.shape != (LOCAL_TRACK_FEATURE_DIM,) or features.dtype != np.float32:
            raise RuntimeError("PPO public feature encoder violated its fixed 100-D schema")
        return features

    def compute_control(
        self,
        obs: Mapping[str, Any],
        info: Mapping[str, Any] | None = None,
    ) -> NDArray[np.float32]:
        """Return deterministic steering and acceleration in physical units."""

        if info is not None and info.get("benchmark_version") != "0.1":
            raise ValueError("PPO Controller received an incompatible public benchmark version")
        action = self._runtime.actor(self._encode(obs))
        if action.shape != (2,) or action.dtype != np.float32 or not np.isfinite(action).all():
            raise RuntimeError("PPO Controller produced an invalid action")
        return np.array(action, dtype=np.float32, copy=True)
