"""Validation-selection observation wrapper built only from the public Challenge observation."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import gymnasium as gym
import jax
import jax.numpy as jnp
import numpy as np
from gymnasium.vector import AutoresetMode, VectorEnv
from gymnasium.vector.utils import batch_space

from controller_learning.envs.vector_racing import VecCarRacingEnv
from controller_learning.rl.configuration import PpoObservationConfig
from controller_learning.rl.features import (
    LOCAL_TRACK_FEATURE_DIM,
    LOCAL_TRACK_FEATURE_SCHEMA_VERSION,
    encode_local_track_features_jax,
)


class SelectionPublicObservationVecEnv(gym.vector.VectorObservationWrapper):
    """Expose policy features and public progress without any private environment state."""

    def __init__(self, env: VectorEnv, *, config: PpoObservationConfig) -> None:
        super().__init__(env)
        base = env.unwrapped
        if not isinstance(base, VecCarRacingEnv):
            raise TypeError("SelectionPublicObservationVecEnv requires VecCarRacingEnv")
        if base.backend != "mjx_warp" or base.level_id != 1:
            raise ValueError("Validation selection requires Level 1 MJX-Warp")
        if env.metadata.get("autoreset_mode") != AutoresetMode.NEXT_STEP:
            raise ValueError("Validation selection requires NEXT_STEP autoreset semantics")
        if not isinstance(config, PpoObservationConfig):
            raise TypeError("config must be PpoObservationConfig")
        project = base.project_config
        if config.max_speed_mps != project.vehicle.vehicle.max_speed_mps:
            raise ValueError("selection max_speed_mps must match the public vehicle limit")
        self.feature_schema_version = LOCAL_TRACK_FEATURE_SCHEMA_VERSION
        self.config = config
        preview_points = config.preview_points
        preview_distance_m = config.preview_distance_m
        max_speed_mps = config.max_speed_mps
        control_dt_s = project.vehicle.simulation.control_dt_s
        max_steering_angle_rad = project.vehicle.actuator.max_steering_angle_rad
        self._encode = jax.jit(
            lambda observation: encode_local_track_features_jax(
                observation,
                preview_points=preview_points,
                preview_distance_m=preview_distance_m,
                max_speed_mps=max_speed_mps,
                control_dt_s=control_dt_s,
                max_steering_angle_rad=max_steering_angle_rad,
            )
        )
        self.single_observation_space = gym.spaces.Dict(
            {
                "features": gym.spaces.Box(
                    low=-np.inf,
                    high=np.inf,
                    shape=(LOCAL_TRACK_FEATURE_DIM,),
                    dtype=np.float32,
                ),
                "track_progress": gym.spaces.Box(
                    low=np.float32(0.0),
                    high=np.float32(1.0),
                    shape=(),
                    dtype=np.float32,
                ),
            }
        )
        self.observation_space = batch_space(self.single_observation_space, self.num_envs)

    def observations(self, observations: Mapping[str, Any]) -> dict[str, jax.Array]:
        if self._encode is None:
            raise gym.error.ClosedEnvironmentError
        return {
            "features": self._encode(observations),
            "track_progress": jnp.asarray(observations["track_progress"], dtype=jnp.float32),
        }

    def close(self) -> None:
        self._encode = None
        super().close()


__all__ = ["SelectionPublicObservationVecEnv"]
