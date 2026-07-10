"""Public-observation reward shaping with exact NEXT_STEP reset accounting."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any, NamedTuple

import gymnasium as gym
import jax
import jax.numpy as jnp
import numpy as np
from gymnasium.vector import AutoresetMode, VectorEnv

from controller_learning.envs.race_core import RaceTermination, wrap_angle
from controller_learning.envs.vector_racing import VecCarRacingEnv
from controller_learning.rl.configuration import PpoRewardConfig
from controller_learning.rl.features import local_track_reference_jax
from controller_learning.rl.schema import PUBLIC_REWARD_SCHEMA_VERSION


class RewardShapingResult(NamedTuple):
    """One device-native shaped reward and its normalized action history value."""

    reward: jax.Array
    normalized_action: jax.Array


def normalize_physical_actions_jax(
    actions: Any,
    *,
    max_steering_angle_rad: float,
    max_acceleration_mps2: float,
    max_deceleration_mps2: float,
) -> jax.Array:
    """Map finite physical actions to ``[-1, 1]`` for smoothness accounting.

    Non-finite values become neutral only in this diagnostic shaping term. The official
    environment still receives the original action and owns invalid-action termination.
    """

    physical = jnp.asarray(actions, dtype=jnp.float32)
    finite = jnp.where(jnp.isfinite(physical), physical, jnp.float32(0.0))
    steering = jnp.clip(
        finite[..., 0],
        -max_steering_angle_rad,
        max_steering_angle_rad,
    ) / jnp.float32(max_steering_angle_rad)
    acceleration = finite[..., 1]
    normalized_acceleration = jnp.where(
        acceleration >= 0.0,
        jnp.clip(acceleration, 0.0, max_acceleration_mps2) / jnp.float32(max_acceleration_mps2),
        jnp.clip(acceleration, -max_deceleration_mps2, 0.0) / jnp.float32(max_deceleration_mps2),
    )
    return jnp.stack((steering, normalized_acceleration), axis=-1).astype(jnp.float32)


def shape_public_reward_jax(
    observation: Mapping[str, Any],
    base_reward: Any,
    termination_reason: Any,
    actions: Any,
    previous_normalized_action: Any,
    reset_only: Any,
    *,
    config: PpoRewardConfig,
    max_steering_angle_rad: float,
    max_acceleration_mps2: float,
    max_deceleration_mps2: float,
) -> RewardShapingResult:
    """Shape one batch using only public observations, reward, info, and submitted actions."""

    reward = jnp.asarray(base_reward, dtype=jnp.float32)
    reason = jnp.asarray(termination_reason, dtype=jnp.int32)
    dummy = jnp.asarray(reset_only, dtype=bool)
    previous_action = jnp.asarray(previous_normalized_action, dtype=jnp.float32)
    normalized_action = normalize_physical_actions_jax(
        actions,
        max_steering_angle_rad=max_steering_angle_rad,
        max_acceleration_mps2=max_acceleration_mps2,
        max_deceleration_mps2=max_deceleration_mps2,
    )

    success = reason == jnp.int32(RaceTermination.SUCCESS)
    failure = (reason == jnp.int32(RaceTermination.OFF_TRACK)) | (
        reason == jnp.int32(RaceTermination.INVALID_ACTION)
    )
    progress_reward = reward - success.astype(jnp.float32) + failure.astype(jnp.float32)

    reference_center, reference_tangent = local_track_reference_jax(observation)
    position = jnp.asarray(observation["position"], dtype=jnp.float32)
    offset = position - reference_center
    lateral_error = (
        reference_tangent[..., 0] * offset[..., 1] - reference_tangent[..., 1] * offset[..., 0]
    )
    reference_heading = jnp.arctan2(
        reference_tangent[..., 1],
        reference_tangent[..., 0],
    )
    heading_error = wrap_angle(
        jnp.asarray(observation["yaw"], dtype=jnp.float32) - reference_heading
    )
    reverse_speed = jnp.maximum(
        -jnp.asarray(observation["velocity_body"], dtype=jnp.float32)[..., 0],
        jnp.float32(0.0),
    )
    action_change = jnp.sum(
        jnp.square(normalized_action - previous_action),
        axis=-1,
    )

    shaped = (
        jnp.float32(config.progress_scale) * progress_reward
        + jnp.float32(config.success_bonus) * success.astype(jnp.float32)
        - jnp.float32(config.offtrack_invalid_penalty) * failure.astype(jnp.float32)
        - jnp.float32(config.lateral_error_weight) * jnp.square(lateral_error)
        - jnp.float32(config.heading_error_weight) * jnp.square(heading_error)
        - jnp.float32(config.reverse_speed_weight) * jnp.square(reverse_speed)
        - jnp.float32(config.action_change_weight) * action_change
    ).astype(jnp.float32)
    shaped = jnp.where(dummy, jnp.float32(0.0), shaped)
    next_action = jnp.where(dummy[..., None], jnp.float32(0.0), normalized_action)
    return RewardShapingResult(shaped, next_action)


class PublicRewardShapingVecEnv(gym.vector.VectorWrapper):
    """Layer versioned public reward shaping over the unchanged official Challenge."""

    def __init__(self, env: VectorEnv, config: PpoRewardConfig) -> None:
        super().__init__(env)
        if not isinstance(config, PpoRewardConfig):
            raise TypeError("config must be a PpoRewardConfig")
        if not isinstance(env, VecCarRacingEnv):
            raise TypeError(
                "PublicRewardShapingVecEnv must directly wrap the official VecCarRacingEnv"
            )
        base = env
        if base.level_id != 1 or base.project_config.benchmark.version != "0.1":
            raise ValueError("public PPO reward shaping requires benchmark 0.1 Level 1")
        if env.metadata.get("autoreset_mode") != AutoresetMode.NEXT_STEP:
            raise ValueError("PublicRewardShapingVecEnv requires NEXT_STEP autoreset semantics")

        actuator = base.project_config.vehicle.actuator
        self.config = config
        self.schema_version = PUBLIC_REWARD_SCHEMA_VERSION
        self._max_steering_angle_rad = actuator.max_steering_angle_rad
        self._max_acceleration_mps2 = actuator.max_acceleration_mps2
        self._max_deceleration_mps2 = actuator.max_deceleration_mps2
        self._pending_reset = jnp.zeros(self.num_envs, dtype=bool)
        self._previous_normalized_action = jnp.zeros((self.num_envs, 2), dtype=jnp.float32)
        reward_config = self.config
        max_steering_angle_rad = self._max_steering_angle_rad
        max_acceleration_mps2 = self._max_acceleration_mps2
        max_deceleration_mps2 = self._max_deceleration_mps2
        self._shape_reward = jax.jit(
            lambda observation, reward, reason, actions, previous, reset_only: (
                shape_public_reward_jax(
                    observation,
                    reward,
                    reason,
                    actions,
                    previous,
                    reset_only,
                    config=reward_config,
                    max_steering_angle_rad=max_steering_angle_rad,
                    max_acceleration_mps2=max_acceleration_mps2,
                    max_deceleration_mps2=max_deceleration_mps2,
                )
            )
        )

    def _actions_for_shaping(self, actions: object) -> jax.Array:
        """Return shape-safe actions without changing what the Challenge receives."""

        if isinstance(actions, jax.Array):
            if actions.shape == (self.num_envs, 2) and not jnp.issubdtype(
                actions.dtype,
                jnp.complexfloating,
            ):
                return jnp.asarray(actions, dtype=jnp.float32)
            return jnp.zeros((self.num_envs, 2), dtype=jnp.float32)
        try:
            source = np.asarray(actions)
            if source.shape != (self.num_envs, 2) or source.dtype.kind == "c":
                raise ValueError
            return jnp.asarray(source, dtype=jnp.float32)
        except (TypeError, ValueError, OverflowError):
            return jnp.zeros((self.num_envs, 2), dtype=jnp.float32)

    def reset(
        self,
        *,
        seed: int | None = None,
        options: dict[str, Any] | None = None,
    ) -> tuple[Any, dict[str, Any]]:
        """Reset the Challenge and clear shaping-only NEXT_STEP state."""

        observation, info = self.env.reset(seed=seed, options=options)
        self._pending_reset = jnp.zeros(self.num_envs, dtype=bool)
        self._previous_normalized_action = jnp.zeros((self.num_envs, 2), dtype=jnp.float32)
        return observation, info

    def step(self, actions: object) -> tuple[Any, Any, Any, Any, dict[str, Any]]:
        """Advance once and replace only the public reward array."""

        if self._shape_reward is None:
            raise gym.error.ClosedEnvironmentError
        reset_only = self._pending_reset
        shaping_actions = self._actions_for_shaping(actions)
        observation, reward, terminated, truncated, info = self.env.step(actions)
        if not isinstance(info, Mapping) or "termination_reason" not in info:
            raise ValueError("official public info must contain termination_reason")
        result = self._shape_reward(
            observation,
            reward,
            info["termination_reason"],
            shaping_actions,
            self._previous_normalized_action,
            reset_only,
        )
        self._previous_normalized_action = result.normalized_action
        self._pending_reset = jnp.asarray(terminated, dtype=bool) | jnp.asarray(
            truncated,
            dtype=bool,
        )
        return observation, result.reward, terminated, truncated, info

    def close(self) -> None:
        """Release shaping JIT/device state before closing the official Challenge."""

        self._shape_reward = None
        self._pending_reset = None
        self._previous_normalized_action = None
        super().close()


__all__ = [
    "PUBLIC_REWARD_SCHEMA_VERSION",
    "PublicRewardShapingVecEnv",
    "RewardShapingResult",
    "normalize_physical_actions_jax",
    "shape_public_reward_jax",
]
