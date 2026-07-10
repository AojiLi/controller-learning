"""Gymnasium registration for the public Controller Learning environment."""

from __future__ import annotations

import gymnasium as gym

ENV_ID = "ControllerLearning/CarRacing-v0"
_SINGLE_ENTRY_POINT = "controller_learning.envs.car_racing:CarRacingEnv"
_VECTOR_ENTRY_POINT = "controller_learning.envs.vector_racing:VecCarRacingEnv"


def register_environments() -> None:
    """Register the single and native vector entry points idempotently."""

    existing = gym.registry.get(ENV_ID)
    if existing is not None:
        compatible = (
            existing.entry_point == _SINGLE_ENTRY_POINT
            and existing.vector_entry_point == _VECTOR_ENTRY_POINT
            and existing.max_episode_steps is None
            and existing.order_enforce is True
            and existing.disable_env_checker is False
        )
        if not compatible:
            raise RuntimeError(f"Gymnasium environment ID {ENV_ID!r} is already registered")
        return
    gym.register(
        id=ENV_ID,
        entry_point=_SINGLE_ENTRY_POINT,
        vector_entry_point=_VECTOR_ENTRY_POINT,
        max_episode_steps=None,
        order_enforce=True,
        disable_env_checker=False,
    )


__all__ = ["ENV_ID", "register_environments"]
