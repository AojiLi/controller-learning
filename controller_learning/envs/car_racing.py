"""Single-world Gymnasium adapter over the official vector Challenge."""

from __future__ import annotations

from numbers import Integral
from typing import TYPE_CHECKING, Any, ClassVar

import gymnasium as gym
import numpy as np
from gymnasium import error

from controller_learning.config import ProjectConfig
from controller_learning.envs._vehicle_driver import VehicleBackend
from controller_learning.envs.episode import PublicScalarInfo, unbatch_public_info
from controller_learning.envs.observation import unbatch_observation
from controller_learning.envs.vector_racing import VecCarRacingEnv
from controller_learning.tracks.pool import TrackPool
from controller_learning.tracks.types import Track

if TYPE_CHECKING:
    from controller_learning.control.debug_draw import DebugDrawCommand
    from controller_learning.visualization import Renderer2D


class CarRacingEnv(gym.Env):
    """Gymnasium single-world API without a second transition implementation."""

    metadata: ClassVar[dict[str, Any]] = {
        "render_modes": ["human", "rgb_array"],
        "render_fps": 20,
    }

    def __init__(
        self,
        *,
        project_config: ProjectConfig,
        level_id: int,
        backend: VehicleBackend,
        track: Track | None = None,
        track_pool: TrackPool | None = None,
        render_mode: str | None = None,
    ) -> None:
        super().__init__()
        if (track is None) == (track_pool is None):
            raise ValueError("provide exactly one of track or track_pool")
        if track is not None and not isinstance(track, Track):
            raise TypeError("track must be an immutable Track value or None")
        if track_pool is not None and not isinstance(track_pool, TrackPool):
            raise TypeError("track_pool must be an immutable TrackPool or None")
        if render_mode not in (None, "human", "rgb_array"):
            raise ValueError("render_mode must be None, 'human', or 'rgb_array'")
        self.render_mode = render_mode
        self.metadata = dict(type(self).metadata)
        self._pool_mode = track_pool is not None
        self._vector_env = VecCarRacingEnv(
            num_envs=1,
            project_config=project_config,
            level_id=level_id,
            tracks=None if track is None else (track,),
            track_pool=track_pool,
            backend=backend,
            render_mode=None,
        )
        # The Runner derives the immutable public Controller configuration from the actual
        # Challenge instance, so these values cannot drift from the environment being executed.
        self.project_config = self._vector_env.project_config
        self.level_id = self._vector_env.level_id
        self.backend = self._vector_env.backend
        self.observation_space = self._vector_env.single_observation_space
        self.action_space = self._vector_env.single_action_space
        self._needs_reset = True
        self._last_observation: dict[str, np.ndarray] | None = None
        self._debug_commands: tuple[DebugDrawCommand, ...] = ()
        self._renderer: Renderer2D | None = None
        if render_mode is not None:
            # Keep Matplotlib out of headless vector training and ``sim --help`` imports.
            from controller_learning.visualization import Renderer2D

            self._renderer = Renderer2D(render_mode)

    def reset(
        self,
        *,
        seed: int | None = None,
        options: dict[str, Any] | None = None,
    ) -> tuple[dict[str, np.ndarray], PublicScalarInfo]:
        """Explicitly begin a new single-world episode."""

        if options is not None and not isinstance(options, dict):
            raise TypeError("reset options must be a dictionary or None")
        vector_options: dict[str, Any] | None = options
        if options:
            if not self._pool_mode:
                raise ValueError("CarRacingEnv does not define reset options without a TrackPool")
            if set(options) != {"track_index"}:
                raise ValueError("TrackPool reset options must contain only 'track_index'")
            track_index = options["track_index"]
            if isinstance(track_index, bool) or not isinstance(track_index, Integral):
                raise TypeError("track_index must be an integer")
            vector_options = {
                "track_indices": np.asarray((int(track_index),), dtype=np.int64),
            }
        super().reset(seed=seed)
        observation, info = self._vector_env.reset(seed=seed, options=vector_options)
        self._needs_reset = False
        self._last_observation = unbatch_observation(observation)
        self._debug_commands = ()
        return self._last_observation, unbatch_public_info(info)

    def step(
        self,
        action: object,
    ) -> tuple[dict[str, np.ndarray], float, bool, bool, PublicScalarInfo]:
        """Advance one transition; terminal episodes require an explicit reset."""

        if self._needs_reset:
            raise error.ResetNeeded("call reset before step or after a terminal transition")
        observation, reward, terminated, truncated, info = self._vector_env.step((action,))
        terminal = bool(np.asarray(terminated)[0])
        timeout = bool(np.asarray(truncated)[0])
        if terminal or timeout:
            self._needs_reset = True
        self._last_observation = unbatch_observation(observation)
        return (
            self._last_observation,
            float(np.asarray(reward)[0]),
            terminal,
            timeout,
            unbatch_public_info(info),
        )

    def render_debug_frame(self, commands: tuple[DebugDrawCommand, ...]) -> None:
        """Store one immutable write-only Controller overlay for the next render call."""

        if self._renderer is None:
            raise gym.error.Error(
                "render_debug_frame requires render_mode='human' or render_mode='rgb_array'"
            )
        if not isinstance(commands, tuple):
            raise TypeError("DebugDraw commands must be an immutable tuple")
        self._debug_commands = commands

    def render(self) -> np.ndarray | None:
        """Render only the latest public observation and write-only DebugDraw commands."""

        if self._renderer is None:
            return None
        if self._last_observation is None:
            raise error.ResetNeeded("call reset before render")
        return self._renderer.render(self._last_observation, self._debug_commands)

    def close(self) -> None:
        """Delegate backend cleanup."""

        if self._renderer is not None:
            self._renderer.close()
        self._vector_env.close()


__all__ = ["CarRacingEnv"]
