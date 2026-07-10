"""Backend-independent interface implemented by every Controller plugin."""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Mapping
from typing import TYPE_CHECKING, Any

import numpy as np
from numpy.typing import NDArray

if TYPE_CHECKING:
    from controller_learning.control.debug_draw import DebugDraw

Observation = Mapping[str, Any]
ControllerInfo = Mapping[str, Any]
ControllerConfig = Mapping[str, Any]
Action = NDArray[np.float32]


class Controller(ABC):
    """Base class for one trusted, single-environment Controller episode.

    The runner constructs a new instance for every episode. Controllers receive only public
    observations, restricted info, and immutable public configuration values.
    """

    def __init__(
        self,
        obs: Observation,
        info: ControllerInfo,
        config: ControllerConfig,
    ) -> None:
        """Initialize one episode's Controller state from public inputs only."""
        return None

    @abstractmethod
    def compute_control(
        self,
        obs: Observation,
        info: ControllerInfo | None = None,
    ) -> Action:
        """Return ``[steering_angle_rad, longitudinal_acceleration_mps2]``."""

    def step_callback(
        self,
        action: Action,
        obs: Observation,
        reward: float,
        terminated: bool,
        truncated: bool,
        info: ControllerInfo,
    ) -> None:
        """Observe the result of one control step, if the Controller needs it."""
        return None

    def episode_callback(self) -> None:
        """Run optional end-of-episode Controller work."""
        return None

    def render_callback(self, debug_draw: DebugDraw) -> None:
        """Submit optional renderer commands through a write-only drawing surface."""
        return None
