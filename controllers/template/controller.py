"""Minimal non-performing Controller used to verify the plugin interface."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import numpy as np
from numpy.typing import NDArray

from controller_learning.control import Controller


class TemplateController(Controller):
    """Return a zero steering and zero acceleration action at every step."""

    def compute_control(
        self,
        obs: Mapping[str, Any],
        info: Mapping[str, Any] | None = None,
    ) -> NDArray[np.float32]:
        """Return the neutral action using the public float32 action contract."""
        return np.zeros(2, dtype=np.float32)
