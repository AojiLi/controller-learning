"""Observation-derived curvature speed profiles for classical Controllers."""

from __future__ import annotations

import math

import numpy as np
from numpy.typing import ArrayLike, NDArray


def _positive(value: float, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"{name} must be a number")
    result = float(value)
    if not math.isfinite(result) or result <= 0.0:
        raise ValueError(f"{name} must be finite and positive")
    return result


def curvature_speed_profile(
    curvature_1pm: ArrayLike,
    distances_m: ArrayLike,
    *,
    minimum_speed_mps: float,
    maximum_speed_mps: float,
    maximum_lateral_acceleration_mps2: float,
    braking_deceleration_mps2: float,
) -> NDArray[np.float64]:
    """Return curvature limits tightened backward by reachable braking speed.

    ``distances_m`` is a non-decreasing preview coordinate starting at the current projection.
    The returned value at each point accounts for every later curvature limit, so index zero is a
    directly usable target speed and the complete array is suitable for an MPC horizon.
    """

    curvature = np.asarray(curvature_1pm, dtype=np.float64)
    distances = np.asarray(distances_m, dtype=np.float64)
    if curvature.ndim != 1 or distances.shape != curvature.shape or curvature.size == 0:
        raise ValueError("curvature_1pm and distances_m must be non-empty matching vectors")
    if not np.isfinite(curvature).all() or not np.isfinite(distances).all():
        raise ValueError("curvature_1pm and distances_m must contain only finite values")
    if distances[0] < 0.0 or np.any(np.diff(distances) < 0.0):
        raise ValueError("distances_m must be non-negative and non-decreasing")

    minimum_speed = _positive(minimum_speed_mps, "minimum_speed_mps")
    maximum_speed = _positive(maximum_speed_mps, "maximum_speed_mps")
    lateral_acceleration = _positive(
        maximum_lateral_acceleration_mps2,
        "maximum_lateral_acceleration_mps2",
    )
    braking_deceleration = _positive(
        braking_deceleration_mps2,
        "braking_deceleration_mps2",
    )
    if minimum_speed > maximum_speed:
        raise ValueError("minimum_speed_mps cannot exceed maximum_speed_mps")

    curvature_limit = np.sqrt(lateral_acceleration / np.maximum(np.abs(curvature), 1.0e-4))
    profile = np.clip(curvature_limit, minimum_speed, maximum_speed)
    for index in range(profile.size - 2, -1, -1):
        spacing = distances[index + 1] - distances[index]
        reachable = math.sqrt(profile[index + 1] ** 2 + 2.0 * braking_deceleration * spacing)
        profile[index] = min(profile[index], reachable)
    profile.setflags(write=False)
    return profile


__all__ = ["curvature_speed_profile"]
