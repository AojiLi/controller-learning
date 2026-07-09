"""Shared physical action clipping and actuator mapping."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from math import isclose, isfinite

import numpy as np

from controller_learning.config import VehicleConfig

WHEEL_STOP_THRESHOLD_RAD_S = 0.5
VEHICLE_STOP_THRESHOLD_M_S = 0.1


class VehicleActionError(ValueError):
    """Raised when a physical vehicle action cannot be applied safely."""


@dataclass(frozen=True, slots=True)
class AppliedVehicleAction:
    """Clipped physical action and the resulting actuator targets."""

    steering_angle_rad: float
    longitudinal_acceleration_mps2: float
    steering_target_rad: float
    wheel_torque_nm: tuple[float, float, float, float]
    steering_saturated: bool
    longitudinal_saturated: bool

    @property
    def saturation_count(self) -> int:
        """Return how many physical action dimensions were clipped."""

        return int(self.steering_saturated) + int(self.longitudinal_saturated)


def _validated_action(action: Sequence[float] | np.ndarray) -> tuple[float, float]:
    try:
        array = np.asarray(action, dtype=np.float64)
    except (TypeError, ValueError) as error:
        raise VehicleActionError("vehicle action must be convertible to float64") from error
    if array.shape != (2,):
        raise VehicleActionError(f"vehicle action must have shape (2,), got {array.shape}")
    if not np.isfinite(array).all():
        raise VehicleActionError("vehicle action must contain only finite values")
    return float(array[0]), float(array[1])


def wheel_torques_for_acceleration(
    config: VehicleConfig,
    longitudinal_acceleration_mps2: float,
    wheel_angular_velocity_rad_s: Sequence[float],
    longitudinal_velocity_mps: float,
) -> tuple[float, float, float, float]:
    """Map acceleration to equal drive torque or rotation-opposing brake torque.

    Positive acceleration drives all four wheels in the forward direction. Negative acceleration
    is braking: torque opposes current wheel rotation and becomes zero at rest, so braking cannot
    turn into unintended reverse propulsion.
    """

    wheel_velocity = np.asarray(wheel_angular_velocity_rad_s, dtype=np.float64)
    if wheel_velocity.shape != (4,) or not np.isfinite(wheel_velocity).all():
        raise VehicleActionError("wheel angular velocity must be four finite values")
    if not isfinite(longitudinal_acceleration_mps2) or not isfinite(longitudinal_velocity_mps):
        raise VehicleActionError("actuator state must contain only finite values")

    torque_magnitude = (
        config.vehicle.mass_kg
        * abs(longitudinal_acceleration_mps2)
        * config.vehicle.wheel_radius_m
        / 4.0
    )
    if longitudinal_acceleration_mps2 > 0.0:
        return (torque_magnitude,) * 4
    if longitudinal_acceleration_mps2 == 0.0:
        return (0.0,) * 4

    vehicle_direction = (
        np.sign(longitudinal_velocity_mps)
        if abs(longitudinal_velocity_mps) >= VEHICLE_STOP_THRESHOLD_M_S
        else 0.0
    )
    directions = np.where(
        np.abs(wheel_velocity) >= WHEEL_STOP_THRESHOLD_RAD_S,
        np.sign(wheel_velocity),
        vehicle_direction,
    )
    return tuple(float(-direction * torque_magnitude) for direction in directions)


def map_vehicle_action(
    config: VehicleConfig,
    action: Sequence[float] | np.ndarray,
    *,
    previous_steering_target_rad: float,
    wheel_angular_velocity_rad_s: Sequence[float],
    longitudinal_velocity_mps: float,
) -> AppliedVehicleAction:
    """Clip one public action and compute rate-limited steering and wheel torque targets."""

    requested_steering, requested_acceleration = _validated_action(action)
    steering = float(
        np.clip(
            requested_steering,
            -config.actuator.max_steering_angle_rad,
            config.actuator.max_steering_angle_rad,
        )
    )
    acceleration = float(
        np.clip(
            requested_acceleration,
            -config.actuator.max_deceleration_mps2,
            config.actuator.max_acceleration_mps2,
        )
    )
    max_steering_delta = config.actuator.max_steering_rate_rad_s * config.simulation.control_dt_s
    steering_target = float(
        np.clip(
            steering,
            previous_steering_target_rad - max_steering_delta,
            previous_steering_target_rad + max_steering_delta,
        )
    )
    wheel_torque = wheel_torques_for_acceleration(
        config,
        acceleration,
        wheel_angular_velocity_rad_s,
        longitudinal_velocity_mps,
    )
    return AppliedVehicleAction(
        steering_angle_rad=steering,
        longitudinal_acceleration_mps2=acceleration,
        steering_target_rad=steering_target,
        wheel_torque_nm=wheel_torque,
        steering_saturated=not isclose(
            requested_steering,
            steering,
            rel_tol=0.0,
            abs_tol=0.0,
        ),
        longitudinal_saturated=not isclose(
            requested_acceleration,
            acceleration,
            rel_tol=0.0,
            abs_tol=0.0,
        ),
    )
