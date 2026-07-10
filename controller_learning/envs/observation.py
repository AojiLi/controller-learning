"""Public Gymnasium observation/action spaces and pure-JAX observation encoding."""

from __future__ import annotations

from collections import OrderedDict
from collections.abc import Mapping
from typing import Any, Protocol, cast

import gymnasium as gym
import jax
import jax.numpy as jnp
import numpy as np
from gymnasium.vector.utils import batch_space

from controller_learning.config.models import ProjectConfig
from controller_learning.envs.race_core import RaceState
from controller_learning.tracks.types import TrackBatch, TrackCapacity

OBSERVATION_KEYS = (
    "position",
    "yaw",
    "velocity_body",
    "yaw_rate",
    "steering_angle",
    "track_progress",
    "centerline",
    "left_boundary",
    "right_boundary",
    "track_mask",
    "track_length",
)


class VehicleStateView(Protocol):
    """Minimal public vehicle-state surface consumed by the Challenge encoder."""

    position_world_m: Any
    yaw_rad: Any
    velocity_body_mps: Any
    angular_velocity_body_rad_s: Any
    steering_angle_rad: Any


def _track_capacity(config: ProjectConfig, capacity: TrackCapacity | None) -> TrackCapacity:
    configured = TrackCapacity(
        max_track_points=config.track.representation.max_track_points,
        max_checkpoints=config.track.representation.max_checkpoints,
    )
    if capacity is None:
        return configured
    if capacity != configured:
        raise ValueError(
            "observation capacity must match the fixed ProjectConfig representation "
            f"{configured}, got {capacity}"
        )
    return capacity


def _num_envs(value: int) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise ValueError("num_envs must be a positive integer")
    return value


def observation_space(
    config: ProjectConfig,
    capacity: TrackCapacity | None = None,
) -> gym.spaces.Dict:
    """Build the exact single-world public observation space."""

    fixed_capacity = _track_capacity(config, capacity)
    points = fixed_capacity.max_track_points
    float32 = np.dtype(np.float32)
    infinity = np.float32(np.inf)
    steering_limit = np.float32(config.vehicle.actuator.max_steering_angle_rad)
    minimum_length = np.float32(config.track.validation.min_length_m)
    maximum_length = np.float32(config.track.validation.max_length_m)

    return gym.spaces.Dict(
        OrderedDict(
            (
                ("position", gym.spaces.Box(-infinity, infinity, shape=(2,), dtype=float32)),
                (
                    "yaw",
                    gym.spaces.Box(
                        np.float32(-np.pi),
                        np.float32(np.pi),
                        shape=(),
                        dtype=float32,
                    ),
                ),
                (
                    "velocity_body",
                    gym.spaces.Box(-infinity, infinity, shape=(2,), dtype=float32),
                ),
                ("yaw_rate", gym.spaces.Box(-infinity, infinity, shape=(), dtype=float32)),
                (
                    "steering_angle",
                    gym.spaces.Box(
                        -steering_limit,
                        steering_limit,
                        shape=(),
                        dtype=float32,
                    ),
                ),
                (
                    "track_progress",
                    gym.spaces.Box(np.float32(0.0), np.float32(1.0), shape=(), dtype=float32),
                ),
                (
                    "centerline",
                    gym.spaces.Box(-infinity, infinity, shape=(points, 2), dtype=float32),
                ),
                (
                    "left_boundary",
                    gym.spaces.Box(-infinity, infinity, shape=(points, 2), dtype=float32),
                ),
                (
                    "right_boundary",
                    gym.spaces.Box(-infinity, infinity, shape=(points, 2), dtype=float32),
                ),
                ("track_mask", gym.spaces.MultiBinary(points)),
                (
                    "track_length",
                    gym.spaces.Box(
                        minimum_length,
                        maximum_length,
                        shape=(),
                        dtype=float32,
                    ),
                ),
            )
        )
    )


def batched_observation_space(
    config: ProjectConfig,
    num_envs: int,
    capacity: TrackCapacity | None = None,
) -> gym.spaces.Dict:
    """Build the leading-dimension vector observation space from the single contract."""

    single = observation_space(config, capacity)
    return cast(gym.spaces.Dict, batch_space(single, _num_envs(num_envs)))


def action_space(config: ProjectConfig) -> gym.spaces.Box:
    """Build the physical single-world action space in steering/acceleration units."""

    actuator = config.vehicle.actuator
    return gym.spaces.Box(
        low=np.asarray(
            (-actuator.max_steering_angle_rad, -actuator.max_deceleration_mps2),
            dtype=np.float32,
        ),
        high=np.asarray(
            (actuator.max_steering_angle_rad, actuator.max_acceleration_mps2),
            dtype=np.float32,
        ),
        dtype=np.float32,
    )


def batched_action_space(config: ProjectConfig, num_envs: int) -> gym.spaces.Box:
    """Build the leading-dimension vector action space from the physical action contract."""

    return cast(gym.spaces.Box, batch_space(action_space(config), _num_envs(num_envs)))


def encode_batched_observation(
    track_batch: TrackBatch,
    race_state: RaceState,
    vehicle_state_view: VehicleStateView,
) -> dict[str, jax.Array]:
    """Encode public observations without depending on simulator state or MJX data.

    Every returned value retains the leading world dimension. The position is the rear-axle
    position exposed by the vehicle-state view, and progress is the monotonic legal Race Core
    progress rather than a nearest-segment shortcut.
    """

    float32 = jnp.float32
    track_length = jnp.asarray(track_batch.length_m, dtype=float32)
    legal_progress = jnp.asarray(race_state.legal_progress_m, dtype=float32)
    progress = jnp.clip(legal_progress / track_length, 0.0, 1.0)

    return {
        "position": jnp.asarray(vehicle_state_view.position_world_m, dtype=float32)[..., :2],
        "yaw": jnp.asarray(vehicle_state_view.yaw_rad, dtype=float32),
        "velocity_body": jnp.asarray(vehicle_state_view.velocity_body_mps, dtype=float32)[..., :2],
        "yaw_rate": jnp.asarray(
            vehicle_state_view.angular_velocity_body_rad_s,
            dtype=float32,
        )[..., 2],
        "steering_angle": jnp.asarray(vehicle_state_view.steering_angle_rad, dtype=float32),
        "track_progress": progress,
        "centerline": jnp.asarray(track_batch.centerline_m, dtype=float32),
        "left_boundary": jnp.asarray(track_batch.left_boundary_m, dtype=float32),
        "right_boundary": jnp.asarray(track_batch.right_boundary_m, dtype=float32),
        "track_mask": jnp.asarray(track_batch.track_mask, dtype=jnp.int8),
        "track_length": track_length,
    }


def observation_to_host(observation: Mapping[str, Any]) -> dict[str, np.ndarray]:
    """Transfer a batched observation to NumPy while preserving shapes and public dtypes."""

    if set(observation) != set(OBSERVATION_KEYS):
        missing = sorted(set(OBSERVATION_KEYS) - set(observation))
        extra = sorted(set(observation) - set(OBSERVATION_KEYS))
        raise ValueError(
            f"observation keys do not match public schema; missing={missing}, extra={extra}"
        )
    return {
        key: np.asarray(observation[key], dtype=np.int8 if key == "track_mask" else np.float32)
        for key in OBSERVATION_KEYS
    }


def unbatch_observation(
    observation: Mapping[str, Any],
    index: int = 0,
) -> dict[str, np.ndarray]:
    """Return one host observation with the exact single-world shapes and dtypes."""

    if isinstance(index, bool) or not isinstance(index, int):
        raise TypeError("observation index must be an integer")
    host = observation_to_host(observation)
    batch_sizes = {value.shape[0] for value in host.values() if value.ndim >= 1}
    if len(batch_sizes) != 1:
        raise ValueError("all observation fields must share one leading batch dimension")
    if not batch_sizes:
        raise ValueError("batched observation fields must have a leading dimension")
    batch_size = batch_sizes.pop()
    if not -batch_size <= index < batch_size:
        raise IndexError(f"observation index {index} is outside batch size {batch_size}")
    return {key: np.asarray(value[index], dtype=value.dtype) for key, value in host.items()}


__all__ = [
    "OBSERVATION_KEYS",
    "VehicleStateView",
    "action_space",
    "batched_action_space",
    "batched_observation_space",
    "encode_batched_observation",
    "observation_space",
    "observation_to_host",
    "unbatch_observation",
]
