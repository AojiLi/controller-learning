"""Versioned public-observation features for the v0.1 PPO policy.

The encoder deliberately reconstructs every local reference from the documented observation
mapping.  It never reads a Track, Race Core state, vehicle backend, or simulator object.  The
formal schema contains four ego-state values followed by center, left-boundary, and right-boundary
previews in body coordinates::

    [
        velocity_body / max_speed,
        yaw_rate * control_dt,
        steering_angle / max_steering_angle,
        center_preview.flatten() / preview_distance,
        left_preview.flatten() / preview_distance,
        right_preview.flatten() / preview_distance,
    ]

With 16 preview points this is a fixed 100-dimensional float32 vector.
"""

from __future__ import annotations

import math
from collections.abc import Mapping
from typing import Any

import gymnasium as gym
import jax
import jax.numpy as jnp
import numpy as np
from gymnasium.vector import AutoresetMode, VectorEnv
from gymnasium.vector.utils import batch_space
from numpy.typing import NDArray

from controller_learning.envs.observation import OBSERVATION_KEYS
from controller_learning.envs.vector_racing import VecCarRacingEnv
from controller_learning.rl.configuration import PpoObservationConfig

LOCAL_TRACK_FEATURE_SCHEMA_VERSION = 1
LOCAL_TRACK_PREVIEW_POINTS = 16
LOCAL_TRACK_FEATURE_DIM = 100

_MIN_CAPACITY = 4
_CLOSURE_ATOL_M = 0.0
_LENGTH_ATOL_M = 1.0e-4
_LENGTH_RTOL = 3.0e-5


def _positive_integer(value: object, *, name: str, minimum: int = 1) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        qualifier = "positive" if minimum == 1 else f"at least {minimum}"
        raise ValueError(f"{name} must be {qualifier}")
    return value


def _finite_positive(value: object, *, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"{name} must be a number")
    result = float(value)
    if not math.isfinite(result) or result <= 0.0:
        raise ValueError(f"{name} must be finite and positive")
    return result


def _exact_observation_keys(observation: object) -> Mapping[str, Any]:
    if not isinstance(observation, Mapping):
        raise TypeError("observation must be a mapping")
    actual = set(observation)
    expected = set(OBSERVATION_KEYS)
    if actual != expected:
        missing = sorted(expected - actual)
        extra = sorted(actual - expected)
        raise ValueError(
            f"observation keys do not match the public schema; missing={missing}, extra={extra}"
        )
    return observation


def _validate_shapes(observation: object, *, batched: bool) -> tuple[Mapping[str, Any], int, int]:
    values = _exact_observation_keys(observation)
    geometry_shape = tuple(values["centerline"].shape)
    expected_rank = 3 if batched else 2
    if len(geometry_shape) != expected_rank or geometry_shape[-1] != 2:
        prefix = "(num_envs, capacity, 2)" if batched else "(capacity, 2)"
        raise ValueError(f"centerline must have shape {prefix}")
    num_envs = geometry_shape[0] if batched else 1
    capacity = geometry_shape[-2]
    if capacity < _MIN_CAPACITY:
        raise ValueError("centerline capacity must contain at least four points")

    scalar_shape = (num_envs,) if batched else ()
    vector_shape = (num_envs, 2) if batched else (2,)
    path_shape = (num_envs, capacity, 2) if batched else (capacity, 2)
    mask_shape = (num_envs, capacity) if batched else (capacity,)
    expected_shapes = {
        "position": vector_shape,
        "yaw": scalar_shape,
        "velocity_body": vector_shape,
        "yaw_rate": scalar_shape,
        "steering_angle": scalar_shape,
        "track_progress": scalar_shape,
        "centerline": path_shape,
        "left_boundary": path_shape,
        "right_boundary": path_shape,
        "track_mask": mask_shape,
        "track_length": scalar_shape,
    }
    for name, expected in expected_shapes.items():
        actual = tuple(values[name].shape)
        if actual != expected:
            raise ValueError(f"observation field {name!r} must have shape {expected}, got {actual}")
    return values, num_envs, capacity


def _jax_geometry(
    observation: Mapping[str, Any],
) -> tuple[jax.Array, jax.Array, jax.Array, jax.Array, jax.Array, jax.Array]:
    """Return masked paths, segment vectors/lengths, and cumulative arc length."""

    center = jnp.asarray(observation["centerline"], dtype=jnp.float32)
    left = jnp.asarray(observation["left_boundary"], dtype=jnp.float32)
    right = jnp.asarray(observation["right_boundary"], dtype=jnp.float32)
    mask = jnp.asarray(observation["track_mask"], dtype=bool)
    valid_segments = mask[:, :-1] & mask[:, 1:]

    # Mask after subtraction so arbitrary or non-finite padding cannot affect a valid segment.
    raw_delta = center[:, 1:] - center[:, :-1]
    delta = jnp.where(valid_segments[..., None], raw_delta, jnp.float32(0.0))
    segment_length = jnp.linalg.norm(delta, axis=-1)
    cumulative = jnp.concatenate(
        (
            jnp.zeros((center.shape[0], 1), dtype=jnp.float32),
            jnp.cumsum(segment_length, axis=1),
        ),
        axis=1,
    )
    return center, left, right, delta, segment_length, cumulative


def _sample_paths_jax(
    observation: Mapping[str, Any],
    offsets_m: jax.Array,
) -> tuple[jax.Array, jax.Array, jax.Array, jax.Array]:
    center, left, right, delta, segment_length, cumulative = _jax_geometry(observation)
    geometry_length = cumulative[:, -1]
    progress = jnp.asarray(observation["track_progress"], dtype=jnp.float32)
    query = progress[:, None] * geometry_length[:, None] + offsets_m[None, :]
    wrapped = jnp.mod(query, geometry_length[:, None])

    # Counting cumulative upper bounds implements searchsorted(..., side="right") in batch.
    segment = jnp.sum(wrapped[..., None] >= cumulative[:, None, 1:], axis=-1)
    valid_segment_count = jnp.sum(
        jnp.asarray(observation["track_mask"], dtype=bool)[:, :-1]
        & jnp.asarray(observation["track_mask"], dtype=bool)[:, 1:],
        axis=1,
    )
    segment = jnp.minimum(segment, valid_segment_count[:, None] - 1).astype(jnp.int32)

    start_s = jnp.take_along_axis(cumulative[:, None, :], segment[..., None], axis=2)[..., 0]
    selected_length = jnp.take_along_axis(segment_length[:, None, :], segment[..., None], axis=2)[
        ..., 0
    ]
    fraction = (wrapped - start_s) / selected_length

    def interpolate(path: jax.Array) -> jax.Array:
        starts = jnp.take_along_axis(path[:, None, :, :], segment[..., None, None], axis=2)[
            ..., 0, :
        ]
        ends = jnp.take_along_axis(path[:, None, 1:, :], segment[..., None, None], axis=2)[
            ..., 0, :
        ]
        return starts + fraction[..., None] * (ends - starts)

    selected_delta = jnp.take_along_axis(delta[:, None, :, :], segment[..., None, None], axis=2)[
        ..., 0, :
    ]
    tangent = selected_delta / selected_length[..., None]
    return interpolate(center), interpolate(left), interpolate(right), tangent


def _world_points_to_body_jax(points: jax.Array, observation: Mapping[str, Any]) -> jax.Array:
    position = jnp.asarray(observation["position"], dtype=jnp.float32)
    yaw = jnp.asarray(observation["yaw"], dtype=jnp.float32)
    relative = points - position[:, None, :]
    cosine = jnp.cos(yaw)[:, None]
    sine = jnp.sin(yaw)[:, None]
    return jnp.stack(
        (
            cosine * relative[..., 0] + sine * relative[..., 1],
            -sine * relative[..., 0] + cosine * relative[..., 1],
        ),
        axis=-1,
    )


def sample_local_track_preview_jax(
    observation: Mapping[str, Any],
    *,
    preview_points: int,
    preview_distance_m: float,
) -> tuple[jax.Array, jax.Array, jax.Array]:
    """Sample body-frame center/left/right previews from a batched public observation.

    Each result has shape ``(num_envs, preview_points, 2)`` and remains in meters.  The lookup uses
    only the masked, explicitly closed centerline to reconstruct arc length; boundary paths reuse
    the same segment and interpolation fraction.
    """

    values, _, _ = _validate_shapes(observation, batched=True)
    points = _positive_integer(preview_points, name="preview_points", minimum=2)
    distance = _finite_positive(preview_distance_m, name="preview_distance_m")
    offsets = jnp.linspace(0.0, distance, points, dtype=jnp.float32)
    center, left, right, _ = _sample_paths_jax(values, offsets)
    return tuple(_world_points_to_body_jax(path, values) for path in (center, left, right))


def local_track_reference_jax(
    observation: Mapping[str, Any],
) -> tuple[jax.Array, jax.Array]:
    """Return the current world-frame center point and unit tangent for each world."""

    values, _, _ = _validate_shapes(observation, batched=True)
    center, _, _, tangent = _sample_paths_jax(values, jnp.zeros((1,), dtype=jnp.float32))
    return center[:, 0], tangent[:, 0]


def encode_local_track_features_jax(
    observation: Mapping[str, Any],
    *,
    preview_points: int,
    preview_distance_m: float,
    max_speed_mps: float,
    control_dt_s: float,
    max_steering_angle_rad: float,
) -> jax.Array:
    """Encode the version-1 fixed-width PPO feature vector for a JAX observation batch."""

    points = _positive_integer(preview_points, name="preview_points", minimum=2)
    if points != LOCAL_TRACK_PREVIEW_POINTS:
        raise ValueError(
            f"feature schema {LOCAL_TRACK_FEATURE_SCHEMA_VERSION} requires "
            f"preview_points={LOCAL_TRACK_PREVIEW_POINTS}"
        )
    distance = _finite_positive(preview_distance_m, name="preview_distance_m")
    speed = _finite_positive(max_speed_mps, name="max_speed_mps")
    control_dt = _finite_positive(control_dt_s, name="control_dt_s")
    steering_limit = _finite_positive(
        max_steering_angle_rad,
        name="max_steering_angle_rad",
    )
    values, num_envs, _ = _validate_shapes(observation, batched=True)
    center, left, right = sample_local_track_preview_jax(
        values,
        preview_points=points,
        preview_distance_m=distance,
    )
    ego = jnp.concatenate(
        (
            jnp.asarray(values["velocity_body"], dtype=jnp.float32) / speed,
            (jnp.asarray(values["yaw_rate"], dtype=jnp.float32) * control_dt)[:, None],
            (jnp.asarray(values["steering_angle"], dtype=jnp.float32) / steering_limit)[:, None],
        ),
        axis=1,
    )
    previews = tuple(
        path.reshape((num_envs, 2 * points)) / distance for path in (center, left, right)
    )
    result = jnp.concatenate((ego, *previews), axis=1).astype(jnp.float32)
    if result.shape != (num_envs, LOCAL_TRACK_FEATURE_DIM):
        raise AssertionError("local-track feature layout does not match the versioned schema")
    return result


def _numeric_numpy(value: Any, *, name: str) -> NDArray[Any]:
    try:
        result = np.asarray(value)
    except (TypeError, ValueError) as error:
        raise ValueError(f"observation field {name!r} must be a numeric array") from error
    if not np.issubdtype(result.dtype, np.number) and result.dtype != np.dtype(np.bool_):
        raise ValueError(f"observation field {name!r} must be a numeric array")
    return result


def _validated_numpy_observation(observation: object) -> dict[str, NDArray[Any]]:
    values, _, capacity = _validate_shapes(observation, batched=False)
    arrays = {name: _numeric_numpy(values[name], name=name) for name in OBSERVATION_KEYS}
    mask = arrays["track_mask"]
    if not np.all((mask == 0) | (mask == 1)):
        raise ValueError("track_mask must contain only zero or one")
    valid_count = int(np.count_nonzero(mask))
    if valid_count < _MIN_CAPACITY:
        raise ValueError("track_mask must select at least four points including closure")
    if not np.array_equal(mask.astype(np.bool_), np.arange(capacity) < valid_count):
        raise ValueError("track_mask must be one contiguous valid prefix")

    finite_fields = (
        "position",
        "yaw",
        "velocity_body",
        "yaw_rate",
        "steering_angle",
        "track_progress",
        "track_length",
    )
    for name in finite_fields:
        if not np.isfinite(arrays[name]).all():
            raise ValueError(f"observation field {name!r} must contain only finite values")
    progress = float(arrays["track_progress"])
    if not 0.0 <= progress <= 1.0:
        raise ValueError("track_progress must be in [0, 1]")
    if float(arrays["track_length"]) <= 0.0:
        raise ValueError("track_length must be positive")

    for name in ("centerline", "left_boundary", "right_boundary"):
        valid = arrays[name][:valid_count]
        if not np.isfinite(valid).all():
            raise ValueError(f"valid {name} points must contain only finite values")
        if not np.allclose(valid[0], valid[-1], rtol=0.0, atol=_CLOSURE_ATOL_M):
            raise ValueError(f"valid {name} points must be explicitly closed")

    center = np.asarray(arrays["centerline"][:valid_count], dtype=np.float64)
    segment_length = np.linalg.norm(np.diff(center, axis=0), axis=1)
    if not np.all(segment_length > 0.0):
        raise ValueError("all valid centerline segments must have positive length")
    geometry_length = float(np.sum(segment_length, dtype=np.float64))
    if not np.isclose(
        float(arrays["track_length"]),
        geometry_length,
        rtol=_LENGTH_RTOL,
        atol=_LENGTH_ATOL_M,
    ):
        raise ValueError("track_length is inconsistent with the masked centerline")
    return arrays


def _sample_paths_numpy(
    observation: Mapping[str, NDArray[Any]],
    offsets_m: NDArray[np.float64],
) -> tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64]]:
    valid_count = int(np.count_nonzero(observation["track_mask"]))
    paths = tuple(
        np.asarray(observation[name][:valid_count], dtype=np.float64)
        for name in ("centerline", "left_boundary", "right_boundary")
    )
    delta = np.diff(paths[0], axis=0)
    segment_length = np.linalg.norm(delta, axis=1)
    cumulative = np.concatenate(
        (np.zeros(1, dtype=np.float64), np.cumsum(segment_length, dtype=np.float64))
    )
    geometry_length = float(cumulative[-1])
    query = float(observation["track_progress"]) * geometry_length + offsets_m
    wrapped = np.mod(query, geometry_length)
    segment = np.searchsorted(cumulative[1:], wrapped, side="right")
    segment = np.minimum(segment, segment_length.size - 1)
    fraction = (wrapped - cumulative[segment]) / segment_length[segment]

    def interpolate(path: NDArray[np.float64]) -> NDArray[np.float64]:
        return path[segment] + fraction[:, None] * (path[segment + 1] - path[segment])

    return tuple(interpolate(path) for path in paths)


def _world_points_to_body_numpy(
    points: NDArray[np.float64],
    observation: Mapping[str, NDArray[Any]],
) -> NDArray[np.float64]:
    relative = points - np.asarray(observation["position"], dtype=np.float64)
    yaw = float(observation["yaw"])
    cosine = math.cos(yaw)
    sine = math.sin(yaw)
    return np.stack(
        (
            cosine * relative[:, 0] + sine * relative[:, 1],
            -sine * relative[:, 0] + cosine * relative[:, 1],
        ),
        axis=-1,
    )


def encode_local_track_features_numpy(
    observation: Mapping[str, Any],
    *,
    preview_points: int,
    preview_distance_m: float,
    max_speed_mps: float,
    control_dt_s: float,
    max_steering_angle_rad: float,
) -> NDArray[np.float32]:
    """Encode one public NumPy observation with the same version-1 feature contract."""

    points = _positive_integer(preview_points, name="preview_points", minimum=2)
    if points != LOCAL_TRACK_PREVIEW_POINTS:
        raise ValueError(
            f"feature schema {LOCAL_TRACK_FEATURE_SCHEMA_VERSION} requires "
            f"preview_points={LOCAL_TRACK_PREVIEW_POINTS}"
        )
    distance = _finite_positive(preview_distance_m, name="preview_distance_m")
    speed = _finite_positive(max_speed_mps, name="max_speed_mps")
    control_dt = _finite_positive(control_dt_s, name="control_dt_s")
    steering_limit = _finite_positive(
        max_steering_angle_rad,
        name="max_steering_angle_rad",
    )
    values = _validated_numpy_observation(observation)
    offsets = np.linspace(0.0, distance, points, dtype=np.float64)
    previews = tuple(
        _world_points_to_body_numpy(path, values) for path in _sample_paths_numpy(values, offsets)
    )
    ego = np.concatenate(
        (
            np.asarray(values["velocity_body"], dtype=np.float64) / speed,
            np.asarray((float(values["yaw_rate"]) * control_dt,), dtype=np.float64),
            np.asarray((float(values["steering_angle"]) / steering_limit,), dtype=np.float64),
        )
    )
    result = np.concatenate(
        (ego, *(path.reshape(2 * points) / distance for path in previews))
    ).astype(np.float32)
    if result.shape != (LOCAL_TRACK_FEATURE_DIM,):
        raise AssertionError("local-track feature layout does not match the versioned schema")
    if not np.isfinite(result).all():
        raise ValueError("encoded local-track features must contain only finite values")
    return result


def _validate_public_observation_spaces(env: VectorEnv, base: VecCarRacingEnv) -> None:
    for label, actual, expected in (
        ("single", env.single_observation_space, base.single_observation_space),
        ("batched", env.observation_space, base.observation_space),
    ):
        if not isinstance(actual, gym.spaces.Dict) or not isinstance(expected, gym.spaces.Dict):
            raise TypeError(f"{label} observation space must be the public Dict schema")
        if set(actual.spaces) != set(OBSERVATION_KEYS):
            raise ValueError(f"{label} observation space does not match the public schema")
        for name in OBSERVATION_KEYS:
            actual_field = actual.spaces[name]
            expected_field = expected.spaces[name]
            if (
                actual_field.shape != expected_field.shape
                or actual_field.dtype != expected_field.dtype
            ):
                raise ValueError(
                    f"{label} observation field {name!r} does not match the public schema"
                )


class LocalTrackObservationVecEnv(gym.vector.VectorObservationWrapper):
    """Replace the exact official public observation with version-1 local-track features."""

    def __init__(self, env: VectorEnv, *, config: PpoObservationConfig) -> None:
        super().__init__(env)
        base = env.unwrapped
        if not isinstance(base, VecCarRacingEnv):
            raise TypeError("LocalTrackObservationVecEnv requires the official VecCarRacingEnv")
        if env.num_envs != base.num_envs:
            raise ValueError("wrapped vector environment width must match VecCarRacingEnv")
        if env.metadata.get("autoreset_mode") != AutoresetMode.NEXT_STEP:
            raise ValueError("LocalTrackObservationVecEnv requires NEXT_STEP autoreset semantics")
        _validate_public_observation_spaces(env, base)
        if not isinstance(config, PpoObservationConfig):
            raise TypeError("config must be a PpoObservationConfig")
        if config.preview_points != LOCAL_TRACK_PREVIEW_POINTS:
            raise ValueError(
                f"feature schema {LOCAL_TRACK_FEATURE_SCHEMA_VERSION} requires "
                f"preview_points={LOCAL_TRACK_PREVIEW_POINTS}"
            )

        project = base.project_config
        if not math.isclose(
            config.max_speed_mps,
            project.vehicle.vehicle.max_speed_mps,
            rel_tol=0.0,
            abs_tol=1.0e-6,
        ):
            raise ValueError("observation max_speed_mps must match the public vehicle limit")
        self.feature_schema_version = LOCAL_TRACK_FEATURE_SCHEMA_VERSION
        self.config = config
        self._control_dt_s = project.vehicle.simulation.control_dt_s
        self._max_steering_angle_rad = project.vehicle.actuator.max_steering_angle_rad
        preview_points = config.preview_points
        preview_distance_m = config.preview_distance_m
        max_speed_mps = config.max_speed_mps
        control_dt_s = self._control_dt_s
        max_steering_angle_rad = self._max_steering_angle_rad
        self._encode_observation = jax.jit(
            lambda observation: encode_local_track_features_jax(
                observation,
                preview_points=preview_points,
                preview_distance_m=preview_distance_m,
                max_speed_mps=max_speed_mps,
                control_dt_s=control_dt_s,
                max_steering_angle_rad=max_steering_angle_rad,
            )
        )
        self.single_observation_space = gym.spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(LOCAL_TRACK_FEATURE_DIM,),
            dtype=np.float32,
        )
        self.observation_space = batch_space(self.single_observation_space, self.num_envs)

    def observations(self, observations: Mapping[str, Any]) -> jax.Array:
        """Run the compiled public-observation encoder without a host transfer."""

        if self._encode_observation is None:
            raise gym.error.ClosedEnvironmentError
        return self._encode_observation(observations)

    def close(self) -> None:
        """Release the instance-owned JIT callable before closing the GPU Challenge."""

        self._encode_observation = None
        super().close()


__all__ = [
    "LOCAL_TRACK_FEATURE_DIM",
    "LOCAL_TRACK_FEATURE_SCHEMA_VERSION",
    "LOCAL_TRACK_PREVIEW_POINTS",
    "LocalTrackObservationVecEnv",
    "encode_local_track_features_jax",
    "encode_local_track_features_numpy",
    "local_track_reference_jax",
    "sample_local_track_preview_jax",
]
