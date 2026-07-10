"""Deterministic episode identities and restricted public environment info."""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from numbers import Integral
from typing import NamedTuple, TypeAlias

import jax
import jax.numpy as jnp
import numpy as np
from numpy.typing import NDArray

from controller_learning.envs.race_core import RaceStep, RaceTermination
from controller_learning.tracks.types import Track

PUBLIC_INFO_KEYS = (
    "episode_seed",
    "controller_seed",
    "track_id",
    "benchmark_version",
    "termination_reason",
    "lap_completed",
    "lap_time_s",
)
"""The complete public info whitelist, in stable insertion order."""

_UINT32_MAX = int(np.iinfo(np.uint32).max)
_EPISODE_DOMAIN = 0
_CONTROLLER_DOMAIN = 1
_UINT32_INIT_A = 0x43B0D7E5
_UINT32_MULT_A = 0x931E8875
_UINT32_INIT_B = 0x8B51F9DD
_UINT32_MULT_B = 0x58F38DED
_UINT32_MIX_LEFT = 0xCA01F9DD
_UINT32_MIX_RIGHT = 0x4973F715

PublicInfoArray: TypeAlias = NDArray[np.generic] | jax.Array
PublicVectorInfo: TypeAlias = dict[str, PublicInfoArray]
PublicScalarInfo: TypeAlias = dict[str, int | float | bool | str]


def _uint32_scalar(value: object, *, name: str) -> np.uint32:
    if isinstance(value, bool) or not isinstance(value, Integral):
        raise TypeError(f"{name} must be an integer in the uint32 range")
    integer = int(value)
    if not 0 <= integer <= _UINT32_MAX:
        raise ValueError(f"{name} must be in the uint32 range")
    return np.uint32(integer)


def _readonly_array(
    value: object,
    *,
    dtype: np.dtype,
    shape: tuple[int, ...],
    name: str,
) -> NDArray:
    source = np.asarray(value)
    if source.shape != shape:
        raise ValueError(f"{name} must have shape {shape}, got {source.shape}")
    if source.dtype != dtype:
        raise TypeError(f"{name} must have dtype {dtype}, got {source.dtype}")
    result = np.array(source, dtype=dtype, copy=True)
    result.setflags(write=False)
    return result


def _readonly_copy(value: object) -> NDArray:
    result = np.array(value, copy=True)
    result.setflags(write=False)
    return result


def _domain_seed(
    root_seed: np.uint32,
    world_index: np.uint32,
    episode_counter: np.uint32,
    domain: int,
) -> np.uint32:
    """Derive one seed using the version-stable NumPy ``SeedSequence`` contract.

    The root seed is entropy. World, per-world episode counter, and domain are the spawn path. This
    avoids global RNG state and makes the result independent of reset ordering and batch size.
    """

    sequence = np.random.SeedSequence(
        entropy=int(root_seed),
        spawn_key=(int(world_index), int(episode_counter), domain),
    )
    return sequence.generate_state(1, dtype=np.uint32)[0]


def _seed_pair(
    root_seed: np.uint32,
    world_index: np.uint32,
    episode_counter: np.uint32,
) -> tuple[np.uint32, np.uint32]:
    episode_seed = _domain_seed(
        root_seed,
        world_index,
        episode_counter,
        _EPISODE_DOMAIN,
    )
    controller_seed = _domain_seed(
        root_seed,
        world_index,
        episode_counter,
        _CONTROLLER_DOMAIN,
    )
    # Domain-separated 32-bit outputs can theoretically collide. Keep the public guarantee strict
    # with a deterministic, non-zero bijection in that single collision case.
    if controller_seed == episode_seed:
        controller_seed ^= np.uint32(0xFFFFFFFF)
    return episode_seed, controller_seed


@dataclass(frozen=True, slots=True)
class EpisodeIdentity:
    """Immutable host state for independently resetting vector-environment episodes."""

    root_seed: np.uint32
    world_index: NDArray[np.uint32]
    episode_counter: NDArray[np.uint32]
    episode_seed: NDArray[np.uint32]
    controller_seed: NDArray[np.uint32]

    def __post_init__(self) -> None:
        root_seed = _uint32_scalar(self.root_seed, name="root_seed")
        world_index_source = np.asarray(self.world_index)
        if world_index_source.ndim != 1 or world_index_source.size < 1:
            raise ValueError("world_index must be a non-empty one-dimensional array")
        shape = world_index_source.shape

        arrays = {}
        for name in (
            "world_index",
            "episode_counter",
            "episode_seed",
            "controller_seed",
        ):
            arrays[name] = _readonly_array(
                getattr(self, name),
                dtype=np.dtype(np.uint32),
                shape=shape,
                name=name,
            )

        expected_worlds = np.arange(shape[0], dtype=np.uint32)
        if not np.array_equal(arrays["world_index"], expected_worlds):
            raise ValueError("world_index must be the contiguous range [0, num_envs)")
        if np.any(arrays["episode_seed"] == arrays["controller_seed"]):
            raise ValueError("episode_seed and controller_seed must differ in every world")

        object.__setattr__(self, "root_seed", root_seed)
        for name, array in arrays.items():
            object.__setattr__(self, name, array)

    @property
    def num_envs(self) -> int:
        """Return the fixed leading world count."""

        return int(self.world_index.shape[0])


class DeviceEpisodeIdentity(NamedTuple):
    """JAX leaves matching ``EpisodeIdentity`` for native vector autoreset."""

    root_seed: jax.Array
    world_index: jax.Array
    episode_counter: jax.Array
    episode_seed: jax.Array
    controller_seed: jax.Array


def _seed_sequence_word_device(
    root_seed: jax.Array,
    world_index: jax.Array,
    episode_counter: jax.Array,
    domain: int,
) -> jax.Array:
    """Return one bit-exact NumPy ``SeedSequence`` word using pure JAX.

    This specializes NumPy's documented uint32 mixer to the project's fixed entropy and spawn-key
    schema: ``root_seed`` and ``(world_index, episode_counter, domain)``. Keeping the derivation on
    device prevents Gym NEXT_STEP identity updates from introducing a GPU-to-host synchronization.
    """

    uint32 = jnp.uint32
    entropy = (
        uint32(root_seed),
        uint32(0),
        uint32(0),
        uint32(0),
        uint32(world_index),
        uint32(episode_counter),
        uint32(domain),
    )
    hash_constant = uint32(_UINT32_INIT_A)

    def hash_mix(value: jax.Array, constant: jax.Array) -> tuple[jax.Array, jax.Array]:
        value = uint32(value) ^ constant
        constant = constant * uint32(_UINT32_MULT_A)
        value = value * constant
        return value ^ (value >> uint32(16)), constant

    def mix(left: jax.Array, right: jax.Array) -> jax.Array:
        result = uint32(_UINT32_MIX_LEFT) * left - uint32(_UINT32_MIX_RIGHT) * right
        return result ^ (result >> uint32(16))

    pool: list[jax.Array] = []
    for value in entropy[:4]:
        mixed, hash_constant = hash_mix(value, hash_constant)
        pool.append(mixed)
    for source in range(4):
        for destination in range(4):
            if source != destination:
                mixed, hash_constant = hash_mix(pool[source], hash_constant)
                pool[destination] = mix(pool[destination], mixed)
    for value in entropy[4:]:
        for destination in range(4):
            mixed, hash_constant = hash_mix(value, hash_constant)
            pool[destination] = mix(pool[destination], mixed)

    state = pool[0] ^ uint32(_UINT32_INIT_B)
    output_constant = uint32(_UINT32_INIT_B) * uint32(_UINT32_MULT_B)
    state = state * output_constant
    return state ^ (state >> uint32(16))


_seed_sequence_batch_device = jax.vmap(
    _seed_sequence_word_device,
    in_axes=(None, 0, 0, None),
)


def episode_identity_to_device(identity: EpisodeIdentity) -> DeviceEpisodeIdentity:
    """Copy a validated host identity to JAX without changing any public seed."""

    return DeviceEpisodeIdentity(
        root_seed=jnp.asarray(identity.root_seed, dtype=jnp.uint32),
        world_index=jnp.asarray(identity.world_index, dtype=jnp.uint32),
        episode_counter=jnp.asarray(identity.episode_counter, dtype=jnp.uint32),
        episode_seed=jnp.asarray(identity.episode_seed, dtype=jnp.uint32),
        controller_seed=jnp.asarray(identity.controller_seed, dtype=jnp.uint32),
    )


def masked_next_episode_device(
    identity: DeviceEpisodeIdentity,
    mask: jax.Array,
) -> DeviceEpisodeIdentity:
    """Advance selected identities on device with host ``SeedSequence`` parity."""

    reset_mask = jnp.asarray(mask, dtype=bool)

    def advance(current: DeviceEpisodeIdentity) -> DeviceEpisodeIdentity:
        counters = current.episode_counter + reset_mask.astype(jnp.uint32)
        episode_candidate = _seed_sequence_batch_device(
            current.root_seed,
            current.world_index,
            counters,
            _EPISODE_DOMAIN,
        )
        controller_candidate = _seed_sequence_batch_device(
            current.root_seed,
            current.world_index,
            counters,
            _CONTROLLER_DOMAIN,
        )
        controller_candidate = jnp.where(
            controller_candidate == episode_candidate,
            controller_candidate ^ jnp.uint32(0xFFFFFFFF),
            controller_candidate,
        )
        return DeviceEpisodeIdentity(
            root_seed=current.root_seed,
            world_index=current.world_index,
            episode_counter=counters,
            episode_seed=jnp.where(reset_mask, episode_candidate, current.episode_seed),
            controller_seed=jnp.where(
                reset_mask,
                controller_candidate,
                current.controller_seed,
            ),
        )

    return jax.lax.cond(jnp.any(reset_mask), advance, lambda current: current, identity)


def initialize_episode_identities(root_seed: int, num_envs: int) -> EpisodeIdentity:
    """Create counter-zero identities for every world from an explicit root seed."""

    root = _uint32_scalar(root_seed, name="root_seed")
    if isinstance(num_envs, bool) or not isinstance(num_envs, Integral):
        raise TypeError("num_envs must be a positive integer")
    world_count = int(num_envs)
    if world_count < 1:
        raise ValueError("num_envs must be positive")
    if world_count > _UINT32_MAX + 1:
        raise ValueError("num_envs cannot exceed the uint32 world-index range")

    worlds = np.arange(world_count, dtype=np.uint32)
    counters = np.zeros(world_count, dtype=np.uint32)
    episode_seeds = np.empty(world_count, dtype=np.uint32)
    controller_seeds = np.empty(world_count, dtype=np.uint32)
    for index in range(world_count):
        episode_seeds[index], controller_seeds[index] = _seed_pair(
            root,
            worlds[index],
            counters[index],
        )
    return EpisodeIdentity(
        root_seed=root,
        world_index=worlds,
        episode_counter=counters,
        episode_seed=episode_seeds,
        controller_seed=controller_seeds,
    )


def masked_next_episode(current: EpisodeIdentity, mask: object) -> EpisodeIdentity:
    """Advance only selected per-world counters and seeds, preserving every other value."""

    reset_mask = np.asarray(mask)
    expected_shape = (current.num_envs,)
    if reset_mask.shape != expected_shape:
        raise ValueError(f"mask must have shape {expected_shape}, got {reset_mask.shape}")
    if reset_mask.dtype != np.dtype(np.bool_):
        raise TypeError(f"mask must have dtype bool, got {reset_mask.dtype}")
    if np.any(reset_mask & (current.episode_counter == np.uint32(_UINT32_MAX))):
        raise OverflowError("selected episode_counter cannot advance beyond uint32")

    counters = np.array(current.episode_counter, copy=True)
    episode_seeds = np.array(current.episode_seed, copy=True)
    controller_seeds = np.array(current.controller_seed, copy=True)
    for index in np.flatnonzero(reset_mask):
        counters[index] += np.uint32(1)
        episode_seeds[index], controller_seeds[index] = _seed_pair(
            current.root_seed,
            current.world_index[index],
            counters[index],
        )
    return EpisodeIdentity(
        root_seed=current.root_seed,
        world_index=current.world_index,
        episode_counter=counters,
        episode_seed=episode_seeds,
        controller_seed=controller_seeds,
    )


def track_id_from_track(track: Track) -> str:
    """Return the stable M4 track identity derived from an injected host Track value."""

    if not isinstance(track, Track):
        raise TypeError("track must be a Track")
    return f"{track.generator_version}:{track.seed}"


def _base_public_info(
    identity: EpisodeIdentity,
    tracks: Sequence[Track],
    benchmark_version: str,
) -> PublicVectorInfo:
    if isinstance(tracks, (str, bytes)) or not isinstance(tracks, Sequence):
        raise TypeError("tracks must be a sequence of Track values")
    if len(tracks) != identity.num_envs:
        raise ValueError(
            "tracks must contain one value per world, "
            f"expected {identity.num_envs}, got {len(tracks)}"
        )
    if not isinstance(benchmark_version, str):
        raise TypeError("benchmark_version must be a string")
    if not benchmark_version:
        raise ValueError("benchmark_version cannot be empty")

    track_ids = np.asarray([track_id_from_track(track) for track in tracks], dtype=np.str_)
    benchmark_versions = np.asarray(
        [benchmark_version] * identity.num_envs,
        dtype=np.str_,
    )
    termination = np.full(
        identity.num_envs,
        np.int32(RaceTermination.NONE),
        dtype=np.int32,
    )
    lap_completed = np.zeros(identity.num_envs, dtype=np.bool_)
    lap_time = np.zeros(identity.num_envs, dtype=np.float32)
    return {
        "episode_seed": identity.episode_seed,
        "controller_seed": identity.controller_seed,
        "track_id": _readonly_copy(track_ids),
        "benchmark_version": _readonly_copy(benchmark_versions),
        "termination_reason": _readonly_copy(termination),
        "lap_completed": _readonly_copy(lap_completed),
        "lap_time_s": _readonly_copy(lap_time),
    }


def build_reset_info(
    identity: EpisodeIdentity,
    tracks: Sequence[Track],
    benchmark_version: str,
) -> PublicVectorInfo:
    """Build reset info with the complete whitelist and neutral terminal fields."""

    return _base_public_info(identity, tracks, benchmark_version)


def _race_step_array(
    value: object,
    *,
    shape: tuple[int, ...],
    name: str,
    kind: str,
) -> NDArray:
    array = np.asarray(value)
    if array.shape != shape:
        raise ValueError(f"RaceStep.{name} must have shape {shape}, got {array.shape}")
    if kind == "bool" and array.dtype != np.dtype(np.bool_):
        raise TypeError(f"RaceStep.{name} must have dtype bool, got {array.dtype}")
    if kind == "integer" and not np.issubdtype(array.dtype, np.integer):
        raise TypeError(f"RaceStep.{name} must have an integer dtype, got {array.dtype}")
    return array


def build_step_info(
    identity: EpisodeIdentity,
    tracks: Sequence[Track],
    benchmark_version: str,
    race_step: RaceStep,
    control_dt_s: float,
) -> PublicVectorInfo:
    """Build restricted public step info from Race Core outcomes only."""

    if not isinstance(race_step, RaceStep):
        raise TypeError("race_step must be a RaceStep")
    if not isinstance(control_dt_s, (int, float, np.integer, np.floating)) or isinstance(
        control_dt_s, bool
    ):
        raise TypeError("control_dt_s must be a finite positive number")
    control_dt = float(control_dt_s)
    if not math.isfinite(control_dt) or control_dt <= 0.0:
        raise ValueError("control_dt_s must be finite and positive")

    shape = (identity.num_envs,)
    reasons_source = _race_step_array(
        race_step.termination_reason,
        shape=shape,
        name="termination_reason",
        kind="integer",
    )
    reasons = np.asarray(reasons_source, dtype=np.int32)
    valid_reasons = np.asarray([int(reason) for reason in RaceTermination], dtype=np.int32)
    if not np.isin(reasons, valid_reasons).all():
        raise ValueError("RaceStep.termination_reason contains an unknown RaceTermination value")
    success = _race_step_array(
        race_step.success,
        shape=shape,
        name="success",
        kind="bool",
    )
    elapsed_source = _race_step_array(
        race_step.state.elapsed_steps,
        shape=shape,
        name="state.elapsed_steps",
        kind="integer",
    )
    if np.any(elapsed_source < 0):
        raise ValueError("RaceStep.state.elapsed_steps cannot be negative")

    elapsed_seconds = np.asarray(elapsed_source, dtype=np.float64) * control_dt
    if not np.isfinite(elapsed_seconds).all() or np.any(elapsed_seconds > np.finfo(np.float32).max):
        raise ValueError("elapsed lap time must fit in float32")
    lap_time = np.where(success, elapsed_seconds, 0.0).astype(np.float32)

    info = _base_public_info(identity, tracks, benchmark_version)
    info["termination_reason"] = _readonly_copy(reasons)
    info["lap_completed"] = _readonly_copy(np.asarray(success, dtype=np.bool_))
    info["lap_time_s"] = _readonly_copy(lap_time)
    return info


def _validate_public_info(info: Mapping[str, object]) -> int:
    if tuple(info) != PUBLIC_INFO_KEYS or set(info) != set(PUBLIC_INFO_KEYS):
        raise ValueError(f"info keys must exactly match the public whitelist {PUBLIC_INFO_KEYS}")

    expected_dtypes: dict[str, np.dtype] = {
        "episode_seed": np.dtype(np.uint32),
        "controller_seed": np.dtype(np.uint32),
        "termination_reason": np.dtype(np.int32),
        "lap_completed": np.dtype(np.bool_),
        "lap_time_s": np.dtype(np.float32),
    }
    arrays = {key: np.asarray(value) for key, value in info.items()}
    first = arrays["episode_seed"]
    if first.ndim != 1 or first.size < 1:
        raise ValueError("public info values must be non-empty one-dimensional arrays")
    shape = first.shape
    for key, array in arrays.items():
        if array.shape != shape:
            raise ValueError(f"info[{key!r}] must have shape {shape}, got {array.shape}")
        if key in expected_dtypes and array.dtype != expected_dtypes[key]:
            raise TypeError(
                f"info[{key!r}] must have dtype {expected_dtypes[key]}, got {array.dtype}"
            )
    for key in ("track_id", "benchmark_version"):
        if arrays[key].dtype.kind != "U":
            raise TypeError(f"info[{key!r}] must contain NumPy unicode strings")
    return int(shape[0])


def unbatch_public_info(info: Mapping[str, object], index: int = 0) -> PublicScalarInfo:
    """Convert one vector-info row to Python scalars for ``CarRacingEnv``."""

    world_count = _validate_public_info(info)
    if isinstance(index, bool) or not isinstance(index, Integral):
        raise TypeError("index must be an integer")
    world_index = int(index)
    if not 0 <= world_index < world_count:
        raise IndexError(f"index must be in [0, {world_count})")

    return {
        "episode_seed": int(np.asarray(info["episode_seed"])[world_index]),
        "controller_seed": int(np.asarray(info["controller_seed"])[world_index]),
        "track_id": str(np.asarray(info["track_id"])[world_index]),
        "benchmark_version": str(np.asarray(info["benchmark_version"])[world_index]),
        "termination_reason": int(np.asarray(info["termination_reason"])[world_index]),
        "lap_completed": bool(np.asarray(info["lap_completed"])[world_index]),
        "lap_time_s": float(np.asarray(info["lap_time_s"])[world_index]),
    }


__all__ = [
    "PUBLIC_INFO_KEYS",
    "DeviceEpisodeIdentity",
    "EpisodeIdentity",
    "PublicScalarInfo",
    "PublicVectorInfo",
    "build_reset_info",
    "build_step_info",
    "episode_identity_to_device",
    "initialize_episode_identities",
    "masked_next_episode",
    "masked_next_episode_device",
    "track_id_from_track",
    "unbatch_public_info",
]
