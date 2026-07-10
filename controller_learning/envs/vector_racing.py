"""Official batched Gymnasium Challenge environment."""

from __future__ import annotations

import math
from collections.abc import Sequence
from numbers import Integral
from typing import Any, ClassVar

import gymnasium as gym
import jax
import jax.numpy as jnp
import numpy as np
from gymnasium import error
from gymnasium.vector import AutoresetMode

from controller_learning.config import ProjectConfig
from controller_learning.envs._vehicle_driver import (
    AppliedActionBatch,
    VehicleBackend,
    create_vehicle_driver,
)
from controller_learning.envs.configuration import race_core_config_from_project
from controller_learning.envs.episode import (
    DeviceEpisodeIdentity,
    episode_identity_to_device,
    initialize_episode_identities,
    masked_next_episode_device,
    track_id_from_track,
)
from controller_learning.envs.observation import (
    action_space,
    batched_action_space,
    batched_observation_space,
    encode_batched_observation,
    observation_space,
)
from controller_learning.envs.race_core import (
    RaceState,
    RaceStep,
    masked_reset_race_state,
    reset_race_state,
    step_race_core,
)
from controller_learning.tracks.types import Track, TrackBatch, TrackCapacity, stack_tracks


def _maybe_reset_race_state(
    track_batch: TrackBatch,
    current: RaceState,
    mask: jax.Array,
) -> RaceState:
    reset_mask = jnp.asarray(mask, dtype=bool)
    return jax.lax.cond(
        jnp.any(reset_mask),
        lambda state: masked_reset_race_state(
            state,
            reset_race_state(track_batch),
            reset_mask,
        ),
        lambda state: state,
        current,
    )


def _normalize_device_actions(
    actions: jax.Array,
    pending_reset: jax.Array,
) -> tuple[jax.Array, jax.Array]:
    active = ~jnp.asarray(pending_reset, dtype=bool)
    converted = jnp.asarray(actions, dtype=jnp.float32)
    finite = jnp.all(jnp.isfinite(converted), axis=1)
    invalid = active & ~finite
    safe = jnp.where(active[:, None] & finite[:, None], converted, jnp.float32(0.0))
    return safe, invalid


def _positive_world_count(value: object) -> int:
    if isinstance(value, bool) or not isinstance(value, Integral):
        raise TypeError("num_envs must be a positive integer")
    count = int(value)
    if count < 1:
        raise ValueError("num_envs must be a positive integer")
    return count


def _selected_level(config: ProjectConfig, level_id: object):
    if isinstance(level_id, bool) or not isinstance(level_id, Integral):
        raise TypeError("level_id must be an integer")
    selected = next(
        (level for level in config.levels if level.level_id == int(level_id)),
        None,
    )
    if selected is None:
        available = sorted(level.level_id for level in config.levels)
        raise ValueError(f"unknown level_id {level_id!r}; available Levels are {available}")
    return selected


def _validated_tracks(
    tracks: object,
    *,
    num_envs: int,
    config: ProjectConfig,
    level_id: int,
) -> tuple[Track, ...]:
    if isinstance(tracks, (str, bytes)) or not isinstance(tracks, Sequence):
        raise TypeError("tracks must be a sequence containing one Track per world")
    values = tuple(tracks)
    if len(values) != num_envs:
        raise ValueError(
            "tracks must contain exactly one Track per world; "
            f"expected {num_envs}, got {len(values)}"
        )
    if not all(isinstance(track, Track) for track in values):
        raise TypeError("tracks must contain only immutable Track values")

    level = _selected_level(config, level_id)
    capacity = TrackCapacity(
        max_track_points=config.track.representation.max_track_points,
        max_checkpoints=config.track.representation.max_checkpoints,
    )
    expected_version = config.track.generator.generator_version
    for index, track in enumerate(values):
        if track.capacity != capacity:
            raise ValueError(
                f"tracks[{index}] capacity {track.capacity} does not match configured {capacity}"
            )
        if track.generator_version != expected_version:
            raise ValueError(
                f"tracks[{index}] generator version {track.generator_version!r} does not match "
                f"configured {expected_version!r}"
            )
        if not math.isclose(
            track.width_m,
            level.track_width_m,
            rel_tol=0.0,
            abs_tol=1.0e-6,
        ):
            raise ValueError(
                f"tracks[{index}] width {track.width_m} does not match Level {level.level_id} "
                f"width {level.track_width_m}"
            )
    return values


def _device_track_batch(tracks: tuple[Track, ...]) -> TrackBatch:
    return jax.tree.map(jnp.asarray, stack_tracks(tracks))


def _readonly_strings(values: Sequence[str]) -> np.ndarray:
    result = np.asarray(values, dtype=np.str_)
    result.setflags(write=False)
    return result


class VecCarRacingEnv(gym.vector.VectorEnv):
    """The sole vectorized Challenge state machine for vehicle racing.

    M4 receives already generated immutable tracks. Track-pool selection is intentionally deferred
    to M5, so an automatic episode reset reuses the same Track in each world.
    """

    metadata: ClassVar[dict[str, Any]] = {
        "render_modes": [],
        "render_fps": 20,
        "jax": True,
        "autoreset_mode": AutoresetMode.NEXT_STEP,
    }

    def __init__(
        self,
        *,
        num_envs: int,
        project_config: ProjectConfig,
        level_id: int,
        tracks: Sequence[Track],
        backend: VehicleBackend,
        render_mode: str | None = None,
    ) -> None:
        super().__init__()
        self.num_envs = _positive_world_count(num_envs)
        if not isinstance(project_config, ProjectConfig):
            raise TypeError("project_config must be a ProjectConfig")
        self.project_config = project_config
        self.level_id = _selected_level(project_config, level_id).level_id
        self._tracks = _validated_tracks(
            tracks,
            num_envs=self.num_envs,
            config=project_config,
            level_id=self.level_id,
        )
        if render_mode is not None:
            raise ValueError("M4 supports only headless rendering with render_mode=None")
        self.render_mode = render_mode
        self.metadata = dict(type(self).metadata)

        self.single_observation_space = observation_space(project_config)
        self.observation_space = batched_observation_space(project_config, self.num_envs)
        self.single_action_space = action_space(project_config)
        self.action_space = batched_action_space(project_config, self.num_envs)

        self._track_batch = _device_track_batch(self._tracks)
        self._start_pose = jnp.asarray(self._track_batch.start_pose, dtype=jnp.float32)
        self._race_config = race_core_config_from_project(project_config)
        self._vehicle_driver = create_vehicle_driver(
            backend,
            project_config.vehicle,
            num_worlds=self.num_envs,
        )
        self.backend = self._vehicle_driver.backend

        # Track values remain dynamic arguments. Different same-capacity tracks therefore reuse the
        # same executable rather than being embedded as constants in a per-track program.
        self._reset_race = jax.jit(reset_race_state)
        self._maybe_reset_race = jax.jit(_maybe_reset_race_state)
        self._advance_identity = jax.jit(masked_next_episode_device)
        self._step_race = jax.jit(
            lambda track_batch, state, position, invalid: step_race_core(
                track_batch,
                state,
                position,
                invalid,
                self._race_config,
            )
        )
        self._encode_observation = jax.jit(encode_batched_observation)
        self._normalize_actions = jax.jit(_normalize_device_actions)
        self._read_vehicle_state = self._vehicle_driver.read_state
        self._planar_position = lambda view: jnp.asarray(
            view.position_world_m,
            dtype=jnp.float32,
        )[..., :2]
        self._finalize_gpu_step = None
        if self.backend == "mjx_warp":
            self._read_vehicle_state = jax.jit(self._vehicle_driver.read_state)
            self._planar_position = jax.jit(lambda view: view.position_world_m[..., :2])

            def finalize_gpu_step(track_batch, vehicle_state, race_step, identity, pending):
                next_identity = masked_next_episode_device(identity, pending)
                next_vehicle = jax.lax.cond(
                    jnp.any(pending),
                    lambda current: self._vehicle_driver.masked_reset(
                        current,
                        pending,
                        self._start_pose,
                    ),
                    lambda current: current,
                    vehicle_state,
                )
                next_race = _maybe_reset_race_state(track_batch, race_step.state, pending)
                vehicle_view = self._vehicle_driver.read_state(next_vehicle)
                observation = encode_batched_observation(track_batch, next_race, vehicle_view)
                reward = jnp.where(pending, jnp.float32(0.0), race_step.reward).astype(jnp.float32)
                terminated = jnp.where(pending, False, race_step.terminated).astype(bool)
                truncated = jnp.where(pending, False, race_step.truncated).astype(bool)
                next_pending = terminated | truncated
                termination_reason = jnp.where(
                    pending,
                    jnp.int32(0),
                    race_step.termination_reason,
                )
                lap_completed = jnp.where(pending, False, race_step.success)
                lap_time_s = jnp.where(
                    lap_completed,
                    race_step.state.elapsed_steps.astype(jnp.float32) * self._control_dt_device,
                    jnp.float32(0.0),
                )
                return (
                    next_vehicle,
                    next_race,
                    next_identity,
                    next_pending,
                    observation,
                    reward,
                    terminated,
                    truncated,
                    termination_reason,
                    lap_completed,
                    lap_time_s,
                )

            self._control_dt_device = jnp.asarray(
                self.project_config.vehicle.simulation.control_dt_s,
                dtype=jnp.float32,
            )
            self._finalize_gpu_step = jax.jit(finalize_gpu_step)
        else:
            self._control_dt_device = jnp.asarray(
                self.project_config.vehicle.simulation.control_dt_s,
                dtype=jnp.float32,
            )

        self._vehicle_state: Any | None = None
        self._race_state: RaceState | None = None
        self._identity: DeviceEpisodeIdentity | None = None
        self._pending_reset = jnp.zeros(self.num_envs, dtype=bool)
        self._zero_actions = jnp.zeros((self.num_envs, 2), dtype=jnp.float32)
        self._track_ids = _readonly_strings([track_id_from_track(track) for track in self._tracks])
        self._benchmark_versions = _readonly_strings(
            [self.project_config.benchmark.version] * self.num_envs
        )
        self._last_race_step: RaceStep | None = None
        self._last_applied_action: AppliedActionBatch | None = None
        self._closed = False

    def _root_seed(self, seed: int | None) -> int:
        if seed is not None:
            if isinstance(seed, bool) or not isinstance(seed, Integral):
                raise TypeError("seed must be an integer in the uint32 range or None")
            root = int(seed)
            if not 0 <= root <= np.iinfo(np.uint32).max:
                raise ValueError("seed must be in the uint32 range")
            super().reset(seed=root)
            return root
        super().reset(seed=None)
        return int(self.np_random.integers(0, 2**32, dtype=np.uint32))

    @staticmethod
    def _validate_options(options: dict[str, Any] | None) -> None:
        if options is not None and not isinstance(options, dict):
            raise TypeError("reset options must be a dictionary or None")
        if options:
            raise ValueError("VecCarRacingEnv does not define reset options")

    def _observation(self) -> dict[str, jax.Array]:
        if self._vehicle_state is None or self._race_state is None:
            raise error.ResetNeeded("call reset before requesting an observation")
        vehicle_view = self._read_vehicle_state(self._vehicle_state)
        return self._encode_observation(
            self._track_batch,
            self._race_state,
            vehicle_view,
        )

    def reset(
        self,
        *,
        seed: int | None = None,
        options: dict[str, Any] | None = None,
    ) -> tuple[dict[str, jax.Array], dict[str, Any]]:
        """Reset every world to its injected Track start pose."""

        self._validate_options(options)
        root_seed = self._root_seed(seed)
        self._identity = episode_identity_to_device(
            initialize_episode_identities(root_seed, self.num_envs)
        )
        self._vehicle_state = self._vehicle_driver.initial_state(self._start_pose)
        self._race_state = self._reset_race(self._track_batch)
        self._pending_reset = jnp.zeros(self.num_envs, dtype=bool)
        self._last_race_step = None
        self._last_applied_action = None
        observation = self._observation()
        info = self._public_info(
            termination_reason=jnp.zeros(self.num_envs, dtype=jnp.int32),
            lap_completed=jnp.zeros(self.num_envs, dtype=bool),
            lap_time_s=jnp.zeros(self.num_envs, dtype=jnp.float32),
        )
        return observation, info

    def _public_info(
        self,
        *,
        termination_reason: jax.Array,
        lap_completed: jax.Array,
        lap_time_s: jax.Array,
    ) -> dict[str, Any]:
        if self._identity is None:
            raise error.ResetNeeded("call reset before requesting public info")
        return {
            "episode_seed": self._identity.episode_seed,
            "controller_seed": self._identity.controller_seed,
            "track_id": self._track_ids,
            "benchmark_version": self._benchmark_versions,
            "termination_reason": termination_reason,
            "lap_completed": lap_completed,
            "lap_time_s": lap_time_s,
        }

    def _safe_actions(self, actions: object) -> tuple[Any, Any]:
        """Convert public actions while assigning conversion failures only to active worlds."""

        active_device = ~jnp.asarray(self._pending_reset, dtype=bool)
        if isinstance(actions, jax.Array):
            if actions.shape != (self.num_envs, 2) or jnp.issubdtype(
                actions.dtype,
                jnp.complexfloating,
            ):
                return self._zero_actions, active_device
            try:
                converted_device = jnp.asarray(actions, dtype=jnp.float32)
            except (TypeError, ValueError, OverflowError):
                return self._zero_actions, active_device
            return self._normalize_actions(converted_device, self._pending_reset)

        try:
            source = np.asarray(actions)
            if source.dtype.kind == "c":
                raise TypeError("complex actions are not supported")
            converted = np.asarray(actions, dtype=np.float32)
        except (TypeError, ValueError, OverflowError):
            return self._zero_actions, active_device
        if converted.shape != (self.num_envs, 2):
            return self._zero_actions, active_device

        converted_device = jnp.asarray(converted, dtype=jnp.float32)
        return self._normalize_actions(converted_device, self._pending_reset)

    def step(
        self,
        actions: object,
    ) -> tuple[
        dict[str, jax.Array],
        jax.Array,
        jax.Array,
        jax.Array,
        dict[str, Any],
    ]:
        """Advance active worlds and apply strict Gymnasium NEXT_STEP masked reset."""

        if self._vehicle_state is None or self._race_state is None or self._identity is None:
            raise error.ResetNeeded("call reset before step")

        pending = self._pending_reset
        safe_actions, boundary_invalid = self._safe_actions(actions)

        # A single native batch transition is always executed. Pending worlds receive a safe zero
        # action and are replaced with exact initial state only after that shared computation.
        vehicle_step = self._vehicle_driver.step(self._vehicle_state, safe_actions)
        self._last_applied_action = vehicle_step.applied
        stepped_view = self._read_vehicle_state(vehicle_step.state)
        invalid = jnp.asarray(boundary_invalid) | jnp.asarray(
            vehicle_step.applied.invalid_action,
            dtype=bool,
        )
        race_step = self._step_race(
            self._track_batch,
            self._race_state,
            self._planar_position(stepped_view),
            invalid,
        )
        self._last_race_step = race_step

        if self.backend == "mjx_warp":
            assert self._finalize_gpu_step is not None
            (
                self._vehicle_state,
                self._race_state,
                self._identity,
                self._pending_reset,
                observation,
                reward,
                terminated,
                truncated,
                termination_reason,
                lap_completed,
                lap_time_s,
            ) = self._finalize_gpu_step(
                self._track_batch,
                vehicle_step.state,
                race_step,
                self._identity,
                pending,
            )
        else:
            self._identity = self._advance_identity(self._identity, pending)
            pending_host = np.asarray(pending, dtype=np.bool_)
            if np.any(pending_host):
                self._vehicle_state = self._vehicle_driver.masked_reset(
                    vehicle_step.state,
                    pending_host,
                    self._start_pose,
                )
            else:
                self._vehicle_state = vehicle_step.state
            self._race_state = self._maybe_reset_race(
                self._track_batch,
                race_step.state,
                pending,
            )
            reward = jnp.where(pending, jnp.float32(0.0), race_step.reward).astype(jnp.float32)
            terminated = jnp.where(pending, False, race_step.terminated).astype(bool)
            truncated = jnp.where(pending, False, race_step.truncated).astype(bool)
            self._pending_reset = terminated | truncated
            termination_reason = jnp.where(
                pending,
                jnp.int32(0),
                race_step.termination_reason,
            )
            lap_completed = jnp.where(pending, False, race_step.success)
            lap_time_s = jnp.where(
                lap_completed,
                race_step.state.elapsed_steps.astype(jnp.float32) * self._control_dt_device,
                jnp.float32(0.0),
            )
            observation = self._observation()
        info = self._public_info(
            termination_reason=termination_reason,
            lap_completed=lap_completed,
            lap_time_s=lap_time_s,
        )

        return observation, reward, terminated, truncated, info

    def render(self) -> None:
        """M4 is headless; the later replay renderer is intentionally separate."""

        return None

    def close(self) -> None:
        """Release backend-owned resources; repeated close calls are safe."""

        if not self._closed:
            self._vehicle_driver.close()
            self._closed = True


__all__ = ["VecCarRacingEnv"]
