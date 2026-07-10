"""Public-observation-only recording for one completed Controller episode."""

from __future__ import annotations

import hashlib
import json
import math
import os
import stat
import tempfile
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType
from typing import TYPE_CHECKING, Any, Final

import numpy as np
from numpy.typing import NDArray

from controller_learning.control import EpisodeRunResult, run_controller_episode
from controller_learning.envs.episode import PUBLIC_INFO_KEYS
from controller_learning.envs.observation import OBSERVATION_KEYS

if TYPE_CHECKING:
    import gymnasium as gym

TRAJECTORY_SCHEMA_VERSION: Final = "controller-learning-trajectory-v1"
MAX_TRAJECTORY_JSON_BYTES: Final = 64 * 1024 * 1024
_UINT32_MAX = int(np.iinfo(np.uint32).max)
_DYNAMIC_OBSERVATION_KEYS: Final = (
    "position",
    "yaw",
    "velocity_body",
    "yaw_rate",
    "steering_angle",
    "track_progress",
)
_TRACK_OBSERVATION_KEYS: Final = (
    "centerline",
    "left_boundary",
    "right_boundary",
    "track_mask",
    "track_length",
)

PublicInfo = Mapping[str, int | float | bool | str]


def _readonly_array(
    value: object,
    *,
    dtype: np.dtype,
    shape: tuple[int, ...],
    name: str,
    finite: bool = True,
) -> NDArray:
    try:
        with np.errstate(over="ignore", invalid="ignore"):
            source = np.asarray(value, dtype=dtype)
    except (OverflowError, TypeError, ValueError) as error:
        raise ValueError(f"{name} must be numeric") from error
    if source.shape != shape:
        raise ValueError(f"{name} must have shape {shape}, got {source.shape}")
    if finite and not np.isfinite(source).all():
        raise ValueError(f"{name} must contain only finite values")
    result = np.array(source, dtype=dtype, copy=True)
    result.setflags(write=False)
    return result


def _readonly_float32(
    value: object,
    *,
    shape: tuple[int, ...],
    name: str,
) -> NDArray[np.float32]:
    return _readonly_array(
        value,
        dtype=np.dtype(np.float32),
        shape=shape,
        name=name,
    )


def _public_info(value: object, *, source: str) -> PublicInfo:
    if not isinstance(value, Mapping):
        raise TypeError(f"{source} info must be a mapping")
    missing = tuple(key for key in PUBLIC_INFO_KEYS if key not in value)
    if missing:
        raise ValueError(f"{source} info is missing public field(s): {', '.join(missing)}")

    expected_types: dict[str, type] = {
        "episode_seed": int,
        "controller_seed": int,
        "track_id": int,
        "benchmark_version": str,
        "termination_reason": int,
        "lap_completed": bool,
        "lap_time_s": float,
    }
    result: dict[str, int | float | bool | str] = {}
    for key in PUBLIC_INFO_KEYS:
        item = value[key]
        expected = expected_types[key]
        if type(item) is not expected:
            raise TypeError(
                f"{source} info field {key!r} must have type {expected.__name__}, "
                f"got {type(item).__name__}"
            )
        result[key] = item

    for key in ("episode_seed", "controller_seed", "track_id"):
        item = int(result[key])
        if not 0 <= item <= _UINT32_MAX:
            raise ValueError(f"{source} info field {key!r} must fit in uint32")
    if not result["benchmark_version"]:
        raise ValueError(f"{source} info field 'benchmark_version' cannot be empty")
    reason = int(result["termination_reason"])
    if reason not in range(5):
        raise ValueError(f"{source} info field 'termination_reason' must be in [0, 4]")
    lap_time_s = float(result["lap_time_s"])
    if not math.isfinite(lap_time_s) or lap_time_s < 0.0:
        raise ValueError(f"{source} info field 'lap_time_s' must be finite and non-negative")
    return MappingProxyType(result)


def _validated_observation(value: object, *, source: str) -> dict[str, NDArray]:
    if not isinstance(value, Mapping):
        raise TypeError(f"{source} observation must be a mapping")
    expected = set(OBSERVATION_KEYS)
    actual = set(value)
    if actual != expected:
        missing = sorted(expected - actual, key=repr)
        extra = sorted(actual - expected, key=repr)
        raise ValueError(
            f"{source} observation keys do not match the public schema; "
            f"missing={missing}, extra={extra}"
        )

    try:
        centerline_source = np.asarray(value["centerline"])
    except (TypeError, ValueError) as error:
        raise ValueError(f"{source} observation field 'centerline' must be numeric") from error
    if centerline_source.ndim != 2 or centerline_source.shape[1:] != (2,):
        raise ValueError(
            f"{source} observation field 'centerline' must have shape (N, 2), "
            f"got {centerline_source.shape}"
        )
    point_count = int(centerline_source.shape[0])
    if point_count < 2:
        raise ValueError(f"{source} observation track capacity must contain at least two points")

    result: dict[str, NDArray] = {
        "position": _readonly_float32(
            value["position"], shape=(2,), name=f"{source} observation position"
        ),
        "yaw": _readonly_float32(value["yaw"], shape=(), name=f"{source} observation yaw"),
        "velocity_body": _readonly_float32(
            value["velocity_body"], shape=(2,), name=f"{source} observation velocity_body"
        ),
        "yaw_rate": _readonly_float32(
            value["yaw_rate"], shape=(), name=f"{source} observation yaw_rate"
        ),
        "steering_angle": _readonly_float32(
            value["steering_angle"], shape=(), name=f"{source} observation steering_angle"
        ),
        "track_progress": _readonly_float32(
            value["track_progress"], shape=(), name=f"{source} observation track_progress"
        ),
        "centerline": _readonly_float32(
            value["centerline"],
            shape=(point_count, 2),
            name=f"{source} observation centerline",
        ),
        "left_boundary": _readonly_float32(
            value["left_boundary"],
            shape=(point_count, 2),
            name=f"{source} observation left_boundary",
        ),
        "right_boundary": _readonly_float32(
            value["right_boundary"],
            shape=(point_count, 2),
            name=f"{source} observation right_boundary",
        ),
        "track_length": _readonly_float32(
            value["track_length"], shape=(), name=f"{source} observation track_length"
        ),
    }
    mask_source = np.asarray(value["track_mask"])
    if mask_source.shape != (point_count,):
        raise ValueError(
            f"{source} observation field 'track_mask' must have shape ({point_count},), "
            f"got {mask_source.shape}"
        )
    if not np.isin(mask_source, (0, 1)).all():
        raise ValueError(f"{source} observation field 'track_mask' must contain only 0 or 1")
    mask = np.array(mask_source, dtype=np.bool_, copy=True)
    mask.setflags(write=False)
    if int(np.count_nonzero(mask)) < 2:
        raise ValueError(f"{source} observation track_mask must select at least two points")
    result["track_mask"] = mask

    progress = float(result["track_progress"])
    if not 0.0 <= progress <= 1.0:
        raise ValueError(f"{source} observation track_progress must lie in [0, 1]")
    if float(result["track_length"]) <= 0.0:
        raise ValueError(f"{source} observation track_length must be positive")
    return result


@dataclass(frozen=True, slots=True)
class EpisodeTrajectory:
    """One immutable, terminal episode reconstructed only from public values."""

    reset_info: PublicInfo
    final_info: PublicInfo
    centerline_m: NDArray[np.float32]
    left_boundary_m: NDArray[np.float32]
    right_boundary_m: NDArray[np.float32]
    track_mask: NDArray[np.bool_]
    track_length_m: float
    position_m: NDArray[np.float32]
    yaw_rad: NDArray[np.float32]
    velocity_body_mps: NDArray[np.float32]
    yaw_rate_rad_s: NDArray[np.float32]
    steering_angle_rad: NDArray[np.float32]
    track_progress: NDArray[np.float32]
    action: NDArray[np.float32]
    reward: NDArray[np.float32]
    terminated: NDArray[np.bool_]
    truncated: NDArray[np.bool_]
    schema_version: str = TRAJECTORY_SCHEMA_VERSION

    def __post_init__(self) -> None:
        if self.schema_version != TRAJECTORY_SCHEMA_VERSION:
            raise ValueError(
                f"schema_version must be {TRAJECTORY_SCHEMA_VERSION!r}, got {self.schema_version!r}"
            )
        reset_info = _public_info(self.reset_info, source="reset")
        final_info = _public_info(self.final_info, source="final")
        identity_keys = ("episode_seed", "controller_seed", "track_id", "benchmark_version")
        if any(reset_info[key] != final_info[key] for key in identity_keys):
            raise ValueError("reset and final info must preserve one public episode identity")
        if (
            reset_info["termination_reason"] != 0
            or reset_info["lap_completed"] is not False
            or reset_info["lap_time_s"] != 0.0
        ):
            raise ValueError("reset info must contain neutral terminal fields")

        centerline_source = np.asarray(self.centerline_m)
        if centerline_source.ndim != 2 or centerline_source.shape[1:] != (2,):
            raise ValueError("centerline_m must have shape (N, 2)")
        point_count = int(centerline_source.shape[0])
        if point_count < 2:
            raise ValueError("track capacity must contain at least two points")
        centerline = _readonly_float32(
            self.centerline_m, shape=(point_count, 2), name="centerline_m"
        )
        left_boundary = _readonly_float32(
            self.left_boundary_m, shape=(point_count, 2), name="left_boundary_m"
        )
        right_boundary = _readonly_float32(
            self.right_boundary_m, shape=(point_count, 2), name="right_boundary_m"
        )
        mask_source = np.asarray(self.track_mask)
        if mask_source.shape != (point_count,) or not np.isin(mask_source, (0, 1)).all():
            raise ValueError("track_mask must have shape (N,) and contain only 0 or 1")
        track_mask = np.array(mask_source, dtype=np.bool_, copy=True)
        track_mask.setflags(write=False)
        if int(np.count_nonzero(track_mask)) < 2:
            raise ValueError("track_mask must select at least two points")
        track_length_m = float(self.track_length_m)
        if not math.isfinite(track_length_m) or track_length_m <= 0.0:
            raise ValueError("track_length_m must be finite and positive")

        position_source = np.asarray(self.position_m)
        if position_source.ndim != 2 or position_source.shape[1:] != (2,):
            raise ValueError("position_m must have shape (frames, 2)")
        frame_count = int(position_source.shape[0])
        if frame_count < 2:
            raise ValueError("a completed trajectory must contain at least two frames")
        step_count = frame_count - 1
        arrays: dict[str, NDArray] = {
            "position_m": _readonly_float32(
                self.position_m, shape=(frame_count, 2), name="position_m"
            ),
            "yaw_rad": _readonly_float32(self.yaw_rad, shape=(frame_count,), name="yaw_rad"),
            "velocity_body_mps": _readonly_float32(
                self.velocity_body_mps,
                shape=(frame_count, 2),
                name="velocity_body_mps",
            ),
            "yaw_rate_rad_s": _readonly_float32(
                self.yaw_rate_rad_s, shape=(frame_count,), name="yaw_rate_rad_s"
            ),
            "steering_angle_rad": _readonly_float32(
                self.steering_angle_rad,
                shape=(frame_count,),
                name="steering_angle_rad",
            ),
            "track_progress": _readonly_float32(
                self.track_progress, shape=(frame_count,), name="track_progress"
            ),
            "action": _readonly_float32(self.action, shape=(step_count, 2), name="action"),
            "reward": _readonly_float32(self.reward, shape=(step_count,), name="reward"),
        }
        if np.any((arrays["track_progress"] < 0.0) | (arrays["track_progress"] > 1.0)):
            raise ValueError("track_progress must lie in [0, 1]")
        flags: dict[str, NDArray[np.bool_]] = {}
        for name, value in (("terminated", self.terminated), ("truncated", self.truncated)):
            source = np.asarray(value)
            if source.shape != (step_count,):
                raise ValueError(f"{name} must have shape ({step_count},)")
            if not np.isin(source, (0, 1)).all():
                raise ValueError(f"{name} must contain only boolean values")
            result = np.array(source, dtype=np.bool_, copy=True)
            result.setflags(write=False)
            flags[name] = result
        terminal = flags["terminated"] | flags["truncated"]
        if np.any(terminal[:-1]) or not bool(terminal[-1]):
            raise ValueError("only the final trajectory transition may be terminal")
        if bool(flags["terminated"][-1]) == bool(flags["truncated"][-1]):
            raise ValueError("the final transition must set exactly one terminal flag")

        reason = int(final_info["termination_reason"])
        success = bool(final_info["lap_completed"])
        lap_time_s = float(final_info["lap_time_s"])
        if reason == 0:
            raise ValueError("final info must contain a terminal reason")
        if (reason == 4) != bool(flags["truncated"][-1]):
            raise ValueError("TIMEOUT must match the final truncated flag")
        if success != (reason == 1):
            raise ValueError("lap_completed must match the SUCCESS termination reason")
        if success and lap_time_s <= 0.0:
            raise ValueError("a successful trajectory must have a positive lap_time_s")
        if not success and lap_time_s != 0.0:
            raise ValueError("an unsuccessful trajectory must have a neutral lap_time_s")

        object.__setattr__(self, "reset_info", reset_info)
        object.__setattr__(self, "final_info", final_info)
        object.__setattr__(self, "centerline_m", centerline)
        object.__setattr__(self, "left_boundary_m", left_boundary)
        object.__setattr__(self, "right_boundary_m", right_boundary)
        object.__setattr__(self, "track_mask", track_mask)
        object.__setattr__(self, "track_length_m", track_length_m)
        for name, array in arrays.items():
            object.__setattr__(self, name, array)
        for name, array in flags.items():
            object.__setattr__(self, name, array)

    @property
    def frame_count(self) -> int:
        """Number of recorded observations, including the reset frame."""

        return int(self.position_m.shape[0])

    @property
    def step_count(self) -> int:
        """Number of recorded environment transitions."""

        return int(self.action.shape[0])

    @property
    def total_reward(self) -> float:
        """Float64 sum of the recorded float32 rewards."""

        return float(np.sum(self.reward, dtype=np.float64))

    def observation(self, frame_index: int) -> Mapping[str, NDArray]:
        """Reconstruct one immutable public observation for deterministic replay."""

        if isinstance(frame_index, bool) or not isinstance(frame_index, int):
            raise TypeError("frame_index must be an integer")
        if not -self.frame_count <= frame_index < self.frame_count:
            raise IndexError(f"frame_index {frame_index} is outside {self.frame_count} frames")
        index = frame_index % self.frame_count
        track_length = np.asarray(self.track_length_m, dtype=np.float32)
        track_length.setflags(write=False)
        return MappingProxyType(
            {
                "position": self.position_m[index],
                "yaw": self.yaw_rad[index],
                "velocity_body": self.velocity_body_mps[index],
                "yaw_rate": self.yaw_rate_rad_s[index],
                "steering_angle": self.steering_angle_rad[index],
                "track_progress": self.track_progress[index],
                "centerline": self.centerline_m,
                "left_boundary": self.left_boundary_m,
                "right_boundary": self.right_boundary_m,
                "track_mask": self.track_mask,
                "track_length": track_length,
            }
        )


@dataclass(frozen=True, slots=True)
class RecordedControllerEpisode:
    """A normal Runner result paired with its public trajectory."""

    result: EpisodeRunResult
    trajectory: EpisodeTrajectory

    def __post_init__(self) -> None:
        if not isinstance(self.result, EpisodeRunResult):
            raise TypeError("result must be an EpisodeRunResult")
        if not isinstance(self.trajectory, EpisodeTrajectory):
            raise TypeError("trajectory must be an EpisodeTrajectory")
        if self.result.steps != self.trajectory.step_count:
            raise ValueError("Runner and trajectory step counts do not match")
        if not math.isclose(
            self.result.total_reward,
            self.trajectory.total_reward,
            rel_tol=0.0,
            abs_tol=1.0e-5 * max(1, self.result.steps),
        ):
            raise ValueError("Runner and trajectory total rewards do not match")
        if self.result.terminated != bool(self.trajectory.terminated[-1]):
            raise ValueError("Runner and trajectory terminated flags do not match")
        if self.result.truncated != bool(self.trajectory.truncated[-1]):
            raise ValueError("Runner and trajectory truncated flags do not match")
        if dict(self.result.final_info) != dict(self.trajectory.final_info):
            raise ValueError("Runner and trajectory final public info do not match")


class _EpisodeRecorder:
    """Mutable implementation detail kept outside the public Controller boundary."""

    def __init__(self) -> None:
        self._reset_info: PublicInfo | None = None
        self._track: dict[str, NDArray] | None = None
        self._frames: dict[str, list[NDArray]] = {key: [] for key in _DYNAMIC_OBSERVATION_KEYS}
        self._actions: list[NDArray[np.float32]] = []
        self._rewards: list[float] = []
        self._terminated: list[bool] = []
        self._truncated: list[bool] = []
        self._final_info: PublicInfo | None = None

    def reset(self, observation: object, info: object) -> None:
        if self._reset_info is not None:
            raise RuntimeError("a trajectory recorder can record exactly one episode")
        values = _validated_observation(observation, source="reset")
        reset_info = _public_info(info, source="reset")
        if (
            reset_info["termination_reason"] != 0
            or reset_info["lap_completed"] is not False
            or reset_info["lap_time_s"] != 0.0
        ):
            raise ValueError("reset info must contain neutral terminal fields")
        self._reset_info = reset_info
        self._track = {key: values[key] for key in _TRACK_OBSERVATION_KEYS}
        self._append_frame(values)

    def step(
        self,
        action: object,
        observation: object,
        reward: object,
        terminated: object,
        truncated: object,
        info: object,
    ) -> None:
        if self._reset_info is None or self._track is None:
            raise RuntimeError("trajectory recording must start with reset")
        if self._final_info is not None:
            raise RuntimeError("cannot append a transition after episode completion")
        values = _validated_observation(observation, source="step")
        mismatched = tuple(
            key
            for key in _TRACK_OBSERVATION_KEYS
            if not np.array_equal(values[key], self._track[key])
        )
        if mismatched:
            raise ValueError(
                "public track observation changed within one episode: " + ", ".join(mismatched)
            )
        action_array = _readonly_float32(action, shape=(2,), name="action")
        try:
            reward_value = float(reward)
        except (TypeError, ValueError) as error:
            raise TypeError("reward must be a real scalar") from error
        if not math.isfinite(reward_value):
            raise ValueError("reward must be finite")
        if type(terminated) is not bool or type(truncated) is not bool:
            raise TypeError("terminated and truncated must be booleans")
        if terminated and truncated:
            raise ValueError("a transition cannot be both terminated and truncated")
        step_info = _public_info(info, source="step")
        identity_keys = ("episode_seed", "controller_seed", "track_id", "benchmark_version")
        if any(step_info[key] != self._reset_info[key] for key in identity_keys):
            raise ValueError("step info changed the public episode identity")
        reason = int(step_info["termination_reason"])
        is_terminal = terminated or truncated
        if (reason != 0) != is_terminal:
            raise ValueError("termination_reason must match the transition terminal flags")
        if (reason == 4) != truncated:
            raise ValueError("TIMEOUT must match the truncated flag")
        if bool(step_info["lap_completed"]) != (reason == 1):
            raise ValueError("lap_completed must match the SUCCESS termination reason")
        lap_time_s = float(step_info["lap_time_s"])
        if (reason == 1 and lap_time_s <= 0.0) or (reason != 1 and lap_time_s != 0.0):
            raise ValueError("lap_time_s must be positive only for SUCCESS")

        self._actions.append(action_array)
        self._rewards.append(reward_value)
        self._terminated.append(terminated)
        self._truncated.append(truncated)
        self._append_frame(values)
        if is_terminal:
            self._final_info = step_info

    def _append_frame(self, observation: Mapping[str, NDArray]) -> None:
        for key in _DYNAMIC_OBSERVATION_KEYS:
            self._frames[key].append(observation[key])

    def finish(self) -> EpisodeTrajectory:
        if self._reset_info is None or self._track is None:
            raise RuntimeError("trajectory recorder did not observe a reset")
        if self._final_info is None:
            raise RuntimeError("trajectory recorder did not observe a terminal transition")
        return EpisodeTrajectory(
            reset_info=self._reset_info,
            final_info=self._final_info,
            centerline_m=self._track["centerline"],
            left_boundary_m=self._track["left_boundary"],
            right_boundary_m=self._track["right_boundary"],
            track_mask=self._track["track_mask"],
            track_length_m=float(self._track["track_length"]),
            position_m=np.stack(self._frames["position"], axis=0),
            yaw_rad=np.asarray(self._frames["yaw"], dtype=np.float32),
            velocity_body_mps=np.stack(self._frames["velocity_body"], axis=0),
            yaw_rate_rad_s=np.asarray(self._frames["yaw_rate"], dtype=np.float32),
            steering_angle_rad=np.asarray(self._frames["steering_angle"], dtype=np.float32),
            track_progress=np.asarray(self._frames["track_progress"], dtype=np.float32),
            action=np.stack(self._actions, axis=0),
            reward=np.asarray(self._rewards, dtype=np.float32),
            terminated=np.asarray(self._terminated, dtype=np.bool_),
            truncated=np.asarray(self._truncated, dtype=np.bool_),
        )


class _RecordingEnvironment:
    def __init__(self, env: object, recorder: _EpisodeRecorder) -> None:
        self._env = env
        self._recorder = recorder

    @property
    def unwrapped(self) -> object:
        try:
            return self._env.unwrapped  # type: ignore[attr-defined]
        except AttributeError as error:
            raise TypeError("env must expose the Gymnasium unwrapped environment") from error

    def reset(self, **kwargs: object) -> tuple[object, object]:
        observation, info = self._env.reset(**kwargs)  # type: ignore[attr-defined]
        self._recorder.reset(observation, info)
        return observation, info

    def step(self, action: object) -> tuple[object, object, object, object, object]:
        transition = self._env.step(action)  # type: ignore[attr-defined]
        if not isinstance(transition, tuple) or len(transition) != 5:
            raise TypeError("env.step must return the five-value Gymnasium transition tuple")
        observation, reward, terminated, truncated, info = transition
        self._recorder.step(action, observation, reward, terminated, truncated, info)
        return transition

    def render(self) -> object:
        return self._env.render()  # type: ignore[attr-defined]


def record_controller_episode(
    env: gym.Env,
    controller_directory: str | Path,
    reset_seed: int,
    render: bool = False,
    max_steps: int | None = None,
    *,
    reset_options: Mapping[str, Any] | None = None,
) -> RecordedControllerEpisode:
    """Run the normal batch-one Controller path and retain one public trajectory.

    The proxy observes only values already returned by ``env.reset`` and ``env.step``. It never
    reads ``env.unwrapped`` except to preserve the Runner's existing Challenge-config boundary.
    Wrapper diagnostics are discarded; persisted info contains exactly ``PUBLIC_INFO_KEYS``.
    The environment remains owned by the caller and is not closed by this function.
    """

    recorder = _EpisodeRecorder()
    proxy = _RecordingEnvironment(env, recorder)
    result = run_controller_episode(
        proxy,  # type: ignore[arg-type]
        controller_directory,
        reset_seed,
        render=render,
        max_steps=max_steps,
        reset_options=reset_options,
    )
    return RecordedControllerEpisode(result=result, trajectory=recorder.finish())


@dataclass(frozen=True, slots=True)
class TrajectoryArtifact:
    """Identity of one atomically written canonical trajectory JSON file."""

    path: Path
    sha256: str
    size_bytes: int

    def __post_init__(self) -> None:
        if not isinstance(self.path, Path):
            raise TypeError("path must be a Path")
        if (
            not isinstance(self.sha256, str)
            or len(self.sha256) != 64
            or any(character not in "0123456789abcdef" for character in self.sha256)
        ):
            raise ValueError("sha256 must be a lowercase 64-character digest")
        if isinstance(self.size_bytes, bool) or not isinstance(self.size_bytes, int):
            raise TypeError("size_bytes must be an integer")
        if not 0 < self.size_bytes <= MAX_TRAJECTORY_JSON_BYTES:
            raise ValueError(f"size_bytes must be in [1, {MAX_TRAJECTORY_JSON_BYTES}]")


def _info_payload(info: PublicInfo) -> dict[str, int | float | bool | str]:
    return {key: info[key] for key in PUBLIC_INFO_KEYS}


def _trajectory_payload(trajectory: EpisodeTrajectory) -> dict[str, object]:
    if not isinstance(trajectory, EpisodeTrajectory):
        raise TypeError("trajectory must be an EpisodeTrajectory")
    return {
        "schema_version": trajectory.schema_version,
        "reset_info": _info_payload(trajectory.reset_info),
        "final_info": _info_payload(trajectory.final_info),
        "track": {
            "centerline_m": trajectory.centerline_m.tolist(),
            "left_boundary_m": trajectory.left_boundary_m.tolist(),
            "right_boundary_m": trajectory.right_boundary_m.tolist(),
            "track_mask": trajectory.track_mask.astype(np.uint8).tolist(),
            "track_length_m": trajectory.track_length_m,
        },
        "frames": {
            "position_m": trajectory.position_m.tolist(),
            "yaw_rad": trajectory.yaw_rad.tolist(),
            "velocity_body_mps": trajectory.velocity_body_mps.tolist(),
            "yaw_rate_rad_s": trajectory.yaw_rate_rad_s.tolist(),
            "steering_angle_rad": trajectory.steering_angle_rad.tolist(),
            "track_progress": trajectory.track_progress.tolist(),
        },
        "transitions": {
            "action": trajectory.action.tolist(),
            "reward": trajectory.reward.tolist(),
            "terminated": trajectory.terminated.astype(np.uint8).tolist(),
            "truncated": trajectory.truncated.astype(np.uint8).tolist(),
        },
    }


def _canonical_json_bytes(payload: object) -> bytes:
    return (
        json.dumps(
            payload,
            ensure_ascii=True,
            allow_nan=False,
            sort_keys=True,
            separators=(",", ":"),
        )
        + "\n"
    ).encode("utf-8")


def _atomic_write(path: Path, content: bytes) -> None:
    if path.is_symlink():
        raise ValueError("trajectory output_path cannot be a symbolic link")
    if path.exists() and not path.is_file():
        raise ValueError("trajectory output_path must be a regular file or absent")
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.parent.is_symlink() or not path.parent.is_dir():
        raise ValueError("trajectory output parent must be a real directory")
    temporary_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="wb",
            dir=path.parent,
            prefix=f".{path.name}.",
            suffix=".tmp",
            delete=False,
        ) as temporary:
            temporary_path = Path(temporary.name)
            temporary.write(content)
            temporary.flush()
            os.fchmod(temporary.fileno(), 0o644)
            os.fsync(temporary.fileno())
        os.replace(temporary_path, path)
        directory_descriptor = os.open(path.parent, os.O_RDONLY | os.O_DIRECTORY)
        try:
            os.fsync(directory_descriptor)
        finally:
            os.close(directory_descriptor)
        if path.read_bytes() != content:
            raise OSError("trajectory artifact readback did not match written bytes")
    finally:
        if temporary_path is not None:
            temporary_path.unlink(missing_ok=True)


def write_trajectory_json(
    trajectory: EpisodeTrajectory,
    output_path: str | Path,
) -> TrajectoryArtifact:
    """Atomically write a deterministic, canonical public trajectory artifact."""

    path = Path(output_path)
    if path.suffix.lower() != ".json":
        raise ValueError("trajectory output_path must use the .json suffix")
    content = _canonical_json_bytes(_trajectory_payload(trajectory))
    if len(content) > MAX_TRAJECTORY_JSON_BYTES:
        raise ValueError(f"trajectory artifact exceeds {MAX_TRAJECTORY_JSON_BYTES} bytes")
    _atomic_write(path, content)
    return TrajectoryArtifact(
        path=path,
        sha256=hashlib.sha256(content).hexdigest(),
        size_bytes=len(content),
    )


def _exact_object(value: object, *, name: str, keys: set[str]) -> Mapping[str, object]:
    if not isinstance(value, Mapping):
        raise TypeError(f"{name} must be a JSON object")
    actual = set(value)
    if actual != keys:
        raise ValueError(
            f"{name} keys do not match schema; missing={sorted(keys - actual)}, "
            f"extra={sorted(actual - keys)}"
        )
    return value


def _reject_json_constant(value: str) -> None:
    raise ValueError(f"non-finite JSON constant is forbidden: {value}")


def _reject_duplicate_keys(pairs: list[tuple[str, object]]) -> dict[str, object]:
    result: dict[str, object] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate JSON key is forbidden: {key!r}")
        result[key] = value
    return result


def load_trajectory_json(
    input_path: str | Path,
    *,
    expected_sha256: str | None = None,
) -> EpisodeTrajectory:
    """Load and strictly validate one canonical trajectory JSON artifact."""

    path = Path(input_path)
    try:
        metadata = path.lstat()
    except FileNotFoundError:
        raise
    if stat.S_ISLNK(metadata.st_mode) or not stat.S_ISREG(metadata.st_mode):
        raise ValueError("trajectory input_path must be a non-symlink regular file")
    if metadata.st_size <= 0 or metadata.st_size > MAX_TRAJECTORY_JSON_BYTES:
        raise ValueError(f"trajectory input size must be in [1, {MAX_TRAJECTORY_JSON_BYTES}] bytes")
    descriptor = os.open(path, os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0))
    try:
        opened_metadata = os.fstat(descriptor)
        if not stat.S_ISREG(opened_metadata.st_mode):
            raise ValueError("trajectory input_path must remain a regular file")
        if opened_metadata.st_size != metadata.st_size:
            raise ValueError("trajectory input_path changed while it was being opened")
        with os.fdopen(descriptor, "rb", closefd=False) as stream:
            content = stream.read(MAX_TRAJECTORY_JSON_BYTES + 1)
    finally:
        os.close(descriptor)
    if len(content) != opened_metadata.st_size:
        raise ValueError("trajectory input_path changed while it was being read")
    digest = hashlib.sha256(content).hexdigest()
    if expected_sha256 is not None:
        if (
            not isinstance(expected_sha256, str)
            or len(expected_sha256) != 64
            or any(character not in "0123456789abcdef" for character in expected_sha256)
        ):
            raise ValueError("expected_sha256 must be a lowercase 64-character SHA-256 digest")
        if digest != expected_sha256:
            raise ValueError("trajectory artifact SHA-256 does not match expected_sha256")
    payload = json.loads(
        content,
        parse_constant=_reject_json_constant,
        object_pairs_hook=_reject_duplicate_keys,
    )
    root = _exact_object(
        payload,
        name="trajectory",
        keys={"schema_version", "reset_info", "final_info", "track", "frames", "transitions"},
    )
    track = _exact_object(
        root["track"],
        name="track",
        keys={
            "centerline_m",
            "left_boundary_m",
            "right_boundary_m",
            "track_mask",
            "track_length_m",
        },
    )
    frames = _exact_object(
        root["frames"],
        name="frames",
        keys={
            "position_m",
            "yaw_rad",
            "velocity_body_mps",
            "yaw_rate_rad_s",
            "steering_angle_rad",
            "track_progress",
        },
    )
    transitions = _exact_object(
        root["transitions"],
        name="transitions",
        keys={"action", "reward", "terminated", "truncated"},
    )
    trajectory = EpisodeTrajectory(
        schema_version=root["schema_version"],  # type: ignore[arg-type]
        reset_info=root["reset_info"],  # type: ignore[arg-type]
        final_info=root["final_info"],  # type: ignore[arg-type]
        centerline_m=track["centerline_m"],  # type: ignore[arg-type]
        left_boundary_m=track["left_boundary_m"],  # type: ignore[arg-type]
        right_boundary_m=track["right_boundary_m"],  # type: ignore[arg-type]
        track_mask=track["track_mask"],  # type: ignore[arg-type]
        track_length_m=track["track_length_m"],  # type: ignore[arg-type]
        position_m=frames["position_m"],  # type: ignore[arg-type]
        yaw_rad=frames["yaw_rad"],  # type: ignore[arg-type]
        velocity_body_mps=frames["velocity_body_mps"],  # type: ignore[arg-type]
        yaw_rate_rad_s=frames["yaw_rate_rad_s"],  # type: ignore[arg-type]
        steering_angle_rad=frames["steering_angle_rad"],  # type: ignore[arg-type]
        track_progress=frames["track_progress"],  # type: ignore[arg-type]
        action=transitions["action"],  # type: ignore[arg-type]
        reward=transitions["reward"],  # type: ignore[arg-type]
        terminated=transitions["terminated"],  # type: ignore[arg-type]
        truncated=transitions["truncated"],  # type: ignore[arg-type]
    )
    if content != _canonical_json_bytes(_trajectory_payload(trajectory)):
        raise ValueError("trajectory JSON is valid but not in canonical serialized form")
    return trajectory


__all__ = [
    "MAX_TRAJECTORY_JSON_BYTES",
    "TRAJECTORY_SCHEMA_VERSION",
    "EpisodeTrajectory",
    "RecordedControllerEpisode",
    "TrajectoryArtifact",
    "load_trajectory_json",
    "record_controller_episode",
    "write_trajectory_json",
]
