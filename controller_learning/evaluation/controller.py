"""Sequential evaluation of trusted Controller plugins on fixed Track batches."""

from __future__ import annotations

import math
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from numbers import Integral, Real
from pathlib import Path
from typing import Literal, Protocol, TypeAlias

import numpy as np
from numpy.typing import NDArray

from controller_learning.config import ProjectConfig
from controller_learning.control import EpisodeRunResult, run_controller_episode
from controller_learning.envs.car_racing import CarRacingEnv
from controller_learning.tracks.assets import validate_track_batch
from controller_learning.tracks.types import TrackBatch, track_from_batch_row

DEFAULT_COMPUTE_DEADLINE_S = 0.05
_UINT32_MAX = int(np.iinfo(np.uint32).max)


class _ClosableEnvironment(Protocol):
    """The only environment lifecycle operation owned by this evaluator."""

    def close(self) -> None: ...


EnvironmentFactory: TypeAlias = Callable[..., _ClosableEnvironment]
EpisodeRunner: TypeAlias = Callable[..., EpisodeRunResult]
EvaluationBackend: TypeAlias = Literal["cpu_reference", "mjx_warp"]


def _finite_nonnegative(value: object, *, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, Real):
        raise TypeError(f"{name} must be a real number")
    result = float(value)
    if not math.isfinite(result) or result < 0.0:
        raise ValueError(f"{name} must be finite and non-negative")
    return result


def _positive_finite(value: object, *, name: str) -> float:
    result = _finite_nonnegative(value, name=name)
    if result <= 0.0:
        raise ValueError(f"{name} must be positive")
    return result


def _nonnegative_integer(value: object, *, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, Integral):
        raise TypeError(f"{name} must be an integer")
    result = int(value)
    if result < 0:
        raise ValueError(f"{name} must be non-negative")
    return result


@dataclass(frozen=True, slots=True)
class TimingSummary:
    """Percentiles and deadline misses for a non-empty compute-time sample."""

    sample_count: int
    deadline_s: float
    p50_s: float
    p95_s: float
    p99_s: float
    max_s: float
    deadline_miss_count: int
    deadline_miss_rate: float

    def __post_init__(self) -> None:
        sample_count = _nonnegative_integer(self.sample_count, name="sample_count")
        if sample_count == 0:
            raise ValueError("sample_count must be positive")
        deadline_s = _positive_finite(self.deadline_s, name="deadline_s")
        percentiles = tuple(
            _finite_nonnegative(value, name=name)
            for name, value in (
                ("p50_s", self.p50_s),
                ("p95_s", self.p95_s),
                ("p99_s", self.p99_s),
                ("max_s", self.max_s),
            )
        )
        if tuple(sorted(percentiles)) != percentiles:
            raise ValueError("timing percentiles must be non-decreasing")
        miss_count = _nonnegative_integer(
            self.deadline_miss_count,
            name="deadline_miss_count",
        )
        if miss_count > sample_count:
            raise ValueError("deadline_miss_count cannot exceed sample_count")
        miss_rate = _finite_nonnegative(
            self.deadline_miss_rate,
            name="deadline_miss_rate",
        )
        if miss_rate > 1.0:
            raise ValueError("deadline_miss_rate cannot exceed one")
        if not math.isclose(
            miss_rate,
            miss_count / sample_count,
            rel_tol=0.0,
            abs_tol=1.0e-15,
        ):
            raise ValueError("deadline_miss_rate must equal miss_count / sample_count")

        object.__setattr__(self, "sample_count", sample_count)
        object.__setattr__(self, "deadline_s", deadline_s)
        for name, value in zip(
            ("p50_s", "p95_s", "p99_s", "max_s"),
            percentiles,
            strict=True,
        ):
            object.__setattr__(self, name, value)
        object.__setattr__(self, "deadline_miss_count", miss_count)
        object.__setattr__(self, "deadline_miss_rate", miss_rate)


@dataclass(frozen=True, slots=True)
class EpisodeEvaluation:
    """One normally completed episode in its fixed evaluation order."""

    track_index: int
    track_id: int
    reset_seed: int
    success: bool
    lap_time_s: float | None
    steps: int
    total_reward: float
    terminated: bool
    truncated: bool
    termination_reason: int
    controller_import_time_s: float
    controller_init_time_s: float
    compute_times_s: tuple[float, ...]
    compute_timing: TimingSummary

    def __post_init__(self) -> None:
        track_index = _nonnegative_integer(self.track_index, name="track_index")
        track_id = _nonnegative_integer(self.track_id, name="track_id")
        reset_seed = _nonnegative_integer(self.reset_seed, name="reset_seed")
        if track_id > _UINT32_MAX:
            raise ValueError("track_id must fit in uint32")
        if reset_seed > _UINT32_MAX:
            raise ValueError("reset_seed must fit in uint32")
        if type(self.success) is not bool:
            raise TypeError("success must be a boolean")
        if type(self.terminated) is not bool or type(self.truncated) is not bool:
            raise TypeError("terminated and truncated must be booleans")
        if self.terminated == self.truncated:
            raise ValueError("exactly one of terminated or truncated must be true")
        if self.success and not self.terminated:
            raise ValueError("a successful episode must be terminated")

        if self.success:
            lap_time_s = _positive_finite(self.lap_time_s, name="lap_time_s")
        elif self.lap_time_s is not None:
            raise ValueError("lap_time_s must be None for an unsuccessful episode")
        else:
            lap_time_s = None
        steps = _nonnegative_integer(self.steps, name="steps")
        if steps == 0:
            raise ValueError("steps must be positive")
        reward = float(self.total_reward)
        if not math.isfinite(reward):
            raise ValueError("total_reward must be finite")
        termination_reason = _nonnegative_integer(
            self.termination_reason,
            name="termination_reason",
        )
        import_time = _finite_nonnegative(
            self.controller_import_time_s,
            name="controller_import_time_s",
        )
        init_time = _finite_nonnegative(
            self.controller_init_time_s,
            name="controller_init_time_s",
        )
        compute_times = tuple(
            _finite_nonnegative(value, name=f"compute_times_s[{index}]")
            for index, value in enumerate(self.compute_times_s)
        )
        if len(compute_times) != steps:
            raise ValueError("compute_times_s must contain exactly one value per step")
        if not isinstance(self.compute_timing, TimingSummary):
            raise TypeError("compute_timing must be a TimingSummary")
        if self.compute_timing.sample_count != len(compute_times):
            raise ValueError("compute_timing.sample_count must match compute_times_s")

        object.__setattr__(self, "track_index", track_index)
        object.__setattr__(self, "track_id", track_id)
        object.__setattr__(self, "reset_seed", reset_seed)
        object.__setattr__(self, "lap_time_s", lap_time_s)
        object.__setattr__(self, "steps", steps)
        object.__setattr__(self, "total_reward", reward)
        object.__setattr__(self, "termination_reason", termination_reason)
        object.__setattr__(self, "controller_import_time_s", import_time)
        object.__setattr__(self, "controller_init_time_s", init_time)
        object.__setattr__(self, "compute_times_s", compute_times)


@dataclass(frozen=True, slots=True)
class ControllerEvaluation:
    """Aggregate result for one Controller and one ordered Track batch."""

    controller_directory: str
    level_id: int
    backend: EvaluationBackend
    episodes: tuple[EpisodeEvaluation, ...]
    track_count: int
    success_count: int
    success_rate: float
    mean_successful_lap_time_s: float | None
    compute_timing: TimingSummary

    def __post_init__(self) -> None:
        if not isinstance(self.controller_directory, str) or not self.controller_directory:
            raise ValueError("controller_directory must be a non-empty string")
        level_id = _nonnegative_integer(self.level_id, name="level_id")
        if self.backend not in ("cpu_reference", "mjx_warp"):
            raise ValueError("backend must be 'cpu_reference' or 'mjx_warp'")
        if not isinstance(self.episodes, tuple) or not all(
            isinstance(episode, EpisodeEvaluation) for episode in self.episodes
        ):
            raise TypeError("episodes must be a tuple of EpisodeEvaluation values")
        if not self.episodes:
            raise ValueError("episodes cannot be empty")
        if tuple(episode.track_index for episode in self.episodes) != tuple(
            range(len(self.episodes))
        ):
            raise ValueError("episode track_index values must preserve contiguous batch order")
        track_count = _nonnegative_integer(self.track_count, name="track_count")
        if track_count != len(self.episodes):
            raise ValueError("track_count must equal len(episodes)")
        success_count = _nonnegative_integer(self.success_count, name="success_count")
        expected_success_count = sum(episode.success for episode in self.episodes)
        if success_count != expected_success_count:
            raise ValueError("success_count must match the episode results")
        success_rate = _finite_nonnegative(self.success_rate, name="success_rate")
        if success_rate > 1.0 or not math.isclose(
            success_rate,
            success_count / track_count,
            rel_tol=0.0,
            abs_tol=1.0e-15,
        ):
            raise ValueError("success_rate must equal success_count / track_count")
        successful_laps = tuple(episode.lap_time_s for episode in self.episodes if episode.success)
        if successful_laps:
            mean_lap_time = _positive_finite(
                self.mean_successful_lap_time_s,
                name="mean_successful_lap_time_s",
            )
            expected_mean = float(np.mean(successful_laps, dtype=np.float64))
            if not math.isclose(mean_lap_time, expected_mean, rel_tol=0.0, abs_tol=1.0e-12):
                raise ValueError("mean_successful_lap_time_s must match successful episodes")
        elif self.mean_successful_lap_time_s is not None:
            raise ValueError("mean_successful_lap_time_s must be None when no episode succeeds")
        else:
            mean_lap_time = None
        if not isinstance(self.compute_timing, TimingSummary):
            raise TypeError("compute_timing must be a TimingSummary")
        expected_samples = sum(len(episode.compute_times_s) for episode in self.episodes)
        if self.compute_timing.sample_count != expected_samples:
            raise ValueError("compute_timing must include every episode compute sample")

        object.__setattr__(self, "level_id", level_id)
        object.__setattr__(self, "track_count", track_count)
        object.__setattr__(self, "success_count", success_count)
        object.__setattr__(self, "success_rate", success_rate)
        object.__setattr__(self, "mean_successful_lap_time_s", mean_lap_time)


def summarize_compute_times(
    compute_times_s: Sequence[Real] | NDArray[np.number],
    *,
    deadline_s: Real = DEFAULT_COMPUTE_DEADLINE_S,
) -> TimingSummary:
    """Summarize one non-empty, finite float64 compute-time sample."""

    deadline = _positive_finite(deadline_s, name="deadline_s")
    if isinstance(compute_times_s, (str, bytes)) or not isinstance(
        compute_times_s,
        (Sequence, np.ndarray),
    ):
        raise TypeError("compute_times_s must be a sequence of real numbers")
    raw_values = tuple(compute_times_s)
    if not raw_values:
        raise ValueError("compute_times_s cannot be empty")
    values = np.asarray(
        tuple(
            _finite_nonnegative(value, name=f"compute_times_s[{index}]")
            for index, value in enumerate(raw_values)
        ),
        dtype=np.float64,
    )
    p50_s, p95_s, p99_s = np.percentile(
        values,
        (50.0, 95.0, 99.0),
        method="linear",
    )
    miss_count = int(np.count_nonzero(values > deadline))
    sample_count = int(values.size)
    return TimingSummary(
        sample_count=sample_count,
        deadline_s=deadline,
        p50_s=float(p50_s),
        p95_s=float(p95_s),
        p99_s=float(p99_s),
        max_s=float(np.max(values)),
        deadline_miss_count=miss_count,
        deadline_miss_rate=miss_count / sample_count,
    )


def _normalized_reset_seeds(
    reset_seeds: Sequence[Integral] | NDArray[np.integer] | None,
    *,
    track_count: int,
) -> tuple[int, ...]:
    if reset_seeds is None:
        return tuple(range(track_count))
    if isinstance(reset_seeds, (str, bytes)) or not isinstance(
        reset_seeds,
        (Sequence, np.ndarray),
    ):
        raise TypeError("reset_seeds must be a sequence of uint32 integers or None")
    if isinstance(reset_seeds, np.ndarray) and reset_seeds.ndim != 1:
        raise ValueError("reset_seeds must be one-dimensional")
    values = tuple(reset_seeds)
    if len(values) != track_count:
        raise ValueError(f"reset_seeds must contain {track_count} values, got {len(values)}")
    normalized = tuple(
        _nonnegative_integer(value, name=f"reset_seeds[{index}]")
        for index, value in enumerate(values)
    )
    if any(value > _UINT32_MAX for value in normalized):
        raise ValueError("reset_seeds values must fit in uint32")
    return normalized


def _episode_from_run(
    result: EpisodeRunResult,
    *,
    track_index: int,
    expected_track_id: int,
    reset_seed: int,
    deadline_s: float,
) -> EpisodeEvaluation:
    if not isinstance(result, EpisodeRunResult):
        raise TypeError("run_episode must return an EpisodeRunResult")
    if not isinstance(result.final_info, Mapping):
        raise TypeError("EpisodeRunResult.final_info must be a mapping")
    for key in ("track_id", "lap_completed", "lap_time_s", "termination_reason"):
        if key not in result.final_info:
            raise ValueError(f"EpisodeRunResult.final_info is missing {key!r}")
    track_id = result.final_info["track_id"]
    if type(track_id) is not int:
        raise TypeError("final_info['track_id'] must be an integer")
    if track_id != expected_track_id:
        raise ValueError(
            f"episode Track ID {track_id} does not match batch Track ID {expected_track_id}"
        )
    success = result.final_info["lap_completed"]
    if type(success) is not bool:
        raise TypeError("final_info['lap_completed'] must be a boolean")
    raw_lap_time = result.final_info["lap_time_s"]
    if type(raw_lap_time) is not float:
        raise TypeError("final_info['lap_time_s'] must be a float")
    lap_time_s = _positive_finite(raw_lap_time, name="lap_time_s") if success else None
    termination_reason = result.final_info["termination_reason"]
    if type(termination_reason) is not int:
        raise TypeError("final_info['termination_reason'] must be an integer")
    compute_timing = summarize_compute_times(result.compute_times_s, deadline_s=deadline_s)
    return EpisodeEvaluation(
        track_index=track_index,
        track_id=track_id,
        reset_seed=reset_seed,
        success=success,
        lap_time_s=lap_time_s,
        steps=result.steps,
        total_reward=result.total_reward,
        terminated=result.terminated,
        truncated=result.truncated,
        termination_reason=termination_reason,
        controller_import_time_s=result.controller_import_time_s,
        controller_init_time_s=result.controller_init_time_s,
        compute_times_s=result.compute_times_s,
        compute_timing=compute_timing,
    )


def evaluate_track_batch(
    project_config: ProjectConfig,
    level_id: int,
    batch: TrackBatch,
    generator_version: str,
    controller_directory: str | Path,
    backend: EvaluationBackend,
    reset_seeds: Sequence[Integral] | NDArray[np.integer] | None = None,
    *,
    env_factory: EnvironmentFactory = CarRacingEnv,
    run_episode: EpisodeRunner = run_controller_episode,
) -> ControllerEvaluation:
    """Evaluate a fresh Controller episode on every Track row, in fixed order.

    Controller and environment exceptions intentionally propagate to the caller. The environment
    created for a row is still closed before propagation, leaving formal failure policy to the
    higher-level evaluation script.
    """

    if not isinstance(project_config, ProjectConfig):
        raise TypeError("project_config must be a ProjectConfig")
    if isinstance(level_id, bool) or not isinstance(level_id, int):
        raise TypeError("level_id must be an integer")
    if level_id not in {level.level_id for level in project_config.levels}:
        raise ValueError(f"level_id {level_id} is not present in project_config")
    if not isinstance(batch, TrackBatch):
        raise TypeError("batch must be a TrackBatch")
    validate_track_batch(batch)
    if not isinstance(generator_version, str) or not generator_version:
        raise ValueError("generator_version must be a non-empty string")
    if not isinstance(controller_directory, (str, Path)):
        raise TypeError("controller_directory must be a string or Path")
    directory = str(controller_directory)
    if not directory:
        raise ValueError("controller_directory cannot be empty")
    if backend not in ("cpu_reference", "mjx_warp"):
        raise ValueError("backend must be 'cpu_reference' or 'mjx_warp'")
    if not callable(env_factory):
        raise TypeError("env_factory must be callable")
    if not callable(run_episode):
        raise TypeError("run_episode must be callable")

    track_count = int(batch.seed.shape[0])
    seeds = _normalized_reset_seeds(reset_seeds, track_count=track_count)
    deadline_s = project_config.benchmark.controller.compute_deadline_s
    episodes: list[EpisodeEvaluation] = []
    all_compute_times: list[float] = []

    for track_index, reset_seed in enumerate(seeds):
        track = track_from_batch_row(
            batch,
            track_index,
            generator_version=generator_version,
        )
        env = env_factory(
            project_config=project_config,
            level_id=level_id,
            track=track,
            backend=backend,
        )
        close = getattr(env, "close", None)
        if not callable(close):
            raise TypeError("env_factory must return an environment with close()")
        try:
            result = run_episode(env, directory, reset_seed)
        finally:
            close()
        episode = _episode_from_run(
            result,
            track_index=track_index,
            expected_track_id=track.seed,
            reset_seed=reset_seed,
            deadline_s=deadline_s,
        )
        episodes.append(episode)
        all_compute_times.extend(episode.compute_times_s)

    successful_laps = tuple(episode.lap_time_s for episode in episodes if episode.success)
    success_count = len(successful_laps)
    return ControllerEvaluation(
        controller_directory=directory,
        level_id=level_id,
        backend=backend,
        episodes=tuple(episodes),
        track_count=track_count,
        success_count=success_count,
        success_rate=success_count / track_count,
        mean_successful_lap_time_s=(
            float(np.mean(successful_laps, dtype=np.float64)) if successful_laps else None
        ),
        compute_timing=summarize_compute_times(all_compute_times, deadline_s=deadline_s),
    )


__all__ = [
    "DEFAULT_COMPUTE_DEADLINE_S",
    "ControllerEvaluation",
    "EpisodeEvaluation",
    "TimingSummary",
    "evaluate_track_batch",
    "summarize_compute_times",
]
