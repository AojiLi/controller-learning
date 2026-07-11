"""Asset-agnostic execution core for the frozen M8 Controller workload.

The caller owns the one batch-one environment and all Track loading.  This module only executes
the fixed Controller/row schedule through the ordinary recording Runner, validates public episode
identity, and derives metrics from that same canonical trajectory.  It deliberately has no asset,
environment-construction, replay, report, or publication responsibilities.
"""

from __future__ import annotations

import math
import re
import time
import traceback
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from numbers import Integral
from pathlib import Path
from types import MappingProxyType
from typing import Final, Literal, TypeAlias

import numpy as np

from controller_learning.envs.episode import initialize_episode_identities
from controller_learning.evaluation.final_benchmark import M8_CONTROLLER_ORDER
from controller_learning.evaluation.final_metrics import (
    EpisodeMetricSamples,
    MetricActionLimits,
    compute_episode_metric_samples,
)
from controller_learning.evaluation.trajectory import (
    RecordedControllerEpisode,
    record_controller_episode,
)

FINAL_WORKLOAD_BENCHMARK_VERSION: Final = "0.1"
CONTROLLER_INIT_SOFT_LIMIT_S: Final = 30.0
MAX_SANITIZED_TRACEBACK_CHARS: Final = 4096
_UINT32_MAX: Final = int(np.iinfo(np.uint32).max)
_MAX_TRACEBACK_FRAMES: Final = 24

FinalWorkloadFailurePhase: TypeAlias = Literal[
    "record_episode",
    "validate_episode",
    "compute_metrics",
    "validate_metrics",
    "episode_sink",
]
EpisodeSink: TypeAlias = Callable[
    [str, int, RecordedControllerEpisode, EpisodeMetricSamples],
    None,
]
ControllerDirectory: TypeAlias = str | Path

_POSIX_ABSOLUTE_PATH = re.compile(r"(?<![A-Za-z0-9_.-])/(?:[^\s\"'<>]|\\ )+")
_WINDOWS_ABSOLUTE_PATH = re.compile(r"(?<![A-Za-z0-9_.-])[A-Za-z]:[\\/][^\s\"'<>]+")


def _finite_nonnegative(value: object, *, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float, np.integer, np.floating)):
        raise TypeError(f"{name} must be a real number")
    result = float(value)
    if not math.isfinite(result) or result < 0.0:
        raise ValueError(f"{name} must be finite and non-negative")
    return result


def _positive_integer(value: object, *, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, Integral):
        raise TypeError(f"{name} must be a positive integer")
    result = int(value)
    if result <= 0:
        raise ValueError(f"{name} must be positive")
    return result


def _clean_exception_text(value: str) -> str:
    """Remove path-shaped and control-character evidence from exception text."""

    normalized = "".join(
        character if character in "\n\t" or character.isprintable() else "?" for character in value
    )
    normalized = _WINDOWS_ABSOLUTE_PATH.sub("<path>", normalized)
    normalized = _POSIX_ABSOLUTE_PATH.sub("<path>", normalized)
    return normalized


def _sanitized_traceback(error: Exception) -> str:
    """Render bounded traceback evidence without absolute filesystem paths or locals."""

    frames = traceback.extract_tb(error.__traceback__)[-_MAX_TRACEBACK_FRAMES:]
    lines = ["Traceback (most recent call last):"]
    for frame in frames:
        filename = frame.filename.replace("\\", "/").rsplit("/", maxsplit=1)[-1]
        function = _clean_exception_text(frame.name)
        lines.append(f'  File "{filename}", line {frame.lineno}, in {function}')
    error_type = type(error).__name__
    detail = _clean_exception_text(str(error)).strip()
    lines.append(f"{error_type}: {detail}" if detail else error_type)
    rendered = "\n".join(lines)
    if len(rendered) <= MAX_SANITIZED_TRACEBACK_CHARS:
        return rendered
    suffix = "\n<truncated>"
    return rendered[: MAX_SANITIZED_TRACEBACK_CHARS - len(suffix)] + suffix


class FinalWorkloadExecutionError(RuntimeError):
    """One fixed Controller row failed without exposing unsafe exception details."""

    def __init__(
        self,
        *,
        controller_name: str,
        row_index: int,
        phase: FinalWorkloadFailurePhase,
        cause_type: str,
        sanitized_traceback: str,
    ) -> None:
        self.controller_name = controller_name
        self.row_index = row_index
        self.phase = phase
        self.cause_type = cause_type
        self.sanitized_traceback = sanitized_traceback
        super().__init__(
            "Final workload execution failed for "
            f"controller={controller_name!r}, row={row_index}, phase={phase!r} "
            f"({cause_type})"
        )


def _execution_error(
    error: Exception,
    *,
    controller_name: str,
    row_index: int,
    phase: FinalWorkloadFailurePhase,
) -> FinalWorkloadExecutionError:
    return FinalWorkloadExecutionError(
        controller_name=controller_name,
        row_index=row_index,
        phase=phase,
        cause_type=type(error).__name__,
        sanitized_traceback=_sanitized_traceback(error),
    )


@dataclass(frozen=True, slots=True)
class ControllerWorkloadExecution:
    """Exactly ordered canonical episodes and metric samples for one Controller."""

    controller_name: str
    track_ids: tuple[int, ...]
    recorded_episodes: tuple[RecordedControllerEpisode, ...]
    metric_samples: tuple[EpisodeMetricSamples, ...]
    wall_time_s: float
    initialization_over_30s_rows: tuple[int, ...]

    def __post_init__(self) -> None:
        if self.controller_name not in M8_CONTROLLER_ORDER:
            raise ValueError(f"controller_name must be one of {M8_CONTROLLER_ORDER!r}")
        if not isinstance(self.track_ids, tuple) or not self.track_ids:
            raise ValueError("track_ids must be a non-empty tuple")
        if not isinstance(self.recorded_episodes, tuple) or not all(
            isinstance(episode, RecordedControllerEpisode) for episode in self.recorded_episodes
        ):
            raise TypeError("recorded_episodes must be a tuple of RecordedControllerEpisode values")
        if not isinstance(self.metric_samples, tuple) or not all(
            isinstance(samples, EpisodeMetricSamples) for samples in self.metric_samples
        ):
            raise TypeError("metric_samples must be a tuple of EpisodeMetricSamples values")
        if len(self.recorded_episodes) != len(self.track_ids) or len(self.metric_samples) != len(
            self.track_ids
        ):
            raise ValueError("each Track row must have one recorded episode and metric sample")
        if (
            tuple(
                int(episode.trajectory.reset_info["track_id"]) for episode in self.recorded_episodes
            )
            != self.track_ids
        ):
            raise ValueError("recorded episode Track IDs differ from the fixed row order")
        if tuple(samples.track_id for samples in self.metric_samples) != self.track_ids:
            raise ValueError("metric sample Track IDs differ from the fixed row order")
        if tuple(samples.reset_seed for samples in self.metric_samples) != tuple(
            range(len(self.track_ids))
        ):
            raise ValueError("metric sample reset seeds differ from row indices")
        wall_time_s = _finite_nonnegative(self.wall_time_s, name="wall_time_s")
        expected_slow_rows = tuple(
            row
            for row, episode in enumerate(self.recorded_episodes)
            if episode.result.controller_init_time_s > CONTROLLER_INIT_SOFT_LIMIT_S
        )
        if self.initialization_over_30s_rows != expected_slow_rows:
            raise ValueError(
                "initialization_over_30s_rows must identify every initialization above 30 seconds"
            )
        object.__setattr__(self, "wall_time_s", wall_time_s)

    @property
    def episode_count(self) -> int:
        """Number of fixed-order episodes executed for this Controller."""

        return len(self.recorded_episodes)


@dataclass(frozen=True, slots=True)
class FinalWorkloadExecution:
    """Complete fixed-order workload result from one caller-owned environment."""

    track_ids: tuple[int, ...]
    controller_results: Mapping[str, ControllerWorkloadExecution]
    wall_time_s: float
    environment_instance_count: int
    fresh_runner_instance_count: int

    def __post_init__(self) -> None:
        if not isinstance(self.track_ids, tuple) or not self.track_ids:
            raise ValueError("track_ids must be a non-empty tuple")
        if not isinstance(self.controller_results, Mapping):
            raise TypeError("controller_results must be a mapping")
        if tuple(self.controller_results) != M8_CONTROLLER_ORDER:
            raise ValueError("controller_results must preserve the fixed PID/MPC/PPO order")
        copied_results = dict(self.controller_results)
        for name in M8_CONTROLLER_ORDER:
            result = copied_results[name]
            if not isinstance(result, ControllerWorkloadExecution):
                raise TypeError("controller_results values must be ControllerWorkloadExecution")
            if result.controller_name != name or result.track_ids != self.track_ids:
                raise ValueError("controller result identity differs from the final workload")
        wall_time_s = _finite_nonnegative(self.wall_time_s, name="wall_time_s")
        if self.environment_instance_count != 1:
            raise ValueError("the final workload must reuse exactly one caller-owned environment")
        expected_runner_instances = len(M8_CONTROLLER_ORDER) * len(self.track_ids)
        if self.fresh_runner_instance_count != expected_runner_instances:
            raise ValueError("fresh_runner_instance_count must equal Controllers times Track rows")
        object.__setattr__(self, "controller_results", MappingProxyType(copied_results))
        object.__setattr__(self, "wall_time_s", wall_time_s)

    @property
    def controller_order(self) -> tuple[str, ...]:
        """The immutable Controller-major execution order."""

        return M8_CONTROLLER_ORDER

    @property
    def episode_count(self) -> int:
        """Total number of fresh ordinary Runner invocations."""

        return self.fresh_runner_instance_count


def _validated_controller_directories(
    value: Mapping[str, ControllerDirectory],
) -> Mapping[str, ControllerDirectory]:
    if not isinstance(value, Mapping):
        raise TypeError("controller_directories must be a mapping")
    if any(type(key) is not str for key in value) or tuple(value) != M8_CONTROLLER_ORDER:
        raise ValueError("controller_directories must contain exactly pid, mpc, ppo in that order")
    copied: dict[str, ControllerDirectory] = {}
    for name in M8_CONTROLLER_ORDER:
        directory = value[name]
        if isinstance(directory, str):
            if not directory:
                raise ValueError(f"controller_directories[{name!r}] cannot be empty")
        elif not isinstance(directory, Path):
            raise TypeError(f"controller_directories[{name!r}] must be a string or pathlib.Path")
        copied[name] = directory
    return MappingProxyType(copied)


def _validated_track_ids(
    value: Sequence[int] | np.ndarray, *, expected_count: int
) -> tuple[int, ...]:
    if isinstance(value, np.ndarray):
        if value.ndim != 1:
            raise ValueError("track_ids must be one-dimensional")
        raw_values = tuple(value)
    elif isinstance(value, (str, bytes)) or not isinstance(value, Sequence):
        raise TypeError("track_ids must be an ordered sequence")
    else:
        raw_values = tuple(value)
    if len(raw_values) != expected_count:
        raise ValueError(f"track_ids must contain exactly {expected_count} ordered values")
    track_ids: list[int] = []
    for row_index, raw in enumerate(raw_values):
        if isinstance(raw, bool) or not isinstance(raw, Integral):
            raise TypeError(f"track_ids[{row_index}] must be an integer")
        track_id = int(raw)
        if not 0 <= track_id <= _UINT32_MAX:
            raise ValueError(f"track_ids[{row_index}] must fit in uint32")
        track_ids.append(track_id)
    if len(set(track_ids)) != len(track_ids):
        raise ValueError("track_ids must be unique while preserving manifest row order")
    return tuple(track_ids)


def _validate_recorded_episode(
    recorded: RecordedControllerEpisode,
    *,
    row_index: int,
    expected_track_id: int,
) -> None:
    if not isinstance(recorded, RecordedControllerEpisode):
        raise TypeError("record_controller_episode must return RecordedControllerEpisode")
    trajectory = recorded.trajectory
    result = recorded.result
    identity = initialize_episode_identities(row_index, 1)
    expected_episode_seed = int(identity.episode_seed[0])
    expected_controller_seed = int(identity.controller_seed[0])

    reset_info = trajectory.reset_info
    final_info = trajectory.final_info
    expected_identity = {
        "episode_seed": expected_episode_seed,
        "controller_seed": expected_controller_seed,
        "track_id": expected_track_id,
        "benchmark_version": FINAL_WORKLOAD_BENCHMARK_VERSION,
    }
    for field, expected in expected_identity.items():
        if reset_info[field] != expected or final_info[field] != expected:
            raise ValueError(f"public episode identity field {field!r} differs from protocol")

    if (
        result.steps != trajectory.step_count
        or len(result.compute_times_s) != trajectory.step_count
    ):
        raise ValueError("Runner, trajectory, and compute-time step counts must match")
    if result.terminated == result.truncated:
        raise ValueError("exactly one terminal flag must end every episode")
    terminal_reason = int(final_info["termination_reason"])
    if terminal_reason == 0:
        raise ValueError("a completed episode must have a non-neutral termination reason")
    if (terminal_reason == 4) != result.truncated:
        raise ValueError("TIMEOUT must be the only truncated terminal reason")
    if bool(final_info["lap_completed"]) != (terminal_reason == 1):
        raise ValueError("lap completion must match the SUCCESS terminal reason")


def _validate_metric_samples(
    samples: EpisodeMetricSamples,
    *,
    recorded: RecordedControllerEpisode,
    row_index: int,
    expected_track_id: int,
) -> None:
    if not isinstance(samples, EpisodeMetricSamples):
        raise TypeError("compute_episode_metric_samples must return EpisodeMetricSamples")
    if samples.track_id != expected_track_id or samples.reset_seed != row_index:
        raise ValueError("metric sample identity differs from the canonical episode row")
    if samples.transition_count != recorded.trajectory.step_count:
        raise ValueError("metric sample count differs from the canonical trajectory")
    if not np.array_equal(samples.requested_action, recorded.trajectory.action):
        raise ValueError("metric requested actions differ from the canonical trajectory")
    if not np.array_equal(
        samples.compute_time_s,
        np.asarray(recorded.result.compute_times_s, dtype=np.float64),
    ):
        raise ValueError("metric compute times differ from the canonical Runner result")


def _elapsed_seconds(started_ns: int) -> float:
    return max(0, time.perf_counter_ns() - started_ns) / 1_000_000_000


def execute_controller_workload(
    *,
    environment: object,
    controller_directories: Mapping[str, ControllerDirectory],
    track_ids: Sequence[int] | np.ndarray,
    action_limits: MetricActionLimits,
    max_episode_steps: int,
    episode_sink: EpisodeSink | None = None,
    expected_track_count: int = 20,
) -> FinalWorkloadExecution:
    """Execute PID, MPC, and PPO row-major through one caller-owned environment.

    The function never creates, closes, unwraps, or replays an environment.  Every episode is run
    once by :func:`record_controller_episode`; metrics are then derived from and checked against
    that exact recording.  A sink can durably persist the validated pair before the next row starts.
    Ordinary row failures stop the workload and surface bounded, path-sanitized evidence.
    """

    directories = _validated_controller_directories(controller_directories)
    expected_count = _positive_integer(expected_track_count, name="expected_track_count")
    if expected_count > _UINT32_MAX + 1:
        raise ValueError("expected_track_count exceeds the uint32 reset-seed row range")
    ordered_track_ids = _validated_track_ids(track_ids, expected_count=expected_count)
    max_steps = _positive_integer(max_episode_steps, name="max_episode_steps")
    if not isinstance(action_limits, MetricActionLimits):
        raise TypeError("action_limits must be MetricActionLimits")
    if episode_sink is not None and not callable(episode_sink):
        raise TypeError("episode_sink must be callable or None")

    workload_started_ns = time.perf_counter_ns()
    controller_results: dict[str, ControllerWorkloadExecution] = {}
    for controller_name in M8_CONTROLLER_ORDER:
        controller_started_ns = time.perf_counter_ns()
        recorded_episodes: list[RecordedControllerEpisode] = []
        metric_samples: list[EpisodeMetricSamples] = []
        for row_index, expected_track_id in enumerate(ordered_track_ids):
            try:
                recorded = record_controller_episode(
                    environment,  # type: ignore[arg-type]
                    directories[controller_name],
                    row_index,
                    render=False,
                    max_steps=max_steps,
                    reset_options={"track_index": row_index},
                )
            except Exception as error:
                raise _execution_error(
                    error,
                    controller_name=controller_name,
                    row_index=row_index,
                    phase="record_episode",
                ) from None

            try:
                _validate_recorded_episode(
                    recorded,
                    row_index=row_index,
                    expected_track_id=expected_track_id,
                )
            except Exception as error:
                raise _execution_error(
                    error,
                    controller_name=controller_name,
                    row_index=row_index,
                    phase="validate_episode",
                ) from None

            try:
                samples = compute_episode_metric_samples(
                    recorded,
                    reset_seed=row_index,
                    action_limits=action_limits,
                )
            except Exception as error:
                raise _execution_error(
                    error,
                    controller_name=controller_name,
                    row_index=row_index,
                    phase="compute_metrics",
                ) from None

            try:
                _validate_metric_samples(
                    samples,
                    recorded=recorded,
                    row_index=row_index,
                    expected_track_id=expected_track_id,
                )
            except Exception as error:
                raise _execution_error(
                    error,
                    controller_name=controller_name,
                    row_index=row_index,
                    phase="validate_metrics",
                ) from None

            if episode_sink is not None:
                try:
                    episode_sink(controller_name, row_index, recorded, samples)
                except Exception as error:
                    raise _execution_error(
                        error,
                        controller_name=controller_name,
                        row_index=row_index,
                        phase="episode_sink",
                    ) from None

            recorded_episodes.append(recorded)
            metric_samples.append(samples)

        recorded_tuple = tuple(recorded_episodes)
        controller_results[controller_name] = ControllerWorkloadExecution(
            controller_name=controller_name,
            track_ids=ordered_track_ids,
            recorded_episodes=recorded_tuple,
            metric_samples=tuple(metric_samples),
            wall_time_s=_elapsed_seconds(controller_started_ns),
            initialization_over_30s_rows=tuple(
                row
                for row, recorded in enumerate(recorded_tuple)
                if recorded.result.controller_init_time_s > CONTROLLER_INIT_SOFT_LIMIT_S
            ),
        )

    return FinalWorkloadExecution(
        track_ids=ordered_track_ids,
        controller_results=controller_results,
        wall_time_s=_elapsed_seconds(workload_started_ns),
        environment_instance_count=1,
        fresh_runner_instance_count=len(M8_CONTROLLER_ORDER) * expected_count,
    )


__all__ = [
    "CONTROLLER_INIT_SOFT_LIMIT_S",
    "FINAL_WORKLOAD_BENCHMARK_VERSION",
    "MAX_SANITIZED_TRACEBACK_CHARS",
    "ControllerWorkloadExecution",
    "EpisodeSink",
    "FinalWorkloadExecution",
    "FinalWorkloadExecutionError",
    "FinalWorkloadFailurePhase",
    "execute_controller_workload",
]
