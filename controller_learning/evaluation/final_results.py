"""Strict deterministic result tables for the frozen M8 Controller comparison.

The types in this module bind the human-readable CSV and JSON summaries to the same public
trajectory and transition samples stored in ``metrics.npz``.  They deliberately contain no
performance threshold or combined score: Test ranks Controllers only by completion rate and then
mean successful lap time.
"""

from __future__ import annotations

import csv
import io
import json
import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from itertools import pairwise
from types import MappingProxyType
from typing import Final

import numpy as np

from controller_learning.envs.episode import initialize_episode_identities
from controller_learning.evaluation.controller import TimingSummary, summarize_compute_times
from controller_learning.evaluation.final_benchmark import (
    M8_CONTROLLER_ORDER,
    M8_RANKING_RULE,
    M8_TEST_TRACK_COUNT,
)
from controller_learning.evaluation.final_metrics import (
    FINAL_METRICS_BENCHMARK_VERSION,
    FINAL_METRICS_CONTROL_DT_S,
    AggregateMetricSummary,
    EpisodeMetricSamples,
    EpisodeMetricSummary,
    FinalMetricsData,
    MetricActionLimits,
    compute_episode_metric_samples,
    summarize_episode_metrics,
    summarize_final_metrics,
)
from controller_learning.evaluation.trajectory import RecordedControllerEpisode

FINAL_CONTROLLER_RESULTS_SCHEMA_VERSION: Final = "controller-learning.m8-controller-results.v1"
FINAL_CONTROLLER_SUMMARY_SCHEMA_VERSION: Final = "controller-learning.m8-controller-summary.v1"
FINAL_COMPARISON_SCHEMA_VERSION: Final = "controller-learning.m8-comparison.v1"
FINAL_RESULTS_COMPUTE_DEADLINE_S: Final = FINAL_METRICS_CONTROL_DT_S
FINAL_RESULTS_REALTIME_P99_LIMIT_S: Final = 0.05
FINAL_RESULTS_REALTIME_MISS_RATE_LIMIT: Final = 0.01
FINAL_RESULTS_ACTION_LIMITS: Final = MetricActionLimits(
    max_steering_angle_rad=0.6,
    max_acceleration_mps2=4.0,
    max_deceleration_mps2=8.0,
)

_TERMINATION_NAMES: Final = MappingProxyType(
    {
        1: "success",
        2: "off_track",
        3: "invalid_action",
        4: "timeout",
    }
)
_TERMINATION_COUNT_KEYS: Final = tuple(_TERMINATION_NAMES.values())

CONTROLLER_RESULTS_CSV_COLUMNS: Final = (
    "schema_version",
    "benchmark_version",
    "controller_name",
    "row_index",
    "track_id",
    "reset_seed",
    "episode_seed",
    "controller_seed",
    "success",
    "lap_time_s",
    "environment_steps",
    "total_reward",
    "terminated",
    "truncated",
    "termination_reason",
    "termination_reason_name",
    "controller_import_time_s",
    "controller_init_time_s",
    "compute_sample_count",
    "compute_deadline_s",
    "compute_p50_s",
    "compute_p95_s",
    "compute_p99_s",
    "compute_max_s",
    "compute_deadline_miss_count",
    "compute_deadline_miss_rate",
    "metric_transition_count",
    "metric_action_delta_count",
    "mean_speed_mps",
    "lateral_error_rms_m",
    "lateral_error_abs_p95_m",
    "lateral_error_abs_max_m",
    "steering_saturation_rate",
    "longitudinal_saturation_rate",
    "steering_rate_rms_rad_s",
    "acceleration_rate_rms_mps3",
)

FINAL_COMPARISON_CSV_COLUMNS: Final = (
    "schema_version",
    "benchmark_version",
    "controller_name",
    "rank",
    "ranking_rule",
    "track_count",
    "success_count",
    "success_rate",
    "mean_successful_lap_time_s",
    "success_termination_count",
    "off_track_termination_count",
    "invalid_action_termination_count",
    "timeout_termination_count",
    "environment_steps",
    "metric_transition_count",
    "metric_action_delta_count",
    "mean_speed_mps",
    "lateral_error_rms_m",
    "lateral_error_abs_p95_m",
    "lateral_error_abs_max_m",
    "steering_saturation_rate",
    "longitudinal_saturation_rate",
    "steering_rate_rms_rad_s",
    "acceleration_rate_rms_mps3",
    "compute_sample_count",
    "compute_deadline_s",
    "compute_p50_s",
    "compute_p95_s",
    "compute_p99_s",
    "compute_max_s",
    "compute_deadline_miss_count",
    "compute_deadline_miss_rate",
    "mean_controller_import_time_s",
    "mean_controller_init_time_s",
)


class FinalResultsArtifactError(ValueError):
    """A final result object or canonical artifact violates the frozen M8 contract."""


def _controller_name(value: object) -> str:
    if type(value) is not str or value not in M8_CONTROLLER_ORDER:
        raise FinalResultsArtifactError(f"controller_name must be one of {M8_CONTROLLER_ORDER!r}")
    return value


def _row_index(value: object) -> int:
    if type(value) is not int or not 0 <= value < M8_TEST_TRACK_COUNT:
        raise FinalResultsArtifactError(
            f"row_index must be an integer in [0, {M8_TEST_TRACK_COUNT})"
        )
    return value


def _finite_nonnegative(value: object, *, field: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float, np.integer, np.floating)):
        raise FinalResultsArtifactError(f"{field} must be a real number")
    result = float(value)
    if not math.isfinite(result) or result < 0.0:
        raise FinalResultsArtifactError(f"{field} must be finite and non-negative")
    return result


def _equal_metric_samples(
    actual: EpisodeMetricSamples,
    expected: EpisodeMetricSamples,
) -> bool:
    return (
        actual.track_id == expected.track_id
        and actual.reset_seed == expected.reset_seed
        and np.array_equal(actual.compute_time_s, expected.compute_time_s)
        and np.array_equal(actual.speed_mps, expected.speed_mps)
        and np.array_equal(actual.lateral_error_m, expected.lateral_error_m)
        and np.array_equal(actual.requested_action, expected.requested_action)
        and np.array_equal(actual.steering_saturated, expected.steering_saturated)
        and np.array_equal(actual.longitudinal_saturated, expected.longitudinal_saturated)
    )


@dataclass(frozen=True, slots=True, init=False)
class FinalEpisodeResult:
    """One fixed-order M8 row derived from one canonical public episode and its metrics."""

    controller_name: str
    row_index: int
    track_id: int
    reset_seed: int
    episode_seed: int
    controller_seed: int
    success: bool
    lap_time_s: float | None
    environment_steps: int
    total_reward: float
    terminated: bool
    truncated: bool
    termination_reason: int
    termination_reason_name: str
    controller_import_time_s: float
    controller_init_time_s: float
    compute_timing: TimingSummary
    metric_summary: EpisodeMetricSummary
    benchmark_version: str
    schema_version: str

    def __init__(
        self,
        controller_name: str,
        row_index: int,
        episode: RecordedControllerEpisode,
        metric_samples: EpisodeMetricSamples,
    ) -> None:
        """Validate and freeze a row without consulting simulator internals."""

        controller = _controller_name(controller_name)
        row = _row_index(row_index)
        if not isinstance(episode, RecordedControllerEpisode):
            raise TypeError("episode must be a RecordedControllerEpisode")
        if not isinstance(metric_samples, EpisodeMetricSamples):
            raise TypeError("metric_samples must be an EpisodeMetricSamples")
        if metric_samples.reset_seed != row:
            raise FinalResultsArtifactError("metric reset_seed must equal row_index")

        expected_metrics = compute_episode_metric_samples(
            episode,
            reset_seed=row,
            action_limits=FINAL_RESULTS_ACTION_LIMITS,
        )
        if not _equal_metric_samples(metric_samples, expected_metrics):
            raise FinalResultsArtifactError(
                "metric_samples do not exactly match the canonical public episode"
            )

        trajectory = episode.trajectory
        reset_info = trajectory.reset_info
        final_info = trajectory.final_info
        if reset_info["benchmark_version"] != FINAL_METRICS_BENCHMARK_VERSION:
            raise FinalResultsArtifactError("episode benchmark_version must be exactly '0.1'")
        if final_info["benchmark_version"] != FINAL_METRICS_BENCHMARK_VERSION:
            raise FinalResultsArtifactError("final benchmark_version must be exactly '0.1'")

        identity = initialize_episode_identities(row, 1)
        expected_episode_seed = int(identity.episode_seed[0])
        expected_controller_seed = int(identity.controller_seed[0])
        if reset_info["episode_seed"] != expected_episode_seed:
            raise FinalResultsArtifactError(
                "public episode_seed does not match initialize_episode_identities(row_index, 1)"
            )
        if reset_info["controller_seed"] != expected_controller_seed:
            raise FinalResultsArtifactError(
                "public controller_seed does not match initialize_episode_identities(row_index, 1)"
            )
        if reset_info["track_id"] != metric_samples.track_id:
            raise FinalResultsArtifactError("public Track ID and metric Track ID differ")

        result = episode.result
        if type(result.terminated) is not bool or type(result.truncated) is not bool:
            raise FinalResultsArtifactError("terminal flags must be booleans")
        reason = int(final_info["termination_reason"])
        if reason not in _TERMINATION_NAMES:
            raise FinalResultsArtifactError("final termination_reason must be one of 1, 2, 3, 4")
        success = reason == 1
        if bool(final_info["lap_completed"]) != success:
            raise FinalResultsArtifactError("lap_completed must be true exactly for success")
        raw_lap = float(final_info["lap_time_s"])
        lap_time = raw_lap if success else None
        if success and (not math.isfinite(raw_lap) or raw_lap <= 0.0):
            raise FinalResultsArtifactError("a successful lap_time_s must be finite and positive")
        if not success and raw_lap != 0.0:
            raise FinalResultsArtifactError("an unsuccessful lap_time_s must be neutral")
        if (reason == 4) != bool(result.truncated):
            raise FinalResultsArtifactError("TIMEOUT must be the only truncated outcome")
        if (reason != 4) != bool(result.terminated):
            raise FinalResultsArtifactError("non-TIMEOUT outcomes must be terminated")

        environment_steps = result.steps
        if type(environment_steps) is not int or environment_steps < 1:
            raise FinalResultsArtifactError("environment_steps must be a positive integer")
        if environment_steps != metric_samples.transition_count:
            raise FinalResultsArtifactError(
                "environment step count and metric transition count differ"
            )
        if success and not math.isclose(
            raw_lap,
            environment_steps * FINAL_METRICS_CONTROL_DT_S,
            rel_tol=1.0e-6,
            abs_tol=1.0e-6,
        ):
            raise FinalResultsArtifactError(
                "successful lap_time_s must equal environment_steps times control_dt"
            )
        total_reward = float(result.total_reward)
        if not math.isfinite(total_reward):
            raise FinalResultsArtifactError("total_reward must be finite")
        import_time = _finite_nonnegative(
            result.controller_import_time_s,
            field="controller_import_time_s",
        )
        init_time = _finite_nonnegative(
            result.controller_init_time_s,
            field="controller_init_time_s",
        )
        timing = summarize_compute_times(
            metric_samples.compute_time_s,
            deadline_s=FINAL_RESULTS_COMPUTE_DEADLINE_S,
        )
        metric_summary = summarize_episode_metrics(metric_samples)

        values = {
            "controller_name": controller,
            "row_index": row,
            "track_id": metric_samples.track_id,
            "reset_seed": metric_samples.reset_seed,
            "episode_seed": expected_episode_seed,
            "controller_seed": expected_controller_seed,
            "success": success,
            "lap_time_s": lap_time,
            "environment_steps": environment_steps,
            "total_reward": total_reward,
            "terminated": bool(result.terminated),
            "truncated": bool(result.truncated),
            "termination_reason": reason,
            "termination_reason_name": _TERMINATION_NAMES[reason],
            "controller_import_time_s": import_time,
            "controller_init_time_s": init_time,
            "compute_timing": timing,
            "metric_summary": metric_summary,
            "benchmark_version": FINAL_METRICS_BENCHMARK_VERSION,
            "schema_version": FINAL_CONTROLLER_RESULTS_SCHEMA_VERSION,
        }
        for field, value in values.items():
            object.__setattr__(self, field, value)


@dataclass(frozen=True, slots=True)
class FinalControllerSummary:
    """Aggregate values recomputed from all raw transitions and fixed episode rows."""

    controller_name: str
    track_count: int
    success_count: int
    success_rate: float
    mean_successful_lap_time_s: float | None
    termination_counts: Mapping[str, int]
    environment_steps: int
    metrics: AggregateMetricSummary
    compute_timing: TimingSummary
    mean_controller_import_time_s: float
    mean_controller_init_time_s: float
    benchmark_version: str = FINAL_METRICS_BENCHMARK_VERSION
    schema_version: str = FINAL_CONTROLLER_SUMMARY_SCHEMA_VERSION

    def __post_init__(self) -> None:
        controller = _controller_name(self.controller_name)
        if type(self.track_count) is not int or self.track_count != M8_TEST_TRACK_COUNT:
            raise FinalResultsArtifactError(f"track_count must be exactly {M8_TEST_TRACK_COUNT}")
        if type(self.success_count) is not int or not 0 <= self.success_count <= self.track_count:
            raise FinalResultsArtifactError("success_count is outside the fixed Track count")
        success_rate = _finite_nonnegative(self.success_rate, field="success_rate")
        if success_rate != self.success_count / self.track_count:
            raise FinalResultsArtifactError("success_rate must equal success_count / track_count")
        if self.success_count:
            if self.mean_successful_lap_time_s is None:
                raise FinalResultsArtifactError("successful rows require a mean lap time")
            mean_lap: float | None = _finite_nonnegative(
                self.mean_successful_lap_time_s,
                field="mean_successful_lap_time_s",
            )
            if mean_lap <= 0.0:
                raise FinalResultsArtifactError("mean successful lap time must be positive")
        elif self.mean_successful_lap_time_s is not None:
            raise FinalResultsArtifactError("zero successes require a null mean lap time")
        else:
            mean_lap = None
        if not isinstance(self.termination_counts, Mapping):
            raise TypeError("termination_counts must be a mapping")
        counts = dict(self.termination_counts)
        if set(counts) != set(_TERMINATION_COUNT_KEYS) or any(
            type(value) is not int or value < 0 for value in counts.values()
        ):
            raise FinalResultsArtifactError("termination_counts must cover the four exact outcomes")
        if sum(counts.values()) != self.track_count or counts["success"] != self.success_count:
            raise FinalResultsArtifactError("termination_counts do not match track/success counts")
        if type(self.environment_steps) is not int or self.environment_steps < self.track_count:
            raise FinalResultsArtifactError(
                "environment_steps must include every non-empty episode"
            )
        if not isinstance(self.metrics, AggregateMetricSummary):
            raise TypeError("metrics must be an AggregateMetricSummary")
        if self.metrics.episode_count != self.track_count:
            raise FinalResultsArtifactError("aggregate metrics must include all 20 episodes")
        if self.metrics.transition_count != self.environment_steps:
            raise FinalResultsArtifactError(
                "aggregate metric transitions must equal environment_steps"
            )
        if not isinstance(self.compute_timing, TimingSummary):
            raise TypeError("compute_timing must be a TimingSummary")
        if self.compute_timing.sample_count != self.environment_steps:
            raise FinalResultsArtifactError("raw compute timing must include every transition")
        import_mean = _finite_nonnegative(
            self.mean_controller_import_time_s,
            field="mean_controller_import_time_s",
        )
        init_mean = _finite_nonnegative(
            self.mean_controller_init_time_s,
            field="mean_controller_init_time_s",
        )
        if self.benchmark_version != FINAL_METRICS_BENCHMARK_VERSION:
            raise FinalResultsArtifactError("benchmark_version must be exactly '0.1'")
        if self.schema_version != FINAL_CONTROLLER_SUMMARY_SCHEMA_VERSION:
            raise FinalResultsArtifactError("controller summary schema_version is invalid")

        object.__setattr__(self, "controller_name", controller)
        object.__setattr__(self, "success_rate", success_rate)
        object.__setattr__(self, "mean_successful_lap_time_s", mean_lap)
        object.__setattr__(self, "termination_counts", MappingProxyType(counts))
        object.__setattr__(self, "mean_controller_import_time_s", import_mean)
        object.__setattr__(self, "mean_controller_init_time_s", init_mean)


@dataclass(frozen=True, slots=True)
class FinalControllerResult:
    """Exactly 20 ordered episode rows bound to one canonical ``metrics.npz`` payload."""

    controller_name: str
    episodes: tuple[FinalEpisodeResult, ...]
    metrics: FinalMetricsData

    def __post_init__(self) -> None:
        controller = _controller_name(self.controller_name)
        if not isinstance(self.episodes, tuple) or len(self.episodes) != M8_TEST_TRACK_COUNT:
            raise FinalResultsArtifactError("episodes must be a tuple of exactly 20 rows")
        if not all(isinstance(row, FinalEpisodeResult) for row in self.episodes):
            raise TypeError("episodes must contain only FinalEpisodeResult values")
        if tuple(row.row_index for row in self.episodes) != tuple(range(M8_TEST_TRACK_COUNT)):
            raise FinalResultsArtifactError("episode rows must preserve exact order 0..19")
        if any(row.controller_name != controller for row in self.episodes):
            raise FinalResultsArtifactError("every episode row must use controller_name")
        if not isinstance(self.metrics, FinalMetricsData):
            raise TypeError("metrics must be a FinalMetricsData")
        if self.metrics.controller_name != controller:
            raise FinalResultsArtifactError("metrics Controller name differs from result")

        track_order = tuple(row.track_id for row in self.episodes)
        if track_order != tuple(int(value) for value in self.metrics.track_id):
            raise FinalResultsArtifactError("episode and metric Track order differ")
        if len(set(track_order)) != M8_TEST_TRACK_COUNT:
            raise FinalResultsArtifactError("the 20 Track IDs must be unique")
        if any(left >= right for left, right in pairwise(track_order)):
            raise FinalResultsArtifactError(
                "Track IDs must preserve the official strictly increasing manifest order"
            )
        for index, row in enumerate(self.episodes):
            samples = self.metrics.episode(index)
            if row.reset_seed != index or samples.reset_seed != index:
                raise FinalResultsArtifactError("reset seeds must equal row indices 0..19")
            if row.environment_steps != samples.transition_count:
                raise FinalResultsArtifactError("episode and metric step counts differ")
            if row.metric_summary != summarize_episode_metrics(samples):
                raise FinalResultsArtifactError("episode scalar metrics differ from metrics.npz")
            if row.compute_timing != summarize_compute_times(
                samples.compute_time_s,
                deadline_s=FINAL_RESULTS_COMPUTE_DEADLINE_S,
            ):
                raise FinalResultsArtifactError("episode timing differs from metrics.npz")

        object.__setattr__(self, "controller_name", controller)

    @property
    def summary(self) -> FinalControllerSummary:
        """Recompute the complete Controller summary from rows and transition arrays."""

        successes = tuple(row for row in self.episodes if row.success)
        lap_times = tuple(row.lap_time_s for row in successes)
        counts = {
            name: sum(row.termination_reason_name == name for row in self.episodes)
            for name in _TERMINATION_COUNT_KEYS
        }
        environment_steps = sum(row.environment_steps for row in self.episodes)
        success_count = len(successes)
        return FinalControllerSummary(
            controller_name=self.controller_name,
            track_count=len(self.episodes),
            success_count=success_count,
            success_rate=success_count / len(self.episodes),
            mean_successful_lap_time_s=(
                float(np.mean(lap_times, dtype=np.float64)) if lap_times else None
            ),
            termination_counts=counts,
            environment_steps=environment_steps,
            metrics=summarize_final_metrics(self.metrics),
            compute_timing=summarize_compute_times(
                self.metrics.compute_time_s,
                deadline_s=FINAL_RESULTS_COMPUTE_DEADLINE_S,
            ),
            mean_controller_import_time_s=float(
                np.mean(
                    tuple(row.controller_import_time_s for row in self.episodes),
                    dtype=np.float64,
                )
            ),
            mean_controller_init_time_s=float(
                np.mean(
                    tuple(row.controller_init_time_s for row in self.episodes),
                    dtype=np.float64,
                )
            ),
        )


def _float_text(value: float) -> str:
    result = float(value)
    if not math.isfinite(result):
        raise FinalResultsArtifactError("canonical artifacts cannot contain NaN or infinity")
    if result == 0.0:
        return "0"
    return format(result, ".17g")


def _bool_text(value: bool) -> str:
    return "true" if value else "false"


def _csv_bytes(columns: tuple[str, ...], rows: Sequence[Sequence[object]]) -> bytes:
    output = io.StringIO(newline="")
    writer = csv.writer(output, lineterminator="\n")
    writer.writerow(columns)
    writer.writerows(rows)
    return output.getvalue().encode("ascii")


def _episode_csv_row(row: FinalEpisodeResult) -> tuple[object, ...]:
    timing = row.compute_timing
    metrics = row.metric_summary
    return (
        row.schema_version,
        row.benchmark_version,
        row.controller_name,
        row.row_index,
        row.track_id,
        row.reset_seed,
        row.episode_seed,
        row.controller_seed,
        _bool_text(row.success),
        "" if row.lap_time_s is None else _float_text(row.lap_time_s),
        row.environment_steps,
        _float_text(row.total_reward),
        _bool_text(row.terminated),
        _bool_text(row.truncated),
        row.termination_reason,
        row.termination_reason_name,
        _float_text(row.controller_import_time_s),
        _float_text(row.controller_init_time_s),
        timing.sample_count,
        _float_text(timing.deadline_s),
        _float_text(timing.p50_s),
        _float_text(timing.p95_s),
        _float_text(timing.p99_s),
        _float_text(timing.max_s),
        timing.deadline_miss_count,
        _float_text(timing.deadline_miss_rate),
        metrics.transition_count,
        metrics.action_delta_count,
        _float_text(metrics.mean_speed_mps),
        _float_text(metrics.lateral_error_rms_m),
        _float_text(metrics.lateral_error_abs_p95_m),
        _float_text(metrics.lateral_error_abs_max_m),
        _float_text(metrics.steering_saturation_rate),
        _float_text(metrics.longitudinal_saturation_rate),
        _float_text(metrics.steering_rate_rms_rad_s),
        _float_text(metrics.acceleration_rate_rms_mps3),
    )


def canonical_controller_results_csv_bytes(result: FinalControllerResult) -> bytes:
    """Return the unique LF-terminated 20-row results CSV for one Controller."""

    if not isinstance(result, FinalControllerResult):
        raise TypeError("result must be a FinalControllerResult")
    return _csv_bytes(
        CONTROLLER_RESULTS_CSV_COLUMNS,
        tuple(_episode_csv_row(row) for row in result.episodes),
    )


def _timing_payload(timing: TimingSummary) -> dict[str, int | float]:
    return {
        "sample_count": timing.sample_count,
        "deadline_s": timing.deadline_s,
        "p50_s": timing.p50_s,
        "p95_s": timing.p95_s,
        "p99_s": timing.p99_s,
        "max_s": timing.max_s,
        "deadline_miss_count": timing.deadline_miss_count,
        "deadline_miss_rate": timing.deadline_miss_rate,
    }


def _aggregate_metric_payload(summary: AggregateMetricSummary) -> dict[str, int | float]:
    return {
        "episode_count": summary.episode_count,
        "transition_count": summary.transition_count,
        "action_delta_count": summary.action_delta_count,
        "mean_speed_mps": summary.mean_speed_mps,
        "lateral_error_rms_m": summary.lateral_error_rms_m,
        "lateral_error_abs_p95_m": summary.lateral_error_abs_p95_m,
        "lateral_error_abs_max_m": summary.lateral_error_abs_max_m,
        "steering_saturation_rate": summary.steering_saturation_rate,
        "longitudinal_saturation_rate": summary.longitudinal_saturation_rate,
        "steering_rate_rms_rad_s": summary.steering_rate_rms_rad_s,
        "acceleration_rate_rms_mps3": summary.acceleration_rate_rms_mps3,
    }


def controller_summary_payload(result: FinalControllerResult) -> Mapping[str, object]:
    """Return the exact JSON-compatible summary object recomputed from canonical samples."""

    if not isinstance(result, FinalControllerResult):
        raise TypeError("result must be a FinalControllerResult")
    summary = result.summary
    payload: dict[str, object] = {
        "schema_version": summary.schema_version,
        "benchmark_version": summary.benchmark_version,
        "controller_name": summary.controller_name,
        "track_count": summary.track_count,
        "success_count": summary.success_count,
        "success_rate": summary.success_rate,
        "mean_successful_lap_time_s": summary.mean_successful_lap_time_s,
        "termination_counts": dict(summary.termination_counts),
        "environment_steps": summary.environment_steps,
        "transition_weighted_metrics": _aggregate_metric_payload(summary.metrics),
        "raw_compute_timing": _timing_payload(summary.compute_timing),
        "realtime_qualification": {
            "compute_p99_within_limit": (
                summary.compute_timing.p99_s <= FINAL_RESULTS_REALTIME_P99_LIMIT_S
            ),
            "deadline_miss_rate_within_limit": (
                summary.compute_timing.deadline_miss_rate <= FINAL_RESULTS_REALTIME_MISS_RATE_LIMIT
            ),
            "miss_rate_limit": FINAL_RESULTS_REALTIME_MISS_RATE_LIMIT,
            "p99_limit_s": FINAL_RESULTS_REALTIME_P99_LIMIT_S,
            "qualified": (
                summary.compute_timing.p99_s <= FINAL_RESULTS_REALTIME_P99_LIMIT_S
                and summary.compute_timing.deadline_miss_rate
                <= FINAL_RESULTS_REALTIME_MISS_RATE_LIMIT
            ),
            "required_for_protocol_pass": False,
        },
        "controller_lifecycle_timing": {
            "mean_import_time_s": summary.mean_controller_import_time_s,
            "mean_init_time_s": summary.mean_controller_init_time_s,
        },
    }
    return MappingProxyType(payload)


def canonical_controller_summary_json_bytes(result: FinalControllerResult) -> bytes:
    """Return the unique sorted, indented, LF-terminated Controller summary JSON."""

    content = json.dumps(
        dict(controller_summary_payload(result)),
        allow_nan=False,
        ensure_ascii=True,
        indent=2,
        sort_keys=True,
    )
    return f"{content}\n".encode("ascii")


def _validated_comparison_results(
    results: Mapping[str, FinalControllerResult],
) -> tuple[FinalControllerResult, ...]:
    if not isinstance(results, Mapping):
        raise TypeError("results must be a mapping")
    if any(type(key) is not str for key in results) or set(results) != set(M8_CONTROLLER_ORDER):
        raise FinalResultsArtifactError("results must contain exactly pid, mpc, and ppo")
    ordered = tuple(results[name] for name in M8_CONTROLLER_ORDER)
    if not all(isinstance(result, FinalControllerResult) for result in ordered):
        raise TypeError("results values must be FinalControllerResult values")
    if any(
        result.controller_name != name
        for name, result in zip(M8_CONTROLLER_ORDER, ordered, strict=True)
    ):
        raise FinalResultsArtifactError("comparison keys must match each result Controller name")
    expected_tracks = tuple(row.track_id for row in ordered[0].episodes)
    for result in ordered[1:]:
        if tuple(row.track_id for row in result.episodes) != expected_tracks:
            raise FinalResultsArtifactError("all Controllers must preserve the same Track order")
    return ordered


def rank_final_controller_results(
    results: Mapping[str, FinalControllerResult],
) -> tuple[str, ...]:
    """Return Controller names ranked by success descending then successful lap ascending."""

    ordered = _validated_comparison_results(results)
    rows = []
    for index, result in enumerate(ordered):
        summary = result.summary
        lap = summary.mean_successful_lap_time_s
        rows.append(
            (
                result.controller_name,
                summary.success_rate,
                math.inf if lap is None else lap,
                index,
            )
        )
    return tuple(
        name
        for name, _rate, _lap, _index in sorted(rows, key=lambda row: (-row[1], row[2], row[3]))
    )


def _comparison_csv_row(summary: FinalControllerSummary, *, rank: int) -> tuple[object, ...]:
    metrics = summary.metrics
    timing = summary.compute_timing
    counts = summary.termination_counts
    return (
        FINAL_COMPARISON_SCHEMA_VERSION,
        summary.benchmark_version,
        summary.controller_name,
        rank,
        M8_RANKING_RULE,
        summary.track_count,
        summary.success_count,
        _float_text(summary.success_rate),
        (
            ""
            if summary.mean_successful_lap_time_s is None
            else _float_text(summary.mean_successful_lap_time_s)
        ),
        counts["success"],
        counts["off_track"],
        counts["invalid_action"],
        counts["timeout"],
        summary.environment_steps,
        metrics.transition_count,
        metrics.action_delta_count,
        _float_text(metrics.mean_speed_mps),
        _float_text(metrics.lateral_error_rms_m),
        _float_text(metrics.lateral_error_abs_p95_m),
        _float_text(metrics.lateral_error_abs_max_m),
        _float_text(metrics.steering_saturation_rate),
        _float_text(metrics.longitudinal_saturation_rate),
        _float_text(metrics.steering_rate_rms_rad_s),
        _float_text(metrics.acceleration_rate_rms_mps3),
        timing.sample_count,
        _float_text(timing.deadline_s),
        _float_text(timing.p50_s),
        _float_text(timing.p95_s),
        _float_text(timing.p99_s),
        _float_text(timing.max_s),
        timing.deadline_miss_count,
        _float_text(timing.deadline_miss_rate),
        _float_text(summary.mean_controller_import_time_s),
        _float_text(summary.mean_controller_init_time_s),
    )


def canonical_final_comparison_csv_bytes(
    results: Mapping[str, FinalControllerResult],
) -> bytes:
    """Return fixed PID/MPC/PPO rows with an explicit protocol rank for each Controller."""

    ordered = _validated_comparison_results(results)
    ranks = {
        controller: rank
        for rank, controller in enumerate(rank_final_controller_results(results), start=1)
    }
    return _csv_bytes(
        FINAL_COMPARISON_CSV_COLUMNS,
        tuple(
            _comparison_csv_row(result.summary, rank=ranks[result.controller_name])
            for result in ordered
        ),
    )


def _validate_canonical_bytes(
    content: bytes,
    expected: bytes,
    *,
    artifact: str,
) -> None:
    if type(content) is not bytes:
        raise TypeError(f"{artifact} content must be bytes")
    if content != expected:
        raise FinalResultsArtifactError(
            f"{artifact} is not the exact canonical recomputation from its typed result"
        )


def validate_controller_results_csv_bytes(
    content: bytes,
    expected: FinalControllerResult,
) -> None:
    """Reject any changed header, row, seed, scalar, order, key, or line ending."""

    _validate_canonical_bytes(
        content,
        canonical_controller_results_csv_bytes(expected),
        artifact="Controller results CSV",
    )


def validate_controller_summary_json_bytes(
    content: bytes,
    expected: FinalControllerResult,
) -> None:
    """Reject any summary not exactly recomputed from the raw canonical result."""

    _validate_canonical_bytes(
        content,
        canonical_controller_summary_json_bytes(expected),
        artifact="Controller summary JSON",
    )


def validate_final_comparison_csv_bytes(
    content: bytes,
    expected: Mapping[str, FinalControllerResult],
) -> None:
    """Reject any comparison not exactly recomputed in fixed PID/MPC/PPO order."""

    _validate_canonical_bytes(
        content,
        canonical_final_comparison_csv_bytes(expected),
        artifact="final comparison CSV",
    )


__all__ = [
    "CONTROLLER_RESULTS_CSV_COLUMNS",
    "FINAL_COMPARISON_CSV_COLUMNS",
    "FINAL_COMPARISON_SCHEMA_VERSION",
    "FINAL_CONTROLLER_RESULTS_SCHEMA_VERSION",
    "FINAL_CONTROLLER_SUMMARY_SCHEMA_VERSION",
    "FINAL_RESULTS_ACTION_LIMITS",
    "FINAL_RESULTS_COMPUTE_DEADLINE_S",
    "FINAL_RESULTS_REALTIME_MISS_RATE_LIMIT",
    "FINAL_RESULTS_REALTIME_P99_LIMIT_S",
    "FinalControllerResult",
    "FinalControllerSummary",
    "FinalEpisodeResult",
    "FinalResultsArtifactError",
    "canonical_controller_results_csv_bytes",
    "canonical_controller_summary_json_bytes",
    "canonical_final_comparison_csv_bytes",
    "controller_summary_payload",
    "rank_final_controller_results",
    "validate_controller_results_csv_bytes",
    "validate_controller_summary_json_bytes",
    "validate_final_comparison_csv_bytes",
]
