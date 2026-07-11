"""Synthetic-only tests for deterministic M8 result tables and summaries."""

from __future__ import annotations

import csv
import io
import json
from dataclasses import replace
from types import MappingProxyType

import numpy as np
import pytest

from controller_learning.control import EpisodeRunResult
from controller_learning.envs.episode import initialize_episode_identities
from controller_learning.evaluation.final_metrics import (
    EpisodeMetricSamples,
    build_final_metrics_data,
    compute_episode_metric_samples,
)
from controller_learning.evaluation.final_results import (
    CONTROLLER_RESULTS_CSV_COLUMNS,
    FINAL_COMPARISON_CSV_COLUMNS,
    FINAL_CONTROLLER_SUMMARY_SCHEMA_VERSION,
    FINAL_RESULTS_ACTION_LIMITS,
    FinalControllerResult,
    FinalEpisodeResult,
    FinalResultsArtifactError,
    canonical_controller_results_csv_bytes,
    canonical_controller_summary_json_bytes,
    canonical_final_comparison_csv_bytes,
    controller_summary_payload,
    rank_final_controller_results,
    validate_controller_results_csv_bytes,
    validate_controller_summary_json_bytes,
    validate_final_comparison_csv_bytes,
)
from controller_learning.evaluation.trajectory import (
    EpisodeTrajectory,
    RecordedControllerEpisode,
)


def _recorded_episode(
    *,
    row_index: int,
    reason: int,
    step_count: int,
    track_offset: int = 0,
    seed_offset: int = 0,
    import_time_s: float = 0.001,
    init_time_s: float = 0.002,
    lap_time_override_s: float | None = None,
) -> RecordedControllerEpisode:
    identity = initialize_episode_identities(row_index, 1)
    track_id = 2_000_000 + track_offset + row_index
    episode_seed = int(identity.episode_seed[0]) + seed_offset
    controller_seed = int(identity.controller_seed[0])
    reset_info = MappingProxyType(
        {
            "episode_seed": episode_seed,
            "controller_seed": controller_seed,
            "track_id": track_id,
            "benchmark_version": "0.1",
            "termination_reason": 0,
            "lap_completed": False,
            "lap_time_s": 0.0,
        }
    )
    success = reason == 1
    final_info = MappingProxyType(
        {
            **dict(reset_info),
            "termination_reason": reason,
            "lap_completed": success,
            "lap_time_s": (
                float(step_count * 0.05)
                if success and lap_time_override_s is None
                else float(lap_time_override_s)
                if success
                else 0.0
            ),
        }
    )
    centerline = np.asarray(
        ((0.0, 0.0), (10.0, 0.0), (10.0, 10.0), (0.0, 10.0), (0.0, 0.0)),
        dtype=np.float32,
    )
    positions = np.column_stack(
        (
            np.linspace(0.0, 0.5 * step_count, step_count + 1, dtype=np.float32),
            np.linspace(0.0, 0.1, step_count + 1, dtype=np.float32),
        )
    )
    velocities = np.column_stack(
        (
            np.linspace(0.0, 2.0, step_count + 1, dtype=np.float32),
            np.zeros(step_count + 1, dtype=np.float32),
        )
    )
    actions = np.column_stack(
        (
            np.linspace(-0.1, 0.1, step_count, dtype=np.float32),
            np.linspace(1.0, 2.0, step_count, dtype=np.float32),
        )
    )
    rewards = np.linspace(0.1, 0.2, step_count, dtype=np.float32)
    terminated = np.zeros(step_count, dtype=np.bool_)
    truncated = np.zeros(step_count, dtype=np.bool_)
    if reason == 4:
        truncated[-1] = True
    else:
        terminated[-1] = True
    trajectory = EpisodeTrajectory(
        reset_info=reset_info,
        final_info=final_info,
        centerline_m=centerline,
        left_boundary_m=centerline,
        right_boundary_m=centerline,
        track_mask=np.ones(centerline.shape[0], dtype=np.bool_),
        track_length_m=40.0,
        position_m=positions,
        yaw_rad=np.zeros(step_count + 1, dtype=np.float32),
        velocity_body_mps=velocities,
        yaw_rate_rad_s=np.zeros(step_count + 1, dtype=np.float32),
        steering_angle_rad=np.zeros(step_count + 1, dtype=np.float32),
        track_progress=np.linspace(0.0, 1.0, step_count + 1, dtype=np.float32),
        action=actions,
        reward=rewards,
        terminated=terminated,
        truncated=truncated,
    )
    compute_times = tuple(0.001 + 0.0001 * index for index in range(step_count))
    result = EpisodeRunResult(
        steps=step_count,
        total_reward=float(np.sum(rewards, dtype=np.float64)),
        terminated=reason != 4,
        truncated=reason == 4,
        final_info=final_info,
        debug_commands=(),
        controller_import_time_s=import_time_s,
        controller_init_time_s=init_time_s,
        compute_times_s=compute_times,
    )
    return RecordedControllerEpisode(result=result, trajectory=trajectory)


def _episode_row(
    *,
    controller: str,
    row_index: int,
    reason: int,
    step_count: int,
    track_offset: int = 0,
    seed_offset: int = 0,
    import_time_s: float = 0.001,
    init_time_s: float = 0.002,
) -> tuple[FinalEpisodeResult, EpisodeMetricSamples]:
    episode = _recorded_episode(
        row_index=row_index,
        reason=reason,
        step_count=step_count,
        track_offset=track_offset,
        seed_offset=seed_offset,
        import_time_s=import_time_s,
        init_time_s=init_time_s,
    )
    samples = compute_episode_metric_samples(
        episode,
        reset_seed=row_index,
        action_limits=FINAL_RESULTS_ACTION_LIMITS,
    )
    return FinalEpisodeResult(controller, row_index, episode, samples), samples


def _controller_result(
    controller: str,
    *,
    success_count: int,
    success_step_count: int,
    track_offset: int = 0,
    row_zero_track_offset: int = 0,
) -> FinalControllerResult:
    rows = []
    samples = []
    for row_index in range(20):
        if row_index < success_count:
            reason = 1
            step_count = success_step_count
        else:
            reason = 2 + row_index % 3
            step_count = 2 + row_index % 2
        row, metric = _episode_row(
            controller=controller,
            row_index=row_index,
            reason=reason,
            step_count=step_count,
            track_offset=track_offset + (row_zero_track_offset if row_index == 0 else 0),
            import_time_s=0.001 + row_index * 0.00001,
            init_time_s=0.002 + row_index * 0.00001,
        )
        rows.append(row)
        samples.append(metric)
    return FinalControllerResult(
        controller_name=controller,
        episodes=tuple(rows),
        metrics=build_final_metrics_data(controller, samples),
    )


def _comparison() -> dict[str, FinalControllerResult]:
    return {
        "ppo": _controller_result("ppo", success_count=12, success_step_count=3),
        "pid": _controller_result("pid", success_count=10, success_step_count=2),
        "mpc": _controller_result("mpc", success_count=12, success_step_count=4),
    }


def test_final_episode_result_binds_public_identity_timing_and_metrics() -> None:
    row, samples = _episode_row(
        controller="pid",
        row_index=7,
        reason=1,
        step_count=3,
    )
    identity = initialize_episode_identities(7, 1)

    assert row.controller_name == "pid"
    assert row.row_index == row.reset_seed == 7
    assert row.track_id == 2_000_007
    assert row.episode_seed == int(identity.episode_seed[0])
    assert row.controller_seed == int(identity.controller_seed[0])
    assert row.success is True
    assert row.lap_time_s == pytest.approx(0.15)
    assert row.termination_reason_name == "success"
    assert row.environment_steps == row.compute_timing.sample_count == 3
    assert row.metric_summary.transition_count == samples.transition_count == 3
    assert row.metric_summary.action_delta_count == 2


def test_final_episode_result_rejects_identity_metric_and_nonfinite_drift() -> None:
    episode = _recorded_episode(row_index=0, reason=1, step_count=2)
    samples = compute_episode_metric_samples(
        episode,
        reset_seed=0,
        action_limits=FINAL_RESULTS_ACTION_LIMITS,
    )
    changed_speed = EpisodeMetricSamples(
        track_id=samples.track_id,
        reset_seed=samples.reset_seed,
        compute_time_s=samples.compute_time_s,
        speed_mps=samples.speed_mps + 1.0,
        lateral_error_m=samples.lateral_error_m,
        requested_action=samples.requested_action,
        steering_saturated=samples.steering_saturated,
        longitudinal_saturated=samples.longitudinal_saturated,
    )
    wrong_reset_seed = EpisodeMetricSamples(
        track_id=samples.track_id,
        reset_seed=1,
        compute_time_s=samples.compute_time_s,
        speed_mps=samples.speed_mps,
        lateral_error_m=samples.lateral_error_m,
        requested_action=samples.requested_action,
        steering_saturated=samples.steering_saturated,
        longitudinal_saturated=samples.longitudinal_saturated,
    )

    with pytest.raises(FinalResultsArtifactError, match="canonical public episode"):
        FinalEpisodeResult("pid", 0, episode, changed_speed)
    with pytest.raises(FinalResultsArtifactError, match="reset_seed"):
        FinalEpisodeResult("pid", 0, episode, wrong_reset_seed)
    with pytest.raises(FinalResultsArtifactError, match="controller_name"):
        FinalEpisodeResult("sac", 0, episode, samples)
    with pytest.raises(FinalResultsArtifactError, match="row_index"):
        FinalEpisodeResult("pid", 20, episode, samples)
    with pytest.raises(FinalResultsArtifactError, match="episode_seed"):
        bad_seed_episode = _recorded_episode(
            row_index=0,
            reason=1,
            step_count=2,
            seed_offset=1,
        )
        bad_seed_metrics = compute_episode_metric_samples(
            bad_seed_episode,
            reset_seed=0,
            action_limits=FINAL_RESULTS_ACTION_LIMITS,
        )
        FinalEpisodeResult("pid", 0, bad_seed_episode, bad_seed_metrics)
    with pytest.raises(FinalResultsArtifactError, match="finite and non-negative"):
        bad_timing_episode = _recorded_episode(
            row_index=0,
            reason=1,
            step_count=2,
            import_time_s=float("nan"),
        )
        bad_timing_metrics = compute_episode_metric_samples(
            bad_timing_episode,
            reset_seed=0,
            action_limits=FINAL_RESULTS_ACTION_LIMITS,
        )
        FinalEpisodeResult("pid", 0, bad_timing_episode, bad_timing_metrics)
    with pytest.raises(FinalResultsArtifactError, match="steps times control_dt"):
        bad_lap_episode = _recorded_episode(
            row_index=0,
            reason=1,
            step_count=2,
            lap_time_override_s=1.0,
        )
        bad_lap_metrics = compute_episode_metric_samples(
            bad_lap_episode,
            reset_seed=0,
            action_limits=FINAL_RESULTS_ACTION_LIMITS,
        )
        FinalEpisodeResult("pid", 0, bad_lap_episode, bad_lap_metrics)


def test_final_controller_result_recomputes_complete_transition_weighted_summary() -> None:
    result = _controller_result("pid", success_count=10, success_step_count=2)
    summary = result.summary

    assert summary.schema_version == FINAL_CONTROLLER_SUMMARY_SCHEMA_VERSION
    assert summary.track_count == 20
    assert summary.success_count == 10
    assert summary.success_rate == 0.5
    assert summary.mean_successful_lap_time_s == pytest.approx(0.1)
    assert sum(summary.termination_counts.values()) == 20
    assert summary.termination_counts["success"] == 10
    assert summary.environment_steps == result.metrics.transition_count
    assert summary.metrics.transition_count == summary.environment_steps
    assert summary.metrics.episode_count == 20
    assert summary.compute_timing.sample_count == summary.environment_steps
    assert summary.mean_controller_import_time_s == pytest.approx(0.001095)
    assert summary.mean_controller_init_time_s == pytest.approx(0.002095)


def test_final_controller_result_rejects_incomplete_reordered_or_mismatched_data() -> None:
    result = _controller_result("pid", success_count=10, success_step_count=2)

    with pytest.raises(FinalResultsArtifactError, match="exactly 20"):
        FinalControllerResult("pid", result.episodes[:-1], result.metrics)
    with pytest.raises(FinalResultsArtifactError, match="exact order"):
        FinalControllerResult(
            "pid",
            (result.episodes[1], result.episodes[0], *result.episodes[2:]),
            result.metrics,
        )
    with pytest.raises(FinalResultsArtifactError, match="metrics Controller"):
        FinalControllerResult(
            "pid",
            result.episodes,
            replace(result.metrics, controller_name="mpc"),
        )
    with pytest.raises(FinalResultsArtifactError, match="strictly increasing"):
        _controller_result(
            "pid",
            success_count=10,
            success_step_count=2,
            row_zero_track_offset=100,
        )


def test_controller_results_csv_is_deterministic_complete_and_uses_empty_null_laps() -> None:
    result = _controller_result("pid", success_count=10, success_step_count=2)
    first = canonical_controller_results_csv_bytes(result)
    second = canonical_controller_results_csv_bytes(result)
    parsed = list(csv.DictReader(io.StringIO(first.decode("ascii"))))

    assert first == second
    assert first.endswith(b"\n") and b"\r" not in first
    assert tuple(parsed[0]) == CONTROLLER_RESULTS_CSV_COLUMNS
    assert len(parsed) == 20
    assert [int(row["row_index"]) for row in parsed] == list(range(20))
    assert parsed[0]["lap_time_s"] != ""
    assert parsed[-1]["lap_time_s"] == ""
    assert parsed[-1]["success"] == "false"
    validate_controller_results_csv_bytes(first, result)


@pytest.mark.parametrize("mutation", ["duplicate_header", "duplicate_row", "reorder", "scalar"])
def test_controller_results_csv_validator_rejects_any_noncanonical_table(mutation: str) -> None:
    result = _controller_result("pid", success_count=10, success_step_count=2)
    lines = canonical_controller_results_csv_bytes(result).splitlines(keepends=True)
    if mutation == "duplicate_header":
        changed = lines[0] + b"".join(lines)
    elif mutation == "duplicate_row":
        changed = b"".join((lines[0], lines[1], *lines[1:]))
    elif mutation == "reorder":
        changed = b"".join((lines[0], lines[2], lines[1], *lines[3:]))
    else:
        changed = b"".join(lines).replace(b"2000000", b"2000099", 1)

    with pytest.raises(FinalResultsArtifactError, match="canonical recomputation"):
        validate_controller_results_csv_bytes(changed, result)


def test_controller_summary_json_is_exact_and_has_no_score_or_performance_gate() -> None:
    result = _controller_result("pid", success_count=0, success_step_count=2)
    content = canonical_controller_summary_json_bytes(result)
    payload = json.loads(content)

    assert content.endswith(b"\n") and b"\r" not in content
    assert payload["mean_successful_lap_time_s"] is None
    assert payload["success_count"] == 0
    assert "transition_weighted_metrics" in payload
    assert "raw_compute_timing" in payload
    assert payload["realtime_qualification"] == {
        "compute_p99_within_limit": True,
        "deadline_miss_rate_within_limit": True,
        "miss_rate_limit": 0.01,
        "p99_limit_s": 0.05,
        "qualified": True,
        "required_for_protocol_pass": False,
    }
    assert "score" not in payload
    assert "passed" not in payload
    assert "pass_gate" not in payload
    assert dict(controller_summary_payload(result)) == payload
    validate_controller_summary_json_bytes(content, result)

    payload["extra"] = True
    changed = (json.dumps(payload, sort_keys=True) + "\n").encode("ascii")
    with pytest.raises(FinalResultsArtifactError, match="canonical recomputation"):
        validate_controller_summary_json_bytes(changed, result)


def test_final_comparison_keeps_fixed_rows_and_exposes_protocol_rank() -> None:
    results = _comparison()
    content = canonical_final_comparison_csv_bytes(results)
    rows = list(csv.DictReader(io.StringIO(content.decode("ascii"))))

    assert content.endswith(b"\n") and b"\r" not in content
    assert tuple(rows[0]) == FINAL_COMPARISON_CSV_COLUMNS
    assert [row["controller_name"] for row in rows] == ["pid", "mpc", "ppo"]
    assert [int(row["rank"]) for row in rows] == [3, 2, 1]
    assert rank_final_controller_results(results) == ("ppo", "mpc", "pid")
    assert all("score" not in row for row in rows)
    validate_final_comparison_csv_bytes(content, results)


def test_final_comparison_rejects_missing_controller_or_different_track_order() -> None:
    results = _comparison()
    with pytest.raises(FinalResultsArtifactError, match="exactly pid, mpc, and ppo"):
        canonical_final_comparison_csv_bytes({"pid": results["pid"], "mpc": results["mpc"]})

    changed = dict(results)
    changed["ppo"] = _controller_result(
        "ppo",
        success_count=12,
        success_step_count=3,
        track_offset=100,
    )
    with pytest.raises(FinalResultsArtifactError, match="same Track order"):
        canonical_final_comparison_csv_bytes(changed)


def test_final_comparison_validator_rejects_changed_rank_or_duplicate_row() -> None:
    results = _comparison()
    lines = canonical_final_comparison_csv_bytes(results).splitlines(keepends=True)
    changed_rank = b"".join(lines).replace(b",3,success_rate", b",1,success_rate", 1)
    duplicate = b"".join((lines[0], lines[1], lines[1], lines[3]))

    for content in (changed_rank, duplicate):
        with pytest.raises(FinalResultsArtifactError, match="canonical recomputation"):
            validate_final_comparison_csv_bytes(content, results)
