"""Tests for the asset-agnostic shared-environment M8 execution core."""

from __future__ import annotations

import inspect
from collections.abc import Callable
from pathlib import Path
from typing import Any

import numpy as np
import pytest

import controller_learning.evaluation.final_execution as execution_module
from controller_learning.control import EpisodeRunResult
from controller_learning.envs.episode import initialize_episode_identities
from controller_learning.evaluation.final_execution import (
    MAX_SANITIZED_TRACEBACK_CHARS,
    FinalWorkloadExecutionError,
    execute_controller_workload,
)
from controller_learning.evaluation.final_metrics import (
    EpisodeMetricSamples,
    MetricActionLimits,
)
from controller_learning.evaluation.trajectory import (
    EpisodeTrajectory,
    RecordedControllerEpisode,
)


class _EnvironmentSentinel:
    """An object that exposes no private simulator or lifecycle surface."""

    close_calls = 0

    @property
    def unwrapped(self) -> object:
        raise AssertionError("the workload core must not inspect the environment")

    def close(self) -> None:
        self.close_calls += 1
        raise AssertionError("the workload core must not close the environment")


CONTROLLER_DIRECTORIES = {
    "pid": "controllers/pid",
    "mpc": "controllers/mpc",
    "ppo": "controllers/ppo",
}
ACTION_LIMITS = MetricActionLimits(
    max_steering_angle_rad=0.6,
    max_acceleration_mps2=4.0,
    max_deceleration_mps2=8.0,
)


def _public_info(
    *,
    row_index: int,
    track_id: int,
    terminal: bool,
    benchmark_version: str = "0.1",
) -> dict[str, int | float | bool | str]:
    identity = initialize_episode_identities(row_index, 1)
    return {
        "episode_seed": int(identity.episode_seed[0]),
        "controller_seed": int(identity.controller_seed[0]),
        "track_id": track_id,
        "benchmark_version": benchmark_version,
        "termination_reason": 1 if terminal else 0,
        "lap_completed": terminal,
        "lap_time_s": 0.05 if terminal else 0.0,
    }


def _recorded_episode(
    *,
    row_index: int,
    track_id: int,
    controller_init_time_s: float = 0.01,
    benchmark_version: str = "0.1",
) -> RecordedControllerEpisode:
    reset_info = _public_info(
        row_index=row_index,
        track_id=track_id,
        terminal=False,
        benchmark_version=benchmark_version,
    )
    final_info = _public_info(
        row_index=row_index,
        track_id=track_id,
        terminal=True,
        benchmark_version=benchmark_version,
    )
    centerline = np.asarray(
        [[0.0, 0.0], [10.0, 0.0], [10.0, 10.0], [0.0, 10.0], [0.0, 0.0]],
        dtype=np.float32,
    )
    trajectory = EpisodeTrajectory(
        reset_info=reset_info,
        final_info=final_info,
        centerline_m=centerline,
        left_boundary_m=centerline + np.asarray([0.0, 2.0], dtype=np.float32),
        right_boundary_m=centerline - np.asarray([0.0, 2.0], dtype=np.float32),
        track_mask=np.ones(5, dtype=np.bool_),
        track_length_m=40.0,
        position_m=np.asarray([[0.0, 0.0], [0.2, 0.0]], dtype=np.float32),
        yaw_rad=np.asarray([0.0, 0.0], dtype=np.float32),
        velocity_body_mps=np.asarray([[0.0, 0.0], [4.0, 0.0]], dtype=np.float32),
        yaw_rate_rad_s=np.asarray([0.0, 0.0], dtype=np.float32),
        steering_angle_rad=np.asarray([0.0, 0.0], dtype=np.float32),
        track_progress=np.asarray([0.0, 1.0], dtype=np.float32),
        action=np.asarray([[0.1, 1.0]], dtype=np.float32),
        reward=np.asarray([1.0], dtype=np.float32),
        terminated=np.asarray([True], dtype=np.bool_),
        truncated=np.asarray([False], dtype=np.bool_),
    )
    result = EpisodeRunResult(
        steps=1,
        total_reward=1.0,
        terminated=True,
        truncated=False,
        final_info=final_info,
        debug_commands=(),
        controller_import_time_s=0.001,
        controller_init_time_s=controller_init_time_s,
        compute_times_s=(0.002,),
    )
    return RecordedControllerEpisode(result=result, trajectory=trajectory)


def _install_recorder(
    monkeypatch: pytest.MonkeyPatch,
    *,
    track_ids: tuple[int, ...],
    event_log: list[tuple[Any, ...]],
    init_time: Callable[[str, int], float] | None = None,
) -> None:
    def record(
        environment: object,
        controller_directory: str | Path,
        reset_seed: int,
        render: bool,
        max_steps: int,
        *,
        reset_options: dict[str, int],
    ) -> RecordedControllerEpisode:
        controller_name = Path(controller_directory).name
        event_log.append(
            (
                "record",
                controller_name,
                reset_seed,
                id(environment),
                render,
                max_steps,
                dict(reset_options),
            )
        )
        controller_init_time_s = (
            0.01 if init_time is None else init_time(controller_name, reset_seed)
        )
        return _recorded_episode(
            row_index=reset_seed,
            track_id=track_ids[reset_seed],
            controller_init_time_s=controller_init_time_s,
        )

    monkeypatch.setattr(execution_module, "record_controller_episode", record)


def test_executes_exact_60_controller_major_rows_through_one_environment(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    environment = _EnvironmentSentinel()
    track_ids = tuple(2_000_000 + row for row in range(20))
    events: list[tuple[Any, ...]] = []
    _install_recorder(monkeypatch, track_ids=track_ids, event_log=events)

    def sink(
        controller_name: str,
        row_index: int,
        recorded: RecordedControllerEpisode,
        samples: EpisodeMetricSamples,
    ) -> None:
        assert recorded.trajectory.reset_info["track_id"] == samples.track_id
        assert recorded.trajectory.step_count == samples.transition_count
        events.append(("sink", controller_name, row_index, id(recorded), id(samples)))

    result = execute_controller_workload(
        environment=environment,
        controller_directories=CONTROLLER_DIRECTORIES,
        track_ids=np.asarray(track_ids, dtype=np.uint32),
        action_limits=ACTION_LIMITS,
        max_episode_steps=4_000,
        episode_sink=sink,
    )

    expected_rows = [
        (controller_name, row_index)
        for controller_name in ("pid", "mpc", "ppo")
        for row_index in range(20)
    ]
    observed_rows = [(event[1], event[2]) for event in events if event[0] == "record"]
    assert observed_rows == expected_rows
    assert len(events) == 120
    for index, expected_row in enumerate(expected_rows):
        record_event = events[2 * index]
        sink_event = events[2 * index + 1]
        assert record_event[:3] == ("record", *expected_row)
        assert sink_event[:3] == ("sink", *expected_row)
        assert record_event[3:] == (id(environment), False, 4_000, {"track_index": expected_row[1]})

    assert result.controller_order == ("pid", "mpc", "ppo")
    assert result.track_ids == track_ids
    assert result.environment_instance_count == 1
    assert result.fresh_runner_instance_count == 60
    assert result.episode_count == 60
    assert environment.close_calls == 0
    recorded_object_ids: list[int] = []
    for controller_name in result.controller_order:
        controller_result = result.controller_results[controller_name]
        assert controller_result.episode_count == 20
        assert (
            tuple(
                int(episode.trajectory.reset_info["track_id"])
                for episode in controller_result.recorded_episodes
            )
            == track_ids
        )
        assert tuple(sample.reset_seed for sample in controller_result.metric_samples) == tuple(
            range(20)
        )
        for row_index, episode in enumerate(controller_result.recorded_episodes):
            identity = initialize_episode_identities(row_index, 1)
            assert episode.trajectory.reset_info["episode_seed"] == int(identity.episode_seed[0])
            assert episode.trajectory.reset_info["controller_seed"] == int(
                identity.controller_seed[0]
            )
            recorded_object_ids.append(id(episode))
        assert controller_result.wall_time_s >= 0.0
    assert len(set(recorded_object_ids)) == 60

    with pytest.raises(TypeError):
        result.controller_results["pid"] = result.controller_results["mpc"]  # type: ignore[index]


def test_initialization_over_30_seconds_is_a_diagnostic_not_a_failure(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    track_ids = (200, 201)
    events: list[tuple[Any, ...]] = []
    _install_recorder(
        monkeypatch,
        track_ids=track_ids,
        event_log=events,
        init_time=lambda controller, row: 30.000_001 if (controller, row) == ("mpc", 1) else 30.0,
    )

    result = execute_controller_workload(
        environment=_EnvironmentSentinel(),
        controller_directories=CONTROLLER_DIRECTORIES,
        track_ids=track_ids,
        action_limits=ACTION_LIMITS,
        max_episode_steps=10,
        expected_track_count=2,
    )

    assert result.controller_results["pid"].initialization_over_30s_rows == ()
    assert result.controller_results["mpc"].initialization_over_30s_rows == (1,)
    assert result.controller_results["ppo"].initialization_over_30s_rows == ()
    assert result.fresh_runner_instance_count == 6


@pytest.mark.parametrize(
    ("controller_directories", "error_type"),
    [
        ({"pid": "pid", "mpc": "mpc"}, ValueError),
        ({"pid": "pid", "mpc": "mpc", "ppo": "ppo", "extra": "extra"}, ValueError),
        ({"mpc": "mpc", "pid": "pid", "ppo": "ppo"}, ValueError),
        ({"pid": "pid", "mpc": "mpc", "ppo": object()}, TypeError),
    ],
)
def test_rejects_noncanonical_controller_mappings(
    controller_directories: dict[str, object],
    error_type: type[Exception],
) -> None:
    with pytest.raises(error_type):
        execute_controller_workload(
            environment=_EnvironmentSentinel(),
            controller_directories=controller_directories,  # type: ignore[arg-type]
            track_ids=(1,),
            action_limits=ACTION_LIMITS,
            max_episode_steps=1,
            expected_track_count=1,
        )


@pytest.mark.parametrize(
    ("track_ids", "expected_count", "error_type"),
    [
        ((1,), 2, ValueError),
        ((1, 1), 2, ValueError),
        ((True,), 1, TypeError),
        ((-1,), 1, ValueError),
        ((2**32,), 1, ValueError),
        (np.asarray([[1]], dtype=np.uint32), 1, ValueError),
        ((1,), 0, ValueError),
        ((1,), True, TypeError),
    ],
)
def test_rejects_invalid_track_schedule(
    track_ids: object,
    expected_count: object,
    error_type: type[Exception],
) -> None:
    with pytest.raises(error_type):
        execute_controller_workload(
            environment=_EnvironmentSentinel(),
            controller_directories=CONTROLLER_DIRECTORIES,
            track_ids=track_ids,  # type: ignore[arg-type]
            action_limits=ACTION_LIMITS,
            max_episode_steps=1,
            expected_track_count=expected_count,  # type: ignore[arg-type]
        )


def test_rejects_public_identity_drift_before_metrics_or_sink(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[str] = []

    def record(*args: object, **kwargs: object) -> RecordedControllerEpisode:
        calls.append("record")
        return _recorded_episode(row_index=0, track_id=99, benchmark_version="0.2")

    monkeypatch.setattr(execution_module, "record_controller_episode", record)

    with pytest.raises(FinalWorkloadExecutionError) as caught:
        execute_controller_workload(
            environment=_EnvironmentSentinel(),
            controller_directories=CONTROLLER_DIRECTORIES,
            track_ids=(99,),
            action_limits=ACTION_LIMITS,
            max_episode_steps=1,
            episode_sink=lambda *args: calls.append("sink"),
            expected_track_count=1,
        )

    assert caught.value.controller_name == "pid"
    assert caught.value.row_index == 0
    assert caught.value.phase == "validate_episode"
    assert calls == ["record"]


def test_rejects_metric_identity_drift_before_sink(monkeypatch: pytest.MonkeyPatch) -> None:
    track_ids = (99,)
    events: list[tuple[Any, ...]] = []
    _install_recorder(monkeypatch, track_ids=track_ids, event_log=events)
    real_compute = execution_module.compute_episode_metric_samples

    def wrong_metrics(*args: object, **kwargs: object) -> EpisodeMetricSamples:
        samples = real_compute(*args, **kwargs)  # type: ignore[arg-type]
        return EpisodeMetricSamples(
            track_id=samples.track_id,
            reset_seed=1,
            compute_time_s=samples.compute_time_s,
            speed_mps=samples.speed_mps,
            lateral_error_m=samples.lateral_error_m,
            requested_action=samples.requested_action,
            steering_saturated=samples.steering_saturated,
            longitudinal_saturated=samples.longitudinal_saturated,
        )

    monkeypatch.setattr(execution_module, "compute_episode_metric_samples", wrong_metrics)

    with pytest.raises(FinalWorkloadExecutionError) as caught:
        execute_controller_workload(
            environment=_EnvironmentSentinel(),
            controller_directories=CONTROLLER_DIRECTORIES,
            track_ids=track_ids,
            action_limits=ACTION_LIMITS,
            max_episode_steps=1,
            episode_sink=lambda *args: events.append(("sink",)),
            expected_track_count=1,
        )

    assert caught.value.phase == "validate_metrics"
    assert not any(event[0] == "sink" for event in events)


def test_ordinary_failure_stops_and_exposes_only_bounded_sanitized_evidence(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    secret_path = tmp_path / "private" / "controller.py"
    calls = 0

    def fail(*args: object, **kwargs: object) -> RecordedControllerEpisode:
        nonlocal calls
        calls += 1
        raise RuntimeError(f"failed at {secret_path} " + "x" * 8_000)

    monkeypatch.setattr(execution_module, "record_controller_episode", fail)

    with pytest.raises(FinalWorkloadExecutionError) as caught:
        execute_controller_workload(
            environment=_EnvironmentSentinel(),
            controller_directories=CONTROLLER_DIRECTORIES,
            track_ids=(10, 11),
            action_limits=ACTION_LIMITS,
            max_episode_steps=1,
            expected_track_count=2,
        )

    error = caught.value
    assert error.controller_name == "pid"
    assert error.row_index == 0
    assert error.phase == "record_episode"
    assert error.cause_type == "RuntimeError"
    assert calls == 1
    assert str(secret_path) not in str(error)
    assert str(secret_path) not in error.sanitized_traceback
    assert str(tmp_path) not in error.sanitized_traceback
    assert len(error.sanitized_traceback) <= MAX_SANITIZED_TRACEBACK_CHARS
    assert error.sanitized_traceback.endswith("<truncated>")


def test_sink_failure_is_typed_and_prevents_the_next_row(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    track_ids = (100, 101)
    events: list[tuple[Any, ...]] = []
    _install_recorder(monkeypatch, track_ids=track_ids, event_log=events)

    def sink(*args: object) -> None:
        events.append(("sink",))
        raise OSError(f"cannot persist {tmp_path / 'bundle.bin'}")

    with pytest.raises(FinalWorkloadExecutionError) as caught:
        execute_controller_workload(
            environment=_EnvironmentSentinel(),
            controller_directories=CONTROLLER_DIRECTORIES,
            track_ids=track_ids,
            action_limits=ACTION_LIMITS,
            max_episode_steps=1,
            episode_sink=sink,
            expected_track_count=2,
        )

    assert caught.value.phase == "episode_sink"
    assert caught.value.controller_name == "pid"
    assert caught.value.row_index == 0
    assert [event[0] for event in events] == ["record", "sink"]
    assert str(tmp_path) not in caught.value.sanitized_traceback


def test_base_exceptions_are_not_converted_to_workload_failures(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def interrupt(*args: object, **kwargs: object) -> RecordedControllerEpisode:
        raise KeyboardInterrupt

    monkeypatch.setattr(execution_module, "record_controller_episode", interrupt)
    with pytest.raises(KeyboardInterrupt):
        execute_controller_workload(
            environment=_EnvironmentSentinel(),
            controller_directories=CONTROLLER_DIRECTORIES,
            track_ids=(1,),
            action_limits=ACTION_LIMITS,
            max_episode_steps=1,
            expected_track_count=1,
        )


def test_core_source_has_no_asset_environment_lifecycle_or_replay_dependencies() -> None:
    source = inspect.getsource(execution_module)
    assert "test_assets" not in source
    assert "load_verified_test_pool" not in source
    assert "CarRacingEnv" not in source
    assert "TrackPool" not in source
    assert ".unwrapped" not in source
    assert ".close(" not in source
    assert "render=True" not in source
