"""Tests for fixed-order sequential Controller evaluation."""

from __future__ import annotations

from dataclasses import FrozenInstanceError, dataclass, replace
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from controller_learning.config import ProjectConfig, load_project_config
from controller_learning.control import EpisodeRunResult
from controller_learning.evaluation import (
    ControllerEvaluation,
    evaluate_track_batch,
    summarize_compute_times,
)
from controller_learning.tracks.level0 import build_level0_track
from controller_learning.tracks.pool import TrackPool
from controller_learning.tracks.types import Track, TrackBatch, stack_tracks, track_from_batch_row

PROJECT_ROOT = Path(__file__).parents[3]


@pytest.fixture(scope="module")
def project_config() -> ProjectConfig:
    return load_project_config(PROJECT_ROOT)


@pytest.fixture(scope="module")
def track_batch() -> TrackBatch:
    template = build_level0_track()
    return stack_tracks(
        (
            replace(template, seed=101),
            replace(template, seed=102),
        )
    )


@pytest.fixture(scope="module")
def track_pool(project_config: ProjectConfig, track_batch: TrackBatch) -> TrackPool:
    return TrackPool(
        benchmark_version=project_config.benchmark.version,
        generator_version="v0.1",
        split="validation",
        batch=track_batch,
    )


@dataclass
class FakeEnv:
    """Minimal reusable environment carrying an ordered immutable TrackPool."""

    track_pool: TrackPool
    events: list[tuple[Any, ...]]
    closed: bool = False
    track: Track | None = None

    def reset(self, *, seed: int, options: dict[str, Any]) -> tuple[dict[str, Any], dict[str, Any]]:
        index = int(options["track_index"])
        self.track = track_from_batch_row(
            self.track_pool.batch,
            index,
            generator_version=self.track_pool.generator_version,
        )
        self.events.append(("reset", self.track.seed, seed, index))
        return {}, {"track_id": self.track.seed}

    def close(self) -> None:
        self.closed = True
        self.events.append(("close",))


def _result(
    *,
    track_id: int,
    success: bool,
    lap_time_s: float,
    compute_times_s: tuple[float, ...],
) -> EpisodeRunResult:
    return EpisodeRunResult(
        steps=len(compute_times_s),
        total_reward=1.5 if success else -0.25,
        terminated=success,
        truncated=not success,
        final_info={
            "track_id": track_id,
            "lap_completed": success,
            "lap_time_s": lap_time_s,
            "termination_reason": 1 if success else 4,
        },
        debug_commands=(),
        controller_import_time_s=0.003,
        controller_init_time_s=0.004,
        compute_times_s=compute_times_s,
    )


def test_summarize_compute_times_uses_float64_percentiles_and_strict_deadline() -> None:
    values = np.asarray((0.01, 0.05, 0.06, 0.10), dtype=np.float64)

    summary = summarize_compute_times(values)

    expected = np.percentile(values, (50, 95, 99), method="linear")
    assert summary.sample_count == 4
    assert summary.deadline_s == 0.05
    assert summary.p50_s == pytest.approx(expected[0])
    assert summary.p95_s == pytest.approx(expected[1])
    assert summary.p99_s == pytest.approx(expected[2])
    assert summary.max_s == pytest.approx(0.10)
    assert summary.deadline_miss_count == 2
    assert summary.deadline_miss_rate == 0.5


@pytest.mark.parametrize(
    ("values", "deadline", "error"),
    [
        ((), 0.05, ValueError),
        ((float("nan"),), 0.05, ValueError),
        ((float("inf"),), 0.05, ValueError),
        ((-0.01,), 0.05, ValueError),
        ((0.01,), 0.0, ValueError),
        ((0.01,), float("nan"), ValueError),
        (("0.01",), 0.05, TypeError),
        (iter((0.01,)), 0.05, TypeError),
    ],
)
def test_summarize_compute_times_rejects_invalid_samples(values, deadline, error) -> None:
    with pytest.raises(error):
        summarize_compute_times(values, deadline_s=deadline)


def test_evaluate_track_batch_reuses_one_env_preserves_order_and_aggregates(
    project_config: ProjectConfig,
    track_batch: TrackBatch,
    track_pool: TrackPool,
) -> None:
    events: list[tuple[Any, ...]] = []
    environments: list[FakeEnv] = []
    progress = []

    def factory(**kwargs) -> FakeEnv:
        assert kwargs["project_config"] is project_config
        assert kwargs["level_id"] == 1
        assert kwargs["backend"] == "mjx_warp"
        assert kwargs["track_pool"] is track_pool
        env = FakeEnv(track_pool=kwargs["track_pool"], events=events)
        environments.append(env)
        events.append(("create",))
        return env

    def runner(
        env: FakeEnv,
        directory: str,
        reset_seed: int,
        *,
        reset_options: dict[str, Any],
    ) -> EpisodeRunResult:
        env.reset(seed=reset_seed, options=reset_options)
        assert env.track is not None
        events.append(("run", env.track.seed, directory, reset_seed))
        if env.track.seed == 101:
            return _result(
                track_id=101,
                success=True,
                lap_time_s=12.5,
                compute_times_s=(0.01, 0.06),
            )
        return _result(
            track_id=102,
            success=False,
            lap_time_s=0.0,
            compute_times_s=(0.02, 0.05),
        )

    evaluation = evaluate_track_batch(
        project_config,
        1,
        track_batch,
        "v0.1",
        Path("controllers/pid"),
        "mjx_warp",
        track_pool=track_pool,
        env_factory=factory,
        run_episode=runner,
        progress_callback=progress.append,
    )

    assert events == [
        ("create",),
        ("reset", 101, 0, 0),
        ("run", 101, "controllers/pid", 0),
        ("reset", 102, 1, 1),
        ("run", 102, "controllers/pid", 1),
        ("close",),
    ]
    assert all(env.closed for env in environments)
    assert tuple(episode.track_index for episode in progress) == (0, 1)
    assert tuple(episode.track_id for episode in progress) == (101, 102)
    assert evaluation.controller_directory == "controllers/pid"
    assert evaluation.level_id == 1
    assert evaluation.backend == "mjx_warp"
    assert evaluation.track_count == 2
    assert evaluation.success_count == 1
    assert evaluation.success_rate == 0.5
    assert evaluation.mean_successful_lap_time_s == 12.5
    assert tuple(episode.track_index for episode in evaluation.episodes) == (0, 1)
    assert tuple(episode.track_id for episode in evaluation.episodes) == (101, 102)
    assert tuple(episode.reset_seed for episode in evaluation.episodes) == (0, 1)
    assert evaluation.episodes[0].lap_time_s == 12.5
    assert evaluation.episodes[1].lap_time_s is None
    assert evaluation.compute_timing.sample_count == 4
    assert evaluation.compute_timing.deadline_miss_count == 1
    assert evaluation.compute_timing.deadline_miss_rate == 0.25
    with pytest.raises(FrozenInstanceError):
        evaluation.success_rate = 1.0  # type: ignore[misc]


def test_evaluate_track_batch_without_pool_preserves_per_track_environment_path(
    project_config: ProjectConfig,
    track_batch: TrackBatch,
) -> None:
    events: list[tuple[Any, ...]] = []

    @dataclass
    class PerTrackEnv:
        track: Track

        def close(self) -> None:
            events.append(("close", self.track.seed))

    def factory(**kwargs) -> PerTrackEnv:
        assert "track_pool" not in kwargs
        env = PerTrackEnv(track=kwargs["track"])
        events.append(("create", env.track.seed))
        return env

    def runner(env: PerTrackEnv, directory: str, reset_seed: int) -> EpisodeRunResult:
        events.append(("run", env.track.seed, directory, reset_seed))
        return _result(
            track_id=env.track.seed,
            success=True,
            lap_time_s=1.0 + reset_seed,
            compute_times_s=(0.001,),
        )

    evaluation = evaluate_track_batch(
        project_config,
        1,
        track_batch,
        "v0.1",
        "controllers/pid",
        "cpu_reference",
        env_factory=factory,
        run_episode=runner,
    )

    assert events == [
        ("create", 101),
        ("run", 101, "controllers/pid", 0),
        ("close", 101),
        ("create", 102),
        ("run", 102, "controllers/pid", 1),
        ("close", 102),
    ]
    assert tuple(episode.track_id for episode in evaluation.episodes) == (101, 102)


def test_evaluate_track_batch_forwards_explicit_reset_seeds(
    project_config: ProjectConfig,
    track_batch: TrackBatch,
    track_pool: TrackPool,
) -> None:
    received: list[int] = []

    def factory(**kwargs) -> FakeEnv:
        return FakeEnv(track_pool=kwargs["track_pool"], events=[])

    def runner(
        env: FakeEnv,
        _directory: str,
        reset_seed: int,
        *,
        reset_options: dict[str, Any],
    ) -> EpisodeRunResult:
        env.reset(seed=reset_seed, options=reset_options)
        assert env.track is not None
        received.append(reset_seed)
        return _result(
            track_id=env.track.seed,
            success=True,
            lap_time_s=float(reset_seed),
            compute_times_s=(0.001,),
        )

    result = evaluate_track_batch(
        project_config,
        1,
        track_batch,
        "v0.1",
        "controllers/pid",
        "cpu_reference",
        reset_seeds=np.asarray((11, 17), dtype=np.uint32),
        track_pool=track_pool,
        env_factory=factory,
        run_episode=runner,
    )

    assert received == [11, 17]
    assert tuple(episode.reset_seed for episode in result.episodes) == (11, 17)
    assert result.mean_successful_lap_time_s == 14.0


def test_evaluate_track_batch_closes_before_propagating_runner_exception(
    project_config: ProjectConfig,
    track_batch: TrackBatch,
    track_pool: TrackPool,
) -> None:
    environments: list[FakeEnv] = []
    failure = RuntimeError("controller failed")

    def factory(**kwargs) -> FakeEnv:
        env = FakeEnv(track_pool=kwargs["track_pool"], events=[])
        environments.append(env)
        return env

    def runner(*_args, **_kwargs) -> EpisodeRunResult:
        raise failure

    with pytest.raises(RuntimeError) as caught:
        evaluate_track_batch(
            project_config,
            1,
            track_batch,
            "v0.1",
            "controllers/pid",
            "mjx_warp",
            track_pool=track_pool,
            env_factory=factory,
            run_episode=runner,
        )

    assert caught.value is failure
    assert len(environments) == 1
    assert environments[0].closed is True


@pytest.mark.parametrize(
    ("overrides", "error"),
    [
        ({"project_config": object()}, TypeError),
        ({"level_id": True}, TypeError),
        ({"level_id": 9}, ValueError),
        ({"batch": object()}, TypeError),
        ({"generator_version": ""}, ValueError),
        ({"controller_directory": ""}, ValueError),
        ({"backend": "other"}, ValueError),
        ({"reset_seeds": (1,)}, ValueError),
        ({"reset_seeds": (1, -1)}, ValueError),
        ({"reset_seeds": (1, 2**32)}, ValueError),
        ({"reset_seeds": (1, 1.5)}, TypeError),
        ({"track_pool": object()}, TypeError),
        ({"env_factory": None}, TypeError),
        ({"run_episode": None}, TypeError),
        ({"progress_callback": 1}, TypeError),
    ],
)
def test_evaluate_track_batch_rejects_invalid_inputs(
    project_config: ProjectConfig,
    track_batch: TrackBatch,
    track_pool: TrackPool,
    overrides: dict[str, Any],
    error: type[Exception],
) -> None:
    arguments: dict[str, Any] = {
        "project_config": project_config,
        "level_id": 1,
        "batch": track_batch,
        "generator_version": "v0.1",
        "controller_directory": "controllers/pid",
        "backend": "cpu_reference",
        "reset_seeds": (3, 4),
        "track_pool": track_pool,
        "env_factory": lambda **_kwargs: None,
        "run_episode": lambda *_args: None,
    }
    arguments.update(overrides)

    with pytest.raises(error):
        evaluate_track_batch(**arguments)


def test_multi_track_evaluation_rejects_a_mismatched_ordered_pool(
    project_config: ProjectConfig,
    track_batch: TrackBatch,
    track_pool: TrackPool,
) -> None:
    arguments = {
        "project_config": project_config,
        "level_id": 1,
        "batch": track_batch,
        "generator_version": "v0.1",
        "controller_directory": "controllers/pid",
        "backend": "cpu_reference",
        "env_factory": lambda **_kwargs: None,
        "run_episode": lambda *_args, **_kwargs: None,
    }
    reversed_batch = TrackBatch(*(np.array(value[::-1], copy=True) for value in track_batch))
    with pytest.raises(ValueError, match="do not exactly match"):
        evaluate_track_batch(
            **arguments,
            track_pool=replace(track_pool, batch=reversed_batch),
        )


def test_evaluate_track_batch_rejects_mismatched_runner_track_id_after_close(
    project_config: ProjectConfig,
    track_batch: TrackBatch,
    track_pool: TrackPool,
) -> None:
    environments: list[FakeEnv] = []

    def factory(**kwargs) -> FakeEnv:
        env = FakeEnv(track_pool=kwargs["track_pool"], events=[])
        environments.append(env)
        return env

    def runner(*_args, **_kwargs) -> EpisodeRunResult:
        return _result(
            track_id=999,
            success=True,
            lap_time_s=2.0,
            compute_times_s=(0.001,),
        )

    with pytest.raises(ValueError, match="does not match"):
        evaluate_track_batch(
            project_config,
            1,
            track_batch,
            "v0.1",
            "controllers/pid",
            "cpu_reference",
            track_pool=track_pool,
            env_factory=factory,
            run_episode=runner,
        )

    assert environments[0].closed is True


def test_controller_evaluation_requires_immutable_episode_tuple(
    project_config: ProjectConfig,
    track_batch: TrackBatch,
    track_pool: TrackPool,
) -> None:
    def factory(**kwargs) -> FakeEnv:
        return FakeEnv(track_pool=kwargs["track_pool"], events=[])

    def runner(
        env: FakeEnv,
        _directory: str,
        reset_seed: int,
        *,
        reset_options: dict[str, Any],
    ) -> EpisodeRunResult:
        env.reset(seed=reset_seed, options=reset_options)
        assert env.track is not None
        return _result(
            track_id=env.track.seed,
            success=True,
            lap_time_s=1.0,
            compute_times_s=(0.001,),
        )

    evaluation = evaluate_track_batch(
        project_config,
        1,
        track_batch,
        "v0.1",
        "controllers/pid",
        "cpu_reference",
        track_pool=track_pool,
        env_factory=factory,
        run_episode=runner,
    )

    with pytest.raises(TypeError, match="tuple"):
        replace(evaluation, episodes=list(evaluation.episodes))  # type: ignore[arg-type]
    assert isinstance(evaluation, ControllerEvaluation)
