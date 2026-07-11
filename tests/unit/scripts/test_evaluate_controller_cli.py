"""Tests for the informal Controller evaluation command."""

from __future__ import annotations

import json
import shutil
import subprocess
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from controller_learning.config import load_project_config
from controller_learning.control import EpisodeRunResult
from controller_learning.evaluation.controller import (
    ControllerEvaluation,
    EpisodeEvaluation,
    summarize_compute_times,
)
from scripts import evaluate_controller

PROJECT_ROOT = Path(__file__).parents[3]


def _run(command: tuple[str, ...], *, cwd: Path) -> None:
    subprocess.run(command, cwd=cwd, check=True, capture_output=True, text=True)


def _project(tmp_path: Path) -> Path:
    root = tmp_path / "project"
    root.mkdir()
    shutil.copytree(PROJECT_ROOT / "configs", root / "configs")
    (root / ".gitignore").write_text("runs/\n", encoding="utf-8")
    controller = root / "controllers" / "example"
    controller.mkdir(parents=True)
    (controller / "controller.py").write_text("VALUE = 1\n", encoding="utf-8")
    (controller / "config.toml").write_text('name = "example"\n', encoding="utf-8")
    _run(("git", "init", "-q"), cwd=root)
    _run(("git", "add", "."), cwd=root)
    _run(
        (
            "git",
            "-c",
            "user.name=Controller Learning Tests",
            "-c",
            "user.email=tests@example.invalid",
            "commit",
            "-qm",
            "test fixture",
        ),
        cwd=root,
    )
    return root


def _episode(track_index: int, track_id: int, reset_seed: int) -> EpisodeEvaluation:
    timing = summarize_compute_times((0.001 + track_index * 0.0001,))
    return EpisodeEvaluation(
        track_index=track_index,
        track_id=track_id,
        reset_seed=reset_seed,
        success=True,
        lap_time_s=10.0 + track_index,
        steps=1,
        total_reward=1.0,
        terminated=True,
        truncated=False,
        termination_reason=1,
        controller_import_time_s=0.002,
        controller_init_time_s=0.003,
        compute_times_s=(timing.p50_s,),
        compute_timing=timing,
    )


def _fake_evaluator(*args, **kwargs) -> ControllerEvaluation:
    project_config, level_id, batch, _generator, controller_directory, backend = args
    reset_seeds = tuple(int(value) for value in kwargs["reset_seeds"])
    episodes = tuple(
        _episode(index, int(track_id), reset_seeds[index])
        for index, track_id in enumerate(batch.seed)
    )
    callback = kwargs.get("progress_callback")
    if callback is not None:
        for episode in episodes:
            callback(episode)
    all_times = tuple(value for episode in episodes for value in episode.compute_times_s)
    return ControllerEvaluation(
        controller_directory=str(controller_directory),
        level_id=level_id,
        backend=backend,
        episodes=episodes,
        track_count=len(episodes),
        success_count=len(episodes),
        success_rate=1.0,
        mean_successful_lap_time_s=float(
            np.mean([episode.lap_time_s for episode in episodes], dtype=np.float64)
        ),
        compute_timing=summarize_compute_times(
            all_times,
            deadline_s=project_config.benchmark.controller.compute_deadline_s,
        ),
    )


def _result(track_id: int) -> EpisodeRunResult:
    return EpisodeRunResult(
        steps=1,
        total_reward=1.0,
        terminated=True,
        truncated=False,
        final_info={
            "track_id": track_id,
            "lap_completed": True,
            "lap_time_s": 1.0,
            "termination_reason": 1,
        },
        debug_commands=(),
        controller_import_time_s=0.001,
        controller_init_time_s=0.001,
        compute_times_s=(0.001,),
    )


def test_parser_exposes_only_development_splits() -> None:
    options = evaluate_controller._parse_args(
        (
            "--controller",
            "controllers/pid",
            "--run-id",
            "pid-validation",
            "--split",
            "validation",
            "--backend",
            "mjx_warp",
            "--count",
            "3",
            "--capture-row",
            "1",
        )
    )
    assert options == evaluate_controller.DevelopmentEvaluationOptions(
        controller_directory=Path("controllers/pid"),
        run_id="pid-validation",
        split="validation",
        backend="mjx_warp",
        count=3,
        capture_row=1,
    )

    with pytest.raises(SystemExit):
        evaluate_controller._parse_args(
            (
                "--controller",
                "controllers/pid",
                "--run-id",
                "forbidden",
                "--split",
                "test",
            )
        )


@pytest.mark.parametrize("run_id", ("UPPER", "../escape", "trailing-", ""))
def test_options_reject_unsafe_run_ids(run_id: str) -> None:
    with pytest.raises(ValueError, match="run_id"):
        evaluate_controller.DevelopmentEvaluationOptions(
            controller_directory=Path("controllers/pid"),
            run_id=run_id,
            split="level0",
        )


def test_level0_loader_opens_only_level0_assets(monkeypatch: pytest.MonkeyPatch) -> None:
    project = load_project_config(PROJECT_ROOT)
    names: list[str] = []
    actual_digest = evaluate_controller.sha256_file
    actual_manifest = evaluate_controller.load_track_asset_manifest
    actual_batch = evaluate_controller.load_verified_manifest_batch

    def digest(path: str | Path) -> str:
        names.append(Path(path).name)
        return actual_digest(path)

    def manifest(path: str | Path):
        names.append(Path(path).name)
        return actual_manifest(path)

    def batch(value, path: str | Path):
        names.append(Path(path).name)
        return actual_batch(value, path)

    monkeypatch.setattr(evaluate_controller, "sha256_file", digest)
    monkeypatch.setattr(evaluate_controller, "load_track_asset_manifest", manifest)
    monkeypatch.setattr(evaluate_controller, "load_verified_manifest_batch", batch)

    prepared = evaluate_controller._load_level0_tracks(project)

    assert prepared.split == "level0"
    assert prepared.track_pool is None
    assert prepared.batch.seed.shape == (1,)
    assert set(names) == {"level0.json", "level0.npz"}
    assert prepared.evidence["loader_accessed_test"] is False


def test_validation_selection_is_an_ordered_prefix_without_test_access() -> None:
    project = load_project_config(PROJECT_ROOT)
    options = evaluate_controller.DevelopmentEvaluationOptions(
        controller_directory=Path("controllers/pid"),
        run_id="validation-prefix",
        split="validation",
        count=3,
    )

    prepared = evaluate_controller._prepare_tracks(project, options)

    assert prepared.level_id == 1
    assert prepared.available_track_count == 100
    assert prepared.track_pool is not None and prepared.track_pool.size == 3
    assert prepared.batch.seed.tolist() == [1_000_000, 1_000_001, 1_000_002]
    assert prepared.evidence["loaded_splits"] == ["validation"]
    assert prepared.evidence["loader_accessed_train"] is False
    assert prepared.evidence["loader_accessed_test"] is False
    source = (PROJECT_ROOT / "scripts/evaluate_controller.py").read_text(encoding="utf-8")
    assert "load_verified_test_pool" not in source
    assert 'official_track_split_spec("test")' not in source


def test_selected_runner_captures_exactly_one_measured_row() -> None:
    calls: list[tuple[str, int]] = []
    trajectory = object()

    def runner(_env, _directory, reset_seed, **_kwargs):
        calls.append(("normal", reset_seed))
        return _result(reset_seed)

    def recorder(_env, _directory, reset_seed, **_kwargs):
        calls.append(("record", reset_seed))
        return SimpleNamespace(result=_result(reset_seed), trajectory=trajectory)

    selected = evaluate_controller.SelectedTrajectoryRunner(
        1,
        runner=runner,
        recorder=recorder,
    )
    for row in range(3):
        selected(object(), "controllers/example", row, reset_options={"track_index": row})

    assert calls == [("normal", 0), ("record", 1), ("normal", 2)]
    assert selected.trajectory is trajectory

    mismatched = evaluate_controller.SelectedTrajectoryRunner(0, runner=runner, recorder=recorder)
    with pytest.raises(RuntimeError, match="row order"):
        mismatched(object(), "controllers/example", 0, reset_options={"track_index": 1})


def test_run_publishes_informal_outputs_and_refuses_overwrite(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    root = _project(tmp_path)
    options = evaluate_controller.DevelopmentEvaluationOptions(
        controller_directory=Path("controllers/example"),
        run_id="unit-evaluation",
        split="level0",
    )

    result = evaluate_controller.run_development_evaluation(
        options,
        project_root=root,
        evaluator=_fake_evaluator,
    )

    assert result == {
        "status": "completed",
        "evaluation_kind": evaluate_controller.DEVELOPMENT_EVALUATION_KIND,
        "run_id": "unit-evaluation",
        "output": "runs/evaluations/unit-evaluation",
        "success_count": 1,
        "track_count": 1,
        "success_rate": 1.0,
    }
    output = root / result["output"]
    summary = json.loads((output / "summary.json").read_text(encoding="ascii"))
    assert summary["formal_benchmark_result"] is False
    assert summary["comparable_to_accepted_test_result"] is False
    assert summary["track_source"]["loaded_splits"] == ["level0"]
    assert summary["track_source"]["loader_accessed_test"] is False
    assert summary["track_selection"]["track_ids"] == [2**32 - 1]
    assert summary["source"]["worktree_clean"] is True
    assert summary["controller"]["directory"] == "controllers/example"
    assert (output / "episodes.csv").read_text(encoding="utf-8").count("\n") == 2
    assert not list((root / "runs/evaluations").glob(".*.staging"))
    assert '"completed":1' in capsys.readouterr().err

    with pytest.raises(evaluate_controller.DevelopmentEvaluationError, match="already exists"):
        evaluate_controller.run_development_evaluation(
            options,
            project_root=root,
            evaluator=_fake_evaluator,
        )
    assert json.loads((output / "summary.json").read_text(encoding="ascii")) == summary


def test_failed_or_mutating_evaluation_leaves_no_published_directory(tmp_path: Path) -> None:
    root = _project(tmp_path)
    options = evaluate_controller.DevelopmentEvaluationOptions(
        controller_directory=Path("controllers/example"),
        run_id="failed-evaluation",
        split="level0",
    )

    def mutating_evaluator(*args, **kwargs):
        result = _fake_evaluator(*args, **kwargs)
        controller = root / "controllers/example/controller.py"
        controller.write_text("VALUE = 2\n", encoding="utf-8")
        return result

    with pytest.raises(
        evaluate_controller.DevelopmentEvaluationError, match="Controller directory"
    ):
        evaluate_controller.run_development_evaluation(
            options,
            project_root=root,
            evaluator=mutating_evaluator,
        )

    base = root / "runs/evaluations"
    assert not (base / "failed-evaluation").exists()
    assert not (base / ".failed-evaluation.staging").exists()
