"""CPU-only protocol tests for the formal M6 PID/MPC benchmark CLI."""

from __future__ import annotations

import copy
import json
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from controller_learning.config import load_project_config
from controller_learning.evaluation import (
    ControllerEvaluation,
    EpisodeEvaluation,
    summarize_compute_times,
)
from scripts import benchmark_m6_controllers as benchmark

PROJECT_ROOT = Path(__file__).parents[3]


def _fake_runtime() -> dict[str, Any]:
    return {
        "python": "3.11.0",
        "platform": "Linux-test",
        "packages": {"jax": "test"},
        "jax_device": {"id": 0, "platform": "gpu", "device_kind": "test GPU"},
        "jax_gpu_error": None,
        "nvidia_inventory": [
            {
                "index": 0,
                "name": "test GPU",
                "driver_version": "1.0",
                "memory_total_mib": 16384.0,
            }
        ],
        "nvidia_inventory_error": None,
    }


def _fake_snapshot(_root: Path) -> dict[str, Any]:
    return {
        "git_revision": "0123456789abcdef",
        "relevant_source_clean": True,
        "source_files_sha256": {relative: "a" * 64 for relative in benchmark.RELEVANT_SOURCE_PATHS},
    }


def _evaluation(
    batch,
    directory: Path,
    level_id: int,
    reset_seeds: np.ndarray,
) -> ControllerEvaluation:
    track_count = int(batch.seed.shape[0])
    name = directory.name
    successes = track_count
    if name == "pid" and level_id == 1:
        successes = 0
    elif name == "mpc" and level_id == 1:
        successes = 80

    episodes = []
    for index in range(track_count):
        success = index < successes
        samples = (0.01,)
        episodes.append(
            EpisodeEvaluation(
                track_index=index,
                track_id=int(batch.seed[index]),
                reset_seed=int(reset_seeds[index]),
                success=success,
                lap_time_s=10.0 + index if success else None,
                steps=1,
                total_reward=1.0 if success else -1.0,
                terminated=success,
                truncated=not success,
                termination_reason=1 if success else 4,
                controller_import_time_s=0.001,
                controller_init_time_s=0.002,
                compute_times_s=samples,
                compute_timing=summarize_compute_times(samples),
            )
        )
    successful_laps = [episode.lap_time_s for episode in episodes if episode.success]
    return ControllerEvaluation(
        controller_directory=str(directory),
        level_id=level_id,
        backend="mjx_warp",
        episodes=tuple(episodes),
        track_count=track_count,
        success_count=successes,
        success_rate=successes / track_count,
        mean_successful_lap_time_s=(
            float(np.mean(successful_laps, dtype=np.float64)) if successful_laps else None
        ),
        compute_timing=summarize_compute_times(tuple(0.01 for _ in episodes)),
    )


@pytest.fixture(scope="module")
def official_assets():
    config = load_project_config(PROJECT_ROOT)
    return benchmark._load_evaluation_assets(config, PROJECT_ROOT)


@pytest.fixture()
def passing_report(official_assets):
    calls: list[dict[str, Any]] = []

    def evaluator(config, level_id, batch, generator_version, directory, backend, **kwargs):
        del config, generator_version
        calls.append(
            {
                "level_id": level_id,
                "track_count": int(batch.seed.shape[0]),
                "directory": Path(directory).name,
                "backend": backend,
                "reset_seeds": tuple(int(value) for value in kwargs["reset_seeds"]),
            }
        )
        return _evaluation(batch, Path(directory), level_id, kwargs["reset_seeds"])

    report = benchmark.run_benchmark(
        benchmark.BenchmarkOptions(),
        project_root=PROJECT_ROOT,
        asset_loader=lambda _config, _root: official_assets,
        evaluator=evaluator,
        snapshot_loader=_fake_snapshot,
        runtime_loader=_fake_runtime,
    )
    return report, calls


def test_cli_exposes_only_the_report_path() -> None:
    assert benchmark._parse_args([]) == benchmark.BenchmarkOptions()
    assert benchmark._parse_args(["--output", "results/m6.json"]).output == Path("results/m6.json")

    with pytest.raises(SystemExit) as backend_error:
        benchmark._parse_args(["--backend", "cpu_reference"])
    assert backend_error.value.code == 2

    with pytest.raises(SystemExit) as suffix_error:
        benchmark._parse_args(["--output", "results/m6.txt"])
    assert suffix_error.value.code == 2


def test_official_loader_reads_only_level0_and_validation(monkeypatch) -> None:
    config = load_project_config(PROJECT_ROOT)
    original = benchmark.load_manifest_track_batch
    accessed: list[str] = []

    def tracked(path):
        accessed.append(Path(path).name)
        return original(path)

    monkeypatch.setattr(benchmark, "load_manifest_track_batch", tracked)
    assets = benchmark._load_evaluation_assets(config, PROJECT_ROOT)

    assert accessed == ["level0.json", "validation.json"]
    assert assets.evidence["loaded_splits"] == ["level0", "validation"]
    assert assets.evidence["test_split_accessed"] is False
    assert "test" not in assets.evidence
    assert not any("test.json" in path for path in benchmark.RELEVANT_SOURCE_PATHS)
    assert not any("test.npz" in path for path in benchmark.RELEVANT_SOURCE_PATHS)


def test_formal_workload_and_row_index_seeds_are_fixed(passing_report) -> None:
    report, calls = passing_report

    assert calls == [
        {
            "level_id": 0,
            "track_count": 1,
            "directory": "pid",
            "backend": "mjx_warp",
            "reset_seeds": (0,),
        },
        {
            "level_id": 1,
            "track_count": 10,
            "directory": "pid",
            "backend": "mjx_warp",
            "reset_seeds": tuple(range(10)),
        },
        {
            "level_id": 0,
            "track_count": 1,
            "directory": "mpc",
            "backend": "mjx_warp",
            "reset_seeds": (0,),
        },
        {
            "level_id": 1,
            "track_count": 100,
            "directory": "mpc",
            "backend": "mjx_warp",
            "reset_seeds": tuple(range(100)),
        },
    ]
    assert report["status"] == "pass"
    assert all(check["passed"] for check in report["checks"])
    assert report["evaluations"]["pid"]["validation"]["success_rate"] == 0.0
    assert report["evaluations"]["mpc"]["validation"]["success_rate"] == 0.8


def test_report_gates_recompute_and_realtime_is_diagnostic(passing_report) -> None:
    report, _calls = passing_report

    assert benchmark.evaluate_report_gates(report) == report["checks"]
    report["evaluations"]["mpc"]["realtime_qualification"]["eligible"] = False
    report["evaluations"]["mpc"]["combined_timing"]["p99_s"] = 0.2

    required_checks = benchmark.evaluate_report_gates(report)

    assert all(check["passed"] for check in required_checks)


def test_mpc_validation_below_eighty_percent_fails_exact_threshold_gate(
    passing_report,
) -> None:
    report, _calls = passing_report
    failing = copy.deepcopy(report)
    validation = failing["evaluations"]["mpc"]["validation"]
    validation["success_count"] = 79
    validation["success_rate"] = 0.79

    failed = {
        check["id"] for check in benchmark.evaluate_report_gates(failing) if not check["passed"]
    }

    assert failed == {"controllers.mpc.validation_success_rate"}


def test_gate_rejects_invalid_action_and_incomplete_timing(passing_report) -> None:
    report, _calls = passing_report
    failing = copy.deepcopy(report)
    episode = failing["evaluations"]["pid"]["validation"]["episodes"][0]
    episode["termination_reason"] = 3
    episode["compute_times_s"] = []

    failed = {
        check["id"] for check in benchmark.evaluate_report_gates(failing) if not check["passed"]
    }

    assert failed == {"controllers.no_invalid_action", "controllers.timing_complete"}


def test_strict_json_is_atomic_and_report_is_private(
    passing_report,
    tmp_path: Path,
) -> None:
    report, _calls = passing_report
    output = tmp_path / "report.json"

    benchmark.write_strict_json(output, report)
    assert json.loads(output.read_text(encoding="utf-8"))["status"] == "pass"
    assert benchmark._privacy_findings(report) == {"absolute_paths": [], "gpu_uuids": []}

    bad_output = tmp_path / "bad.json"
    with pytest.raises(ValueError, match="Out of range float values"):
        benchmark.write_strict_json(bad_output, {"bad": float("nan")})
    assert not bad_output.exists()


def test_privacy_gate_rejects_absolute_paths_and_gpu_uuids(passing_report) -> None:
    report, _calls = passing_report
    failing = copy.deepcopy(report)
    failing["runtime"]["leak"] = [
        "/home/user/controller_learning",
        "GPU-12345678-1234-1234-1234-123456789abc",
    ]

    failed = {
        check["id"] for check in benchmark.evaluate_report_gates(failing) if not check["passed"]
    }

    assert failed == {"report.privacy"}


def test_main_writes_the_report_and_exits_nonzero_after_a_failed_gate(
    passing_report,
    monkeypatch,
    tmp_path: Path,
) -> None:
    report, _calls = passing_report
    passing_output = tmp_path / "passing.json"
    monkeypatch.setattr(benchmark, "run_benchmark", lambda _options: report)

    benchmark.main(["--output", str(passing_output)])

    assert json.loads(passing_output.read_text(encoding="utf-8"))["status"] == "pass"

    failing = copy.deepcopy(report)
    failing["status"] = "fail"
    failing["checks"][0]["passed"] = False
    failing_output = tmp_path / "failing.json"
    monkeypatch.setattr(benchmark, "run_benchmark", lambda _options: failing)

    with pytest.raises(SystemExit, match="failed one or more gates"):
        benchmark.main(["--output", str(failing_output)])
    assert json.loads(failing_output.read_text(encoding="utf-8"))["status"] == "fail"


def test_relevant_sources_are_unique_relative_and_present() -> None:
    assert len(benchmark.RELEVANT_SOURCE_PATHS) == len(set(benchmark.RELEVANT_SOURCE_PATHS))
    for relative in benchmark.RELEVANT_SOURCE_PATHS:
        assert not Path(relative).is_absolute()
        assert (PROJECT_ROOT / relative).is_file(), relative
