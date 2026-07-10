"""CPU-only contract tests for the formal M4 environment benchmark."""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from controller_learning.tracks import TrackGenerationError
from scripts import benchmark_racing_env as benchmark

PROJECT_ROOT = Path(__file__).parents[3]


def _passing_report() -> dict[str, Any]:
    hashes = {"source.py": "a" * 64}
    report: dict[str, Any] = {
        "schema_version": benchmark.REPORT_SCHEMA_VERSION,
        "protocol_version": benchmark.PROTOCOL_VERSION,
        "protocol": {
            "backend": "mjx_warp",
            "level_id": benchmark.FORMAL_LEVEL_ID,
            "num_worlds": benchmark.FORMAL_NUM_WORLDS,
            "reset_seed": benchmark.FORMAL_RESET_SEED,
            "environment_steps": benchmark.DEFAULT_ENVIRONMENT_STEPS,
            "warmup_steps": benchmark.DEFAULT_WARMUP_STEPS,
            "transitions": benchmark.FORMAL_NUM_WORLDS * benchmark.DEFAULT_ENVIRONMENT_STEPS,
            "action_device_platform": "gpu",
            "per_step_host_synchronization": False,
        },
        "track_scan": {
            "accepted_count": benchmark.FORMAL_NUM_WORLDS,
            "rejected_count": 7,
            "attempted_count": benchmark.FORMAL_NUM_WORLDS + 7,
            "all_track_ids_distinct": True,
            "all_seeds_distinct": True,
        },
        "timing": {
            "reset_compile_seconds": 1.0,
            "first_step_compile_seconds": 2.0,
            "warmup_seconds": 0.5,
            "steady_seconds": 4.0,
            "environment_steps_per_second": 2.5,
            "transitions_per_second": 2560.0,
        },
        "transfer_guard": {
            "active_step": {"passed": True},
            "mixed_next_step_autoreset": {"passed": True},
        },
        "health": {
            "bound_sufficient": True,
            "maximum_steps": 4001,
            "required_steps_for_all_timeouts_and_next_step_reset": 4001,
            "all_worlds_observed_timeout": True,
            "timeout_event_count": benchmark.FORMAL_NUM_WORLDS,
            "all_worlds_observed_autoreset": True,
            "autoreset_world_count": benchmark.FORMAL_NUM_WORLDS,
            "unexpected_termination_event_count": 0,
            "numerical_failure_event_count": 0,
            "final_output_finite": True,
            "final_nonfinite_fields": [],
        },
        "runtime": {"jax_device": {"platform": "gpu"}},
        "memory": {
            "peak_sampled_process_vram_mib": 4096.0,
            "steady_process_vram_growth_mib": 0.0,
            "steady_growth_within_limit": True,
        },
        "source_evidence": {
            "before": {
                "git_revision": "0123456789abcdef",
                "relevant_source_clean": True,
                "source_files_sha256": hashes,
            },
            "after": {
                "git_revision": "0123456789abcdef",
                "relevant_source_clean": True,
                "source_files_sha256": hashes,
            },
        },
    }
    return report


def test_cli_defaults_lock_the_formal_protocol() -> None:
    options = benchmark._parse_args([])

    assert options == benchmark.BenchmarkOptions()
    assert benchmark.FORMAL_NUM_WORLDS == 1024
    assert benchmark.FORMAL_LEVEL_ID == 1
    assert options.environment_steps == 10_000
    assert options.output == Path("benchmarks/v0.1/m4_environment_report.json")


@pytest.mark.parametrize(
    "arguments",
    (
        ["--steps", "0"],
        ["--warmup-steps", "-1"],
        ["--health-max-steps", "nope"],
        ["--max-track-candidates", "1023"],
    ),
)
def test_cli_rejects_non_formal_bounds(arguments: list[str]) -> None:
    with pytest.raises(SystemExit) as caught:
        benchmark._parse_args(arguments)
    assert caught.value.code == 2


def test_track_scan_records_every_accepted_and_rejected_seed(monkeypatch) -> None:
    monkeypatch.setattr(
        benchmark,
        "generation_spec_from_project",
        lambda config: SimpleNamespace(generator_version="v0.1"),
    )
    monkeypatch.setattr(benchmark, "validation_spec_from_project", lambda config: object())
    monkeypatch.setattr(benchmark, "track_capacity_from_project", lambda config: object())

    def generate(seed, spec):
        if seed == 0:
            raise TrackGenerationError("length_out_of_range", "test rejection")
        return SimpleNamespace(seed=seed)

    def validate(candidate, spec):
        if candidate.seed == 1:
            return SimpleNamespace(
                valid=False,
                primary_reason="curvature_exceeded",
                reasons=("curvature_exceeded",),
            )
        return SimpleNamespace(valid=True, primary_reason=None, reasons=())

    monkeypatch.setattr(benchmark, "generate_track_candidate", generate)
    monkeypatch.setattr(benchmark, "validate_track_candidate", validate)
    monkeypatch.setattr(
        benchmark,
        "pack_track",
        lambda candidate, capacity: SimpleNamespace(
            seed=candidate.seed,
            generator_version="v0.1",
        ),
    )

    tracks, evidence = benchmark._generate_valid_tracks(object(), count=2, max_candidates=5)

    assert [track.seed for track in tracks] == [2, 3]
    assert evidence["attempted_count"] == 4
    assert evidence["accepted_seeds"] == [2, 3]
    assert evidence["accepted_track_ids"] == ["v0.1:2", "v0.1:3"]
    assert evidence["rejected_candidates"] == [
        {"seed": 0, "stage": "generation", "reason": "length_out_of_range"},
        {
            "seed": 1,
            "stage": "validation",
            "reason": "curvature_exceeded",
            "reasons": ["curvature_exceeded"],
        },
    ]
    assert evidence["accepted_count"] + evidence["rejected_count"] == 4


def test_all_relevant_source_paths_exist_and_are_repository_relative() -> None:
    assert len(benchmark.RELEVANT_SOURCE_PATHS) == len(set(benchmark.RELEVANT_SOURCE_PATHS))
    for relative in benchmark.RELEVANT_SOURCE_PATHS:
        assert not Path(relative).is_absolute()
        assert (PROJECT_ROOT / relative).is_file(), relative


def test_report_gates_accept_complete_evidence_and_reject_numerical_failure() -> None:
    report = _passing_report()
    passing_checks = benchmark.evaluate_report_gates(report)
    assert passing_checks
    assert all(check["passed"] for check in passing_checks)

    report["health"]["numerical_failure_event_count"] = 1
    failing_checks = benchmark.evaluate_report_gates(report)
    failed_ids = {check["id"] for check in failing_checks if not check["passed"]}
    assert failed_ids == {"health.numerical"}


def test_report_gates_require_1024_distinct_tracks_and_mixed_autoreset() -> None:
    report = _passing_report()
    report["track_scan"]["all_track_ids_distinct"] = False
    report["transfer_guard"]["mixed_next_step_autoreset"]["passed"] = False

    failed = {
        check["id"] for check in benchmark.evaluate_report_gates(report) if not check["passed"]
    }

    assert failed == {"tracks.distinct", "transfer_guard.mixed_autoreset"}


def test_diagnostic_step_counts_cannot_pass_the_formal_protocol() -> None:
    report = _passing_report()
    report["protocol"]["environment_steps"] = 1
    report["protocol"]["transitions"] = benchmark.FORMAL_NUM_WORLDS
    report["protocol"]["warmup_steps"] = benchmark.DEFAULT_WARMUP_STEPS - 1

    failed = {
        check["id"] for check in benchmark.evaluate_report_gates(report) if not check["passed"]
    }

    assert failed == {"protocol.environment_steps", "protocol.warmup_steps"}


@pytest.mark.parametrize(
    ("memory", "expected_failed"),
    (
        (
            {
                "peak_sampled_process_vram_mib": None,
                "steady_process_vram_growth_mib": None,
                "steady_growth_within_limit": None,
            },
            True,
        ),
        (
            {
                "peak_sampled_process_vram_mib": 4096.0,
                "steady_process_vram_growth_mib": 128.0,
                "steady_growth_within_limit": False,
            },
            True,
        ),
    ),
)
def test_formal_report_requires_measurable_stable_process_vram(
    memory: dict[str, Any],
    expected_failed: bool,
) -> None:
    report = _passing_report()
    report["memory"] = memory

    failed = {
        check["id"] for check in benchmark.evaluate_report_gates(report) if not check["passed"]
    }

    assert ("memory.steady_growth" in failed) is expected_failed


def test_strict_json_writer_rejects_nonfinite_values(tmp_path: Path) -> None:
    output = tmp_path / "report.json"

    with pytest.raises(ValueError, match="Out of range float values"):
        benchmark.write_strict_json(output, {"bad": float("nan")})

    assert not output.exists()


def test_main_writes_mocked_report_and_exits_nonzero_on_failed_gate(
    monkeypatch,
    tmp_path: Path,
    capsys,
) -> None:
    output = tmp_path / "report.json"
    passing = _passing_report()
    passing["status"] = "pass"
    passing["checks"] = benchmark.evaluate_report_gates(passing)
    monkeypatch.setattr(benchmark, "run_benchmark", lambda options: passing)

    benchmark.main(["--output", str(output), "--steps", "10"])

    assert json.loads(output.read_text(encoding="utf-8"))["status"] == "pass"
    assert "M4 environment status: pass" in capsys.readouterr().out

    failing = _passing_report()
    failing["health"]["numerical_failure_event_count"] = 1
    failing["checks"] = benchmark.evaluate_report_gates(failing)
    failing["status"] = "fail"
    monkeypatch.setattr(benchmark, "run_benchmark", lambda options: failing)
    with pytest.raises(SystemExit) as caught:
        benchmark.main(["--output", str(output), "--steps", "10"])
    assert caught.value.code == 1
