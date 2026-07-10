"""CPU-only CLI contracts for the formal M5 Track materializer."""

from __future__ import annotations

from pathlib import Path

import pytest

from scripts import build_track_assets as admission_cli

PROJECT_ROOT = Path(__file__).parents[3]


def test_cli_defaults_lock_asset_cache_and_report_locations() -> None:
    options = admission_cli._parse_args([])

    assert options == admission_cli.AdmissionOptions()
    assert options.asset_directory == Path("controller_learning/assets/tracks/v0.1")
    assert options.train_cache_directory == Path(".track-cache/v0.1")
    assert options.output == Path("benchmarks/v0.1/m5_track_admission_report.json")
    assert admission_cli.FORMAL_ADMISSION_WORLDS == 1024
    assert admission_cli.FORMAL_CONTROL_BLOCK_STEPS == 100


def test_cli_accepts_explicit_filesystem_outputs_only() -> None:
    options = admission_cli._parse_args(
        [
            "--asset-directory",
            "repo-assets",
            "--train-cache-directory",
            "local-cache",
            "--output",
            "report.json",
        ]
    )
    assert options == admission_cli.AdmissionOptions(
        Path("repo-assets"),
        Path("local-cache"),
        Path("report.json"),
    )


def test_all_evidence_sources_exist_and_are_repository_relative() -> None:
    assert len(admission_cli.RELEVANT_SOURCE_PATHS) == len(set(admission_cli.RELEVANT_SOURCE_PATHS))
    for relative in admission_cli.RELEVANT_SOURCE_PATHS:
        assert not Path(relative).is_absolute()
        assert (PROJECT_ROOT / relative).is_file(), relative


def test_main_writes_passing_fake_report_without_initializing_gpu(monkeypatch, tmp_path) -> None:
    report = {
        "status": "pass",
        "splits": {
            "train": {"selected_count": 10_000},
            "validation": {"selected_count": 100},
            "test": {"selected_count": 20},
        },
    }
    calls = []
    monkeypatch.setattr(admission_cli, "PROJECT_ROOT", tmp_path)
    monkeypatch.setattr(
        admission_cli,
        "run_formal_admission",
        lambda project_root, options: report,
    )
    monkeypatch.setattr(
        admission_cli,
        "write_strict_json",
        lambda value, path: calls.append((value, path)),
    )

    admission_cli.main(["--output", "evidence.json"])

    assert calls == [(report, tmp_path / "evidence.json")]


def test_main_records_failure_before_reraising(monkeypatch, tmp_path) -> None:
    calls = []
    monkeypatch.setattr(admission_cli, "PROJECT_ROOT", tmp_path)

    def fail(project_root, options):
        raise RuntimeError("fake GPU failure")

    monkeypatch.setattr(admission_cli, "run_formal_admission", fail)
    monkeypatch.setattr(
        admission_cli,
        "write_strict_json",
        lambda value, path: calls.append((value, path)),
    )

    with pytest.raises(RuntimeError, match="fake GPU failure"):
        admission_cli.main(["--output", "failure.json"])

    assert calls[0][0]["status"] == "fail"
    assert calls[0][0]["failure"] == {
        "type": "RuntimeError",
        "message": "fake GPU failure",
    }
    assert calls[0][1] == tmp_path / "failure.json"
