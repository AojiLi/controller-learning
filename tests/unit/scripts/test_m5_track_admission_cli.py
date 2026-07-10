"""CPU-only CLI contracts for the formal M5 Track materializer."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

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


def test_cli_parses_paths_but_formal_run_rejects_nonofficial_locations() -> None:
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
    evidence = admission_cli._formal_path_evidence(PROJECT_ROOT, options)
    assert not any(evidence.values())
    with pytest.raises(ValueError, match="official output paths"):
        admission_cli._require_formal_output_paths(PROJECT_ROOT, options)


def test_default_paths_are_the_only_formal_locations() -> None:
    evidence = admission_cli._require_formal_output_paths(
        PROJECT_ROOT,
        admission_cli.AdmissionOptions(),
    )
    assert evidence == {
        "official_asset_directory": True,
        "official_train_cache_directory": True,
        "official_report_path": True,
    }


def test_all_evidence_sources_exist_and_are_repository_relative() -> None:
    assert len(admission_cli.RELEVANT_SOURCE_PATHS) == len(set(admission_cli.RELEVANT_SOURCE_PATHS))
    for relative in admission_cli.RELEVANT_SOURCE_PATHS:
        assert not Path(relative).is_absolute()
        assert (PROJECT_ROOT / relative).is_file(), relative


def test_artifact_readback_rehashes_every_manifest_fixed_asset_and_train_cache(
    tmp_path,
) -> None:
    asset_directory = tmp_path / "assets"
    train_cache = tmp_path / "cache" / "train_pool.npz"
    asset_directory.mkdir()
    train_cache.parent.mkdir()
    manifests = {}
    materialized = {}
    fixed_batches = {}
    for spec in admission_cli.OFFICIAL_TRACK_SPLITS:
        manifest_path = asset_directory / spec.manifest_file
        asset_path = train_cache if spec.split == "train" else asset_directory / spec.asset_file
        manifest_path.write_bytes(f"manifest-{spec.split}".encode())
        asset_path.write_bytes(f"asset-{spec.split}".encode())
        manifest_digest = admission_cli.sha256_file(manifest_path)
        asset_digest = admission_cli.sha256_file(asset_path)
        manifests[spec.split] = SimpleNamespace(asset_sha256=asset_digest)
        materialized[spec.split] = {
            "manifest_sha256": manifest_digest,
            "asset_sha256": asset_digest,
        }
        if spec.package_asset:
            fixed_batches[spec.split] = object()
    verification = SimpleNamespace(
        manifests=manifests,
        fixed_batches=fixed_batches,
        train_cache_verified=True,
    )

    evidence = admission_cli._artifact_readback_evidence(
        verification=verification,
        asset_directory=asset_directory,
        train_cache_path=train_cache,
        materialized=materialized,
    )
    assert evidence["passed"] is True
    assert set(evidence["asset_files_sha256"]) == {
        "level0",
        "train",
        "validation",
        "test",
    }

    train_cache.write_bytes(b"corrupt")
    with pytest.raises(RuntimeError, match="train asset changed"):
        admission_cli._artifact_readback_evidence(
            verification=verification,
            asset_directory=asset_directory,
            train_cache_path=train_cache,
            materialized=materialized,
        )


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

    admission_cli.main([])

    assert calls == [(report, tmp_path / "benchmarks/v0.1/m5_track_admission_report.json")]


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
