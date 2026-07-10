"""CPU-only protocol, asset-loading, privacy, and CLI tests for the M5 GPU benchmark."""

from __future__ import annotations

import copy
from dataclasses import replace
from pathlib import Path

import pytest

from controller_learning.config import load_project_config
from controller_learning.tracks.assets import TrackAssetManifest, TrackAssetRecord
from controller_learning.tracks.generator import generate_track_candidate, pack_track
from controller_learning.tracks.hashing import track_geometry_sha256
from controller_learning.tracks.specs import (
    generation_spec_from_project,
    track_capacity_from_project,
)
from controller_learning.tracks.types import stack_tracks
from scripts import benchmark_track_pool as benchmark

PROJECT_ROOT = Path(__file__).parents[3]


def _passing_report() -> dict:
    digest = "a" * 64
    source = {
        "git_revision": "1" * 40,
        "relevant_source_clean": True,
        "tracked_worktree_clean": True,
        "source_files_sha256": {"relative/source.py": "b" * 64},
    }
    transitions = benchmark.FORMAL_NUM_WORLDS * benchmark.DEFAULT_ENVIRONMENT_STEPS
    return {
        "schema_version": benchmark.REPORT_SCHEMA_VERSION,
        "protocol_version": benchmark.PROTOCOL_VERSION,
        "protocol": {
            "backend": "mjx_warp",
            "level_id": benchmark.FORMAL_LEVEL_ID,
            "num_worlds": benchmark.FORMAL_NUM_WORLDS,
            "environment_steps": benchmark.DEFAULT_ENVIRONMENT_STEPS,
            "transitions": transitions,
            "warmup_steps": benchmark.DEFAULT_WARMUP_STEPS,
            "reset_seed": benchmark.FORMAL_RESET_SEED,
            "reset_heavy_cycles": benchmark.DEFAULT_RESET_HEAVY_CYCLES,
            "action_device_platform": "gpu",
            "per_step_host_synchronization": False,
        },
        "assets": {
            "formal_manifest_location": True,
            "formal_cache_location": True,
            "track_count": benchmark.FORMAL_TRAIN_TRACK_COUNT,
            "configured_track_count": benchmark.FORMAL_TRAIN_TRACK_COUNT,
            "cache_matches_manifest_sha256": True,
            "unique_seed_count": benchmark.FORMAL_TRAIN_TRACK_COUNT,
            "geometry_hash_count": benchmark.FORMAL_TRAIN_TRACK_COUNT,
            "geometry_admission_pass_count": benchmark.FORMAL_TRAIN_TRACK_COUNT,
            "driveability_admission_pass_count": benchmark.FORMAL_TRAIN_TRACK_COUNT,
            "allowed_seed_uint32_sha256": digest,
        },
        "pool_residency": {
            "track_count": benchmark.FORMAL_TRAIN_TRACK_COUNT,
            "leaf_count": 17,
            "all_leaves_on_gpu": True,
            "byte_count_matches_host": True,
            "device_seed_uint32_sha256": digest,
        },
        "deterministic_reset": {"passed": True},
        "transfer_guard": {
            "active_step": {"passed": True},
            "mixed_next_step_autoreset": {"passed": True},
        },
        "timing": {
            "environment_create_seconds": 1.0,
            "pool_upload_ready_seconds": 1.2,
            "reset_compile_seconds": 0.5,
            "first_step_compile_seconds": 2.0,
            "warmup_seconds": 1.0,
            "steady_seconds": 60.0,
            "environment_steps_per_second": 166.6,
            "transitions_per_second": 170_000.0,
            "pool_to_fixed_throughput_ratio": 0.8,
        },
        "fixed_track_baseline": {
            "steps": benchmark.DEFAULT_ENVIRONMENT_STEPS,
            "transitions": transitions,
            "steady_seconds": 50.0,
            "transitions_per_second": 212_500.0,
            "final_output_finite": True,
            "matches_pool_initial_selection": True,
            "per_step_host_synchronization": False,
        },
        "reset_heavy": {
            "seconds": 2.0,
            "reset_events_per_second": 10_000.0,
            "passed": True,
        },
        "health": {
            "bound_sufficient": True,
            "maximum_steps": 5_000,
            "required_steps_for_all_timeouts_and_next_step_reset": 4_001,
            "all_worlds_observed_timeout": True,
            "timeout_event_count": benchmark.FORMAL_NUM_WORLDS,
            "all_worlds_observed_autoreset": True,
            "autoreset_world_count": benchmark.FORMAL_NUM_WORLDS,
            "unexpected_termination_event_count": 0,
            "numerical_failure_event_count": 0,
            "final_output_finite": True,
            "final_nonfinite_fields": [],
            "disallowed_track_id_event_count": 0,
        },
        "executable_cache": {"passed": True},
        "runtime": {"jax_device": {"platform": "gpu", "device_kind": "NVIDIA Test GPU"}},
        "memory": {
            "peak_sampled_process_vram_mib": 1_000.0,
            "steady_process_vram_growth_mib": 2.0,
            "steady_growth_within_limit": True,
        },
        "source_evidence": {"before": copy.deepcopy(source), "after": copy.deepcopy(source)},
        "final_output": {
            "finite": True,
            "nonfinite_fields": [],
            "all_track_ids_allowed": True,
        },
    }


def _gate(report: dict, identifier: str) -> dict:
    return next(
        item for item in benchmark.evaluate_report_gates(report) if item["id"] == identifier
    )


def test_cli_defaults_lock_formal_manifest_cache_and_protocol() -> None:
    options = benchmark._parse_args([])

    assert options == benchmark.BenchmarkOptions()
    assert options.manifest == Path("controller_learning/assets/tracks/v0.1/train.json")
    assert options.cache == Path(".track-cache/v0.1/train_pool.npz")
    assert options.environment_steps == 10_000
    assert options.health_max_steps == 5_000
    assert options.reset_heavy_cycles == 64


@pytest.mark.parametrize(
    "arguments",
    (
        ("--steps", "0"),
        ("--warmup-steps", "-1"),
        ("--health-max-steps", "bad"),
        ("--reset-heavy-cycles", "0"),
    ),
)
def test_cli_rejects_nonpositive_integer_protocol_values(arguments: tuple[str, str]) -> None:
    with pytest.raises(SystemExit) as caught:
        benchmark._parse_args(list(arguments))
    assert caught.value.code == 2


def test_complete_fake_report_passes_every_formal_gate() -> None:
    checks = benchmark.evaluate_report_gates(_passing_report())

    assert len(checks) >= 35
    assert len({check["id"] for check in checks}) == len(checks)
    assert all(check["passed"] for check in checks)


@pytest.mark.parametrize(
    ("gate_id", "mutate"),
    (
        (
            "protocol.environment_steps",
            lambda report: report["protocol"].update(environment_steps=9),
        ),
        (
            "assets.cache_integrity",
            lambda report: report["assets"].update(cache_matches_manifest_sha256=False),
        ),
        (
            "assets.driveability_admission",
            lambda report: report["assets"].update(driveability_admission_pass_count=9_999),
        ),
        ("pool.resident", lambda report: report["pool_residency"].update(all_leaves_on_gpu=False)),
        ("reset.deterministic", lambda report: report["deterministic_reset"].update(passed=False)),
        (
            "transfer.active",
            lambda report: report["transfer_guard"]["active_step"].update(passed=False),
        ),
        (
            "transfer.mixed_reset",
            lambda report: report["transfer_guard"]["mixed_next_step_autoreset"].update(
                passed=False
            ),
        ),
        (
            "timing.pool_ratio",
            lambda report: report["timing"].update(pool_to_fixed_throughput_ratio=0.749),
        ),
        ("reset_heavy.protocol", lambda report: report["reset_heavy"].update(passed=False)),
        (
            "health.autoreset",
            lambda report: report["health"].update(all_worlds_observed_autoreset=False),
        ),
        (
            "health.unexpected_termination",
            lambda report: report["health"].update(unexpected_termination_event_count=1),
        ),
        (
            "health.numerical",
            lambda report: report["health"].update(numerical_failure_event_count=1),
        ),
        (
            "health.allowed_track_ids",
            lambda report: report["health"].update(disallowed_track_id_event_count=1),
        ),
        ("cache.no_recompile", lambda report: report["executable_cache"].update(passed=False)),
        (
            "source.clean",
            lambda report: report["source_evidence"]["after"].update(tracked_worktree_clean=False),
        ),
        ("final_output.finite", lambda report: report["final_output"].update(finite=False)),
    ),
)
def test_report_validator_rejects_each_critical_failure(gate_id, mutate) -> None:
    report = _passing_report()
    mutate(report)

    assert _gate(report, gate_id)["passed"] is False


def test_privacy_gate_rejects_absolute_paths_and_uuids_but_allows_repo_paths() -> None:
    report = _passing_report()
    report["relative_manifest"] = "controller_learning/assets/tracks/v0.1/train.json"
    assert _gate(report, "privacy.redacted")["passed"] is True

    report["leak"] = "/home/user/controller_learning/.track-cache/v0.1/train_pool.npz"
    assert _gate(report, "privacy.redacted")["passed"] is False
    report.pop("leak")
    report["leak"] = "GPU-12345678-1234-1234-1234-123456789abc"
    assert _gate(report, "privacy.redacted")["passed"] is False


def test_report_writer_fails_closed_before_persisting_private_values(
    monkeypatch,
    tmp_path: Path,
) -> None:
    writes = []
    monkeypatch.setattr(
        benchmark.m4_benchmark,
        "write_strict_json",
        lambda path, payload: writes.append((path, payload)),
    )

    with pytest.raises(ValueError, match="refusing to persist"):
        benchmark.write_strict_json(tmp_path / "report.json", {"leak": "/home/user/cache"})
    assert writes == []

    benchmark.write_strict_json(
        tmp_path / "report.json",
        {"manifest": "controller_learning/assets/tracks/v0.1/train.json"},
    )
    assert len(writes) == 1


def test_source_snapshot_hashes_relative_files_and_requires_clean_tracked_state(
    monkeypatch,
    tmp_path: Path,
) -> None:
    source = tmp_path / "source.py"
    source.write_text("value = 1\n", encoding="utf-8")
    monkeypatch.setattr(benchmark, "RELEVANT_SOURCE_PATHS", ("source.py",))

    def clean_git(root, *arguments):
        assert root == tmp_path
        return "f" * 40 if arguments[:2] == ("rev-parse", "HEAD") else ""

    monkeypatch.setattr(benchmark.m4_benchmark, "_git", clean_git)
    snapshot = benchmark._source_snapshot(tmp_path)

    assert snapshot["git_revision"] == "f" * 40
    assert snapshot["relevant_source_clean"] is True
    assert snapshot["tracked_worktree_clean"] is True
    assert tuple(snapshot["source_files_sha256"]) == ("source.py",)


def test_verified_cache_loader_binds_manifest_records_to_track_pool(
    monkeypatch,
) -> None:
    project = load_project_config(PROJECT_ROOT)
    project = replace(
        project,
        benchmark=replace(project.benchmark, train_track_count=1),
    )
    track = pack_track(
        generate_track_candidate(42, generation_spec_from_project(project)),
        track_capacity_from_project(project),
    )
    batch = stack_tracks((track,))
    geometry_sha256 = track_geometry_sha256(track)
    manifest = TrackAssetManifest(
        schema_version=1,
        benchmark_version=project.benchmark.version,
        level_id=1,
        split="train",
        generator_version=project.track.generator.generator_version,
        geometry_validation_version="geometry-v1",
        driveability_protocol_version="driveability-v1",
        track_width_m=track.width_m,
        track_count=1,
        capacity=track.capacity,
        asset_file="train_pool.npz",
        asset_sha256="c" * 64,
        tracks=(
            TrackAssetRecord(
                seed=track.seed,
                geometry_sha256=geometry_sha256,
                geometry_validation="passed",
                driveability_validation="passed",
            ),
        ),
    )
    calls = []
    monkeypatch.setattr(benchmark, "load_track_asset_manifest", lambda path: manifest)

    def fake_load(path, **kwargs):
        calls.append((path, kwargs))
        return batch

    monkeypatch.setattr(benchmark, "load_track_batch_npz", fake_load)
    monkeypatch.setattr(benchmark, "sha256_file", lambda path: "c" * 64)
    loaded_manifest, pool, evidence = benchmark._load_verified_train_pool(
        project,
        Path("train.json"),
        Path("cache/train_pool.npz"),
    )

    assert loaded_manifest == manifest
    assert pool.size == 1
    assert pool.split == "train"
    assert calls == [
        (
            Path("cache/train_pool.npz"),
            {
                "expected_sha256": "c" * 64,
                "expected_track_count": 1,
                "expected_capacity": track.capacity,
            },
        )
    ]
    assert evidence["cache_matches_manifest_sha256"] is True
    assert evidence["geometry_admission_pass_count"] == 1
    assert evidence["driveability_admission_pass_count"] == 1


def test_main_writes_report_and_fails_only_for_failed_gates(monkeypatch, capsys) -> None:
    written = []
    passing = {
        "status": "pass",
        "checks": [{"id": "ok", "passed": True}],
        "protocol": {"num_worlds": 1024, "environment_steps": 10_000},
        "assets": {"track_count": 10_000},
        "timing": {"transitions_per_second": 1.0, "pool_to_fixed_throughput_ratio": 0.8},
    }
    monkeypatch.setattr(benchmark, "run_benchmark", lambda options: passing)
    monkeypatch.setattr(
        benchmark,
        "write_strict_json",
        lambda path, report: written.append((path, report)),
    )

    benchmark.main([])
    assert written == [(benchmark.DEFAULT_OUTPUT, passing)]
    assert "M5 TrackPool status: pass" in capsys.readouterr().out

    failing = copy.deepcopy(passing)
    failing["status"] = "fail"
    failing["checks"] = [{"id": "health.autoreset", "passed": False}]
    monkeypatch.setattr(benchmark, "run_benchmark", lambda options: failing)
    with pytest.raises(SystemExit) as caught:
        benchmark.main([])
    assert caught.value.code == 1
    assert "health.autoreset" in capsys.readouterr().err
