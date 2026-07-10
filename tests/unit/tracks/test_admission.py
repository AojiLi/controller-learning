"""CPU contracts for deterministic formal Track admission."""

from __future__ import annotations

from dataclasses import replace

import pytest

from controller_learning.tracks.admission import (
    ADMISSION_PROTOCOL_VERSION,
    ADMISSION_REPORT_SCHEMA_VERSION,
    FORMAL_ADMISSION_WORLDS,
    FORMAL_CONTROL_BLOCK_STEPS,
    FORMAL_SPLIT_RULES,
    AdmissionInfrastructureError,
    DriveabilityOutcome,
    GeometryAttempt,
    SplitAdmissionResult,
    SplitAdmissionRule,
    admit_split,
    evaluate_admission_report,
    materialize_admitted_assets,
    require_global_admission_diagnostics,
    split_result_dict,
    validate_split_rules,
    verify_selected_disjointness,
)
from controller_learning.tracks.generator import generate_track_candidate, pack_track
from controller_learning.tracks.hashing import track_geometry_sha256
from controller_learning.tracks.level0 import build_level0_track
from controller_learning.tracks.types import Track, TrackCapacity


def _fake_attempt(seed: int) -> GeometryAttempt:
    track = replace(build_level0_track(), seed=seed)
    return GeometryAttempt(
        seed=seed,
        status="accepted",
        reasons=(),
        track=track,
        geometry_sha256=f"{seed + 1:064x}",
    )


def _successful(tracks: list[Track] | tuple[Track, ...]):
    return tuple(DriveabilityOutcome(track.seed, "success") for track in tracks)


def test_formal_split_rules_lock_disjoint_seed_ranges_and_quotas() -> None:
    validate_split_rules(FORMAL_SPLIT_RULES)

    actual = [
        (rule.split, rule.seed_start, rule.seed_stop, rule.track_count)
        for rule in FORMAL_SPLIT_RULES
    ]
    assert actual == [
        ("train", 0, 1_000_000, 10_000),
        ("validation", 1_000_000, 2_000_000, 100),
        ("test", 2_000_000, 3_000_000, 20),
    ]
    with pytest.raises(ValueError, match="locked"):
        validate_split_rules(FORMAL_SPLIT_RULES[::-1])


def test_geometry_attempt_requires_one_explicit_outcome() -> None:
    with pytest.raises(ValueError, match="at least one reason"):
        GeometryAttempt(1, "generation_rejected", ())
    with pytest.raises(ValueError, match="provide a Track"):
        GeometryAttempt(1, "accepted", ())


def test_selection_is_chunk_size_independent_and_attempts_each_seed_once() -> None:
    rule = SplitAdmissionRule("train", 0, 30, 7)

    def geometry(seed: int) -> GeometryAttempt:
        if seed in {1, 6}:
            return GeometryAttempt(seed, "validation_rejected", ("curvature_exceeded",))
        return _fake_attempt(seed)

    def physical(tracks):
        return tuple(
            DriveabilityOutcome(
                track.seed,
                "off_track" if track.seed in {2, 8} else "success",
            )
            for track in tracks
        )

    single = admit_split(
        rule,
        geometry_builder=geometry,
        driveability_admitter=physical,
        admission_chunk_size=1,
    )
    batched = admit_split(
        rule,
        geometry_builder=geometry,
        driveability_admitter=physical,
        admission_chunk_size=5,
    )

    assert [track.seed for track in single.selected_tracks] == [0, 3, 4, 5, 7, 9, 10]
    assert [track.seed for track in batched.selected_tracks] == [0, 3, 4, 5, 7, 9, 10]
    assert len({record.seed for record in batched.candidate_records}) == len(
        batched.candidate_records
    )
    assert any(
        record.selection_status == "quota_already_satisfied" for record in batched.candidate_records
    )


def test_admission_records_geometry_physics_and_duplicate_rejections() -> None:
    rule = SplitAdmissionRule("validation", 100, 110, 2)

    def geometry(seed: int) -> GeometryAttempt:
        if seed == 100:
            return GeometryAttempt(seed, "generation_rejected", ("length_out_of_range",))
        attempt = _fake_attempt(seed)
        if seed == 103:
            return replace(attempt, geometry_sha256=f"{102 + 1:064x}")
        return attempt

    def physical(tracks):
        return tuple(
            DriveabilityOutcome(track.seed, "timeout" if track.seed == 101 else "success")
            for track in tracks
        )

    result = admit_split(
        rule,
        geometry_builder=geometry,
        driveability_admitter=physical,
        admission_chunk_size=2,
    )
    by_seed = {record.seed: record for record in result.candidate_records}

    assert [track.seed for track in result.selected_tracks] == [102, 104]
    assert by_seed[100].selection_status == "geometry_rejected"
    assert by_seed[101].selection_status == "driveability_rejected"
    assert by_seed[103].selection_status == "duplicate_geometry_rejected"
    assert result.complete


def test_admission_enforces_callback_identity_and_result_count() -> None:
    rule = SplitAdmissionRule("test", 200, 203, 1)
    with pytest.raises(ValueError, match="one outcome"):
        admit_split(
            rule,
            geometry_builder=_fake_attempt,
            driveability_admitter=lambda tracks: (),
            admission_chunk_size=2,
        )
    with pytest.raises(ValueError, match="seed identity"):
        admit_split(
            rule,
            geometry_builder=_fake_attempt,
            driveability_admitter=lambda tracks: (
                DriveabilityOutcome(tracks[0].seed + 1, "success"),
                DriveabilityOutcome(tracks[1].seed, "success"),
            ),
            admission_chunk_size=2,
        )


def test_global_gpu_fault_aborts_instead_of_becoming_track_rejections() -> None:
    passing = {
        "finite": True,
        "time_monotonic": True,
        "contact_overflow": False,
        "constraint_overflow": False,
        "unexpected_contact": False,
    }
    require_global_admission_diagnostics(passing)
    with pytest.raises(AdmissionInfrastructureError, match="contact_overflow"):
        require_global_admission_diagnostics({**passing, "contact_overflow": True})

    calls: list[tuple[int, ...]] = []

    def infrastructure_failure(tracks):
        calls.append(tuple(track.seed for track in tracks))
        raise AdmissionInfrastructureError("fake batch invariant failure")

    with pytest.raises(AdmissionInfrastructureError, match="fake batch invariant failure"):
        admit_split(
            SplitAdmissionRule("train", 0, 20, 5),
            geometry_builder=_fake_attempt,
            driveability_admitter=infrastructure_failure,
            admission_chunk_size=3,
        )
    assert calls == [(0, 1, 2)]


def _actual_track(seed: int) -> Track:
    capacity = TrackCapacity(640, 48)
    return pack_track(generate_track_candidate(seed), capacity)


def _one_track_result(split: str, seed: int) -> SplitAdmissionResult:
    track = _actual_track(seed)
    digest = track_geometry_sha256(track)
    rule = SplitAdmissionRule(split, seed, seed + 10, 1)  # type: ignore[arg-type]
    result = admit_split(
        rule,
        geometry_builder=lambda current: GeometryAttempt(
            current,
            "accepted",
            (),
            _actual_track(current),
            track_geometry_sha256(_actual_track(current)),
        ),
        driveability_admitter=_successful,
        admission_chunk_size=1,
    )
    assert result.selected_hashes == (digest,)
    return result


def test_materialization_uses_locked_repository_and_local_cache_names(tmp_path) -> None:
    level0 = build_level0_track()
    level0_hash = track_geometry_sha256(level0)
    results = (
        _one_track_result("train", 10),
        _one_track_result("validation", 20),
        _one_track_result("test", 30),
    )
    verify_selected_disjointness(level0, level0_hash, results)

    outputs = materialize_admitted_assets(
        benchmark_version="0.1",
        asset_directory=tmp_path / "assets",
        train_cache_directory=tmp_path / "cache",
        level0_track=level0,
        level0_hash=level0_hash,
        split_results=results,
    )

    assert {path.name for path in (tmp_path / "assets").iterdir()} == {
        "level0.json",
        "level0.npz",
        "train.json",
        "validation.json",
        "validation.npz",
        "test.json",
        "test.npz",
    }
    assert {path.name for path in (tmp_path / "cache").iterdir()} == {"train_pool.npz"}
    assert outputs["train"]["asset_file"] == "train_pool.npz"
    assert outputs["train"]["storage"] == "local_cache"


def _passing_report() -> dict:
    split_data = {}
    for rule in FORMAL_SPLIT_RULES:
        split_data[rule.split] = {
            "complete": True,
            "selected_count": rule.track_count,
            "attempted_seed_count": rule.track_count,
            "candidate_results": [
                {
                    "seed": rule.seed_start + index,
                    "selection_status": "selected",
                    "driveability_status": "success",
                }
                for index in range(rule.track_count)
            ],
        }
    source = {
        "git_revision": "0123456789abcdef",
        "relevant_source_clean": True,
        "source_files_sha256": {"source.py": "0" * 64},
    }
    return {
        "schema_version": ADMISSION_REPORT_SCHEMA_VERSION,
        "protocol_version": ADMISSION_PROTOCOL_VERSION,
        "protocol": {
            "admission_worlds": FORMAL_ADMISSION_WORLDS,
            "fixed_shape_reused": True,
            "ascending_seed_order": True,
            "one_candidate_per_seed": True,
            "hidden_retry": False,
            "bounded_control_step_chunks": True,
            "control_block_steps": FORMAL_CONTROL_BLOCK_STEPS,
        },
        "splits": split_data,
        "disjointness": {
            "all_selected_seeds_disjoint": True,
            "all_selected_geometry_hashes_disjoint": True,
        },
        "runtime": {
            "jax_device": {"platform": "gpu"},
            "physics_backend": "MJX-Warp",
            "python_version": "3.11.15",
            "numpy_version": "2.4.0",
            "jax_version": "0.10.2",
            "mujoco_version": "3.10.0",
            "mjx_warp_version": "3.10.0",
        },
        "timing": {
            "total_s": 10.0,
            "gpu": {
                "adapter_creation_s": 1.0,
                "compilation_s": 2.0,
                "measured_execution_s": 7.0,
                "compiled_executable_sets": 1,
                "batch_calls": 11,
                "executed_control_steps": 100,
                "executed_transitions": 102_400,
                "host_sync_count": 11,
            },
        },
        "source_evidence": {"before": source, "after": dict(source)},
        "artifacts": {name: {} for name in ("level0", "train", "validation", "test")},
    }


def test_report_gates_require_fixed_shape_gpu_clean_source_and_complete_quotas() -> None:
    report = _passing_report()
    assert all(check["passed"] for check in evaluate_admission_report(report))

    report["protocol"]["fixed_shape_reused"] = False
    report["splits"]["validation"]["selected_count"] = 99
    report["source_evidence"]["after"]["relevant_source_clean"] = False
    failed = {check["id"] for check in evaluate_admission_report(report) if not check["passed"]}
    assert failed == {
        "protocol.shape",
        "source.clean",
        "splits.quotas",
        "splits.rows",
    }


def test_split_report_keeps_every_rejection_row() -> None:
    result = admit_split(
        SplitAdmissionRule("test", 0, 4, 1),
        geometry_builder=lambda seed: (
            GeometryAttempt(seed, "capacity_rejected", ("capacity",))
            if seed == 0
            else _fake_attempt(seed)
        ),
        driveability_admitter=_successful,
        admission_chunk_size=2,
    )
    payload = split_result_dict(result)

    assert payload["attempted_seed_count"] == len(payload["candidate_results"])
    assert payload["candidate_results"][0]["geometry_reasons"] == ["capacity"]
