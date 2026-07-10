"""Tests for deterministic M3 Track-capacity evidence."""

from __future__ import annotations

import json
from types import SimpleNamespace

import pytest

from controller_learning.tracks import capacity_benchmark
from controller_learning.tracks.capacity_benchmark import (
    REPORT_SCHEMA_VERSION,
    run_track_capacity_benchmark,
    write_track_capacity_report,
)
from controller_learning.tracks.generator import TrackGenerationSpec


@pytest.fixture(scope="module")
def small_report():
    spec = TrackGenerationSpec(dense_samples_per_control_point=128)
    return run_track_capacity_benchmark(
        seed_start=20,
        seed_count=3,
        arc_spacings_m=(1.0,),
        base_spec=spec,
        reproducibility_sample_count=2,
    )


def test_report_schema_seed_range_and_percentiles(small_report) -> None:
    assert small_report["schema_version"] == REPORT_SCHEMA_VERSION
    assert small_report["protocol"]["seed_range"] == {
        "start_inclusive": 20,
        "end_exclusive": 23,
        "count": 3,
        "contiguous": True,
    }
    assert small_report["protocol"]["percentiles"] == [
        "min",
        "p1",
        "p5",
        "p50",
        "p95",
        "p99",
        "p99_9",
        "max",
    ]
    result = small_report["spacing_results"][0]
    generated = result["statistics"]["generated_candidates"]
    assert generated["count"] == result["generation"]["succeeded_count"]
    if generated["count"]:
        assert set(generated["point_count"]) == set(small_report["protocol"]["percentiles"])
        assert set(generated["checkpoint_count"]) == set(small_report["protocol"]["percentiles"])


def test_report_is_exactly_deterministic_and_reproducible(small_report) -> None:
    repeated = run_track_capacity_benchmark(
        seed_start=20,
        seed_count=3,
        arc_spacings_m=(1.0,),
        base_spec=TrackGenerationSpec(dense_samples_per_control_point=128),
        reproducibility_sample_count=2,
    )
    assert repeated == small_report
    reproducibility = repeated["spacing_results"][0]["reproducibility"]
    assert reproducibility["passed"] is True
    assert reproducibility["mismatch_seeds"] == []


def test_rejections_and_optional_validation_are_explicit(small_report) -> None:
    result = small_report["spacing_results"][0]
    rejections = result["rejections"]
    assert rejections["total"] == sum(rejections["by_stage"].values())
    if small_report["validator"]["validation_available"]:
        assert result["validation"]["accepted_count"] is not None
        assert result["validation"]["rejected_count"] is not None
    else:
        assert result["validation"]["accepted_count"] is None
        assert result["validation"]["rejected_count"] is None
        assert result["statistics"]["validated_accepted_candidates"] is None


def test_generation_rejections_have_stable_structured_reasons() -> None:
    report = run_track_capacity_benchmark(
        seed_start=7,
        seed_count=1,
        arc_spacings_m=(1.0,),
        base_spec=TrackGenerationSpec(
            min_length_m=10.0,
            max_length_m=11.0,
            dense_samples_per_control_point=128,
        ),
        reproducibility_sample_count=1,
    )
    result = report["spacing_results"][0]
    assert result["generation"] == {
        "attempted_count": 1,
        "succeeded_count": 0,
        "failed_count": 1,
    }
    assert result["rejections"]["by_stage"] == {"generation": 1, "validation": 0}
    assert result["rejections"]["primary_reason_counts"] == {"length_out_of_range": 1}
    assert result["rejections"]["sample_seeds_by_primary_reason"] == {"length_out_of_range": [7]}
    assert result["reproducibility"]["passed"] is True


def test_validation_rejections_count_primary_and_all_reasons(monkeypatch) -> None:
    def reject(_candidate):
        return SimpleNamespace(
            valid=False,
            reasons=("curvature_exceeded", "nonlocal_clearance"),
            primary_reason="curvature_exceeded",
            metrics={"synthetic": 1},
        )

    monkeypatch.setattr(capacity_benchmark, "_load_candidate_validator", lambda: reject)
    report = run_track_capacity_benchmark(
        seed_start=0,
        seed_count=1,
        arc_spacings_m=(1.0,),
        base_spec=TrackGenerationSpec(dense_samples_per_control_point=128),
        reproducibility_sample_count=1,
    )
    result = report["spacing_results"][0]
    assert result["rejections"]["by_stage"] == {"generation": 0, "validation": 1}
    assert result["rejections"]["primary_reason_counts"] == {"curvature_exceeded": 1}
    assert result["rejections"]["all_reason_counts"] == {
        "curvature_exceeded": 1,
        "nonlocal_clearance": 1,
    }
    assert result["validation"]["accepted_count"] == 0


def test_missing_validator_never_marks_generated_candidates_accepted(monkeypatch) -> None:
    monkeypatch.setattr(capacity_benchmark, "_load_candidate_validator", lambda: None)
    report = run_track_capacity_benchmark(
        seed_start=0,
        seed_count=1,
        arc_spacings_m=(1.0,),
        base_spec=TrackGenerationSpec(dense_samples_per_control_point=128),
        reproducibility_sample_count=1,
    )
    result = report["spacing_results"][0]
    assert report["validator"]["validation_available"] is False
    assert report["validator"]["validation_spec"] is None
    assert result["generation"]["succeeded_count"] == 1
    assert result["validation"]["accepted_count"] is None
    assert result["statistics"]["validated_accepted_candidates"] is None


def test_theoretical_capacity_selection_and_exact_memory(small_report) -> None:
    result = small_report["spacing_results"][0]
    theoretical = result["theoretical_capacity"]
    selected = result["selected_capacity_candidate"]
    assert theoretical["max_track_points_required"] == 601
    assert theoretical["max_checkpoints_required"] == 40
    assert selected["max_track_points"] >= theoretical["max_track_points_required"]
    assert selected["max_checkpoints"] >= theoretical["max_checkpoints_required"]

    memory = result["memory_estimates"]
    track_field_bytes = 41 * selected["max_track_points"] + 21 * selected["max_checkpoints"] + 12
    expected_per_track = track_field_bytes + 20
    assert memory["track_field_array_bytes_per_track"] == track_field_bytes
    assert memory["runtime_scalar_array_bytes_per_track"] == 20
    assert memory["bytes_per_track"] == expected_per_track
    assert memory["bytes_1024_world_batch"] == expected_per_track * 1024
    assert memory["bytes_10000_track_pool"] == expected_per_track * 10_000


def test_writer_emits_strict_json_without_machine_metadata(tmp_path) -> None:
    output = tmp_path / "capacity.json"
    report = write_track_capacity_report(
        output,
        seed_start=4,
        seed_count=1,
        arc_spacings_m=(1.25,),
        base_spec=TrackGenerationSpec(dense_samples_per_control_point=128),
        reproducibility_sample_count=1,
    )
    persisted = json.loads(output.read_text(encoding="utf-8"))
    assert persisted == report
    serialized = output.read_text(encoding="utf-8")
    assert "/home/" not in serialized
    assert "timestamp" not in serialized.lower()


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"seed_count": 0}, "seed_count"),
        ({"arc_spacings_m": ()}, "arc_spacings_m"),
        ({"arc_spacings_m": (1.0, 1.0)}, "duplicates"),
        ({"reproducibility_sample_count": 0}, "reproducibility_sample_count"),
    ],
)
def test_invalid_protocol_inputs_are_rejected(kwargs, message) -> None:
    with pytest.raises(ValueError, match=message):
        run_track_capacity_benchmark(**kwargs)
