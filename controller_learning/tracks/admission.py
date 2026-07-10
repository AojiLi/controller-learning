"""Deterministic, chunk-independent admission for versioned benchmark Track pools.

The orchestration in this module is deliberately independent from JAX and the physics backend.
Formal GPU execution is supplied as one injected batch callback by
``scripts/build_track_assets.py``; CPU tests can therefore exercise the complete seed-selection,
rejection, manifest, and report logic without pretending to validate physics.
"""

from __future__ import annotations

import json
import math
import os
import tempfile
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from itertools import pairwise
from pathlib import Path
from types import MappingProxyType
from typing import Any, Literal

import numpy as np

from controller_learning.tracks.assets import (
    TRACK_ASSET_SCHEMA_VERSION,
    TrackAssetManifest,
    TrackAssetRecord,
    save_track_batch_npz,
    write_track_asset_manifest,
)
from controller_learning.tracks.generator import (
    TrackGenerationError,
    TrackGenerationSpec,
    generate_track_candidate,
    pack_track,
)
from controller_learning.tracks.hashing import track_geometry_sha256
from controller_learning.tracks.types import Track, TrackCapacity, stack_tracks
from controller_learning.tracks.validator import TrackValidationSpec, validate_track_candidate

ADMISSION_REPORT_SCHEMA_VERSION = 1
ADMISSION_PROTOCOL_VERSION = "m5-track-admission-v1"
GEOMETRY_VALIDATION_VERSION = "m3-geometry-v1"
DRIVEABILITY_PROTOCOL_VERSION = "m5-driveability-v1"
FORMAL_ADMISSION_WORLDS = 1024
FORMAL_CONTROL_BLOCK_STEPS = 100

SplitName = Literal["train", "validation", "test"]
DriveabilityStatus = Literal[
    "success",
    "off_track",
    "timeout",
    "invalid_action",
    "numerical_failure",
]

_DRIVEABILITY_STATUSES = {
    "success",
    "off_track",
    "timeout",
    "invalid_action",
    "numerical_failure",
}


class AdmissionInfrastructureError(RuntimeError):
    """Raised when formal admission evidence is invalid for the entire physical batch."""


def require_global_admission_diagnostics(
    diagnostics: Mapping[str, bool],
) -> None:
    """Fail the whole admission run on a batch-wide physics invariant violation.

    Global finite/contact-capacity/constraint/contact-semantics failures are properties of the
    formal executable and its evidence, not evidence that every Track in the current chunk is
    individually undriveable. Converting such a fault to per-Track rejections would make the
    selected first-N pool depend on chunk boundaries.
    """

    expected = {
        "finite": True,
        "time_monotonic": True,
        "contact_overflow": False,
        "constraint_overflow": False,
        "unexpected_contact": False,
    }
    failed = [
        field
        for field, expected_value in expected.items()
        if type(diagnostics.get(field)) is not bool or diagnostics.get(field) is not expected_value
    ]
    if failed:
        raise AdmissionInfrastructureError(
            "formal MJX-Warp admission invariant failed: " + ", ".join(failed)
        )


@dataclass(frozen=True, slots=True)
class SplitAdmissionRule:
    """Fixed public seed interval and quota for one Level 1 split."""

    split: SplitName
    seed_start: int
    seed_stop: int
    track_count: int

    def __post_init__(self) -> None:
        if self.split not in ("train", "validation", "test"):
            raise ValueError("split must be train, validation, or test")
        if type(self.seed_start) is not int or type(self.seed_stop) is not int:
            raise TypeError("seed interval bounds must be integers")
        if not 0 <= self.seed_start < self.seed_stop <= 2**32:
            raise ValueError("seed interval must be a non-empty uint32 half-open range")
        if type(self.track_count) is not int or self.track_count < 1:
            raise ValueError("track_count must be a positive integer")
        if self.track_count > self.seed_stop - self.seed_start:
            raise ValueError("track_count cannot exceed the seed interval")


FORMAL_SPLIT_RULES: tuple[SplitAdmissionRule, ...] = (
    SplitAdmissionRule("train", 0, 1_000_000, 10_000),
    SplitAdmissionRule("validation", 1_000_000, 2_000_000, 100),
    SplitAdmissionRule("test", 2_000_000, 3_000_000, 20),
)


def validate_split_rules(rules: Sequence[SplitAdmissionRule]) -> None:
    """Reject missing, reordered, overlapping, or non-formal split definitions."""

    if tuple(rules) != FORMAL_SPLIT_RULES:
        raise ValueError("formal M5 admission requires the locked train/validation/test rules")
    for previous, current in pairwise(rules):
        if previous.seed_stop > current.seed_start:
            raise ValueError("split seed intervals cannot overlap")


@dataclass(frozen=True, slots=True)
class GeometryAttempt:
    """Result of exactly one generator/validator/packer attempt for one seed."""

    seed: int
    status: Literal["accepted", "generation_rejected", "validation_rejected", "capacity_rejected"]
    reasons: tuple[str, ...]
    track: Track | None = field(default=None, repr=False)
    geometry_sha256: str | None = None

    def __post_init__(self) -> None:
        if type(self.seed) is not int or not 0 <= self.seed <= np.iinfo(np.uint32).max:
            raise ValueError("GeometryAttempt.seed must fit uint32")
        statuses = {
            "accepted",
            "generation_rejected",
            "validation_rejected",
            "capacity_rejected",
        }
        if self.status not in statuses:
            raise ValueError("invalid GeometryAttempt.status")
        reasons = tuple(self.reasons)
        object.__setattr__(self, "reasons", reasons)
        accepted = self.status == "accepted"
        if accepted != (self.track is not None and self.geometry_sha256 is not None):
            raise ValueError("accepted geometry must provide a Track and geometry hash")
        if accepted:
            assert self.track is not None
            if self.track.seed != self.seed:
                raise ValueError("accepted Track seed must match GeometryAttempt.seed")
            if reasons:
                raise ValueError("accepted geometry cannot have rejection reasons")
            digest = self.geometry_sha256
            if not isinstance(digest, str) or len(digest) != 64:
                raise ValueError("accepted geometry must provide a SHA-256 digest")
        elif not reasons:
            raise ValueError("rejected geometry must provide at least one reason")


@dataclass(frozen=True, slots=True)
class DriveabilityOutcome:
    """One physical admission outcome returned in the same order as its input Tracks."""

    seed: int
    status: DriveabilityStatus
    metrics: Mapping[str, bool | int | float | None] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if type(self.seed) is not int or not 0 <= self.seed <= np.iinfo(np.uint32).max:
            raise ValueError("DriveabilityOutcome.seed must fit uint32")
        if self.status not in _DRIVEABILITY_STATUSES:
            raise ValueError("invalid driveability status")
        metrics = dict(self.metrics)
        for key, value in metrics.items():
            if not isinstance(key, str) or not key:
                raise ValueError("driveability metric names must be non-empty strings")
            if not (value is None or type(value) in (bool, int, float)):
                raise TypeError("driveability metrics must be strict JSON scalar values")
            if isinstance(value, float) and not math.isfinite(value):
                raise ValueError("driveability metrics must be finite")
        object.__setattr__(self, "metrics", MappingProxyType(metrics))


@dataclass(frozen=True, slots=True)
class CandidateAdmissionRecord:
    """Stable report row for one explicitly attempted seed."""

    seed: int
    geometry_status: str
    geometry_reasons: tuple[str, ...]
    geometry_sha256: str | None
    driveability_status: str
    selection_status: str
    metrics: Mapping[str, bool | int | float | None] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "geometry_reasons", tuple(self.geometry_reasons))
        object.__setattr__(self, "metrics", MappingProxyType(dict(self.metrics)))

    @property
    def selected(self) -> bool:
        return self.selection_status == "selected"


@dataclass(frozen=True, slots=True)
class SplitAdmissionResult:
    """Complete deterministic result for one split scan."""

    rule: SplitAdmissionRule
    selected_tracks: tuple[Track, ...]
    selected_hashes: tuple[str, ...]
    candidate_records: tuple[CandidateAdmissionRecord, ...]
    admission_batch_calls: int
    padded_world_count: int

    def __post_init__(self) -> None:
        if len(self.selected_tracks) != len(self.selected_hashes):
            raise ValueError("selected Tracks and hashes must have equal lengths")
        if any(
            track.seed != record.seed
            for track, record in zip(
                self.selected_tracks,
                (record for record in self.candidate_records if record.selected),
                strict=True,
            )
        ):
            raise ValueError("selected Track order must match selected candidate records")

    @property
    def complete(self) -> bool:
        return len(self.selected_tracks) == self.rule.track_count

    @property
    def attempted_seed_count(self) -> int:
        return len(self.candidate_records)

    @property
    def rejection_count(self) -> int:
        return sum(
            record.selection_status not in ("selected", "quota_already_satisfied")
            for record in self.candidate_records
        )


GeometryBuilder = Callable[[int], GeometryAttempt]
DriveabilityAdmitter = Callable[[Sequence[Track]], Sequence[DriveabilityOutcome]]


def build_geometry_attempt(
    seed: int,
    *,
    generation_spec: TrackGenerationSpec,
    validation_spec: TrackValidationSpec,
    capacity: TrackCapacity,
) -> GeometryAttempt:
    """Generate, validate, and pack one seed exactly once, preserving rejection reasons."""

    try:
        candidate = generate_track_candidate(seed, generation_spec)
    except TrackGenerationError as error:
        return GeometryAttempt(seed, "generation_rejected", (error.reason,))
    validation = validate_track_candidate(candidate, validation_spec)
    if not validation.valid:
        return GeometryAttempt(seed, "validation_rejected", validation.reasons)
    try:
        track = pack_track(candidate, capacity)
    except TrackGenerationError as error:
        return GeometryAttempt(seed, "capacity_rejected", (error.reason,))
    return GeometryAttempt(
        seed=seed,
        status="accepted",
        reasons=(),
        track=track,
        geometry_sha256=track_geometry_sha256(track),
    )


def _geometry_rejection_record(attempt: GeometryAttempt) -> CandidateAdmissionRecord:
    return CandidateAdmissionRecord(
        seed=attempt.seed,
        geometry_status=attempt.status,
        geometry_reasons=attempt.reasons,
        geometry_sha256=None,
        driveability_status="not_run",
        selection_status="geometry_rejected",
    )


def admit_split(
    rule: SplitAdmissionRule,
    *,
    geometry_builder: GeometryBuilder,
    driveability_admitter: DriveabilityAdmitter,
    admission_chunk_size: int = FORMAL_ADMISSION_WORLDS,
    excluded_geometry_hashes: frozenset[str] = frozenset(),
) -> SplitAdmissionResult:
    """Select the first driveable, globally unique Tracks in ascending seed order.

    The physical callback may run any positive chunk size. Results are always consumed in seed
    order, so the selected first-N set is invariant to chunk boundaries. A seed is generated once
    and admitted at most once. Candidates physically processed after the quota is reached within
    the final batch remain in the report as ``quota_already_satisfied`` rather than being silently
    discarded.
    """

    if type(admission_chunk_size) is not int or admission_chunk_size < 1:
        raise ValueError("admission_chunk_size must be a positive integer")
    selected_tracks: list[Track] = []
    selected_hashes: list[str] = []
    known_hashes = set(excluded_geometry_hashes)
    records: list[CandidateAdmissionRecord] = []
    pending_attempts: list[GeometryAttempt] = []
    batch_calls = 0
    padded_world_count = 0
    next_seed = rule.seed_start

    def flush() -> None:
        nonlocal batch_calls, padded_world_count
        if not pending_attempts:
            return
        tracks = [attempt.track for attempt in pending_attempts]
        assert all(track is not None for track in tracks)
        typed_tracks = [track for track in tracks if track is not None]
        outcomes = tuple(driveability_admitter(typed_tracks))
        batch_calls += 1
        padded_world_count += admission_chunk_size - len(typed_tracks)
        if len(outcomes) != len(typed_tracks):
            raise ValueError("driveability callback must return one outcome per Track")
        for attempt, track, outcome in zip(pending_attempts, typed_tracks, outcomes, strict=True):
            if outcome.seed != attempt.seed:
                raise ValueError("driveability callback changed Track order or seed identity")
            digest = attempt.geometry_sha256
            assert digest is not None
            if len(selected_tracks) >= rule.track_count:
                selection = "quota_already_satisfied"
            elif outcome.status != "success":
                selection = "driveability_rejected"
            elif digest in known_hashes:
                selection = "duplicate_geometry_rejected"
            else:
                selection = "selected"
                selected_tracks.append(track)
                selected_hashes.append(digest)
                known_hashes.add(digest)
            records.append(
                CandidateAdmissionRecord(
                    seed=attempt.seed,
                    geometry_status="accepted",
                    geometry_reasons=(),
                    geometry_sha256=digest,
                    driveability_status=outcome.status,
                    selection_status=selection,
                    metrics=outcome.metrics,
                )
            )
        pending_attempts.clear()

    while next_seed < rule.seed_stop and len(selected_tracks) < rule.track_count:
        attempt = geometry_builder(next_seed)
        if attempt.seed != next_seed:
            raise ValueError("geometry callback changed seed identity")
        next_seed += 1
        if attempt.status == "accepted":
            pending_attempts.append(attempt)
            if len(pending_attempts) == admission_chunk_size:
                flush()
        else:
            records.append(_geometry_rejection_record(attempt))
    flush()
    records.sort(key=lambda record: record.seed)
    if len({record.seed for record in records}) != len(records):
        raise RuntimeError("a seed was attempted more than once")
    return SplitAdmissionResult(
        rule=rule,
        selected_tracks=tuple(selected_tracks),
        selected_hashes=tuple(selected_hashes),
        candidate_records=tuple(records),
        admission_batch_calls=batch_calls,
        padded_world_count=padded_world_count,
    )


def verify_selected_disjointness(
    level0_track: Track,
    level0_hash: str,
    split_results: Sequence[SplitAdmissionResult],
) -> dict[str, bool]:
    """Return and enforce exact seed/hash disjointness across Level 0 and all Level 1 splits."""

    seeds = [level0_track.seed]
    hashes = [level0_hash]
    for result in split_results:
        seeds.extend(track.seed for track in result.selected_tracks)
        hashes.extend(result.selected_hashes)
    checks = {
        "all_selected_seeds_disjoint": len(seeds) == len(set(seeds)),
        "all_selected_geometry_hashes_disjoint": len(hashes) == len(set(hashes)),
    }
    if not all(checks.values()):
        raise ValueError("selected benchmark Tracks are not seed/hash disjoint")
    return checks


def _manifest_for_tracks(
    *,
    benchmark_version: str,
    split: Literal["level0", "train", "validation", "test"],
    tracks: Sequence[Track],
    hashes: Sequence[str],
    asset_file: str,
    asset_sha256: str,
) -> TrackAssetManifest:
    if not tracks or len(tracks) != len(hashes):
        raise ValueError("manifest requires non-empty, aligned Tracks and hashes")
    return TrackAssetManifest(
        schema_version=TRACK_ASSET_SCHEMA_VERSION,
        benchmark_version=benchmark_version,
        level_id=0 if split == "level0" else 1,
        split=split,
        generator_version=tracks[0].generator_version,
        geometry_validation_version=GEOMETRY_VALIDATION_VERSION,
        driveability_protocol_version=DRIVEABILITY_PROTOCOL_VERSION,
        track_width_m=float(tracks[0].width_m),
        track_count=len(tracks),
        capacity=tracks[0].capacity,
        asset_file=asset_file,
        asset_sha256=asset_sha256,
        tracks=tuple(
            TrackAssetRecord(
                seed=track.seed,
                geometry_sha256=digest,
                geometry_validation="passed",
                driveability_validation="passed",
            )
            for track, digest in zip(tracks, hashes, strict=True)
        ),
    )


def materialize_admitted_assets(
    *,
    benchmark_version: str,
    asset_directory: Path,
    train_cache_directory: Path,
    level0_track: Track,
    level0_hash: str,
    split_results: Sequence[SplitAdmissionResult],
) -> dict[str, dict[str, str]]:
    """Write canonical assets using the locked repository/cache layout."""

    by_split = {result.rule.split: result for result in split_results}
    if set(by_split) != {"train", "validation", "test"}:
        raise ValueError("materialization requires one result for every Level 1 split")
    if not all(result.complete for result in by_split.values()):
        raise ValueError("cannot materialize an incomplete admission result")
    asset_directory.mkdir(parents=True, exist_ok=True)
    train_cache_directory.mkdir(parents=True, exist_ok=True)
    outputs: dict[str, dict[str, str]] = {}

    entries: tuple[tuple[str, Sequence[Track], Sequence[str], Path, str], ...] = (
        (
            "level0",
            (level0_track,),
            (level0_hash,),
            asset_directory / "level0.npz",
            "level0.npz",
        ),
        (
            "train",
            by_split["train"].selected_tracks,
            by_split["train"].selected_hashes,
            train_cache_directory / "train_pool.npz",
            "train_pool.npz",
        ),
        (
            "validation",
            by_split["validation"].selected_tracks,
            by_split["validation"].selected_hashes,
            asset_directory / "validation.npz",
            "validation.npz",
        ),
        (
            "test",
            by_split["test"].selected_tracks,
            by_split["test"].selected_hashes,
            asset_directory / "test.npz",
            "test.npz",
        ),
    )
    for split, tracks, hashes, npz_path, asset_file in entries:
        batch = stack_tracks(list(tracks))
        asset_sha256 = save_track_batch_npz(batch, npz_path)
        manifest = _manifest_for_tracks(
            benchmark_version=benchmark_version,
            split=split,  # type: ignore[arg-type]
            tracks=tracks,
            hashes=hashes,
            asset_file=asset_file,
            asset_sha256=asset_sha256,
        )
        manifest_path = asset_directory / f"{split}.json"
        manifest_sha256 = write_track_asset_manifest(manifest, manifest_path)
        outputs[split] = {
            "asset_file": asset_file,
            "asset_sha256": asset_sha256,
            "manifest_file": manifest_path.name,
            "manifest_sha256": manifest_sha256,
            "storage": "local_cache" if split == "train" else "repository_asset",
        }
    return outputs


def candidate_record_dict(record: CandidateAdmissionRecord) -> dict[str, Any]:
    """Convert one result row to strict-JSON-compatible data."""

    return {
        "driveability_status": record.driveability_status,
        "geometry_reasons": list(record.geometry_reasons),
        "geometry_sha256": record.geometry_sha256,
        "geometry_status": record.geometry_status,
        "metrics": dict(record.metrics),
        "seed": record.seed,
        "selection_status": record.selection_status,
    }


def split_result_dict(result: SplitAdmissionResult) -> dict[str, Any]:
    """Convert a complete split result to report data without losing rejections."""

    selection_counts: dict[str, int] = {}
    for record in result.candidate_records:
        selection_counts[record.selection_status] = (
            selection_counts.get(record.selection_status, 0) + 1
        )
    return {
        "rule": {
            "seed_start_inclusive": result.rule.seed_start,
            "seed_stop_exclusive": result.rule.seed_stop,
            "track_count": result.rule.track_count,
        },
        "complete": result.complete,
        "attempted_seed_count": result.attempted_seed_count,
        "admission_batch_calls": result.admission_batch_calls,
        "padded_world_count": result.padded_world_count,
        "selected_count": len(result.selected_tracks),
        "selected_seeds": [track.seed for track in result.selected_tracks],
        "selected_geometry_sha256": list(result.selected_hashes),
        "selection_counts": dict(sorted(selection_counts.items())),
        "candidate_results": [candidate_record_dict(row) for row in result.candidate_records],
    }


def evaluate_admission_report(report: Mapping[str, Any]) -> tuple[dict[str, Any], ...]:
    """Evaluate strict M5 publication gates without trusting a top-level status string."""

    protocol = report.get("protocol", {})
    splits = report.get("splits", {})
    disjointness = report.get("disjointness", {})
    source = report.get("source_evidence", {})
    before = source.get("before", {})
    after = source.get("after", {})
    runtime = report.get("runtime", {})
    timing = report.get("timing", {})
    gpu_timing = timing.get("gpu", {})
    path_evidence = protocol.get("official_output_paths", {})
    artifacts = report.get("artifacts", {})
    artifact_readback = report.get("artifact_readback", {})
    expected_artifact_splits = {"level0", "train", "validation", "test"}
    expected_fixed_splits = {"level0", "validation", "test"}

    readback_manifest_hashes = artifact_readback.get("manifest_files_sha256", {})
    readback_asset_hashes = artifact_readback.get("asset_files_sha256", {})
    readback_hashes_match = (
        set(artifacts) == expected_artifact_splits
        and set(readback_manifest_hashes) == expected_artifact_splits
        and set(readback_asset_hashes) == expected_artifact_splits
        and all(
            artifacts[split].get("manifest_sha256") == readback_manifest_hashes[split]
            and artifacts[split].get("asset_sha256") == readback_asset_hashes[split]
            for split in expected_artifact_splits
        )
    )

    split_rows_match = True
    split_order_valid = True
    split_selected_success = True
    for rule in FORMAL_SPLIT_RULES:
        payload = splits.get(rule.split, {})
        rows = payload.get("candidate_results", [])
        selected_rows = [row for row in rows if row.get("selection_status") == "selected"]
        split_rows_match &= payload.get("attempted_seed_count") == len(rows) and len(
            selected_rows
        ) == payload.get("selected_count")
        seeds = [row.get("seed") for row in rows]
        split_order_valid &= (
            all(type(seed) is int for seed in seeds)
            and seeds == sorted(seeds)
            and len(seeds) == len(set(seeds))
            and all(rule.seed_start <= seed < rule.seed_stop for seed in seeds)
        )
        split_selected_success &= all(
            row.get("driveability_status") == "success" for row in selected_rows
        )

    timing_values = (
        timing.get("total_s"),
        gpu_timing.get("adapter_creation_s"),
        gpu_timing.get("compilation_s"),
        gpu_timing.get("measured_execution_s"),
    )
    timing_complete = all(
        type(value) in (int, float) and math.isfinite(value) and value > 0.0
        for value in timing_values
    ) and all(
        type(gpu_timing.get(field)) is int and gpu_timing.get(field) > 0
        for field in (
            "compiled_executable_sets",
            "batch_calls",
            "executed_control_steps",
            "executed_transitions",
            "host_sync_count",
        )
    )
    checks = (
        (
            "report.schema",
            report.get("schema_version") == ADMISSION_REPORT_SCHEMA_VERSION
            and report.get("protocol_version") == ADMISSION_PROTOCOL_VERSION,
        ),
        (
            "protocol.shape",
            protocol.get("admission_worlds") == FORMAL_ADMISSION_WORLDS
            and protocol.get("fixed_shape_reused") is True,
        ),
        (
            "protocol.seed_order",
            protocol.get("ascending_seed_order") is True
            and protocol.get("one_candidate_per_seed") is True
            and protocol.get("hidden_retry") is False,
        ),
        (
            "protocol.bounded_sync",
            protocol.get("bounded_control_step_chunks") is True
            and protocol.get("control_block_steps") == FORMAL_CONTROL_BLOCK_STEPS,
        ),
        (
            "paths.official",
            set(path_evidence)
            == {
                "official_asset_directory",
                "official_report_path",
                "official_train_cache_directory",
            }
            and all(value is True for value in path_evidence.values()),
        ),
        (
            "splits.complete",
            set(splits) == {"train", "validation", "test"}
            and all(splits[name].get("complete") is True for name in splits),
        ),
        (
            "splits.quotas",
            all(
                splits.get(rule.split, {}).get("selected_count") == rule.track_count
                for rule in FORMAL_SPLIT_RULES
            ),
        ),
        ("splits.rows", split_rows_match),
        ("splits.seed_order", split_order_valid),
        (
            "splits.success",
            split_selected_success,
        ),
        (
            "tracks.disjoint",
            disjointness.get("all_selected_seeds_disjoint") is True
            and disjointness.get("all_selected_geometry_hashes_disjoint") is True,
        ),
        (
            "runtime.gpu",
            runtime.get("jax_device", {}).get("platform") == "gpu"
            and runtime.get("physics_backend") == "MJX-Warp",
        ),
        (
            "runtime.versions",
            all(
                isinstance(runtime.get(field), str) and bool(runtime.get(field))
                for field in (
                    "python_version",
                    "numpy_version",
                    "jax_version",
                    "mujoco_version",
                    "mjx_warp_version",
                )
            ),
        ),
        ("timing.complete", timing_complete),
        (
            "source.stable",
            before.get("git_revision") is not None
            and before.get("git_revision") == after.get("git_revision")
            and before.get("source_files_sha256") == after.get("source_files_sha256"),
        ),
        (
            "source.clean",
            before.get("relevant_source_clean") is True
            and after.get("relevant_source_clean") is True,
        ),
        (
            "artifacts.complete",
            set(artifacts) == expected_artifact_splits,
        ),
        (
            "artifacts.readback",
            artifact_readback.get("passed") is True
            and set(artifact_readback.get("official_manifest_splits", ()))
            == expected_artifact_splits
            and set(artifact_readback.get("fixed_asset_splits", ())) == expected_fixed_splits
            and artifact_readback.get("train_cache_verified") is True
            and readback_hashes_match,
        ),
    )
    return tuple({"id": check_id, "passed": bool(passed)} for check_id, passed in checks)


def write_strict_json(value: Mapping[str, Any], path: Path) -> None:
    """Atomically write deterministic strict JSON."""

    payload = (json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n").encode()
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            prefix=f".{path.name}.", suffix=".tmp", dir=path.parent, delete=False
        ) as file:
            temporary = Path(file.name)
            file.write(payload)
        os.replace(temporary, path)
        temporary = None
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


__all__ = [
    "ADMISSION_PROTOCOL_VERSION",
    "ADMISSION_REPORT_SCHEMA_VERSION",
    "DRIVEABILITY_PROTOCOL_VERSION",
    "FORMAL_ADMISSION_WORLDS",
    "FORMAL_CONTROL_BLOCK_STEPS",
    "FORMAL_SPLIT_RULES",
    "GEOMETRY_VALIDATION_VERSION",
    "AdmissionInfrastructureError",
    "CandidateAdmissionRecord",
    "DriveabilityOutcome",
    "GeometryAttempt",
    "SplitAdmissionResult",
    "SplitAdmissionRule",
    "admit_split",
    "build_geometry_attempt",
    "candidate_record_dict",
    "evaluate_admission_report",
    "materialize_admitted_assets",
    "require_global_admission_diagnostics",
    "split_result_dict",
    "validate_split_rules",
    "verify_selected_disjointness",
    "write_strict_json",
]
