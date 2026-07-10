"""Deterministic evidence for selecting fixed Track array capacities."""

from __future__ import annotations

import hashlib
import importlib
import json
import math
import subprocess
from collections import Counter, defaultdict
from collections.abc import Callable, Mapping, Sequence
from dataclasses import asdict, replace
from pathlib import Path
from typing import Any

import numpy as np

from controller_learning.tracks.generator import (
    TrackCandidate,
    TrackGenerationError,
    TrackGenerationSpec,
    generate_track_candidate,
)
from controller_learning.tracks.types import Track, TrackCapacity, track_array_bytes

REPORT_SCHEMA_VERSION = "controller-learning.track-capacity.v1"
DEFAULT_ARC_SPACINGS_M = (0.75, 1.0, 1.25)
PERCENTILES = (
    ("min", 0.0),
    ("p1", 1.0),
    ("p5", 5.0),
    ("p50", 50.0),
    ("p95", 95.0),
    ("p99", 99.0),
    ("p99_9", 99.9),
    ("max", 100.0),
)
CAPACITY_HEADROOM_FRACTION = 0.05
TRACK_POINT_ROUNDING = 64
CHECKPOINT_ROUNDING = 8

CandidateValidator = Callable[[TrackCandidate], Any]

_RELEVANT_SOURCE_PATHS = (
    "controller_learning/tracks/types.py",
    "controller_learning/tracks/geometry.py",
    "controller_learning/tracks/generator.py",
    "controller_learning/tracks/validator.py",
    "controller_learning/tracks/capacity_benchmark.py",
    "scripts/benchmark_track_capacity.py",
)


def _load_candidate_validator() -> CandidateValidator | None:
    """Load the optional M3 validator without making it an import-time dependency."""

    module_name = "controller_learning.tracks.validator"
    try:
        module = importlib.import_module(module_name)
    except ModuleNotFoundError as error:
        if error.name == module_name:
            return None
        raise
    validator = getattr(module, "validate_track_candidate", None)
    if not callable(validator):
        raise RuntimeError(f"{module_name} does not export validate_track_candidate")
    return validator


def _validator_spec() -> dict[str, Any]:
    """Record the default validation contract used by the optional validator."""

    module = importlib.import_module("controller_learning.tracks.validator")
    spec_type = getattr(module, "TrackValidationSpec", None)
    if spec_type is None:
        raise RuntimeError("track validator does not export TrackValidationSpec")
    return _json_value(asdict(spec_type()))


def _json_value(value: Any) -> Any:
    """Convert NumPy and immutable container values to strict-JSON-compatible values."""

    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, Mapping):
        return {str(key): _json_value(item) for key, item in sorted(value.items())}
    if isinstance(value, (tuple, list)):
        return [_json_value(item) for item in value]
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return repr(value)


def _implementation_evidence() -> dict[str, Any]:
    """Record revision and content hashes without leaking local filesystem paths."""

    root = Path(__file__).resolve().parents[2]
    hashes = {
        relative: hashlib.sha256((root / relative).read_bytes()).hexdigest()
        for relative in _RELEVANT_SOURCE_PATHS
    }
    try:
        revision_process = subprocess.run(
            ("git", "rev-parse", "HEAD"),
            cwd=root,
            check=True,
            capture_output=True,
            text=True,
            timeout=10,
        )
        status_process = subprocess.run(
            ("git", "status", "--porcelain", "--", *_RELEVANT_SOURCE_PATHS),
            cwd=root,
            check=True,
            capture_output=True,
            text=True,
            timeout=10,
        )
        revision: str | None = revision_process.stdout.strip()
        relevant_source_clean: bool | None = not bool(status_process.stdout.strip())
    except (FileNotFoundError, subprocess.SubprocessError):
        revision = None
        relevant_source_clean = None
    return {
        "git_revision": revision,
        "relevant_source_clean": relevant_source_clean,
        "source_files_sha256": hashes,
    }


def _candidate_fingerprint(candidate: TrackCandidate) -> str:
    """Hash every deterministic candidate field used by later Track processing."""

    digest = hashlib.sha256()
    digest.update(str(candidate.seed).encode())
    digest.update(candidate.generator_version.encode())
    digest.update(np.float64(candidate.length_m).tobytes())
    digest.update(np.float64(candidate.width_m).tobytes())
    for name in (
        "control_points_m",
        "centerline_m",
        "left_boundary_m",
        "right_boundary_m",
        "tangent",
        "curvature_1pm",
        "cumulative_s_m",
        "checkpoint_center_m",
        "checkpoint_tangent",
        "checkpoint_s_m",
        "start_pose",
    ):
        array = np.ascontiguousarray(getattr(candidate, name))
        digest.update(name.encode())
        digest.update(str(array.shape).encode())
        digest.update(array.dtype.str.encode())
        digest.update(array.tobytes())
    return digest.hexdigest()


def _validation_signature(result: Any) -> dict[str, Any]:
    """Return the deterministic public portion of a validator result."""

    return {
        "valid": bool(result.valid),
        "reasons": [str(reason) for reason in result.reasons],
        "primary_reason": (None if result.primary_reason is None else str(result.primary_reason)),
        "metrics": _json_value(result.metrics),
    }


def _attempt_signature(seed: int, spec: TrackGenerationSpec, validator: CandidateValidator | None):
    """Generate and optionally validate one seed for reproducibility comparison."""

    try:
        candidate = generate_track_candidate(seed, spec)
    except TrackGenerationError as error:
        return {
            "stage": "generation",
            "reason": error.reason,
            "context": _json_value(error.context),
        }
    signature: dict[str, Any] = {
        "stage": "candidate",
        "fingerprint": _candidate_fingerprint(candidate),
    }
    if validator is not None:
        signature["validation"] = _validation_signature(validator(candidate))
    return signature


def _percentile_summary(values: Sequence[float | int]) -> dict[str, float] | None:
    if not values:
        return None
    array = np.asarray(values, dtype=np.float64)
    quantiles = np.percentile(
        array,
        [percentile for _, percentile in PERCENTILES],
        method="linear",
    )
    return {name: float(value) for (name, _), value in zip(PERCENTILES, quantiles, strict=True)}


def _statistics(records: Sequence[dict[str, float | int]]) -> dict[str, Any]:
    return {
        "count": len(records),
        "point_count": _percentile_summary([record["point_count"] for record in records]),
        "checkpoint_count": _percentile_summary([record["checkpoint_count"] for record in records]),
        "length_m": _percentile_summary([record["length_m"] for record in records]),
        "max_abs_curvature_1pm": _percentile_summary(
            [record["max_abs_curvature_1pm"] for record in records]
        ),
    }


def _candidate_record(candidate: TrackCandidate) -> dict[str, float | int]:
    return {
        "point_count": candidate.point_count,
        "checkpoint_count": candidate.checkpoint_count,
        "length_m": float(candidate.length_m),
        "max_abs_curvature_1pm": float(np.max(np.abs(candidate.curvature_1pm))),
    }


def _round_up(value: float, quantum: int) -> int:
    return int(math.ceil(value / quantum) * quantum)


def _theoretical_capacity(spec: TrackGenerationSpec) -> dict[str, Any]:
    """Derive guaranteed capacities from the permitted maximum track length."""

    full_steps = math.floor(spec.max_length_m / spec.arc_spacing_m)
    remainder_m = spec.max_length_m - full_steps * spec.arc_spacing_m
    if remainder_m <= 1.0e-10 or remainder_m < spec.arc_spacing_m * spec.tail_merge_fraction:
        track_points = full_steps + 1
    else:
        track_points = full_steps + 2
    checkpoints = math.ceil(spec.max_length_m / spec.checkpoint_spacing_m)
    return {
        "max_length_m": spec.max_length_m,
        "arc_spacing_m": spec.arc_spacing_m,
        "checkpoint_spacing_m": spec.checkpoint_spacing_m,
        "max_track_points_required": track_points,
        "max_checkpoints_required": checkpoints,
        "track_point_formula": (
            "mirror fixed-spacing tail merge at max_length_m, including explicit closure"
        ),
        "checkpoint_formula": "ceil(max_length_m / checkpoint_spacing_m)",
    }


def _selected_capacity(
    theoretical: Mapping[str, Any], records: Sequence[dict[str, float | int]]
) -> dict[str, Any]:
    observed_points = max((int(record["point_count"]) for record in records), default=0)
    observed_checkpoints = max((int(record["checkpoint_count"]) for record in records), default=0)
    base_points = max(int(theoretical["max_track_points_required"]), observed_points)
    base_checkpoints = max(int(theoretical["max_checkpoints_required"]), observed_checkpoints)
    selected_points = _round_up(
        base_points * (1.0 + CAPACITY_HEADROOM_FRACTION), TRACK_POINT_ROUNDING
    )
    selected_checkpoints = _round_up(
        base_checkpoints * (1.0 + CAPACITY_HEADROOM_FRACTION), CHECKPOINT_ROUNDING
    )
    return {
        "max_track_points": selected_points,
        "max_checkpoints": selected_checkpoints,
        "base_track_points": base_points,
        "base_checkpoints": base_checkpoints,
        "observed_max_track_points": observed_points,
        "observed_max_checkpoints": observed_checkpoints,
        "headroom_fraction": CAPACITY_HEADROOM_FRACTION,
        "track_point_rounding": TRACK_POINT_ROUNDING,
        "checkpoint_rounding": CHECKPOINT_ROUNDING,
        "rule": "max(theoretical_required, observed_max), add 5%, then round upward",
    }


def _memory_probe_track(capacity: TrackCapacity) -> Track:
    """Build a minimal valid Track so memory uses the production byte counter."""

    point_shape = (capacity.max_track_points,)
    checkpoint_shape = (capacity.max_checkpoints,)
    centerline = np.zeros((*point_shape, 2), dtype=np.float32)
    centerline[:4] = ((0.0, 0.0), (1.0, 0.0), (0.0, 1.0), (0.0, 0.0))
    tangent = np.zeros_like(centerline)
    tangent[:4] = ((1.0, 0.0), (-1.0, 1.0), (0.0, -1.0), (1.0, 0.0))
    cumulative = np.zeros(point_shape, dtype=np.float32)
    cumulative[:4] = (0.0, 1.0, 2.0, 3.0)
    checkpoint_tangent = np.zeros((*checkpoint_shape, 2), dtype=np.float32)
    checkpoint_tangent[0] = (1.0, 0.0)
    checkpoint_s = np.zeros(checkpoint_shape, dtype=np.float32)
    checkpoint_s[0] = 3.0
    return Track(
        seed=0,
        generator_version="capacity-memory-probe-v1",
        centerline_m=centerline,
        left_boundary_m=centerline,
        right_boundary_m=centerline,
        tangent=tangent,
        curvature_1pm=np.zeros(point_shape, dtype=np.float32),
        cumulative_s_m=cumulative,
        track_mask=np.arange(capacity.max_track_points) < 4,
        checkpoint_center_m=np.zeros((*checkpoint_shape, 2), dtype=np.float32),
        checkpoint_tangent=checkpoint_tangent,
        checkpoint_s_m=checkpoint_s,
        checkpoint_mask=np.arange(capacity.max_checkpoints) < 1,
        start_pose=np.zeros(3, dtype=np.float32),
        point_count=4,
        checkpoint_count=1,
        length_m=3.0,
        width_m=7.0,
    )


def _memory_estimates(selection: Mapping[str, Any]) -> dict[str, Any]:
    capacity = TrackCapacity(
        max_track_points=int(selection["max_track_points"]),
        max_checkpoints=int(selection["max_checkpoints"]),
    )
    track_field_arrays = track_array_bytes(_memory_probe_track(capacity))
    # ``TrackBatch`` materializes the host scalar metadata as five device arrays:
    # seed, point_count, checkpoint_count, length_m, and width_m.
    runtime_scalars = 5 * np.dtype(np.float32).itemsize
    per_track = track_field_arrays + runtime_scalars
    worlds = 1024
    pool_size = 10_000
    return {
        "method": "Track arrays measured by track_array_bytes plus TrackBatch scalar arrays",
        "scope": "all runtime numerical TrackBatch leaves; generator_version is host metadata",
        "track_field_array_bytes_per_track": track_field_arrays,
        "runtime_scalar_array_bytes_per_track": runtime_scalars,
        "bytes_per_track": per_track,
        "bytes_1024_world_batch": per_track * worlds,
        "bytes_10000_track_pool": per_track * pool_size,
        "mib_per_track": per_track / (1024**2),
        "mib_1024_world_batch": per_track * worlds / (1024**2),
        "mib_10000_track_pool": per_track * pool_size / (1024**2),
    }


def _reproducibility_seeds(seed_start: int, seed_count: int, sample_count: int) -> list[int]:
    count = min(seed_count, sample_count)
    if count == seed_count:
        return list(range(seed_start, seed_start + seed_count))
    offsets = np.linspace(0, seed_count - 1, count, dtype=np.int64)
    return [seed_start + int(offset) for offset in np.unique(offsets)]


def _record_rejection(
    *,
    seed: int,
    stage: str,
    primary_reason: str,
    all_reasons: Sequence[str],
    stage_counts: Counter[str],
    primary_counts: Counter[str],
    all_counts: Counter[str],
    sample_seeds: defaultdict[str, list[int]],
) -> None:
    stage_counts[stage] += 1
    primary_counts[primary_reason] += 1
    for reason in all_reasons:
        all_counts[reason] += 1
    if len(sample_seeds[primary_reason]) < 10:
        sample_seeds[primary_reason].append(seed)


def _run_spacing(
    *,
    spec: TrackGenerationSpec,
    seed_start: int,
    seed_count: int,
    validator: CandidateValidator | None,
    reproducibility_sample_count: int,
) -> dict[str, Any]:
    generated_records: list[dict[str, float | int]] = []
    accepted_records: list[dict[str, float | int]] = []
    stage_counts: Counter[str] = Counter()
    primary_counts: Counter[str] = Counter()
    all_counts: Counter[str] = Counter()
    sample_seeds: defaultdict[str, list[int]] = defaultdict(list)
    reference_signatures: dict[int, dict[str, Any]] = {}
    checked_seeds = _reproducibility_seeds(seed_start, seed_count, reproducibility_sample_count)
    checked_seed_set = set(checked_seeds)

    generation_failure_count = 0
    validation_rejection_count = 0
    for seed in range(seed_start, seed_start + seed_count):
        try:
            candidate = generate_track_candidate(seed, spec)
        except TrackGenerationError as error:
            generation_failure_count += 1
            _record_rejection(
                seed=seed,
                stage="generation",
                primary_reason=error.reason,
                all_reasons=(error.reason,),
                stage_counts=stage_counts,
                primary_counts=primary_counts,
                all_counts=all_counts,
                sample_seeds=sample_seeds,
            )
            if seed in checked_seed_set:
                reference_signatures[seed] = {
                    "stage": "generation",
                    "reason": error.reason,
                    "context": _json_value(error.context),
                }
            continue

        record = _candidate_record(candidate)
        generated_records.append(record)
        validation_signature = None
        if validator is not None:
            validation_result = validator(candidate)
            validation_signature = _validation_signature(validation_result)
            if validation_result.valid:
                accepted_records.append(record)
            else:
                validation_rejection_count += 1
                reasons = tuple(str(reason) for reason in validation_result.reasons)
                primary_reason = (
                    str(validation_result.primary_reason)
                    if validation_result.primary_reason is not None
                    else reasons[0]
                    if reasons
                    else "validation_rejected_without_reason"
                )
                all_reasons = reasons or (primary_reason,)
                _record_rejection(
                    seed=seed,
                    stage="validation",
                    primary_reason=primary_reason,
                    all_reasons=all_reasons,
                    stage_counts=stage_counts,
                    primary_counts=primary_counts,
                    all_counts=all_counts,
                    sample_seeds=sample_seeds,
                )

        if seed in checked_seed_set:
            reference_signatures[seed] = {
                "stage": "candidate",
                "fingerprint": _candidate_fingerprint(candidate),
            }
            if validation_signature is not None:
                reference_signatures[seed]["validation"] = validation_signature

    mismatches = [
        seed
        for seed in checked_seeds
        if reference_signatures[seed] != _attempt_signature(seed, spec, validator)
    ]
    theoretical = _theoretical_capacity(spec)
    selection = _selected_capacity(theoretical, generated_records)
    validation_available = validator is not None
    return {
        "arc_spacing_m": spec.arc_spacing_m,
        "generator_spec": _json_value(asdict(spec)),
        "generation": {
            "attempted_count": seed_count,
            "succeeded_count": len(generated_records),
            "failed_count": generation_failure_count,
        },
        "validation": {
            "available": validation_available,
            "validated_count": len(generated_records) if validation_available else 0,
            "accepted_count": len(accepted_records) if validation_available else None,
            "rejected_count": validation_rejection_count if validation_available else None,
        },
        "rejections": {
            "total": generation_failure_count + validation_rejection_count,
            "by_stage": {
                "generation": stage_counts["generation"],
                "validation": stage_counts["validation"],
            },
            "primary_reason_counts": dict(sorted(primary_counts.items())),
            "all_reason_counts": dict(sorted(all_counts.items())),
            "sample_seeds_by_primary_reason": {
                reason: seeds for reason, seeds in sorted(sample_seeds.items())
            },
        },
        "statistics": {
            "generated_candidates": _statistics(generated_records),
            "validated_accepted_candidates": (
                _statistics(accepted_records) if validation_available else None
            ),
        },
        "reproducibility": {
            "checked_seeds": checked_seeds,
            "checked_count": len(checked_seeds),
            "passed": not mismatches,
            "mismatch_seeds": mismatches,
        },
        "theoretical_capacity": theoretical,
        "selected_capacity_candidate": selection,
        "memory_estimates": _memory_estimates(selection),
    }


def run_track_capacity_benchmark(
    *,
    seed_start: int = 0,
    seed_count: int = 10_000,
    arc_spacings_m: Sequence[float] = DEFAULT_ARC_SPACINGS_M,
    base_spec: TrackGenerationSpec | None = None,
    reproducibility_sample_count: int = 8,
) -> dict[str, Any]:
    """Sweep a contiguous seed range and return deterministic capacity evidence."""

    if seed_count <= 0:
        raise ValueError("seed_count must be positive")
    if not 0 <= seed_start <= np.iinfo(np.uint32).max:
        raise ValueError("seed_start must fit in uint32")
    if seed_start + seed_count - 1 > np.iinfo(np.uint32).max:
        raise ValueError("the requested seed range exceeds uint32")
    spacings = tuple(float(spacing) for spacing in arc_spacings_m)
    if not spacings or any(not np.isfinite(spacing) or spacing <= 0.0 for spacing in spacings):
        raise ValueError("arc_spacings_m must contain finite positive values")
    if len(set(spacings)) != len(spacings):
        raise ValueError("arc_spacings_m cannot contain duplicates")
    if reproducibility_sample_count <= 0:
        raise ValueError("reproducibility_sample_count must be positive")

    spec = TrackGenerationSpec() if base_spec is None else base_spec
    validator = _load_candidate_validator()
    spacing_results = [
        _run_spacing(
            spec=replace(spec, arc_spacing_m=spacing),
            seed_start=seed_start,
            seed_count=seed_count,
            validator=validator,
            reproducibility_sample_count=reproducibility_sample_count,
        )
        for spacing in spacings
    ]
    return {
        "schema_version": REPORT_SCHEMA_VERSION,
        "benchmark_version": "v0.1",
        "implementation": _implementation_evidence(),
        "protocol": {
            "seed_range": {
                "start_inclusive": seed_start,
                "end_exclusive": seed_start + seed_count,
                "count": seed_count,
                "contiguous": True,
            },
            "arc_spacings_m": list(spacings),
            "percentiles": [name for name, _ in PERCENTILES],
            "percentile_method": "linear",
            "reproducibility_sample_count": min(seed_count, reproducibility_sample_count),
        },
        "validator": {
            "module": "controller_learning.tracks.validator",
            "function": "validate_track_candidate",
            "validation_available": validator is not None,
            "validation_spec": _validator_spec() if validator is not None else None,
            "unavailable_means_unvalidated_not_accepted": True,
        },
        "spacing_results": spacing_results,
    }


def write_track_capacity_report(
    output: Path,
    *,
    seed_start: int = 0,
    seed_count: int = 10_000,
    arc_spacings_m: Sequence[float] = DEFAULT_ARC_SPACINGS_M,
    base_spec: TrackGenerationSpec | None = None,
    reproducibility_sample_count: int = 8,
) -> dict[str, Any]:
    """Run the protocol and write stable strict JSON without host-specific metadata."""

    report = run_track_capacity_benchmark(
        seed_start=seed_start,
        seed_count=seed_count,
        arc_spacings_m=arc_spacings_m,
        base_spec=base_spec,
        reproducibility_sample_count=reproducibility_sample_count,
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(report, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    return report


__all__ = [
    "DEFAULT_ARC_SPACINGS_M",
    "REPORT_SCHEMA_VERSION",
    "run_track_capacity_benchmark",
    "write_track_capacity_report",
]
