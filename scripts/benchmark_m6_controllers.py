"""Run the formal M6 PID and MPC Controller benchmark on official GPU assets.

The formal workload is intentionally not configurable from the command line. PID is evaluated on
Level 0 and the first ten validation Tracks; MPC is evaluated on Level 0 and all one hundred
validation Tracks. The Test split is never loaded by this script.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import platform
import re
import subprocess
import sys
import tempfile
from collections.abc import Callable, Mapping, Sequence
from dataclasses import asdict, dataclass, is_dataclass
from datetime import UTC, datetime
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from typing import Any, TypeAlias

import numpy as np

from controller_learning.config import ProjectConfig, load_project_config
from controller_learning.envs.race_core import RaceTermination
from controller_learning.evaluation import (
    ControllerEvaluation,
    evaluate_track_batch,
    summarize_compute_times,
)
from controller_learning.tracks.assets import (
    TrackAssetManifest,
    load_manifest_track_batch,
    sha256_file,
)
from controller_learning.tracks.official_assets import (
    official_track_asset_directory,
    official_track_split_spec,
    validate_official_manifest,
)
from controller_learning.tracks.types import TrackBatch

REPORT_SCHEMA_VERSION = "controller-learning.m6-controllers.v1"
PROTOCOL_VERSION = "m6-pid-mpc-gpu-v1"
FORMAL_BACKEND = "mjx_warp"
FORMAL_PID_VALIDATION_TRACKS = 10
FORMAL_MPC_VALIDATION_TRACKS = 100
FORMAL_LEVEL0_TRACKS = 1
REALTIME_P99_LIMIT_S = 0.05
REALTIME_MISS_RATE_LIMIT = 0.01
FORMAL_INIT_TIMEOUT_S = 30.0
DEFAULT_OUTPUT = Path("benchmarks/v0.1/m6_controller_report.json")
PROJECT_ROOT = Path(__file__).resolve().parents[1]
_SHA256_PATTERN = re.compile(r"^[0-9a-f]{64}$")
_GPU_UUID_PATTERN = re.compile(
    r"\bGPU-[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}\b",
    flags=re.IGNORECASE,
)
_WINDOWS_ABSOLUTE_PATTERN = re.compile(r"^[A-Za-z]:[\\/]")

_NON_PACKAGE_SOURCE_PATHS = (
    "pixi.lock",
    "pyproject.toml",
    "configs/benchmark.toml",
    "configs/levels/level0.toml",
    "configs/levels/level1.toml",
    "configs/track.toml",
    "configs/vehicle.toml",
    "controller_learning/assets/tracks/v0.1/level0.json",
    "controller_learning/assets/tracks/v0.1/level0.npz",
    "controller_learning/assets/tracks/v0.1/validation.json",
    "controller_learning/assets/tracks/v0.1/validation.npz",
    "controller_learning/assets/vehicle/car.xml",
    "controllers/pid/config.toml",
    "controllers/pid/controller.py",
    "controllers/pid/helpers.py",
    "controllers/mpc/config.toml",
    "controllers/mpc/controller.py",
    "controllers/mpc/helpers.py",
    "controllers/mpc/solver.py",
    "scripts/benchmark_m6_controllers.py",
)
_PACKAGE_SOURCE_PATHS = tuple(
    path.relative_to(PROJECT_ROOT).as_posix()
    for path in sorted((PROJECT_ROOT / "controller_learning").rglob("*.py"))
)
RELEVANT_SOURCE_PATHS = tuple(sorted((*_NON_PACKAGE_SOURCE_PATHS, *_PACKAGE_SOURCE_PATHS)))


@dataclass(frozen=True, slots=True)
class BenchmarkOptions:
    """The report destination is the only configurable formal runtime option."""

    output: Path = DEFAULT_OUTPUT

    def __post_init__(self) -> None:
        output = Path(self.output)
        if output.suffix != ".json":
            raise ValueError("output must use the .json suffix")
        object.__setattr__(self, "output", output)


@dataclass(frozen=True, slots=True)
class EvaluationAssets:
    """The two verified Track batches that the M6 protocol is permitted to read."""

    level0_manifest: TrackAssetManifest
    level0_batch: TrackBatch
    validation_manifest: TrackAssetManifest
    validation_batch: TrackBatch
    evidence: Mapping[str, Any]


AssetLoader: TypeAlias = Callable[[ProjectConfig, Path], EvaluationAssets]
ControllerEvaluator: TypeAlias = Callable[..., ControllerEvaluation]
SnapshotLoader: TypeAlias = Callable[[Path], Mapping[str, Any]]
RuntimeLoader: TypeAlias = Callable[[], Mapping[str, Any]]


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT,
        help=f"Strict JSON report path (default: {DEFAULT_OUTPUT})",
    )
    return parser


def _parse_args(argv: list[str] | None = None) -> BenchmarkOptions:
    values = _build_parser().parse_args(argv)
    try:
        return BenchmarkOptions(output=values.output)
    except ValueError as error:
        _build_parser().error(str(error))


def _json_value(value: Any) -> Any:
    """Convert dataclasses, NumPy values, and paths to strict-JSON-compatible builtins."""

    if is_dataclass(value) and not isinstance(value, type):
        return _json_value(asdict(value))
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, Path):
        return value.as_posix()
    if isinstance(value, Mapping):
        return {str(key): _json_value(item) for key, item in sorted(value.items())}
    if isinstance(value, (tuple, list)):
        return [_json_value(item) for item in value]
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    raise TypeError(f"value of type {type(value).__name__} is not JSON serializable")


def write_strict_json(path: Path, payload: Mapping[str, Any]) -> None:
    """Atomically write indented JSON while rejecting NaN and infinity."""

    destination = path.expanduser().resolve()
    serialized = (
        json.dumps(
            _json_value(payload),
            indent=2,
            sort_keys=True,
            allow_nan=False,
        )
        + "\n"
    )
    destination.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{destination.name}.",
        suffix=".tmp",
        dir=destination.parent,
        text=True,
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as file:
            file.write(serialized)
            file.flush()
            os.fsync(file.fileno())
        os.replace(temporary, destination)
        destination.chmod(0o644)
    finally:
        temporary.unlink(missing_ok=True)


def _sha256_text(values: Sequence[str]) -> str:
    digest = hashlib.sha256()
    for value in values:
        digest.update(value.encode("ascii"))
        digest.update(b"\n")
    return digest.hexdigest()


def _git(project_root: Path, *arguments: str) -> str | None:
    try:
        completed = subprocess.run(
            ("git", *arguments),
            cwd=project_root,
            check=True,
            capture_output=True,
            text=True,
            timeout=15,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    return completed.stdout.strip()


def _source_snapshot(project_root: Path) -> dict[str, Any]:
    missing = [path for path in RELEVANT_SOURCE_PATHS if not (project_root / path).is_file()]
    if missing:
        raise FileNotFoundError(f"M6 benchmark source inputs are missing: {', '.join(missing)}")
    hashes = {path: sha256_file(project_root / path) for path in RELEVANT_SOURCE_PATHS}
    status = _git(project_root, "status", "--porcelain", "--", *RELEVANT_SOURCE_PATHS)
    return {
        "git_revision": _git(project_root, "rev-parse", "HEAD"),
        "relevant_source_clean": None if status is None else not bool(status),
        "source_files_sha256": hashes,
    }


def _require_source_preflight(snapshot: Mapping[str, Any]) -> None:
    """Reject an unreproducible checkout before starting the expensive formal workload."""

    revision = snapshot.get("git_revision")
    if not isinstance(revision, str) or not revision:
        raise RuntimeError("formal M6 evaluation requires a readable non-empty Git revision")
    if snapshot.get("relevant_source_clean") is not True:
        raise RuntimeError("formal M6 evaluation requires a clean relevant source checkout")
    hashes = snapshot.get("source_files_sha256")
    if not isinstance(hashes, Mapping) or set(hashes) != set(RELEVANT_SOURCE_PATHS):
        raise RuntimeError("formal M6 evaluation source snapshot does not cover every input")
    if any(
        not isinstance(value, str) or _SHA256_PATTERN.fullmatch(value) is None
        for value in hashes.values()
    ):
        raise RuntimeError("formal M6 evaluation source snapshot contains an invalid SHA-256")


def _package_version(package: str) -> str | None:
    try:
        return version(package)
    except PackageNotFoundError:
        return None


def _nvidia_inventory() -> tuple[list[dict[str, Any]], str | None]:
    command = (
        "nvidia-smi",
        "--query-gpu=index,name,driver_version,memory.total",
        "--format=csv,noheader,nounits",
    )
    try:
        completed = subprocess.run(
            command,
            check=True,
            capture_output=True,
            text=True,
            timeout=15,
        )
    except (OSError, subprocess.SubprocessError) as error:
        return [], f"{type(error).__name__}: nvidia-smi inventory unavailable"
    inventory: list[dict[str, Any]] = []
    for line in completed.stdout.splitlines():
        fields = [field.strip() for field in line.split(",")]
        if len(fields) != 4:
            continue
        try:
            inventory.append(
                {
                    "index": int(fields[0]),
                    "name": fields[1],
                    "driver_version": fields[2],
                    "memory_total_mib": float(fields[3]),
                }
            )
        except ValueError:
            continue
    return inventory, None if inventory else "nvidia-smi returned no parseable GPUs"


def _runtime_evidence() -> dict[str, Any]:
    """Collect versioned runtime and sanitized GPU identity evidence."""

    import jax

    try:
        gpu_devices = jax.devices("gpu")
        gpu_error = None
    except RuntimeError as error:
        gpu_devices = []
        gpu_error = f"{type(error).__name__}: no JAX GPU device available"
    inventory, inventory_error = _nvidia_inventory()
    device = gpu_devices[0] if gpu_devices else None
    return {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "packages": {
            package: _package_version(package)
            for package in (
                "casadi",
                "controller-learning",
                "jax",
                "mujoco",
                "mujoco-mjx",
                "numpy",
            )
        },
        "jax_device": (
            None
            if device is None
            else {
                "id": int(device.id),
                "platform": str(device.platform),
                "device_kind": str(device.device_kind),
            }
        ),
        "jax_gpu_error": gpu_error,
        "nvidia_inventory": inventory,
        "nvidia_inventory_error": inventory_error,
    }


def _asset_evidence(
    manifest_path: Path,
    manifest: TrackAssetManifest,
    batch: TrackBatch,
) -> dict[str, Any]:
    geometry_hashes = tuple(record.geometry_sha256 for record in manifest.tracks)
    asset_path = manifest_path.parent / manifest.asset_file
    return {
        "split": manifest.split,
        "manifest_file": manifest_path.name,
        "asset_file": manifest.asset_file,
        "track_count": manifest.track_count,
        "loaded_track_count": int(batch.seed.shape[0]),
        "manifest_sha256": sha256_file(manifest_path),
        "manifest_asset_sha256": manifest.asset_sha256,
        "asset_file_sha256": sha256_file(asset_path),
        "track_ids": [int(value) for value in batch.seed],
        "geometry_hash_count": len(geometry_hashes),
        "geometry_hashes_sha256": _sha256_text(geometry_hashes),
        "readback_verified": True,
    }


def _load_evaluation_assets(config: ProjectConfig, project_root: Path) -> EvaluationAssets:
    """Load only Level 0 and Validation through their strict official manifests."""

    del project_root  # Package-relative official assets also work from an installed wheel.
    directory = official_track_asset_directory(config.benchmark.version)
    loaded: dict[str, tuple[TrackAssetManifest, TrackBatch]] = {}
    evidence: dict[str, Any] = {
        "loaded_splits": ["level0", "validation"],
        "test_split_accessed": False,
    }
    for split in ("level0", "validation"):
        spec = official_track_split_spec(split)
        manifest_path = directory / spec.manifest_file
        manifest, batch = load_manifest_track_batch(manifest_path)
        validate_official_manifest(config, manifest)
        loaded[split] = (manifest, batch)
        evidence[split] = _asset_evidence(manifest_path, manifest, batch)

    level0_manifest, level0_batch = loaded["level0"]
    validation_manifest, validation_batch = loaded["validation"]
    return EvaluationAssets(
        level0_manifest=level0_manifest,
        level0_batch=level0_batch,
        validation_manifest=validation_manifest,
        validation_batch=validation_batch,
        evidence=evidence,
    )


def _first_rows(batch: TrackBatch, count: int) -> TrackBatch:
    if not 0 < count <= int(batch.seed.shape[0]):
        raise ValueError("row count must be positive and cannot exceed the Track batch")
    arrays = []
    for array in batch:
        selected = np.array(array[:count], copy=True)
        selected.setflags(write=False)
        arrays.append(selected)
    return TrackBatch(*arrays)


def _controller_config_evidence(project_root: Path, name: str) -> dict[str, Any]:
    relative_directory = Path("controllers") / name
    relative_config = relative_directory / "config.toml"
    config_path = project_root / relative_config
    if not config_path.is_file():
        raise FileNotFoundError(f"missing Controller config: {relative_config.as_posix()}")
    return {
        "directory": relative_directory.as_posix(),
        "config_file": relative_config.as_posix(),
        "config_sha256": sha256_file(config_path),
    }


def _evaluation_payload(
    evaluation: ControllerEvaluation,
    *,
    controller_directory: str,
    project_root: Path,
    expected_level_id: int,
) -> dict[str, Any]:
    if not isinstance(evaluation, ControllerEvaluation):
        raise TypeError("Controller evaluator must return ControllerEvaluation")
    expected_path = (project_root / controller_directory).resolve()
    actual_path = Path(evaluation.controller_directory).expanduser()
    if not actual_path.is_absolute():
        actual_path = project_root / actual_path
    if actual_path.resolve() != expected_path:
        raise ValueError(
            "Controller evaluator returned a result for an unexpected Controller directory"
        )
    if evaluation.level_id != expected_level_id:
        raise ValueError("Controller evaluator returned a result for an unexpected Level")
    if evaluation.backend != FORMAL_BACKEND:
        raise ValueError("Controller evaluator returned a result for a non-formal backend")
    payload = _json_value(evaluation)
    if not isinstance(payload, dict):  # pragma: no cover - dataclass conversion invariant
        raise AssertionError("ControllerEvaluation must serialize to an object")
    payload["controller_directory"] = controller_directory
    return payload


def _combined_timing(evaluations: Sequence[ControllerEvaluation]) -> dict[str, Any]:
    samples = tuple(
        sample
        for evaluation in evaluations
        for episode in evaluation.episodes
        for sample in episode.compute_times_s
    )
    return _json_value(summarize_compute_times(samples, deadline_s=REALTIME_P99_LIMIT_S))


def _realtime_qualification(timing: Mapping[str, Any]) -> dict[str, Any]:
    p99_s = timing.get("p99_s")
    miss_rate = timing.get("deadline_miss_rate")
    eligible = (
        isinstance(p99_s, (int, float))
        and not isinstance(p99_s, bool)
        and isinstance(miss_rate, (int, float))
        and not isinstance(miss_rate, bool)
        and float(p99_s) <= REALTIME_P99_LIMIT_S
        and float(miss_rate) <= REALTIME_MISS_RATE_LIMIT
    )
    return {
        "p99_limit_s": REALTIME_P99_LIMIT_S,
        "deadline_miss_rate_limit": REALTIME_MISS_RATE_LIMIT,
        "eligible": eligible,
        "required_for_m6_pass": False,
    }


def _controller_result(
    level0: ControllerEvaluation,
    validation: ControllerEvaluation,
    *,
    directory: str,
    project_root: Path,
) -> dict[str, Any]:
    combined_timing = _combined_timing((level0, validation))
    return {
        "level0": _evaluation_payload(
            level0,
            controller_directory=directory,
            project_root=project_root,
            expected_level_id=0,
        ),
        "validation": _evaluation_payload(
            validation,
            controller_directory=directory,
            project_root=project_root,
            expected_level_id=1,
        ),
        "combined_timing": combined_timing,
        "realtime_qualification": _realtime_qualification(combined_timing),
    }


def _run_controller_evaluations(
    config: ProjectConfig,
    assets: EvaluationAssets,
    project_root: Path,
    evaluator: ControllerEvaluator,
) -> dict[str, Any]:
    generator_version0 = assets.level0_manifest.generator_version
    validation_version = assets.validation_manifest.generator_version
    pid_validation = _first_rows(assets.validation_batch, FORMAL_PID_VALIDATION_TRACKS)
    reset0 = np.arange(FORMAL_LEVEL0_TRACKS, dtype=np.uint32)
    pid_reset = np.arange(FORMAL_PID_VALIDATION_TRACKS, dtype=np.uint32)
    mpc_reset = np.arange(FORMAL_MPC_VALIDATION_TRACKS, dtype=np.uint32)
    pid_directory = project_root / "controllers/pid"
    mpc_directory = project_root / "controllers/mpc"

    pid_level0 = evaluator(
        config,
        0,
        assets.level0_batch,
        generator_version0,
        pid_directory,
        FORMAL_BACKEND,
        reset_seeds=reset0,
    )
    pid_validation_result = evaluator(
        config,
        1,
        pid_validation,
        validation_version,
        pid_directory,
        FORMAL_BACKEND,
        reset_seeds=pid_reset,
    )
    mpc_level0 = evaluator(
        config,
        0,
        assets.level0_batch,
        generator_version0,
        mpc_directory,
        FORMAL_BACKEND,
        reset_seeds=reset0,
    )
    mpc_validation_result = evaluator(
        config,
        1,
        assets.validation_batch,
        validation_version,
        mpc_directory,
        FORMAL_BACKEND,
        reset_seeds=mpc_reset,
    )
    return {
        "pid": _controller_result(
            pid_level0,
            pid_validation_result,
            directory="controllers/pid",
            project_root=project_root,
        ),
        "mpc": _controller_result(
            mpc_level0,
            mpc_validation_result,
            directory="controllers/mpc",
            project_root=project_root,
        ),
    }


def _mapping(value: object) -> Mapping[str, Any]:
    return value if isinstance(value, Mapping) else {}


def _episodes(report: Mapping[str, Any], controller: str, split: str) -> list[Any]:
    result = _evaluation_result(report, controller, split)
    episodes = result.get("episodes")
    return episodes if isinstance(episodes, list) else []


def _evaluation_result(
    report: Mapping[str, Any],
    controller: str,
    split: str,
) -> Mapping[str, Any]:
    evaluations = _mapping(report.get("evaluations"))
    controller_result = _mapping(evaluations.get(controller))
    return _mapping(controller_result.get(split))


def _track_count(report: Mapping[str, Any], controller: str, split: str) -> object:
    return _evaluation_result(report, controller, split).get("track_count")


def _success_rate(report: Mapping[str, Any], controller: str, split: str) -> object:
    return _evaluation_result(report, controller, split).get("success_rate")


def _episode_values(episodes: Sequence[object], key: str) -> list[Any]:
    return [episode.get(key) if isinstance(episode, Mapping) else None for episode in episodes]


def _normal_episode(episode: object) -> bool:
    if not isinstance(episode, Mapping):
        return False
    terminated = episode.get("terminated")
    truncated = episode.get("truncated")
    return (
        type(terminated) is bool
        and type(truncated) is bool
        and terminated != truncated
        and isinstance(episode.get("steps"), int)
        and not isinstance(episode.get("steps"), bool)
        and int(episode["steps"]) > 0
    )


def _finite_number(value: object, *, minimum: float = 0.0) -> bool:
    return (
        isinstance(value, (int, float))
        and not isinstance(value, bool)
        and math.isfinite(float(value))
        and float(value) >= minimum
    )


def _close_number(actual: object, expected: float, *, tolerance: float = 1.0e-15) -> bool:
    return _finite_number(actual) and math.isclose(
        float(actual),
        expected,
        rel_tol=0.0,
        abs_tol=tolerance,
    )


def _evaluation_identity_findings(report: Mapping[str, Any]) -> list[str]:
    findings: list[str] = []
    for controller in ("pid", "mpc"):
        for split, level_id in (("level0", 0), ("validation", 1)):
            result = _evaluation_result(report, controller, split)
            prefix = f"{controller}.{split}"
            expected = {
                "backend": FORMAL_BACKEND,
                "level_id": level_id,
                "controller_directory": f"controllers/{controller}",
            }
            for key, value in expected.items():
                actual = result.get(key)
                if actual != value or (key == "level_id" and type(actual) is not int):
                    findings.append(f"{prefix}.{key}")
    return findings


def _aggregate_findings(report: Mapping[str, Any]) -> list[str]:
    findings: list[str] = []
    for controller in ("pid", "mpc"):
        for split in ("level0", "validation"):
            prefix = f"{controller}.{split}"
            result = _evaluation_result(report, controller, split)
            episodes = result.get("episodes")
            if not isinstance(episodes, list) or not episodes:
                findings.append(f"{prefix}.episodes")
                continue
            track_count = result.get("track_count")
            if type(track_count) is not int or track_count != len(episodes):
                findings.append(f"{prefix}.track_count")

            successes: list[bool] = []
            successful_laps: list[float] = []
            episode_schema_valid = True
            for index, episode in enumerate(episodes):
                if not isinstance(episode, Mapping) or type(episode.get("success")) is not bool:
                    findings.append(f"{prefix}.episodes[{index}].success")
                    episode_schema_valid = False
                    continue
                success = bool(episode["success"])
                successes.append(success)
                lap_time = episode.get("lap_time_s")
                if success:
                    if not _finite_number(lap_time, minimum=np.finfo(np.float64).tiny):
                        findings.append(f"{prefix}.episodes[{index}].lap_time_s")
                        episode_schema_valid = False
                    else:
                        successful_laps.append(float(lap_time))
                elif lap_time is not None:
                    findings.append(f"{prefix}.episodes[{index}].lap_time_s")
                    episode_schema_valid = False

            if not episode_schema_valid or len(successes) != len(episodes):
                continue
            success_count = sum(successes)
            success_rate = success_count / len(episodes)
            reported_success_count = result.get("success_count")
            if type(reported_success_count) is not int or reported_success_count != success_count:
                findings.append(f"{prefix}.success_count")
            if not _close_number(result.get("success_rate"), success_rate):
                findings.append(f"{prefix}.success_rate")
            mean_lap = result.get("mean_successful_lap_time_s")
            if successful_laps:
                expected_mean = float(np.mean(successful_laps, dtype=np.float64))
                if not _close_number(mean_lap, expected_mean, tolerance=1.0e-12):
                    findings.append(f"{prefix}.mean_successful_lap_time_s")
            elif mean_lap is not None:
                findings.append(f"{prefix}.mean_successful_lap_time_s")
    return findings


def _timing_summary_matches(summary: object, samples: Sequence[object]) -> bool:
    if not isinstance(summary, Mapping):
        return False
    try:
        expected = _json_value(
            summarize_compute_times(samples, deadline_s=REALTIME_P99_LIMIT_S)  # type: ignore[arg-type]
        )
    except (TypeError, ValueError):
        return False
    if not isinstance(expected, Mapping):  # pragma: no cover - dataclass conversion invariant
        raise AssertionError("TimingSummary must serialize to an object")
    integer_fields = ("sample_count", "deadline_miss_count")
    float_fields = (
        "deadline_s",
        "p50_s",
        "p95_s",
        "p99_s",
        "max_s",
        "deadline_miss_rate",
    )
    return (
        set(summary) == set(expected)
        and all(
            type(summary.get(field)) is int and summary.get(field) == expected[field]
            for field in integer_fields
        )
        and all(_close_number(summary.get(field), float(expected[field])) for field in float_fields)
    )


def _timing_findings(report: Mapping[str, Any]) -> list[str]:
    findings: list[str] = []
    evaluations = _mapping(report.get("evaluations"))
    for controller in ("pid", "mpc"):
        controller_result = _mapping(evaluations.get(controller))
        combined_samples: list[object] = []
        for split in ("level0", "validation"):
            result = _mapping(controller_result.get(split))
            episodes = result.get("episodes")
            if not isinstance(episodes, list) or not episodes:
                findings.append(f"{controller}.{split}.episodes")
                continue
            aggregate_samples: list[object] = []
            for index, episode in enumerate(episodes):
                if not isinstance(episode, Mapping):
                    findings.append(f"{controller}.{split}.episodes[{index}]")
                    continue
                steps = episode.get("steps")
                samples = episode.get("compute_times_s")
                if (
                    not isinstance(steps, int)
                    or isinstance(steps, bool)
                    or not isinstance(samples, list)
                    or len(samples) != steps
                ):
                    findings.append(f"{controller}.{split}.episodes[{index}].compute_times_s")
                    continue
                if not _timing_summary_matches(episode.get("compute_timing"), samples):
                    findings.append(f"{controller}.{split}.episodes[{index}].compute_timing")
                aggregate_samples.extend(samples)
            if not aggregate_samples or not _timing_summary_matches(
                result.get("compute_timing"), aggregate_samples
            ):
                findings.append(f"{controller}.{split}.compute_timing")
            combined_samples.extend(aggregate_samples)
        if not combined_samples or not _timing_summary_matches(
            controller_result.get("combined_timing"), combined_samples
        ):
            findings.append(f"{controller}.combined_timing")
    return findings


def _realtime_findings(report: Mapping[str, Any]) -> list[str]:
    findings: list[str] = []
    evaluations = _mapping(report.get("evaluations"))
    for controller in ("pid", "mpc"):
        controller_result = _mapping(evaluations.get(controller))
        expected = _realtime_qualification(_mapping(controller_result.get("combined_timing")))
        if controller_result.get("realtime_qualification") != expected:
            findings.append(f"{controller}.realtime_qualification")
    return findings


def _init_timeout_findings(report: Mapping[str, Any]) -> list[str]:
    findings: list[str] = []
    for controller in ("pid", "mpc"):
        for split in ("level0", "validation"):
            for index, episode in enumerate(_episodes(report, controller, split)):
                value = (
                    episode.get("controller_init_time_s") if isinstance(episode, Mapping) else None
                )
                if not _finite_number(value) or float(value) > FORMAL_INIT_TIMEOUT_S:
                    findings.append(
                        f"{controller}.{split}.episodes[{index}].controller_init_time_s"
                    )
    return findings


def _privacy_findings(value: object) -> dict[str, list[str]]:
    absolute_paths: list[str] = []
    gpu_uuids: list[str] = []

    def visit(item: object) -> None:
        if isinstance(item, Mapping):
            for nested in item.values():
                visit(nested)
        elif isinstance(item, (list, tuple)):
            for nested in item:
                visit(nested)
        elif isinstance(item, str):
            if item.startswith("/") or _WINDOWS_ABSOLUTE_PATTERN.match(item):
                absolute_paths.append(item)
            gpu_uuids.extend(match.group(0) for match in _GPU_UUID_PATTERN.finditer(item))

    visit(value)
    return {
        "absolute_paths": sorted(set(absolute_paths)),
        "gpu_uuids": sorted(set(gpu_uuids)),
    }


def _check(identifier: str, passed: bool, observed: Any, expected: Any) -> dict[str, Any]:
    return {
        "id": identifier,
        "passed": bool(passed),
        "observed": _json_value(observed),
        "expected": _json_value(expected),
    }


def evaluate_report_gates(report: Mapping[str, Any]) -> list[dict[str, Any]]:
    """Independently recompute every formal M6 pass gate from report evidence."""

    protocol = _mapping(report.get("protocol"))
    assets = _mapping(report.get("assets"))
    runtime = _mapping(report.get("runtime"))
    source = _mapping(report.get("source_evidence"))
    before = _mapping(source.get("before"))
    after = _mapping(source.get("after"))
    checks = [
        _check(
            "report.schema",
            report.get("schema_version") == REPORT_SCHEMA_VERSION,
            report.get("schema_version"),
            REPORT_SCHEMA_VERSION,
        ),
        _check(
            "protocol.version",
            report.get("protocol_version") == PROTOCOL_VERSION,
            report.get("protocol_version"),
            PROTOCOL_VERSION,
        ),
        _check(
            "protocol.fixed_workload",
            protocol.get("backend") == FORMAL_BACKEND
            and protocol.get("level0_track_count") == FORMAL_LEVEL0_TRACKS
            and protocol.get("pid_validation_track_count") == FORMAL_PID_VALIDATION_TRACKS
            and protocol.get("mpc_validation_track_count") == FORMAL_MPC_VALIDATION_TRACKS
            and protocol.get("reset_seed_policy") == "row_index_uint32"
            and protocol.get("validation_selection") == "manifest_order_prefix"
            and protocol.get("test_split_policy") == "not_loaded_or_evaluated"
            and protocol.get("compute_deadline_s") == REALTIME_P99_LIMIT_S
            and protocol.get("controller_init_timeout_s") == FORMAL_INIT_TIMEOUT_S
            and protocol.get("realtime_p99_limit_s") == REALTIME_P99_LIMIT_S
            and protocol.get("realtime_deadline_miss_rate_limit") == REALTIME_MISS_RATE_LIMIT
            and protocol.get("realtime_qualification_required_for_m6_pass") is False,
            protocol,
            {
                "backend": FORMAL_BACKEND,
                "level0_track_count": FORMAL_LEVEL0_TRACKS,
                "pid_validation_track_count": FORMAL_PID_VALIDATION_TRACKS,
                "mpc_validation_track_count": FORMAL_MPC_VALIDATION_TRACKS,
                "reset_seed_policy": "row_index_uint32",
                "validation_selection": "manifest_order_prefix",
                "test_split_policy": "not_loaded_or_evaluated",
                "compute_deadline_s": REALTIME_P99_LIMIT_S,
                "controller_init_timeout_s": FORMAL_INIT_TIMEOUT_S,
                "realtime_p99_limit_s": REALTIME_P99_LIMIT_S,
                "realtime_deadline_miss_rate_limit": REALTIME_MISS_RATE_LIMIT,
                "realtime_qualification_required_for_m6_pass": False,
            },
        ),
    ]
    for split, expected_count in (("level0", 1), ("validation", 100)):
        evidence = _mapping(assets.get(split))
        hash_values = (
            evidence.get("manifest_sha256"),
            evidence.get("manifest_asset_sha256"),
            evidence.get("asset_file_sha256"),
            evidence.get("geometry_hashes_sha256"),
        )
        checks.extend(
            (
                _check(
                    f"assets.{split}.count",
                    evidence.get("track_count") == expected_count
                    and evidence.get("loaded_track_count") == expected_count
                    and evidence.get("geometry_hash_count") == expected_count,
                    {
                        key: evidence.get(key)
                        for key in (
                            "track_count",
                            "loaded_track_count",
                            "geometry_hash_count",
                        )
                    },
                    expected_count,
                ),
                _check(
                    f"assets.{split}.hash",
                    all(
                        isinstance(value, str) and _SHA256_PATTERN.fullmatch(value)
                        for value in hash_values
                    )
                    and evidence.get("asset_file_sha256") == evidence.get("manifest_asset_sha256"),
                    hash_values,
                    "four SHA-256 digests with asset_file == manifest_asset",
                ),
                _check(
                    f"assets.{split}.readback",
                    evidence.get("readback_verified") is True,
                    evidence.get("readback_verified"),
                    True,
                ),
            )
        )
    loaded_splits = assets.get("loaded_splits")
    checks.append(
        _check(
            "assets.test_not_accessed",
            loaded_splits == ["level0", "validation"]
            and assets.get("test_split_accessed") is False
            and "test" not in assets,
            {
                "loaded_splits": loaded_splits,
                "test_split_accessed": assets.get("test_split_accessed"),
            },
            {"loaded_splits": ["level0", "validation"], "test_split_accessed": False},
        )
    )

    pid_level0 = _episodes(report, "pid", "level0")
    pid_validation = _episodes(report, "pid", "validation")
    mpc_level0 = _episodes(report, "mpc", "level0")
    mpc_validation = _episodes(report, "mpc", "validation")
    level0_ids = _mapping(assets.get("level0")).get("track_ids")
    validation_ids = _mapping(assets.get("validation")).get("track_ids")
    checks.extend(
        (
            _check(
                "protocol.track_selection",
                isinstance(level0_ids, list)
                and isinstance(validation_ids, list)
                and len(level0_ids) == FORMAL_LEVEL0_TRACKS
                and len(validation_ids) == FORMAL_MPC_VALIDATION_TRACKS
                and _episode_values(pid_level0, "track_id") == level0_ids
                and _episode_values(mpc_level0, "track_id") == level0_ids
                and _episode_values(pid_validation, "track_id")
                == validation_ids[:FORMAL_PID_VALIDATION_TRACKS]
                and _episode_values(mpc_validation, "track_id") == validation_ids,
                {
                    "pid_level0": _episode_values(pid_level0, "track_id"),
                    "pid_validation": _episode_values(pid_validation, "track_id"),
                    "mpc_level0": _episode_values(mpc_level0, "track_id"),
                    "mpc_validation": _episode_values(mpc_validation, "track_id"),
                },
                "Level 0 and manifest-order Validation prefixes",
            ),
            _check(
                "protocol.row_index_reset_seeds",
                _episode_values(pid_level0, "reset_seed") == list(range(1))
                and _episode_values(pid_validation, "reset_seed")
                == list(range(FORMAL_PID_VALIDATION_TRACKS))
                and _episode_values(mpc_level0, "reset_seed") == list(range(1))
                and _episode_values(mpc_validation, "reset_seed")
                == list(range(FORMAL_MPC_VALIDATION_TRACKS)),
                {
                    "pid_level0": _episode_values(pid_level0, "reset_seed"),
                    "pid_validation": _episode_values(pid_validation, "reset_seed"),
                    "mpc_level0": _episode_values(mpc_level0, "reset_seed"),
                    "mpc_validation": _episode_values(mpc_validation, "reset_seed"),
                },
                "zero-based row indices",
            ),
            _check(
                "controllers.pid.level0_success",
                _track_count(report, "pid", "level0") == 1
                and len(pid_level0) == 1
                and isinstance(pid_level0[0], Mapping)
                and pid_level0[0].get("success") is True
                and _success_rate(report, "pid", "level0") == 1.0,
                _success_rate(report, "pid", "level0"),
                1.0,
            ),
            _check(
                "controllers.mpc.level0_success",
                _track_count(report, "mpc", "level0") == 1
                and len(mpc_level0) == 1
                and isinstance(mpc_level0[0], Mapping)
                and mpc_level0[0].get("success") is True
                and _success_rate(report, "mpc", "level0") == 1.0,
                _success_rate(report, "mpc", "level0"),
                1.0,
            ),
            _check(
                "controllers.pid.validation_complete",
                _track_count(report, "pid", "validation") == FORMAL_PID_VALIDATION_TRACKS
                and len(pid_validation) == FORMAL_PID_VALIDATION_TRACKS
                and all(_normal_episode(episode) for episode in pid_validation),
                {
                    "track_count": _track_count(report, "pid", "validation"),
                    "episode_count": len(pid_validation),
                    "all_normal": all(_normal_episode(episode) for episode in pid_validation),
                },
                {"track_count": FORMAL_PID_VALIDATION_TRACKS, "all_normal": True},
            ),
            _check(
                "controllers.mpc.validation_success_rate",
                _track_count(report, "mpc", "validation") == FORMAL_MPC_VALIDATION_TRACKS
                and len(mpc_validation) == FORMAL_MPC_VALIDATION_TRACKS
                and all(_normal_episode(episode) for episode in mpc_validation)
                and isinstance(_success_rate(report, "mpc", "validation"), (int, float))
                and not isinstance(_success_rate(report, "mpc", "validation"), bool)
                and float(_success_rate(report, "mpc", "validation")) >= 0.80,
                {
                    "track_count": _track_count(report, "mpc", "validation"),
                    "episode_count": len(mpc_validation),
                    "success_rate": _success_rate(report, "mpc", "validation"),
                },
                {"track_count": FORMAL_MPC_VALIDATION_TRACKS, "minimum_success_rate": 0.80},
            ),
        )
    )
    all_episodes = pid_level0 + pid_validation + mpc_level0 + mpc_validation
    invalid_count = sum(
        isinstance(episode, Mapping)
        and episode.get("termination_reason") == int(RaceTermination.INVALID_ACTION)
        for episode in all_episodes
    )
    identity_findings = _evaluation_identity_findings(report)
    aggregate_findings = _aggregate_findings(report)
    timing_findings = _timing_findings(report)
    realtime_findings = _realtime_findings(report)
    init_timeout_findings = _init_timeout_findings(report)
    checks.extend(
        (
            _check(
                "protocol.evaluation_identity",
                not identity_findings,
                identity_findings,
                [],
            ),
            _check(
                "controllers.aggregate_consistency",
                not aggregate_findings,
                aggregate_findings,
                [],
            ),
            _check(
                "controllers.no_invalid_action",
                len(all_episodes) == 112 and invalid_count == 0,
                {"episode_count": len(all_episodes), "invalid_action_count": invalid_count},
                {"episode_count": 112, "invalid_action_count": 0},
            ),
            _check(
                "controllers.timing_consistency",
                not timing_findings,
                timing_findings,
                [],
            ),
            _check(
                "controllers.realtime_qualification_consistency",
                not realtime_findings,
                realtime_findings,
                [],
            ),
            _check(
                "controllers.init_timeout",
                not init_timeout_findings,
                init_timeout_findings,
                [],
            ),
        )
    )

    controller_configs = _mapping(report.get("controller_configs"))
    config_hashes_valid = all(
        _mapping(controller_configs.get(name)).get("directory") == f"controllers/{name}"
        and _mapping(controller_configs.get(name)).get("config_file")
        == f"controllers/{name}/config.toml"
        and isinstance(_mapping(controller_configs.get(name)).get("config_sha256"), str)
        and _SHA256_PATTERN.fullmatch(
            str(_mapping(controller_configs.get(name)).get("config_sha256"))
        )
        for name in ("pid", "mpc")
    )
    checks.append(
        _check(
            "controllers.config_hashes",
            config_hashes_valid,
            controller_configs,
            "repository-relative PID/MPC configs with SHA-256 digests",
        )
    )

    device = _mapping(runtime.get("jax_device"))
    checks.append(
        _check(
            "runtime.gpu",
            device.get("platform") == "gpu" and runtime.get("jax_gpu_error") is None,
            {"jax_device": device, "jax_gpu_error": runtime.get("jax_gpu_error")},
            {"platform": "gpu", "jax_gpu_error": None},
        )
    )
    checks.extend(
        (
            _check(
                "source.clean",
                before.get("relevant_source_clean") is True
                and after.get("relevant_source_clean") is True,
                {
                    "before": before.get("relevant_source_clean"),
                    "after": after.get("relevant_source_clean"),
                },
                {"before": True, "after": True},
            ),
            _check(
                "source.coverage",
                isinstance(before.get("source_files_sha256"), Mapping)
                and set(before["source_files_sha256"]) == set(RELEVANT_SOURCE_PATHS)
                and all(
                    isinstance(value, str) and _SHA256_PATTERN.fullmatch(value)
                    for value in before["source_files_sha256"].values()
                )
                and isinstance(after.get("source_files_sha256"), Mapping)
                and set(after["source_files_sha256"]) == set(RELEVANT_SOURCE_PATHS)
                and all(
                    isinstance(value, str) and _SHA256_PATTERN.fullmatch(value)
                    for value in after["source_files_sha256"].values()
                ),
                {
                    "before_count": len(_mapping(before.get("source_files_sha256"))),
                    "after_count": len(_mapping(after.get("source_files_sha256"))),
                },
                {"source_path_count": len(RELEVANT_SOURCE_PATHS)},
            ),
            _check(
                "source.stable",
                isinstance(before.get("git_revision"), str)
                and bool(before.get("git_revision"))
                and before.get("git_revision") == after.get("git_revision")
                and isinstance(before.get("source_files_sha256"), Mapping)
                and before.get("source_files_sha256") == after.get("source_files_sha256"),
                {
                    "revision_before": before.get("git_revision"),
                    "revision_after": after.get("git_revision"),
                    "hashes_equal": before.get("source_files_sha256")
                    == after.get("source_files_sha256"),
                },
                "same non-empty revision and source hashes",
            ),
        )
    )
    privacy = _privacy_findings(
        {key: value for key, value in report.items() if key not in ("checks", "privacy")}
    )
    checks.append(
        _check(
            "report.privacy",
            not privacy["absolute_paths"] and not privacy["gpu_uuids"],
            privacy,
            {"absolute_paths": [], "gpu_uuids": []},
        )
    )
    return checks


def run_benchmark(
    options: BenchmarkOptions,
    *,
    project_root: Path = PROJECT_ROOT,
    asset_loader: AssetLoader | None = None,
    evaluator: ControllerEvaluator | None = None,
    snapshot_loader: SnapshotLoader | None = None,
    runtime_loader: RuntimeLoader | None = None,
) -> dict[str, Any]:
    """Execute the locked formal workload and return a strict-JSON-compatible report."""

    if not isinstance(options, BenchmarkOptions):
        raise TypeError("options must be BenchmarkOptions")
    root = Path(project_root).expanduser().resolve()
    load_assets = _load_evaluation_assets if asset_loader is None else asset_loader
    run_evaluation = evaluate_track_batch if evaluator is None else evaluator
    take_snapshot = _source_snapshot if snapshot_loader is None else snapshot_loader
    load_runtime = _runtime_evidence if runtime_loader is None else runtime_loader

    before = dict(take_snapshot(root))
    _require_source_preflight(before)
    config = load_project_config(root)
    if config.benchmark.version != "0.1":
        raise RuntimeError("formal M6 evaluation is locked to benchmark version 0.1")
    if config.benchmark.validation_track_count != FORMAL_MPC_VALIDATION_TRACKS:
        raise RuntimeError("formal M6 evaluation requires exactly 100 Validation Tracks")
    if config.benchmark.controller.init_timeout_s != FORMAL_INIT_TIMEOUT_S:
        raise RuntimeError("formal M6 evaluation requires the fixed 30 second init timeout")
    assets = load_assets(config, root)
    if int(assets.level0_batch.seed.shape[0]) != FORMAL_LEVEL0_TRACKS:
        raise RuntimeError("formal Level 0 asset must contain exactly one Track")
    if int(assets.validation_batch.seed.shape[0]) != FORMAL_MPC_VALIDATION_TRACKS:
        raise RuntimeError("formal Validation asset must contain exactly 100 Tracks")

    runtime = dict(load_runtime())
    controller_configs = {name: _controller_config_evidence(root, name) for name in ("pid", "mpc")}
    evaluations = _run_controller_evaluations(
        config,
        assets,
        root,
        run_evaluation,
    )
    after = dict(take_snapshot(root))
    report: dict[str, Any] = {
        "schema_version": REPORT_SCHEMA_VERSION,
        "protocol_version": PROTOCOL_VERSION,
        "generated_at_utc": datetime.now(UTC).isoformat(),
        "protocol": {
            "backend": FORMAL_BACKEND,
            "benchmark_version": config.benchmark.version,
            "level0_track_count": FORMAL_LEVEL0_TRACKS,
            "pid_validation_track_count": FORMAL_PID_VALIDATION_TRACKS,
            "mpc_validation_track_count": FORMAL_MPC_VALIDATION_TRACKS,
            "validation_selection": "manifest_order_prefix",
            "reset_seed_policy": "row_index_uint32",
            "test_split_policy": "not_loaded_or_evaluated",
            "compute_deadline_s": config.benchmark.controller.compute_deadline_s,
            "controller_init_timeout_s": config.benchmark.controller.init_timeout_s,
            "realtime_p99_limit_s": REALTIME_P99_LIMIT_S,
            "realtime_deadline_miss_rate_limit": REALTIME_MISS_RATE_LIMIT,
            "realtime_qualification_required_for_m6_pass": False,
        },
        "assets": _json_value(assets.evidence),
        "controller_configs": controller_configs,
        "evaluations": evaluations,
        "runtime": runtime,
        "source_evidence": {"before": before, "after": after},
    }
    report["privacy"] = _privacy_findings(report)
    report["checks"] = evaluate_report_gates(report)
    report["status"] = (
        "pass"
        if report["checks"] and all(check["passed"] for check in report["checks"])
        else "fail"
    )
    return _json_value(report)


def main(argv: list[str] | None = None) -> None:
    """Write the formal report and exit non-zero when any required gate fails."""

    options = _parse_args(argv)
    report = run_benchmark(options)
    write_strict_json(options.output, report)
    print(
        json.dumps(
            {
                "output": options.output.as_posix(),
                "status": report["status"],
                "gate_count": len(report["checks"]),
                "failed_gates": [check["id"] for check in report["checks"] if not check["passed"]],
            },
            sort_keys=True,
        )
    )
    if report["status"] != "pass":
        raise SystemExit("formal M6 Controller benchmark failed one or more gates")


if __name__ == "__main__":  # pragma: no cover - exercised through Pixi/CLI
    main(sys.argv[1:])
