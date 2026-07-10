"""Benchmark the formal M5 10,000-Track pool on 1,024 MJX-Warp worlds.

The formal throughput loops enqueue device-native ``VecCarRacingEnv.step`` calls and synchronize
only once at the end. Correctness, transfer-guard, reset-heavy, and health checks are deliberately
separate from those timing intervals.
"""

from __future__ import annotations

import os

# Make process VRAM observable and choose physical GPUs deterministically before importing JAX.
os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
os.environ.setdefault("CUDA_DEVICE_ORDER", "PCI_BUS_ID")

import argparse
import gc
import hashlib
import json
import math
import re
import sys
import time
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np

from controller_learning.config import ProjectConfig, load_project_config
from controller_learning.envs.episode import (
    initialize_episode_identities,
    masked_next_episode,
    track_pool_seeds,
)
from controller_learning.tracks.admission import (
    ADMISSION_PROTOCOL_VERSION,
    ADMISSION_REPORT_SCHEMA_VERSION,
    DRIVEABILITY_PROTOCOL_VERSION,
    FORMAL_ADMISSION_WORLDS,
    evaluate_admission_report,
)
from controller_learning.tracks.assets import (
    TrackAssetManifest,
    load_track_asset_manifest,
    load_track_batch_npz,
    sha256_file,
)
from controller_learning.tracks.hashing import track_batch_geometry_sha256
from controller_learning.tracks.official_assets import (
    OFFICIAL_TRACK_SPLITS,
    OfficialAssetVerification,
    verify_official_track_assets,
)
from controller_learning.tracks.pool import TrackPool
from controller_learning.tracks.types import Track, TrackBatch
from scripts import benchmark_racing_env as m4_benchmark

REPORT_SCHEMA_VERSION = "controller-learning.m5-track-pool.v2"
PROTOCOL_VERSION = "m5-track-pool-gpu-v2"
FORMAL_NUM_WORLDS = 1024
FORMAL_LEVEL_ID = 1
FORMAL_RESET_SEED = 20260710
ALLOCATOR_STABILIZATION_SEED = 20260711
MEASUREMENT_EPOCHS = (
    ("E1", FORMAL_RESET_SEED),
    ("E2", 20260712),
    ("E3", 20260713),
)
HEADLINE_EPOCH = "E1"
FORMAL_TRAIN_TRACK_COUNT = 10_000
DEFAULT_ENVIRONMENT_STEPS = 10_000
DEFAULT_ALLOCATOR_STABILIZATION_STEPS = 10_000
DEFAULT_WARMUP_STEPS = 8
DEFAULT_HEALTH_MAX_STEPS = 5_000
DEFAULT_RESET_HEAVY_CYCLES = 64
MINIMUM_POOL_TO_FIXED_THROUGHPUT_RATIO = 0.75
PROCESS_VRAM_GROWTH_LIMIT_MIB = 64.0
HOST_RSS_GROWTH_LIMIT_MIB = 64.0
MEMORY_SLOPE_LIMIT_MIB_PER_EPOCH = 4.0
LIVE_BYTES_GROWTH_LIMIT = 32 * 1024 * 1024
LIVE_BYTES_MAX_WINDOW_GROWTH = 32 * 1024 * 1024
LIVE_BYTES_SLOPE_LIMIT_PER_EPOCH = 4 * 1024 * 1024
PEAK_BYTES_GROWTH_LIMIT = 64 * 1024 * 1024
POOL_BYTES_GROWTH_TOLERANCE = 0
MAXIMUM_PROCESS_VRAM_FRACTION = 0.80
MINIMUM_GPU_FREE_MIB = 1024.0
EXPECTED_MEMORY_SAMPLE_PHASES = (
    "before_environment",
    "after_environment_create",
    "after_initial_compile_and_warmup",
    "after_health_preflight",
    "after_reset_heavy_preflight",
    "before_allocator_stabilization",
    "allocator_stabilized_E0",
    "post_stabilization_E1",
    "post_stabilization_E2",
    "post_stabilization_E3",
    "after_fixed_baseline",
)
DEFAULT_MANIFEST = Path("controller_learning/assets/tracks/v0.1/train.json")
DEFAULT_CACHE = Path(".track-cache/v0.1/train_pool.npz")
DEFAULT_ADMISSION_REPORT = Path("benchmarks/v0.1/m5_track_admission_report.json")
DEFAULT_OUTPUT = Path("benchmarks/v0.1/m5_track_pool_report.json")
PROJECT_ROOT = Path(__file__).resolve().parents[1]
TERMINATION_NAMES = ("none", "success", "off_track", "invalid_action", "timeout")
DYNAMIC_OBSERVATION_KEYS = (
    "position",
    "yaw",
    "velocity_body",
    "yaw_rate",
    "steering_angle",
    "track_progress",
)

RELEVANT_SOURCE_PATHS = (
    "pixi.lock",
    "pyproject.toml",
    "configs/benchmark.toml",
    "configs/levels/level1.toml",
    "configs/track.toml",
    "configs/vehicle.toml",
    "benchmarks/v0.1/m5_track_admission_report.json",
    "controller_learning/assets/tracks/v0.1/train.json",
    "controller_learning/assets/vehicle/car.xml",
    "controller_learning/config/loader.py",
    "controller_learning/config/models.py",
    "controller_learning/envs/_vehicle_driver.py",
    "controller_learning/envs/configuration.py",
    "controller_learning/envs/episode.py",
    "controller_learning/envs/observation.py",
    "controller_learning/envs/race_core.py",
    "controller_learning/envs/vector_racing.py",
    "controller_learning/physics/actuation.py",
    "controller_learning/physics/mjx_warp.py",
    "controller_learning/physics/model.py",
    "controller_learning/tracks/assets.py",
    "controller_learning/tracks/admission.py",
    "controller_learning/tracks/driveability.py",
    "controller_learning/tracks/generator.py",
    "controller_learning/tracks/geometry.py",
    "controller_learning/tracks/hashing.py",
    "controller_learning/tracks/pool.py",
    "controller_learning/tracks/specs.py",
    "controller_learning/tracks/types.py",
    "controller_learning/tracks/validator.py",
    "scripts/benchmark_racing_env.py",
    "scripts/benchmark_track_pool.py",
    "scripts/build_track_assets.py",
    "scripts/validate_track_driveability.py",
)

_UUID_PATTERN = re.compile(
    r"(?i)(?:GPU-)?[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}"
)
_WINDOWS_ABSOLUTE_PATTERN = re.compile(r"^[A-Za-z]:[\\/]")


@dataclass(frozen=True, slots=True)
class BenchmarkOptions:
    """Validated controls; formal world count, backend, and comparison protocol stay fixed."""

    output: Path = DEFAULT_OUTPUT
    manifest: Path = DEFAULT_MANIFEST
    cache: Path = DEFAULT_CACHE
    admission_report: Path = DEFAULT_ADMISSION_REPORT
    environment_steps: int = DEFAULT_ENVIRONMENT_STEPS
    allocator_stabilization_steps: int = DEFAULT_ALLOCATOR_STABILIZATION_STEPS
    warmup_steps: int = DEFAULT_WARMUP_STEPS
    health_max_steps: int = DEFAULT_HEALTH_MAX_STEPS
    reset_heavy_cycles: int = DEFAULT_RESET_HEAVY_CYCLES

    def __post_init__(self) -> None:
        for name in (
            "environment_steps",
            "allocator_stabilization_steps",
            "warmup_steps",
            "health_max_steps",
            "reset_heavy_cycles",
        ):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
                raise ValueError(f"{name} must be a positive integer")
        for name in ("output", "manifest", "cache", "admission_report"):
            if not isinstance(getattr(self, name), Path):
                raise TypeError(f"{name} must be a pathlib.Path")


def _positive_integer(value: str) -> int:
    try:
        parsed = int(value)
    except ValueError as error:
        raise argparse.ArgumentTypeError("must be a positive integer") from error
    if parsed <= 0:
        raise argparse.ArgumentTypeError("must be a positive integer")
    return parsed


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--cache", type=Path, default=DEFAULT_CACHE)
    parser.add_argument(
        "--admission-report",
        type=Path,
        default=DEFAULT_ADMISSION_REPORT,
        help="Committed formal M5 admission evidence bound to every Track artifact",
    )
    parser.add_argument(
        "--steps",
        dest="environment_steps",
        type=_positive_integer,
        default=DEFAULT_ENVIRONMENT_STEPS,
        help="Timed TrackPool and fixed-baseline environment steps (default: 10000 each)",
    )
    parser.add_argument(
        "--warmup-steps",
        type=_positive_integer,
        default=DEFAULT_WARMUP_STEPS,
    )
    parser.add_argument(
        "--allocator-stabilization-steps",
        type=_positive_integer,
        default=DEFAULT_ALLOCATOR_STABILIZATION_STEPS,
        help=(
            "Untimed no-sync steps used to stabilize allocator growth before the formal memory "
            "baseline (default: 10000)"
        ),
    )
    parser.add_argument(
        "--health-max-steps",
        type=_positive_integer,
        default=DEFAULT_HEALTH_MAX_STEPS,
    )
    parser.add_argument(
        "--reset-heavy-cycles",
        type=_positive_integer,
        default=DEFAULT_RESET_HEAVY_CYCLES,
    )
    return parser


def _parse_args(argv: list[str] | None = None) -> BenchmarkOptions:
    values = _build_parser().parse_args(argv)
    return BenchmarkOptions(
        output=values.output,
        manifest=values.manifest,
        cache=values.cache,
        admission_report=values.admission_report,
        environment_steps=values.environment_steps,
        allocator_stabilization_steps=values.allocator_stabilization_steps,
        warmup_steps=values.warmup_steps,
        health_max_steps=values.health_max_steps,
        reset_heavy_cycles=values.reset_heavy_cycles,
    )


def _resolve(root: Path, path: Path) -> Path:
    return path.expanduser().resolve() if path.is_absolute() else (root / path).resolve()


def _same_path(root: Path, actual: Path, expected: Path) -> bool:
    return _resolve(root, actual) == _resolve(root, expected)


def _source_snapshot(project_root: Path) -> dict[str, Any]:
    missing = [path for path in RELEVANT_SOURCE_PATHS if not (project_root / path).is_file()]
    if missing:
        raise FileNotFoundError(f"M5 benchmark source inputs are missing: {', '.join(missing)}")
    hashes = {path: sha256_file(project_root / path) for path in RELEVANT_SOURCE_PATHS}
    relevant_status = m4_benchmark._git(
        project_root,
        "status",
        "--porcelain",
        "--",
        *RELEVANT_SOURCE_PATHS,
    )
    tracked_status = m4_benchmark._git(
        project_root,
        "status",
        "--porcelain",
        "--untracked-files=no",
    )
    return {
        "git_revision": m4_benchmark._git(project_root, "rev-parse", "HEAD"),
        "relevant_source_clean": None if relevant_status is None else not bool(relevant_status),
        "tracked_worktree_clean": None if tracked_status is None else not bool(tracked_status),
        "source_files_sha256": hashes,
    }


def _reject_duplicate_json_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    value: dict[str, Any] = {}
    for key, item in pairs:
        if key in value:
            raise ValueError(f"duplicate JSON key in admission report: {key}")
        value[key] = item
    return value


def _reject_nonfinite_json(value: str) -> None:
    raise ValueError(f"non-finite JSON value in admission report: {value}")


def _verify_official_asset_set(
    config: ProjectConfig,
    *,
    asset_directory: Path,
    train_cache_path: Path,
) -> tuple[OfficialAssetVerification, dict[str, Any]]:
    """Verify all official splits and require the manifest-bound local train cache."""

    verification = verify_official_track_assets(
        config,
        asset_directory=asset_directory,
        train_cache_path=train_cache_path,
        require_train_cache=True,
    )
    expected_splits = {spec.split for spec in OFFICIAL_TRACK_SPLITS}
    manifest_splits = set(verification.manifests)
    fixed_splits = set(verification.fixed_batches)
    expected_fixed = {spec.split for spec in OFFICIAL_TRACK_SPLITS if spec.package_asset}
    evidence = {
        "manifest_splits": sorted(manifest_splits),
        "fixed_package_asset_splits": sorted(fixed_splits),
        "complete_manifest_set_verified": manifest_splits == expected_splits,
        "fixed_package_assets_verified": fixed_splits == expected_fixed,
        "train_cache_verified": verification.train_cache_verified,
        "split_namespaces_and_protocols_verified": True,
    }
    evidence["passed"] = bool(
        evidence["complete_manifest_set_verified"]
        and evidence["fixed_package_assets_verified"]
        and evidence["train_cache_verified"]
    )
    if not evidence["passed"]:
        raise RuntimeError("official Track asset verification returned incomplete evidence")
    return verification, evidence


def _load_verified_admission_evidence(
    report_path: Path,
    *,
    config: ProjectConfig,
    asset_directory: Path,
    train_cache_path: Path,
    official_verification: OfficialAssetVerification,
) -> dict[str, Any]:
    """Bind a strict passing admission report to all current manifests and the train cache."""

    try:
        report = json.loads(
            report_path.read_text(encoding="utf-8"),
            object_pairs_hook=_reject_duplicate_json_keys,
            parse_constant=_reject_nonfinite_json,
        )
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise RuntimeError("cannot load the formal M5 admission report") from error
    if not isinstance(report, dict):
        raise RuntimeError("formal M5 admission report root must be an object")
    if report.get("status") != "pass":
        raise RuntimeError("formal M5 admission report status must be 'pass'")
    if report.get("schema_version") != ADMISSION_REPORT_SCHEMA_VERSION:
        raise RuntimeError("formal M5 admission report schema version is not supported")
    if report.get("protocol_version") != ADMISSION_PROTOCOL_VERSION:
        raise RuntimeError("formal M5 admission protocol version is not supported")

    protocol = report.get("protocol")
    if not isinstance(protocol, dict):
        raise RuntimeError("formal M5 admission report protocol must be an object")
    required_protocol = {
        "benchmark_version": config.benchmark.version,
        "generator_version": config.track.generator.generator_version,
        "driveability_protocol_version": DRIVEABILITY_PROTOCOL_VERSION,
        "formal_physics_backend": "MJX-Warp",
        "admission_worlds": FORMAL_ADMISSION_WORLDS,
    }
    if any(protocol.get(key) != value for key, value in required_protocol.items()):
        raise RuntimeError("formal M5 admission report protocol does not match the benchmark")

    recomputed_checks = tuple(evaluate_admission_report(report))
    actual_checks = report.get("checks")
    if actual_checks != list(recomputed_checks) or not all(
        check.get("passed") is True for check in recomputed_checks
    ):
        raise RuntimeError("formal M5 admission report gates are missing, stale, or failing")
    source = report.get("source_evidence")
    if not isinstance(source, dict):
        raise RuntimeError("formal M5 admission report source evidence is missing")
    before = source.get("before")
    after = source.get("after")
    if not isinstance(before, dict) or not isinstance(after, dict):
        raise RuntimeError("formal M5 admission report source snapshots are missing")
    source_passed = bool(
        before.get("git_revision") is not None
        and before.get("git_revision") == after.get("git_revision")
        and before.get("relevant_source_clean") is True
        and after.get("relevant_source_clean") is True
        and before.get("source_files_sha256") == after.get("source_files_sha256")
    )
    if not source_passed:
        raise RuntimeError("formal M5 admission report source evidence is not clean and stable")

    artifacts = report.get("artifacts")
    expected_splits = {spec.split for spec in OFFICIAL_TRACK_SPLITS}
    if not isinstance(artifacts, dict) or set(artifacts) != expected_splits:
        raise RuntimeError("formal M5 admission report artifact set is incomplete")
    current_manifest_sha256 = {
        spec.split: sha256_file(asset_directory / spec.manifest_file)
        for spec in OFFICIAL_TRACK_SPLITS
    }
    manifest_sha256_matches = {
        spec.split: artifacts[spec.split].get("manifest_sha256")
        == current_manifest_sha256[spec.split]
        for spec in OFFICIAL_TRACK_SPLITS
    }
    artifact_names_match = {
        spec.split: artifacts[spec.split].get("manifest_file") == spec.manifest_file
        and artifacts[spec.split].get("asset_file") == spec.asset_file
        for spec in OFFICIAL_TRACK_SPLITS
    }
    manifest_asset_sha256_matches = {
        spec.split: artifacts[spec.split].get("asset_sha256")
        == official_verification.manifests[spec.split].asset_sha256
        for spec in OFFICIAL_TRACK_SPLITS
    }
    train_cache_sha256 = sha256_file(train_cache_path)
    train_cache_matches = (
        artifacts["train"].get("asset_sha256") == train_cache_sha256
        and train_cache_sha256 == official_verification.manifests["train"].asset_sha256
    )
    if not (
        all(manifest_sha256_matches.values())
        and all(artifact_names_match.values())
        and all(manifest_asset_sha256_matches.values())
        and train_cache_matches
    ):
        raise RuntimeError(
            "formal M5 admission report does not identify the current Track artifacts"
        )
    return {
        "report_sha256": sha256_file(report_path),
        "status": report["status"],
        "schema_version": report["schema_version"],
        "protocol_version": report["protocol_version"],
        "protocol": required_protocol,
        "recomputed_gate_count": len(recomputed_checks),
        "all_recomputed_gates_passed": True,
        "source_evidence_passed": source_passed,
        "source_git_revision": before["git_revision"],
        "manifest_sha256_matches": manifest_sha256_matches,
        "artifact_names_match": artifact_names_match,
        "manifest_asset_sha256_matches": manifest_asset_sha256_matches,
        "train_cache_sha256": train_cache_sha256,
        "train_cache_sha256_matches": train_cache_matches,
        "passed": True,
    }


def _uint32_digest(values: Any) -> str:
    array = np.ascontiguousarray(np.asarray(values, dtype="<u4"))
    return hashlib.sha256(array.tobytes(order="C")).hexdigest()


def _expected_track_ids(pool: TrackPool, identity: Any) -> np.ndarray:
    """Return exact host-reference Track IDs for one episode-identity snapshot."""

    selection_seeds = track_pool_seeds(identity)
    indices = np.remainder(selection_seeds, np.uint32(pool.size)).astype(np.int64)
    return np.asarray(pool.batch.seed[indices], dtype=np.uint32)


def _advance_all_episodes(identity: Any, count: int) -> Any:
    mask = np.ones(identity.num_envs, dtype=np.bool_)
    current = identity
    for _ in range(count):
        current = masked_next_episode(current, mask)
    return current


def _load_verified_train_pool(
    config: ProjectConfig,
    manifest_path: Path,
    cache_path: Path,
) -> tuple[TrackAssetManifest, TrackPool, dict[str, Any]]:
    """Load the committed manifest and verify the external cache byte-for-byte and row-for-row."""

    started = time.perf_counter()
    manifest = load_track_asset_manifest(manifest_path)
    if manifest.split != "train" or manifest.level_id != FORMAL_LEVEL_ID:
        raise RuntimeError("the formal M5 manifest must be the Level 1 train split")
    if manifest.benchmark_version != config.benchmark.version:
        raise RuntimeError("train manifest benchmark version does not match ProjectConfig")
    if manifest.generator_version != config.track.generator.generator_version:
        raise RuntimeError("train manifest generator version does not match ProjectConfig")
    if manifest.track_count != config.benchmark.train_track_count:
        raise RuntimeError("train manifest count does not match ProjectConfig")
    if cache_path.name != manifest.asset_file:
        raise RuntimeError("the local train cache filename does not match manifest.asset_file")

    batch = load_track_batch_npz(
        cache_path,
        expected_sha256=manifest.asset_sha256,
        expected_track_count=manifest.track_count,
        expected_capacity=manifest.capacity,
    )
    manifest_seeds = np.asarray([record.seed for record in manifest.tracks], dtype=np.uint32)
    if not np.array_equal(batch.seed, manifest_seeds):
        raise RuntimeError("train cache seed order does not match the manifest")
    manifest_hashes = tuple(record.geometry_sha256 for record in manifest.tracks)
    if track_batch_geometry_sha256(batch) != manifest_hashes:
        raise RuntimeError("train cache geometry hashes do not match the manifest")
    if not np.all(batch.width_m == np.float32(manifest.track_width_m)):
        raise RuntimeError("train cache width does not match the manifest")

    pool = TrackPool(
        benchmark_version=manifest.benchmark_version,
        generator_version=manifest.generator_version,
        split="train",
        batch=batch,
    )
    bytes_total = sum(int(np.asarray(leaf).nbytes) for leaf in pool.batch)
    cache_sha256 = sha256_file(cache_path)
    evidence = {
        "manifest_sha256": sha256_file(manifest_path),
        "cache_sha256": cache_sha256,
        "cache_matches_manifest_sha256": cache_sha256 == manifest.asset_sha256,
        "manifest_asset_file": manifest.asset_file,
        "split": manifest.split,
        "level_id": manifest.level_id,
        "track_count": manifest.track_count,
        "configured_track_count": config.benchmark.train_track_count,
        "capacity": asdict(manifest.capacity),
        "generator_version": manifest.generator_version,
        "geometry_validation_version": manifest.geometry_validation_version,
        "driveability_protocol_version": manifest.driveability_protocol_version,
        "geometry_admission_pass_count": sum(
            record.geometry_validation == "passed" for record in manifest.tracks
        ),
        "driveability_admission_pass_count": sum(
            record.driveability_validation == "passed" for record in manifest.tracks
        ),
        "unique_seed_count": int(np.unique(manifest_seeds).size),
        "allowed_seed_uint32_sha256": _uint32_digest(manifest_seeds),
        "geometry_hash_count": len(set(manifest_hashes)),
        "pool_array_bytes": bytes_total,
        "load_and_verify_seconds": time.perf_counter() - started,
    }
    return manifest, pool, evidence


def _create_environment(**kwargs: Any):
    # Keep CLI parsing and CPU report tests independent from eager GPU backend construction.
    from controller_learning.envs import VecCarRacingEnv

    return VecCarRacingEnv(**kwargs)


def _block_public_step(jax: Any, output: tuple[Any, ...]) -> None:
    observation, reward, terminated, truncated, info = output
    numeric_info = {
        key: info[key]
        for key in (
            "episode_seed",
            "controller_seed",
            "track_id",
            "termination_reason",
            "lap_completed",
            "lap_time_s",
        )
    }
    jax.block_until_ready((observation, reward, terminated, truncated, numeric_info))


def _all_public_finite(output: tuple[Any, ...]) -> tuple[bool, list[str]]:
    observation, reward, _, _, info = output
    failures: list[str] = []
    for key, value in observation.items():
        array = np.asarray(value)
        if np.issubdtype(array.dtype, np.number) and not np.isfinite(array).all():
            failures.append(f"observation.{key}")
    for key, value in (("reward", reward), ("info.lap_time_s", info["lap_time_s"])):
        if not np.isfinite(np.asarray(value)).all():
            failures.append(key)
    return not failures, failures


def _finite_dynamic_rows(jnp: Any, observation: Mapping[str, Any], reward: Any, info: Any):
    finite = jnp.ones((FORMAL_NUM_WORLDS,), dtype=bool)
    for key in DYNAMIC_OBSERVATION_KEYS:
        values = jnp.asarray(observation[key])
        finite &= jnp.all(jnp.isfinite(values).reshape((FORMAL_NUM_WORLDS, -1)), axis=1)
    finite &= jnp.isfinite(reward)
    finite &= jnp.isfinite(info["lap_time_s"])
    return finite


def _ids_allowed(jnp: Any, track_ids: Any, sorted_allowed_ids: Any):
    ids = jnp.asarray(track_ids, dtype=jnp.uint32)
    allowed = jnp.asarray(sorted_allowed_ids, dtype=jnp.uint32)
    positions = jnp.searchsorted(allowed, ids, side="left", method="scan")
    safe_positions = jnp.minimum(positions, allowed.shape[0] - 1)
    return (positions < allowed.shape[0]) & (allowed[safe_positions] == ids)


def _array_device_platform(value: Any) -> str | None:
    device = getattr(value, "device", None)
    if callable(device):
        device = device()
    return None if device is None else getattr(device, "platform", None)


def _pool_residency_evidence(jax: Any, env: Any, expected_bytes: int) -> dict[str, Any]:
    device_batch = env._pool_batch
    jax.block_until_ready(device_batch)
    leaves = jax.tree.leaves(device_batch)
    platforms = [_array_device_platform(leaf) for leaf in leaves]
    device_bytes = sum(int(leaf.size * leaf.dtype.itemsize) for leaf in leaves)
    return {
        "track_count": int(device_batch.seed.shape[0]),
        "leaf_count": len(leaves),
        "all_leaves_on_gpu": bool(platforms and all(value == "gpu" for value in platforms)),
        "device_platforms": sorted(set(platforms), key=lambda value: str(value)),
        "host_verified_pool_bytes": int(expected_bytes),
        "device_pool_bytes": device_bytes,
        "byte_count_matches_host": device_bytes == expected_bytes,
        "device_seed_uint32_sha256": _uint32_digest(np.asarray(device_batch.seed)),
    }


def _tree_equal(jax: Any, jnp: Any, left: Any, right: Any) -> bool:
    comparisons = [
        jnp.array_equal(left_leaf, right_leaf)
        for left_leaf, right_leaf in zip(
            jax.tree.leaves(left),
            jax.tree.leaves(right),
            strict=True,
        )
    ]
    result = jnp.all(jnp.stack(comparisons))
    return bool(np.asarray(jax.block_until_ready(result)))


def _deterministic_reset_evidence(
    jax: Any,
    jnp: Any,
    env: Any,
    sorted_allowed_ids: Any,
) -> tuple[dict[str, Any], float]:
    started = time.perf_counter()
    first_observation, first_info = env.reset(seed=FORMAL_RESET_SEED)
    jax.block_until_ready((first_observation, first_info["track_id"]))
    first_tracks = env._track_batch
    first_ids = np.asarray(first_info["track_id"], dtype=np.uint32)

    second_observation, second_info = env.reset(seed=FORMAL_RESET_SEED)
    jax.block_until_ready((second_observation, second_info["track_id"]))
    reset_seconds = time.perf_counter() - started
    second_tracks = env._track_batch
    numeric_info_keys = (
        "episode_seed",
        "controller_seed",
        "track_id",
        "termination_reason",
        "lap_completed",
        "lap_time_s",
    )
    info_equal = all(
        np.array_equal(np.asarray(first_info[key]), np.asarray(second_info[key]))
        for key in numeric_info_keys
    ) and np.array_equal(
        np.asarray(first_info["benchmark_version"]),
        np.asarray(second_info["benchmark_version"]),
    )
    allowed = bool(
        np.asarray(jnp.all(_ids_allowed(jnp, second_info["track_id"], sorted_allowed_ids)))
    )
    evidence = {
        "same_seed_observation_bit_exact": _tree_equal(
            jax, jnp, first_observation, second_observation
        ),
        "same_seed_track_batch_bit_exact": _tree_equal(jax, jnp, first_tracks, second_tracks),
        "same_seed_info_bit_exact": info_equal,
        "all_initial_track_ids_allowed": allowed,
        "initial_track_id_uint32_sha256": _uint32_digest(first_ids),
        "initial_unique_track_id_count": int(np.unique(first_ids).size),
        "sampling_with_replacement": True,
    }
    evidence["passed"] = bool(
        evidence["same_seed_observation_bit_exact"]
        and evidence["same_seed_track_batch_bit_exact"]
        and evidence["same_seed_info_bit_exact"]
        and evidence["all_initial_track_ids_allowed"]
    )
    return evidence, reset_seconds


def _jit_cache_objects(env: Any) -> dict[str, Any]:
    vehicle_driver = getattr(env, "_vehicle_driver", None)
    vehicle = getattr(vehicle_driver, "_vehicle", None)
    return {
        "encode_observation": getattr(env, "_encode_observation", None),
        "finalize_gpu_step": getattr(env, "_finalize_gpu_step", None),
        "normalize_actions": getattr(env, "_normalize_actions", None),
        "planar_position": getattr(env, "_planar_position", None),
        "read_vehicle_state": getattr(env, "_read_vehicle_state", None),
        "reset_race": getattr(env, "_reset_race", None),
        "select_pool_tracks": getattr(env, "_select_pool_tracks", None),
        "step_race": getattr(env, "_step_race", None),
        "vehicle_step": getattr(vehicle, "_step_function", None),
    }


def _jit_cache_snapshot(env: Any) -> dict[str, int | None]:
    values: dict[str, int | None] = {}
    for name, function in _jit_cache_objects(env).items():
        cache_size = getattr(function, "_cache_size", None)
        try:
            values[name] = int(cache_size()) if callable(cache_size) else None
        except (RuntimeError, TypeError, ValueError):
            values[name] = None
    return values


def _cache_evidence(
    after_compile: Mapping[str, int | None],
    after_all_pool_work: Mapping[str, int | None],
) -> dict[str, Any]:
    supported = all(value is not None for value in after_compile.values())
    compiled = supported and all(int(value) >= 1 for value in after_compile.values())
    stable = dict(after_compile) == dict(after_all_pool_work)
    return {
        "after_compile_and_warmup": dict(after_compile),
        "after_all_pool_work": dict(after_all_pool_work),
        "cache_size_introspection_supported": supported,
        "all_expected_executables_compiled": compiled,
        "cache_sizes_stable_after_warmup": stable,
        "recompile_detected": not stable,
        "passed": bool(supported and compiled and stable),
    }


def _add_epoch_cache_evidence(
    evidence: dict[str, Any],
    snapshots: Mapping[str, Mapping[str, int | None]],
) -> None:
    ordered = {label: dict(snapshot) for label, snapshot in snapshots.items()}
    values = list(ordered.values())
    stable = bool(values) and all(snapshot == values[0] for snapshot in values[1:])
    evidence.update(
        {
            "epoch_snapshots": ordered,
            "cache_sizes_stable_from_E0_through_E3": stable,
            "passed": bool(evidence["passed"] and stable),
        }
    )


def _track_rows_preserved(
    before: TrackBatch,
    after: TrackBatch,
    selected: np.ndarray,
) -> bool:
    unselected = ~selected
    return all(
        np.array_equal(np.asarray(old)[unselected], np.asarray(new)[unselected])
        for old, new in zip(before, after, strict=True)
    )


def _track_observation_matches(
    observation: Mapping[str, Any],
    tracks: TrackBatch,
    selected: np.ndarray,
) -> bool:
    fields = {
        "centerline": tracks.centerline_m,
        "left_boundary": tracks.left_boundary_m,
        "right_boundary": tracks.right_boundary_m,
        "track_mask": tracks.track_mask.astype(np.int8),
        "track_length": tracks.length_m,
    }
    return all(
        np.array_equal(np.asarray(observation[key])[selected], np.asarray(value)[selected])
        for key, value in fields.items()
    )


def _run_transfer_and_mixed_reset_checks(
    env: Any,
    action: Any,
    pool: TrackPool,
    sorted_allowed_ids: Any,
) -> dict[str, Any]:
    """Check transfer guards plus old-terminal/new-reset Track identity and row isolation."""

    import jax
    import jax.numpy as jnp

    initial_observation, initial_info = env.reset(seed=FORMAL_RESET_SEED)
    jax.block_until_ready((initial_observation, initial_info["track_id"]))
    try:
        with jax.transfer_guard("disallow"):
            active = env.step(action)
            _block_public_step(jax, active)
        active_guard_passed = True
        active_error = None
    except Exception as error:  # pragma: no cover - only on a broken GPU runtime
        active_guard_passed = False
        active_error = type(error).__name__
        env.reset(seed=FORMAL_RESET_SEED)
        active = env.step(action)
        _block_public_step(jax, active)

    selected_device = jnp.mod(jnp.arange(FORMAL_NUM_WORLDS), 97) == 3
    selected = np.asarray(selected_device, dtype=np.bool_)
    invalid = action.at[:, 0].set(jnp.where(selected_device, jnp.nan, action[:, 0]))
    before_tracks = env._track_batch
    before_episode_seeds = np.asarray(active[4]["episode_seed"], dtype=np.uint32)
    terminal = env.step(invalid)
    _block_public_step(jax, terminal)
    try:
        with jax.transfer_guard("disallow"):
            autoreset = env.step(action)
            _block_public_step(jax, autoreset)
        mixed_guard_passed = True
        mixed_error = None
    except Exception as error:  # pragma: no cover - only on a broken GPU runtime
        mixed_guard_passed = False
        mixed_error = type(error).__name__
        autoreset = terminal
    after_tracks = env._track_batch

    terminal_ids = np.asarray(terminal[4]["track_id"], dtype=np.uint32)
    reset_ids = np.asarray(autoreset[4]["track_id"], dtype=np.uint32)
    terminal_reasons = np.asarray(terminal[4]["termination_reason"], dtype=np.int32)
    terminal_flags = np.asarray(terminal[2], dtype=np.bool_)
    terminal_truncated = np.asarray(terminal[3], dtype=np.bool_)
    reset_episode_seeds = np.asarray(autoreset[4]["episode_seed"], dtype=np.uint32)
    reset_reward = np.asarray(autoreset[1], dtype=np.float32)
    reset_terminated = np.asarray(autoreset[2], dtype=np.bool_)
    reset_truncated = np.asarray(autoreset[3], dtype=np.bool_)
    old_ids = np.asarray(before_tracks.seed, dtype=np.uint32)
    current_ids = np.asarray(after_tracks.seed, dtype=np.uint32)
    initial_identity = initialize_episode_identities(FORMAL_RESET_SEED, FORMAL_NUM_WORLDS)
    expected_terminal_ids = _expected_track_ids(pool, initial_identity)
    advanced_identity = masked_next_episode(initial_identity, selected)
    expected_reset_ids = _expected_track_ids(pool, advanced_identity)
    terminal_ids_exact = np.array_equal(terminal_ids, expected_terminal_ids)
    reset_ids_exact = np.array_equal(reset_ids, expected_reset_ids)

    terminal_contract = bool(
        np.array_equal(terminal_ids, old_ids)
        and terminal_ids_exact
        and np.all(terminal_flags[selected])
        and not np.any(terminal_flags[~selected])
        and not np.any(terminal_truncated)
        and np.all(terminal_reasons[selected] == 3)
        and np.all(terminal_reasons[~selected] == 0)
        and _track_observation_matches(terminal[0], before_tracks, selected)
    )
    reset_contract = bool(
        np.array_equal(reset_ids, current_ids)
        and reset_ids_exact
        and np.all(reset_reward[selected] == 0.0)
        and not np.any(reset_terminated[selected])
        and not np.any(reset_truncated[selected])
        and np.all(reset_episode_seeds[selected] != before_episode_seeds[selected])
        and np.array_equal(
            reset_episode_seeds[~selected],
            before_episode_seeds[~selected],
        )
        and _track_observation_matches(autoreset[0], after_tracks, selected)
        and np.allclose(
            np.asarray(autoreset[0]["position"])[selected],
            np.asarray(after_tracks.start_pose)[selected, :2],
            rtol=0.0,
            atol=1.0e-5,
        )
        and np.allclose(
            np.asarray(autoreset[0]["yaw"])[selected],
            np.asarray(after_tracks.start_pose)[selected, 2],
            rtol=0.0,
            atol=1.0e-5,
        )
    )
    allowed = bool(
        np.asarray(jnp.all(_ids_allowed(jnp, autoreset[4]["track_id"], sorted_allowed_ids)))
    )
    unselected_preserved = _track_rows_preserved(before_tracks, after_tracks, selected)
    evidence = {
        "active_step": {"passed": active_guard_passed, "error_type": active_error},
        "mixed_next_step_autoreset": {
            "guard_passed": mixed_guard_passed,
            "error_type": mixed_error,
            "selected_world_count": int(np.count_nonzero(selected)),
            "terminal_reports_old_track": terminal_contract,
            "next_step_reports_current_reset_track": reset_contract,
            "terminal_track_ids_match_host_domain2_reference": terminal_ids_exact,
            "reset_track_ids_match_advanced_host_domain2_reference": reset_ids_exact,
            "expected_reset_track_id_uint32_sha256": _uint32_digest(expected_reset_ids),
            "actual_reset_track_id_uint32_sha256": _uint32_digest(reset_ids),
            "unselected_track_rows_bit_exact": unselected_preserved,
            "all_result_track_ids_allowed": allowed,
            "selected_track_id_changed_count": int(
                np.count_nonzero(reset_ids[selected] != terminal_ids[selected])
            ),
            "selected_expected_unique_track_id_count": int(
                np.unique(expected_reset_ids[selected]).size
            ),
            "selected_actual_unique_track_id_count": int(np.unique(reset_ids[selected]).size),
            "selected_world_diversity_ratio": (
                float(np.unique(reset_ids[selected]).size) / float(np.count_nonzero(selected))
            ),
            "same_track_can_be_resampled": True,
        },
    }
    mixed = evidence["mixed_next_step_autoreset"]
    mixed["passed"] = bool(
        mixed_guard_passed
        and terminal_contract
        and reset_contract
        and unselected_preserved
        and allowed
    )
    evidence["passed"] = bool(active_guard_passed and mixed["passed"])
    return evidence


def _timeout_step_bound(config: ProjectConfig, pool: TrackPool) -> int:
    episode = config.benchmark.episode
    dt = config.vehicle.simulation.control_dt_s
    lengths = np.asarray(pool.batch.length_m, dtype=np.float64)
    maximum_time = np.maximum(
        episode.minimum_timeout_s,
        lengths / episode.timeout_reference_speed_mps,
    )
    return int(np.max(np.ceil(maximum_time / dt)))


def _run_health_validation(
    env: Any,
    action: Any,
    *,
    config: ProjectConfig,
    pool: TrackPool,
    sorted_allowed_ids: Any,
    maximum_steps: int,
) -> dict[str, Any]:
    """Run bounded zero-action health checks with device-side reductions."""

    import jax
    import jax.numpy as jnp

    observation, info = env.reset(seed=FORMAL_RESET_SEED)
    jax.block_until_ready((observation, info["track_id"]))
    previous_episode_seed = jnp.asarray(info["episode_seed"], dtype=jnp.uint32)
    reason_counts = jnp.zeros((len(TERMINATION_NAMES),), dtype=jnp.int32)
    numerical_failure_events = jnp.asarray(0, dtype=jnp.int32)
    numerical_failure_worlds = jnp.zeros((FORMAL_NUM_WORLDS,), dtype=bool)
    disallowed_track_id_events = jnp.asarray(0, dtype=jnp.int32)
    disallowed_track_id_worlds = jnp.zeros((FORMAL_NUM_WORLDS,), dtype=bool)
    autoreset_count = jnp.asarray(0, dtype=jnp.int32)
    autoreset_worlds = jnp.zeros((FORMAL_NUM_WORLDS,), dtype=bool)
    timeout_bound = _timeout_step_bound(config, pool)
    required_steps = timeout_bound + 1
    executed_steps = min(maximum_steps, required_steps)
    final: tuple[Any, ...] | None = None
    for _ in range(executed_steps):
        final = env.step(action)
        observation, reward, _, _, info = final
        reasons = jnp.asarray(info["termination_reason"], dtype=jnp.int32)
        reason_counts += jnp.bincount(reasons, length=len(TERMINATION_NAMES))
        finite = _finite_dynamic_rows(jnp, observation, reward, info)
        numerical_failure_events += jnp.sum(~finite, dtype=jnp.int32)
        numerical_failure_worlds |= ~finite
        allowed = _ids_allowed(jnp, info["track_id"], sorted_allowed_ids)
        disallowed_track_id_events += jnp.sum(~allowed, dtype=jnp.int32)
        disallowed_track_id_worlds |= ~allowed
        episode_seed = jnp.asarray(info["episode_seed"], dtype=jnp.uint32)
        reset = episode_seed != previous_episode_seed
        autoreset_count += jnp.sum(reset, dtype=jnp.int32)
        autoreset_worlds |= reset
        previous_episode_seed = episode_seed
    assert final is not None
    jax.block_until_ready(
        (
            final[0],
            final[1],
            final[4]["track_id"],
            reason_counts,
            numerical_failure_events,
            numerical_failure_worlds,
            disallowed_track_id_events,
            disallowed_track_id_worlds,
            autoreset_count,
            autoreset_worlds,
        )
    )
    counts = np.asarray(reason_counts, dtype=np.int64)
    count_by_reason = {name: int(counts[index]) for index, name in enumerate(TERMINATION_NAMES)}
    failed_world_indices = np.flatnonzero(np.asarray(numerical_failure_worlds)).tolist()
    disallowed_world_indices = np.flatnonzero(np.asarray(disallowed_track_id_worlds)).tolist()
    reset_world_indices = np.flatnonzero(np.asarray(autoreset_worlds)).tolist()
    final_finite, final_failures = _all_public_finite(final)
    return {
        "maximum_steps": maximum_steps,
        "required_steps_for_all_timeouts_and_next_step_reset": required_steps,
        "executed_steps": executed_steps,
        "bound_sufficient": maximum_steps >= required_steps,
        "termination_event_counts": count_by_reason,
        "timeout_event_count": count_by_reason["timeout"],
        "unexpected_termination_event_count": sum(
            count_by_reason[name] for name in ("success", "off_track", "invalid_action")
        ),
        "autoreset_event_count": int(np.asarray(autoreset_count)),
        "autoreset_world_count": len(reset_world_indices),
        "autoreset_world_indices": reset_world_indices,
        "all_worlds_observed_timeout": count_by_reason["timeout"] >= FORMAL_NUM_WORLDS,
        "all_worlds_observed_autoreset": len(reset_world_indices) == FORMAL_NUM_WORLDS,
        "numerical_failure_event_count": int(np.asarray(numerical_failure_events)),
        "numerical_failure_world_count": len(failed_world_indices),
        "numerical_failure_world_indices": failed_world_indices,
        "disallowed_track_id_event_count": int(np.asarray(disallowed_track_id_events)),
        "disallowed_track_id_world_count": len(disallowed_world_indices),
        "disallowed_track_id_world_indices": disallowed_world_indices,
        "final_output_finite": final_finite,
        "final_nonfinite_fields": final_failures,
        "numerical_scope": {
            "every_health_step": [
                *(f"observation.{key}" for key in DYNAMIC_OBSERVATION_KEYS),
                "reward",
                "info.lap_time_s",
            ],
            "all_track_ids_checked_against_verified_manifest": True,
            "final_step_all_public_numeric_fields": True,
        },
    }


def _measure_steady_steps(
    jax: Any,
    env: Any,
    action: Any,
    *,
    steps: int,
    reset_seed: int,
) -> tuple[float, tuple[Any, ...]]:
    observation, info = env.reset(seed=reset_seed)
    jax.block_until_ready((observation, info["track_id"]))
    started = time.perf_counter()
    final: tuple[Any, ...] | None = None
    for _ in range(steps):
        final = env.step(action)
    assert final is not None
    _block_public_step(jax, final)
    return time.perf_counter() - started, final


def _run_no_sync_epoch(
    jax: Any,
    jnp: Any,
    env: Any,
    action: Any,
    sorted_allowed_ids: Any,
    *,
    label: str,
    steps: int,
    reset_seed: int,
    headline: bool,
) -> dict[str, Any]:
    """Run one reset/step epoch, fully settle effects, and release outputs before sampling."""

    seconds, final = _measure_steady_steps(
        jax,
        env,
        action,
        steps=steps,
        reset_seed=reset_seed,
    )
    finite, failures = _all_public_finite(final)
    track_ids_allowed = bool(
        np.asarray(jnp.all(_ids_allowed(jnp, final[4]["track_id"], sorted_allowed_ids)))
    )
    effects_barrier = getattr(jax, "effects_barrier", None)
    effects_barrier_used = callable(effects_barrier)
    settle_started = time.perf_counter()
    if effects_barrier_used:
        effects_barrier()
    del final
    collected_objects = gc.collect()
    settle_seconds = time.perf_counter() - settle_started
    transitions = FORMAL_NUM_WORLDS * steps
    return {
        "label": label,
        "steps": steps,
        "transitions": transitions,
        "seconds": seconds,
        "settle_seconds": settle_seconds,
        "environment_steps_per_second": steps / seconds,
        "transitions_per_second": transitions / seconds,
        "reset_seed": reset_seed,
        "performance_action": [0.0, 0.0],
        "per_step_host_synchronization": False,
        "full_final_tree_synchronized": True,
        "effects_barrier_before_memory_sample": effects_barrier_used,
        "final_output_released_before_memory_sample": True,
        "gc_before_memory_sample": True,
        "gc_collected_objects": collected_objects,
        "included_in_formal_throughput": headline,
        "included_in_formal_transition_count": headline,
        "final_output_finite": finite,
        "final_nonfinite_fields": failures,
        "final_track_ids_allowed": track_ids_allowed,
        "passed": bool(finite and track_ids_allowed and effects_barrier_used),
    }


def _host_rss_mib() -> tuple[float | None, str | None]:
    """Read current Linux resident-set size without adding a runtime dependency."""

    try:
        fields = Path("/proc/self/statm").read_text(encoding="utf-8").split()
        resident_pages = int(fields[1])
        page_size = int(os.sysconf("SC_PAGE_SIZE"))
    except (IndexError, OSError, TypeError, ValueError) as error:
        return None, f"procfs RSS unavailable: {type(error).__name__}"
    return resident_pages * page_size / (1024.0 * 1024.0), None


def _selected_gpu_memory_mib(gpu_uuid: str | None) -> tuple[dict[str, float], str | None]:
    """Return selected-device totals without persisting its private UUID."""

    if gpu_uuid is None:
        return {}, "selected physical GPU is unavailable"
    stdout, error = m4_benchmark._run_command(
        (
            "nvidia-smi",
            "--query-gpu=uuid,memory.total,memory.used,memory.free",
            "--format=csv,noheader,nounits",
        )
    )
    if stdout is None:
        return {}, error
    for line in stdout.splitlines():
        fields = [value.strip() for value in line.split(",")]
        if len(fields) != 4 or fields[0] != gpu_uuid:
            continue
        try:
            return {
                "total_mib": float(fields[1]),
                "used_mib": float(fields[2]),
                "free_mib": float(fields[3]),
            }, None
        except ValueError:
            break
    return {}, "selected GPU memory row was unavailable"


def _formal_memory_sample(device: Any, phase: str, gpu_uuid: str | None) -> dict[str, Any]:
    sample = m4_benchmark._memory_sample(device, phase, gpu_uuid)
    host_rss, host_error = _host_rss_mib()
    gpu_memory, gpu_error = _selected_gpu_memory_mib(gpu_uuid)
    sample.update(
        {
            "host_rss_mib": host_rss,
            "host_rss_error": host_error,
            "selected_gpu_memory_mib": gpu_memory,
            "selected_gpu_memory_error": gpu_error,
        }
    )
    return sample


def _number(value: Any) -> float | None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    converted = float(value)
    return converted if math.isfinite(converted) else None


def _linear_slope(values: Sequence[float]) -> float:
    mean_x = (len(values) - 1) / 2.0
    mean_y = sum(values) / len(values)
    denominator = sum((index - mean_x) ** 2 for index in range(len(values)))
    return (
        sum((index - mean_x) * (value - mean_y) for index, value in enumerate(values)) / denominator
        if denominator
        else 0.0
    )


def _series_delta_evidence(
    samples: Mapping[str, Mapping[str, Any]],
    phases: Sequence[str],
    getter: Any,
) -> dict[str, Any]:
    values = [_number(getter(samples.get(phase, {}))) for phase in phases]
    labelled = [
        {"phase": phase, "value": value} for phase, value in zip(phases, values, strict=True)
    ]
    if any(value is None for value in values):
        return {
            "available": False,
            "values": labelled,
            "window_deltas": [],
            "cumulative_deltas_from_baseline": [],
            "max_growth_from_baseline": None,
            "end_growth_from_baseline": None,
            "max_positive_window_growth": None,
            "monotonic_non_decreasing_with_positive_growth": None,
            "linear_slope_per_epoch": None,
            "measurement_end_growth_from_E1": None,
            "measurement_linear_slope_per_epoch": None,
        }

    numeric = [float(value) for value in values if value is not None]
    windows = [
        {
            "from_phase": first_phase,
            "to_phase": second_phase,
            "delta": second - first,
        }
        for first_phase, second_phase, first, second in zip(
            phases[:-1],
            phases[1:],
            numeric[:-1],
            numeric[1:],
            strict=True,
        )
    ]
    cumulative = [
        {"phase": phase, "delta": value - numeric[0]}
        for phase, value in zip(phases[1:], numeric[1:], strict=True)
    ]
    cumulative_values = [item["delta"] for item in cumulative]
    window_values = [item["delta"] for item in windows]
    return {
        "available": True,
        "values": labelled,
        "window_deltas": windows,
        "cumulative_deltas_from_baseline": cumulative,
        "max_growth_from_baseline": max(0.0, *cumulative_values),
        "end_growth_from_baseline": numeric[-1] - numeric[0],
        "max_positive_window_growth": max(0.0, *window_values),
        "monotonic_non_decreasing_with_positive_growth": bool(
            all(delta >= 0.0 for delta in window_values)
            and any(delta > 0.0 for delta in window_values)
        ),
        "linear_slope_per_epoch": _linear_slope(numeric),
        "measurement_end_growth_from_E1": numeric[-1] - numeric[1],
        "measurement_linear_slope_per_epoch": _linear_slope(numeric[1:]),
    }


def _stabilization_memory_delta(
    before: Mapping[str, Any],
    after: Mapping[str, Any],
) -> dict[str, Any]:
    """Record one-time allocator expansion separately from the formal leak measurement."""

    before_process = _number(before.get("process_vram_mib"))
    after_process = _number(after.get("process_vram_mib"))
    process_delta = (
        after_process - before_process
        if before_process is not None and after_process is not None
        else None
    )
    before_allocator = before.get("jax_allocator", {})
    after_allocator = after.get("jax_allocator", {})
    allocator_fields: dict[str, dict[str, float] | None] = {}
    for field in ("bytes_in_use", "peak_bytes_in_use", "pool_bytes"):
        first = _number(before_allocator.get(field))
        second = _number(after_allocator.get(field))
        allocator_fields[field] = (
            {
                "before": first,
                "after": second,
                "delta": second - first,
            }
            if first is not None and second is not None
            else None
        )
    before_host = _number(before.get("host_rss_mib"))
    after_host = _number(after.get("host_rss_mib"))
    return {
        "before_phase": before["phase"],
        "after_phase": after["phase"],
        "process_vram_mib": {
            "before": before_process,
            "after": after_process,
            "delta": process_delta,
        },
        "host_rss_mib": {
            "before": before_host,
            "after": after_host,
            "delta": (
                after_host - before_host
                if before_host is not None and after_host is not None
                else None
            ),
        },
        "jax_allocator_bytes": allocator_fields,
    }


def _pool_memory_report(samples: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """Build the fixed-E0 plateau evidence without adapting its baseline."""

    phases = [sample.get("phase") for sample in samples]
    if phases != list(EXPECTED_MEMORY_SAMPLE_PHASES):
        raise ValueError("memory samples must match the exact ordered phase protocol")
    by_phase = {str(sample["phase"]): sample for sample in samples}
    plateau_phases = (
        "allocator_stabilized_E0",
        *(f"post_stabilization_{label}" for label, _ in MEASUREMENT_EPOCHS),
    )
    process = _series_delta_evidence(
        by_phase,
        plateau_phases,
        lambda sample: sample.get("process_vram_mib"),
    )
    host_rss = _series_delta_evidence(
        by_phase,
        plateau_phases,
        lambda sample: sample.get("host_rss_mib"),
    )
    allocator = {
        field: _series_delta_evidence(
            by_phase,
            plateau_phases,
            lambda sample, field=field: sample.get("jax_allocator", {}).get(field),
        )
        for field in ("bytes_in_use", "pool_bytes", "peak_bytes_in_use")
    }
    process["growth_limit_mib"] = PROCESS_VRAM_GROWTH_LIMIT_MIB
    process["slope_limit_mib_per_epoch"] = MEMORY_SLOPE_LIMIT_MIB_PER_EPOCH
    process["passed"] = bool(
        process["available"]
        and process["max_growth_from_baseline"] <= PROCESS_VRAM_GROWTH_LIMIT_MIB
        and process["end_growth_from_baseline"] <= PROCESS_VRAM_GROWTH_LIMIT_MIB
        and process["linear_slope_per_epoch"] <= MEMORY_SLOPE_LIMIT_MIB_PER_EPOCH
        and process["measurement_linear_slope_per_epoch"] <= MEMORY_SLOPE_LIMIT_MIB_PER_EPOCH
    )
    host_rss["growth_limit_mib"] = HOST_RSS_GROWTH_LIMIT_MIB
    host_rss["slope_limit_mib_per_epoch"] = MEMORY_SLOPE_LIMIT_MIB_PER_EPOCH
    host_rss["passed"] = bool(
        host_rss["available"]
        and host_rss["max_growth_from_baseline"] <= HOST_RSS_GROWTH_LIMIT_MIB
        and host_rss["end_growth_from_baseline"] <= HOST_RSS_GROWTH_LIMIT_MIB
        and host_rss["linear_slope_per_epoch"] <= MEMORY_SLOPE_LIMIT_MIB_PER_EPOCH
        and host_rss["measurement_linear_slope_per_epoch"] <= MEMORY_SLOPE_LIMIT_MIB_PER_EPOCH
    )
    pool_bytes = allocator["pool_bytes"]
    pool_bytes["growth_tolerance_bytes"] = POOL_BYTES_GROWTH_TOLERANCE
    pool_bytes["passed"] = bool(
        pool_bytes["available"]
        and pool_bytes["max_growth_from_baseline"] <= POOL_BYTES_GROWTH_TOLERANCE
        and pool_bytes["end_growth_from_baseline"] <= POOL_BYTES_GROWTH_TOLERANCE
    )
    live_bytes = allocator["bytes_in_use"]
    live_bytes["growth_limit_bytes"] = LIVE_BYTES_GROWTH_LIMIT
    live_bytes["max_window_growth_limit_bytes"] = LIVE_BYTES_MAX_WINDOW_GROWTH
    live_bytes["slope_limit_bytes_per_epoch"] = LIVE_BYTES_SLOPE_LIMIT_PER_EPOCH
    live_bytes["passed"] = bool(
        live_bytes["available"]
        and live_bytes["max_growth_from_baseline"] <= LIVE_BYTES_GROWTH_LIMIT
        and live_bytes["end_growth_from_baseline"] <= LIVE_BYTES_GROWTH_LIMIT
        and live_bytes["measurement_end_growth_from_E1"] <= LIVE_BYTES_GROWTH_LIMIT
        and live_bytes["max_positive_window_growth"] <= LIVE_BYTES_MAX_WINDOW_GROWTH
        and live_bytes["linear_slope_per_epoch"] <= LIVE_BYTES_SLOPE_LIMIT_PER_EPOCH
        and live_bytes["measurement_linear_slope_per_epoch"] <= LIVE_BYTES_SLOPE_LIMIT_PER_EPOCH
    )
    peak_bytes = allocator["peak_bytes_in_use"]
    peak_bytes["growth_limit_bytes"] = PEAK_BYTES_GROWTH_LIMIT
    peak_bytes["max_window_growth_limit_bytes"] = PEAK_BYTES_GROWTH_LIMIT
    peak_bytes["passed"] = bool(
        peak_bytes["available"]
        and peak_bytes["max_growth_from_baseline"] <= PEAK_BYTES_GROWTH_LIMIT
        and peak_bytes["end_growth_from_baseline"] <= PEAK_BYTES_GROWTH_LIMIT
        and peak_bytes["max_positive_window_growth"] <= PEAK_BYTES_GROWTH_LIMIT
    )

    process_values = [
        value
        for sample in samples
        if (value := _number(sample.get("process_vram_mib"))) is not None
    ]
    gpu_totals = [
        value
        for sample in samples
        if (value := _number(sample.get("selected_gpu_memory_mib", {}).get("total_mib")))
        is not None
    ]
    gpu_free = [
        value
        for sample in samples
        if (value := _number(sample.get("selected_gpu_memory_mib", {}).get("free_mib"))) is not None
    ]
    peak_process = max(process_values, default=None)
    selected_total = min(gpu_totals, default=None)
    minimum_free = min(gpu_free, default=None)
    peak_fraction = (
        peak_process / selected_total
        if peak_process is not None and selected_total not in (None, 0.0)
        else None
    )
    fraction_passed = peak_fraction is not None and peak_fraction <= MAXIMUM_PROCESS_VRAM_FRACTION
    free_passed = minimum_free is not None and minimum_free >= MINIMUM_GPU_FREE_MIB
    headroom = {
        "peak_process_vram_mib": peak_process,
        "selected_gpu_total_mib": selected_total,
        "peak_process_vram_fraction": peak_fraction,
        "maximum_process_vram_fraction": MAXIMUM_PROCESS_VRAM_FRACTION,
        "minimum_sampled_gpu_free_mib": minimum_free,
        "minimum_gpu_free_mib": MINIMUM_GPU_FREE_MIB,
        "fraction_criterion_passed": fraction_passed,
        "free_criterion_passed": free_passed,
        "passed": bool(fraction_passed and free_passed),
    }
    initial_compiled = by_phase.get("after_initial_compile_and_warmup", {})
    stabilized = by_phase.get("allocator_stabilized_E0", {})
    cold_to_stabilized = _stabilization_memory_delta(initial_compiled, stabilized)
    return {
        "claim": "post-stabilization allocator plateau",
        "samples": list(samples),
        "baseline_phase": "allocator_stabilized_E0",
        "baseline_fixed_before_repeated_epochs": True,
        "comparison_phases": list(plateau_phases),
        "initial_compiled_to_stabilized": cold_to_stabilized,
        "post_stabilization": {
            "process_vram_mib": process,
            "host_rss_mib": host_rss,
            "jax_allocator_bytes": allocator,
        },
        "absolute_headroom": headroom,
        "peak_sampled_process_vram_mib": peak_process,
        "steady_process_vram_growth_mib": process["end_growth_from_baseline"],
        "steady_growth_limit_mib": PROCESS_VRAM_GROWTH_LIMIT_MIB,
        "steady_growth_within_limit": process["passed"],
        "baseline_after_allocator_stabilization": True,
        "formal_comparison": "fixed E0 baseline through distinct-seed E1-E3 epochs",
        "passed": bool(
            process["passed"]
            and pool_bytes["passed"]
            and live_bytes["passed"]
            and peak_bytes["passed"]
            and host_rss["passed"]
            and headroom["passed"]
        ),
    }


def _run_reset_heavy_measurement(
    env: Any,
    action: Any,
    pool: TrackPool,
    sorted_allowed_ids: Any,
    *,
    cycles: int,
) -> dict[str, Any]:
    """Measure maximal pool replacement: all worlds terminate, then NEXT_STEP reset."""

    import jax
    import jax.numpy as jnp

    invalid = action.at[:, 0].set(jnp.nan)
    observation, info = env.reset(seed=FORMAL_RESET_SEED)
    jax.block_until_ready((observation, info["track_id"]))
    preflight_terminal = env.step(invalid)
    preflight_reset = env.step(action)
    _block_public_step(jax, preflight_reset)
    initial_identity = initialize_episode_identities(FORMAL_RESET_SEED, FORMAL_NUM_WORLDS)
    expected_initial_ids = _expected_track_ids(pool, initial_identity)
    one_reset_identity = _advance_all_episodes(initial_identity, 1)
    expected_preflight_reset_ids = _expected_track_ids(pool, one_reset_identity)
    preflight_terminal_ids = np.asarray(preflight_terminal[4]["track_id"], dtype=np.uint32)
    preflight_reset_ids = np.asarray(preflight_reset[4]["track_id"], dtype=np.uint32)
    preflight_ids_exact = bool(
        np.array_equal(preflight_terminal_ids, expected_initial_ids)
        and np.array_equal(preflight_reset_ids, expected_preflight_reset_ids)
    )
    terminal_ok = bool(
        np.all(np.asarray(preflight_terminal[2], dtype=np.bool_))
        and not np.any(np.asarray(preflight_terminal[3], dtype=np.bool_))
        and np.all(np.asarray(preflight_terminal[4]["termination_reason"]) == 3)
    )
    reset_ok = bool(
        np.all(np.asarray(preflight_reset[1]) == 0.0)
        and not np.any(np.asarray(preflight_reset[2], dtype=np.bool_))
        and not np.any(np.asarray(preflight_reset[3], dtype=np.bool_))
    )

    observation, info = env.reset(seed=FORMAL_RESET_SEED)
    jax.block_until_ready((observation, info["track_id"]))
    started = time.perf_counter()
    terminal: tuple[Any, ...] | None = None
    final: tuple[Any, ...] | None = None
    for _ in range(cycles):
        terminal = env.step(invalid)
        final = env.step(action)
    assert terminal is not None and final is not None
    _block_public_step(jax, final)
    elapsed = time.perf_counter() - started
    final_allowed = bool(
        np.asarray(jnp.all(_ids_allowed(jnp, final[4]["track_id"], sorted_allowed_ids)))
    )
    final_ids = np.asarray(final[4]["track_id"], dtype=np.uint32)
    final_identity = _advance_all_episodes(initial_identity, cycles)
    expected_final_ids = _expected_track_ids(pool, final_identity)
    final_ids_exact = np.array_equal(final_ids, expected_final_ids)
    final_semantics = bool(
        np.all(np.asarray(final[1]) == 0.0)
        and not np.any(np.asarray(final[2], dtype=np.bool_))
        and not np.any(np.asarray(final[3], dtype=np.bool_))
    )
    environment_steps = 2 * cycles
    reset_events = FORMAL_NUM_WORLDS * cycles
    return {
        "cycles": cycles,
        "environment_steps": environment_steps,
        "transitions": FORMAL_NUM_WORLDS * environment_steps,
        "requested_reset_events": reset_events,
        "seconds": elapsed,
        "environment_steps_per_second": environment_steps / elapsed,
        "transitions_per_second": FORMAL_NUM_WORLDS * environment_steps / elapsed,
        "reset_events_per_second": reset_events / elapsed,
        "preflight_all_invalid_terminals": terminal_ok,
        "preflight_all_next_step_resets": reset_ok,
        "preflight_track_ids_match_host_domain2_reference": preflight_ids_exact,
        "final_reset_semantics_passed": final_semantics,
        "final_track_ids_allowed": final_allowed,
        "final_track_ids_match_advanced_host_domain2_reference": final_ids_exact,
        "expected_final_track_id_uint32_sha256": _uint32_digest(expected_final_ids),
        "actual_final_track_id_uint32_sha256": _uint32_digest(final_ids),
        "initial_unique_track_id_count": int(np.unique(expected_initial_ids).size),
        "final_expected_unique_track_id_count": int(np.unique(expected_final_ids).size),
        "final_actual_unique_track_id_count": int(np.unique(final_ids).size),
        "changed_track_id_count": int(np.count_nonzero(final_ids != expected_initial_ids)),
        "final_world_diversity_ratio": float(np.unique(final_ids).size) / FORMAL_NUM_WORLDS,
        "per_step_host_synchronization": False,
        "passed": bool(
            terminal_ok
            and reset_ok
            and preflight_ids_exact
            and final_semantics
            and final_allowed
            and final_ids_exact
        ),
    }


def _track_from_batch_row(batch: TrackBatch, index: int, generator_version: str) -> Track:
    return Track(
        seed=int(batch.seed[index]),
        generator_version=generator_version,
        centerline_m=batch.centerline_m[index],
        left_boundary_m=batch.left_boundary_m[index],
        right_boundary_m=batch.right_boundary_m[index],
        tangent=batch.tangent[index],
        curvature_1pm=batch.curvature_1pm[index],
        cumulative_s_m=batch.cumulative_s_m[index],
        track_mask=batch.track_mask[index],
        checkpoint_center_m=batch.checkpoint_center_m[index],
        checkpoint_tangent=batch.checkpoint_tangent[index],
        checkpoint_s_m=batch.checkpoint_s_m[index],
        checkpoint_mask=batch.checkpoint_mask[index],
        start_pose=batch.start_pose[index],
        point_count=int(batch.point_count[index]),
        checkpoint_count=int(batch.checkpoint_count[index]),
        length_m=float(batch.length_m[index]),
        width_m=float(batch.width_m[index]),
    )


def _fixed_tracks_for_reset(pool: TrackPool, reset_seed: int) -> tuple[Track, ...]:
    identity = initialize_episode_identities(reset_seed, FORMAL_NUM_WORLDS)
    selection_seeds = track_pool_seeds(identity)
    indices = np.remainder(selection_seeds, np.uint32(pool.size)).astype(np.int64)
    return tuple(
        _track_from_batch_row(pool.batch, int(index), pool.generator_version) for index in indices
    )


def _measure_fixed_track_baseline(
    jax: Any,
    config: ProjectConfig,
    tracks: Sequence[Track],
    action: Any,
    *,
    steps: int,
    warmup_steps: int,
    reset_seed: int,
) -> dict[str, Any]:
    create_started = time.perf_counter()
    env = _create_environment(
        num_envs=FORMAL_NUM_WORLDS,
        project_config=config,
        level_id=FORMAL_LEVEL_ID,
        backend="mjx_warp",
        tracks=tracks,
        render_mode=None,
    )
    create_seconds = time.perf_counter() - create_started
    try:
        observation, info = env.reset(seed=reset_seed)
        jax.block_until_ready((observation, info["track_id"]))
        first_started = time.perf_counter()
        first = env.step(action)
        _block_public_step(jax, first)
        first_step_compile_seconds = time.perf_counter() - first_started
        warm_started = time.perf_counter()
        warm: tuple[Any, ...] | None = None
        for _ in range(warmup_steps):
            warm = env.step(action)
        assert warm is not None
        _block_public_step(jax, warm)
        warmup_seconds = time.perf_counter() - warm_started
        steady_seconds, final = _measure_steady_steps(
            jax,
            env,
            action,
            steps=steps,
            reset_seed=reset_seed,
        )
        final_finite, final_failures = _all_public_finite(final)
        backend = env.backend
    finally:
        env.close()
    return {
        "backend": backend,
        "track_mode": "fixed_injected",
        "reset_seed": reset_seed,
        "steps": steps,
        "transitions": FORMAL_NUM_WORLDS * steps,
        "environment_create_seconds": create_seconds,
        "first_step_compile_seconds": first_step_compile_seconds,
        "warmup_seconds": warmup_seconds,
        "steady_seconds": steady_seconds,
        "environment_steps_per_second": steps / steady_seconds,
        "transitions_per_second": FORMAL_NUM_WORLDS * steps / steady_seconds,
        "final_output_finite": final_finite,
        "final_nonfinite_fields": final_failures,
        "per_step_host_synchronization": False,
    }


def _check(identifier: str, passed: bool, observed: Any, expected: Any) -> dict[str, Any]:
    return {
        "id": identifier,
        "passed": bool(passed),
        "observed": m4_benchmark._json_value(observed),
        "expected": m4_benchmark._json_value(expected),
    }


def _finite_positive(values: Sequence[Any]) -> bool:
    return all(math.isfinite(float(value)) and float(value) > 0.0 for value in values)


def _privacy_violations(value: Any, path: str = "report") -> list[str]:
    """Return schema locations containing an absolute local path or a GPU/general UUID."""

    failures: list[str] = []
    if isinstance(value, Mapping):
        for key, item in value.items():
            failures.extend(_privacy_violations(item, f"{path}.{key}"))
    elif isinstance(value, (tuple, list)):
        for index, item in enumerate(value):
            failures.extend(_privacy_violations(item, f"{path}[{index}]"))
    elif isinstance(value, str):
        absolute_path = (
            value.startswith("/")
            or value.startswith("file://")
            or _WINDOWS_ABSOLUTE_PATTERN.match(value) is not None
        )
        if absolute_path or _UUID_PATTERN.search(value) is not None:
            failures.append(path)
    return failures


def evaluate_report_gates(report: Mapping[str, Any]) -> list[dict[str, Any]]:
    """Evaluate formal M5 gates from serialized evidence; intentionally GPU-independent."""

    protocol = report["protocol"]
    assets = report["assets"]
    official_assets = report["official_assets"]
    admission = report["admission"]
    residency = report["pool_residency"]
    timing = report["timing"]
    baseline = report["fixed_track_baseline"]
    transfer = report["transfer_guard"]
    stabilization = report["allocator_stabilization"]
    measurement_epochs = report["measurement_epochs"]
    health = report["health"]
    reset_heavy = report["reset_heavy"]
    cache = report["executable_cache"]
    runtime = report["runtime"]
    source = report["source_evidence"]
    memory = report["memory"]
    determinism = report["deterministic_reset"]
    final_output = report["final_output"]
    expected_transitions = FORMAL_NUM_WORLDS * int(protocol["environment_steps"])
    expected_epoch_pairs = list(MEASUREMENT_EPOCHS)
    expected_extra_steps = (
        protocol["allocator_stabilization_steps"]
        + (len(MEASUREMENT_EPOCHS) - 1) * protocol["environment_steps"]
    )
    expected_extra_transitions = FORMAL_NUM_WORLDS * expected_extra_steps
    timing_values = (
        timing["environment_create_seconds"],
        timing["pool_upload_ready_seconds"],
        timing["reset_compile_seconds"],
        timing["first_step_compile_seconds"],
        timing["warmup_seconds"],
        timing["steady_seconds"],
        timing["environment_steps_per_second"],
        timing["transitions_per_second"],
        baseline["steady_seconds"],
        baseline["transitions_per_second"],
        reset_heavy["seconds"],
        reset_heavy["reset_events_per_second"],
    )
    before = source["before"]
    after = source["after"]
    privacy_failures = _privacy_violations(
        {key: value for key, value in report.items() if key not in {"checks", "privacy"}}
    )
    post_stabilization = memory["post_stabilization"]
    process_plateau = post_stabilization["process_vram_mib"]
    host_plateau = post_stabilization["host_rss_mib"]
    pool_plateau = post_stabilization["jax_allocator_bytes"]["pool_bytes"]
    live_plateau = post_stabilization["jax_allocator_bytes"]["bytes_in_use"]
    peak_plateau = post_stabilization["jax_allocator_bytes"]["peak_bytes_in_use"]
    headroom = memory["absolute_headroom"]
    sample_phases = [sample.get("phase") for sample in memory["samples"]]
    epoch_memory_phases = [
        stabilization.get("memory_sample_phase"),
        *(epoch.get("memory_sample_phase") for epoch in measurement_epochs),
    ]
    expected_epoch_memory_phases = [
        "allocator_stabilized_E0",
        *(f"post_stabilization_{label}" for label, _ in MEASUREMENT_EPOCHS),
    ]
    memory_phase_protocol_passed = bool(
        sample_phases == list(EXPECTED_MEMORY_SAMPLE_PHASES)
        and epoch_memory_phases == expected_epoch_memory_phases
    )
    recomputed_memory = (
        _pool_memory_report(memory["samples"]) if memory_phase_protocol_passed else None
    )
    peak_process_fraction = _number(headroom["peak_process_vram_fraction"])
    minimum_sampled_gpu_free = _number(headroom["minimum_sampled_gpu_free_mib"])
    fraction_criterion = bool(
        peak_process_fraction is not None and peak_process_fraction <= MAXIMUM_PROCESS_VRAM_FRACTION
    )
    free_criterion = bool(
        minimum_sampled_gpu_free is not None and minimum_sampled_gpu_free >= MINIMUM_GPU_FREE_MIB
    )
    return [
        _check(
            "protocol.schema",
            report["schema_version"] == REPORT_SCHEMA_VERSION,
            report["schema_version"],
            REPORT_SCHEMA_VERSION,
        ),
        _check(
            "protocol.version",
            report["protocol_version"] == PROTOCOL_VERSION,
            report["protocol_version"],
            PROTOCOL_VERSION,
        ),
        _check(
            "protocol.backend", protocol["backend"] == "mjx_warp", protocol["backend"], "mjx_warp"
        ),
        _check(
            "protocol.level",
            protocol["level_id"] == FORMAL_LEVEL_ID,
            protocol["level_id"],
            FORMAL_LEVEL_ID,
        ),
        _check(
            "protocol.world_count",
            protocol["num_worlds"] == FORMAL_NUM_WORLDS,
            protocol["num_worlds"],
            FORMAL_NUM_WORLDS,
        ),
        _check(
            "protocol.environment_steps",
            protocol["environment_steps"] == DEFAULT_ENVIRONMENT_STEPS,
            protocol["environment_steps"],
            DEFAULT_ENVIRONMENT_STEPS,
        ),
        _check(
            "protocol.allocator_stabilization_steps",
            protocol["allocator_stabilization_steps"] == DEFAULT_ALLOCATOR_STABILIZATION_STEPS
            and protocol["allocator_stabilization_transitions"]
            == FORMAL_NUM_WORLDS * DEFAULT_ALLOCATOR_STABILIZATION_STEPS,
            {
                "steps": protocol["allocator_stabilization_steps"],
                "transitions": protocol["allocator_stabilization_transitions"],
            },
            {
                "steps": DEFAULT_ALLOCATOR_STABILIZATION_STEPS,
                "transitions": FORMAL_NUM_WORLDS * DEFAULT_ALLOCATOR_STABILIZATION_STEPS,
            },
        ),
        _check(
            "protocol.repeated_epochs",
            protocol["headline_epoch"] == HEADLINE_EPOCH
            and protocol["measurement_epoch_labels"] == [label for label, _ in expected_epoch_pairs]
            and protocol["measurement_epoch_seeds"] == [seed for _, seed in expected_epoch_pairs]
            and protocol["measurement_epoch_count"] == len(expected_epoch_pairs)
            and len(
                {
                    protocol["allocator_stabilization_seed"],
                    *protocol["measurement_epoch_seeds"],
                }
            )
            == len(expected_epoch_pairs) + 1
            and protocol["extra_non_headline_steps"] == expected_extra_steps
            and protocol["extra_non_headline_transitions"] == expected_extra_transitions
            and protocol["total_long_run_steps"]
            == expected_extra_steps + protocol["environment_steps"]
            and protocol["total_long_run_transitions"]
            == expected_extra_transitions + expected_transitions
            and protocol["same_environment_E0_through_E3"] is True
            and protocol["environment_recreations_between_E0_E3"] == 0
            and protocol["jax_cache_clear_calls"] == 0,
            {
                "headline_epoch": protocol["headline_epoch"],
                "allocator_stabilization_seed": protocol["allocator_stabilization_seed"],
                "measurement_epoch_labels": protocol["measurement_epoch_labels"],
                "measurement_epoch_seeds": protocol["measurement_epoch_seeds"],
                "extra_non_headline_steps": protocol["extra_non_headline_steps"],
                "extra_non_headline_transitions": protocol["extra_non_headline_transitions"],
                "same_environment_E0_through_E3": protocol["same_environment_E0_through_E3"],
                "environment_recreations_between_E0_E3": protocol[
                    "environment_recreations_between_E0_E3"
                ],
                "jax_cache_clear_calls": protocol["jax_cache_clear_calls"],
            },
            "fixed distinct-seed E0 plus E1-E3 protocol with all extra work disclosed",
        ),
        _check(
            "protocol.transition_count",
            protocol["transitions"] == expected_transitions,
            protocol["transitions"],
            expected_transitions,
        ),
        _check(
            "protocol.warmup",
            protocol["warmup_steps"] >= DEFAULT_WARMUP_STEPS,
            protocol["warmup_steps"],
            f">= {DEFAULT_WARMUP_STEPS}",
        ),
        _check(
            "protocol.reset_seed",
            protocol["reset_seed"] == FORMAL_RESET_SEED,
            protocol["reset_seed"],
            FORMAL_RESET_SEED,
        ),
        _check(
            "protocol.reset_heavy_cycles",
            protocol["reset_heavy_cycles"] >= DEFAULT_RESET_HEAVY_CYCLES,
            protocol["reset_heavy_cycles"],
            f">= {DEFAULT_RESET_HEAVY_CYCLES}",
        ),
        _check(
            "protocol.device_action",
            protocol["action_device_platform"] == "gpu",
            protocol["action_device_platform"],
            "gpu",
        ),
        _check(
            "protocol.no_step_sync",
            protocol["per_step_host_synchronization"] is False,
            protocol["per_step_host_synchronization"],
            False,
        ),
        _check(
            "assets.formal_locations",
            assets["formal_manifest_location"] and assets["formal_cache_location"],
            [assets["formal_manifest_location"], assets["formal_cache_location"]],
            [True, True],
        ),
        _check(
            "assets.train_count",
            assets["track_count"] == FORMAL_TRAIN_TRACK_COUNT
            and assets["configured_track_count"] == FORMAL_TRAIN_TRACK_COUNT,
            [assets["track_count"], assets["configured_track_count"]],
            [FORMAL_TRAIN_TRACK_COUNT, FORMAL_TRAIN_TRACK_COUNT],
        ),
        _check(
            "assets.cache_integrity",
            assets["cache_matches_manifest_sha256"],
            assets["cache_matches_manifest_sha256"],
            True,
        ),
        _check(
            "assets.seed_integrity",
            assets["unique_seed_count"] == FORMAL_TRAIN_TRACK_COUNT,
            assets["unique_seed_count"],
            FORMAL_TRAIN_TRACK_COUNT,
        ),
        _check(
            "assets.geometry_integrity",
            assets["geometry_hash_count"] == FORMAL_TRAIN_TRACK_COUNT,
            assets["geometry_hash_count"],
            FORMAL_TRAIN_TRACK_COUNT,
        ),
        _check(
            "assets.geometry_admission",
            assets["geometry_admission_pass_count"] == FORMAL_TRAIN_TRACK_COUNT,
            assets["geometry_admission_pass_count"],
            FORMAL_TRAIN_TRACK_COUNT,
        ),
        _check(
            "assets.driveability_admission",
            assets["driveability_admission_pass_count"] == FORMAL_TRAIN_TRACK_COUNT,
            assets["driveability_admission_pass_count"],
            FORMAL_TRAIN_TRACK_COUNT,
        ),
        _check(
            "official_assets.complete",
            official_assets["passed"],
            official_assets,
            {"passed": True},
        ),
        _check(
            "admission.formal_report",
            admission["formal_report_location"],
            admission["formal_report_location"],
            True,
        ),
        _check(
            "admission.protocol_and_source",
            admission["passed"]
            and admission["status"] == "pass"
            and admission["schema_version"] == ADMISSION_REPORT_SCHEMA_VERSION
            and admission["protocol_version"] == ADMISSION_PROTOCOL_VERSION
            and admission["all_recomputed_gates_passed"]
            and admission["source_evidence_passed"],
            admission,
            "passing strict protocol and clean stable source evidence",
        ),
        _check(
            "admission.manifest_binding",
            set(admission["manifest_sha256_matches"])
            == {spec.split for spec in OFFICIAL_TRACK_SPLITS}
            and all(admission["manifest_sha256_matches"].values())
            and all(admission["artifact_names_match"].values())
            and all(admission["manifest_asset_sha256_matches"].values()),
            {
                "manifest_sha256_matches": admission["manifest_sha256_matches"],
                "artifact_names_match": admission["artifact_names_match"],
                "manifest_asset_sha256_matches": admission["manifest_asset_sha256_matches"],
            },
            "all four official artifacts match the admission report",
        ),
        _check(
            "admission.train_cache_binding",
            admission["train_cache_sha256_matches"]
            and admission["train_cache_sha256"] == assets["cache_sha256"],
            [admission["train_cache_sha256_matches"], admission["train_cache_sha256"]],
            [True, assets["cache_sha256"]],
        ),
        _check(
            "pool.resident",
            residency["track_count"] == FORMAL_TRAIN_TRACK_COUNT
            and residency["leaf_count"] == len(TrackBatch._fields)
            and residency["all_leaves_on_gpu"]
            and residency["byte_count_matches_host"],
            {
                "track_count": residency["track_count"],
                "leaf_count": residency["leaf_count"],
                "all_leaves_on_gpu": residency["all_leaves_on_gpu"],
                "byte_count_matches_host": residency["byte_count_matches_host"],
            },
            True,
        ),
        _check(
            "pool.seed_identity",
            residency["device_seed_uint32_sha256"] == assets["allowed_seed_uint32_sha256"],
            residency["device_seed_uint32_sha256"],
            assets["allowed_seed_uint32_sha256"],
        ),
        _check("reset.deterministic", determinism["passed"], determinism, {"passed": True}),
        _check(
            "transfer.active",
            transfer["active_step"]["passed"],
            transfer["active_step"],
            {"passed": True},
        ),
        _check(
            "transfer.mixed_reset",
            transfer["mixed_next_step_autoreset"]["passed"]
            and transfer["mixed_next_step_autoreset"][
                "terminal_track_ids_match_host_domain2_reference"
            ]
            and transfer["mixed_next_step_autoreset"][
                "reset_track_ids_match_advanced_host_domain2_reference"
            ],
            transfer["mixed_next_step_autoreset"],
            {"passed": True},
        ),
        _check(
            "transfer.mixed_diversity",
            transfer["mixed_next_step_autoreset"]["selected_actual_unique_track_id_count"]
            == transfer["mixed_next_step_autoreset"]["selected_expected_unique_track_id_count"]
            and transfer["mixed_next_step_autoreset"]["selected_actual_unique_track_id_count"] > 1,
            {
                "expected": transfer["mixed_next_step_autoreset"][
                    "selected_expected_unique_track_id_count"
                ],
                "actual": transfer["mixed_next_step_autoreset"][
                    "selected_actual_unique_track_id_count"
                ],
            },
            "equal and greater than one",
        ),
        _check(
            "allocator.stabilization",
            stabilization["passed"]
            and stabilization["label"] == "E0"
            and stabilization["steps"] == protocol["allocator_stabilization_steps"]
            and stabilization["transitions"]
            == FORMAL_NUM_WORLDS * protocol["allocator_stabilization_steps"]
            and stabilization["steps"] == DEFAULT_ALLOCATOR_STABILIZATION_STEPS
            and stabilization["reset_seed"] == ALLOCATOR_STABILIZATION_SEED
            and stabilization["per_step_host_synchronization"] is False
            and stabilization["full_final_tree_synchronized"] is True
            and stabilization["effects_barrier_before_memory_sample"] is True
            and stabilization["final_output_released_before_memory_sample"] is True
            and stabilization["gc_before_memory_sample"] is True
            and stabilization["included_in_formal_throughput"] is False
            and stabilization["included_in_formal_transition_count"] is False
            and stabilization["final_output_finite"]
            and stabilization["final_track_ids_allowed"]
            and _finite_positive(
                (
                    stabilization["seconds"],
                    stabilization["environment_steps_per_second"],
                    stabilization["transitions_per_second"],
                )
            ),
            stabilization,
            "one fixed-seed 10000-step settled E0 excluded from formal counts",
        ),
        _check(
            "allocator.repeated_epochs",
            len(measurement_epochs) == len(expected_epoch_pairs)
            and all(
                epoch["label"] == expected_label
                and epoch["reset_seed"] == expected_seed
                and epoch["steps"] == protocol["environment_steps"]
                and epoch["transitions"] == expected_transitions
                and epoch["per_step_host_synchronization"] is False
                and epoch["full_final_tree_synchronized"] is True
                and epoch["effects_barrier_before_memory_sample"] is True
                and epoch["final_output_released_before_memory_sample"] is True
                and epoch["gc_before_memory_sample"] is True
                and epoch["included_in_formal_throughput"] == (expected_label == HEADLINE_EPOCH)
                and epoch["included_in_formal_transition_count"]
                == (expected_label == HEADLINE_EPOCH)
                and epoch["final_output_finite"]
                and epoch["final_track_ids_allowed"]
                and epoch["passed"]
                and _finite_positive(
                    (
                        epoch["seconds"],
                        epoch["environment_steps_per_second"],
                        epoch["transitions_per_second"],
                    )
                )
                and _number(epoch["settle_seconds"]) is not None
                and epoch["settle_seconds"] >= 0.0
                for epoch, (expected_label, expected_seed) in zip(
                    measurement_epochs, expected_epoch_pairs, strict=True
                )
            ),
            measurement_epochs,
            "three settled, finite, distinct-seed epochs with E1 as the sole headline epoch",
        ),
        _check(
            "timing.finite_positive",
            _finite_positive(timing_values),
            timing_values,
            "all finite and positive",
        ),
        _check(
            "timing.pool_ratio",
            timing["pool_to_fixed_throughput_ratio"] >= MINIMUM_POOL_TO_FIXED_THROUGHPUT_RATIO,
            timing["pool_to_fixed_throughput_ratio"],
            f">= {MINIMUM_POOL_TO_FIXED_THROUGHPUT_RATIO}",
        ),
        _check(
            "timing.headline_epoch",
            timing["headline_epoch"] == HEADLINE_EPOCH
            and timing["headline_reset_seed"] == FORMAL_RESET_SEED
            and timing["steady_seconds"] == measurement_epochs[0]["seconds"]
            and timing["transitions_per_second"] == measurement_epochs[0]["transitions_per_second"],
            {
                "headline_epoch": timing["headline_epoch"],
                "headline_reset_seed": timing["headline_reset_seed"],
                "steady_seconds": timing["steady_seconds"],
            },
            "E1 timing and seed exactly",
        ),
        _check(
            "baseline.protocol",
            baseline["steps"] == protocol["environment_steps"]
            and baseline["transitions"] == protocol["transitions"]
            and baseline["reset_seed"] == timing["headline_reset_seed"]
            and baseline["final_output_finite"]
            and baseline["matches_pool_initial_selection"]
            and baseline["per_step_host_synchronization"] is False,
            baseline,
            "same step/transition count, finite, no per-step synchronization",
        ),
        _check(
            "reset_heavy.protocol",
            reset_heavy["passed"]
            and reset_heavy["preflight_track_ids_match_host_domain2_reference"]
            and reset_heavy["final_track_ids_match_advanced_host_domain2_reference"],
            reset_heavy,
            {"passed": True, "host_domain2_references": True},
        ),
        _check(
            "reset_heavy.diversity",
            reset_heavy["final_actual_unique_track_id_count"]
            == reset_heavy["final_expected_unique_track_id_count"]
            and reset_heavy["final_actual_unique_track_id_count"] > 1,
            {
                "expected": reset_heavy["final_expected_unique_track_id_count"],
                "actual": reset_heavy["final_actual_unique_track_id_count"],
            },
            "equal and greater than one",
        ),
        _check(
            "health.bound",
            health["bound_sufficient"],
            health["maximum_steps"],
            f">= {health['required_steps_for_all_timeouts_and_next_step_reset']}",
        ),
        _check(
            "health.timeout",
            health["all_worlds_observed_timeout"],
            health["timeout_event_count"],
            f">= {FORMAL_NUM_WORLDS}",
        ),
        _check(
            "health.autoreset",
            health["all_worlds_observed_autoreset"],
            health["autoreset_world_count"],
            FORMAL_NUM_WORLDS,
        ),
        _check(
            "health.unexpected_termination",
            health["unexpected_termination_event_count"] == 0,
            health["unexpected_termination_event_count"],
            0,
        ),
        _check(
            "health.numerical",
            health["numerical_failure_event_count"] == 0 and health["final_output_finite"],
            [health["numerical_failure_event_count"], health["final_nonfinite_fields"]],
            [0, []],
        ),
        _check(
            "health.allowed_track_ids",
            health["disallowed_track_id_event_count"] == 0,
            health["disallowed_track_id_event_count"],
            0,
        ),
        _check("cache.no_recompile", cache["passed"], cache, {"passed": True}),
        _check(
            "cache.epoch_plateau",
            cache["cache_sizes_stable_from_E0_through_E3"] is True
            and list(cache["epoch_snapshots"]) == ["E0", "E1", "E2", "E3"]
            and all(
                snapshot == cache["epoch_snapshots"]["E0"]
                for snapshot in cache["epoch_snapshots"].values()
            ),
            cache["epoch_snapshots"],
            "identical E0-E3 cache-size snapshots",
        ),
        _check(
            "runtime.jax_gpu",
            runtime["jax_device"]["platform"] == "gpu",
            runtime["jax_device"]["platform"],
            "gpu",
        ),
        _check(
            "runtime.nvidia",
            "NVIDIA" in str(runtime["jax_device"]["device_kind"]).upper(),
            runtime["jax_device"]["device_kind"],
            "contains NVIDIA",
        ),
        _check(
            "memory.raw_sample_binding",
            memory_phase_protocol_passed and memory == recomputed_memory,
            {
                "sample_phases": sample_phases,
                "epoch_memory_phases": epoch_memory_phases,
                "summary_matches_samples": memory == recomputed_memory,
            },
            {
                "sample_phases": list(EXPECTED_MEMORY_SAMPLE_PHASES),
                "epoch_memory_phases": expected_epoch_memory_phases,
                "summary_matches_samples": True,
            },
        ),
        _check(
            "memory.steady_growth",
            memory["claim"] == "post-stabilization allocator plateau"
            and memory["baseline_phase"] == "allocator_stabilized_E0"
            and memory["baseline_fixed_before_repeated_epochs"] is True
            and process_plateau["available"] is True
            and process_plateau["max_growth_from_baseline"] <= PROCESS_VRAM_GROWTH_LIMIT_MIB
            and process_plateau["end_growth_from_baseline"] <= PROCESS_VRAM_GROWTH_LIMIT_MIB
            and process_plateau["linear_slope_per_epoch"] <= MEMORY_SLOPE_LIMIT_MIB_PER_EPOCH
            and process_plateau["measurement_linear_slope_per_epoch"]
            <= MEMORY_SLOPE_LIMIT_MIB_PER_EPOCH
            and process_plateau["growth_limit_mib"] == PROCESS_VRAM_GROWTH_LIMIT_MIB
            and process_plateau["slope_limit_mib_per_epoch"] == MEMORY_SLOPE_LIMIT_MIB_PER_EPOCH
            and process_plateau["passed"] is True
            and memory["formal_comparison"]
            == "fixed E0 baseline through distinct-seed E1-E3 epochs",
            {
                "claim": memory["claim"],
                "process_vram_mib": process_plateau,
                "formal_comparison": memory["formal_comparison"],
            },
            "fixed E0, max/end <=64 MiB and fitted slopes <=4 MiB/epoch",
        ),
        _check(
            "memory.allocator_pool_plateau",
            pool_plateau["available"] is True
            and pool_plateau["max_growth_from_baseline"] <= POOL_BYTES_GROWTH_TOLERANCE
            and pool_plateau["end_growth_from_baseline"] <= POOL_BYTES_GROWTH_TOLERANCE
            and pool_plateau["growth_tolerance_bytes"] == POOL_BYTES_GROWTH_TOLERANCE
            and pool_plateau["passed"] is True,
            pool_plateau,
            "no max/end pool_bytes growth above fixed E0",
        ),
        _check(
            "memory.live_bytes_plateau",
            live_plateau["available"] is True
            and live_plateau["max_growth_from_baseline"] <= LIVE_BYTES_GROWTH_LIMIT
            and live_plateau["end_growth_from_baseline"] <= LIVE_BYTES_GROWTH_LIMIT
            and live_plateau["measurement_end_growth_from_E1"] <= LIVE_BYTES_GROWTH_LIMIT
            and live_plateau["max_positive_window_growth"] <= LIVE_BYTES_MAX_WINDOW_GROWTH
            and live_plateau["linear_slope_per_epoch"] <= LIVE_BYTES_SLOPE_LIMIT_PER_EPOCH
            and live_plateau["measurement_linear_slope_per_epoch"]
            <= LIVE_BYTES_SLOPE_LIMIT_PER_EPOCH
            and live_plateau["growth_limit_bytes"] == LIVE_BYTES_GROWTH_LIMIT
            and live_plateau["max_window_growth_limit_bytes"] == LIVE_BYTES_MAX_WINDOW_GROWTH
            and live_plateau["slope_limit_bytes_per_epoch"] == LIVE_BYTES_SLOPE_LIMIT_PER_EPOCH
            and live_plateau["passed"] is True,
            live_plateau,
            "max/end/E1-E3/window <=32 MiB and fitted slopes <=4 MiB/epoch",
        ),
        _check(
            "memory.allocator_peak_plateau",
            peak_plateau["available"] is True
            and peak_plateau["max_growth_from_baseline"] <= PEAK_BYTES_GROWTH_LIMIT
            and peak_plateau["end_growth_from_baseline"] <= PEAK_BYTES_GROWTH_LIMIT
            and peak_plateau["max_positive_window_growth"] <= PEAK_BYTES_GROWTH_LIMIT
            and peak_plateau["growth_limit_bytes"] == PEAK_BYTES_GROWTH_LIMIT
            and peak_plateau["max_window_growth_limit_bytes"] == PEAK_BYTES_GROWTH_LIMIT
            and peak_plateau["passed"] is True,
            peak_plateau,
            "max/end/window peak_bytes_in_use growth <=64 MiB",
        ),
        _check(
            "memory.host_rss_plateau",
            host_plateau["available"] is True
            and host_plateau["max_growth_from_baseline"] <= HOST_RSS_GROWTH_LIMIT_MIB
            and host_plateau["end_growth_from_baseline"] <= HOST_RSS_GROWTH_LIMIT_MIB
            and host_plateau["linear_slope_per_epoch"] <= MEMORY_SLOPE_LIMIT_MIB_PER_EPOCH
            and host_plateau["measurement_linear_slope_per_epoch"]
            <= MEMORY_SLOPE_LIMIT_MIB_PER_EPOCH
            and host_plateau["growth_limit_mib"] == HOST_RSS_GROWTH_LIMIT_MIB
            and host_plateau["slope_limit_mib_per_epoch"] == MEMORY_SLOPE_LIMIT_MIB_PER_EPOCH
            and host_plateau["passed"] is True,
            host_plateau,
            "host RSS max/end <=64 MiB and fitted slopes <=4 MiB/epoch",
        ),
        _check(
            "memory.absolute_headroom",
            headroom["maximum_process_vram_fraction"] == MAXIMUM_PROCESS_VRAM_FRACTION
            and headroom["minimum_gpu_free_mib"] == MINIMUM_GPU_FREE_MIB
            and headroom["fraction_criterion_passed"] == fraction_criterion
            and headroom["free_criterion_passed"] == free_criterion
            and headroom["passed"]
            == (headroom["fraction_criterion_passed"] and headroom["free_criterion_passed"])
            and headroom["passed"] is True,
            headroom,
            "peak process VRAM <=80% of total GPU VRAM and sampled free VRAM >=1 GiB",
        ),
        _check(
            "source.revision_stable",
            before["git_revision"] is not None and before["git_revision"] == after["git_revision"],
            [before["git_revision"], after["git_revision"]],
            "same non-null revision",
        ),
        _check(
            "source.hashes_stable",
            before["source_files_sha256"] == after["source_files_sha256"],
            before["source_files_sha256"] == after["source_files_sha256"],
            True,
        ),
        _check(
            "source.clean",
            before["relevant_source_clean"] is True
            and after["relevant_source_clean"] is True
            and before["tracked_worktree_clean"] is True
            and after["tracked_worktree_clean"] is True,
            [
                before["relevant_source_clean"],
                after["relevant_source_clean"],
                before["tracked_worktree_clean"],
                after["tracked_worktree_clean"],
            ],
            [True, True, True, True],
        ),
        _check("privacy.redacted", not privacy_failures, privacy_failures, []),
        _check(
            "final_output.finite",
            final_output["finite"] and final_output["all_track_ids_allowed"],
            {
                "nonfinite_fields": final_output["nonfinite_fields"],
                "all_track_ids_allowed": final_output["all_track_ids_allowed"],
            },
            {"nonfinite_fields": [], "all_track_ids_allowed": True},
        ),
    ]


def run_benchmark(
    options: BenchmarkOptions,
    *,
    project_root: str | Path = PROJECT_ROOT,
) -> dict[str, Any]:
    """Execute the formal M5 TrackPool protocol and return a strict JSON-compatible report."""

    import jax
    import jax.numpy as jnp

    root = Path(project_root).expanduser().resolve()
    source_before = _source_snapshot(root)
    config = load_project_config(root)
    if config.benchmark.official_level != FORMAL_LEVEL_ID:
        raise RuntimeError("ProjectConfig does not designate Level 1 as the official benchmark")
    if config.benchmark.train_track_count != FORMAL_TRAIN_TRACK_COUNT:
        raise RuntimeError(
            f"formal M5 requires {FORMAL_TRAIN_TRACK_COUNT} training Tracks, "
            f"got {config.benchmark.train_track_count}"
        )

    manifest_path = _resolve(root, options.manifest)
    cache_path = _resolve(root, options.cache)
    admission_report_path = _resolve(root, options.admission_report)
    official_verification, official_asset_evidence = _verify_official_asset_set(
        config,
        asset_directory=manifest_path.parent,
        train_cache_path=cache_path,
    )
    admission_evidence = _load_verified_admission_evidence(
        admission_report_path,
        config=config,
        asset_directory=manifest_path.parent,
        train_cache_path=cache_path,
        official_verification=official_verification,
    )
    admission_evidence["formal_report_location"] = _same_path(
        root,
        options.admission_report,
        DEFAULT_ADMISSION_REPORT,
    )
    _manifest, pool, asset_evidence = _load_verified_train_pool(
        config,
        manifest_path,
        cache_path,
    )
    if _manifest != official_verification.manifests["train"]:
        raise RuntimeError("train manifest changed after official asset-set verification")
    asset_evidence.update(
        {
            "formal_manifest_location": _same_path(root, options.manifest, DEFAULT_MANIFEST),
            "formal_cache_location": _same_path(root, options.cache, DEFAULT_CACHE),
            "manifest_repository_location": DEFAULT_MANIFEST.as_posix(),
            "cache_role": "verified local generated artifact; path deliberately not persisted",
        }
    )

    devices = jax.devices("gpu")
    if not devices:
        raise RuntimeError("JAX found no GPU device; use the Pixi gpu environment")
    device = devices[0]
    inventory, inventory_error = m4_benchmark._nvidia_inventory()
    runtime, gpu_uuid = m4_benchmark._runtime_evidence(device, inventory, inventory_error)
    cuda_visible = runtime.pop("cuda_visible_devices", None)
    runtime["jax_device"].pop("description", None)
    nvidia_smi_error = runtime.pop("nvidia_smi_error", None)
    gpu_selection_error = runtime.pop("gpu_selection_error", None)
    runtime["cuda_visible_devices_configured"] = bool(cuda_visible)
    runtime["nvidia_smi_available"] = nvidia_smi_error is None
    runtime["gpu_selection_succeeded"] = gpu_selection_error is None
    allowed_ids_host = np.sort(np.asarray(pool.batch.seed, dtype=np.uint32))
    allowed_ids_device = jax.device_put(allowed_ids_host, device=device)
    action = jax.device_put(
        jnp.zeros((FORMAL_NUM_WORLDS, 2), dtype=jnp.float32),
        device=device,
    )
    memory_samples = [_formal_memory_sample(device, "before_environment", gpu_uuid)]

    create_started = time.perf_counter()
    env = _create_environment(
        num_envs=FORMAL_NUM_WORLDS,
        project_config=config,
        level_id=FORMAL_LEVEL_ID,
        backend="mjx_warp",
        track_pool=pool,
        render_mode=None,
    )
    environment_create_seconds = time.perf_counter() - create_started
    try:
        pool_residency = _pool_residency_evidence(
            jax,
            env,
            asset_evidence["pool_array_bytes"],
        )
        pool_upload_ready_seconds = time.perf_counter() - create_started
        memory_samples.append(_formal_memory_sample(device, "after_environment_create", gpu_uuid))

        reset_started = time.perf_counter()
        reset_observation, reset_info = env.reset(seed=FORMAL_RESET_SEED)
        jax.block_until_ready((reset_observation, reset_info["track_id"]))
        reset_compile_seconds = time.perf_counter() - reset_started

        deterministic_reset, repeated_reset_seconds = _deterministic_reset_evidence(
            jax,
            jnp,
            env,
            allowed_ids_device,
        )

        first_started = time.perf_counter()
        first = env.step(action)
        _block_public_step(jax, first)
        first_step_compile_seconds = time.perf_counter() - first_started

        warm_started = time.perf_counter()
        warm: tuple[Any, ...] | None = None
        for _ in range(options.warmup_steps):
            warm = env.step(action)
        assert warm is not None
        _block_public_step(jax, warm)
        warmup_seconds = time.perf_counter() - warm_started
        cache_after_compile = _jit_cache_snapshot(env)
        memory_samples.append(
            _formal_memory_sample(device, "after_initial_compile_and_warmup", gpu_uuid)
        )

        transfer_guard = _run_transfer_and_mixed_reset_checks(
            env,
            action,
            pool,
            allowed_ids_device,
        )

        health = _run_health_validation(
            env,
            action,
            config=config,
            pool=pool,
            sorted_allowed_ids=allowed_ids_device,
            maximum_steps=options.health_max_steps,
        )
        memory_samples.append(_formal_memory_sample(device, "after_health_preflight", gpu_uuid))

        reset_heavy = _run_reset_heavy_measurement(
            env,
            action,
            pool,
            allowed_ids_device,
            cycles=options.reset_heavy_cycles,
        )
        memory_samples.append(
            _formal_memory_sample(device, "after_reset_heavy_preflight", gpu_uuid)
        )

        # Release every preflight output before fixing E0; the baseline never moves afterwards.
        del reset_observation, reset_info, first, warm
        jax.effects_barrier()
        gc.collect()
        memory_samples.append(
            _formal_memory_sample(device, "before_allocator_stabilization", gpu_uuid)
        )
        allocator_stabilization = _run_no_sync_epoch(
            jax,
            jnp,
            env,
            action,
            allowed_ids_device,
            label="E0",
            steps=options.allocator_stabilization_steps,
            reset_seed=ALLOCATOR_STABILIZATION_SEED,
            headline=False,
        )
        allocator_stabilization["memory_sample_phase"] = "allocator_stabilized_E0"
        memory_samples.append(_formal_memory_sample(device, "allocator_stabilized_E0", gpu_uuid))
        epoch_cache_snapshots: dict[str, dict[str, int | None]] = {"E0": _jit_cache_snapshot(env)}

        measurement_epochs: list[dict[str, Any]] = []
        for label, reset_seed in MEASUREMENT_EPOCHS:
            epoch = _run_no_sync_epoch(
                jax,
                jnp,
                env,
                action,
                allowed_ids_device,
                label=label,
                steps=options.environment_steps,
                reset_seed=reset_seed,
                headline=label == HEADLINE_EPOCH,
            )
            phase = f"post_stabilization_{label}"
            epoch["memory_sample_phase"] = phase
            measurement_epochs.append(epoch)
            memory_samples.append(_formal_memory_sample(device, phase, gpu_uuid))
            epoch_cache_snapshots[label] = _jit_cache_snapshot(env)

        cache_after_all_pool_work = _jit_cache_snapshot(env)
        executable_cache = _cache_evidence(cache_after_compile, cache_after_all_pool_work)
        _add_epoch_cache_evidence(executable_cache, epoch_cache_snapshots)
        headline_epoch = next(
            epoch for epoch in measurement_epochs if epoch["label"] == HEADLINE_EPOCH
        )
        pool_backend = env.backend
    finally:
        env.close()

    fixed_tracks = _fixed_tracks_for_reset(pool, FORMAL_RESET_SEED)
    fixed_track_id_digest = _uint32_digest([track.seed for track in fixed_tracks])
    expected_initial_digest = deterministic_reset["initial_track_id_uint32_sha256"]
    del env
    gc.collect()
    fixed_baseline = _measure_fixed_track_baseline(
        jax,
        config,
        fixed_tracks,
        action,
        steps=options.environment_steps,
        warmup_steps=options.warmup_steps,
        reset_seed=FORMAL_RESET_SEED,
    )
    fixed_baseline["initial_track_id_uint32_sha256"] = fixed_track_id_digest
    fixed_baseline["matches_pool_initial_selection"] = (
        fixed_track_id_digest == expected_initial_digest
    )
    jax.effects_barrier()
    gc.collect()
    memory_samples.append(_formal_memory_sample(device, "after_fixed_baseline", gpu_uuid))

    transitions = FORMAL_NUM_WORLDS * options.environment_steps
    pool_transitions_per_second = headline_epoch["transitions_per_second"]
    throughput_ratio = pool_transitions_per_second / fixed_baseline["transitions_per_second"]
    extra_non_headline_steps = (
        options.allocator_stabilization_steps
        + (len(MEASUREMENT_EPOCHS) - 1) * options.environment_steps
    )
    extra_non_headline_transitions = FORMAL_NUM_WORLDS * extra_non_headline_steps
    total_long_run_steps = extra_non_headline_steps + options.environment_steps
    total_long_run_transitions = FORMAL_NUM_WORLDS * total_long_run_steps
    memory_report = _pool_memory_report(memory_samples)
    allocator_stabilization["memory_delta_from_initial_compile"] = memory_report[
        "initial_compiled_to_stabilized"
    ]
    source_after = _source_snapshot(root)
    report: dict[str, Any] = {
        "schema_version": REPORT_SCHEMA_VERSION,
        "protocol_version": PROTOCOL_VERSION,
        "generated_at_utc": datetime.now(UTC).isoformat(),
        "benchmark_version": config.benchmark.version,
        "protocol": {
            "backend": pool_backend,
            "level_id": FORMAL_LEVEL_ID,
            "num_worlds": FORMAL_NUM_WORLDS,
            "environment_steps": options.environment_steps,
            "transitions": transitions,
            "allocator_stabilization_steps": options.allocator_stabilization_steps,
            "allocator_stabilization_transitions": (
                FORMAL_NUM_WORLDS * options.allocator_stabilization_steps
            ),
            "allocator_stabilization_seed": ALLOCATOR_STABILIZATION_SEED,
            "measurement_epoch_count": len(MEASUREMENT_EPOCHS),
            "measurement_epoch_labels": [label for label, _ in MEASUREMENT_EPOCHS],
            "measurement_epoch_seeds": [seed for _, seed in MEASUREMENT_EPOCHS],
            "headline_epoch": HEADLINE_EPOCH,
            "extra_non_headline_steps": extra_non_headline_steps,
            "extra_non_headline_transitions": extra_non_headline_transitions,
            "total_long_run_steps": total_long_run_steps,
            "total_long_run_transitions": total_long_run_transitions,
            "same_environment_E0_through_E3": True,
            "environment_recreations_between_E0_E3": 0,
            "jax_cache_clear_calls": 0,
            "warmup_steps": options.warmup_steps,
            "reset_seed": FORMAL_RESET_SEED,
            "reset_heavy_cycles": options.reset_heavy_cycles,
            "action_shape": [FORMAL_NUM_WORLDS, 2],
            "action_dtype": str(action.dtype),
            "action_device_platform": _array_device_platform(action),
            "performance_action": [0.0, 0.0],
            "pool_sampling": (
                "domain-separated uint32 SeedSequence modulo pool size, with replacement"
            ),
            "next_step_autoreset": True,
            "per_step_host_synchronization": False,
            "timing_method": (
                "enqueue consecutive VecCarRacingEnv.step calls, then synchronize the complete "
                "public final output once after each measured loop"
            ),
        },
        "assets": asset_evidence,
        "official_assets": official_asset_evidence,
        "admission": admission_evidence,
        "pool_residency": pool_residency,
        "deterministic_reset": deterministic_reset,
        "transfer_guard": transfer_guard,
        "allocator_stabilization": allocator_stabilization,
        "measurement_epochs": measurement_epochs,
        "timing": {
            "environment_create_seconds": environment_create_seconds,
            "pool_upload_ready_seconds": pool_upload_ready_seconds,
            "reset_compile_seconds": reset_compile_seconds,
            "two_cached_deterministic_resets_seconds": repeated_reset_seconds,
            "first_step_compile_seconds": first_step_compile_seconds,
            "warmup_seconds": warmup_seconds,
            "headline_epoch": HEADLINE_EPOCH,
            "headline_reset_seed": FORMAL_RESET_SEED,
            "steady_seconds": headline_epoch["seconds"],
            "environment_steps_per_second": headline_epoch["environment_steps_per_second"],
            "transitions_per_second": pool_transitions_per_second,
            "pool_to_fixed_throughput_ratio": throughput_ratio,
            "minimum_pool_to_fixed_throughput_ratio": (MINIMUM_POOL_TO_FIXED_THROUGHPUT_RATIO),
        },
        "fixed_track_baseline": fixed_baseline,
        "reset_heavy": reset_heavy,
        "health": health,
        "numerical": {
            "failure_event_count": health["numerical_failure_event_count"],
            "failure_world_count": health["numerical_failure_world_count"],
            "failure_world_indices": health["numerical_failure_world_indices"],
            "evidence_scope": health["numerical_scope"],
        },
        "executable_cache": executable_cache,
        "final_output": {
            "epoch": HEADLINE_EPOCH,
            "finite": headline_epoch["final_output_finite"],
            "nonfinite_fields": headline_epoch["final_nonfinite_fields"],
            "all_track_ids_allowed": headline_epoch["final_track_ids_allowed"],
        },
        "runtime": runtime,
        "memory": memory_report,
        "configuration": {
            "project": m4_benchmark._json_value(asdict(config)),
            "environment_steps": options.environment_steps,
            "allocator_stabilization_steps": options.allocator_stabilization_steps,
            "warmup_steps": options.warmup_steps,
            "health_max_steps": options.health_max_steps,
            "reset_heavy_cycles": options.reset_heavy_cycles,
        },
        "source_evidence": {"before": source_before, "after": source_after},
    }
    privacy_failures = _privacy_violations(report)
    report["privacy"] = {
        "absolute_path_or_uuid_violations": privacy_failures,
        "passed": not privacy_failures,
    }
    report["checks"] = evaluate_report_gates(report)
    report["status"] = "pass" if all(check["passed"] for check in report["checks"]) else "fail"
    return report


def write_strict_json(path: Path, payload: Mapping[str, Any]) -> None:
    """Persist strict JSON atomically while rejecting NaN and infinity."""

    privacy_failures = _privacy_violations(payload)
    if privacy_failures:
        raise ValueError(
            "refusing to persist a report containing an absolute path or UUID at: "
            + ", ".join(privacy_failures)
        )
    m4_benchmark.write_strict_json(path, payload)


def main(argv: list[str] | None = None) -> None:
    """Run, write evidence, and fail the process when any formal gate fails."""

    options = _parse_args(argv)
    report = run_benchmark(options)
    write_strict_json(options.output, report)
    print(f"M5 TrackPool status: {report['status']}")
    print(
        f"worlds={report['protocol']['num_worlds']} "
        f"pool_tracks={report['assets']['track_count']} "
        f"steps={report['protocol']['environment_steps']} "
        f"transitions/s={report['timing']['transitions_per_second']:.3f} "
        f"fixed_ratio={report['timing']['pool_to_fixed_throughput_ratio']:.3f}"
    )
    print(f"Wrote {options.output}")
    if report["status"] != "pass":
        failed = [check["id"] for check in report["checks"] if not check["passed"]]
        print(f"Failed gates: {', '.join(failed)}", file=sys.stderr)
        raise SystemExit(1)


if __name__ == "__main__":
    main()
