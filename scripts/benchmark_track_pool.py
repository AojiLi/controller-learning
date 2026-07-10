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
    track_pool_seeds,
)
from controller_learning.tracks.assets import (
    TrackAssetManifest,
    load_track_asset_manifest,
    load_track_batch_npz,
    sha256_file,
)
from controller_learning.tracks.hashing import track_batch_geometry_sha256
from controller_learning.tracks.pool import TrackPool
from controller_learning.tracks.types import Track, TrackBatch
from scripts import benchmark_racing_env as m4_benchmark

REPORT_SCHEMA_VERSION = "controller-learning.m5-track-pool.v1"
PROTOCOL_VERSION = "m5-track-pool-gpu-v1"
FORMAL_NUM_WORLDS = 1024
FORMAL_LEVEL_ID = 1
FORMAL_RESET_SEED = 20260710
FORMAL_TRAIN_TRACK_COUNT = 10_000
DEFAULT_ENVIRONMENT_STEPS = 10_000
DEFAULT_WARMUP_STEPS = 8
DEFAULT_HEALTH_MAX_STEPS = 5_000
DEFAULT_RESET_HEAVY_CYCLES = 64
MINIMUM_POOL_TO_FIXED_THROUGHPUT_RATIO = 0.75
DEFAULT_MANIFEST = Path("controller_learning/assets/tracks/v0.1/train.json")
DEFAULT_CACHE = Path(".track-cache/v0.1/train_pool.npz")
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
    environment_steps: int = DEFAULT_ENVIRONMENT_STEPS
    warmup_steps: int = DEFAULT_WARMUP_STEPS
    health_max_steps: int = DEFAULT_HEALTH_MAX_STEPS
    reset_heavy_cycles: int = DEFAULT_RESET_HEAVY_CYCLES

    def __post_init__(self) -> None:
        for name in (
            "environment_steps",
            "warmup_steps",
            "health_max_steps",
            "reset_heavy_cycles",
        ):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
                raise ValueError(f"{name} must be a positive integer")
        for name in ("output", "manifest", "cache"):
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
        environment_steps=values.environment_steps,
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


def _uint32_digest(values: Any) -> str:
    array = np.ascontiguousarray(np.asarray(values, dtype="<u4"))
    return hashlib.sha256(array.tobytes(order="C")).hexdigest()


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

    terminal_contract = bool(
        np.array_equal(terminal_ids, old_ids)
        and np.all(terminal_flags[selected])
        and not np.any(terminal_flags[~selected])
        and not np.any(terminal_truncated)
        and np.all(terminal_reasons[selected] == 3)
        and np.all(terminal_reasons[~selected] == 0)
        and _track_observation_matches(terminal[0], before_tracks, selected)
    )
    reset_contract = bool(
        np.array_equal(reset_ids, current_ids)
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
            "unselected_track_rows_bit_exact": unselected_preserved,
            "all_result_track_ids_allowed": allowed,
            "selected_track_id_changed_count": int(
                np.count_nonzero(reset_ids[selected] != terminal_ids[selected])
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


def _run_reset_heavy_measurement(
    env: Any,
    action: Any,
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
        "final_reset_semantics_passed": final_semantics,
        "final_track_ids_allowed": final_allowed,
        "per_step_host_synchronization": False,
        "passed": bool(terminal_ok and reset_ok and final_semantics and final_allowed),
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


def _fixed_tracks_for_formal_reset(pool: TrackPool) -> tuple[Track, ...]:
    identity = initialize_episode_identities(FORMAL_RESET_SEED, FORMAL_NUM_WORLDS)
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
        observation, info = env.reset(seed=FORMAL_RESET_SEED)
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
            reset_seed=FORMAL_RESET_SEED,
        )
        final_finite, final_failures = _all_public_finite(final)
        backend = env.backend
    finally:
        env.close()
    return {
        "backend": backend,
        "track_mode": "fixed_injected",
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
    residency = report["pool_residency"]
    timing = report["timing"]
    baseline = report["fixed_track_baseline"]
    transfer = report["transfer_guard"]
    health = report["health"]
    reset_heavy = report["reset_heavy"]
    cache = report["executable_cache"]
    runtime = report["runtime"]
    source = report["source_evidence"]
    memory = report["memory"]
    determinism = report["deterministic_reset"]
    final_output = report["final_output"]
    expected_transitions = FORMAL_NUM_WORLDS * int(protocol["environment_steps"])
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
            transfer["mixed_next_step_autoreset"]["passed"],
            transfer["mixed_next_step_autoreset"],
            {"passed": True},
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
            "baseline.protocol",
            baseline["steps"] == protocol["environment_steps"]
            and baseline["transitions"] == protocol["transitions"]
            and baseline["final_output_finite"]
            and baseline["matches_pool_initial_selection"]
            and baseline["per_step_host_synchronization"] is False,
            baseline,
            "same step/transition count, finite, no per-step synchronization",
        ),
        _check("reset_heavy.protocol", reset_heavy["passed"], reset_heavy, {"passed": True}),
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
            "memory.steady_growth",
            memory["peak_sampled_process_vram_mib"] is not None
            and memory["steady_process_vram_growth_mib"] is not None
            and memory["steady_growth_within_limit"] is True,
            memory["steady_growth_within_limit"],
            True,
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
    _manifest, pool, asset_evidence = _load_verified_train_pool(
        config,
        manifest_path,
        cache_path,
    )
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
    memory_samples = [m4_benchmark._memory_sample(device, "before_environment", gpu_uuid)]

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
        memory_samples.append(
            m4_benchmark._memory_sample(device, "after_environment_create", gpu_uuid)
        )

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
            m4_benchmark._memory_sample(device, "after_compile_and_warmup", gpu_uuid)
        )

        transfer_guard = _run_transfer_and_mixed_reset_checks(
            env,
            action,
            allowed_ids_device,
        )

        steady_seconds, steady_final = _measure_steady_steps(
            jax,
            env,
            action,
            steps=options.environment_steps,
            reset_seed=FORMAL_RESET_SEED,
        )
        memory_samples.append(m4_benchmark._memory_sample(device, "after_steady", gpu_uuid))

        health = _run_health_validation(
            env,
            action,
            config=config,
            pool=pool,
            sorted_allowed_ids=allowed_ids_device,
            maximum_steps=options.health_max_steps,
        )
        memory_samples.append(m4_benchmark._memory_sample(device, "after_health", gpu_uuid))

        reset_heavy = _run_reset_heavy_measurement(
            env,
            action,
            allowed_ids_device,
            cycles=options.reset_heavy_cycles,
        )
        memory_samples.append(m4_benchmark._memory_sample(device, "after_reset_heavy", gpu_uuid))
        cache_after_all_pool_work = _jit_cache_snapshot(env)
        executable_cache = _cache_evidence(cache_after_compile, cache_after_all_pool_work)
        steady_finite, steady_failures = _all_public_finite(steady_final)
        steady_ids_allowed = bool(
            np.asarray(
                jnp.all(
                    _ids_allowed(
                        jnp,
                        steady_final[4]["track_id"],
                        allowed_ids_device,
                    )
                )
            )
        )
        pool_backend = env.backend
    finally:
        env.close()

    fixed_tracks = _fixed_tracks_for_formal_reset(pool)
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
    )
    fixed_baseline["initial_track_id_uint32_sha256"] = fixed_track_id_digest
    fixed_baseline["matches_pool_initial_selection"] = (
        fixed_track_id_digest == expected_initial_digest
    )

    transitions = FORMAL_NUM_WORLDS * options.environment_steps
    pool_transitions_per_second = transitions / steady_seconds
    throughput_ratio = pool_transitions_per_second / fixed_baseline["transitions_per_second"]
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
        "pool_residency": pool_residency,
        "deterministic_reset": deterministic_reset,
        "transfer_guard": transfer_guard,
        "timing": {
            "environment_create_seconds": environment_create_seconds,
            "pool_upload_ready_seconds": pool_upload_ready_seconds,
            "reset_compile_seconds": reset_compile_seconds,
            "two_cached_deterministic_resets_seconds": repeated_reset_seconds,
            "first_step_compile_seconds": first_step_compile_seconds,
            "warmup_seconds": warmup_seconds,
            "steady_seconds": steady_seconds,
            "environment_steps_per_second": options.environment_steps / steady_seconds,
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
            "finite": steady_finite,
            "nonfinite_fields": steady_failures,
            "all_track_ids_allowed": steady_ids_allowed,
        },
        "runtime": runtime,
        "memory": m4_benchmark._memory_report(memory_samples),
        "configuration": {
            "project": m4_benchmark._json_value(asdict(config)),
            "environment_steps": options.environment_steps,
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
