"""Benchmark the formal 1,024-world M4 ``VecCarRacingEnv`` GPU path.

The timed steady-state loop never synchronizes or copies values to the host per step.  Transfer,
timeout/autoreset, and numerical-health checks run outside that timing interval and are recorded as
separate evidence.
"""

from __future__ import annotations

import os

# Keep allocator behavior observable and select physical GPUs deterministically before JAX loads.
os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
os.environ.setdefault("CUDA_DEVICE_ORDER", "PCI_BUS_ID")

import argparse
import hashlib
import json
import math
import platform
import subprocess
import sys
import time
from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from typing import Any

import numpy as np

from controller_learning.config import ProjectConfig, load_project_config
from controller_learning.tracks import (
    Track,
    TrackGenerationError,
    generate_track_candidate,
    generation_spec_from_project,
    pack_track,
    track_capacity_from_project,
    validate_track_candidate,
    validation_spec_from_project,
)

REPORT_SCHEMA_VERSION = "controller-learning.m4-environment.v1"
PROTOCOL_VERSION = "m4-vec-car-racing-gpu-v1"
FORMAL_NUM_WORLDS = 1024
FORMAL_LEVEL_ID = 1
FORMAL_RESET_SEED = 20260710
DEFAULT_ENVIRONMENT_STEPS = 10_000
DEFAULT_WARMUP_STEPS = 8
DEFAULT_HEALTH_MAX_STEPS = 5_000
DEFAULT_MAX_TRACK_CANDIDATES = 100_000
DEFAULT_OUTPUT = Path("benchmarks/v0.1/m4_environment_report.json")
PROJECT_ROOT = Path(__file__).resolve().parents[1]
MEMORY_GROWTH_FLOOR_MIB = 64.0
MEMORY_GROWTH_FRACTION = 0.02
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
    "controller_learning/tracks/generator.py",
    "controller_learning/tracks/geometry.py",
    "controller_learning/tracks/specs.py",
    "controller_learning/tracks/types.py",
    "controller_learning/tracks/validator.py",
    "scripts/benchmark_racing_env.py",
)


@dataclass(frozen=True, slots=True)
class BenchmarkOptions:
    """Validated runtime controls that do not change the formal world count or backend."""

    output: Path = DEFAULT_OUTPUT
    environment_steps: int = DEFAULT_ENVIRONMENT_STEPS
    warmup_steps: int = DEFAULT_WARMUP_STEPS
    health_max_steps: int = DEFAULT_HEALTH_MAX_STEPS
    max_track_candidates: int = DEFAULT_MAX_TRACK_CANDIDATES

    def __post_init__(self) -> None:
        for name in ("environment_steps", "warmup_steps", "health_max_steps"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
                raise ValueError(f"{name} must be a positive integer")
        if (
            isinstance(self.max_track_candidates, bool)
            or not isinstance(self.max_track_candidates, int)
            or self.max_track_candidates < FORMAL_NUM_WORLDS
        ):
            raise ValueError(f"max_track_candidates must be at least {FORMAL_NUM_WORLDS}")


def _positive_integer(value: str) -> int:
    try:
        parsed = int(value)
    except ValueError as error:
        raise argparse.ArgumentTypeError("must be a positive integer") from error
    if parsed <= 0:
        raise argparse.ArgumentTypeError("must be a positive integer")
    return parsed


def _track_candidate_limit(value: str) -> int:
    parsed = _positive_integer(value)
    if parsed < FORMAL_NUM_WORLDS:
        raise argparse.ArgumentTypeError(f"must be at least {FORMAL_NUM_WORLDS}")
    return parsed


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT,
        help=f"Strict JSON report path (default: {DEFAULT_OUTPUT})",
    )
    parser.add_argument(
        "--steps",
        dest="environment_steps",
        type=_positive_integer,
        default=DEFAULT_ENVIRONMENT_STEPS,
        help="Timed environment steps (default: 10000)",
    )
    parser.add_argument(
        "--warmup-steps",
        type=_positive_integer,
        default=DEFAULT_WARMUP_STEPS,
        help="Untimed warm steps after first-step compilation (default: 8)",
    )
    parser.add_argument(
        "--health-max-steps",
        type=_positive_integer,
        default=DEFAULT_HEALTH_MAX_STEPS,
        help="Bound for the separate timeout/autoreset health run (default: 5000)",
    )
    parser.add_argument(
        "--max-track-candidates",
        type=_track_candidate_limit,
        default=DEFAULT_MAX_TRACK_CANDIDATES,
        help="Maximum contiguous seeds scanned to obtain 1024 valid Tracks",
    )
    return parser


def _parse_args(argv: list[str] | None = None) -> BenchmarkOptions:
    values = _build_parser().parse_args(argv)
    return BenchmarkOptions(
        output=values.output,
        environment_steps=values.environment_steps,
        warmup_steps=values.warmup_steps,
        health_max_steps=values.health_max_steps,
        max_track_candidates=values.max_track_candidates,
    )


def _json_value(value: Any) -> Any:
    """Convert dataclass/NumPy values to strict-JSON-compatible builtins."""

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
    return repr(value)


def write_strict_json(path: Path, payload: Mapping[str, Any]) -> None:
    """Atomically write indented JSON while rejecting NaN and infinity."""

    destination = path.expanduser().resolve()
    destination.parent.mkdir(parents=True, exist_ok=True)
    serialized = json.dumps(_json_value(payload), indent=2, sort_keys=True, allow_nan=False) + "\n"
    temporary = destination.with_name(f".{destination.name}.{os.getpid()}.tmp")
    temporary.write_text(serialized, encoding="utf-8")
    temporary.replace(destination)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


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
        raise FileNotFoundError(f"M4 benchmark source inputs are missing: {', '.join(missing)}")
    hashes = {path: _sha256(project_root / path) for path in RELEVANT_SOURCE_PATHS}
    status = _git(project_root, "status", "--porcelain", "--", *RELEVANT_SOURCE_PATHS)
    return {
        "git_revision": _git(project_root, "rev-parse", "HEAD"),
        "relevant_source_clean": None if status is None else not bool(status),
        "source_files_sha256": hashes,
    }


def _package_version(package: str) -> str | None:
    try:
        return version(package)
    except PackageNotFoundError:
        return None


def _run_command(command: Sequence[str]) -> tuple[str | None, str | None]:
    try:
        completed = subprocess.run(
            tuple(command),
            check=True,
            capture_output=True,
            text=True,
            timeout=15,
        )
    except (OSError, subprocess.SubprocessError) as error:
        return None, f"{type(error).__name__}: {error}"
    return completed.stdout.strip(), None


def _nvidia_inventory() -> tuple[list[dict[str, Any]], str | None]:
    fields = ("index", "name", "uuid", "driver_version", "memory.total")
    stdout, error = _run_command(
        (
            "nvidia-smi",
            f"--query-gpu={','.join(fields)}",
            "--format=csv,noheader,nounits",
        )
    )
    if stdout is None:
        return [], error
    inventory: list[dict[str, Any]] = []
    for line in stdout.splitlines():
        values = [value.strip() for value in line.split(",")]
        if len(values) != len(fields):
            continue
        try:
            inventory.append(
                {
                    "index": int(values[0]),
                    "name": values[1],
                    "uuid": values[2],
                    "driver_version": values[3],
                    "memory_total_mib": float(values[4]),
                }
            )
        except ValueError:
            continue
    return inventory, None if inventory else "nvidia-smi returned no parseable GPUs"


def _selected_gpu(device: Any, inventory: Sequence[Mapping[str, Any]]):
    logical_id = int(getattr(device, "id", 0))
    visible = os.environ.get("CUDA_VISIBLE_DEVICES")
    token: str | None = None
    if visible:
        tokens = [item.strip() for item in visible.split(",") if item.strip()]
        if logical_id >= len(tokens):
            return None, f"logical GPU {logical_id} is not in CUDA_VISIBLE_DEVICES"
        token = tokens[logical_id]
    if token is None or token.isdigit():
        physical_index = logical_id if token is None else int(token)
        selected = next((item for item in inventory if item["index"] == physical_index), None)
    else:
        selected = next(
            (
                item
                for item in inventory
                if item["uuid"] == token or str(item["uuid"]).startswith(token)
            ),
            None,
        )
    if selected is None:
        return None, f"cannot map logical JAX GPU {logical_id} to nvidia-smi inventory"
    return selected, None


def _process_vram_mib(gpu_uuid: str | None) -> tuple[float | None, str | None]:
    if gpu_uuid is None:
        return None, "selected physical GPU is unavailable"
    stdout, error = _run_command(
        (
            "nvidia-smi",
            "--query-compute-apps=gpu_uuid,pid,used_gpu_memory",
            "--format=csv,noheader,nounits",
        )
    )
    if stdout is None:
        return None, error
    total = 0.0
    found = False
    for line in stdout.splitlines():
        fields = [value.strip() for value in line.split(",")]
        if len(fields) != 3:
            continue
        try:
            process_id = int(fields[1])
            memory = float(fields[2])
        except ValueError:
            continue
        if fields[0] == gpu_uuid and process_id == os.getpid():
            found = True
            total += memory
    return (total if found else None), None


def _memory_sample(device: Any, phase: str, gpu_uuid: str | None) -> dict[str, Any]:
    process_vram, process_error = _process_vram_mib(gpu_uuid)
    try:
        raw_stats = device.memory_stats() or {}
        allocator = {
            str(key): value
            for key, raw in raw_stats.items()
            if isinstance((value := _json_value(raw)), (int, float, bool, str))
        }
        allocator_error = None
    except (RuntimeError, TypeError) as error:
        allocator = {}
        allocator_error = f"{type(error).__name__}: {error}"
    return {
        "phase": phase,
        "process_vram_mib": process_vram,
        "process_vram_error": process_error,
        "jax_allocator": allocator,
        "jax_allocator_error": allocator_error,
    }


def _allocator_bytes(sample: Mapping[str, Any]) -> float | None:
    allocator = sample.get("jax_allocator", {})
    for key in ("bytes_in_use", "bytes_reserved", "peak_bytes_in_use"):
        value = allocator.get(key)
        if isinstance(value, (int, float)):
            return float(value)
    return None


def _memory_report(samples: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    by_phase = {str(sample["phase"]): sample for sample in samples}
    before = by_phase.get("before_environment", {})
    after = by_phase.get("after_steady", {})
    process_values = [
        float(value)
        for sample in samples
        if isinstance((value := sample.get("process_vram_mib")), (int, float))
    ]
    allocator_values = [
        value for sample in samples if (value := _allocator_bytes(sample)) is not None
    ]
    before_process = before.get("process_vram_mib")
    after_process = after.get("process_vram_mib")
    process_growth = (
        float(after_process) - float(before_process)
        if isinstance(before_process, (int, float)) and isinstance(after_process, (int, float))
        else None
    )
    compiled = by_phase.get("after_compile_and_warmup", {})
    compiled_process = compiled.get("process_vram_mib")
    steady_growth = (
        float(after_process) - float(compiled_process)
        if isinstance(compiled_process, (int, float)) and isinstance(after_process, (int, float))
        else None
    )
    peak_process = max(process_values, default=None)
    growth_limit = max(
        MEMORY_GROWTH_FLOOR_MIB,
        MEMORY_GROWTH_FRACTION * peak_process if peak_process is not None else 0.0,
    )
    return {
        "samples": list(samples),
        "before_process_vram_mib": before_process,
        "after_process_vram_mib": after_process,
        "peak_sampled_process_vram_mib": peak_process,
        "process_vram_growth_mib": process_growth,
        "steady_process_vram_growth_mib": steady_growth,
        "steady_growth_limit_mib": growth_limit,
        "peak_sampled_jax_allocator_bytes": max(allocator_values, default=None),
        "steady_growth_evaluated": steady_growth is not None,
        "steady_growth_within_limit": (
            None if steady_growth is None else steady_growth <= growth_limit
        ),
    }


def _generate_valid_tracks(
    config: ProjectConfig,
    *,
    count: int = FORMAL_NUM_WORLDS,
    max_candidates: int = DEFAULT_MAX_TRACK_CANDIDATES,
) -> tuple[tuple[Track, ...], dict[str, Any]]:
    """Scan contiguous uint32 seeds and retain the first ``count`` geometrically valid Tracks."""

    generation = generation_spec_from_project(config)
    validation = validation_spec_from_project(config)
    capacity = track_capacity_from_project(config)
    accepted: list[Track] = []
    accepted_seeds: list[int] = []
    rejected: list[dict[str, Any]] = []
    rejection_counts: Counter[str] = Counter()
    for seed in range(max_candidates):
        try:
            candidate = generate_track_candidate(seed, generation)
        except TrackGenerationError as error:
            reason = f"generation:{error.reason}"
            rejection_counts[reason] += 1
            rejected.append({"seed": seed, "stage": "generation", "reason": error.reason})
            continue
        result = validate_track_candidate(candidate, validation)
        if not result.valid:
            reason = result.primary_reason or "unspecified"
            rejection_counts[f"validation:{reason}"] += 1
            rejected.append(
                {
                    "seed": seed,
                    "stage": "validation",
                    "reason": reason,
                    "reasons": list(result.reasons),
                }
            )
            continue
        try:
            track = pack_track(candidate, capacity)
        except TrackGenerationError as error:
            reason = f"packing:{error.reason}"
            rejection_counts[reason] += 1
            rejected.append({"seed": seed, "stage": "packing", "reason": error.reason})
            continue
        accepted.append(track)
        accepted_seeds.append(seed)
        if len(accepted) == count:
            break
    if len(accepted) != count:
        raise RuntimeError(
            f"only {len(accepted)} valid Tracks found in {max_candidates} contiguous seeds"
        )
    attempted = accepted_seeds[-1] + 1
    track_ids = [f"{track.generator_version}:{track.seed}" for track in accepted]
    return tuple(accepted), {
        "seed_start": 0,
        "seed_stop_exclusive": attempted,
        "candidate_limit": max_candidates,
        "attempted_count": attempted,
        "accepted_count": len(accepted),
        "rejected_count": len(rejected),
        "accepted_seeds": accepted_seeds,
        "accepted_track_ids": track_ids,
        "rejected_candidates": rejected,
        "rejection_counts": dict(sorted(rejection_counts.items())),
        "all_track_ids_distinct": len(set(track_ids)) == len(track_ids),
        "all_seeds_distinct": len(set(accepted_seeds)) == len(accepted_seeds),
        "generator_version": generation.generator_version,
        "geometry_validation_required": True,
    }


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
        if np.issubdtype(np.asarray(value).dtype, np.number) and not np.isfinite(value).all():
            failures.append(f"observation.{key}")
    for key, value in (("reward", reward), ("info.lap_time_s", info["lap_time_s"])):
        if not np.isfinite(np.asarray(value)).all():
            failures.append(key)
    return not failures, failures


def _run_transfer_guard_checks(env: Any, action: Any, *, reset_seed: int) -> dict[str, Any]:
    import jax
    import jax.numpy as jnp

    initial_observation, initial_info = env.reset(seed=reset_seed)
    jax.block_until_ready((initial_observation, initial_info["episode_seed"]))
    try:
        with jax.transfer_guard("disallow"):
            active = env.step(action)
            _block_public_step(jax, active)
        active_guard_passed = True
        active_error = None
    except Exception as error:  # pragma: no cover - only exercised on a broken GPU runtime
        active_guard_passed = False
        active_error = f"{type(error).__name__}: {error}"

    invalid = action.at[0, 0].set(jnp.nan)
    terminal = env.step(invalid)
    _block_public_step(jax, terminal)
    try:
        with jax.transfer_guard("disallow"):
            autoreset = env.step(action)
            _block_public_step(jax, autoreset)
        autoreset_guard_passed = True
        autoreset_error = None
    except Exception as error:  # pragma: no cover - only exercised on a broken GPU runtime
        autoreset_guard_passed = False
        autoreset_error = f"{type(error).__name__}: {error}"
        autoreset = terminal

    terminal_reason = np.asarray(terminal[4]["termination_reason"])
    terminal_flags = np.asarray(terminal[2])
    autoreset_reward = np.asarray(autoreset[1])
    autoreset_terminated = np.asarray(autoreset[2])
    autoreset_truncated = np.asarray(autoreset[3])
    initial_seeds = np.asarray(initial_info["episode_seed"])
    autoreset_seeds = np.asarray(autoreset[4]["episode_seed"])
    reset_observation_matches = all(
        np.allclose(
            np.asarray(autoreset[0][key])[0],
            np.asarray(initial_observation[key])[0],
            rtol=0.0,
            atol=1.0e-6,
        )
        for key in initial_observation
    )
    mixed_semantics = bool(
        terminal_flags[0]
        and terminal_reason[0] == 3
        and not np.any(terminal_flags[1:])
        and autoreset_reward[0] == 0.0
        and not autoreset_terminated[0]
        and not autoreset_truncated[0]
        and autoreset_seeds[0] != initial_seeds[0]
        and np.array_equal(autoreset_seeds[1:], initial_seeds[1:])
        and reset_observation_matches
    )
    return {
        "active_step": {"passed": active_guard_passed, "error": active_error},
        "mixed_next_step_autoreset": {
            "passed": autoreset_guard_passed and mixed_semantics,
            "guard_passed": autoreset_guard_passed,
            "semantics_passed": mixed_semantics,
            "error": autoreset_error,
            "invalid_world_index": 0,
            "invalid_terminal_reason": int(terminal_reason[0]),
            "reset_observation_matches": reset_observation_matches,
            "only_selected_episode_seed_changed": bool(
                autoreset_seeds[0] != initial_seeds[0]
                and np.array_equal(autoreset_seeds[1:], initial_seeds[1:])
            ),
        },
    }


def _finite_dynamic_rows(jnp: Any, observation: Mapping[str, Any], reward: Any, info: Any):
    finite = jnp.ones((FORMAL_NUM_WORLDS,), dtype=bool)
    for key in DYNAMIC_OBSERVATION_KEYS:
        values = jnp.asarray(observation[key])
        finite &= jnp.all(jnp.isfinite(values).reshape((FORMAL_NUM_WORLDS, -1)), axis=1)
    finite &= jnp.isfinite(reward)
    finite &= jnp.isfinite(info["lap_time_s"])
    return finite


def _timeout_step_bound(config: ProjectConfig, tracks: Sequence[Track]) -> int:
    episode = config.benchmark.episode
    dt = config.vehicle.simulation.control_dt_s
    return max(
        math.ceil(
            max(episode.minimum_timeout_s, track.length_m / episode.timeout_reference_speed_mps)
            / dt
        )
        for track in tracks
    )


def _run_health_validation(
    env: Any,
    action: Any,
    *,
    config: ProjectConfig,
    tracks: Sequence[Track],
    reset_seed: int,
    maximum_steps: int,
) -> dict[str, Any]:
    """Run a bounded untimed validation with device-side event/numerical reductions."""

    import jax
    import jax.numpy as jnp

    observation, info = env.reset(seed=reset_seed)
    jax.block_until_ready((observation, info["episode_seed"]))
    previous_seed = jnp.asarray(info["episode_seed"])
    reason_counts = jnp.zeros((len(TERMINATION_NAMES),), dtype=jnp.int32)
    numerical_failure_events = jnp.asarray(0, dtype=jnp.int32)
    numerical_failure_worlds = jnp.zeros((FORMAL_NUM_WORLDS,), dtype=bool)
    autoreset_count = jnp.asarray(0, dtype=jnp.int32)
    autoreset_worlds = jnp.zeros((FORMAL_NUM_WORLDS,), dtype=bool)
    timeout_bound = _timeout_step_bound(config, tracks)
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
        episode_seed = jnp.asarray(info["episode_seed"])
        reset = episode_seed != previous_seed
        autoreset_count += jnp.sum(reset, dtype=jnp.int32)
        autoreset_worlds |= reset
        previous_seed = episode_seed
    assert final is not None
    jax.block_until_ready(
        (
            final[0],
            final[1],
            reason_counts,
            numerical_failure_events,
            numerical_failure_worlds,
            autoreset_count,
            autoreset_worlds,
        )
    )
    counts = np.asarray(reason_counts, dtype=np.int64)
    count_by_reason = {name: int(counts[index]) for index, name in enumerate(TERMINATION_NAMES)}
    final_finite, final_failures = _all_public_finite(final)
    failed_world_indices = np.flatnonzero(np.asarray(numerical_failure_worlds)).tolist()
    reset_world_indices = np.flatnonzero(np.asarray(autoreset_worlds)).tolist()
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
        "numerical_scope": {
            "every_health_step": [
                *(f"observation.{key}" for key in DYNAMIC_OBSERVATION_KEYS),
                "reward",
                "info.lap_time_s",
            ],
            "final_step_all_public_numeric_fields": True,
        },
        "final_output_finite": final_finite,
        "final_nonfinite_fields": final_failures,
    }


def _runtime_evidence(device: Any, inventory: Sequence[Mapping[str, Any]], error: str | None):
    import jax
    import mujoco

    selected, selection_error = _selected_gpu(device, inventory)
    public_selected = (
        None
        if selected is None
        else {key: value for key, value in selected.items() if key != "uuid"}
    )
    return {
        "python_version": platform.python_version(),
        "platform": platform.platform(),
        "numpy_version": np.__version__,
        "jax_version": jax.__version__,
        "jaxlib_version": _package_version("jaxlib"),
        "mujoco_version": mujoco.__version__,
        "mujoco_mjx_version": _package_version("mujoco-mjx"),
        "warp_version": _package_version("warp-lang"),
        "jax_device": {
            "description": str(device),
            "platform": getattr(device, "platform", None),
            "device_kind": getattr(device, "device_kind", None),
            "id": getattr(device, "id", None),
        },
        "selected_nvidia_gpu": public_selected,
        "nvidia_smi_error": error,
        "gpu_selection_error": selection_error,
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "xla_python_client_preallocate": os.environ.get("XLA_PYTHON_CLIENT_PREALLOCATE"),
    }, (None if selected is None else str(selected["uuid"]))


def _check(identifier: str, passed: bool, observed: Any, expected: Any) -> dict[str, Any]:
    return {
        "id": identifier,
        "passed": bool(passed),
        "observed": _json_value(observed),
        "expected": _json_value(expected),
    }


def evaluate_report_gates(report: Mapping[str, Any]) -> list[dict[str, Any]]:
    """Evaluate every formal pass gate from report evidence (also used by CPU tests)."""

    protocol = report["protocol"]
    tracks = report["track_scan"]
    timing = report["timing"]
    transfer = report["transfer_guard"]
    health = report["health"]
    runtime = report["runtime"]
    source = report["source_evidence"]
    memory = report["memory"]
    transitions = FORMAL_NUM_WORLDS * int(protocol["environment_steps"])
    timing_values = (
        timing["reset_compile_seconds"],
        timing["first_step_compile_seconds"],
        timing["warmup_seconds"],
        timing["steady_seconds"],
        timing["environment_steps_per_second"],
        timing["transitions_per_second"],
    )
    memory_stable = memory["steady_growth_within_limit"]
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
            "protocol.reset_seed",
            protocol["reset_seed"] == FORMAL_RESET_SEED,
            protocol["reset_seed"],
            FORMAL_RESET_SEED,
        ),
        _check(
            "protocol.environment_steps",
            protocol["environment_steps"] == DEFAULT_ENVIRONMENT_STEPS,
            protocol["environment_steps"],
            DEFAULT_ENVIRONMENT_STEPS,
        ),
        _check(
            "protocol.warmup_steps",
            protocol["warmup_steps"] >= DEFAULT_WARMUP_STEPS,
            protocol["warmup_steps"],
            f">= {DEFAULT_WARMUP_STEPS}",
        ),
        _check(
            "tracks.accepted",
            tracks["accepted_count"] == FORMAL_NUM_WORLDS,
            tracks["accepted_count"],
            FORMAL_NUM_WORLDS,
        ),
        _check(
            "tracks.accounting",
            tracks["attempted_count"] == tracks["accepted_count"] + tracks["rejected_count"],
            tracks["attempted_count"],
            tracks["accepted_count"] + tracks["rejected_count"],
        ),
        _check(
            "tracks.distinct",
            tracks["all_track_ids_distinct"] and tracks["all_seeds_distinct"],
            {"track_ids": tracks["all_track_ids_distinct"], "seeds": tracks["all_seeds_distinct"]},
            True,
        ),
        _check(
            "action.device",
            protocol["action_device_platform"] == "gpu",
            protocol["action_device_platform"],
            "gpu",
        ),
        _check(
            "timing.transition_count",
            protocol["transitions"] == transitions,
            protocol["transitions"],
            transitions,
        ),
        _check(
            "timing.no_per_step_host_sync",
            protocol["per_step_host_synchronization"] is False,
            protocol["per_step_host_synchronization"],
            False,
        ),
        _check(
            "timing.finite_positive",
            all(math.isfinite(float(value)) and float(value) > 0 for value in timing_values),
            timing_values,
            "all finite and positive",
        ),
        _check(
            "transfer_guard.active",
            transfer["active_step"]["passed"],
            transfer["active_step"],
            {"passed": True},
        ),
        _check(
            "transfer_guard.mixed_autoreset",
            transfer["mixed_next_step_autoreset"]["passed"],
            transfer["mixed_next_step_autoreset"],
            {"passed": True},
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
            health["numerical_failure_event_count"] == 0,
            health["numerical_failure_event_count"],
            0,
        ),
        _check(
            "health.final_finite",
            health["final_output_finite"],
            health["final_nonfinite_fields"],
            [],
        ),
        _check(
            "runtime.jax_gpu",
            runtime["jax_device"]["platform"] == "gpu",
            runtime["jax_device"]["platform"],
            "gpu",
        ),
        _check(
            "source.revision_stable",
            source["before"]["git_revision"] is not None
            and source["before"]["git_revision"] == source["after"]["git_revision"],
            [source["before"]["git_revision"], source["after"]["git_revision"]],
            "same non-null revision",
        ),
        _check(
            "source.hashes_stable",
            source["before"]["source_files_sha256"] == source["after"]["source_files_sha256"],
            source["before"]["source_files_sha256"] == source["after"]["source_files_sha256"],
            True,
        ),
        _check(
            "source.clean",
            source["before"]["relevant_source_clean"] is True
            and source["after"]["relevant_source_clean"] is True,
            [source["before"]["relevant_source_clean"], source["after"]["relevant_source_clean"]],
            [True, True],
        ),
        _check(
            "memory.steady_growth",
            memory["peak_sampled_process_vram_mib"] is not None
            and memory["steady_process_vram_growth_mib"] is not None
            and memory_stable is True,
            memory_stable,
            "measurable process VRAM and steady_growth_within_limit=true",
        ),
    ]


def run_benchmark(
    options: BenchmarkOptions,
    *,
    project_root: str | Path = PROJECT_ROOT,
) -> dict[str, Any]:
    """Execute the formal M4 benchmark and return a strict-JSON-compatible report."""

    import jax
    import jax.numpy as jnp

    root = Path(project_root).expanduser().resolve()
    source_before = _source_snapshot(root)
    config = load_project_config(root)
    if config.benchmark.official_level != FORMAL_LEVEL_ID:
        raise RuntimeError("the project config does not designate Level 1 as the official Level")
    devices = jax.devices("gpu")
    if not devices:
        raise RuntimeError("JAX found no GPU device; use the Pixi gpu environment")
    device = devices[0]
    inventory, inventory_error = _nvidia_inventory()
    runtime, gpu_uuid = _runtime_evidence(device, inventory, inventory_error)
    memory_samples = [_memory_sample(device, "before_environment", gpu_uuid)]

    tracks, track_evidence = _generate_valid_tracks(
        config,
        max_candidates=options.max_track_candidates,
    )
    create_start = time.perf_counter()
    env = _create_environment(
        num_envs=FORMAL_NUM_WORLDS,
        project_config=config,
        level_id=FORMAL_LEVEL_ID,
        tracks=tracks,
        backend="mjx_warp",
        render_mode=None,
    )
    create_seconds = time.perf_counter() - create_start
    memory_samples.append(_memory_sample(device, "after_environment_create", gpu_uuid))
    action = jax.device_put(
        jnp.zeros((FORMAL_NUM_WORLDS, 2), dtype=jnp.float32),
        device=device,
    )
    try:
        reset_start = time.perf_counter()
        reset_observation, reset_info = env.reset(seed=FORMAL_RESET_SEED)
        jax.block_until_ready((reset_observation, reset_info["episode_seed"]))
        reset_compile_seconds = time.perf_counter() - reset_start

        first_start = time.perf_counter()
        first = env.step(action)
        _block_public_step(jax, first)
        first_step_compile_seconds = time.perf_counter() - first_start

        warm_start = time.perf_counter()
        warm: tuple[Any, ...] | None = None
        for _ in range(options.warmup_steps):
            warm = env.step(action)
        assert warm is not None
        _block_public_step(jax, warm)
        warmup_seconds = time.perf_counter() - warm_start
        memory_samples.append(_memory_sample(device, "after_compile_and_warmup", gpu_uuid))

        transfer_guard = _run_transfer_guard_checks(
            env,
            action,
            reset_seed=FORMAL_RESET_SEED,
        )

        measured_reset, measured_reset_info = env.reset(seed=FORMAL_RESET_SEED)
        jax.block_until_ready((measured_reset, measured_reset_info["episode_seed"]))
        steady_start = time.perf_counter()
        final: tuple[Any, ...] | None = None
        for _ in range(options.environment_steps):
            final = env.step(action)
        assert final is not None
        _block_public_step(jax, final)
        steady_seconds = time.perf_counter() - steady_start
        memory_samples.append(_memory_sample(device, "after_steady", gpu_uuid))

        health = _run_health_validation(
            env,
            action,
            config=config,
            tracks=tracks,
            reset_seed=FORMAL_RESET_SEED,
            maximum_steps=options.health_max_steps,
        )
        memory_samples.append(_memory_sample(device, "after_health", gpu_uuid))
    finally:
        env.close()

    transitions = FORMAL_NUM_WORLDS * options.environment_steps
    source_after = _source_snapshot(root)
    report: dict[str, Any] = {
        "schema_version": REPORT_SCHEMA_VERSION,
        "protocol_version": PROTOCOL_VERSION,
        "generated_at_utc": datetime.now(UTC).isoformat(),
        "benchmark_version": config.benchmark.version,
        "protocol": {
            "backend": env.backend,
            "level_id": FORMAL_LEVEL_ID,
            "num_worlds": FORMAL_NUM_WORLDS,
            "environment_steps": options.environment_steps,
            "transitions": transitions,
            "warmup_steps": options.warmup_steps,
            "reset_seed": FORMAL_RESET_SEED,
            "action_shape": [FORMAL_NUM_WORLDS, 2],
            "action_dtype": str(action.dtype),
            "action_device_platform": action.device.platform,
            "performance_action": [0.0, 0.0],
            "per_step_host_synchronization": False,
            "timing_method": (
                "enqueue consecutive VecCarRacingEnv.step calls, then synchronize the complete "
                "public final output once after the measured loop"
            ),
            "health_validation_timed_with_throughput": False,
        },
        "track_scan": track_evidence,
        "timing": {
            "environment_create_seconds": create_seconds,
            "reset_compile_seconds": reset_compile_seconds,
            "first_step_compile_seconds": first_step_compile_seconds,
            "warmup_seconds": warmup_seconds,
            "steady_seconds": steady_seconds,
            "environment_steps_per_second": options.environment_steps / steady_seconds,
            "transitions_per_second": transitions / steady_seconds,
        },
        "transfer_guard": transfer_guard,
        "health": health,
        "numerical": {
            "failure_event_count": health["numerical_failure_event_count"],
            "failure_world_count": health["numerical_failure_world_count"],
            "failure_world_indices": health["numerical_failure_world_indices"],
            "evidence_scope": health["numerical_scope"],
        },
        "final_output": {
            "finite": health["final_output_finite"],
            "nonfinite_fields": health["final_nonfinite_fields"],
        },
        "runtime": runtime,
        "memory": _memory_report(memory_samples),
        "configuration": {
            "project": _json_value(asdict(config)),
            "benchmark_options": _json_value(asdict(options)),
        },
        "source_evidence": {"before": source_before, "after": source_after},
    }
    report["checks"] = evaluate_report_gates(report)
    report["status"] = "pass" if all(check["passed"] for check in report["checks"]) else "fail"
    return report


def main(argv: list[str] | None = None) -> None:
    """Run the benchmark, persist all evidence, and fail the process on any gate."""

    options = _parse_args(argv)
    report = run_benchmark(options)
    write_strict_json(options.output, report)
    print(f"M4 environment status: {report['status']}")
    print(
        f"worlds={report['protocol']['num_worlds']} "
        f"steps={report['protocol']['environment_steps']} "
        f"transitions/s={report['timing']['transitions_per_second']:.3f}"
    )
    print(f"Wrote {options.output}")
    if report["status"] != "pass":
        failed = [check["id"] for check in report["checks"] if not check["passed"]]
        print(f"Failed gates: {', '.join(failed)}", file=sys.stderr)
        raise SystemExit(1)


if __name__ == "__main__":
    main()
