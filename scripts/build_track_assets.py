"""Build the formal v0.1 Track pools through fixed-shape MJX-Warp admission."""

from __future__ import annotations

import os

# The complete train pool is intentionally large; never preallocate most device memory.
os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")

import argparse
import hashlib
import importlib.metadata
import json
import platform
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import jax
import jax.numpy as jnp
import mujoco
import numpy as np

from controller_learning.config import ProjectConfig, load_project_config
from controller_learning.envs.configuration import race_core_config_from_project
from controller_learning.envs.race_core import (
    RaceState,
    RaceTermination,
    project_to_track,
    reset_race_state,
    step_race_core,
)
from controller_learning.tracks.admission import (
    ADMISSION_PROTOCOL_VERSION,
    ADMISSION_REPORT_SCHEMA_VERSION,
    DRIVEABILITY_PROTOCOL_VERSION,
    FORMAL_ADMISSION_WORLDS,
    FORMAL_CONTROL_BLOCK_STEPS,
    FORMAL_SPLIT_RULES,
    AdmissionInfrastructureError,
    DriveabilityOutcome,
    admit_split,
    build_geometry_attempt,
    evaluate_admission_report,
    materialize_admitted_assets,
    require_global_admission_diagnostics,
    split_result_dict,
    validate_split_rules,
    verify_selected_disjointness,
    write_strict_json,
)
from controller_learning.tracks.assets import sha256_file
from controller_learning.tracks.hashing import track_geometry_sha256
from controller_learning.tracks.level0 import build_level0_candidate, build_level0_track
from controller_learning.tracks.official_assets import (
    DEFAULT_TRAIN_CACHE,
    OFFICIAL_TRACK_SPLITS,
    OfficialAssetVerification,
    official_track_asset_directory,
    verify_official_track_assets,
)
from controller_learning.tracks.specs import (
    generation_spec_from_project,
    track_capacity_from_project,
    validation_spec_from_project,
)
from controller_learning.tracks.types import Track, TrackBatch, stack_tracks
from controller_learning.tracks.validator import validate_track_candidate

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_ASSET_DIRECTORY = Path("controller_learning/assets/tracks/v0.1")
DEFAULT_TRAIN_CACHE_DIRECTORY = Path(".track-cache/v0.1")
DEFAULT_REPORT_PATH = Path("benchmarks/v0.1/m5_track_admission_report.json")
FORMAL_TARGET_SPEED_MPS = 4.0

RELEVANT_SOURCE_PATHS = (
    "controller_learning/config/loader.py",
    "controller_learning/config/models.py",
    "controller_learning/tracks/admission.py",
    "controller_learning/tracks/assets.py",
    "controller_learning/tracks/driveability.py",
    "controller_learning/tracks/generator.py",
    "controller_learning/tracks/geometry.py",
    "controller_learning/tracks/hashing.py",
    "controller_learning/tracks/level0.py",
    "controller_learning/tracks/official_assets.py",
    "controller_learning/tracks/specs.py",
    "controller_learning/tracks/types.py",
    "controller_learning/tracks/validator.py",
    "controller_learning/envs/race_core.py",
    "controller_learning/envs/configuration.py",
    "controller_learning/physics/actuation.py",
    "controller_learning/physics/model.py",
    "controller_learning/physics/mjx_warp.py",
    "controller_learning/assets/vehicle/car.xml",
    "scripts/build_track_assets.py",
    "configs/benchmark.toml",
    "configs/levels/level0.toml",
    "configs/levels/level1.toml",
    "configs/track.toml",
    "configs/vehicle.toml",
    "pyproject.toml",
    "pixi.lock",
)


@dataclass(frozen=True, slots=True)
class AdmissionOptions:
    """Filesystem outputs for the otherwise locked formal protocol."""

    asset_directory: Path = DEFAULT_ASSET_DIRECTORY
    train_cache_directory: Path = DEFAULT_TRAIN_CACHE_DIRECTORY
    output: Path = DEFAULT_REPORT_PATH


def _parse_args(arguments: list[str] | None = None) -> AdmissionOptions:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--asset-directory",
        type=Path,
        default=DEFAULT_ASSET_DIRECTORY,
        help="Repository directory for manifests and fixed validation/test assets",
    )
    parser.add_argument(
        "--train-cache-directory",
        type=Path,
        default=DEFAULT_TRAIN_CACHE_DIRECTORY,
        help="Local-only directory for the large train_pool.npz",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_REPORT_PATH,
        help="Formal strict-JSON admission report",
    )
    namespace = parser.parse_args(arguments)
    return AdmissionOptions(
        asset_directory=namespace.asset_directory,
        train_cache_directory=namespace.train_cache_directory,
        output=namespace.output,
    )


def _resolve_output(project_root: Path, path: Path) -> Path:
    return path if path.is_absolute() else project_root / path


def _formal_path_evidence(
    project_root: Path,
    options: AdmissionOptions,
) -> dict[str, bool]:
    """Return whether every output resolves to its one official v0.1 location."""

    root = project_root.resolve()
    expected_asset_directory = official_track_asset_directory(
        "0.1",
        package_root=root / "controller_learning",
    ).resolve()
    expected_train_directory = (root / DEFAULT_TRAIN_CACHE.parent).resolve()
    expected_report = (root / DEFAULT_REPORT_PATH).resolve()
    return {
        "official_asset_directory": (
            _resolve_output(root, options.asset_directory).resolve() == expected_asset_directory
        ),
        "official_train_cache_directory": (
            _resolve_output(root, options.train_cache_directory).resolve()
            == expected_train_directory
        ),
        "official_report_path": (
            _resolve_output(root, options.output).resolve() == expected_report
        ),
    }


def _require_formal_output_paths(
    project_root: Path,
    options: AdmissionOptions,
) -> dict[str, bool]:
    evidence = _formal_path_evidence(project_root, options)
    failed = [name for name, passed in evidence.items() if not passed]
    if failed:
        raise ValueError(
            "formal M5 admission requires the official output paths; invalid: " + ", ".join(failed)
        )
    return evidence


def _source_evidence(project_root: Path) -> dict[str, Any]:
    revision_result = subprocess.run(
        ("git", "rev-parse", "HEAD"),
        cwd=project_root,
        check=False,
        capture_output=True,
        text=True,
    )
    status_result = subprocess.run(
        ("git", "status", "--porcelain", "--", *RELEVANT_SOURCE_PATHS),
        cwd=project_root,
        check=False,
        capture_output=True,
        text=True,
    )
    return {
        "git_revision": (
            revision_result.stdout.strip()
            if revision_result.returncode == 0 and revision_result.stdout.strip()
            else None
        ),
        "relevant_source_clean": (
            not bool(status_result.stdout.strip()) if status_result.returncode == 0 else None
        ),
        "source_files_sha256": {
            relative: hashlib.sha256((project_root / relative).read_bytes()).hexdigest()
            for relative in RELEVANT_SOURCE_PATHS
        },
    }


def _nvidia_smi() -> dict[str, Any]:
    result = subprocess.run(
        (
            "nvidia-smi",
            "--query-gpu=name,memory.total,memory.used,driver_version",
            "--format=csv,noheader,nounits",
            "--id=0",
        ),
        check=False,
        capture_output=True,
        text=True,
    )
    if result.returncode != 0 or not result.stdout.strip():
        return {}
    fields = [field.strip() for field in result.stdout.splitlines()[0].split(",")]
    if len(fields) != 4:
        return {}
    return {
        "gpu_name": fields[0],
        "gpu_memory_total_mib": int(fields[1]),
        "gpu_memory_used_mib": int(fields[2]),
        "nvidia_driver_version": fields[3],
    }


def _runtime_evidence(device: Any) -> dict[str, Any]:
    return {
        "os": platform.system(),
        "kernel": platform.release(),
        "machine": platform.machine(),
        "python_version": sys.version.split()[0],
        "numpy_version": np.__version__,
        "jax_version": jax.__version__,
        "mujoco_version": mujoco.__version__,
        "mjx_warp_version": importlib.metadata.version("mujoco-mjx"),
        "physics_backend": "MJX-Warp",
        "jax_device": {
            "platform": str(getattr(device, "platform", "unknown")),
            "device_kind": str(getattr(device, "device_kind", "unknown")),
        },
        **_nvidia_smi(),
    }


def _artifact_readback_evidence(
    *,
    verification: OfficialAssetVerification,
    asset_directory: Path,
    train_cache_path: Path,
    materialized: dict[str, dict[str, str]],
) -> dict[str, Any]:
    """Prove the just-written manifests and all four assets form one official set."""

    expected_splits = tuple(spec.split for spec in OFFICIAL_TRACK_SPLITS)
    expected_fixed = tuple(spec.split for spec in OFFICIAL_TRACK_SPLITS if spec.package_asset)
    manifest_splits = tuple(verification.manifests)
    fixed_splits = tuple(verification.fixed_batches)
    if set(manifest_splits) != set(expected_splits):
        raise RuntimeError("official readback did not return all four manifests")
    if set(fixed_splits) != set(expected_fixed):
        raise RuntimeError("official readback did not return every fixed package asset")
    if not verification.train_cache_verified:
        raise RuntimeError("official readback did not verify the training cache")

    manifest_hashes: dict[str, str] = {}
    asset_hashes: dict[str, str] = {}
    for spec in OFFICIAL_TRACK_SPLITS:
        manifest = verification.manifests[spec.split]
        manifest_digest = sha256_file(asset_directory / spec.manifest_file)
        asset_path = (
            train_cache_path if spec.split == "train" else asset_directory / spec.asset_file
        )
        asset_digest = sha256_file(asset_path)
        emitted = materialized[spec.split]
        if emitted["manifest_sha256"] != manifest_digest:
            raise RuntimeError(f"{spec.split} manifest changed before official readback")
        if emitted["asset_sha256"] != asset_digest:
            raise RuntimeError(f"{spec.split} asset changed before official readback")
        if manifest.asset_sha256 != asset_digest:
            raise RuntimeError(f"{spec.split} readback asset digest does not match its manifest")
        manifest_hashes[spec.split] = manifest_digest
        asset_hashes[spec.split] = asset_digest

    return {
        "passed": True,
        "official_manifest_splits": list(expected_splits),
        "fixed_asset_splits": list(expected_fixed),
        "train_cache_verified": True,
        "manifest_files_sha256": manifest_hashes,
        "asset_files_sha256": asset_hashes,
    }


def _device_track_batch(batch: TrackBatch, device: Any) -> TrackBatch:
    return jax.tree.map(lambda value: jax.device_put(value, device), batch)


def _select_race_state(previous: RaceState, candidate: RaceState, active: jax.Array) -> RaceState:
    def select(old: jax.Array, new: jax.Array) -> jax.Array:
        mask = active.reshape((active.shape[0],) + (1,) * (old.ndim - 1))
        return jnp.where(mask, new, old)

    return jax.tree.map(select, previous, candidate)


def _outcome_name(reason: int) -> str:
    names = {
        int(RaceTermination.SUCCESS): "success",
        int(RaceTermination.OFF_TRACK): "off_track",
        int(RaceTermination.INVALID_ACTION): "invalid_action",
        int(RaceTermination.TIMEOUT): "timeout",
        5: "numerical_failure",
    }
    return names.get(reason, "numerical_failure")


class FixedShapeGpuAdmitter:
    """Reuse one 1024-world MJX-Warp executable set across every candidate chunk.

    MJX-Warp is dispatched one compiled control step at a time. Device work is synchronized only
    after a bounded block of 100 steps (or the final step). This avoids assuming that Warp custom
    calls can safely be nested inside an outer ``lax.scan`` while still making every host boundary
    explicit and bounded.
    """

    def __init__(self, project_config: ProjectConfig) -> None:
        # Keep CPU-only CLI inspection and unit tests independent from the optional Warp import.
        from controller_learning.physics.mjx_warp import MjxWarpVehicle
        from controller_learning.tracks.driveability import (
            ConservativeDriveabilityPolicyConfig,
            conservative_driveability_action,
        )

        self._project_config = project_config
        self._race_config = race_core_config_from_project(project_config)
        vehicle_config = project_config.vehicle
        self._policy_config = ConservativeDriveabilityPolicyConfig(
            target_speed_mps=FORMAL_TARGET_SPEED_MPS,
            wheelbase_m=vehicle_config.vehicle.wheelbase_m,
            maximum_steering_angle_rad=vehicle_config.actuator.max_steering_angle_rad,
            maximum_acceleration_mps2=vehicle_config.actuator.max_acceleration_mps2,
            maximum_deceleration_mps2=vehicle_config.actuator.max_deceleration_mps2,
        )
        self._policy_action = conservative_driveability_action
        started = time.perf_counter()
        self._vehicle = MjxWarpVehicle.create(
            vehicle_config,
            num_worlds=FORMAL_ADMISSION_WORLDS,
        )
        self.adapter_creation_s = time.perf_counter() - started
        self.compilation_s = 0.0
        self.measured_execution_s = 0.0
        self.host_to_device_s = 0.0
        self.executed_control_steps = 0
        self.executed_transitions = 0
        self.transfer_sync_count = 0
        self.compilation_sync_count = 0
        self.control_block_sync_count = 0
        self.result_readback_count = 0
        self.batch_calls = 0
        self._compiled_physics: Any | None = None
        self._compiled_policy: Any | None = None
        self._compiled_race: Any | None = None
        self._chunk_evidence: list[dict[str, Any]] = []

    @property
    def device(self) -> Any:
        return self._vehicle.device

    @property
    def compile_count(self) -> int:
        return int(self._compiled_physics is not None)

    def _compile(
        self,
        track_batch: TrackBatch,
        physics_state: Any,
        race_state: RaceState,
        projection: Any,
        view: Any,
    ) -> None:
        if self._compiled_physics is not None:
            return

        def policy_step(batch: TrackBatch, current_projection: Any, current_view: Any) -> Any:
            return self._policy_action(
                batch,
                current_projection,
                current_view,
                self._policy_config,
            )

        def race_step(
            batch: TrackBatch,
            current_race: RaceState,
            positions: jax.Array,
            invalid: jax.Array,
        ) -> Any:
            return step_race_core(
                batch,
                current_race,
                positions,
                invalid,
                self._race_config,
            )

        policy_function = jax.jit(policy_step)
        race_function = jax.jit(race_step)
        started = time.perf_counter()
        self._compiled_policy = policy_function.lower(track_batch, projection, view).compile()
        initial_actions = self._compiled_policy(track_batch, projection, view)
        self._compiled_physics = self._vehicle.lower_step(physics_state, initial_actions).compile()
        self._compiled_race = race_function.lower(
            track_batch,
            race_state,
            view.position_world_m[:, :2],
            jnp.zeros(FORMAL_ADMISSION_WORLDS, dtype=bool),
        ).compile()
        jax.block_until_ready(initial_actions)
        self.compilation_sync_count += 1
        self.compilation_s = time.perf_counter() - started

    def __call__(self, tracks: list[Track] | tuple[Track, ...]) -> tuple[DriveabilityOutcome, ...]:
        if not tracks or len(tracks) > FORMAL_ADMISSION_WORLDS:
            raise ValueError(f"physical admission requires 1..{FORMAL_ADMISSION_WORLDS} Tracks")
        if any(track.capacity != tracks[0].capacity for track in tracks):
            raise ValueError("all admission Tracks must share one fixed capacity")
        real_worlds = len(tracks)
        padded = [*tracks, *([tracks[-1]] * (FORMAL_ADMISSION_WORLDS - real_worlds))]
        host_batch = stack_tracks(padded)
        transfer_started = time.perf_counter()
        track_batch = _device_track_batch(host_batch, self.device)
        physics_state = self._vehicle.initial_state(track_batch.start_pose)
        race_state = reset_race_state(track_batch)
        view = self._vehicle.read_state(physics_state)
        projection = project_to_track(
            track_batch,
            view.position_world_m[:, :2],
            race_state.segment_index,
            self._race_config,
        )
        jax.block_until_ready((track_batch.seed, physics_state.data.qpos, projection.projected_s_m))
        self.transfer_sync_count += 1
        self.host_to_device_s += time.perf_counter() - transfer_started
        self._compile(track_batch, physics_state, race_state, projection, view)
        assert self._compiled_physics is not None
        assert self._compiled_policy is not None
        assert self._compiled_race is not None

        world_index = jnp.arange(FORMAL_ADMISSION_WORLDS)
        real_mask = world_index < real_worlds
        active = real_mask
        outcome_reason = jnp.zeros(FORMAL_ADMISSION_WORLDS, dtype=jnp.int32)
        outcome_step = jnp.zeros(FORMAL_ADMISSION_WORLDS, dtype=jnp.int32)
        maximum_lateral_error = jnp.zeros(FORMAL_ADMISSION_WORLDS, dtype=jnp.float32)
        maximum_speed = jnp.zeros(FORMAL_ADMISSION_WORLDS, dtype=jnp.float32)
        finite_per_world = jnp.ones(FORMAL_ADMISSION_WORLDS, dtype=bool)
        global_finite = jnp.asarray(True)
        time_monotonic = jnp.asarray(True)
        contact_overflow = jnp.asarray(False)
        constraint_overflow = jnp.asarray(False)
        unexpected_contact = jnp.asarray(False)
        maximum_steps = int(
            np.max(
                np.ceil(
                    np.maximum(
                        self._race_config.min_timeout_s,
                        np.asarray(host_batch.length_m[:real_worlds])
                        / self._race_config.timeout_reference_speed_mps,
                    )
                    / self._race_config.control_dt_s
                )
            )
        )

        measured_started = time.perf_counter()
        executed_steps = 0
        sync_count = 0
        for step_index in range(maximum_steps):
            actions = self._compiled_policy(track_batch, projection, view)
            stopped_actions = jnp.column_stack(
                (
                    jnp.zeros(FORMAL_ADMISSION_WORLDS, dtype=jnp.float32),
                    jnp.full(
                        FORMAL_ADMISSION_WORLDS,
                        -self._project_config.vehicle.actuator.max_deceleration_mps2,
                        dtype=jnp.float32,
                    ),
                )
            )
            actions = jnp.where(active[:, None], actions, stopped_actions)
            next_physics, applied, diagnostics = self._compiled_physics(physics_state, actions)
            next_view = self._vehicle.read_state(next_physics)
            candidate_race = self._compiled_race(
                track_batch,
                race_state,
                next_view.position_world_m[:, :2],
                applied.invalid_action,
            )
            numerical_failure = active & ~diagnostics.finite_per_world
            race_done = candidate_race.terminated | candidate_race.truncated
            new_done = active & (race_done | numerical_failure)
            reason = jnp.where(
                numerical_failure,
                jnp.int32(5),
                candidate_race.termination_reason,
            )
            outcome_reason = jnp.where(new_done, reason, outcome_reason)
            outcome_step = jnp.where(new_done, jnp.int32(step_index + 1), outcome_step)
            maximum_lateral_error = jnp.maximum(
                maximum_lateral_error,
                jnp.where(active, jnp.abs(candidate_race.projection.lateral_error_m), 0.0),
            )
            speed = jnp.linalg.norm(next_view.velocity_body_mps[:, :2], axis=1)
            maximum_speed = jnp.maximum(maximum_speed, jnp.where(active, speed, 0.0))
            finite_per_world &= diagnostics.finite_per_world
            global_finite &= diagnostics.finite
            time_monotonic &= diagnostics.time_monotonic
            contact_overflow |= diagnostics.contact_overflow
            constraint_overflow |= diagnostics.constraint_overflow
            unexpected_contact |= diagnostics.unexpected_contact
            race_state = _select_race_state(race_state, candidate_race.state, active)
            projection = candidate_race.projection
            physics_state = next_physics
            view = next_view
            active &= ~new_done
            executed_steps = step_index + 1
            block_boundary = executed_steps % FORMAL_CONTROL_BLOCK_STEPS == 0
            if block_boundary or executed_steps == maximum_steps:
                sync_count += 1
                block_values = jax.device_get(
                    (
                        jnp.any(active),
                        global_finite,
                        time_monotonic,
                        contact_overflow,
                        constraint_overflow,
                        unexpected_contact,
                    )
                )
                require_global_admission_diagnostics(
                    {
                        "finite": bool(block_values[1]),
                        "time_monotonic": bool(block_values[2]),
                        "contact_overflow": bool(block_values[3]),
                        "constraint_overflow": bool(block_values[4]),
                        "unexpected_contact": bool(block_values[5]),
                    }
                )
                if not bool(block_values[0]):
                    break
        host_values = jax.device_get(
            (
                outcome_reason,
                outcome_step,
                race_state.legal_progress_m,
                maximum_lateral_error,
                maximum_speed,
                finite_per_world,
                global_finite,
                time_monotonic,
                contact_overflow,
                constraint_overflow,
                unexpected_contact,
            )
        )
        result_readback_count = 1
        elapsed = time.perf_counter() - measured_started

        (
            host_reason,
            host_steps,
            host_progress,
            host_lateral,
            host_speed,
            host_finite,
            host_global_finite,
            host_time_monotonic,
            host_contact_overflow,
            host_constraint_overflow,
            host_unexpected_contact,
        ) = host_values
        host_reason = np.asarray(host_reason)[:real_worlds]
        host_steps = np.asarray(host_steps)[:real_worlds]
        host_progress = np.asarray(host_progress)[:real_worlds]
        host_lateral = np.asarray(host_lateral)[:real_worlds]
        host_speed = np.asarray(host_speed)[:real_worlds]
        host_finite = np.asarray(host_finite)[:real_worlds]
        global_diagnostics = {
            "finite": bool(host_global_finite),
            "time_monotonic": bool(host_time_monotonic),
            "contact_overflow": bool(host_contact_overflow),
            "constraint_overflow": bool(host_constraint_overflow),
            "unexpected_contact": bool(host_unexpected_contact),
        }
        require_global_admission_diagnostics(global_diagnostics)
        invalid_outcomes = [
            tracks[index].seed
            for index, reason in enumerate(host_reason)
            if int(reason)
            not in (
                int(RaceTermination.SUCCESS),
                int(RaceTermination.OFF_TRACK),
                int(RaceTermination.TIMEOUT),
            )
        ]
        if invalid_outcomes:
            raise AdmissionInfrastructureError(
                "formal admission produced an invalid internal outcome for seeds: "
                + ", ".join(str(seed) for seed in invalid_outcomes)
            )
        outcomes: list[DriveabilityOutcome] = []
        for index, track in enumerate(tracks):
            reason = int(host_reason[index])
            status = _outcome_name(reason)
            if not bool(host_finite[index]):
                raise AdmissionInfrastructureError(
                    f"formal admission produced non-finite per-world state for seed {track.seed}"
                )
            outcomes.append(
                DriveabilityOutcome(
                    seed=track.seed,
                    status=status,  # type: ignore[arg-type]
                    metrics={
                        "termination_step": int(host_steps[index]),
                        "lap_time_s": (
                            float(host_steps[index] * self._race_config.control_dt_s)
                            if status == "success"
                            else None
                        ),
                        "legal_progress_m": float(host_progress[index]),
                        "progress_fraction": float(host_progress[index] / track.length_m),
                        "maximum_abs_lateral_error_m": float(host_lateral[index]),
                        "maximum_planar_speed_mps": float(host_speed[index]),
                    },
                )
            )

        self.batch_calls += 1
        self.measured_execution_s += elapsed
        self.executed_control_steps += executed_steps
        self.executed_transitions += FORMAL_ADMISSION_WORLDS * executed_steps
        self.control_block_sync_count += sync_count
        self.result_readback_count += result_readback_count
        self._chunk_evidence.append(
            {
                "batch_index": self.batch_calls - 1,
                "candidate_worlds": real_worlds,
                "padded_worlds": FORMAL_ADMISSION_WORLDS - real_worlds,
                "executed_control_steps": executed_steps,
                "control_block_sync_count": sync_count,
                "result_readback_count": result_readback_count,
                "measured_execution_s": elapsed,
                "global_diagnostics": global_diagnostics,
            }
        )
        return tuple(outcomes)

    def evidence(self) -> dict[str, Any]:
        throughput = (
            self.executed_transitions / self.measured_execution_s
            if self.measured_execution_s > 0.0
            else None
        )
        host_sync_count = (
            self.transfer_sync_count
            + self.compilation_sync_count
            + self.control_block_sync_count
            + self.result_readback_count
        )
        return {
            "adapter_creation_s": self.adapter_creation_s,
            "compilation_s": self.compilation_s,
            "host_to_device_s": self.host_to_device_s,
            "measured_execution_s": self.measured_execution_s,
            "compiled_executable_sets": self.compile_count,
            "batch_calls": self.batch_calls,
            "executed_control_steps": self.executed_control_steps,
            "executed_transitions": self.executed_transitions,
            "transitions_per_second": throughput,
            "host_sync_count": host_sync_count,
            "transfer_sync_count": self.transfer_sync_count,
            "compilation_sync_count": self.compilation_sync_count,
            "control_block_sync_count": self.control_block_sync_count,
            "result_readback_count": self.result_readback_count,
            "chunks": list(self._chunk_evidence),
        }


def _validate_project_protocol(project_config: ProjectConfig) -> None:
    validate_split_rules(FORMAL_SPLIT_RULES)
    expected_counts = {
        "train": project_config.benchmark.train_track_count,
        "validation": project_config.benchmark.validation_track_count,
        "test": project_config.benchmark.test_track_count,
    }
    if any(rule.track_count != expected_counts[rule.split] for rule in FORMAL_SPLIT_RULES):
        raise ValueError("configs/benchmark.toml does not match the formal M5 split quotas")
    if project_config.benchmark.version != "0.1":
        raise ValueError("formal M5 admission is locked to benchmark version 0.1")
    if project_config.vehicle.vehicle.max_speed_mps < FORMAL_TARGET_SPEED_MPS:
        raise ValueError("formal admission target speed exceeds the vehicle speed limit")


def run_formal_admission(
    project_root: Path,
    options: AdmissionOptions,
) -> dict[str, Any]:
    """Run the formal Level 0 and Level 1 admission/materialization protocol."""

    path_evidence = _require_formal_output_paths(project_root, options)
    source_before = _source_evidence(project_root)
    started = time.perf_counter()
    project_config = load_project_config(project_root)
    _validate_project_protocol(project_config)
    generation_spec = generation_spec_from_project(project_config)
    validation_spec = validation_spec_from_project(project_config)
    capacity = track_capacity_from_project(project_config)
    admitter = FixedShapeGpuAdmitter(project_config)

    level0_candidate = build_level0_candidate()
    level0_validation = validate_track_candidate(level0_candidate, validation_spec)
    if not level0_validation.valid:
        raise RuntimeError(
            "Level 0 geometry failed formal validation: " + ", ".join(level0_validation.reasons)
        )
    level0_track = build_level0_track(capacity)
    level0_hash = track_geometry_sha256(level0_track)
    level0_outcome = admitter((level0_track,))[0]
    if level0_outcome.status != "success":
        raise RuntimeError(f"Level 0 failed physical admission: {level0_outcome.status}")

    def geometry_builder(seed: int) -> Any:
        return build_geometry_attempt(
            seed,
            generation_spec=generation_spec,
            validation_spec=validation_spec,
            capacity=capacity,
        )

    split_results = []
    excluded_hashes = frozenset({level0_hash})
    split_timing: dict[str, float] = {}
    for rule in FORMAL_SPLIT_RULES:
        split_started = time.perf_counter()
        result = admit_split(
            rule,
            geometry_builder=geometry_builder,
            driveability_admitter=admitter,
            admission_chunk_size=FORMAL_ADMISSION_WORLDS,
            excluded_geometry_hashes=excluded_hashes,
        )
        split_timing[rule.split] = time.perf_counter() - split_started
        if not result.complete:
            raise RuntimeError(
                f"{rule.split} exhausted its seed interval before reaching its quota"
            )
        split_results.append(result)
        excluded_hashes = frozenset((*excluded_hashes, *result.selected_hashes))

    disjointness = verify_selected_disjointness(
        level0_track,
        level0_hash,
        split_results,
    )
    asset_directory = _resolve_output(project_root, options.asset_directory)
    train_cache_directory = _resolve_output(project_root, options.train_cache_directory)
    artifacts = materialize_admitted_assets(
        benchmark_version=project_config.benchmark.version,
        asset_directory=asset_directory,
        train_cache_directory=train_cache_directory,
        level0_track=level0_track,
        level0_hash=level0_hash,
        split_results=split_results,
    )
    train_cache_path = train_cache_directory / "train_pool.npz"
    verification = verify_official_track_assets(
        project_config,
        asset_directory=asset_directory,
        train_cache_path=train_cache_path,
        require_train_cache=True,
    )
    artifact_readback = _artifact_readback_evidence(
        verification=verification,
        asset_directory=asset_directory,
        train_cache_path=train_cache_path,
        materialized=artifacts,
    )
    source_after = _source_evidence(project_root)
    report: dict[str, Any] = {
        "schema_version": ADMISSION_REPORT_SCHEMA_VERSION,
        "protocol_version": ADMISSION_PROTOCOL_VERSION,
        "status": "pending_gates",
        "protocol": {
            "benchmark_version": project_config.benchmark.version,
            "generator_version": generation_spec.generator_version,
            "driveability_protocol_version": DRIVEABILITY_PROTOCOL_VERSION,
            "formal_physics_backend": "MJX-Warp",
            "admission_worlds": FORMAL_ADMISSION_WORLDS,
            "fixed_shape_reused": admitter.compile_count == 1,
            "bounded_control_step_chunks": True,
            "control_block_steps": FORMAL_CONTROL_BLOCK_STEPS,
            "host_sync_semantics": (
                "one active check per bounded block, one final result readback per chunk, "
                "plus explicit transfer and first-batch compilation boundaries"
            ),
            "ascending_seed_order": True,
            "one_candidate_per_seed": True,
            "hidden_retry": False,
            "selection_rule": "first-N geometry-valid, physically successful, unique hashes",
            "target_speed_mps": FORMAL_TARGET_SPEED_MPS,
            "train_asset_storage": "local cache only",
            "official_output_paths": path_evidence,
        },
        "level0": {
            "seed": level0_track.seed,
            "geometry_sha256": level0_hash,
            "geometry_validation": "passed",
            "driveability_status": level0_outcome.status,
            "metrics": dict(level0_outcome.metrics),
        },
        "splits": {result.rule.split: split_result_dict(result) for result in split_results},
        "disjointness": disjointness,
        "artifacts": artifacts,
        "artifact_readback": artifact_readback,
        "timing": {
            "total_s": time.perf_counter() - started,
            "split_scan_s": split_timing,
            "gpu": admitter.evidence(),
        },
        "runtime": _runtime_evidence(admitter.device),
        "source_evidence": {"before": source_before, "after": source_after},
    }
    checks = evaluate_admission_report(report)
    report["checks"] = list(checks)
    report["status"] = "pass" if all(check["passed"] for check in checks) else "fail"
    return report


def main(arguments: list[str] | None = None) -> None:
    options = _parse_args(arguments)
    report_path = _resolve_output(PROJECT_ROOT, options.output)
    try:
        report = run_formal_admission(PROJECT_ROOT, options)
    except Exception as error:
        failure = {
            "schema_version": ADMISSION_REPORT_SCHEMA_VERSION,
            "protocol_version": ADMISSION_PROTOCOL_VERSION,
            "status": "fail",
            "failure": {"type": type(error).__name__, "message": str(error)},
        }
        write_strict_json(failure, report_path)
        print(json.dumps(failure, sort_keys=True))
        raise
    write_strict_json(report, report_path)
    print(f"M5 Track admission status: {report['status']}")
    counts = {name: value["selected_count"] for name, value in report["splits"].items()}
    print(json.dumps(counts, sort_keys=True))
    print(f"Wrote {report_path}")
    if report["status"] != "pass":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
