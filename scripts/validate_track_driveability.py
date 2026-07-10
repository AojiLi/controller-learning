"""Validate generated tracks at low speed on the formal MJX-Warp four-wheel backend."""

from __future__ import annotations

import os

# Keep this diagnostic from reserving most device memory before JAX is imported.
os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")

import argparse
import hashlib
import importlib.metadata
import json
import math
import platform
import subprocess
import sys
import time
from collections import Counter
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
from controller_learning.physics.mjx_warp import MjxWarpVehicle
from controller_learning.tracks.driveability import (
    ConservativeDriveabilityPolicyConfig,
    conservative_driveability_action,
)
from controller_learning.tracks.generator import (
    TrackGenerationError,
    generate_track_candidate,
    pack_track,
)
from controller_learning.tracks.specs import (
    generation_spec_from_project,
    track_capacity_from_project,
    validation_spec_from_project,
)
from controller_learning.tracks.types import stack_tracks
from controller_learning.tracks.validator import validate_track_candidate

PROTOCOL_VERSION = "m3-driveability-v1"

_RELEVANT_SOURCE_PATHS = (
    "controller_learning/tracks/driveability.py",
    "controller_learning/tracks/generator.py",
    "controller_learning/tracks/validator.py",
    "controller_learning/tracks/specs.py",
    "controller_learning/envs/race_core.py",
    "controller_learning/envs/configuration.py",
    "controller_learning/physics/mjx_warp.py",
    "scripts/validate_track_driveability.py",
    "configs/track.toml",
    "configs/vehicle.toml",
    "configs/benchmark.toml",
    "pixi.lock",
)


def _source_evidence(project_root: Path) -> dict[str, Any]:
    """Record reconstructible source evidence without local paths or device identifiers."""

    revision_result = subprocess.run(
        ("git", "rev-parse", "HEAD"),
        cwd=project_root,
        check=False,
        capture_output=True,
        text=True,
    )
    revision = revision_result.stdout.strip()
    status_result = subprocess.run(
        ("git", "status", "--porcelain", "--", *_RELEVANT_SOURCE_PATHS),
        cwd=project_root,
        check=False,
        capture_output=True,
        text=True,
    )
    return {
        "git_revision": (revision if revision_result.returncode == 0 and revision else None),
        "relevant_source_clean": (
            not bool(status_result.stdout.strip()) if status_result.returncode == 0 else None
        ),
        "source_files_sha256": {
            relative: hashlib.sha256((project_root / relative).read_bytes()).hexdigest()
            for relative in _RELEVANT_SOURCE_PATHS
        },
    }


def _gpu_hardware(device: Any) -> dict[str, Any]:
    hardware: dict[str, Any] = {
        "jax_device_kind": str(getattr(device, "device_kind", "unknown")),
        "jax_platform": str(getattr(device, "platform", "unknown")),
    }
    result = subprocess.run(
        (
            "nvidia-smi",
            "--query-gpu=name,memory.total,driver_version",
            "--format=csv,noheader,nounits",
            "--id=0",
        ),
        check=False,
        capture_output=True,
        text=True,
    )
    if result.returncode == 0 and result.stdout.strip():
        fields = [field.strip() for field in result.stdout.splitlines()[0].split(",")]
        if len(fields) == 3:
            hardware.update(
                {
                    "gpu_name": fields[0],
                    "gpu_memory_total_mib": int(fields[1]),
                    "nvidia_driver_version": fields[2],
                }
            )
    return hardware


def _select_race_state(previous: RaceState, candidate: RaceState, active: jax.Array) -> RaceState:
    def select(old: jax.Array, new: jax.Array) -> jax.Array:
        mask = active.reshape((active.shape[0],) + (1,) * (old.ndim - 1))
        return jnp.where(mask, new, old)

    return jax.tree.map(select, previous, candidate)


def _outcome_name(reason: int) -> str:
    names = {
        int(RaceTermination.NONE): "incomplete",
        int(RaceTermination.SUCCESS): "success",
        int(RaceTermination.OFF_TRACK): "off_track",
        int(RaceTermination.INVALID_ACTION): "invalid_action",
        int(RaceTermination.TIMEOUT): "timeout",
        5: "numerical_failure",
    }
    return names.get(reason, "unknown")


def _generate_tracks(
    seed_start: int,
    seed_count: int,
    project_config: ProjectConfig,
) -> tuple[list[Any], list[dict[str, Any]]]:
    capacity = track_capacity_from_project(project_config)
    generation_spec = generation_spec_from_project(project_config)
    validation_spec = validation_spec_from_project(project_config)
    tracks: list[Any] = []
    seed_results: list[dict[str, Any]] = []
    for seed in range(seed_start, seed_start + seed_count):
        result: dict[str, Any] = {"seed": seed}
        try:
            candidate = generate_track_candidate(seed, generation_spec)
        except TrackGenerationError as error:
            result.update(
                {
                    "geometry_status": "generation_rejected",
                    "geometry_reasons": [error.reason],
                    "driveability_status": "not_run",
                }
            )
            seed_results.append(result)
            continue
        validation = validate_track_candidate(candidate, validation_spec)
        if not validation.valid:
            result.update(
                {
                    "geometry_status": "validation_rejected",
                    "geometry_reasons": list(validation.reasons),
                    "driveability_status": "not_run",
                }
            )
            seed_results.append(result)
            continue
        try:
            track = pack_track(candidate, capacity)
        except TrackGenerationError as error:
            result.update(
                {
                    "geometry_status": "capacity_rejected",
                    "geometry_reasons": [error.reason],
                    "driveability_status": "not_run",
                }
            )
            seed_results.append(result)
            continue
        result.update(
            {
                "geometry_status": "accepted",
                "geometry_reasons": [],
                "length_m": float(track.length_m),
                "point_count": track.point_count,
                "checkpoint_count": track.checkpoint_count,
                "driveability_status": "pending",
            }
        )
        tracks.append(track)
        seed_results.append(result)
    return tracks, seed_results


def run_driveability_validation(
    project_root: Path,
    *,
    seed_start: int,
    seed_count: int,
    target_speed_mps: float,
) -> dict[str, Any]:
    """Run one no-retry candidate per seed and return a strict-JSON-compatible report."""

    if seed_count <= 0:
        raise ValueError("seed_count must be positive")
    if not 0 <= seed_start <= np.iinfo(np.uint32).max:
        raise ValueError("seed_start must fit in uint32")
    if seed_start + seed_count - 1 > np.iinfo(np.uint32).max:
        raise ValueError("the final seed must fit in uint32")
    if not math.isfinite(target_speed_mps) or target_speed_mps <= 3.0:
        raise ValueError("target_speed_mps must be finite and greater than the 3 m/s timeout speed")

    project_config = load_project_config(project_root)
    generation_started = time.perf_counter()
    tracks, seed_results = _generate_tracks(seed_start, seed_count, project_config)
    generation_s = time.perf_counter() - generation_started
    accepted_indices = [
        index
        for index, result in enumerate(seed_results)
        if result["geometry_status"] == "accepted"
    ]
    if not tracks:
        return {
            "schema_version": 1,
            "protocol_version": PROTOCOL_VERSION,
            "status": "fail",
            "failure": "no_geometry_candidates_accepted",
            "request": {
                "seed_start": seed_start,
                "seed_count": seed_count,
                "target_speed_mps": target_speed_mps,
            },
            "capacity": {
                "max_track_points": project_config.track.representation.max_track_points,
                "max_checkpoints": project_config.track.representation.max_checkpoints,
            },
            "seed_results": seed_results,
            "timing": {"geometry_generation_and_validation_s": generation_s},
        }

    vehicle_config = project_config.vehicle
    track_config = project_config.track
    if target_speed_mps > vehicle_config.vehicle.max_speed_mps:
        raise ValueError("target_speed_mps cannot exceed the configured vehicle speed limit")
    track_batch = stack_tracks(tracks)
    race_config = race_core_config_from_project(project_config)
    policy_config = ConservativeDriveabilityPolicyConfig(
        target_speed_mps=target_speed_mps,
        wheelbase_m=vehicle_config.vehicle.wheelbase_m,
        maximum_steering_angle_rad=vehicle_config.actuator.max_steering_angle_rad,
        maximum_acceleration_mps2=vehicle_config.actuator.max_acceleration_mps2,
        maximum_deceleration_mps2=vehicle_config.actuator.max_deceleration_mps2,
    )

    adapter_started = time.perf_counter()
    vehicle = MjxWarpVehicle.create(vehicle_config, num_worlds=len(tracks))
    adapter_creation_s = time.perf_counter() - adapter_started
    physics_state = vehicle.initial_state(track_batch.start_pose)
    race_state = reset_race_state(track_batch)
    view = vehicle.read_state(physics_state)
    projection = project_to_track(
        track_batch,
        view.position_world_m[:, :2],
        race_state.segment_index,
        race_config,
    )
    policy_step = jax.jit(
        lambda current_projection, current_view: conservative_driveability_action(
            track_batch,
            current_projection,
            current_view,
            policy_config,
        )
    )
    race_step = jax.jit(
        lambda current_race, positions, invalid: step_race_core(
            track_batch,
            current_race,
            positions,
            invalid,
            race_config,
        )
    )

    first_step_started = time.perf_counter()
    initial_actions = policy_step(projection, view)
    compiled_physics = vehicle.lower_step(physics_state, initial_actions).compile()
    compiled_policy = policy_step.lower(projection, view).compile()
    compiled_race = race_step.lower(
        race_state,
        view.position_world_m[:, :2],
        jnp.zeros(len(tracks), dtype=bool),
    ).compile()
    jax.block_until_ready(initial_actions)
    compilation_s = time.perf_counter() - first_step_started

    num_worlds = len(tracks)
    active = jnp.ones(num_worlds, dtype=bool)
    outcome_reason = jnp.zeros(num_worlds, dtype=jnp.int32)
    outcome_step = jnp.zeros(num_worlds, dtype=jnp.int32)
    maximum_lateral_error = jnp.zeros(num_worlds, dtype=jnp.float32)
    maximum_speed = jnp.zeros(num_worlds, dtype=jnp.float32)
    maximum_abs_steering = jnp.zeros(num_worlds, dtype=jnp.float32)
    maximum_abs_acceleration = jnp.zeros(num_worlds, dtype=jnp.float32)
    finite_per_world = jnp.ones(num_worlds, dtype=bool)
    minimum_wheel_contact_fraction = jnp.ones((num_worlds, 4), dtype=jnp.float32)
    finite = jnp.asarray(True)
    time_monotonic = jnp.asarray(True)
    contact_overflow = jnp.asarray(False)
    constraint_overflow = jnp.asarray(False)
    unexpected_contact = jnp.asarray(False)
    peak_nacon = jnp.asarray(0, dtype=jnp.int32)
    peak_ncollision = jnp.asarray(0, dtype=jnp.int32)
    peak_nefc = jnp.asarray(0, dtype=jnp.int32)
    maximum_penetration = jnp.asarray(0.0, dtype=jnp.float32)
    maximum_contact_gap = jnp.asarray(0.0, dtype=jnp.float32)
    maximum_quaternion_error = jnp.asarray(0.0, dtype=jnp.float32)
    maximum_roll_pitch = jnp.asarray(0.0, dtype=jnp.float32)
    maximum_vertical_speed = jnp.asarray(0.0, dtype=jnp.float32)
    maximum_steps = int(
        np.max(
            np.ceil(
                np.maximum(
                    race_config.min_timeout_s,
                    np.asarray(track_batch.length_m) / race_config.timeout_reference_speed_mps,
                )
                / race_config.control_dt_s
            )
        )
    )

    measured_started = time.perf_counter()
    executed_steps = 0
    for step_index in range(maximum_steps):
        actions = compiled_policy(projection, view)
        stopped_actions = jnp.column_stack(
            (
                jnp.zeros(num_worlds, dtype=jnp.float32),
                jnp.full(
                    num_worlds,
                    -vehicle_config.actuator.max_deceleration_mps2,
                    dtype=jnp.float32,
                ),
            )
        )
        actions = jnp.where(active[:, None], actions, stopped_actions)
        next_physics, applied, diagnostics = compiled_physics(physics_state, actions)
        next_view = vehicle.read_state(next_physics)
        candidate_race = compiled_race(
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
        maximum_abs_steering = jnp.maximum(
            maximum_abs_steering,
            jnp.where(active, jnp.abs(applied.steering_angle_rad), 0.0),
        )
        maximum_abs_acceleration = jnp.maximum(
            maximum_abs_acceleration,
            jnp.where(active, jnp.abs(applied.longitudinal_acceleration_mps2), 0.0),
        )
        finite &= diagnostics.finite
        finite_per_world &= diagnostics.finite_per_world
        time_monotonic &= diagnostics.time_monotonic
        contact_overflow |= diagnostics.contact_overflow
        constraint_overflow |= diagnostics.constraint_overflow
        unexpected_contact |= diagnostics.unexpected_contact
        peak_nacon = jnp.maximum(peak_nacon, diagnostics.peak_nacon)
        peak_ncollision = jnp.maximum(peak_ncollision, diagnostics.peak_ncollision)
        peak_nefc = jnp.maximum(peak_nefc, diagnostics.peak_nefc)
        maximum_penetration = jnp.maximum(
            maximum_penetration,
            diagnostics.maximum_penetration_m,
        )
        maximum_contact_gap = jnp.maximum(
            maximum_contact_gap,
            diagnostics.maximum_wheel_contact_gap_s,
        )
        maximum_quaternion_error = jnp.maximum(
            maximum_quaternion_error,
            diagnostics.maximum_quaternion_norm_error,
        )
        maximum_roll_pitch = jnp.maximum(
            maximum_roll_pitch,
            diagnostics.maximum_abs_roll_pitch_rad,
        )
        maximum_vertical_speed = jnp.maximum(
            maximum_vertical_speed,
            diagnostics.maximum_abs_vertical_speed_mps,
        )
        minimum_wheel_contact_fraction = jnp.minimum(
            minimum_wheel_contact_fraction,
            diagnostics.wheel_ground_contact_fraction,
        )
        race_state = _select_race_state(race_state, candidate_race.state, active)
        projection = candidate_race.projection
        physics_state = next_physics
        view = next_view
        active &= ~new_done
        executed_steps = step_index + 1
        if executed_steps % 100 == 0 and not bool(jax.device_get(jnp.any(active))):
            break
    jax.block_until_ready((physics_state.data.qpos, outcome_reason, race_state.legal_progress_m))
    measured_execution_s = time.perf_counter() - measured_started

    host_reason = np.asarray(jax.device_get(outcome_reason))
    host_steps = np.asarray(jax.device_get(outcome_step))
    host_progress = np.asarray(jax.device_get(race_state.legal_progress_m))
    host_lateral = np.asarray(jax.device_get(maximum_lateral_error))
    host_speed = np.asarray(jax.device_get(maximum_speed))
    host_steering = np.asarray(jax.device_get(maximum_abs_steering))
    host_acceleration = np.asarray(jax.device_get(maximum_abs_acceleration))
    for world, result_index in enumerate(accepted_indices):
        reason = int(host_reason[world])
        outcome = _outcome_name(reason)
        seed_results[result_index].update(
            {
                "driveability_status": outcome,
                "termination_step": int(host_steps[world]),
                "lap_time_s": (
                    float(host_steps[world] * race_config.control_dt_s)
                    if outcome == "success"
                    else None
                ),
                "legal_progress_m": float(host_progress[world]),
                "progress_fraction": float(host_progress[world] / tracks[world].length_m),
                "maximum_abs_lateral_error_m": float(host_lateral[world]),
                "maximum_planar_speed_mps": float(host_speed[world]),
                "maximum_abs_steering_action_rad": float(host_steering[world]),
                "maximum_abs_acceleration_action_mps2": float(host_acceleration[world]),
            }
        )

    geometry_counts = Counter(result["geometry_status"] for result in seed_results)
    outcome_counts = Counter(
        result["driveability_status"]
        for result in seed_results
        if result["geometry_status"] == "accepted"
    )
    diagnostics_host = {
        "finite": bool(jax.device_get(finite)),
        "finite_per_world": np.asarray(jax.device_get(finite_per_world)).tolist(),
        "time_monotonic": bool(jax.device_get(time_monotonic)),
        "contact_overflow": bool(jax.device_get(contact_overflow)),
        "constraint_overflow": bool(jax.device_get(constraint_overflow)),
        "unexpected_contact": bool(jax.device_get(unexpected_contact)),
        "peak_nacon_global": int(jax.device_get(peak_nacon)),
        "peak_ncollision_global": int(jax.device_get(peak_ncollision)),
        "peak_nefc_per_world": int(jax.device_get(peak_nefc)),
        "maximum_penetration_m": float(jax.device_get(maximum_penetration)),
        "maximum_wheel_contact_gap_s": float(jax.device_get(maximum_contact_gap)),
        "minimum_control_step_wheel_contact_fraction": float(
            jnp.min(minimum_wheel_contact_fraction)
        ),
        "maximum_quaternion_norm_error": float(jax.device_get(maximum_quaternion_error)),
        "maximum_abs_roll_pitch_rad": float(jax.device_get(maximum_roll_pitch)),
        "maximum_abs_vertical_speed_mps": float(jax.device_get(maximum_vertical_speed)),
    }
    all_successful = outcome_counts["success"] == len(tracks)
    numerically_valid = (
        diagnostics_host["finite"]
        and diagnostics_host["time_monotonic"]
        and not diagnostics_host["contact_overflow"]
        and not diagnostics_host["constraint_overflow"]
        and not diagnostics_host["unexpected_contact"]
    )
    transitions = num_worlds * executed_steps
    device = jax.devices("gpu")[0]
    level1 = next(level for level in project_config.levels if level.level_id == 1)
    return {
        "schema_version": 1,
        "protocol_version": PROTOCOL_VERSION,
        "status": "pass" if all_successful and numerically_valid else "fail",
        "source_evidence": _source_evidence(project_root),
        "request": {
            "seed_start": seed_start,
            "seed_count": seed_count,
            "target_speed_mps": target_speed_mps,
        },
        "capacity": {
            "max_track_points": track_config.representation.max_track_points,
            "max_checkpoints": track_config.representation.max_checkpoints,
        },
        "protocol": {
            "one_candidate_per_seed": True,
            "hidden_retry": False,
            "formal_physics_backend": "MJX-Warp",
            "track_width_m": level1.track_width_m,
            "vehicle_width_m": vehicle_config.vehicle.vehicle_width_m,
            "safety_margin_m": track_config.race.safety_margin_m,
            "projection_backward_segments": track_config.race.projection_backward_segments,
            "projection_forward_segments": track_config.race.projection_forward_segments,
            "control_dt_s": race_config.control_dt_s,
            "timeout_minimum_s": race_config.min_timeout_s,
            "timeout_reference_speed_mps": race_config.timeout_reference_speed_mps,
            "policy_role": "internal_offline_track_admission_only",
        },
        "summary": {
            "geometry_counts": dict(sorted(geometry_counts.items())),
            "driveability_counts": dict(sorted(outcome_counts.items())),
            "accepted_count": len(tracks),
            "success_count": outcome_counts["success"],
            "off_track_count": outcome_counts["off_track"],
            "timeout_count": outcome_counts["timeout"],
            "invalid_action_count": outcome_counts["invalid_action"],
            "numerical_failure_count": outcome_counts["numerical_failure"],
            "success_rate": outcome_counts["success"] / len(tracks),
        },
        "seed_results": seed_results,
        "numerical_diagnostics": diagnostics_host,
        "timing": {
            "geometry_generation_and_validation_s": generation_s,
            "adapter_creation_s": adapter_creation_s,
            "compilation_s": compilation_s,
            "measured_execution_s": measured_execution_s,
            "executed_environment_steps": executed_steps,
            "executed_transitions": transitions,
            "transitions_per_second": transitions / measured_execution_s,
        },
        "runtime": {
            "os": platform.system(),
            "kernel": platform.release(),
            "machine": platform.machine(),
            "python_version": sys.version.split()[0],
            "jax_version": jax.__version__,
            "mujoco_version": mujoco.__version__,
            "mjx_warp_version": importlib.metadata.version("mujoco-mjx"),
            **_gpu_hardware(device),
        },
    }


def main() -> None:
    """Parse the bounded validation protocol and write its strict JSON report."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--count", type=int, default=16, help="Number of contiguous seeds")
    parser.add_argument("--start-seed", type=int, default=0, help="First seed")
    parser.add_argument(
        "--target-speed",
        type=float,
        default=4.0,
        help="Low-speed policy target in m/s (must exceed 3 m/s)",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("driveability_report.json"),
        help="Strict JSON output path",
    )
    args = parser.parse_args()
    project_root = Path(__file__).resolve().parents[1]
    report = run_driveability_validation(
        project_root,
        seed_start=args.start_seed,
        seed_count=args.count,
        target_speed_mps=args.target_speed,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(report, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    print(f"M3 driveability status: {report['status']}")
    print(json.dumps(report.get("summary", {}), sort_keys=True))
    print(f"Wrote {args.output}")
    if report["status"] != "pass":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
