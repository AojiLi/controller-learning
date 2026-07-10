"""Run one isolated M2 MJX-Warp scale and emit a strict JSON result."""

from __future__ import annotations

import os

# This must be set before any import can initialize JAX's GPU backend.
os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
os.environ.setdefault("CUDA_DEVICE_ORDER", "PCI_BUS_ID")

import argparse
import gc
import math
import platform
import statistics
import subprocess
import sys
import tempfile
import time
import traceback
from contextlib import contextmanager
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from typing import Any, NamedTuple

from controller_learning.physics.m2_benchmark import (
    DEFAULT_WARMUP_CHUNKS,
    MAXIMUM_ABS_QVEL,
    MAXIMUM_ABS_ROLL_PITCH_RAD,
    MAXIMUM_ABS_VERTICAL_SPEED_MPS,
    MAXIMUM_PENETRATION_M,
    MAXIMUM_QUATERNION_NORM_ERROR,
    MAXIMUM_WHEEL_CONTACT_GAP_S,
    MINIMUM_CHUNK_MEAN_WHEEL_CONTACT_FRACTION,
    PROTOCOL_VERSION,
    WORKER_JSON_PREFIX,
    WORKER_SCHEMA_VERSION,
    write_strict_json,
)

RESET_INTERVAL_STEPS = 257
RESET_GROUPS = 31
MEMORY_GROWTH_FLOOR_MIB = 64.0
MEMORY_GROWTH_FRACTION = 0.02


class ChunkAggregate(NamedTuple):
    """Device-side reductions over one synchronized benchmark chunk."""

    finite: Any
    time_monotonic: Any
    contact_overflow: Any
    constraint_overflow: Any
    unexpected_contact: Any
    peak_nacon: Any
    peak_ncollision: Any
    peak_nefc: Any
    maximum_penetration_m: Any
    maximum_wheel_contact_gap_s: Any
    maximum_quaternion_norm_error: Any
    maximum_abs_roll_pitch_rad: Any
    maximum_abs_vertical_speed_mps: Any
    minimum_chunk_mean_wheel_contact_fraction: Any
    minimum_chassis_z_m: Any
    maximum_chassis_z_m: Any
    maximum_abs_qvel: Any
    maximum_planar_speed_mps: Any
    invalid_action_count: Any
    masked_reset_count: Any


@contextmanager
def _capture_native_output(directory: Path):
    """Capture Python and native-library stdout/stderr without corrupting JSON stdout."""

    stdout_path = directory / "captured.stdout"
    stderr_path = directory / "captured.stderr"
    sys.stdout.flush()
    sys.stderr.flush()
    saved_stdout = os.dup(1)
    saved_stderr = os.dup(2)
    try:
        with stdout_path.open("wb") as stdout_file, stderr_path.open("wb") as stderr_file:
            os.dup2(stdout_file.fileno(), 1)
            os.dup2(stderr_file.fileno(), 2)
            yield stdout_path, stderr_path
            sys.stdout.flush()
            sys.stderr.flush()
    finally:
        os.dup2(saved_stdout, 1)
        os.dup2(saved_stderr, 2)
        os.close(saved_stdout)
        os.close(saved_stderr)


def _bounded_text(path: Path, *, maximum_bytes: int = 24_000) -> str:
    try:
        content = path.read_bytes()
    except OSError:
        return ""
    return content[-maximum_bytes:].decode(errors="replace")


def _warning_lines(*paths: Path, maximum_lines: int = 100) -> list[str]:
    """Scan complete native output files while bounding retained report evidence."""

    indicators = (
        "warning",
        "overflow",
        "traceback",
        "failed",
        "error",
        "non-finite",
        "nonfinite",
        "nan detected",
    )
    matches: list[str] = []
    for path in paths:
        try:
            lines = path.read_text(errors="replace").splitlines()
        except OSError as error:
            return [f"output scan failed for {path.name}: {error}"]
        for line in lines:
            if any(indicator in line.casefold() for indicator in indicators):
                matches.append(line[:1_000])
                if len(matches) >= maximum_lines:
                    matches.append("additional warning lines omitted")
                    return matches
    return matches


def _package_version(name: str) -> str | None:
    try:
        return version(name)
    except PackageNotFoundError:
        return None


def _run_command(command: tuple[str, ...]) -> tuple[str | None, str | None]:
    try:
        completed = subprocess.run(
            command,
            check=True,
            capture_output=True,
            text=True,
            timeout=15,
        )
    except (OSError, subprocess.SubprocessError) as error:
        return None, f"{type(error).__name__}: {error}"
    return completed.stdout.strip(), None


def _nvidia_hardware() -> tuple[list[dict[str, Any]], str | None]:
    fields = (
        "index",
        "name",
        "uuid",
        "driver_version",
        "memory.total",
        "memory.used",
        "memory.free",
    )
    stdout, error = _run_command(
        (
            "nvidia-smi",
            f"--query-gpu={','.join(fields)}",
            "--format=csv,noheader,nounits",
        )
    )
    if stdout is None:
        return [], error
    gpus: list[dict[str, Any]] = []
    for line in stdout.splitlines():
        values = [value.strip() for value in line.split(",")]
        if len(values) != len(fields):
            continue
        gpus.append(
            {
                "index": int(values[0]),
                "name": values[1],
                "uuid": values[2],
                "driver_version": values[3],
                "memory_total_mib": float(values[4]),
                "memory_used_mib": float(values[5]),
                "memory_free_mib": float(values[6]),
            }
        )
    return gpus, None if gpus else "nvidia-smi returned no parseable GPUs"


def _selected_nvidia_gpu(
    device: Any,
    gpus: list[dict[str, Any]],
) -> tuple[dict[str, Any] | None, str | None]:
    """Map one logical JAX device to the selected physical NVIDIA GPU."""

    logical_id = int(getattr(device, "id", 0))
    visible = os.environ.get("CUDA_VISIBLE_DEVICES")
    if visible:
        tokens = [token.strip() for token in visible.split(",") if token.strip()]
        if logical_id >= len(tokens):
            return None, f"logical GPU {logical_id} is absent from CUDA_VISIBLE_DEVICES={visible!r}"
        token = tokens[logical_id]
        if token.isdigit():
            physical_index = int(token)
            selected = next((gpu for gpu in gpus if gpu["index"] == physical_index), None)
        else:
            selected = next(
                (gpu for gpu in gpus if gpu["uuid"] == token or gpu["uuid"].startswith(token)),
                None,
            )
    else:
        selected = next((gpu for gpu in gpus if gpu["index"] == logical_id), None)
    if selected is None:
        return None, f"cannot map logical JAX GPU {logical_id} to nvidia-smi inventory"
    return selected, None


def _process_vram_mib(
    process_id: int,
    gpu_uuid: str | None,
) -> tuple[float | None, str | None]:
    if gpu_uuid is None:
        return None, "selected physical GPU UUID is unavailable"
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
    matched = False
    for line in stdout.splitlines():
        values = [value.strip() for value in line.split(",")]
        if len(values) != 3:
            continue
        try:
            row_pid = int(values[1])
            used_mib = float(values[2])
        except ValueError:
            continue
        if values[0] == gpu_uuid and row_pid == process_id:
            matched = True
            total += used_mib
    return (total if matched else None), None


def _json_scalar(value: Any) -> int | float | bool | str | None:
    if value is None or isinstance(value, (bool, int, float, str)):
        return value
    try:
        return int(value)
    except (TypeError, ValueError):
        return str(value)


def _memory_sample(
    device: Any,
    phase: str,
    environment_step: int,
    *,
    gpu_uuid: str | None,
) -> dict[str, Any]:
    process_vram, process_error = _process_vram_mib(os.getpid(), gpu_uuid)
    try:
        raw_stats = device.memory_stats() or {}
        allocator = {str(key): _json_scalar(value) for key, value in raw_stats.items()}
        allocator_error = None
    except (RuntimeError, TypeError) as error:
        allocator = {}
        allocator_error = f"{type(error).__name__}: {error}"
    return {
        "phase": phase,
        "environment_step": environment_step,
        "process_vram_mib": process_vram,
        "process_vram_error": process_error,
        "jax_allocator": allocator,
        "jax_allocator_error": allocator_error,
    }


def _allocator_bytes(sample: dict[str, Any]) -> float | None:
    allocator = sample["jax_allocator"]
    for key in ("bytes_in_use", "bytes_reserved", "peak_bytes_in_use"):
        value = allocator.get(key)
        if isinstance(value, (int, float)):
            return float(value)
    return None


def _median_edge_growth(values: list[float]) -> float | None:
    if len(values) < 4:
        return None
    analysis = values[max(1, len(values) // 5) :]
    width = min(5, max(1, len(analysis) // 3))
    return statistics.median(analysis[-width:]) - statistics.median(analysis[:width])


def _memory_report(samples: list[dict[str, Any]]) -> dict[str, Any]:
    measured = [sample for sample in samples if sample["phase"] == "measured"]
    all_process_values = [
        float(sample["process_vram_mib"])
        for sample in samples
        if sample["process_vram_mib"] is not None
    ]
    process_values = [
        float(sample["process_vram_mib"])
        for sample in measured
        if sample["process_vram_mib"] is not None
    ]
    allocator_values = [value for sample in measured if (value := _allocator_bytes(sample))]
    process_growth = _median_edge_growth(process_values)
    allocator_growth_bytes = _median_edge_growth(allocator_values)
    peak_process = max(all_process_values, default=None)
    process_threshold = max(
        MEMORY_GROWTH_FLOOR_MIB,
        MEMORY_GROWTH_FRACTION * peak_process if peak_process is not None else 0.0,
    )
    process_stable = process_growth is not None and process_growth <= process_threshold
    allocator_threshold_bytes = MEMORY_GROWTH_FLOOR_MIB * 1024 * 1024
    allocator_coverage = len(allocator_values) == len(measured) and len(measured) >= 4
    allocator_stable = (
        allocator_coverage
        and allocator_growth_bytes is not None
        and allocator_growth_bytes <= allocator_threshold_bytes
    )
    return {
        "samples": samples,
        "measured_sample_count": len(measured),
        "process_vram_coverage": len(process_values) == len(measured) and len(measured) >= 4,
        "jax_allocator_coverage": allocator_coverage,
        "peak_process_vram_mib": peak_process,
        "process_vram_growth_mib": process_growth,
        "process_growth_limit_mib": process_threshold,
        "jax_allocator_growth_bytes": allocator_growth_bytes,
        "jax_allocator_growth_limit_bytes": allocator_threshold_bytes,
        "stable": bool(process_stable and allocator_stable),
    }


def _empty_aggregate(jnp: Any) -> ChunkAggregate:
    return ChunkAggregate(
        finite=jnp.asarray(True),
        time_monotonic=jnp.asarray(True),
        contact_overflow=jnp.asarray(False),
        constraint_overflow=jnp.asarray(False),
        unexpected_contact=jnp.asarray(False),
        peak_nacon=jnp.asarray(0, dtype=jnp.int32),
        peak_ncollision=jnp.asarray(0, dtype=jnp.int32),
        peak_nefc=jnp.asarray(0, dtype=jnp.int32),
        maximum_penetration_m=jnp.asarray(0.0, dtype=jnp.float32),
        maximum_wheel_contact_gap_s=jnp.asarray(0.0, dtype=jnp.float32),
        maximum_quaternion_norm_error=jnp.asarray(0.0, dtype=jnp.float32),
        maximum_abs_roll_pitch_rad=jnp.asarray(0.0, dtype=jnp.float32),
        maximum_abs_vertical_speed_mps=jnp.asarray(0.0, dtype=jnp.float32),
        minimum_chunk_mean_wheel_contact_fraction=jnp.asarray(1.0, dtype=jnp.float32),
        minimum_chassis_z_m=jnp.asarray(math.inf, dtype=jnp.float32),
        maximum_chassis_z_m=jnp.asarray(-math.inf, dtype=jnp.float32),
        maximum_abs_qvel=jnp.asarray(0.0, dtype=jnp.float32),
        maximum_planar_speed_mps=jnp.asarray(0.0, dtype=jnp.float32),
        invalid_action_count=jnp.asarray(0, dtype=jnp.int32),
        masked_reset_count=jnp.asarray(0, dtype=jnp.int32),
    )


def _build_chunk_function(vehicle: Any, *, chunk_steps: int, jax: Any, jnp: Any):
    world_ids = jnp.arange(vehicle.num_worlds, dtype=jnp.int32)
    denominator = max(vehicle.num_worlds, 1)
    phase = world_ids.astype(jnp.float32) * (2.0 * math.pi / denominator)
    target_speed = 6.0 + 2.0 * (world_ids % 5).astype(jnp.float32) / 4.0
    reset_groups = min(RESET_GROUPS, vehicle.num_worlds)

    def run_chunk(state: Any, start_step: Any) -> tuple[Any, ChunkAggregate]:
        def body(carry: tuple[Any, ChunkAggregate, Any], offset: Any):
            current, aggregate, contact_fraction_sum = carry
            global_step = start_step + offset
            view = vehicle.read_state(current)
            speed = view.velocity_body_mps[:, 0]
            enabled = global_step >= 20
            acceleration = jnp.where(
                enabled,
                jnp.clip(1.25 * (target_speed - speed), -6.0, 3.0),
                0.0,
            )
            time_s = global_step.astype(jnp.float32) * vehicle.config.simulation.control_dt_s
            steering = jnp.where(
                global_step >= 40,
                0.12 * jnp.sin((2.0 * math.pi / 6.0) * time_s + phase),
                0.0,
            )
            actions = jnp.stack((steering, acceleration), axis=1)
            stepped, applied, diagnostics = vehicle._step_function(current, actions)
            stepped_view = vehicle.read_state(stepped)
            planar_speed = jnp.linalg.norm(stepped_view.velocity_body_mps[:, :2], axis=1)
            completed_step = global_step + 1
            reset_cycle = completed_step // RESET_INTERVAL_STEPS
            reset_mask = (completed_step % RESET_INTERVAL_STEPS == 0) & (
                (world_ids + reset_cycle) % reset_groups == 0
            )
            next_state = vehicle.masked_reset(stepped, reset_mask)
            aggregate = ChunkAggregate(
                finite=aggregate.finite & diagnostics.finite,
                time_monotonic=aggregate.time_monotonic & diagnostics.time_monotonic,
                contact_overflow=aggregate.contact_overflow | diagnostics.contact_overflow,
                constraint_overflow=(
                    aggregate.constraint_overflow | diagnostics.constraint_overflow
                ),
                unexpected_contact=aggregate.unexpected_contact | diagnostics.unexpected_contact,
                peak_nacon=jnp.maximum(aggregate.peak_nacon, diagnostics.peak_nacon),
                peak_ncollision=jnp.maximum(
                    aggregate.peak_ncollision,
                    diagnostics.peak_ncollision,
                ),
                peak_nefc=jnp.maximum(aggregate.peak_nefc, diagnostics.peak_nefc),
                maximum_penetration_m=jnp.maximum(
                    aggregate.maximum_penetration_m,
                    diagnostics.maximum_penetration_m,
                ),
                maximum_wheel_contact_gap_s=jnp.maximum(
                    aggregate.maximum_wheel_contact_gap_s,
                    diagnostics.maximum_wheel_contact_gap_s,
                ),
                maximum_quaternion_norm_error=jnp.maximum(
                    aggregate.maximum_quaternion_norm_error,
                    diagnostics.maximum_quaternion_norm_error,
                ),
                maximum_abs_roll_pitch_rad=jnp.maximum(
                    aggregate.maximum_abs_roll_pitch_rad,
                    diagnostics.maximum_abs_roll_pitch_rad,
                ),
                maximum_abs_vertical_speed_mps=jnp.maximum(
                    aggregate.maximum_abs_vertical_speed_mps,
                    diagnostics.maximum_abs_vertical_speed_mps,
                ),
                minimum_chunk_mean_wheel_contact_fraction=(
                    aggregate.minimum_chunk_mean_wheel_contact_fraction
                ),
                minimum_chassis_z_m=jnp.minimum(
                    aggregate.minimum_chassis_z_m,
                    jnp.min(stepped.data.qpos[:, 2]),
                ),
                maximum_chassis_z_m=jnp.maximum(
                    aggregate.maximum_chassis_z_m,
                    jnp.max(stepped.data.qpos[:, 2]),
                ),
                maximum_abs_qvel=jnp.maximum(
                    aggregate.maximum_abs_qvel,
                    jnp.max(jnp.abs(stepped.data.qvel)),
                ),
                maximum_planar_speed_mps=jnp.maximum(
                    aggregate.maximum_planar_speed_mps,
                    jnp.max(planar_speed),
                ),
                invalid_action_count=(
                    aggregate.invalid_action_count + jnp.sum(applied.invalid_action)
                ),
                masked_reset_count=(
                    aggregate.masked_reset_count + jnp.sum(reset_mask, dtype=jnp.int32)
                ),
            )
            contact_fraction_sum += diagnostics.wheel_ground_contact_fraction
            return (next_state, aggregate, contact_fraction_sum), None

        (state, aggregate, contact_fraction_sum), _ = jax.lax.scan(
            body,
            (
                state,
                _empty_aggregate(jnp),
                jnp.zeros((vehicle.num_worlds, 4), dtype=jnp.float32),
            ),
            jnp.arange(chunk_steps, dtype=jnp.int32),
        )
        aggregate = aggregate._replace(
            minimum_chunk_mean_wheel_contact_fraction=jnp.min(contact_fraction_sum / chunk_steps)
        )
        return state, aggregate

    return jax.jit(run_chunk)


def _new_host_aggregate() -> dict[str, Any]:
    return {
        "finite": True,
        "time_monotonic": True,
        "contact_overflow": False,
        "constraint_overflow": False,
        "unexpected_contact": False,
        "peak_nacon": 0,
        "peak_ncollision": 0,
        "peak_nefc": 0,
        "maximum_penetration_m": 0.0,
        "maximum_wheel_contact_gap_s": 0.0,
        "maximum_quaternion_norm_error": 0.0,
        "maximum_abs_roll_pitch_rad": 0.0,
        "maximum_abs_vertical_speed_mps": 0.0,
        "minimum_chunk_mean_wheel_contact_fraction": 1.0,
        "minimum_chassis_z_m": math.inf,
        "maximum_chassis_z_m": -math.inf,
        "maximum_abs_qvel": 0.0,
        "maximum_planar_speed_mps": 0.0,
        "invalid_action_count": 0,
        "masked_reset_count": 0,
    }


def _merge_aggregate(host: dict[str, Any], device: ChunkAggregate) -> None:
    for field in ("finite", "time_monotonic"):
        host[field] = bool(host[field] and bool(getattr(device, field)))
    for field in ("contact_overflow", "constraint_overflow", "unexpected_contact"):
        host[field] = bool(host[field] or bool(getattr(device, field)))
    for field in (
        "peak_nacon",
        "peak_ncollision",
        "peak_nefc",
        "maximum_penetration_m",
        "maximum_wheel_contact_gap_s",
        "maximum_quaternion_norm_error",
        "maximum_abs_roll_pitch_rad",
        "maximum_abs_vertical_speed_mps",
        "maximum_chassis_z_m",
        "maximum_abs_qvel",
        "maximum_planar_speed_mps",
    ):
        host[field] = max(host[field], float(getattr(device, field)))
    host["minimum_chunk_mean_wheel_contact_fraction"] = min(
        host["minimum_chunk_mean_wheel_contact_fraction"],
        float(device.minimum_chunk_mean_wheel_contact_fraction),
    )
    host["minimum_chassis_z_m"] = min(
        host["minimum_chassis_z_m"],
        float(device.minimum_chassis_z_m),
    )
    for field in ("invalid_action_count", "masked_reset_count"):
        host[field] += int(getattr(device, field))
    for field in ("peak_nacon", "peak_ncollision", "peak_nefc"):
        host[field] = int(host[field])


def _consistency_action(step: int) -> tuple[float, float]:
    if step < 10:
        return (0.0, 0.0)
    if step < 30:
        return (0.0, 1.5)
    if step < 50:
        return (0.15, 0.5)
    if step < 60:
        return (0.0, 0.0)
    if step < 80:
        return (0.0, -3.0)
    return (0.0, 0.0)


def _cpu_gpu_consistency(
    vehicle: Any, config: Any, *, jax: Any, jnp: Any, np: Any
) -> dict[str, Any]:
    from controller_learning.physics import CpuVehicle

    cpu = CpuVehicle(config)
    gpu_state = vehicle.initial_state()
    lowered = vehicle.lower_step(gpu_state, jnp.zeros((1, 2), dtype=jnp.float32))
    compile_started = time.perf_counter()
    compiled_step = lowered.compile()
    compilation_s = time.perf_counter() - compile_started
    cpu_states = [cpu.state()]
    gpu_states = [jax.device_get(vehicle.read_state(gpu_state))]
    cpu_contact_fractions: list[tuple[float, float, float, float]] = []
    gpu_contact_fractions: list[Any] = []
    cpu_contact_gaps: list[float] = []
    gpu_contact_gaps: list[float] = []
    cpu_penetrations: list[float] = []
    gpu_penetrations: list[float] = []
    finite = True
    no_overflow = True
    contacts_valid = True
    run_started = time.perf_counter()
    for step in range(100):
        action = _consistency_action(step)
        cpu_states.append(cpu.step(action))
        gpu_state, _, diagnostics = compiled_step(
            gpu_state,
            jnp.asarray((action,), dtype=jnp.float32),
        )
        jax.block_until_ready(gpu_state.data.qpos)
        diagnostics = jax.device_get(diagnostics)
        cpu_diagnostics = cpu.last_step_diagnostics
        finite = finite and bool(diagnostics.finite)
        no_overflow = no_overflow and not bool(
            diagnostics.contact_overflow | diagnostics.constraint_overflow
        )
        contacts_valid = contacts_valid and (
            cpu_diagnostics.maximum_unexpected_contact_count == 0
            and not bool(diagnostics.unexpected_contact)
        )
        cpu_contact_fractions.append(cpu_diagnostics.wheel_ground_contact_fraction)
        gpu_contact_fractions.append(np.asarray(diagnostics.wheel_ground_contact_fraction[0]))
        cpu_contact_gaps.append(cpu_diagnostics.maximum_wheel_contact_gap_s)
        gpu_contact_gaps.append(float(diagnostics.maximum_wheel_contact_gap_s))
        cpu_penetrations.append(cpu_diagnostics.maximum_penetration_m)
        gpu_penetrations.append(float(diagnostics.maximum_penetration_m))
        gpu_states.append(jax.device_get(vehicle.read_state(gpu_state)))
    run_s = time.perf_counter() - run_started

    def cpu_array(attribute: str):
        return np.asarray([getattr(state, attribute) for state in cpu_states])

    def gpu_array(attribute: str):
        return np.asarray([getattr(state, attribute)[0] for state in gpu_states])

    position_error = gpu_array("position_world_m") - cpu_array("position_world_m")
    velocity_error = gpu_array("velocity_body_mps") - cpu_array("velocity_body_mps")
    angular_error = gpu_array("angular_velocity_body_rad_s") - cpu_array(
        "angular_velocity_body_rad_s"
    )
    steering_error = gpu_array("front_steering_angles_rad") - cpu_array("front_steering_angles_rad")
    wheel_error = gpu_array("wheel_angular_velocity_rad_s") - cpu_array(
        "wheel_angular_velocity_rad_s"
    )
    yaw_error = np.arctan2(
        np.sin(gpu_array("yaw_rad") - cpu_array("yaw_rad")),
        np.cos(gpu_array("yaw_rad") - cpu_array("yaw_rad")),
    )
    cpu_quaternion = cpu_array("quaternion_wxyz").astype(np.float64)
    gpu_quaternion = gpu_array("quaternion_wxyz").astype(np.float64)
    cpu_quaternion /= np.linalg.norm(cpu_quaternion, axis=1, keepdims=True)
    gpu_quaternion /= np.linalg.norm(gpu_quaternion, axis=1, keepdims=True)
    quaternion_dot = np.abs(np.sum(cpu_quaternion * gpu_quaternion, axis=1))
    attitude_error = 2.0 * np.arccos(np.clip(quaternion_dot, 0.0, 1.0))
    contact_fraction_error = np.asarray(gpu_contact_fractions) - np.asarray(cpu_contact_fractions)
    contact_gap_error = np.asarray(gpu_contact_gaps) - np.asarray(cpu_contact_gaps)
    penetration_error = np.asarray(gpu_penetrations) - np.asarray(cpu_penetrations)
    metrics = {
        "position_xy_rmse_m": float(np.sqrt(np.mean(np.square(position_error[:, :2])))),
        "position_xy_max_abs_m": float(np.max(np.abs(position_error[:, :2]))),
        "position_z_max_abs_m": float(np.max(np.abs(position_error[:, 2]))),
        "attitude_rmse_rad": float(np.sqrt(np.mean(np.square(attitude_error)))),
        "attitude_max_abs_rad": float(np.max(np.abs(attitude_error))),
        "yaw_max_abs_rad": float(np.max(np.abs(yaw_error))),
        "velocity_rmse_mps": float(np.sqrt(np.mean(np.square(velocity_error)))),
        "velocity_max_abs_mps": float(np.max(np.abs(velocity_error))),
        "angular_velocity_rmse_rad_s": float(np.sqrt(np.mean(np.square(angular_error)))),
        "angular_velocity_max_abs_rad_s": float(np.max(np.abs(angular_error))),
        "front_steering_max_abs_rad": float(np.max(np.abs(steering_error))),
        "wheel_speed_max_abs_rad_s": float(np.max(np.abs(wheel_error))),
        "wheel_contact_fraction_rmse": float(np.sqrt(np.mean(np.square(contact_fraction_error)))),
        "wheel_contact_fraction_max_abs": float(np.max(np.abs(contact_fraction_error))),
        "contact_gap_max_abs_difference_s": float(np.max(np.abs(contact_gap_error))),
        "penetration_max_abs_difference_m": float(np.max(np.abs(penetration_error))),
        "finite": finite,
        "no_buffer_overflow": no_overflow,
        "contacts_valid": contacts_valid,
    }
    tolerances = {
        "position_xy_rmse_m": 5e-4,
        "position_xy_max_abs_m": 2e-3,
        "position_z_max_abs_m": 5e-4,
        "attitude_rmse_rad": 5e-5,
        "attitude_max_abs_rad": 2e-4,
        "yaw_max_abs_rad": 2e-4,
        "velocity_rmse_mps": 5e-4,
        "velocity_max_abs_mps": 2e-3,
        "angular_velocity_rmse_rad_s": 5e-4,
        "angular_velocity_max_abs_rad_s": 2e-3,
        "front_steering_max_abs_rad": 2e-4,
        "wheel_speed_max_abs_rad_s": 5e-3,
        "wheel_contact_fraction_rmse": 0.05,
        "wheel_contact_fraction_max_abs": 0.2,
        "contact_gap_max_abs_difference_s": 0.01,
        "penetration_max_abs_difference_m": 5e-4,
    }
    checks = [
        {
            "id": f"cpu_gpu_consistency.{metric}",
            "passed": metrics[metric] <= limit,
            "value": metrics[metric],
            "operator": "<=",
            "limit": limit,
        }
        for metric, limit in tolerances.items()
    ]
    checks.extend(
        (
            {
                "id": "cpu_gpu_consistency.finite",
                "passed": finite,
                "value": finite,
                "operator": "==",
                "limit": True,
            },
            {
                "id": "cpu_gpu_consistency.no_buffer_overflow",
                "passed": no_overflow,
                "value": no_overflow,
                "operator": "==",
                "limit": True,
            },
            {
                "id": "cpu_gpu_consistency.contacts_valid",
                "passed": contacts_valid,
                "value": contacts_valid,
                "operator": "==",
                "limit": True,
            },
        )
    )
    return {
        "status": "pass" if all(check["passed"] for check in checks) else "fail",
        "duration_s": 5.0,
        "environment_steps": 100,
        "schedule_timebase": "integer control-step index",
        "metrics": metrics,
        "tolerances": tolerances,
        "checks": checks,
        "timing": {"compilation_s": compilation_s, "run_s": run_s},
    }


def _check(check_id: str, passed: bool, value: Any, expected: Any) -> dict[str, Any]:
    return {"id": check_id, "passed": bool(passed), "value": value, "expected": expected}


def _run_gpu_scale(args: argparse.Namespace) -> dict[str, Any]:
    import jax
    import jax.numpy as jnp
    import mujoco
    import numpy as np
    import torch

    from controller_learning.config import load_vehicle_config
    from controller_learning.physics.mjx_warp import MjxWarpVehicle

    project_root = args.project_root.resolve()
    config = load_vehicle_config(project_root / "configs/vehicle.toml")
    devices = jax.devices("gpu")
    if not devices:
        raise RuntimeError("JAX found no GPU device")
    device = devices[0]
    nvidia_gpus, nvidia_error = _nvidia_hardware()
    selected_gpu, gpu_selection_error = _selected_nvidia_gpu(device, nvidia_gpus)
    selected_gpu_uuid = selected_gpu["uuid"] if selected_gpu else None
    samples = [
        _memory_sample(
            device,
            "baseline",
            0,
            gpu_uuid=selected_gpu_uuid,
        )
    ]
    creation_started = time.perf_counter()
    vehicle = MjxWarpVehicle.create(
        config,
        num_worlds=args.num_worlds,
        contacts_per_world=args.contacts_per_world,
        constraints_per_world=args.constraints_per_world,
    )
    creation_s = time.perf_counter() - creation_started
    samples.append(_memory_sample(device, "created", 0, gpu_uuid=selected_gpu_uuid))
    chunk_function = _build_chunk_function(
        vehicle,
        chunk_steps=args.chunk_steps,
        jax=jax,
        jnp=jnp,
    )
    initial_state = vehicle.initial_state()
    start_step = jnp.asarray(0, dtype=jnp.int32)
    compilation_started = time.perf_counter()
    compiled_chunk = chunk_function.lower(initial_state, start_step).compile()
    compilation_s = time.perf_counter() - compilation_started
    samples.append(_memory_sample(device, "compiled", 0, gpu_uuid=selected_gpu_uuid))
    warmup_started = time.perf_counter()
    warmup_state = initial_state
    for warmup_index in range(DEFAULT_WARMUP_CHUNKS):
        warmup_state, warmup_aggregate = compiled_chunk(
            warmup_state,
            jnp.asarray(warmup_index * args.chunk_steps, dtype=jnp.int32),
        )
        jax.block_until_ready((warmup_state, warmup_aggregate))
    warmup_s = time.perf_counter() - warmup_started
    del warmup_state, warmup_aggregate
    gc.collect()
    samples.append(_memory_sample(device, "warmup", 0, gpu_uuid=selected_gpu_uuid))

    state = initial_state
    host_aggregate = _new_host_aggregate()
    execution_s = 0.0
    monitored_started = time.perf_counter()
    for start in range(0, args.environment_steps, args.chunk_steps):
        chunk_started = time.perf_counter()
        state, chunk_aggregate = compiled_chunk(
            state,
            jnp.asarray(start, dtype=jnp.int32),
        )
        jax.block_until_ready((state, chunk_aggregate))
        execution_s += time.perf_counter() - chunk_started
        _merge_aggregate(host_aggregate, jax.device_get(chunk_aggregate))
        samples.append(
            _memory_sample(
                device,
                "measured",
                start + args.chunk_steps,
                gpu_uuid=selected_gpu_uuid,
            )
        )
    monitored_wall_s = time.perf_counter() - monitored_started
    view = jax.device_get(vehicle.read_state(state))
    signatures = np.round(
        np.column_stack(
            (
                np.asarray(view.position_world_m)[:, :2],
                np.asarray(view.velocity_body_mps)[:, 0],
                np.asarray(view.yaw_rad),
            )
        ),
        decimals=3,
    )
    unique_terminal_states = int(np.unique(signatures, axis=0).shape[0])
    terminal_state_spread = float(np.max(np.std(signatures, axis=0)))
    minimum_unique_states = min(args.num_worlds, 4)
    independence_passed = args.num_worlds == 1 or (
        unique_terminal_states >= minimum_unique_states and terminal_state_spread > 1e-4
    )
    cpu_gpu_consistency = None
    if args.num_worlds == 1:
        cpu_gpu_consistency = _cpu_gpu_consistency(
            vehicle,
            config,
            jax=jax,
            jnp=jnp,
            np=np,
        )
        samples.append(
            _memory_sample(
                device,
                "post_consistency",
                args.environment_steps,
                gpu_uuid=selected_gpu_uuid,
            )
        )
    memory = _memory_report(samples)
    implementation = vehicle.initial_state().data._impl
    capacities = {
        "contacts_per_world_requested": args.contacts_per_world,
        "naconmax_global": int(implementation.naconmax),
        "constraints_per_world_requested": args.constraints_per_world,
        "njmax_runtime": int(implementation.njmax),
        "njmax_nnz_runtime": int(implementation.njmax_nnz),
        "njmax_pad_runtime": int(implementation.njmax_pad),
        "naccdmax_global": int(implementation.naccdmax),
        "peak_nacon_global": host_aggregate["peak_nacon"],
        "peak_ncollision_global": host_aggregate["peak_ncollision"],
        "peak_nefc_per_world": host_aggregate["peak_nefc"],
        "nacon_headroom_fraction": host_aggregate["peak_nacon"] / int(implementation.naconmax),
        "ncollision_headroom_fraction": (
            host_aggregate["peak_ncollision"] / int(implementation.naconmax)
        ),
        "nefc_headroom_fraction": host_aggregate["peak_nefc"] / int(implementation.njmax),
        "scope_note": (
            "naconmax/naccdmax are global flattened Warp buffers; peak nefc is the maximum "
            "per-world count observed after vmap. njmax_nnz is recorded from Data._impl."
        ),
    }
    transitions = args.num_worlds * args.environment_steps
    physics_steps = transitions * vehicle.physics_substeps_per_control
    timing = {
        "adapter_creation_s": creation_s,
        "compilation_s": compilation_s,
        "warmup_s": warmup_s,
        "warmup_chunks": DEFAULT_WARMUP_CHUNKS,
        "warmup_environment_steps": DEFAULT_WARMUP_CHUNKS * args.chunk_steps,
        "measured_execution_s": execution_s,
        "monitored_wall_s": monitored_wall_s,
        "environment_steps_per_second": args.environment_steps / execution_s,
        "transitions_per_second": transitions / execution_s,
        "world_physics_steps_per_second": physics_steps / execution_s,
        "total_transitions": transitions,
        "total_world_physics_steps": physics_steps,
    }
    numerical = {
        **host_aggregate,
        "terminal_unique_state_signatures": unique_terminal_states,
        "terminal_state_spread": terminal_state_spread,
    }
    checks = [
        _check("numerical.finite", host_aggregate["finite"], host_aggregate["finite"], True),
        _check(
            "numerical.time_monotonic",
            host_aggregate["time_monotonic"],
            host_aggregate["time_monotonic"],
            True,
        ),
        _check(
            "buffers.contact_overflow",
            not host_aggregate["contact_overflow"],
            host_aggregate["contact_overflow"],
            False,
        ),
        _check(
            "buffers.constraint_overflow",
            not host_aggregate["constraint_overflow"],
            host_aggregate["constraint_overflow"],
            False,
        ),
        _check(
            "contacts.unexpected",
            not host_aggregate["unexpected_contact"],
            host_aggregate["unexpected_contact"],
            False,
        ),
        _check(
            "buffers.contact_headroom",
            capacities["nacon_headroom_fraction"] <= 0.8,
            capacities["nacon_headroom_fraction"],
            "<= 0.8",
        ),
        _check(
            "buffers.collision_headroom",
            capacities["ncollision_headroom_fraction"] <= 0.8,
            capacities["ncollision_headroom_fraction"],
            "<= 0.8",
        ),
        _check(
            "buffers.constraint_headroom",
            capacities["nefc_headroom_fraction"] <= 0.8,
            capacities["nefc_headroom_fraction"],
            "<= 0.8",
        ),
        _check(
            "state.chassis_z_bounds",
            host_aggregate["minimum_chassis_z_m"] >= 0.2
            and host_aggregate["maximum_chassis_z_m"] <= 2.0,
            [host_aggregate["minimum_chassis_z_m"], host_aggregate["maximum_chassis_z_m"]],
            "within [0.2, 2.0] m",
        ),
        _check(
            "state.planar_speed_bound",
            host_aggregate["maximum_planar_speed_mps"] <= config.vehicle.max_speed_mps + 2.0,
            host_aggregate["maximum_planar_speed_mps"],
            f"<= {config.vehicle.max_speed_mps + 2.0} m/s",
        ),
        _check(
            "contacts.penetration_bound",
            host_aggregate["maximum_penetration_m"] <= MAXIMUM_PENETRATION_M,
            host_aggregate["maximum_penetration_m"],
            f"<= {MAXIMUM_PENETRATION_M} m",
        ),
        _check(
            "contacts.maximum_wheel_gap",
            host_aggregate["maximum_wheel_contact_gap_s"] <= MAXIMUM_WHEEL_CONTACT_GAP_S,
            host_aggregate["maximum_wheel_contact_gap_s"],
            f"<= {MAXIMUM_WHEEL_CONTACT_GAP_S} s",
        ),
        _check(
            "contacts.minimum_chunk_mean_wheel_coverage",
            host_aggregate["minimum_chunk_mean_wheel_contact_fraction"]
            >= MINIMUM_CHUNK_MEAN_WHEEL_CONTACT_FRACTION,
            host_aggregate["minimum_chunk_mean_wheel_contact_fraction"],
            f">= {MINIMUM_CHUNK_MEAN_WHEEL_CONTACT_FRACTION}",
        ),
        _check(
            "state.quaternion_norm",
            host_aggregate["maximum_quaternion_norm_error"] <= MAXIMUM_QUATERNION_NORM_ERROR,
            host_aggregate["maximum_quaternion_norm_error"],
            f"<= {MAXIMUM_QUATERNION_NORM_ERROR}",
        ),
        _check(
            "state.roll_pitch_bound",
            host_aggregate["maximum_abs_roll_pitch_rad"] <= MAXIMUM_ABS_ROLL_PITCH_RAD,
            host_aggregate["maximum_abs_roll_pitch_rad"],
            f"<= {MAXIMUM_ABS_ROLL_PITCH_RAD} rad",
        ),
        _check(
            "state.vertical_speed_bound",
            host_aggregate["maximum_abs_vertical_speed_mps"] <= MAXIMUM_ABS_VERTICAL_SPEED_MPS,
            host_aggregate["maximum_abs_vertical_speed_mps"],
            f"<= {MAXIMUM_ABS_VERTICAL_SPEED_MPS} m/s",
        ),
        _check(
            "state.generalized_velocity_bound",
            host_aggregate["maximum_abs_qvel"] <= MAXIMUM_ABS_QVEL,
            host_aggregate["maximum_abs_qvel"],
            f"<= {MAXIMUM_ABS_QVEL}",
        ),
        _check(
            "state.independent", independence_passed, unique_terminal_states, minimum_unique_states
        ),
        _check(
            "reset.masked_exercised",
            args.num_worlds == 1 or host_aggregate["masked_reset_count"] > 0,
            host_aggregate["masked_reset_count"],
            "> 0 for batched scales",
        ),
        _check(
            "memory.coverage",
            memory["process_vram_coverage"],
            memory["measured_sample_count"],
            ">= 4",
        ),
        _check(
            "memory.stable", memory["stable"], memory["process_vram_growth_mib"], "within limit"
        ),
        _check(
            "timing.positive_throughput",
            math.isfinite(timing["transitions_per_second"])
            and timing["transitions_per_second"] > 0.0,
            timing["transitions_per_second"],
            "> 0",
        ),
        _check(
            "actions.valid",
            host_aggregate["invalid_action_count"] == 0,
            host_aggregate["invalid_action_count"],
            0,
        ),
    ]
    if cpu_gpu_consistency is not None:
        checks.append(
            _check(
                "cpu_gpu_consistency.pass",
                cpu_gpu_consistency["status"] == "pass",
                cpu_gpu_consistency["status"],
                "pass",
            )
        )
    gpu_total_mib = selected_gpu["memory_total_mib"] if selected_gpu else None
    peak_process_mib = memory["peak_process_vram_mib"]
    peak_memory_fraction = (
        peak_process_mib / gpu_total_mib
        if peak_process_mib is not None and gpu_total_mib is not None
        else None
    )
    memory["selected_gpu_total_mib"] = gpu_total_mib
    memory["peak_process_memory_fraction"] = peak_memory_fraction
    runtime = {
        "os": platform.platform(),
        "kernel": platform.release(),
        "machine": platform.machine(),
        "python_version": sys.version.split()[0],
        "jax_version": jax.__version__,
        "jaxlib_version": _package_version("jaxlib"),
        "jax_cuda12_plugin_version": _package_version("jax-cuda12-plugin"),
        "mujoco_version": mujoco.__version__,
        "mujoco_mjx_version": _package_version("mujoco-mjx"),
        "warp_version": _package_version("warp-lang"),
        "numpy_version": np.__version__,
        "torch_version": torch.__version__,
        "torch_cuda_version": torch.version.cuda,
        "jax_device": {
            "description": str(device),
            "platform": device.platform,
            "device_kind": getattr(device, "device_kind", None),
            "id": getattr(device, "id", None),
        },
        "nvidia_gpus": nvidia_gpus,
        "nvidia_smi_error": nvidia_error,
        "gpu_selection_error": gpu_selection_error,
        "xla_python_client_preallocate": os.environ.get("XLA_PYTHON_CLIENT_PREALLOCATE"),
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "gpu_environment": {
            key: value
            for key, value in sorted(os.environ.items())
            if key.startswith(("CUDA_", "JAX_", "XLA_"))
        },
    }
    checks.append(
        _check(
            "runtime.nvidia_smi",
            nvidia_error is None and gpu_selection_error is None,
            {"inventory_error": nvidia_error, "selection_error": gpu_selection_error},
            {"inventory_error": None, "selection_error": None},
        )
    )
    checks.extend(
        (
            _check(
                "memory.jax_allocator_coverage",
                memory["jax_allocator_coverage"],
                memory["measured_sample_count"],
                ">= 4",
            ),
            _check(
                "memory.peak_process_budget",
                peak_memory_fraction is not None and peak_memory_fraction <= 0.95,
                peak_memory_fraction,
                "<= 0.95 of selected GPU memory",
            ),
        )
    )
    passed = all(check["passed"] for check in checks)
    return {
        "schema_version": WORKER_SCHEMA_VERSION,
        "protocol_version": PROTOCOL_VERSION,
        "status": "pass" if passed else "fail",
        "process_id": os.getpid(),
        "num_worlds": args.num_worlds,
        "environment_steps": args.environment_steps,
        "chunk_steps": args.chunk_steps,
        "physics_substeps_per_environment_step": vehicle.physics_substeps_per_control,
        "runtime": runtime,
        "capacities": capacities,
        "numerical": numerical,
        "timing": timing,
        "memory": memory,
        "cpu_gpu_consistency": cpu_gpu_consistency,
        "checks": checks,
    }


def _failure_result(args: argparse.Namespace, error: BaseException) -> dict[str, Any]:
    return {
        "schema_version": WORKER_SCHEMA_VERSION,
        "protocol_version": PROTOCOL_VERSION,
        "status": "fail",
        "process_id": os.getpid(),
        "num_worlds": args.num_worlds,
        "environment_steps": args.environment_steps,
        "chunk_steps": args.chunk_steps,
        "physics_substeps_per_environment_step": 10,
        "runtime": {
            "os": platform.platform(),
            "python_version": sys.version.split()[0],
            "xla_python_client_preallocate": os.environ.get("XLA_PYTHON_CLIENT_PREALLOCATE"),
        },
        "capacities": {},
        "numerical": {},
        "timing": {},
        "memory": {},
        "cpu_gpu_consistency": (
            {"status": "fail", "error": "worker failed before consistency"}
            if args.num_worlds == 1
            else None
        ),
        "checks": [],
        "error": {
            "type": type(error).__name__,
            "message": str(error),
            "traceback": traceback.format_exc()[-16_000:],
        },
    }


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", type=Path, required=True)
    parser.add_argument("--num-worlds", type=int, required=True)
    parser.add_argument("--environment-steps", type=int, required=True)
    parser.add_argument("--chunk-steps", type=int, required=True)
    parser.add_argument("--contacts-per-world", type=int, required=True)
    parser.add_argument("--constraints-per-world", type=int, required=True)
    parser.add_argument("--output", type=Path)
    return parser


def main() -> None:
    """Run one worker while keeping its machine-readable stdout unambiguous."""

    args = _parser().parse_args()
    if args.environment_steps <= 0 or args.chunk_steps <= 0:
        raise SystemExit("step counts must be positive")
    if args.environment_steps % args.chunk_steps:
        raise SystemExit("environment steps must be divisible by chunk steps")
    with tempfile.TemporaryDirectory(prefix="controller-learning-m2-capture-") as temporary:
        capture_directory = Path(temporary)
        with _capture_native_output(capture_directory) as (stdout_path, stderr_path):
            try:
                result = _run_gpu_scale(args)
            except Exception as error:
                result = _failure_result(args, error)
        warnings = _warning_lines(stdout_path, stderr_path)
        captured_stdout = _bounded_text(stdout_path)
        captured_stderr = _bounded_text(stderr_path)
    result["captured_output"] = {
        "stdout_tail": captured_stdout,
        "stderr_tail": captured_stderr,
        "warning_lines": warnings,
        "unexpected_warning_count": len(warnings),
    }
    result["checks"].append(_check("runtime.unexpected_warnings", not warnings, len(warnings), 0))
    result["status"] = (
        "pass"
        if "error" not in result
        and result["checks"]
        and all(check["passed"] for check in result["checks"])
        else "fail"
    )
    if args.output is not None:
        write_strict_json(args.output, result)
    print(
        WORKER_JSON_PREFIX
        + __import__("json").dumps(result, separators=(",", ":"), allow_nan=False)
    )
    raise SystemExit(0 if result["status"] == "pass" else 1)


if __name__ == "__main__":
    main()
