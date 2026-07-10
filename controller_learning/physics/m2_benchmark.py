"""Formal M2 MJX-Warp benchmark orchestration and report contracts.

This module deliberately avoids importing JAX.  Each scale is executed by a
fresh worker process so device initialization, compilation, and memory
measurements cannot leak from one scale into another.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import subprocess
import sys
import tempfile
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

PROTOCOL_VERSION = "m2-mjx-warp-v1"
WORKER_SCHEMA_VERSION = 1
REPORT_SCHEMA_VERSION = 1
WORKER_JSON_PREFIX = "M2_WORKER_JSON="
REQUIRED_WORLD_COUNTS = (1, 64, 256, 1024)
DEFAULT_SCALE_STEPS = {
    1: 1_000,
    64: 1_000,
    256: 1_000,
    1024: 10_000,
}
DEFAULT_CHUNK_STEPS = 100
DEFAULT_WARMUP_CHUNKS = 8
DEFAULT_CONTACTS_PER_WORLD = 16
DEFAULT_CONSTRAINTS_PER_WORLD = 64
MAXIMUM_PENETRATION_M = 0.005
MAXIMUM_WHEEL_CONTACT_GAP_S = 0.1
MAXIMUM_QUATERNION_NORM_ERROR = 1e-5
MAXIMUM_ABS_ROLL_PITCH_RAD = 0.2
MAXIMUM_ABS_VERTICAL_SPEED_MPS = 2.0
MAXIMUM_ABS_QVEL = 100.0
MINIMUM_CHUNK_MEAN_WHEEL_CONTACT_FRACTION = 0.75


@dataclass(frozen=True, slots=True)
class ScaleSpec:
    """One isolated world-count benchmark invocation."""

    num_worlds: int
    environment_steps: int
    chunk_steps: int = DEFAULT_CHUNK_STEPS

    def __post_init__(self) -> None:
        if self.num_worlds <= 0:
            raise ValueError("num_worlds must be positive")
        if self.environment_steps <= 0:
            raise ValueError("environment_steps must be positive")
        if self.chunk_steps <= 0:
            raise ValueError("chunk_steps must be positive")
        if self.environment_steps % self.chunk_steps:
            raise ValueError("environment_steps must be divisible by chunk_steps")


FORMAL_SCALE_SPECS = tuple(
    ScaleSpec(num_worlds, DEFAULT_SCALE_STEPS[num_worlds]) for num_worlds in REQUIRED_WORLD_COUNTS
)


def sha256_file(path: Path) -> str:
    """Return the SHA-256 digest of one evidence input."""

    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_strict_json(path: Path, payload: Mapping[str, Any]) -> None:
    """Atomically write strict JSON, rejecting NaN and infinity."""

    path = path.resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    serialized = json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n"
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(serialized)
    temporary.replace(path)


def extract_worker_json(stdout: str) -> dict[str, Any] | None:
    """Extract the last sentinel-delimited worker object from noisy stdout."""

    for line in reversed(stdout.splitlines()):
        if not line.startswith(WORKER_JSON_PREFIX):
            continue
        try:
            payload = json.loads(line.removeprefix(WORKER_JSON_PREFIX))
        except json.JSONDecodeError:
            continue
        if isinstance(payload, dict):
            return payload
    return None


def validate_worker_result(payload: Mapping[str, Any], spec: ScaleSpec) -> tuple[str, ...]:
    """Return contract violations for a scale worker result."""

    violations: list[str] = []
    expected = {
        "schema_version": WORKER_SCHEMA_VERSION,
        "protocol_version": PROTOCOL_VERSION,
        "num_worlds": spec.num_worlds,
        "environment_steps": spec.environment_steps,
        "chunk_steps": spec.chunk_steps,
        "physics_substeps_per_environment_step": 10,
    }
    for field, value in expected.items():
        if payload.get(field) != value:
            violations.append(f"{field}: expected {value!r}, got {payload.get(field)!r}")
    if payload.get("status") not in {"pass", "fail"}:
        violations.append("status must be 'pass' or 'fail'")
    if not isinstance(payload.get("process_id"), int):
        violations.append("process_id must be an integer")
    for field in ("runtime", "capacities", "numerical", "timing", "memory"):
        if not isinstance(payload.get(field), dict):
            violations.append(f"{field} must be an object")
    checks = payload.get("checks")
    if (
        not isinstance(checks, list)
        or not checks
        or not all(isinstance(check, dict) for check in checks)
    ):
        violations.append("checks must be a non-empty list of objects")
    elif payload.get("status") == "pass" and not all(
        check.get("passed") is True and isinstance(check.get("id"), str) for check in checks
    ):
        violations.append("passing worker must contain only passing named checks")
    if spec.num_worlds == 1 and not isinstance(payload.get("cpu_gpu_consistency"), dict):
        violations.append("batch-one worker must include cpu_gpu_consistency")
    return tuple(violations)


def _git(project_root: Path, *args: str) -> str:
    return subprocess.run(
        ("git", *args),
        cwd=project_root,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


def _source_hashes(project_root: Path, *, worker_path: Path) -> dict[str, str]:
    paths = {
        "pixi_lock": project_root / "pixi.lock",
        "vehicle_model": project_root / "controller_learning/assets/vehicle/car.xml",
        "vehicle_config": project_root / "configs/vehicle.toml",
        "mjx_warp_adapter": project_root / "controller_learning/physics/mjx_warp.py",
        "benchmark_protocol": project_root / "controller_learning/physics/m2_benchmark.py",
        "worker": worker_path,
        "launcher": project_root / "scripts/benchmark_gpu.py",
    }
    return {name: sha256_file(path) for name, path in paths.items()}


def _read_worker_payload(output: Path, stdout: str) -> dict[str, Any] | None:
    if output.is_file():
        try:
            payload = json.loads(output.read_text())
        except (OSError, json.JSONDecodeError):
            payload = None
        if isinstance(payload, dict):
            return payload
    return extract_worker_json(stdout)


def _output_tail(output: str | bytes | None, maximum_chars: int = 8_000) -> str:
    """Normalize subprocess output, including TimeoutExpired byte payloads."""

    if output is None:
        return ""
    if isinstance(output, bytes):
        output = output.decode(errors="replace")
    return output[-maximum_chars:]


def _redact_launcher_text(
    value: str,
    *,
    project_root: Path,
    worker_path: Path,
    output_path: Path,
) -> str:
    replacements = (
        (str(output_path), "<temporary_output>"),
        (str(worker_path), "<worker>"),
        (str(project_root), "<project_root>"),
        (str(Path.home()), "$HOME"),
    )
    for source, replacement in replacements:
        value = value.replace(source, replacement)
    return re.sub(r"(?:GPU|MIG)-[0-9A-Fa-f-]+", "<gpu_uuid>", value)


def _display_command(
    command: Sequence[str],
    *,
    project_root: Path,
    worker_path: Path,
    output_path: Path,
) -> list[str]:
    return [
        "<python>"
        if value == sys.executable
        else _redact_launcher_text(
            value,
            project_root=project_root,
            worker_path=worker_path,
            output_path=output_path,
        )
        for value in command
    ]


def redact_evidence_payload(
    value: Any,
    *,
    project_root: Path,
    temporary_root: Path,
) -> Any:
    """Remove machine-unique paths and GPU identifiers from persisted evidence."""

    if isinstance(value, str):
        replacements = (
            (str(project_root.resolve()), "<project_root>"),
            (str(temporary_root.resolve()), "<temporary_dir>"),
            (str(Path.home()), "$HOME"),
        )
        for source, replacement in replacements:
            value = value.replace(source, replacement)
        return re.sub(r"(?:GPU|MIG)-[0-9A-Fa-f-]+", "<gpu_uuid>", value)
    if isinstance(value, dict):
        return {
            key: redact_evidence_payload(
                item,
                project_root=project_root,
                temporary_root=temporary_root,
            )
            for key, item in value.items()
            if key != "uuid"
        }
    if isinstance(value, list):
        return [
            redact_evidence_payload(
                item,
                project_root=project_root,
                temporary_root=temporary_root,
            )
            for item in value
        ]
    return value


def _failed_worker_payload(
    spec: ScaleSpec,
    *,
    message: str,
    returncode: int | None,
) -> dict[str, Any]:
    return {
        "schema_version": WORKER_SCHEMA_VERSION,
        "protocol_version": PROTOCOL_VERSION,
        "status": "fail",
        "process_id": -1,
        "num_worlds": spec.num_worlds,
        "environment_steps": spec.environment_steps,
        "chunk_steps": spec.chunk_steps,
        "physics_substeps_per_environment_step": 10,
        "runtime": {},
        "capacities": {},
        "numerical": {},
        "timing": {},
        "memory": {},
        "checks": [],
        "error": {
            "type": "WorkerProcessError",
            "message": message,
            "returncode": returncode,
        },
    }


def _run_scale_worker(
    project_root: Path,
    worker_path: Path,
    spec: ScaleSpec,
    *,
    contacts_per_world: int,
    constraints_per_world: int,
    timeout_s: float | None,
    extra_environment: Mapping[str, str] | None,
) -> dict[str, Any]:
    with tempfile.TemporaryDirectory(prefix=f"controller-learning-m2-{spec.num_worlds}-") as tmp:
        output = Path(tmp) / "worker.json"
        command = (
            sys.executable,
            str(worker_path),
            "--project-root",
            str(project_root),
            "--num-worlds",
            str(spec.num_worlds),
            "--environment-steps",
            str(spec.environment_steps),
            "--chunk-steps",
            str(spec.chunk_steps),
            "--contacts-per-world",
            str(contacts_per_world),
            "--constraints-per-world",
            str(constraints_per_world),
            "--output",
            str(output),
        )
        environment = os.environ.copy()
        environment["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
        environment.setdefault("CUDA_DEVICE_ORDER", "PCI_BUS_ID")
        if extra_environment:
            environment.update(extra_environment)
        try:
            completed = subprocess.run(
                command,
                cwd=project_root,
                env=environment,
                capture_output=True,
                text=True,
                timeout=timeout_s,
                check=False,
            )
        except subprocess.TimeoutExpired as error:
            result = _failed_worker_payload(
                spec,
                message=f"worker exceeded timeout of {timeout_s} seconds",
                returncode=None,
            )
            result["launcher"] = {
                "command": _display_command(
                    command,
                    project_root=project_root,
                    worker_path=worker_path,
                    output_path=output,
                ),
                "stdout_tail": _redact_launcher_text(
                    _output_tail(error.stdout),
                    project_root=project_root,
                    worker_path=worker_path,
                    output_path=output,
                ),
                "stderr_tail": _redact_launcher_text(
                    _output_tail(error.stderr),
                    project_root=project_root,
                    worker_path=worker_path,
                    output_path=output,
                ),
            }
            return result

        payload = _read_worker_payload(output, completed.stdout)
        if payload is None:
            payload = _failed_worker_payload(
                spec,
                message="worker produced neither a valid output file nor sentinel JSON",
                returncode=completed.returncode,
            )
        else:
            payload = dict(payload)
        violations = validate_worker_result(payload, spec)
        if violations:
            payload["status"] = "fail"
            payload["contract_violations"] = list(violations)
        if completed.returncode != 0 and payload.get("status") == "pass":
            payload["status"] = "fail"
            payload.setdefault("contract_violations", []).append(
                f"passing worker exited with status {completed.returncode}"
            )
        payload["launcher"] = {
            "command": _display_command(
                command,
                project_root=project_root,
                worker_path=worker_path,
                output_path=output,
            ),
            "returncode": completed.returncode,
            "stdout_tail": _redact_launcher_text(
                completed.stdout[-8_000:],
                project_root=project_root,
                worker_path=worker_path,
                output_path=output,
            ),
            "stderr_tail": _redact_launcher_text(
                completed.stderr[-8_000:],
                project_root=project_root,
                worker_path=worker_path,
                output_path=output,
            ),
        }
        return payload


def _formal_specs_match(specs: Sequence[ScaleSpec]) -> bool:
    return tuple(specs) == FORMAL_SCALE_SPECS


def _check(check_id: str, passed: bool, value: Any, expected: Any) -> dict[str, Any]:
    return {
        "id": check_id,
        "passed": bool(passed),
        "value": value,
        "expected": expected,
    }


def run_m2_benchmark(
    project_root: Path,
    *,
    scale_specs: Sequence[ScaleSpec] = FORMAL_SCALE_SPECS,
    worker_path: Path | None = None,
    contacts_per_world: int = DEFAULT_CONTACTS_PER_WORLD,
    constraints_per_world: int = DEFAULT_CONSTRAINTS_PER_WORLD,
    timeout_s: float | None = 7_200.0,
    extra_environment: Mapping[str, str] | None = None,
) -> dict[str, Any]:
    """Run every GPU scale in a fresh process and merge the formal M2 report."""

    project_root = project_root.resolve()
    specs = tuple(scale_specs)
    if not specs:
        raise ValueError("at least one scale spec is required")
    if contacts_per_world <= 0 or constraints_per_world <= 0:
        raise ValueError("contact and constraint capacities must be positive")
    canonical_worker_path = (project_root / "scripts/benchmark_gpu_worker.py").resolve()
    worker_path = (worker_path or canonical_worker_path).resolve()
    git_commit_before = _git(project_root, "rev-parse", "HEAD")
    git_dirty_before = bool(_git(project_root, "status", "--porcelain"))
    source_hashes_before = _source_hashes(project_root, worker_path=worker_path)
    results = [
        _run_scale_worker(
            project_root,
            worker_path,
            spec,
            contacts_per_world=contacts_per_world,
            constraints_per_world=constraints_per_world,
            timeout_s=timeout_s,
            extra_environment=extra_environment,
        )
        for spec in specs
    ]
    git_commit_after = _git(project_root, "rev-parse", "HEAD")
    git_dirty_after = bool(_git(project_root, "status", "--porcelain"))
    source_hashes_after = _source_hashes(project_root, worker_path=worker_path)
    formal_specs = _formal_specs_match(specs)
    canonical_worker = worker_path == canonical_worker_path
    canonical_environment = not extra_environment
    stable_sources = (
        git_commit_before == git_commit_after and source_hashes_before == source_hashes_after
    )
    clean_worktree = not git_dirty_before and not git_dirty_after
    process_ids = [result.get("process_id") for result in results]
    valid_process_ids = all(
        isinstance(process_id, int) and process_id > 0 for process_id in process_ids
    )
    fresh_processes = valid_process_ids and len(set(process_ids)) == len(process_ids)
    all_workers_passed = all(result.get("status") == "pass" for result in results)
    consistency = next(
        (result.get("cpu_gpu_consistency") for result in results if result.get("num_worlds") == 1),
        None,
    )
    consistency_passed = isinstance(consistency, dict) and consistency.get("status") == "pass"
    checks = [
        _check(
            "protocol.formal_scale_specs",
            formal_specs,
            [
                {
                    "num_worlds": spec.num_worlds,
                    "environment_steps": spec.environment_steps,
                    "chunk_steps": spec.chunk_steps,
                }
                for spec in specs
            ],
            "1/64/256 worlds for 1,000 steps and 1024 worlds for 10,000 steps",
        ),
        _check(
            "launcher.fresh_worker_processes", fresh_processes, process_ids, "unique positive PIDs"
        ),
        _check(
            "protocol.canonical_worker",
            canonical_worker,
            worker_path.name,
            canonical_worker_path.name,
        ),
        _check(
            "protocol.canonical_environment",
            canonical_environment,
            sorted(extra_environment or {}),
            [],
        ),
        _check("workers.all_passed", all_workers_passed, all_workers_passed, True),
        _check("batch_one.cpu_gpu_consistency", consistency_passed, consistency_passed, True),
        _check("evidence.clean_worktree", clean_worktree, clean_worktree, True),
        _check("evidence.stable_sources", stable_sources, stable_sources, True),
    ]
    passed = all(check["passed"] for check in checks)
    protocol = {
        "protocol_version": PROTOCOL_VERSION,
        "scale_specs": [
            {
                "num_worlds": spec.num_worlds,
                "environment_steps": spec.environment_steps,
                "chunk_steps": spec.chunk_steps,
            }
            for spec in specs
        ],
        "control_dt_s": 0.05,
        "physics_dt_s": 0.005,
        "physics_substeps_per_environment_step": 10,
        "contacts_per_world": contacts_per_world,
        "constraints_per_world_requested": constraints_per_world,
        "extra_environment_keys": sorted(extra_environment or {}),
        "physical_limits": {
            "maximum_penetration_m": MAXIMUM_PENETRATION_M,
            "maximum_wheel_contact_gap_s": MAXIMUM_WHEEL_CONTACT_GAP_S,
            "maximum_quaternion_norm_error": MAXIMUM_QUATERNION_NORM_ERROR,
            "maximum_abs_roll_pitch_rad": MAXIMUM_ABS_ROLL_PITCH_RAD,
            "maximum_abs_vertical_speed_mps": MAXIMUM_ABS_VERTICAL_SPEED_MPS,
            "maximum_abs_qvel": MAXIMUM_ABS_QVEL,
            "minimum_chunk_mean_wheel_contact_fraction": (
                MINIMUM_CHUNK_MEAN_WHEEL_CONTACT_FRACTION
            ),
        },
        "compilation_timing": "explicit lower().compile(), excluded from measured execution",
        "warmup": {
            "chunks": DEFAULT_WARMUP_CHUNKS,
            "purpose": (
                "settle lazy kernels and allocator pools before memory and throughput sampling"
            ),
        },
        "memory_sampling": (
            "process VRAM from nvidia-smi and allocator statistics from the JAX device, "
            "sampled after synchronized chunks"
        ),
        "track_scope": (
            "M2 is a physics-only gate. Per-world deterministic actions and masked resets are "
            "exercised here; independent procedural tracks are deferred until the M3 Track layer."
        ),
        "cpu_gpu_consistency": {
            "batch_size": 1,
            "duration_s": 5.0,
            "environment_steps": 100,
            "schedule": "fixed piecewise action schedule using integer control-step indices",
        },
    }
    protocol_sha256 = hashlib.sha256(
        json.dumps(protocol, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    return {
        "schema_version": REPORT_SCHEMA_VERSION,
        "milestone": "M2",
        "status": "pass" if passed else "fail",
        "generated_at_utc": datetime.now(UTC).isoformat(),
        "provenance": {
            "git_commit": git_commit_before,
            "git_commit_after": git_commit_after,
            "git_dirty": not clean_worktree,
            "git_dirty_before": git_dirty_before,
            "git_dirty_after": git_dirty_after,
            "protocol_sha256": protocol_sha256,
            "source_sha256": source_hashes_before,
            "source_sha256_after": source_hashes_after,
        },
        "protocol": protocol,
        "cpu_gpu_consistency": consistency,
        "scale_results": results,
        "checks": checks,
        "selection": {
            "m2_passed": passed,
            "ready_for_m3": passed,
            "formal_protocol": formal_specs and canonical_worker and canonical_environment,
            "all_workers_passed": all_workers_passed,
            "evidence_valid": clean_worktree and stable_sources,
        },
    }


def write_m2_report(
    project_root: Path,
    output: Path,
    **kwargs: Any,
) -> dict[str, Any]:
    """Run the formal protocol and atomically write strict versioned evidence."""

    report = run_m2_benchmark(project_root, **kwargs)
    write_strict_json(output, report)
    return report


__all__ = [
    "DEFAULT_CHUNK_STEPS",
    "DEFAULT_CONSTRAINTS_PER_WORLD",
    "DEFAULT_CONTACTS_PER_WORLD",
    "DEFAULT_WARMUP_CHUNKS",
    "FORMAL_SCALE_SPECS",
    "MAXIMUM_ABS_QVEL",
    "MAXIMUM_ABS_ROLL_PITCH_RAD",
    "MAXIMUM_ABS_VERTICAL_SPEED_MPS",
    "MAXIMUM_PENETRATION_M",
    "MAXIMUM_QUATERNION_NORM_ERROR",
    "MAXIMUM_WHEEL_CONTACT_GAP_S",
    "MINIMUM_CHUNK_MEAN_WHEEL_CONTACT_FRACTION",
    "PROTOCOL_VERSION",
    "ScaleSpec",
    "extract_worker_json",
    "redact_evidence_payload",
    "run_m2_benchmark",
    "validate_worker_result",
    "write_m2_report",
    "write_strict_json",
]
