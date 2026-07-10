"""Run the formal M6 PID and MPC Controller benchmark on official GPU assets.

The formal workload is intentionally not configurable from the command line. PID is evaluated on
Level 0 and the first ten validation Tracks; MPC is evaluated on Level 0 and all one hundred
validation Tracks. The Test split is never loaded by this script.
"""

from __future__ import annotations

import os

# Keep GPU memory observable and physical-device ordering deterministic before JAX is imported.
os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
os.environ.setdefault("CUDA_DEVICE_ORDER", "PCI_BUS_ID")

import argparse
import hashlib
import json
import math
import platform
import re
import subprocess
import sys
import tempfile
import time
from collections import Counter
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
from scripts import benchmark_racing_env as m4_benchmark

REPORT_SCHEMA_VERSION = "controller-learning.m6-controllers.v1"
PROTOCOL_VERSION = "m6-pid-mpc-gpu-v1"
FORMAL_BACKEND = "mjx_warp"
FORMAL_PID_VALIDATION_TRACKS = 10
FORMAL_MPC_VALIDATION_TRACKS = 100
FORMAL_LEVEL0_TRACKS = 1
REALTIME_P99_LIMIT_S = 0.05
REALTIME_MISS_RATE_LIMIT = 0.01
FORMAL_INIT_TIMEOUT_S = 30.0
FORMAL_ENVIRONMENTS_PER_EPISODE = 1
FORMAL_EPISODE_COUNT = 112
FORMAL_PHYSICS_SUBSTEPS_PER_ENVIRONMENT_STEP = 10
M2_EVIDENCE_PATH = Path("benchmarks/v0.1/gpu_report.json")
M2_EVIDENCE_SHA256 = "22c885cbef07632e7a6fd8bb19cc4abd316cd64c065663823985a56e0ad4a702"
M2_EXPECTED_COMPILATION_S = 0.8990262070001336
M2_EXPECTED_TRANSITIONS_PER_SECOND = 77750.56470048921
M2_EXPECTED_PEAK_PROCESS_VRAM_MIB = 346.0
M5_EVIDENCE_PATH = Path("benchmarks/v0.1/m5_track_pool_report.json")
M5_EVIDENCE_SHA256 = "4e2acf751be2ffcc379ee11baf02bee44ccffa4364e7ab388ca00f1e57888916"
M5_EXPECTED_FIRST_STEP_COMPILE_S = 1.865571317001013
M5_EXPECTED_TRANSITIONS_PER_SECOND = 210371.5072336413
M5_EXPECTED_PEAK_PROCESS_VRAM_MIB = 1334.0
EXECUTION_GROUPS = (
    "pid.level0",
    "pid.validation",
    "mpc.level0",
    "mpc.validation",
)
MEMORY_SAMPLE_PHASES = (
    "before_evaluation",
    "after_first_environment_create",
    "after_first_reset",
    "after_first_step",
    "after_pid_level0",
    "after_pid_validation",
    "after_mpc_level0",
    "after_mpc_validation",
    "after_evaluation",
)
FIRST_USE_TIMING_METHOD = (
    "wall clock around the first actual batch-one environment create, reset, and step calls after "
    "runtime/device discovery; create includes environment/backend initialization, reset/step each "
    "include one execution, JAX context and IPOPT plugin discovery are excluded, and persistent "
    "compilation caches are not cleared"
)
CONTROLLER_EXECUTION_MODEL = "sequential host Controller with batch-one MJX-Warp physics"
THROUGHPUT_SCOPE = (
    "closed-loop batch-one evaluation with per-step host synchronization; end-to-end wall time "
    "includes in-band public numerical checks and first-call evidence sampling, while environment "
    "step-call timing excludes those checks; neither value is native GPU physics throughput"
)
MEMORY_SAMPLING_METHOD = (
    "selected-process VRAM from nvidia-smi plus JAX allocator statistics sampled at synchronized "
    "formal phases"
)
DEFAULT_OUTPUT = Path("benchmarks/v0.1/m6_controller_report.json")
PROJECT_ROOT = Path(__file__).resolve().parents[1]
_SHA256_PATTERN = re.compile(r"^[0-9a-f]{64}$")
_GPU_UUID_PATTERN = re.compile(
    r"\bGPU-[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}\b",
    flags=re.IGNORECASE,
)
_POSIX_ABSOLUTE_PATTERN = re.compile(r"(?<![:/A-Za-z0-9._-])(/[^\s\"'<>]+)")
_WINDOWS_ABSOLUTE_PATTERN = re.compile(r"(?<![A-Za-z0-9._-])([A-Za-z]:[\\/][^\s\"'<>]+)")

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
    M2_EVIDENCE_PATH.as_posix(),
    M5_EVIDENCE_PATH.as_posix(),
    "controllers/pid/config.toml",
    "controllers/pid/controller.py",
    "controllers/pid/helpers.py",
    "controllers/mpc/config.toml",
    "controllers/mpc/controller.py",
    "controllers/mpc/helpers.py",
    "controllers/mpc/solver.py",
    "scripts/benchmark_m6_controllers.py",
    "scripts/benchmark_racing_env.py",
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
ExecutionEvidenceLoader: TypeAlias = Callable[[Mapping[str, Any], ProjectConfig], Mapping[str, Any]]
HistoricalEvidenceLoader: TypeAlias = Callable[[Path], Mapping[str, Any]]


@dataclass(slots=True)
class _EnvironmentCallRecord:
    """Wall-clock evidence for one batch-one environment instance."""

    group: str
    create_s: float
    reset_count: int = 0
    reset_wall_s: float = 0.0
    first_reset_s: float | None = None
    step_count: int = 0
    step_wall_s: float = 0.0
    first_step_s: float | None = None
    closed: bool = False


@dataclass(frozen=True, slots=True)
class _EvaluationGroupRecord:
    """One Controller/split call, excluding only its after-group memory sample."""

    label: str
    episode_count: int
    environment_steps: int
    wall_s: float


class _MeasuredEnvironment:
    """Transparent benchmark-only wrapper around one official ``CarRacingEnv``."""

    def __init__(
        self,
        environment: Any,
        recorder: _ExecutionRecorder,
        record: _EnvironmentCallRecord,
    ) -> None:
        self._environment = environment
        self._recorder = recorder
        self._record = record

    @property
    def unwrapped(self) -> Any:
        return self._environment.unwrapped

    def reset(self, *args: Any, **kwargs: Any) -> Any:
        started = time.perf_counter()
        result = self._environment.reset(*args, **kwargs)
        elapsed = time.perf_counter() - started
        self._recorder.record_reset(self._record, elapsed)
        return result

    def step(self, action: object) -> Any:
        started = time.perf_counter()
        result = self._environment.step(action)
        elapsed = time.perf_counter() - started
        self._recorder.record_step(self._record, elapsed, result)
        return result

    def render(self) -> Any:
        return self._environment.render()

    def close(self) -> None:
        try:
            self._environment.close()
        finally:
            self._record.closed = True


class _ExecutionRecorder:
    """Collect scoped GPU execution evidence without exposing it to a Controller."""

    def __init__(
        self,
        *,
        device: Any,
        gpu_uuid: str | None,
        gpu_selection_error: str | None,
        environment_factory: Callable[..., Any],
        memory_sampler: Callable[[Any, str, str | None], Mapping[str, Any]],
    ) -> None:
        self._device = device
        self._gpu_uuid = gpu_uuid
        self._gpu_selection_error = gpu_selection_error
        self._environment_factory = environment_factory
        self._memory_sampler = memory_sampler
        self._current_group: str | None = None
        self._instances: list[_EnvironmentCallRecord] = []
        self._groups: list[_EvaluationGroupRecord] = []
        self._memory_samples: list[dict[str, Any]] = []
        self._numerical_failure_count = 0
        self._numerical_fields: Counter[str] = Counter()
        self._checked_transitions = 0
        self._first_create_sampled = False
        self._first_reset_sampled = False
        self._first_step_sampled = False
        self.sample_memory("before_evaluation")

    def sample_memory(self, phase: str) -> None:
        sample = dict(self._memory_sampler(self._device, phase, self._gpu_uuid))
        sample["phase"] = phase
        self._memory_samples.append(sample)

    def begin_group(self, label: str) -> None:
        if label not in EXECUTION_GROUPS:
            raise ValueError(f"unknown execution group {label!r}")
        if self._current_group is not None:
            raise RuntimeError("an execution group is already active")
        self._current_group = label

    def end_group(self, label: str, evaluation: ControllerEvaluation, wall_s: float) -> None:
        if self._current_group != label:
            raise RuntimeError("execution group lifecycle is inconsistent")
        self._current_group = None
        self._groups.append(
            _EvaluationGroupRecord(
                label=label,
                episode_count=len(evaluation.episodes),
                environment_steps=sum(episode.steps for episode in evaluation.episodes),
                wall_s=float(wall_s),
            )
        )
        self.sample_memory(f"after_{label.replace('.', '_')}")

    def create_environment(self, **kwargs: Any) -> _MeasuredEnvironment:
        if self._current_group is None:
            raise RuntimeError("environment creation requires an active execution group")
        started = time.perf_counter()
        environment = self._environment_factory(**kwargs)
        elapsed = time.perf_counter() - started
        record = _EnvironmentCallRecord(group=self._current_group, create_s=float(elapsed))
        self._instances.append(record)
        if not self._first_create_sampled:
            self._first_create_sampled = True
            self.sample_memory("after_first_environment_create")
        return _MeasuredEnvironment(environment, self, record)

    def record_reset(self, record: _EnvironmentCallRecord, elapsed_s: float) -> None:
        elapsed = float(elapsed_s)
        record.reset_count += 1
        record.reset_wall_s += elapsed
        if record.first_reset_s is None:
            record.first_reset_s = elapsed
        if not self._first_reset_sampled:
            self._first_reset_sampled = True
            self.sample_memory("after_first_reset")

    def record_step(
        self,
        record: _EnvironmentCallRecord,
        elapsed_s: float,
        output: tuple[Any, ...],
    ) -> None:
        elapsed = float(elapsed_s)
        record.step_count += 1
        record.step_wall_s += elapsed
        if record.first_step_s is None:
            record.first_step_s = elapsed
        self._checked_transitions += 1
        finite, failures = m4_benchmark._all_public_finite(output)
        if not finite:
            self._numerical_failure_count += 1
            self._numerical_fields.update(failures)
        if not self._first_step_sampled:
            self._first_step_sampled = True
            self.sample_memory("after_first_step")

    def finish_evaluation(self) -> None:
        if self._current_group is not None:
            raise RuntimeError("cannot finish execution evidence while a group is active")
        self.sample_memory("after_evaluation")

    @property
    def instances(self) -> tuple[_EnvironmentCallRecord, ...]:
        return tuple(self._instances)

    @property
    def groups(self) -> tuple[_EvaluationGroupRecord, ...]:
        return tuple(self._groups)

    @property
    def memory_samples(self) -> tuple[Mapping[str, Any], ...]:
        return tuple(self._memory_samples)

    @property
    def checked_transitions(self) -> int:
        return self._checked_transitions

    @property
    def numerical_failure_count(self) -> int:
        return self._numerical_failure_count

    @property
    def numerical_fields(self) -> Mapping[str, int]:
        return dict(sorted(self._numerical_fields.items()))


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


def _conda_package_version(package: str) -> str | None:
    """Read one installed Conda package version without persisting its environment path."""

    metadata_directory = Path(sys.prefix) / "conda-meta"
    try:
        candidates = sorted(metadata_directory.glob(f"{package}-*.json"))
    except OSError:
        return None
    for candidate in candidates:
        try:
            payload = json.loads(candidate.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            continue
        if payload.get("name") == package and isinstance(payload.get("version"), str):
            return str(payload["version"])
    return None


def _cpu_model() -> str | None:
    try:
        lines = Path("/proc/cpuinfo").read_text(encoding="utf-8").splitlines()
    except OSError:
        lines = []
    for line in lines:
        key, separator, value = line.partition(":")
        if separator and key.strip() in {"model name", "Hardware"} and value.strip():
            return value.strip()
    processor = platform.processor().strip()
    return processor or None


def _runtime_evidence() -> dict[str, Any]:
    """Collect versioned runtime and sanitized GPU identity evidence."""

    import casadi as ca
    import jax

    try:
        gpu_devices = jax.devices("gpu")
        gpu_error = None
    except RuntimeError as error:
        gpu_devices = []
        gpu_error = f"{type(error).__name__}: no JAX GPU device available"
    device = gpu_devices[0] if gpu_devices else None
    inventory, inventory_error = m4_benchmark._nvidia_inventory()
    selected_gpu: Mapping[str, Any] | None = None
    selection_error: str | None = "JAX GPU device unavailable"
    if device is not None:
        base_runtime, _gpu_uuid = m4_benchmark._runtime_evidence(
            device,
            inventory,
            inventory_error,
        )
        selected = base_runtime.get("selected_nvidia_gpu")
        selected_gpu = selected if isinstance(selected, Mapping) else None
        raw_selection_error = base_runtime.get("gpu_selection_error")
        selection_error = str(raw_selection_error) if raw_selection_error is not None else None
    packages = {
        package: _package_version(package)
        for package in (
            "casadi",
            "controller-learning",
            "jax",
            "jax-cuda12-plugin",
            "jaxlib",
            "mujoco",
            "mujoco-mjx",
            "nvidia-cuda-nvcc-cu12",
            "nvidia-cuda-runtime-cu12",
            "numpy",
            "warp-lang",
        )
    }
    packages["ipopt"] = _conda_package_version("ipopt")
    return {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "kernel": platform.release(),
        "machine": platform.machine(),
        "cpu": {
            "model": _cpu_model(),
            "logical_count": os.cpu_count(),
        },
        "packages": packages,
        "casadi_ipopt_available": bool(ca.has_nlpsol("ipopt")),
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
        "selected_nvidia_gpu": None if selected_gpu is None else dict(selected_gpu),
        "nvidia_smi_error": inventory_error,
        "gpu_selection_error": selection_error,
        "xla_python_client_preallocate": os.environ.get("XLA_PYTHON_CLIENT_PREALLOCATE"),
        "cuda_device_order": os.environ.get("CUDA_DEVICE_ORDER"),
        "cuda_visible_devices_configured": bool(os.environ.get("CUDA_VISIBLE_DEVICES")),
    }


def _strict_json_object(path: Path) -> Mapping[str, Any]:
    def reject_constant(value: str) -> None:
        raise ValueError(f"non-finite JSON constant {value!r}")

    def reject_duplicates(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in pairs:
            if key in result:
                raise ValueError(f"duplicate JSON key {key!r}")
            result[key] = value
        return result

    payload = json.loads(
        path.read_text(encoding="utf-8"),
        parse_constant=reject_constant,
        object_pairs_hook=reject_duplicates,
    )
    if not isinstance(payload, Mapping):
        raise ValueError(f"reviewed GPU evidence {path.name} must contain a JSON object")
    return payload


def _all_checks_pass(report: Mapping[str, Any]) -> bool:
    checks = report.get("checks")
    return (
        bool(checks)
        and isinstance(checks, list)
        and all(isinstance(check, Mapping) and check.get("passed") is True for check in checks)
    )


def _historical_gpu_evidence(project_root: Path) -> dict[str, Any]:
    """Bind M6 to reviewed physics/vector scaling reports without relabeling their metrics."""

    m2_path = project_root / M2_EVIDENCE_PATH
    m5_path = project_root / M5_EVIDENCE_PATH
    for path, expected_hash in (
        (m2_path, M2_EVIDENCE_SHA256),
        (m5_path, M5_EVIDENCE_SHA256),
    ):
        actual_hash = sha256_file(path)
        if actual_hash != expected_hash:
            raise RuntimeError(
                f"reviewed GPU evidence {path.name} has SHA-256 {actual_hash}, "
                f"expected {expected_hash}"
            )

    m2 = _strict_json_object(m2_path)
    m5 = _strict_json_object(m5_path)
    m2_scales = m2.get("scale_results")
    if not isinstance(m2_scales, list):
        raise RuntimeError("reviewed M2 report is missing scale_results")
    m2_1024 = next(
        (
            item
            for item in m2_scales
            if isinstance(item, Mapping) and item.get("num_worlds") == 1024
        ),
        None,
    )
    if not isinstance(m2_1024, Mapping):
        raise RuntimeError("reviewed M2 report is missing the 1,024-world result")

    m2_protocol = _mapping(m2.get("protocol"))
    m2_timing = _mapping(m2_1024.get("timing"))
    m2_memory = _mapping(m2_1024.get("memory"))
    m2_numerical = _mapping(m2_1024.get("numerical"))
    m2_failure_count = sum(
        (
            m2_numerical.get("finite") is not True,
            m2_numerical.get("time_monotonic") is not True,
            m2_numerical.get("contact_overflow") is not False,
            m2_numerical.get("constraint_overflow") is not False,
            m2_numerical.get("unexpected_contact") is not False,
        )
    )

    m5_protocol = _mapping(m5.get("protocol"))
    m5_timing = _mapping(m5.get("timing"))
    m5_memory = _mapping(m5.get("memory"))
    m5_numerical = _mapping(m5.get("numerical"))
    return {
        "scope": (
            "historical reviewed GPU backend/scaling evidence; these metrics are not M6 "
            "Controller throughput"
        ),
        "m2_physics": {
            "path": M2_EVIDENCE_PATH.as_posix(),
            "sha256": M2_EVIDENCE_SHA256,
            "schema_version": m2.get("schema_version"),
            "protocol_version": m2_protocol.get("protocol_version"),
            "status": m2.get("status"),
            "all_checks_passed": _all_checks_pass(m2),
            "num_worlds": m2_1024.get("num_worlds"),
            "environment_steps": m2_1024.get("environment_steps"),
            "compilation_s": m2_timing.get("compilation_s"),
            "transitions_per_second": m2_timing.get("transitions_per_second"),
            "peak_sampled_process_vram_mib": m2_memory.get("peak_process_vram_mib"),
            "numerical_failure_count": m2_failure_count,
        },
        "m5_vector_environment": {
            "path": M5_EVIDENCE_PATH.as_posix(),
            "sha256": M5_EVIDENCE_SHA256,
            "schema_version": m5.get("schema_version"),
            "protocol_version": m5.get("protocol_version"),
            "status": m5.get("status"),
            "all_checks_passed": _all_checks_pass(m5),
            "num_worlds": m5_protocol.get("num_worlds"),
            "environment_steps": m5_protocol.get("environment_steps"),
            "transitions": m5_protocol.get("transitions"),
            "first_step_compile_seconds": m5_timing.get("first_step_compile_seconds"),
            "transitions_per_second": m5_timing.get("transitions_per_second"),
            "peak_sampled_process_vram_mib": m5_memory.get("peak_sampled_process_vram_mib"),
            "numerical_failure_count": m5_numerical.get("failure_event_count"),
        },
    }


def _formal_execution_recorder() -> _ExecutionRecorder:
    import jax

    from controller_learning.envs import CarRacingEnv

    devices = jax.devices("gpu")
    if not devices:
        raise RuntimeError("JAX found no GPU device; use the Pixi gpu environment")
    device = devices[0]
    inventory, inventory_error = m4_benchmark._nvidia_inventory()
    selected, selection_error = m4_benchmark._selected_gpu(device, inventory)
    if inventory_error is not None and selection_error is None:
        selection_error = inventory_error
    gpu_uuid = None if selected is None else str(selected["uuid"])
    return _ExecutionRecorder(
        device=device,
        gpu_uuid=gpu_uuid,
        gpu_selection_error=selection_error,
        environment_factory=CarRacingEnv,
        memory_sampler=m4_benchmark._memory_sample,
    )


def _positive_rate(count: int, seconds: float) -> float:
    return count / seconds if seconds > 0.0 else 0.0


def _instance_payload(record: _EnvironmentCallRecord) -> dict[str, Any]:
    return {
        "group": record.group,
        "create_s": record.create_s,
        "reset_count": record.reset_count,
        "reset_wall_s": record.reset_wall_s,
        "first_reset_s": record.first_reset_s,
        "step_count": record.step_count,
        "step_wall_s": record.step_wall_s,
        "first_step_s": record.first_step_s,
        "closed": record.closed,
    }


def _execution_evidence(
    recorder: _ExecutionRecorder,
    evaluations: Mapping[str, Any],
    config: ProjectConfig,
) -> dict[str, Any]:
    instances = recorder.instances
    groups = recorder.groups
    if not instances:
        raise RuntimeError("formal M6 execution recorder observed no environments")
    if tuple(group.label for group in groups) != EXECUTION_GROUPS:
        raise RuntimeError("formal M6 execution recorder did not observe the fixed group order")

    environment_steps = sum(record.step_count for record in instances)
    environment_step_wall_s = sum(record.step_wall_s for record in instances)
    evaluation_wall_s = sum(group.wall_s for group in groups)
    episode_count = sum(group.episode_count for group in groups)
    physics_substeps = config.vehicle.simulation.physics_steps_per_control
    first = instances[0]
    if first.first_reset_s is None or first.first_step_s is None:
        raise RuntimeError("the first formal environment did not execute reset and step")

    group_payload = {
        group.label: {
            "episode_count": group.episode_count,
            "environment_steps": group.environment_steps,
            "wall_s": group.wall_s,
            "end_to_end_transitions_per_second": _positive_rate(
                group.environment_steps,
                group.wall_s,
            ),
        }
        for group in groups
    }
    samples = [dict(sample) for sample in recorder.memory_samples]
    process_values = [
        float(value)
        for sample in samples
        if isinstance((value := sample.get("process_vram_mib")), (int, float))
        and not isinstance(value, bool)
        and math.isfinite(float(value))
    ]
    allocator_peak_values = [
        float(value)
        for sample in samples
        if isinstance(sample.get("jax_allocator"), Mapping)
        and isinstance(
            (value := _mapping(sample.get("jax_allocator")).get("peak_bytes_in_use")),
            (int, float),
        )
        and not isinstance(value, bool)
        and math.isfinite(float(value))
    ]
    all_episodes = [
        episode
        for controller in ("pid", "mpc")
        for split in ("level0", "validation")
        for episode in _episodes({"evaluations": evaluations}, controller, split)
    ]
    invalid_action_count = sum(
        isinstance(episode, Mapping)
        and episode.get("termination_reason") == int(RaceTermination.INVALID_ACTION)
        for episode in all_episodes
    )
    return {
        "controller_evaluation": {
            "execution_model": CONTROLLER_EXECUTION_MODEL,
            "throughput_scope": THROUGHPUT_SCOPE,
            "num_envs_per_environment": FORMAL_ENVIRONMENTS_PER_EPISODE,
            "maximum_concurrent_worlds": FORMAL_ENVIRONMENTS_PER_EPISODE,
            "environment_instances": len(instances),
            "episode_count": episode_count,
            "environment_steps": environment_steps,
            "transitions": environment_steps,
            "physics_substeps_per_environment_step": physics_substeps,
            "world_physics_steps": environment_steps * physics_substeps,
            "per_step_host_synchronization": True,
            "evaluation_wall_s": evaluation_wall_s,
            "end_to_end_transitions_per_second": _positive_rate(
                environment_steps,
                evaluation_wall_s,
            ),
            "environment_step_call_wall_s": environment_step_wall_s,
            "environment_step_call_transitions_per_second": _positive_rate(
                environment_steps,
                environment_step_wall_s,
            ),
            "groups": group_payload,
            "instances": [_instance_payload(record) for record in instances],
        },
        "first_use_timing": {
            "method": FIRST_USE_TIMING_METHOD,
            "first_environment_create_and_backend_initialization_s": first.create_s,
            "first_reset_compile_and_execute_s": first.first_reset_s,
            "first_step_compile_and_execute_s": first.first_step_s,
            "combined_first_create_reset_step_s": (
                first.create_s + first.first_reset_s + first.first_step_s
            ),
        },
        "memory": {
            "method": MEMORY_SAMPLING_METHOD,
            "gpu_selection_error": recorder._gpu_selection_error,
            "required_phases": list(MEMORY_SAMPLE_PHASES),
            "sample_count": len(samples),
            "samples": samples,
            "peak_sampled_process_vram_mib": max(process_values, default=None),
            "peak_jax_allocator_bytes": max(allocator_peak_values, default=None),
        },
        "numerical": {
            "scope": [
                "all numeric public observation fields",
                "reward",
                "info.lap_time_s",
            ],
            "checked_transition_count": recorder.checked_transitions,
            "failure_event_count": recorder.numerical_failure_count,
            "failure_field_counts": dict(recorder.numerical_fields),
            "invalid_action_count": invalid_action_count,
            "internal_physics_diagnostics_claimed": False,
        },
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
    *,
    recorder: _ExecutionRecorder | None = None,
) -> dict[str, Any]:
    generator_version0 = assets.level0_manifest.generator_version
    validation_version = assets.validation_manifest.generator_version
    pid_validation = _first_rows(assets.validation_batch, FORMAL_PID_VALIDATION_TRACKS)
    reset0 = np.arange(FORMAL_LEVEL0_TRACKS, dtype=np.uint32)
    pid_reset = np.arange(FORMAL_PID_VALIDATION_TRACKS, dtype=np.uint32)
    mpc_reset = np.arange(FORMAL_MPC_VALIDATION_TRACKS, dtype=np.uint32)
    pid_directory = project_root / "controllers/pid"
    mpc_directory = project_root / "controllers/mpc"

    def run_group(
        label: str,
        level_id: int,
        batch: TrackBatch,
        generator_version: str,
        directory: Path,
        reset_seeds: np.ndarray,
    ) -> ControllerEvaluation:
        kwargs: dict[str, Any] = {"reset_seeds": reset_seeds}
        if recorder is not None:
            recorder.begin_group(label)
            kwargs["env_factory"] = recorder.create_environment
        started = time.perf_counter()
        evaluation = evaluator(
            config,
            level_id,
            batch,
            generator_version,
            directory,
            FORMAL_BACKEND,
            **kwargs,
        )
        elapsed = time.perf_counter() - started
        if recorder is not None:
            recorder.end_group(label, evaluation, elapsed)
        return evaluation

    pid_level0 = run_group(
        "pid.level0",
        0,
        assets.level0_batch,
        generator_version0,
        pid_directory,
        reset0,
    )
    pid_validation_result = run_group(
        "pid.validation",
        1,
        pid_validation,
        validation_version,
        pid_directory,
        pid_reset,
    )
    mpc_level0 = run_group(
        "mpc.level0",
        0,
        assets.level0_batch,
        generator_version0,
        mpc_directory,
        reset0,
    )
    mpc_validation_result = run_group(
        "mpc.validation",
        1,
        assets.validation_batch,
        validation_version,
        mpc_directory,
        mpc_reset,
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


def _execution_group_results(report: Mapping[str, Any]) -> dict[str, Mapping[str, Any]]:
    return {
        f"{controller}.{split}": _evaluation_result(report, controller, split)
        for controller in ("pid", "mpc")
        for split in ("level0", "validation")
    }


def _execution_count_findings(report: Mapping[str, Any]) -> list[str]:
    findings: list[str] = []
    execution = _mapping(report.get("execution"))
    evaluation = _mapping(execution.get("controller_evaluation"))
    instances = evaluation.get("instances")
    groups = evaluation.get("groups")
    if not isinstance(instances, list):
        return ["controller_evaluation.instances"]
    if not isinstance(groups, Mapping):
        return ["controller_evaluation.groups"]

    expected_results = _execution_group_results(report)
    expected_episode_count = sum(
        len(_episodes(report, *label.split("."))) for label in EXECUTION_GROUPS
    )
    expected_steps = sum(
        int(episode["steps"])
        for label in EXECUTION_GROUPS
        for episode in _episodes(report, *label.split("."))
        if isinstance(episode, Mapping) and type(episode.get("steps")) is int
    )
    instance_counts: Counter[str] = Counter()
    instance_steps: Counter[str] = Counter()
    total_step_wall_s = 0.0
    valid_instances = True
    for index, instance in enumerate(instances):
        if not isinstance(instance, Mapping):
            findings.append(f"controller_evaluation.instances[{index}]")
            valid_instances = False
            continue
        group = instance.get("group")
        step_count = instance.get("step_count")
        valid = (
            group in EXECUTION_GROUPS
            and type(step_count) is int
            and step_count > 0
            and type(instance.get("reset_count")) is int
            and instance.get("reset_count") == 1
            and instance.get("closed") is True
            and all(
                _finite_number(instance.get(field_name), minimum=np.finfo(np.float64).tiny)
                for field_name in (
                    "create_s",
                    "reset_wall_s",
                    "first_reset_s",
                    "step_wall_s",
                    "first_step_s",
                )
            )
            and float(instance["reset_wall_s"]) >= float(instance["first_reset_s"])
            and float(instance["step_wall_s"]) >= float(instance["first_step_s"])
        )
        if not valid:
            findings.append(f"controller_evaluation.instances[{index}]")
            valid_instances = False
            continue
        label = str(group)
        instance_counts[label] += 1
        instance_steps[label] += int(step_count)
        total_step_wall_s += float(instance["step_wall_s"])

    group_wall_total = 0.0
    for label in EXECUTION_GROUPS:
        group = _mapping(groups.get(label))
        result = expected_results[label]
        episodes = result.get("episodes")
        expected_group_episodes = len(episodes) if isinstance(episodes, list) else 0
        expected_group_steps = sum(
            int(episode["steps"])
            for episode in episodes or []
            if isinstance(episode, Mapping) and type(episode.get("steps")) is int
        )
        wall_s = group.get("wall_s")
        if (
            group.get("episode_count") != expected_group_episodes
            or type(group.get("episode_count")) is not int
            or group.get("environment_steps") != expected_group_steps
            or type(group.get("environment_steps")) is not int
            or instance_counts[label] != expected_group_episodes
            or instance_steps[label] != expected_group_steps
            or not _finite_number(wall_s, minimum=np.finfo(np.float64).tiny)
            or not _close_number(
                group.get("end_to_end_transitions_per_second"),
                _positive_rate(expected_group_steps, float(wall_s or 0.0)),
                tolerance=1.0e-12,
            )
        ):
            findings.append(f"controller_evaluation.groups.{label}")
        if _finite_number(wall_s):
            group_wall_total += float(wall_s)

    physics_substeps = FORMAL_PHYSICS_SUBSTEPS_PER_ENVIRONMENT_STEP
    if (
        evaluation.get("execution_model") != CONTROLLER_EXECUTION_MODEL
        or evaluation.get("throughput_scope") != THROUGHPUT_SCOPE
        or evaluation.get("num_envs_per_environment") != FORMAL_ENVIRONMENTS_PER_EPISODE
        or type(evaluation.get("num_envs_per_environment")) is not int
        or evaluation.get("maximum_concurrent_worlds") != FORMAL_ENVIRONMENTS_PER_EPISODE
        or type(evaluation.get("maximum_concurrent_worlds")) is not int
        or evaluation.get("environment_instances") != len(instances)
        or type(evaluation.get("environment_instances")) is not int
        or len(instances) != FORMAL_EPISODE_COUNT
        or evaluation.get("episode_count") != expected_episode_count
        or type(evaluation.get("episode_count")) is not int
        or expected_episode_count != FORMAL_EPISODE_COUNT
        or evaluation.get("environment_steps") != expected_steps
        or type(evaluation.get("environment_steps")) is not int
        or evaluation.get("transitions") != expected_steps
        or type(evaluation.get("transitions")) is not int
        or evaluation.get("physics_substeps_per_environment_step") != physics_substeps
        or type(evaluation.get("physics_substeps_per_environment_step")) is not int
        or evaluation.get("world_physics_steps") != expected_steps * physics_substeps
        or type(evaluation.get("world_physics_steps")) is not int
        or evaluation.get("per_step_host_synchronization") is not True
        or not _close_number(evaluation.get("evaluation_wall_s"), group_wall_total, tolerance=1e-12)
        or not _close_number(
            evaluation.get("end_to_end_transitions_per_second"),
            _positive_rate(expected_steps, group_wall_total),
            tolerance=1.0e-12,
        )
        or not _close_number(
            evaluation.get("environment_step_call_wall_s"),
            total_step_wall_s,
            tolerance=1.0e-12,
        )
        or not _close_number(
            evaluation.get("environment_step_call_transitions_per_second"),
            _positive_rate(expected_steps, total_step_wall_s),
            tolerance=1.0e-12,
        )
        or not valid_instances
        or set(groups) != set(EXECUTION_GROUPS)
    ):
        findings.append("controller_evaluation.aggregate")
    return findings


def _first_use_timing_findings(report: Mapping[str, Any]) -> list[str]:
    execution = _mapping(report.get("execution"))
    first_use = _mapping(execution.get("first_use_timing"))
    evaluation = _mapping(execution.get("controller_evaluation"))
    instances = evaluation.get("instances")
    if not isinstance(instances, list) or not instances or not isinstance(instances[0], Mapping):
        return ["first_use_timing.first_instance"]
    first = instances[0]
    create = first.get("create_s")
    reset = first.get("first_reset_s")
    step = first.get("first_step_s")
    if not all(
        _finite_number(value, minimum=np.finfo(np.float64).tiny) for value in (create, reset, step)
    ):
        return ["first_use_timing.first_instance"]
    findings: list[str] = []
    if first_use.get("method") != FIRST_USE_TIMING_METHOD:
        findings.append("first_use_timing.method")
    expected = {
        "first_environment_create_and_backend_initialization_s": float(create),
        "first_reset_compile_and_execute_s": float(reset),
        "first_step_compile_and_execute_s": float(step),
        "combined_first_create_reset_step_s": float(create) + float(reset) + float(step),
    }
    for field_name, value in expected.items():
        if not _close_number(first_use.get(field_name), value, tolerance=1.0e-12):
            findings.append(f"first_use_timing.{field_name}")
    return findings


def _memory_findings(report: Mapping[str, Any]) -> list[str]:
    memory = _mapping(_mapping(report.get("execution")).get("memory"))
    samples = memory.get("samples")
    if not isinstance(samples, list):
        return ["memory.samples"]
    phases = [sample.get("phase") if isinstance(sample, Mapping) else None for sample in samples]
    findings: list[str] = []
    if (
        memory.get("method") != MEMORY_SAMPLING_METHOD
        or memory.get("required_phases") != list(MEMORY_SAMPLE_PHASES)
        or phases != list(MEMORY_SAMPLE_PHASES)
        or memory.get("sample_count") != len(samples)
        or type(memory.get("sample_count")) is not int
        or memory.get("gpu_selection_error") is not None
    ):
        findings.append("memory.coverage")

    process_values: list[float] = []
    allocator_values: list[float] = []
    for index, sample in enumerate(samples):
        if not isinstance(sample, Mapping):
            findings.append(f"memory.samples[{index}]")
            continue
        process = sample.get("process_vram_mib")
        allocator = _mapping(sample.get("jax_allocator"))
        allocator_peak = allocator.get("peak_bytes_in_use")
        if _finite_number(process):
            process_values.append(float(process))
        if _finite_number(allocator_peak):
            allocator_values.append(float(allocator_peak))
        if index > 0 and (
            not _finite_number(process, minimum=np.finfo(np.float64).tiny)
            or sample.get("process_vram_error") is not None
            or not _finite_number(allocator_peak, minimum=np.finfo(np.float64).tiny)
            or sample.get("jax_allocator_error") is not None
        ):
            findings.append(f"memory.samples[{index}]")
    expected_process_peak = max(process_values, default=None)
    expected_allocator_peak = max(allocator_values, default=None)
    if expected_process_peak is None or not _close_number(
        memory.get("peak_sampled_process_vram_mib"), expected_process_peak
    ):
        findings.append("memory.peak_sampled_process_vram_mib")
    if expected_allocator_peak is None or not _close_number(
        memory.get("peak_jax_allocator_bytes"), expected_allocator_peak
    ):
        findings.append("memory.peak_jax_allocator_bytes")
    return findings


def _execution_numerical_findings(report: Mapping[str, Any]) -> list[str]:
    numerical = _mapping(_mapping(report.get("execution")).get("numerical"))
    evaluation = _mapping(_mapping(report.get("execution")).get("controller_evaluation"))
    expected_steps = evaluation.get("environment_steps")
    valid = (
        numerical.get("scope")
        == [
            "all numeric public observation fields",
            "reward",
            "info.lap_time_s",
        ]
        and type(numerical.get("checked_transition_count")) is int
        and numerical.get("checked_transition_count") == expected_steps
        and numerical.get("failure_event_count") == 0
        and type(numerical.get("failure_event_count")) is int
        and numerical.get("failure_field_counts") == {}
        and numerical.get("invalid_action_count") == 0
        and type(numerical.get("invalid_action_count")) is int
        and numerical.get("internal_physics_diagnostics_claimed") is False
    )
    return [] if valid else ["execution.numerical"]


def _runtime_findings(report: Mapping[str, Any]) -> list[str]:
    runtime = _mapping(report.get("runtime"))
    packages = _mapping(runtime.get("packages"))
    cpu = _mapping(runtime.get("cpu"))
    selected_gpu = _mapping(runtime.get("selected_nvidia_gpu"))
    required_packages = (
        "casadi",
        "controller-learning",
        "ipopt",
        "jax",
        "jax-cuda12-plugin",
        "jaxlib",
        "mujoco",
        "mujoco-mjx",
        "nvidia-cuda-nvcc-cu12",
        "nvidia-cuda-runtime-cu12",
        "numpy",
        "warp-lang",
    )
    valid = (
        all(
            isinstance(packages.get(name), str) and bool(packages[name])
            for name in required_packages
        )
        and runtime.get("casadi_ipopt_available") is True
        and isinstance(cpu.get("model"), str)
        and bool(cpu.get("model"))
        and type(cpu.get("logical_count")) is int
        and int(cpu["logical_count"]) > 0
        and isinstance(runtime.get("python"), str)
        and bool(runtime.get("python"))
        and isinstance(runtime.get("platform"), str)
        and bool(runtime.get("platform"))
        and isinstance(runtime.get("kernel"), str)
        and bool(runtime.get("kernel"))
        and isinstance(runtime.get("machine"), str)
        and bool(runtime.get("machine"))
        and isinstance(selected_gpu.get("name"), str)
        and bool(selected_gpu.get("name"))
        and isinstance(selected_gpu.get("driver_version"), str)
        and bool(selected_gpu.get("driver_version"))
        and _finite_number(selected_gpu.get("memory_total_mib"), minimum=np.finfo(np.float64).tiny)
        and runtime.get("nvidia_smi_error") is None
        and runtime.get("gpu_selection_error") is None
        and runtime.get("xla_python_client_preallocate") == "false"
        and runtime.get("cuda_device_order") == "PCI_BUS_ID"
        and type(runtime.get("cuda_visible_devices_configured")) is bool
    )
    return [] if valid else ["runtime.hardware_software_versions"]


def _historical_evidence_findings(report: Mapping[str, Any]) -> list[str]:
    evidence = _mapping(report.get("historical_gpu_evidence"))
    m2 = _mapping(evidence.get("m2_physics"))
    m5 = _mapping(evidence.get("m5_vector_environment"))
    source = _mapping(report.get("source_evidence"))
    before_hashes = _mapping(_mapping(source.get("before")).get("source_files_sha256"))
    after_hashes = _mapping(_mapping(source.get("after")).get("source_files_sha256"))
    valid = (
        evidence.get("scope")
        == (
            "historical reviewed GPU backend/scaling evidence; these metrics are not M6 "
            "Controller throughput"
        )
        and m2.get("path") == M2_EVIDENCE_PATH.as_posix()
        and m2.get("sha256") == M2_EVIDENCE_SHA256
        and before_hashes.get(M2_EVIDENCE_PATH.as_posix()) == M2_EVIDENCE_SHA256
        and after_hashes.get(M2_EVIDENCE_PATH.as_posix()) == M2_EVIDENCE_SHA256
        and m2.get("schema_version") == 1
        and m2.get("protocol_version") == "m2-mjx-warp-v1"
        and m2.get("status") == "pass"
        and m2.get("all_checks_passed") is True
        and m2.get("num_worlds") == 1024
        and m2.get("environment_steps") == 10_000
        and _close_number(
            m2.get("compilation_s"),
            M2_EXPECTED_COMPILATION_S,
            tolerance=1.0e-12,
        )
        and _close_number(
            m2.get("transitions_per_second"),
            M2_EXPECTED_TRANSITIONS_PER_SECOND,
            tolerance=1.0e-9,
        )
        and _close_number(
            m2.get("peak_sampled_process_vram_mib"),
            M2_EXPECTED_PEAK_PROCESS_VRAM_MIB,
            tolerance=1.0e-12,
        )
        and m2.get("numerical_failure_count") == 0
        and m5.get("path") == M5_EVIDENCE_PATH.as_posix()
        and m5.get("sha256") == M5_EVIDENCE_SHA256
        and before_hashes.get(M5_EVIDENCE_PATH.as_posix()) == M5_EVIDENCE_SHA256
        and after_hashes.get(M5_EVIDENCE_PATH.as_posix()) == M5_EVIDENCE_SHA256
        and m5.get("schema_version") == "controller-learning.m5-track-pool.v2"
        and m5.get("protocol_version") == "m5-track-pool-gpu-v2"
        and m5.get("status") == "pass"
        and m5.get("all_checks_passed") is True
        and m5.get("num_worlds") == 1024
        and m5.get("environment_steps") == 10_000
        and m5.get("transitions") == 10_240_000
        and _close_number(
            m5.get("first_step_compile_seconds"),
            M5_EXPECTED_FIRST_STEP_COMPILE_S,
            tolerance=1.0e-12,
        )
        and _close_number(
            m5.get("transitions_per_second"),
            M5_EXPECTED_TRANSITIONS_PER_SECOND,
            tolerance=1.0e-9,
        )
        and _close_number(
            m5.get("peak_sampled_process_vram_mib"),
            M5_EXPECTED_PEAK_PROCESS_VRAM_MIB,
            tolerance=1.0e-12,
        )
        and m5.get("numerical_failure_count") == 0
    )
    return [] if valid else ["historical_gpu_evidence"]


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
            absolute_paths.extend(
                match.group(1) for match in _POSIX_ABSOLUTE_PATTERN.finditer(item)
            )
            absolute_paths.extend(
                match.group(1) for match in _WINDOWS_ABSOLUTE_PATTERN.finditer(item)
            )
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
            and protocol.get("realtime_qualification_required_for_m6_pass") is False
            and protocol.get("num_envs_per_environment") == FORMAL_ENVIRONMENTS_PER_EPISODE
            and protocol.get("maximum_concurrent_worlds") == FORMAL_ENVIRONMENTS_PER_EPISODE
            and protocol.get("physics_substeps_per_environment_step")
            == FORMAL_PHYSICS_SUBSTEPS_PER_ENVIRONMENT_STEP
            and protocol.get("controller_execution_model") == CONTROLLER_EXECUTION_MODEL
            and protocol.get("throughput_scope") == THROUGHPUT_SCOPE,
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
                "num_envs_per_environment": FORMAL_ENVIRONMENTS_PER_EPISODE,
                "maximum_concurrent_worlds": FORMAL_ENVIRONMENTS_PER_EPISODE,
                "physics_substeps_per_environment_step": (
                    FORMAL_PHYSICS_SUBSTEPS_PER_ENVIRONMENT_STEP
                ),
                "controller_execution_model": CONTROLLER_EXECUTION_MODEL,
                "throughput_scope": THROUGHPUT_SCOPE,
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
    execution_count_findings = _execution_count_findings(report)
    first_use_findings = _first_use_timing_findings(report)
    memory_findings = _memory_findings(report)
    execution_numerical_findings = _execution_numerical_findings(report)
    runtime_findings = _runtime_findings(report)
    historical_findings = _historical_evidence_findings(report)
    checks.extend(
        (
            _check(
                "runtime.gpu",
                device.get("platform") == "gpu" and runtime.get("jax_gpu_error") is None,
                {"jax_device": device, "jax_gpu_error": runtime.get("jax_gpu_error")},
                {"platform": "gpu", "jax_gpu_error": None},
            ),
            _check(
                "runtime.hardware_software_versions",
                not runtime_findings,
                runtime_findings,
                [],
            ),
            _check(
                "execution.counts_and_throughput",
                not execution_count_findings,
                execution_count_findings,
                [],
            ),
            _check(
                "execution.first_use_timing",
                not first_use_findings,
                first_use_findings,
                [],
            ),
            _check(
                "execution.memory",
                not memory_findings,
                memory_findings,
                [],
            ),
            _check(
                "execution.public_numerical_health",
                not execution_numerical_findings,
                execution_numerical_findings,
                [],
            ),
            _check(
                "evidence.historical_gpu_reports",
                not historical_findings,
                historical_findings,
                [],
            ),
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
    execution_evidence_loader: ExecutionEvidenceLoader | None = None,
    historical_evidence_loader: HistoricalEvidenceLoader | None = None,
) -> dict[str, Any]:
    """Execute the locked formal workload and return a strict-JSON-compatible report."""

    if not isinstance(options, BenchmarkOptions):
        raise TypeError("options must be BenchmarkOptions")
    root = Path(project_root).expanduser().resolve()
    load_assets = _load_evaluation_assets if asset_loader is None else asset_loader
    run_evaluation = evaluate_track_batch if evaluator is None else evaluator
    take_snapshot = _source_snapshot if snapshot_loader is None else snapshot_loader
    load_runtime = _runtime_evidence if runtime_loader is None else runtime_loader
    load_historical = (
        _historical_gpu_evidence
        if historical_evidence_loader is None
        else historical_evidence_loader
    )

    before = dict(take_snapshot(root))
    _require_source_preflight(before)
    config = load_project_config(root)
    if config.benchmark.version != "0.1":
        raise RuntimeError("formal M6 evaluation is locked to benchmark version 0.1")
    if config.benchmark.validation_track_count != FORMAL_MPC_VALIDATION_TRACKS:
        raise RuntimeError("formal M6 evaluation requires exactly 100 Validation Tracks")
    if config.benchmark.controller.init_timeout_s != FORMAL_INIT_TIMEOUT_S:
        raise RuntimeError("formal M6 evaluation requires the fixed 30 second init timeout")
    if (
        config.vehicle.simulation.physics_steps_per_control
        != FORMAL_PHYSICS_SUBSTEPS_PER_ENVIRONMENT_STEP
    ):
        raise RuntimeError("formal M6 evaluation requires ten physics substeps per control step")
    assets = load_assets(config, root)
    if int(assets.level0_batch.seed.shape[0]) != FORMAL_LEVEL0_TRACKS:
        raise RuntimeError("formal Level 0 asset must contain exactly one Track")
    if int(assets.validation_batch.seed.shape[0]) != FORMAL_MPC_VALIDATION_TRACKS:
        raise RuntimeError("formal Validation asset must contain exactly 100 Tracks")

    historical_gpu_evidence = dict(load_historical(root))
    runtime = dict(load_runtime())
    controller_configs = {name: _controller_config_evidence(root, name) for name in ("pid", "mpc")}
    if execution_evidence_loader is None:
        recorder = _formal_execution_recorder()
        evaluations = _run_controller_evaluations(
            config,
            assets,
            root,
            run_evaluation,
            recorder=recorder,
        )
        recorder.finish_evaluation()
        execution = _execution_evidence(recorder, evaluations, config)
    else:
        evaluations = _run_controller_evaluations(
            config,
            assets,
            root,
            run_evaluation,
        )
        execution = dict(execution_evidence_loader(evaluations, config))
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
            "num_envs_per_environment": FORMAL_ENVIRONMENTS_PER_EPISODE,
            "maximum_concurrent_worlds": FORMAL_ENVIRONMENTS_PER_EPISODE,
            "physics_substeps_per_environment_step": (
                config.vehicle.simulation.physics_steps_per_control
            ),
            "controller_execution_model": CONTROLLER_EXECUTION_MODEL,
            "throughput_scope": THROUGHPUT_SCOPE,
        },
        "assets": _json_value(assets.evidence),
        "controller_configs": controller_configs,
        "evaluations": evaluations,
        "execution": execution,
        "historical_gpu_evidence": historical_gpu_evidence,
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
