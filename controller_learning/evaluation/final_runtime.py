"""Measured runtime, lifecycle, and memory evidence for the M8 final evaluation.

The helpers in this module are deliberately independent from benchmark assets and Controller
execution.  They measure one caller-owned environment, the selected GPU process, and the local
runtime without exposing machine paths or the physical GPU UUID in public evidence.
"""

from __future__ import annotations

import math
import os
import platform
import re
import shutil
import stat
import subprocess
import time
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from importlib.metadata import version as installed_package_version
from numbers import Integral, Real
from pathlib import Path
from typing import Any, Final, TypeAlias

FINAL_ENVIRONMENT_LIFECYCLE_SCHEMA_VERSION: Final = 1
FINAL_MEMORY_EVIDENCE_SCHEMA_VERSION: Final = 1
FINAL_RUNTIME_EVIDENCE_SCHEMA_VERSION: Final = 1
FINAL_RUNTIME_PACKAGE_NAMES: Final = (
    "controller-learning",
    "jax",
    "jaxlib",
    "matplotlib",
    "mujoco",
    "mujoco-mjx",
    "numpy",
    "torch",
    "warp-lang",
)

CommandRunner: TypeAlias = Callable[[Sequence[str]], str]
PackageVersionLoader: TypeAlias = Callable[[str], str]

_GPU_UUID = re.compile(
    r"GPU-[0-9a-fA-F]{8}(?:-[0-9a-fA-F]{4}){3}-[0-9a-fA-F]{12}",
    flags=re.IGNORECASE,
)
_POSIX_ABSOLUTE_PATH = re.compile(r"(?:^|[\s\"'=<(])/(?!/)[^\s\"'<>]*")
_WINDOWS_ABSOLUTE_PATH = re.compile(r"(?:^|[\s\"'=<(])[A-Za-z]:[\\/][^\s\"'<>]*")
_FILE_URI = re.compile(r"file:(?://[^/\s\"'<>]*)?/[^\s\"'<>]*", flags=re.IGNORECASE)
_PYTHON_311 = re.compile(r"3\.11(?:\.[0-9]+)?")
_NVIDIA_SMI_ENVIRONMENT_ITEMS: Final = (
    ("HOME", "/nonexistent"),
    ("LANG", "C"),
    ("LC_ALL", "C"),
)


class FinalRuntimeEvidenceError(ValueError):
    """Runtime or memory evidence is incomplete, invalid, or unsafe to publish."""


def _plain_nonnegative_integer(value: object, *, field: str) -> int:
    if isinstance(value, bool) or not isinstance(value, Integral):
        raise FinalRuntimeEvidenceError(f"{field} must be an integer")
    result = int(value)
    if result < 0:
        raise FinalRuntimeEvidenceError(f"{field} must be non-negative")
    return result


def _finite_number(value: object, *, field: str, positive: bool = False) -> float:
    if isinstance(value, bool) or not isinstance(value, Real):
        raise FinalRuntimeEvidenceError(f"{field} must be a real number")
    result = float(value)
    if not math.isfinite(result) or result < 0.0 or (positive and result <= 0.0):
        qualifier = "positive" if positive else "non-negative"
        raise FinalRuntimeEvidenceError(f"{field} must be finite and {qualifier}")
    return result


def _safe_nonempty_string(value: object, *, field: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise FinalRuntimeEvidenceError(f"{field} must be a non-empty string")
    result = value.strip()
    if (
        _GPU_UUID.search(result) is not None
        or _POSIX_ABSOLUTE_PATH.search(result) is not None
        or _WINDOWS_ABSOLUTE_PATH.search(result) is not None
        or _FILE_URI.search(result) is not None
        or any(not character.isprintable() for character in result)
    ):
        raise FinalRuntimeEvidenceError(f"{field} contains private or unsafe text")
    return result


def _private_safe_value(value: object) -> bool:
    if value is None or isinstance(value, bool):
        return True
    if isinstance(value, Integral):
        return True
    if isinstance(value, Real):
        return math.isfinite(float(value))
    if isinstance(value, str):
        return bool(value) and (
            _GPU_UUID.search(value) is None
            and _POSIX_ABSOLUTE_PATH.search(value) is None
            and _WINDOWS_ABSOLUTE_PATH.search(value) is None
            and _FILE_URI.search(value) is None
            and all(character.isprintable() for character in value)
        )
    if isinstance(value, Mapping):
        return all(
            isinstance(key, str) and _private_safe_value(key) and _private_safe_value(item)
            for key, item in value.items()
        )
    if isinstance(value, (tuple, list)):
        return all(_private_safe_value(item) for item in value)
    return False


def runtime_evidence_is_private_safe(value: object) -> bool:
    """Return whether evidence contains only finite JSON-like values and no paths or GPU UUIDs."""

    return _private_safe_value(value)


def _require_exact_keys(value: Mapping[str, Any], expected: set[str], *, field: str) -> None:
    if set(value) != expected:
        raise FinalRuntimeEvidenceError(f"{field} must contain exactly {sorted(expected)!r}")


@dataclass(slots=True)
class _EnvironmentMeasurement:
    create_wall_time_s: float
    reset_count: int = 0
    step_count: int = 0
    close_count: int = 0
    first_reset_wall_time_s: float | None = None
    first_step_wall_time_s: float | None = None
    closed: bool = False


class _MeasuredFinalEnvironment:
    """Narrow Runner-compatible proxy around the single formal environment."""

    def __init__(self, environment: Any, measurement: _EnvironmentMeasurement) -> None:
        self._environment = environment
        self._measurement = measurement

    def _require_open(self) -> None:
        if self._measurement.close_count:
            raise RuntimeError("the measured final environment is already closed")

    @property
    def unwrapped(self) -> Any:
        self._require_open()
        return self._environment.unwrapped

    def reset(self, *args: Any, **kwargs: Any) -> Any:
        self._require_open()
        started = time.perf_counter()
        self._measurement.reset_count += 1
        try:
            return self._environment.reset(*args, **kwargs)
        finally:
            if self._measurement.first_reset_wall_time_s is None:
                self._measurement.first_reset_wall_time_s = time.perf_counter() - started

    def step(self, action: object) -> Any:
        self._require_open()
        started = time.perf_counter()
        self._measurement.step_count += 1
        try:
            return self._environment.step(action)
        finally:
            if self._measurement.first_step_wall_time_s is None:
                self._measurement.first_step_wall_time_s = time.perf_counter() - started

    def render(self, *args: Any, **kwargs: Any) -> Any:
        self._require_open()
        return self._environment.render(*args, **kwargs)

    def close(self) -> None:
        if self._measurement.close_count:
            raise RuntimeError("the measured final environment may be closed exactly once")
        self._measurement.close_count += 1
        self._environment.close()
        self._measurement.closed = True


class MeasuredFinalEnvironmentFactory:
    """Construct and measure exactly one environment for all 60 final episodes."""

    def __init__(self, factory: Callable[..., Any]) -> None:
        if not callable(factory):
            raise TypeError("factory must be callable")
        self._factory = factory
        self._create_attempted = False
        self._measurement: _EnvironmentMeasurement | None = None

    def create(self, **kwargs: Any) -> _MeasuredFinalEnvironment:
        """Create the sole environment; a failed or successful second attempt is forbidden."""

        if self._create_attempted:
            raise RuntimeError("the final evaluation may construct exactly one environment")
        self._create_attempted = True
        started = time.perf_counter()
        environment = self._factory(**kwargs)
        elapsed = time.perf_counter() - started
        self._measurement = _EnvironmentMeasurement(create_wall_time_s=elapsed)
        return _MeasuredFinalEnvironment(environment, self._measurement)

    def evidence(self, *, expected_resets: int, expected_steps: int) -> dict[str, Any]:
        """Return strict lifecycle evidence after the sole environment has closed."""

        expected_reset_count = _plain_nonnegative_integer(expected_resets, field="expected_resets")
        expected_step_count = _plain_nonnegative_integer(expected_steps, field="expected_steps")
        measurement = self._measurement
        if measurement is None:
            raise RuntimeError("the measured final environment has not been created")
        if not measurement.closed:
            raise RuntimeError("environment lifecycle evidence is available only after close")
        if measurement.reset_count != expected_reset_count:
            raise RuntimeError("environment reset count differs from the frozen workload")
        if measurement.step_count != expected_step_count:
            raise RuntimeError("environment step count differs from the completed workload")
        if measurement.close_count != 1:
            raise RuntimeError("the final environment must be closed exactly once")
        if measurement.first_reset_wall_time_s is None:
            raise RuntimeError("the first environment reset was not measured")
        if measurement.first_step_wall_time_s is None:
            raise RuntimeError("the first environment step was not measured")
        evidence = {
            "schema_version": FINAL_ENVIRONMENT_LIFECYCLE_SCHEMA_VERSION,
            "environment_instance_count": 1,
            "environment_create_wall_time_s": _finite_number(
                measurement.create_wall_time_s,
                field="environment_create_wall_time_s",
            ),
            "first_reset_wall_time_including_lazy_compilation_s": _finite_number(
                measurement.first_reset_wall_time_s,
                field="first_reset_wall_time_including_lazy_compilation_s",
            ),
            "first_step_wall_time_including_lazy_compilation_s": _finite_number(
                measurement.first_step_wall_time_s,
                field="first_step_wall_time_including_lazy_compilation_s",
            ),
            "reset_count": measurement.reset_count,
            "expected_reset_count": expected_reset_count,
            "step_count": measurement.step_count,
            "expected_step_count": expected_step_count,
            "close_count": measurement.close_count,
            "method": (
                "wall clock around environment construction and the first public reset and step; "
                "the first-call timings include any lazy compilation"
            ),
        }
        if not runtime_evidence_is_private_safe(evidence):
            raise FinalRuntimeEvidenceError("environment lifecycle evidence is unsafe to publish")
        return evidence


def _default_command_runner(command: Sequence[str]) -> str:
    completed = subprocess.run(
        tuple(command),
        stdin=subprocess.DEVNULL,
        capture_output=True,
        cwd=None,
        env=dict(_NVIDIA_SMI_ENVIRONMENT_ITEMS),
        check=True,
        close_fds=True,
        shell=False,
        start_new_session=True,
        text=True,
        timeout=15,
    )
    return completed.stdout


def resolve_nvidia_smi_executable(executable: str | os.PathLike[str] = "nvidia-smi") -> str:
    """Return one absolute, non-symlink regular executable without publishing its path."""

    supplied = os.fspath(executable)
    if not isinstance(supplied, str) or not supplied:
        raise TypeError("nvidia-smi executable must be a non-empty filesystem path")
    located = shutil.which(supplied)
    if located is None:
        raise RuntimeError("nvidia-smi executable is unavailable")
    try:
        resolved = Path(located).resolve(strict=True)
        metadata = resolved.lstat()
    except (FileNotFoundError, OSError) as error:
        raise RuntimeError("nvidia-smi executable is unavailable") from error
    if (
        not resolved.is_absolute()
        or stat.S_ISLNK(metadata.st_mode)
        or not stat.S_ISREG(metadata.st_mode)
    ):
        raise RuntimeError("nvidia-smi executable must resolve to an absolute regular file")
    if metadata.st_mode & 0o111 == 0 or not os.access(resolved, os.X_OK):
        raise RuntimeError("nvidia-smi executable is not executable")
    return str(resolved)


def _validated_nvidia_smi_executable(value: str | os.PathLike[str]) -> str:
    supplied = os.fspath(value)
    if not isinstance(supplied, str) or not supplied:
        raise TypeError("nvidia_smi_executable must be a non-empty filesystem path")
    path = Path(supplied)
    if not path.is_absolute():
        raise ValueError("nvidia_smi_executable must be an absolute pre-resolved path")
    try:
        resolved = path.resolve(strict=True)
        metadata = path.lstat()
    except (FileNotFoundError, OSError) as error:
        raise ValueError("nvidia_smi_executable is unavailable") from error
    if resolved != path or stat.S_ISLNK(metadata.st_mode) or not stat.S_ISREG(metadata.st_mode):
        raise ValueError("nvidia_smi_executable must be a non-symlink regular file")
    if metadata.st_mode & 0o111 == 0:
        raise ValueError("nvidia_smi_executable must have an executable mode")
    return str(path)


def _validated_gpu_uuid(value: object) -> str:
    if not isinstance(value, str) or _GPU_UUID.fullmatch(value) is None:
        raise FinalRuntimeEvidenceError("gpu_uuid must be a canonical physical NVIDIA GPU UUID")
    return value


def _run_command(command_runner: CommandRunner, command: Sequence[str], *, purpose: str) -> str:
    try:
        output = command_runner(tuple(command))
    except Exception as error:
        raise RuntimeError(f"{purpose} command failed") from error
    if not isinstance(output, str):
        raise RuntimeError(f"{purpose} command did not return text")
    return output


def _process_vram_mib(
    *,
    gpu_uuid: str,
    nvidia_smi_executable: str,
    command_runner: CommandRunner,
    pid: int,
) -> float:
    output = _run_command(
        command_runner,
        (
            nvidia_smi_executable,
            "--query-compute-apps=gpu_uuid,pid,used_gpu_memory",
            "--format=csv,noheader,nounits",
        ),
        purpose="selected-process VRAM sampling",
    )
    matched_memory: list[float] = []
    for line in output.splitlines():
        if not line.strip():
            continue
        fields = tuple(field.strip() for field in line.split(","))
        if len(fields) != 3:
            raise RuntimeError("selected-process VRAM evidence is malformed")
        try:
            row_pid = int(fields[1])
            memory_mib = float(fields[2])
        except ValueError as error:
            raise RuntimeError("selected-process VRAM evidence is malformed") from error
        if not math.isfinite(memory_mib) or memory_mib < 0.0:
            raise RuntimeError("selected-process VRAM evidence is invalid")
        if fields[0] == gpu_uuid and row_pid == pid:
            matched_memory.append(memory_mib)
    if not matched_memory:
        raise RuntimeError("nvidia-smi did not report the selected GPU process")
    total = sum(matched_memory)
    if total <= 0.0:
        raise RuntimeError("selected-process VRAM must be positive")
    return total


@dataclass(frozen=True, slots=True)
class _FinalMemorySample:
    label: str
    process_vram_mib: float
    jax_bytes_in_use: int
    jax_peak_bytes_in_use: int


class FinalMemoryRecorder:
    """Record synchronized selected-process VRAM and JAX allocator samples."""

    def __init__(
        self,
        jax_module: Any,
        device: Any,
        gpu_uuid: str,
        nvidia_smi_executable: str | os.PathLike[str],
        command_runner: CommandRunner = _default_command_runner,
    ) -> None:
        if not callable(command_runner):
            raise TypeError("command_runner must be callable")
        self._jax = jax_module
        self._device = device
        self._gpu_uuid = _validated_gpu_uuid(gpu_uuid)
        self._nvidia_smi_executable = _validated_nvidia_smi_executable(nvidia_smi_executable)
        self._command_runner = command_runner
        self._samples: list[_FinalMemorySample] = []

    def sample(self, label: str) -> dict[str, Any]:
        """Synchronize and capture one uniquely labelled memory sample."""

        safe_label = _safe_nonempty_string(label, field="memory sample label")
        if safe_label in {sample.label for sample in self._samples}:
            raise ValueError("memory sample labels must be unique")
        barrier = getattr(self._jax, "effects_barrier", None)
        if not callable(barrier):
            raise RuntimeError("JAX effects_barrier is required for synchronized memory sampling")
        try:
            barrier()
        except Exception as error:
            raise RuntimeError("JAX synchronization failed before memory sampling") from error
        process_vram_mib = _process_vram_mib(
            gpu_uuid=self._gpu_uuid,
            nvidia_smi_executable=self._nvidia_smi_executable,
            command_runner=self._command_runner,
            pid=os.getpid(),
        )
        try:
            statistics = self._device.memory_stats()
        except Exception as error:
            raise RuntimeError("JAX allocator statistics are unavailable") from error
        if not isinstance(statistics, Mapping):
            raise RuntimeError("JAX allocator statistics are unavailable")
        try:
            live_bytes = _plain_nonnegative_integer(
                statistics["bytes_in_use"], field="JAX bytes_in_use"
            )
            peak_bytes = _plain_nonnegative_integer(
                statistics["peak_bytes_in_use"], field="JAX peak_bytes_in_use"
            )
        except KeyError as error:
            raise RuntimeError("JAX allocator statistics omit required byte counters") from error
        if peak_bytes < live_bytes:
            raise RuntimeError("JAX peak allocator bytes are smaller than live bytes")
        sample = _FinalMemorySample(
            label=safe_label,
            process_vram_mib=process_vram_mib,
            jax_bytes_in_use=live_bytes,
            jax_peak_bytes_in_use=peak_bytes,
        )
        self._samples.append(sample)
        return self._sample_payload(sample)

    @staticmethod
    def _sample_payload(sample: _FinalMemorySample) -> dict[str, Any]:
        return {
            "label": sample.label,
            "synchronized": True,
            "process_vram_mib": sample.process_vram_mib,
            "jax_bytes_in_use": sample.jax_bytes_in_use,
            "jax_peak_bytes_in_use": sample.jax_peak_bytes_in_use,
        }

    def evidence(self) -> dict[str, Any]:
        """Return complete public memory evidence without the private physical GPU UUID."""

        if not self._samples:
            raise RuntimeError("at least one final memory sample is required")
        evidence = {
            "schema_version": FINAL_MEMORY_EVIDENCE_SCHEMA_VERSION,
            "sampling_method": (
                "JAX-synchronized selected-process nvidia-smi compute-app memory plus JAX "
                "allocator statistics at labelled phase boundaries"
            ),
            "sample_count": len(self._samples),
            "samples": [self._sample_payload(sample) for sample in self._samples],
            "peak_sampled_process_vram_mib": max(
                sample.process_vram_mib for sample in self._samples
            ),
            "peak_jax_allocator_bytes": max(
                sample.jax_peak_bytes_in_use for sample in self._samples
            ),
            "final_jax_live_bytes": self._samples[-1].jax_bytes_in_use,
        }
        validate_final_memory_evidence(evidence)
        return evidence


def validate_final_memory_evidence(value: object) -> None:
    """Reject incomplete, internally inconsistent, non-finite, or private memory evidence."""

    if not isinstance(value, Mapping):
        raise FinalRuntimeEvidenceError("memory evidence must be an object")
    _require_exact_keys(
        value,
        {
            "schema_version",
            "sampling_method",
            "sample_count",
            "samples",
            "peak_sampled_process_vram_mib",
            "peak_jax_allocator_bytes",
            "final_jax_live_bytes",
        },
        field="memory evidence",
    )
    if (
        type(value["schema_version"]) is not int
        or value["schema_version"] != FINAL_MEMORY_EVIDENCE_SCHEMA_VERSION
    ):
        raise FinalRuntimeEvidenceError("memory evidence schema version differs from v0.1")
    _safe_nonempty_string(value["sampling_method"], field="sampling_method")
    sample_count = _plain_nonnegative_integer(value["sample_count"], field="sample_count")
    samples = value["samples"]
    if not isinstance(samples, (tuple, list)) or not samples or len(samples) != sample_count:
        raise FinalRuntimeEvidenceError("memory samples must match the positive sample count")
    labels: list[str] = []
    process_values: list[float] = []
    allocator_peaks: list[int] = []
    live_values: list[int] = []
    for index, sample in enumerate(samples):
        if not isinstance(sample, Mapping):
            raise FinalRuntimeEvidenceError("each memory sample must be an object")
        _require_exact_keys(
            sample,
            {
                "label",
                "synchronized",
                "process_vram_mib",
                "jax_bytes_in_use",
                "jax_peak_bytes_in_use",
            },
            field=f"samples[{index}]",
        )
        labels.append(_safe_nonempty_string(sample["label"], field=f"samples[{index}].label"))
        if sample["synchronized"] is not True:
            raise FinalRuntimeEvidenceError("every memory sample must be synchronized")
        process_values.append(
            _finite_number(
                sample["process_vram_mib"],
                field=f"samples[{index}].process_vram_mib",
                positive=True,
            )
        )
        live = _plain_nonnegative_integer(
            sample["jax_bytes_in_use"], field=f"samples[{index}].jax_bytes_in_use"
        )
        peak = _plain_nonnegative_integer(
            sample["jax_peak_bytes_in_use"],
            field=f"samples[{index}].jax_peak_bytes_in_use",
        )
        if peak < live:
            raise FinalRuntimeEvidenceError("JAX peak allocator bytes cannot be below live bytes")
        live_values.append(live)
        allocator_peaks.append(peak)
    if len(labels) != len(set(labels)):
        raise FinalRuntimeEvidenceError("memory sample labels must be unique")
    if _finite_number(
        value["peak_sampled_process_vram_mib"],
        field="peak_sampled_process_vram_mib",
        positive=True,
    ) != max(process_values):
        raise FinalRuntimeEvidenceError("peak sampled process VRAM does not match samples")
    if _plain_nonnegative_integer(
        value["peak_jax_allocator_bytes"], field="peak_jax_allocator_bytes"
    ) != max(allocator_peaks):
        raise FinalRuntimeEvidenceError("peak JAX allocator bytes do not match samples")
    if (
        _plain_nonnegative_integer(value["final_jax_live_bytes"], field="final_jax_live_bytes")
        != live_values[-1]
    ):
        raise FinalRuntimeEvidenceError("final JAX live bytes do not match the final sample")
    if not runtime_evidence_is_private_safe(value):
        raise FinalRuntimeEvidenceError("memory evidence contains private or unsafe values")


@dataclass(frozen=True, slots=True)
class _NvidiaGpu:
    index: int
    uuid: str
    name: str
    driver_version: str
    memory_total_mib: float


def _nvidia_inventory(
    command_runner: CommandRunner,
    *,
    nvidia_smi_executable: str,
) -> tuple[_NvidiaGpu, ...]:
    output = _run_command(
        command_runner,
        (
            nvidia_smi_executable,
            "--query-gpu=index,uuid,name,driver_version,memory.total",
            "--format=csv,noheader,nounits",
        ),
        purpose="NVIDIA GPU inventory",
    )
    inventory: list[_NvidiaGpu] = []
    for line in output.splitlines():
        if not line.strip():
            continue
        fields = tuple(field.strip() for field in line.split(","))
        if len(fields) != 5:
            raise RuntimeError("nvidia-smi GPU evidence is malformed")
        try:
            index = int(fields[0])
            memory_total_mib = float(fields[4])
        except ValueError as error:
            raise RuntimeError("nvidia-smi GPU evidence is malformed") from error
        if index < 0 or not math.isfinite(memory_total_mib) or memory_total_mib <= 0.0:
            raise RuntimeError("nvidia-smi GPU evidence is invalid")
        try:
            uuid = _validated_gpu_uuid(fields[1])
            name = _safe_nonempty_string(fields[2], field="GPU name")
            driver = _safe_nonempty_string(fields[3], field="GPU driver version")
        except FinalRuntimeEvidenceError as error:
            raise RuntimeError("nvidia-smi GPU evidence is invalid") from error
        inventory.append(
            _NvidiaGpu(
                index=index,
                uuid=uuid,
                name=name,
                driver_version=driver,
                memory_total_mib=memory_total_mib,
            )
        )
    if not inventory:
        raise RuntimeError("nvidia-smi returned no GPU evidence")
    if len({gpu.index for gpu in inventory}) != len(inventory) or len(
        {gpu.uuid for gpu in inventory}
    ) != len(inventory):
        raise RuntimeError("nvidia-smi GPU identities are not unique")
    return tuple(inventory)


def _selected_gpu(inventory: tuple[_NvidiaGpu, ...]) -> tuple[_NvidiaGpu, bool]:
    visible = os.environ.get("CUDA_VISIBLE_DEVICES")
    if visible is None:
        selected = next((gpu for gpu in inventory if gpu.index == 0), None)
    else:
        token = visible.split(",", maxsplit=1)[0].strip()
        if not token:
            raise RuntimeError("CUDA_VISIBLE_DEVICES contains no selected GPU")
        selected = next(
            (
                gpu
                for gpu in inventory
                if token == gpu.uuid or (token.isdecimal() and int(token) == gpu.index)
            ),
            None,
        )
    if selected is None:
        raise RuntimeError("CUDA_VISIBLE_DEVICES does not map JAX GPU 0 to nvidia-smi")
    return selected, visible is not None


def _cpu_model() -> str:
    try:
        cpuinfo = Path("/proc/cpuinfo").read_text(encoding="utf-8", errors="strict")
    except (OSError, UnicodeError):
        cpuinfo = ""
    cpu_fields: list[tuple[str, str]] = []
    for line in cpuinfo.splitlines():
        key, separator, value = line.partition(":")
        if separator and value.strip():
            cpu_fields.append((key.strip().lower(), value.strip()))
    for preferred_key in ("model name", "hardware", "processor"):
        for key, value in cpu_fields:
            if key == preferred_key and (preferred_key != "processor" or not value.isdecimal()):
                return _safe_nonempty_string(value, field="CPU model")
    candidates = (platform.processor(), platform.uname().processor)
    for candidate in candidates:
        if isinstance(candidate, str) and candidate.strip():
            return _safe_nonempty_string(candidate, field="CPU model")
    raise RuntimeError("the CPU model is unavailable")


def _cuda_runtime(device: Any) -> str:
    client = getattr(device, "client", None)
    runtime = getattr(client, "platform_version", None)
    if not isinstance(runtime, str) or not runtime.strip() or "cuda" not in runtime.lower():
        raise RuntimeError("JAX does not report a CUDA runtime")
    try:
        return _safe_nonempty_string(runtime, field="CUDA runtime")
    except FinalRuntimeEvidenceError as error:
        raise RuntimeError("JAX CUDA runtime evidence is unsafe") from error


def validate_final_runtime_evidence(value: object) -> None:
    """Reject runtime evidence that drifts from the v0.1 platform or leaks private values."""

    if not isinstance(value, Mapping):
        raise FinalRuntimeEvidenceError("runtime evidence must be an object")
    _require_exact_keys(
        value,
        {
            "schema_version",
            "platform",
            "machine",
            "kernel",
            "python",
            "cpu_model",
            "cuda_runtime",
            "cuda_driver",
            "cuda_device_order",
            "cuda_visible_devices_configured",
            "xla_python_client_preallocate",
            "jax_device",
            "packages",
            "selected_gpu",
        },
        field="runtime evidence",
    )
    if (
        type(value["schema_version"]) is not int
        or value["schema_version"] != FINAL_RUNTIME_EVIDENCE_SCHEMA_VERSION
    ):
        raise FinalRuntimeEvidenceError("runtime evidence schema version differs from v0.1")
    if value["platform"] != "Linux" or value["machine"] != "x86_64":
        raise FinalRuntimeEvidenceError("formal runtime must be Linux x86_64")
    _safe_nonempty_string(value["kernel"], field="kernel")
    if not isinstance(value["python"], str) or _PYTHON_311.fullmatch(value["python"]) is None:
        raise FinalRuntimeEvidenceError("formal runtime must use Python 3.11")
    _safe_nonempty_string(value["cpu_model"], field="cpu_model")
    cuda_runtime = _safe_nonempty_string(value["cuda_runtime"], field="cuda_runtime")
    if "cuda" not in cuda_runtime.lower():
        raise FinalRuntimeEvidenceError("cuda_runtime must identify CUDA")
    cuda_driver = _safe_nonempty_string(value["cuda_driver"], field="cuda_driver")
    if value["cuda_device_order"] != "PCI_BUS_ID":
        raise FinalRuntimeEvidenceError("CUDA_DEVICE_ORDER must be PCI_BUS_ID")
    if type(value["cuda_visible_devices_configured"]) is not bool:
        raise FinalRuntimeEvidenceError("cuda_visible_devices_configured must be boolean")
    if value["xla_python_client_preallocate"] != "false":
        raise FinalRuntimeEvidenceError("XLA_PYTHON_CLIENT_PREALLOCATE must be false")

    jax_device = value["jax_device"]
    if not isinstance(jax_device, Mapping):
        raise FinalRuntimeEvidenceError("jax_device must be an object")
    _require_exact_keys(jax_device, {"id", "platform", "device_kind"}, field="jax_device")
    if (
        _plain_nonnegative_integer(jax_device["id"], field="jax_device.id") != 0
        or jax_device["platform"] != "gpu"
    ):
        raise FinalRuntimeEvidenceError("formal JAX device must be logical GPU 0")
    _safe_nonempty_string(jax_device["device_kind"], field="jax_device.device_kind")

    packages = value["packages"]
    if not isinstance(packages, Mapping) or tuple(packages) != FINAL_RUNTIME_PACKAGE_NAMES:
        raise FinalRuntimeEvidenceError("runtime package inventory is incomplete or reordered")
    for name in FINAL_RUNTIME_PACKAGE_NAMES:
        _safe_nonempty_string(packages[name], field=f"packages.{name}")

    selected_gpu = value["selected_gpu"]
    if not isinstance(selected_gpu, Mapping):
        raise FinalRuntimeEvidenceError("selected_gpu must be an object")
    _require_exact_keys(
        selected_gpu,
        {"index", "uuid", "name", "driver_version", "memory_total_mib"},
        field="selected_gpu",
    )
    _plain_nonnegative_integer(selected_gpu["index"], field="selected_gpu.index")
    if selected_gpu["uuid"] != "redacted":
        raise FinalRuntimeEvidenceError("the public GPU UUID must be redacted")
    _safe_nonempty_string(selected_gpu["name"], field="selected_gpu.name")
    if (
        _safe_nonempty_string(selected_gpu["driver_version"], field="selected_gpu.driver_version")
        != cuda_driver
    ):
        raise FinalRuntimeEvidenceError("CUDA driver and selected GPU driver differ")
    _finite_number(
        selected_gpu["memory_total_mib"], field="selected_gpu.memory_total_mib", positive=True
    )
    if not runtime_evidence_is_private_safe(value):
        raise FinalRuntimeEvidenceError("runtime evidence contains private or unsafe values")


def collect_final_runtime_evidence(
    jax_module: Any,
    nvidia_smi_executable: str | os.PathLike[str],
    command_runner: CommandRunner = _default_command_runner,
    package_version: PackageVersionLoader = installed_package_version,
) -> tuple[dict[str, Any], str]:
    """Collect strict public runtime identity and return the private UUID for memory sampling."""

    if not callable(command_runner) or not callable(package_version):
        raise TypeError("command_runner and package_version must be callable")
    absolute_nvidia_smi = _validated_nvidia_smi_executable(nvidia_smi_executable)
    if (
        platform.system() != "Linux"
        or platform.machine() != "x86_64"
        or _PYTHON_311.fullmatch(platform.python_version()) is None
    ):
        raise RuntimeError("formal final evaluation requires Linux x86_64 and Python 3.11")
    if (
        os.environ.get("CUDA_DEVICE_ORDER") != "PCI_BUS_ID"
        or os.environ.get("XLA_PYTHON_CLIENT_PREALLOCATE") != "false"
    ):
        raise RuntimeError("formal final evaluation CUDA environment policy is not active")

    inventory = _nvidia_inventory(
        command_runner,
        nvidia_smi_executable=absolute_nvidia_smi,
    )
    selected_gpu, visible_configured = _selected_gpu(inventory)
    try:
        devices = jax_module.devices("gpu")
    except Exception as error:
        raise RuntimeError("formal final evaluation requires a JAX GPU device") from error
    if not isinstance(devices, (tuple, list)) or not devices:
        raise RuntimeError("formal final evaluation requires a JAX GPU device")
    device = devices[0]
    if (
        isinstance(getattr(device, "id", None), bool)
        or not isinstance(getattr(device, "id", None), Integral)
        or int(device.id) != 0
        or getattr(device, "platform", None) != "gpu"
        or not isinstance(getattr(device, "device_kind", None), str)
        or not device.device_kind.strip()
    ):
        raise RuntimeError("formal JAX device must be logical GPU 0")

    packages: dict[str, str] = {}
    for name in FINAL_RUNTIME_PACKAGE_NAMES:
        try:
            package_value = package_version(name)
        except Exception as error:
            raise RuntimeError("formal runtime package inventory is incomplete") from error
        try:
            packages[name] = _safe_nonempty_string(
                package_value, field=f"package version for {name}"
            )
        except FinalRuntimeEvidenceError as error:
            raise RuntimeError("formal runtime package inventory is incomplete") from error

    public = {
        "schema_version": FINAL_RUNTIME_EVIDENCE_SCHEMA_VERSION,
        "platform": platform.system(),
        "machine": platform.machine(),
        "kernel": platform.release(),
        "python": platform.python_version(),
        "cpu_model": _cpu_model(),
        "cuda_runtime": _cuda_runtime(device),
        "cuda_driver": selected_gpu.driver_version,
        "cuda_device_order": os.environ.get("CUDA_DEVICE_ORDER"),
        "cuda_visible_devices_configured": visible_configured,
        "xla_python_client_preallocate": os.environ.get("XLA_PYTHON_CLIENT_PREALLOCATE"),
        "jax_device": {
            "id": int(device.id),
            "platform": str(device.platform),
            "device_kind": device.device_kind.strip(),
        },
        "packages": packages,
        "selected_gpu": {
            "index": selected_gpu.index,
            "uuid": "redacted",
            "name": selected_gpu.name,
            "driver_version": selected_gpu.driver_version,
            "memory_total_mib": selected_gpu.memory_total_mib,
        },
    }
    validate_final_runtime_evidence(public)
    return public, selected_gpu.uuid
