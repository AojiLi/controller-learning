"""Evaluate the selected PPO as an ordinary Controller and publish one deterministic replay.

The formal workload is fixed by ``configs/ppo_controller_evaluation.toml``. It may read only the
official Validation manifest and NPZ; Train caches and the Test split are blocked before opening.
"""

from __future__ import annotations

import os

# Set allocator and device-order policy before the formal subprocess imports JAX.
os.environ.setdefault("CUDA_DEVICE_ORDER", "PCI_BUS_ID")
os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")

import argparse
import dataclasses
import hashlib
import json
import math
import platform
import re
import shutil
import stat
import subprocess
import sys
import tempfile
import time
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path, PurePosixPath
from typing import Any, ClassVar, Final

PROJECT_ROOT: Final = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG: Final = Path("configs/ppo_controller_evaluation.toml")
OUTPUT_TRANSACTION_DIRECTORY: Final = Path(
    "runs/m7_ppo_controller_evaluation_publication_transaction"
)
OUTPUT_TRANSACTION_CLEANUP_DIRECTORY: Final = Path(
    "runs/m7_ppo_controller_evaluation_publication_transaction.cleanup"
)
FORMAL_OUTPUT_PATHS: Final = (
    "benchmarks/v0.1/m7_ppo_replay_trajectory.json",
    "benchmarks/v0.1/m7_ppo_replay_overview.png",
    "benchmarks/v0.1/m7_ppo_controller_evaluation_report.json",
)
FIRST_USE_TIMING_METHOD: Final = (
    "wall clock around the first actual batch-one environment create, reset, and step calls; "
    "the create sample includes MJX-Warp backend initialization"
)


class ForbiddenControllerEvaluationAssetAccessError(RuntimeError):
    """Raised before a non-Validation official Track asset can be opened."""


@dataclass(slots=True)
class OfficialValidationAssetAccessGuard:
    """Process-wide read-only audit guard for exactly validation.json and validation.npz."""

    official_track_root: Path
    validation_manifest: Path
    validation_asset: Path
    track_cache_root: Path
    _installed: bool = False
    _allowed_event_counts: dict[str, int] = field(default_factory=dict)
    _allowed_event_sequence: list[dict[str, str | int | None]] = field(default_factory=list)
    _denied_event_count: int = 0
    _denied_mutation_event_count: int = 0
    _denied_mutation_event_types: dict[str, int] = field(default_factory=dict)
    _validation_reads_enabled: bool = False

    _MUTATION_PATH_ARGUMENTS: ClassVar[Mapping[str, tuple[tuple[int, int | None], ...]]] = {
        "os.chmod": ((0, 2),),
        "os.chown": ((0, 3),),
        "os.link": ((0, 2), (1, 3)),
        "os.mkfifo": ((0, 2),),
        "os.mkdir": ((0, 2),),
        "os.mknod": ((0, 3),),
        "os.remove": ((0, 1),),
        "os.removexattr": ((0, None),),
        "os.rename": ((0, 2), (1, 3)),
        "os.replace": ((0, 2), (1, 3)),
        "os.rmdir": ((0, 1),),
        "os.setxattr": ((0, None),),
        "os.symlink": ((1, 2),),
        "os.truncate": ((0, None),),
        "os.unlink": ((0, 1),),
        "os.utime": ((0, 3),),
        "shutil.rmtree": ((0, 1),),
    }

    def __post_init__(self) -> None:
        for name in (
            "official_track_root",
            "validation_manifest",
            "validation_asset",
            "track_cache_root",
        ):
            object.__setattr__(self, name, Path(getattr(self, name)).resolve(strict=False))
        for allowed in (self.validation_manifest, self.validation_asset):
            if not allowed.is_relative_to(self.official_track_root):
                raise ValueError("allowed Validation assets must be inside official_track_root")

    def _category(self, candidate: Path) -> str | None:
        if candidate == self.validation_manifest:
            return "official_validation_manifest"
        if candidate == self.validation_asset:
            return "official_validation_asset"
        return None

    @staticmethod
    def _audit_path(source: object, directory_descriptor: object = None) -> Path | None:
        if type(source) is int:
            if source < 0:
                return None
            try:
                return Path(f"/proc/self/fd/{source}").resolve(strict=True)
            except (FileNotFoundError, OSError):
                return None
        if not isinstance(source, (str, bytes, os.PathLike)):
            return None
        path = Path(os.fsdecode(os.fspath(source)))
        if (
            not path.is_absolute()
            and type(directory_descriptor) is int
            and directory_descriptor >= 0
        ):
            descriptor_path = Path(f"/proc/self/fd/{directory_descriptor}")
            try:
                path = descriptor_path.resolve(strict=True) / path
            except (FileNotFoundError, OSError):
                return None
        return path.resolve(strict=False)

    def _is_protected(self, candidate: Path) -> bool:
        return candidate.is_relative_to(self.official_track_root) or candidate.is_relative_to(
            self.track_cache_root
        )

    def _block_mutation(self, event: str, arguments: tuple[Any, ...]) -> None:
        specifications = self._MUTATION_PATH_ARGUMENTS.get(event)
        if specifications is None:
            return
        for path_index, descriptor_index in specifications:
            if path_index >= len(arguments):
                continue
            descriptor = (
                arguments[descriptor_index]
                if descriptor_index is not None and descriptor_index < len(arguments)
                else None
            )
            candidate = self._audit_path(arguments[path_index], descriptor)
            if candidate is not None and self._is_protected(candidate):
                self._denied_event_count += 1
                self._denied_mutation_event_count += 1
                self._denied_mutation_event_types[event] = (
                    self._denied_mutation_event_types.get(event, 0) + 1
                )
                raise ForbiddenControllerEvaluationAssetAccessError(
                    f"M7 Controller evaluation forbids protected filesystem mutation {event}"
                )

    def _audit(self, event: str, arguments: tuple[Any, ...]) -> None:
        self._block_mutation(event, arguments)
        if event != "open" or not arguments:
            return
        source = arguments[0]
        candidate = self._audit_path(source)
        if candidate is None or not self._is_protected(candidate):
            return
        category = self._category(candidate)
        if category is None:
            self._denied_event_count += 1
            raise ForbiddenControllerEvaluationAssetAccessError(
                "M7 ordinary Controller evaluation forbids Train, Test, and Track-cache access"
            )
        if not self._validation_reads_enabled:
            self._denied_event_count += 1
            raise ForbiddenControllerEvaluationAssetAccessError(
                "Validation reads remain disabled until selection, export, and plugin preflight "
                "complete"
            )
        mode = arguments[1] if len(arguments) > 1 else None
        flags = arguments[2] if len(arguments) > 2 else None
        write_mode = isinstance(mode, str) and any(token in mode for token in "wax+")
        write_flags = type(flags) is int and bool(
            flags & (os.O_WRONLY | os.O_RDWR | os.O_CREAT | os.O_TRUNC | os.O_APPEND)
        )
        if write_mode or write_flags:
            self._denied_event_count += 1
            raise ForbiddenControllerEvaluationAssetAccessError("Validation assets are read-only")
        self._allowed_event_counts[category] = self._allowed_event_counts.get(category, 0) + 1
        self._allowed_event_sequence.append(
            {
                "category": category,
                "flags": flags if type(flags) is int else None,
                "mode": mode if isinstance(mode, str) else None,
            }
        )

    def install(self) -> None:
        if self._installed:
            raise RuntimeError("Validation asset guard is already installed")
        sys.addaudithook(self._audit)
        original_mkfifo = os.mkfifo
        original_mknod = os.mknod

        def guarded_mkfifo(
            path: str | bytes | os.PathLike[str] | os.PathLike[bytes],
            mode: int = 0o666,
            *,
            dir_fd: int | None = None,
        ) -> None:
            self._block_mutation("os.mkfifo", (path, mode, -1 if dir_fd is None else dir_fd))
            original_mkfifo(path, mode, dir_fd=dir_fd)

        def guarded_mknod(
            path: str | bytes | os.PathLike[str] | os.PathLike[bytes],
            mode: int = 0o600,
            device: int = 0,
            *,
            dir_fd: int | None = None,
        ) -> None:
            self._block_mutation(
                "os.mknod",
                (path, mode, device, -1 if dir_fd is None else dir_fd),
            )
            original_mknod(path, mode, device, dir_fd=dir_fd)

        # CPython does not emit audit events for these two functions. The formal subprocess is
        # single-purpose, so process-local wrappers close that gap for its remaining lifetime.
        os.mkfifo = guarded_mkfifo  # type: ignore[assignment]
        os.mknod = guarded_mknod  # type: ignore[assignment]
        self._installed = True

    def enable_validation_reads(self) -> None:
        """Open the one-way Validation phase only after every immutable input passes preflight."""

        if not self._installed:
            raise RuntimeError("Validation asset guard must be installed before phase transition")
        if self._validation_reads_enabled:
            raise RuntimeError("Validation reads are already enabled")
        if self._denied_event_count != 0 or self._denied_mutation_event_count != 0:
            raise RuntimeError("a denied asset access permanently closed the Validation phase")
        if self._allowed_event_counts:
            raise RuntimeError("Validation assets were opened before the phase transition")
        self._validation_reads_enabled = True

    def evidence(self, *, validation_loaded: bool) -> dict[str, Any]:
        if type(validation_loaded) is not bool:
            raise TypeError("validation_loaded must be boolean")
        observed = set(self._allowed_event_counts)
        expected = {"official_validation_manifest", "official_validation_asset"}
        if validation_loaded and observed != expected:
            raise RuntimeError("successful Validation loading did not audit both allowed files")
        if not validation_loaded and observed:
            raise RuntimeError("Validation assets were opened before the preflight boundary")
        return {
            "audit_hook_installed_before_preflight": self._installed,
            "denied_event_count": self._denied_event_count,
            "denied_mutation_event_count": self._denied_mutation_event_count,
            "denied_mutation_event_types": dict(sorted(self._denied_mutation_event_types.items())),
            "open_event_counts": dict(sorted(self._allowed_event_counts.items())),
            "open_event_sequence": list(self._allowed_event_sequence),
            "opened_path_categories": sorted(observed),
            "opened_splits": ["validation"] if validation_loaded else [],
            "pre_validation_open_event_count": 0,
            "test_opened": False,
            "track_cache_opened": False,
            "train_opened": False,
            "validation_loaded": validation_loaded,
            "validation_reads_enabled": self._validation_reads_enabled,
        }


@dataclass(frozen=True, slots=True)
class BenchmarkOptions:
    """The frozen protocol config is the only command-line option."""

    config: Path = DEFAULT_CONFIG

    def __post_init__(self) -> None:
        path = Path(self.config)
        if path.suffix != ".toml":
            raise ValueError("Controller evaluation config must use the .toml suffix")
        object.__setattr__(self, "config", path)


def _parse_args(argv: Sequence[str] | None = None) -> BenchmarkOptions:
    parser = argparse.ArgumentParser(
        description="Evaluate the selected PPO through the ordinary batch-one Controller Runner"
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=DEFAULT_CONFIG,
        help="Frozen post-selection evaluation TOML inside the repository",
    )
    return BenchmarkOptions(config=parser.parse_args(argv).config)


def _run_command(command: Sequence[str], *, cwd: Path | None = None) -> str:
    try:
        completed = subprocess.run(
            tuple(command),
            cwd=cwd,
            check=True,
            capture_output=True,
            text=True,
            timeout=20,
        )
    except (OSError, subprocess.SubprocessError) as error:
        raise RuntimeError(f"command failed: {' '.join(command)}") from error
    return completed.stdout.strip()


def _source_snapshot(project_root: Path) -> dict[str, Any]:
    revision = _run_command(("git", "rev-parse", "--verify", "HEAD"), cwd=project_root)
    status = _run_command(
        ("git", "status", "--porcelain", "--untracked-files=normal"), cwd=project_root
    )
    if len(revision) != 40 or any(character not in "0123456789abcdef" for character in revision):
        raise RuntimeError("formal Controller evaluation requires a full lowercase Git revision")
    if status:
        raise RuntimeError("formal Controller evaluation requires a clean worktree")
    return {"revision": revision, "worktree_clean": True}


def _source_snapshot_allowing_outputs(
    project_root: Path,
    *,
    expected_revision: str,
    allowed_paths: Sequence[str],
) -> dict[str, Any]:
    revision = _run_command(("git", "rev-parse", "--verify", "HEAD"), cwd=project_root)
    if revision != expected_revision:
        raise RuntimeError("source revision changed during Controller evaluation")
    status = _run_command(
        ("git", "status", "--porcelain", "--untracked-files=all"), cwd=project_root
    )
    observed: set[str] = set()
    for line in status.splitlines():
        if len(line) < 4:
            raise RuntimeError("Git worktree status output is malformed")
        path = line[3:]
        if " -> " in path:
            path = path.rsplit(" -> ", maxsplit=1)[1]
        observed.add(path)
    allowed_outputs = set(allowed_paths)
    unexpected = observed - allowed_outputs
    if unexpected:
        raise RuntimeError(
            "unexpected worktree changes appeared during Controller evaluation: "
            + ", ".join(sorted(unexpected))
        )
    return {
        "allowed_generated_output_paths": sorted(allowed_outputs),
        "observed_changed_paths": sorted(observed),
        "only_allowed_generated_outputs": True,
        "revision": revision,
        "unexpected_changed_paths": [],
    }


def _project_file(project_root: Path, relative: str | Path, *, label: str) -> Path:
    root = project_root.resolve(strict=True)
    source = Path(relative)
    candidate = source if source.is_absolute() else root / source
    if candidate.is_symlink():
        raise ValueError(f"{label} must be a regular non-symlink file")
    try:
        resolved = candidate.resolve(strict=True)
        resolved.relative_to(root)
    except (FileNotFoundError, ValueError) as error:
        raise ValueError(f"{label} must be an existing file inside the project root") from error
    if not resolved.is_file():
        raise ValueError(f"{label} must be a regular file")
    return resolved


def _canonical_evaluation_config_path(project_root: Path, requested: Path) -> Path:
    config_path = _project_file(
        project_root,
        requested,
        label="Controller evaluation config",
    )
    if config_path.relative_to(project_root).as_posix() != DEFAULT_CONFIG.as_posix():
        raise RuntimeError(
            "formal Controller evaluation requires configs/ppo_controller_evaluation.toml"
        )
    return config_path


def _project_directory(project_root: Path, relative: str | Path, *, label: str) -> Path:
    root = project_root.resolve(strict=True)
    source = Path(relative)
    candidate = source if source.is_absolute() else root / source
    if candidate.is_symlink():
        raise ValueError(f"{label} must be a regular non-symlink directory")
    try:
        resolved = candidate.resolve(strict=True)
        resolved.relative_to(root)
    except (FileNotFoundError, ValueError) as error:
        raise ValueError(
            f"{label} must be an existing directory inside the project root"
        ) from error
    if not resolved.is_dir():
        raise ValueError(f"{label} must be a directory")
    return resolved


def _json_value(value: Any) -> Any:
    if dataclasses.is_dataclass(value) and not isinstance(value, type):
        return _json_value(dataclasses.asdict(value))
    if isinstance(value, Path):
        return value.as_posix()
    if isinstance(value, Mapping):
        return {str(key): _json_value(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [_json_value(item) for item in value]
    if isinstance(value, (str, bool, int)) or value is None:
        return value
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ValueError("Controller evaluation evidence cannot contain NaN or Infinity")
        return value
    raise TypeError(f"unsupported Controller evaluation evidence type {type(value).__name__}")


def _package_version(name: str) -> str | None:
    try:
        return version(name)
    except PackageNotFoundError:
        return None


def _runtime_evidence(jax_module: Any) -> tuple[dict[str, Any], str]:
    inventory = _run_command(
        (
            "nvidia-smi",
            "--query-gpu=index,uuid,name,driver_version,memory.total",
            "--format=csv,noheader,nounits",
        )
    )
    rows = [[part.strip() for part in line.split(",")] for line in inventory.splitlines()]
    if not rows or any(len(row) != 5 for row in rows):
        raise RuntimeError("nvidia-smi GPU evidence is malformed")
    visible = os.environ.get("CUDA_VISIBLE_DEVICES")
    if visible is None:
        selected = rows[0]
    else:
        token = visible.split(",", maxsplit=1)[0].strip()
        selected = next(
            (
                row
                for row in rows
                if token == row[1] or (token.isdecimal() and int(token) == int(row[0]))
            ),
            None,
        )
        if selected is None:
            raise RuntimeError("CUDA_VISIBLE_DEVICES does not identify an inventoried NVIDIA GPU")
    devices = jax_module.devices("gpu")
    if not devices:
        raise RuntimeError("formal Controller evaluation requires a JAX GPU device")
    device = devices[0]
    package_names = (
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
    packages = {name: _package_version(name) for name in package_names}
    if any(not isinstance(value, str) or not value for value in packages.values()):
        raise RuntimeError("formal Controller evaluation package inventory is incomplete")
    if (
        os.environ.get("CUDA_DEVICE_ORDER") != "PCI_BUS_ID"
        or os.environ.get("XLA_PYTHON_CLIENT_PREALLOCATE") != "false"
        or platform.system() != "Linux"
        or platform.machine() != "x86_64"
        or re.fullmatch(r"3\.11(?:\.[0-9]+)?", platform.python_version()) is None
        or int(selected[0]) != 0
        or not selected[2]
        or not selected[3]
        or float(selected[4]) <= 0.0
        or re.fullmatch(
            r"GPU-[0-9a-fA-F]{8}(?:-[0-9a-fA-F]{4}){3}-[0-9a-fA-F]{12}",
            selected[1],
        )
        is None
        or int(getattr(device, "id", -1)) != 0
        or str(getattr(device, "platform", "")) != "gpu"
        or not str(getattr(device, "device_kind", ""))
    ):
        raise RuntimeError("formal Controller evaluation runtime differs from Linux GPU v0.1")
    return (
        {
            "cuda_device_order": os.environ.get("CUDA_DEVICE_ORDER"),
            "cuda_visible_devices_configured": visible is not None,
            "jax_device": {
                "device_kind": str(getattr(device, "device_kind", "")),
                "id": int(getattr(device, "id", 0)),
                "platform": str(getattr(device, "platform", "")),
            },
            "kernel": platform.release(),
            "machine": platform.machine(),
            "packages": packages,
            "platform": platform.system(),
            "python": platform.python_version(),
            "selected_gpu": {
                "driver_version": selected[3],
                "index": int(selected[0]),
                "memory_total_mib": float(selected[4]),
                "name": selected[2],
                "uuid": selected[1],
            },
            "xla_python_client_preallocate": os.environ.get("XLA_PYTHON_CLIENT_PREALLOCATE"),
        },
        selected[1],
    )


def _process_vram_mib(gpu_uuid: str) -> tuple[float, str | None]:
    try:
        output = _run_command(
            (
                "nvidia-smi",
                "--query-compute-apps=gpu_uuid,pid,used_gpu_memory",
                "--format=csv,noheader,nounits",
            )
        )
    except RuntimeError as error:
        return 0.0, str(error)
    total = 0.0
    for line in output.splitlines():
        parts = [part.strip() for part in line.split(",")]
        if len(parts) != 3:
            continue
        try:
            pid = int(parts[1])
            memory = float(parts[2])
        except ValueError:
            continue
        if parts[0] == gpu_uuid and pid == os.getpid():
            total += memory
    return total, None


@dataclass(slots=True)
class MemoryRecorder:
    """Synchronized process and JAX allocator samples at formal phase boundaries."""

    jax: Any
    device: Any
    gpu_uuid: str
    samples: list[dict[str, Any]] = field(default_factory=list)

    def sample(self, phase: str) -> None:
        barrier = getattr(self.jax, "effects_barrier", None)
        if callable(barrier):
            barrier()
        process, error = _process_vram_mib(self.gpu_uuid)
        statistics = self.device.memory_stats() or {}
        self.samples.append(
            {
                "jax_bytes_in_use": int(statistics.get("bytes_in_use", 0)),
                "jax_peak_bytes_in_use": int(statistics.get("peak_bytes_in_use", 0)),
                "phase": phase,
                "process_vram_error": error,
                "process_vram_mib": process,
            }
        )

    def report(self) -> dict[str, Any]:
        process = [float(sample["process_vram_mib"]) for sample in self.samples]
        allocator = [int(sample["jax_peak_bytes_in_use"]) for sample in self.samples]
        return {
            "peak_jax_allocator_bytes": max(allocator, default=0),
            "peak_sampled_process_vram_mib": max(process, default=0.0),
            "sample_count": len(self.samples),
            "samples": list(self.samples),
        }


@dataclass(slots=True)
class _EnvironmentRecord:
    label: str
    create_s: float
    reset_count: int = 0
    step_count: int = 0
    first_reset_s: float | None = None
    first_step_s: float | None = None
    closed: bool = False


class _MeasuredEnvironment:
    def __init__(self, environment: Any, record: _EnvironmentRecord) -> None:
        self._environment = environment
        self._record = record

    @property
    def unwrapped(self) -> Any:
        return self._environment.unwrapped

    def reset(self, *args: Any, **kwargs: Any) -> Any:
        started = time.perf_counter()
        result = self._environment.reset(*args, **kwargs)
        elapsed = time.perf_counter() - started
        self._record.reset_count += 1
        if self._record.first_reset_s is None:
            self._record.first_reset_s = elapsed
        return result

    def step(self, action: object) -> Any:
        started = time.perf_counter()
        result = self._environment.step(action)
        elapsed = time.perf_counter() - started
        self._record.step_count += 1
        if self._record.first_step_s is None:
            self._record.first_step_s = elapsed
        return result

    def render(self) -> Any:
        return self._environment.render()

    def close(self) -> None:
        try:
            self._environment.close()
        finally:
            self._record.closed = True


@dataclass(slots=True)
class ExecutionRecorder:
    """Own measured construction for the evaluation and one replay environment."""

    environment_factory: Callable[..., Any]
    records: list[_EnvironmentRecord] = field(default_factory=list)

    def create(self, label: str, **kwargs: Any) -> _MeasuredEnvironment:
        started = time.perf_counter()
        environment = self.environment_factory(**kwargs)
        record = _EnvironmentRecord(label=label, create_s=time.perf_counter() - started)
        self.records.append(record)
        return _MeasuredEnvironment(environment, record)

    def evaluation_factory(self, **kwargs: Any) -> _MeasuredEnvironment:
        if any(record.label == "evaluation" for record in self.records):
            raise RuntimeError("formal evaluation may construct exactly one reusable environment")
        return self.create("evaluation", **kwargs)

    def first_use_timing(self) -> dict[str, Any]:
        if len(self.records) != 2 or [record.label for record in self.records] != [
            "evaluation",
            "replay",
        ]:
            raise RuntimeError("formal execution must contain evaluation then replay")
        first = self.records[0]
        if first.first_reset_s is None or first.first_step_s is None:
            raise RuntimeError("first-use reset and step timings were not observed")
        if not all(record.closed for record in self.records):
            raise RuntimeError("every formal environment must be closed")
        return {
            "first_environment_create_s": first.create_s,
            "first_reset_s": first.first_reset_s,
            "first_step_s": first.first_step_s,
            "method": FIRST_USE_TIMING_METHOD,
        }


@dataclass(frozen=True, slots=True)
class _OutputSnapshot:
    """Exact pre-publication state used for multi-artifact rollback."""

    relative_path: str
    content: bytes | None
    mode: int


def _capture_output_snapshots(
    root: Path,
    relative_paths: Sequence[str],
) -> tuple[_OutputSnapshot, ...]:
    snapshots: list[_OutputSnapshot] = []
    for relative in relative_paths:
        path = root / relative
        if path.is_symlink():
            raise RuntimeError("formal output paths cannot be symbolic links")
        if path.exists() and not path.is_file():
            raise RuntimeError("formal output paths must be regular files or absent")
        snapshots.append(
            _OutputSnapshot(
                relative_path=relative,
                content=path.read_bytes() if path.exists() else None,
                mode=stat.S_IMODE(path.stat().st_mode) if path.exists() else 0o644,
            )
        )
    return tuple(snapshots)


@dataclass(slots=True)
class _PersistentOutputTransaction:
    """Durably restore all formal outputs after interruption or ordinary failure."""

    project_root: Path
    relative_paths: tuple[str, ...] = FORMAL_OUTPUT_PATHS
    _entered: bool = False
    _committed: bool = False

    def __post_init__(self) -> None:
        self.project_root = Path(self.project_root).resolve(strict=True)
        self.relative_paths = tuple(self.relative_paths)
        if self.relative_paths != FORMAL_OUTPUT_PATHS:
            raise ValueError("persistent publication must cover exactly the three formal outputs")
        for relative in self.relative_paths:
            path = PurePosixPath(relative)
            if (
                path.is_absolute()
                or path.as_posix() != relative
                or any(part in {"", ".", ".."} for part in path.parts)
            ):
                raise ValueError("formal output paths must be normalized relative POSIX paths")

    @property
    def transaction_directory(self) -> Path:
        return self.project_root / OUTPUT_TRANSACTION_DIRECTORY

    @property
    def cleanup_directory(self) -> Path:
        return self.project_root / OUTPUT_TRANSACTION_CLEANUP_DIRECTORY

    @property
    def runs_directory(self) -> Path:
        return self.transaction_directory.parent

    @property
    def staging_directory(self) -> Path:
        return self.transaction_directory / "staging"

    @staticmethod
    def _fsync_directory(path: Path) -> None:
        descriptor = os.open(path, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
        try:
            os.fsync(descriptor)
        finally:
            os.close(descriptor)

    @staticmethod
    def _require_directory(path: Path, *, label: str) -> None:
        try:
            metadata = path.lstat()
        except FileNotFoundError as error:
            raise RuntimeError(f"{label} is missing") from error
        if stat.S_ISLNK(metadata.st_mode) or not stat.S_ISDIR(metadata.st_mode):
            raise RuntimeError(f"{label} must be a non-symlink directory")

    def _ensure_runs_directory(self) -> None:
        runs = self.runs_directory
        try:
            metadata = runs.lstat()
        except FileNotFoundError:
            os.mkdir(runs, 0o755)
            self._fsync_directory(self.project_root)
            return
        if stat.S_ISLNK(metadata.st_mode) or not stat.S_ISDIR(metadata.st_mode):
            raise RuntimeError("runs transaction parent must be a non-symlink directory")

    def _remove_tree(self, path: Path) -> None:
        if not path.exists() and not path.is_symlink():
            return
        self._require_directory(path, label="publication transaction cleanup directory")
        shutil.rmtree(path)
        self._fsync_directory(path.parent)

    def _retire_transaction(self) -> None:
        transaction = self.transaction_directory
        self._require_directory(transaction, label="publication transaction directory")
        if self.cleanup_directory.exists() or self.cleanup_directory.is_symlink():
            self._remove_tree(self.cleanup_directory)
        os.replace(transaction, self.cleanup_directory)
        self._fsync_directory(self.runs_directory)
        self._remove_tree(self.cleanup_directory)

    def _destination(self, relative_path: str) -> Path:
        if relative_path not in self.relative_paths:
            raise ValueError("publication path is not one of the three formal outputs")
        destination = self.project_root / relative_path
        try:
            resolved_parent = destination.parent.resolve(strict=True)
        except FileNotFoundError as error:
            raise RuntimeError("formal output parent directory must already exist") from error
        if resolved_parent != destination.parent:
            raise RuntimeError("formal output parent directories cannot contain symlinks")
        if destination.is_symlink() or (destination.exists() and not destination.is_file()):
            raise RuntimeError("formal output destination must be a regular file or absent")
        return destination

    def _ensure_staging_directory(self) -> None:
        try:
            metadata = self.staging_directory.lstat()
        except FileNotFoundError:
            os.mkdir(self.staging_directory, 0o700)
            self._fsync_directory(self.transaction_directory)
            return
        if stat.S_ISLNK(metadata.st_mode) or not stat.S_ISDIR(metadata.st_mode):
            raise RuntimeError("publication staging path must be a non-symlink directory")

    def _stage_bytes(
        self,
        relative_path: str,
        payload: bytes,
        *,
        mode: int,
        purpose: str,
    ) -> Path:
        if not isinstance(payload, bytes):
            raise TypeError("staged publication payload must be bytes")
        if type(mode) is not int or mode < 0 or mode > 0o777:
            raise ValueError("staged publication mode must be between 0o000 and 0o777")
        if purpose not in {"publish", "restore"}:
            raise ValueError("staged publication purpose differs")
        index = self.relative_paths.index(relative_path)
        self._ensure_staging_directory()
        staged = self.staging_directory / f"{index:03d}.{purpose}"
        flags = os.O_WRONLY | os.O_CREAT | os.O_TRUNC | getattr(os, "O_CLOEXEC", 0)
        flags |= getattr(os, "O_NOFOLLOW", 0)
        descriptor = os.open(staged, flags, mode)
        try:
            os.fchmod(descriptor, mode)
            with os.fdopen(descriptor, "wb") as file:
                descriptor = -1
                file.write(payload)
                file.flush()
                os.fsync(file.fileno())
        finally:
            if descriptor >= 0:
                os.close(descriptor)
        self._fsync_directory(self.staging_directory)
        if staged.is_symlink() or not staged.is_file() or staged.read_bytes() != payload:
            raise RuntimeError("transaction-local staged bytes failed exact readback")
        if stat.S_IMODE(staged.stat().st_mode) != mode:
            raise RuntimeError("transaction-local staged mode failed exact readback")
        return staged

    def _replace_from_stage(
        self,
        relative_path: str,
        payload: bytes,
        *,
        mode: int,
        purpose: str,
    ) -> None:
        staged = self._stage_bytes(relative_path, payload, mode=mode, purpose=purpose)
        destination = self._destination(relative_path)
        os.replace(staged, destination)
        self._fsync_directory(self.staging_directory)
        self._fsync_directory(destination.parent)
        if destination.read_bytes() != payload:
            raise RuntimeError("published formal output bytes failed exact readback")
        if stat.S_IMODE(destination.stat().st_mode) != mode:
            raise RuntimeError("published formal output mode failed exact readback")

    def _restore_snapshots(self, snapshots: Sequence[_OutputSnapshot]) -> None:
        for snapshot in reversed(tuple(snapshots)):
            destination = self._destination(snapshot.relative_path)
            if snapshot.content is None:
                if destination.exists():
                    destination.unlink()
                    self._fsync_directory(destination.parent)
                continue
            self._replace_from_stage(
                snapshot.relative_path,
                snapshot.content,
                mode=snapshot.mode,
                purpose="restore",
            )

    def _manifest(self, snapshots: Sequence[_OutputSnapshot]) -> dict[str, Any]:
        from controller_learning.rl.artifacts import sha256_bytes
        from controller_learning.rl.controller_benchmark import (
            FORMAL_OUTPUT_CRASH_RECOVERY_METHOD,
        )

        outputs = []
        for index, snapshot in enumerate(snapshots):
            existed = snapshot.content is not None
            outputs.append(
                {
                    "backup_relative_path": f"backups/{index:03d}.bin" if existed else None,
                    "existed": existed,
                    "mode": snapshot.mode,
                    "relative_path": snapshot.relative_path,
                    "sha256": sha256_bytes(snapshot.content) if existed else None,
                    "size_bytes": len(snapshot.content) if existed else 0,
                }
            )
        return {
            "outputs": outputs,
            "recovery_method": FORMAL_OUTPUT_CRASH_RECOVERY_METHOD,
            "schema_version": 1,
        }

    def _load_ready_snapshots(self) -> tuple[_OutputSnapshot, ...]:
        from controller_learning.rl.artifacts import canonical_json_bytes, read_strict_json
        from controller_learning.rl.controller_benchmark import (
            FORMAL_OUTPUT_CRASH_RECOVERY_METHOD,
        )

        transaction = self.transaction_directory
        manifest = read_strict_json(transaction, "manifest.json", require_canonical=True)
        if not isinstance(manifest, Mapping) or set(manifest) != {
            "outputs",
            "recovery_method",
            "schema_version",
        }:
            raise RuntimeError("publication transaction manifest keys differ")
        if (
            manifest["schema_version"] != 1
            or manifest["recovery_method"] != FORMAL_OUTPUT_CRASH_RECOVERY_METHOD
        ):
            raise RuntimeError("publication transaction recovery protocol differs")
        outputs = manifest["outputs"]
        if not isinstance(outputs, list) or len(outputs) != len(self.relative_paths):
            raise RuntimeError("publication transaction output coverage differs")
        snapshots: list[_OutputSnapshot] = []
        for index, (record, expected_relative) in enumerate(
            zip(outputs, self.relative_paths, strict=True)
        ):
            if not isinstance(record, Mapping) or set(record) != {
                "backup_relative_path",
                "existed",
                "mode",
                "relative_path",
                "sha256",
                "size_bytes",
            }:
                raise RuntimeError("publication transaction output record keys differ")
            existed = record["existed"]
            mode = record["mode"]
            if (
                type(existed) is not bool
                or type(mode) is not int
                or mode < 0
                or mode > 0o777
                or record["relative_path"] != expected_relative
            ):
                raise RuntimeError("publication transaction output identity differs")
            if not existed:
                if (
                    record["backup_relative_path"] is not None
                    or record["sha256"] is not None
                    or record["size_bytes"] != 0
                ):
                    raise RuntimeError("absent output has unexpected backup content")
                snapshots.append(
                    _OutputSnapshot(
                        relative_path=expected_relative,
                        content=None,
                        mode=mode,
                    )
                )
                continue
            expected_backup = f"backups/{index:03d}.bin"
            if (
                record["backup_relative_path"] != expected_backup
                or not isinstance(record["sha256"], str)
                or re.fullmatch(r"[0-9a-f]{64}", record["sha256"]) is None
                or type(record["size_bytes"]) is not int
                or record["size_bytes"] < 0
            ):
                raise RuntimeError("existing output backup identity differs")
            backup = transaction / expected_backup
            if backup.is_symlink() or not backup.is_file():
                raise RuntimeError("publication output backup must be a regular file")
            content = backup.read_bytes()
            if (
                len(content) != record["size_bytes"]
                or hashlib.sha256(content).hexdigest() != record["sha256"]
            ):
                raise RuntimeError("publication output backup content differs")
            snapshots.append(
                _OutputSnapshot(
                    relative_path=expected_relative,
                    content=content,
                    mode=mode,
                )
            )
        ready = transaction / "READY"
        if ready.is_symlink() or not ready.is_file():
            raise RuntimeError("publication transaction READY marker is invalid")
        expected_ready = hashlib.sha256(canonical_json_bytes(manifest)).hexdigest().encode("ascii")
        if ready.read_bytes() != expected_ready + b"\n":
            raise RuntimeError("publication transaction READY marker differs")
        return tuple(snapshots)

    def _verify_restored(self, snapshots: Sequence[_OutputSnapshot]) -> None:
        for snapshot in snapshots:
            output = self.project_root / snapshot.relative_path
            if snapshot.content is None:
                if output.exists() or output.is_symlink():
                    raise RuntimeError("an originally absent formal output remains after recovery")
                continue
            if output.is_symlink() or not output.is_file():
                raise RuntimeError("a restored formal output is not a regular file")
            if output.read_bytes() != snapshot.content:
                raise RuntimeError("restored formal output bytes differ from backup")
            if stat.S_IMODE(output.stat().st_mode) != snapshot.mode:
                raise RuntimeError("restored formal output mode differs from backup")

    def recover_startup(self) -> str:
        """Recover a READY transaction before the formal clean-worktree gate."""

        runs = self.runs_directory
        if not runs.exists() and not runs.is_symlink():
            return "none"
        self._require_directory(runs, label="runs transaction parent")
        if self.cleanup_directory.exists() or self.cleanup_directory.is_symlink():
            self._remove_tree(self.cleanup_directory)
        transaction = self.transaction_directory
        if not transaction.exists() and not transaction.is_symlink():
            return "none"
        self._require_directory(transaction, label="publication transaction directory")
        ready = transaction / "READY"
        if ready.exists() or ready.is_symlink():
            snapshots = self._load_ready_snapshots()
            self._restore_snapshots(snapshots)
            self._verify_restored(snapshots)
            self._retire_transaction()
            return "restored_ready_transaction"
        self._retire_transaction()
        return "discarded_unready_transaction"

    def __enter__(self) -> _PersistentOutputTransaction:
        from controller_learning.rl.artifacts import (
            atomic_write_bytes,
            atomic_write_json,
            canonical_json_bytes,
        )

        if self._entered:
            raise RuntimeError("persistent output transaction cannot be entered twice")
        self._ensure_runs_directory()
        if self.transaction_directory.exists() or self.transaction_directory.is_symlink():
            raise RuntimeError("pending publication transaction requires startup recovery")
        if self.cleanup_directory.exists() or self.cleanup_directory.is_symlink():
            self._remove_tree(self.cleanup_directory)
        snapshots = _capture_output_snapshots(self.project_root, self.relative_paths)
        os.mkdir(self.transaction_directory, 0o700)
        self._fsync_directory(self.runs_directory)
        os.mkdir(self.transaction_directory / "backups", 0o700)
        self._fsync_directory(self.transaction_directory)
        os.mkdir(self.staging_directory, 0o700)
        self._fsync_directory(self.transaction_directory)
        try:
            manifest = self._manifest(snapshots)
            for record, snapshot in zip(manifest["outputs"], snapshots, strict=True):
                if snapshot.content is not None:
                    atomic_write_bytes(
                        self.transaction_directory,
                        record["backup_relative_path"],
                        snapshot.content,
                        mode=0o600,
                    )
            atomic_write_json(self.transaction_directory, "manifest.json", manifest)
            ready = hashlib.sha256(canonical_json_bytes(manifest)).hexdigest().encode("ascii")
            # READY is the final durable transaction write. Callers may publish only after return.
            atomic_write_bytes(
                self.transaction_directory,
                "READY",
                ready + b"\n",
                mode=0o600,
            )
        except BaseException:
            if self.transaction_directory.exists():
                self._retire_transaction()
            raise
        self._entered = True
        return self

    def commit(self) -> None:
        """Retire durable backups only after every published byte has been verified."""

        if not self._entered or self._committed:
            raise RuntimeError("persistent output transaction is not active")
        self._load_ready_snapshots()
        self._retire_transaction()
        self._committed = True

    def ready_snapshots(self) -> tuple[_OutputSnapshot, ...]:
        """Return the disk-verified backups captured before READY."""

        if not self._entered or self._committed:
            raise RuntimeError("persistent output transaction is not active")
        return self._load_ready_snapshots()

    def prepare_staged_output(
        self,
        relative_path: str,
        payload: bytes,
        *,
        mode: int = 0o644,
    ) -> Path:
        """Durably stage one output inside the READY transaction without publishing it."""

        if not self._entered or self._committed:
            raise RuntimeError("persistent output transaction is not active")
        self._load_ready_snapshots()
        return self._stage_bytes(relative_path, payload, mode=mode, purpose="publish")

    def publish_bytes(
        self,
        relative_path: str,
        payload: bytes,
        *,
        mode: int = 0o644,
    ) -> None:
        """Replace one formal output from deterministic transaction-local staging."""

        if not self._entered or self._committed:
            raise RuntimeError("persistent output transaction is not active")
        self._load_ready_snapshots()
        self._replace_from_stage(
            relative_path,
            payload,
            mode=mode,
            purpose="publish",
        )

    def __exit__(
        self,
        exception_type: type[BaseException] | None,
        exception: BaseException | None,
        traceback: object,
    ) -> bool:
        del exception_type, traceback
        if self._committed:
            return False
        try:
            action = self.recover_startup()
            if action != "restored_ready_transaction":
                raise RuntimeError("active publication transaction was not recoverable")
        except BaseException as recovery_error:
            if exception is None:
                raise
            raise RuntimeError(
                "formal publication failed and persistent recovery also failed"
            ) from recovery_error
        if exception is None:
            raise RuntimeError("persistent output transaction exited without commit")
        return False


def _artifact_record_from_bytes(root: Path, relative_path: str, payload: bytes) -> dict[str, Any]:
    from controller_learning.rl.artifacts import ArtifactRecord, sha256_bytes

    return ArtifactRecord(
        relative_path=(root / relative_path).relative_to(root).as_posix(),
        sha256=sha256_bytes(payload),
        size_bytes=len(payload),
    ).to_dict()


def _artifact_record(root: Path, path: Path) -> dict[str, Any]:
    from controller_learning.rl.artifacts import ArtifactRecord, sha256_file

    return ArtifactRecord(
        relative_path=path.relative_to(root).as_posix(),
        sha256=sha256_file(path),
        size_bytes=path.stat().st_size,
    ).to_dict()


def _input_hashes(paths: Mapping[str, Path]) -> dict[str, str]:
    from controller_learning.rl.artifacts import sha256_file

    return {name: sha256_file(path) for name, path in sorted(paths.items())}


def _selected_checkpoint(selection_report: Mapping[str, Any]) -> tuple[int, str]:
    selected_update = selection_report["selection"]["selected_update"]
    for candidate in selection_report["training_run"]["candidate_checkpoints"]:
        if candidate["update_index"] == selected_update:
            return selected_update, candidate["checkpoint"]["sha256"]
    raise RuntimeError("selection report does not bind the selected retained checkpoint")


def _outcome_matches(recorded: Any, episode: Any, evaluation_identity: Mapping[str, Any]) -> bool:
    result = recorded.result
    info = result.final_info
    lap = float(info["lap_time_s"]) if episode.success else None
    return bool(
        result.steps == episode.steps
        and result.terminated == episode.terminated
        and result.truncated == episode.truncated
        and int(info["episode_seed"]) == evaluation_identity["episode_seed"]
        and int(info["controller_seed"]) == evaluation_identity["controller_seed"]
        and int(info["track_id"]) == episode.track_id
        and str(info["benchmark_version"]) == evaluation_identity["benchmark_version"]
        and bool(info["lap_completed"]) == episode.success
        and int(info["termination_reason"]) == episode.termination_reason
        and lap == episode.lap_time_s
        and math.isclose(result.total_reward, episode.total_reward, rel_tol=0.0, abs_tol=1.0e-6)
    )


def run_benchmark(
    options: BenchmarkOptions,
    *,
    access_guard: OfficialValidationAssetAccessGuard,
    project_root: Path = PROJECT_ROOT,
) -> dict[str, Any]:
    """Execute the frozen post-selection workload and publish strict replay evidence."""

    if not isinstance(options, BenchmarkOptions):
        raise TypeError("options must be BenchmarkOptions")
    if (
        not isinstance(access_guard, OfficialValidationAssetAccessGuard)
        or not access_guard._installed
    ):
        raise RuntimeError("the Validation audit hook must be installed before run_benchmark")

    root = Path(project_root).resolve(strict=True)
    config_path = _canonical_evaluation_config_path(root, options.config)
    publication_transaction = _PersistentOutputTransaction(root)
    # A killed prior publication is restored before the clean-worktree gate can reject its debris.
    publication_transaction.recover_startup()
    preflight = _source_snapshot(root)

    # All project and GPU imports occur after the audit hook is active.
    import jax
    import numpy as np

    from controller_learning.config import load_project_config
    from controller_learning.control import (
        build_public_controller_config,
        load_controller,
        load_controller_config,
        run_controller_episode,
    )
    from controller_learning.envs.car_racing import CarRacingEnv
    from controller_learning.evaluation import (
        evaluate_track_batch,
        record_controller_episode,
        write_trajectory_json,
    )
    from controller_learning.rl.artifacts import (
        canonical_json_bytes,
        read_strict_json,
    )
    from controller_learning.rl.controller_benchmark import (
        CONTROLLER_EVALUATION_REPORT_SCHEMA_VERSION,
        FORMAL_CONTROLLER_EXECUTION_MODEL,
        FORMAL_OUTPUT_CRASH_RECOVERY_METHOD,
        episode_to_report_row,
        evaluation_summary,
        load_ppo_controller_evaluation_config,
        replay_track_index,
        validate_controller_evaluation_report,
    )
    from controller_learning.rl.controller_export import load_ppo_controller_runtime
    from controller_learning.rl.selection import (
        load_ppo_selection_config,
        validate_selection_report,
    )
    from controller_learning.rl.validation_assets import load_verified_validation_pool
    from controller_learning.visualization import write_trajectory_overview_png

    config = load_ppo_controller_evaluation_config(config_path)
    selection_config_path = _project_file(root, config.selection_config, label="selection config")
    selection_report_path = _project_file(root, config.selection_report, label="selection report")
    export_report_path = _project_file(root, config.export_report, label="export report")
    controller_directory = _project_directory(
        root, config.controller_directory, label="PPO Controller"
    )
    lock_path = _project_file(root, "pixi.lock", label="Pixi lock")
    training_config_path = _project_file(root, "configs/ppo.toml", label="PPO training config")
    controller_paths = {
        "controller_config": _project_file(
            root, controller_directory / "config.toml", label="PPO config"
        ),
        "controller_metadata": _project_file(
            root, controller_directory / "metadata.json", label="PPO metadata"
        ),
        "controller_policy": _project_file(
            root, controller_directory / "policy.npz", label="PPO policy"
        ),
        "controller_source": _project_file(
            root, controller_directory / "controller.py", label="PPO source"
        ),
    }
    non_validation_input_paths = {
        **controller_paths,
        "evaluation_config": config_path,
        "export_report": export_report_path,
        "pixi_lock": lock_path,
        "selection_config": selection_config_path,
        "selection_report": selection_report_path,
        "training_config": training_config_path,
    }
    non_validation_initial_sha256 = _input_hashes(non_validation_input_paths)

    selection_config = load_ppo_selection_config(selection_config_path)
    selection_report = read_strict_json(root, config.selection_report, require_canonical=True)
    validate_selection_report(selection_report, config=selection_config)
    if selection_report["status"] != "passed" or selection_report["gates"]["passed"] is not True:
        raise RuntimeError("ordinary Controller evaluation requires a passed selection report")
    selected_update, selected_checkpoint_sha256 = _selected_checkpoint(selection_report)
    selected_candidate = next(
        candidate
        for candidate in selection_report["training_run"]["candidate_checkpoints"]
        if candidate["update_index"] == selected_update
    )
    selected_inference_policy = selected_candidate["inference_policy"]
    training_identity = selection_report["training_run"]["identity"]
    from controller_learning.rl.export_protocol import validate_export_report

    export_report = read_strict_json(root, config.export_report, require_canonical=True)
    validate_export_report(export_report)
    exported_selection = export_report["selection"]
    exported_candidate = exported_selection["selected_candidate"]
    exported_training = export_report["training"]
    exported_controller = export_report["controller"]
    if (
        exported_selection["report"]["sha256"] != non_validation_initial_sha256["selection_report"]
        or exported_selection["report"]["size_bytes"] != selection_report_path.stat().st_size
        or exported_selection["config"]["sha256"]
        != non_validation_initial_sha256["selection_config"]
        or exported_selection["config"]["size_bytes"] != selection_config_path.stat().st_size
        or exported_candidate != selected_candidate
        or exported_training["identity"] != training_identity
        or exported_training["training_config"]["sha256"]
        != non_validation_initial_sha256["training_config"]
        or exported_training["training_config"]["size_bytes"] != training_config_path.stat().st_size
    ):
        raise RuntimeError("canonical export report differs from the frozen selection inputs")

    project = load_project_config(root)
    if project.benchmark.version != config.benchmark_version:
        raise RuntimeError("project benchmark differs from the frozen Controller evaluation")
    controller_parameters = load_controller_config(controller_directory)
    public_controller_config = build_public_controller_config(
        project, config.level_id, controller_parameters
    )
    runtime_controller = load_ppo_controller_runtime(
        public_controller_config, plugin_directory=controller_directory
    )
    load_controller(controller_directory)
    checkpoint = runtime_controller.checkpoint
    exported_artifacts = exported_controller["artifacts"]
    if (
        checkpoint.update_index != selected_update
        or checkpoint.checkpoint_sha256 != selected_checkpoint_sha256
        or checkpoint.training_configuration_sha256 != training_identity["configuration_sha256"]
        or checkpoint.run_id != training_identity["run_id"]
        or checkpoint.source_revision != training_identity["source_revision"]
        or checkpoint.vector_steps != selected_candidate["vector_steps"]
        or checkpoint.valid_transitions != selected_candidate["valid_transitions"]
        or runtime_controller.policy_evidence.sha256 != selected_inference_policy["sha256"]
        or runtime_controller.policy_evidence.size_bytes != selected_inference_policy["size_bytes"]
        or runtime_controller.policy_evidence.schema_version
        != selected_inference_policy["schema_version"]
        or runtime_controller.policy_evidence.sha256 != controller_parameters["policy"]["sha256"]
        or exported_controller["checkpoint"] != checkpoint.to_dict()
        or exported_artifacts["config"]["sha256"]
        != non_validation_initial_sha256["controller_config"]
        or exported_artifacts["config"]["size_bytes"]
        != controller_paths["controller_config"].stat().st_size
        or exported_artifacts["metadata"]["sha256"]
        != non_validation_initial_sha256["controller_metadata"]
        or exported_artifacts["metadata"]["size_bytes"]
        != controller_paths["controller_metadata"].stat().st_size
        or exported_artifacts["policy"]["sha256"] != runtime_controller.policy_evidence.sha256
        or exported_artifacts["policy"]["size_bytes"]
        != runtime_controller.policy_evidence.size_bytes
    ):
        raise RuntimeError("finalized PPO Controller differs from the selected candidate")
    if "torch" in sys.modules:
        raise RuntimeError("ordinary PPO Controller evaluation must remain Torch-free")

    validation_manifest_path = access_guard.validation_manifest
    validation_asset_path = access_guard.validation_asset
    input_sha256_before = _input_hashes(non_validation_input_paths)
    if input_sha256_before != non_validation_initial_sha256:
        raise RuntimeError("formal inputs changed during selection or Controller preflight")
    runtime, gpu_uuid = _runtime_evidence(jax)
    memory = MemoryRecorder(jax=jax, device=jax.devices("gpu")[0], gpu_uuid=gpu_uuid)
    memory.sample("before_environment_create")

    pre_validation_access = access_guard.evidence(validation_loaded=False)
    if (
        pre_validation_access["pre_validation_open_event_count"] != 0
        or pre_validation_access["denied_event_count"] != 0
        or pre_validation_access["denied_mutation_event_count"] != 0
        or pre_validation_access["denied_mutation_event_types"] != {}
    ):
        raise RuntimeError("asset guard recorded denied activity before the Validation boundary")
    if pre_validation_access["validation_reads_enabled"] is not False:
        raise RuntimeError("Validation phase opened before all preflight work completed")
    access_guard.enable_validation_reads()
    validation = load_verified_validation_pool(project)
    if validation.pool.size != config.validation_track_count:
        raise RuntimeError("Validation pool does not contain exactly 100 Tracks")
    expected_track_ids = tuple(int(value) for value in validation.pool.batch.seed)
    reset_seeds = np.arange(config.validation_track_count, dtype=np.uint32)
    validation_evidence = _json_value(validation.evidence)
    input_sha256_before.update(
        {
            "validation_asset": validation_evidence["asset_file_sha256"],
            "validation_manifest": validation_evidence["manifest_sha256"],
        }
    )

    recorder = ExecutionRecorder(environment_factory=CarRacingEnv)
    episode_identities: list[dict[str, int | str]] = []

    def bounded_episode(environment: Any, directory: str, seed: int, **kwargs: Any) -> Any:
        result = run_controller_episode(
            environment,
            directory,
            seed,
            max_steps=config.max_episode_steps,
            **kwargs,
        )
        episode_identities.append(
            {
                "benchmark_version": str(result.final_info["benchmark_version"]),
                "controller_seed": int(result.final_info["controller_seed"]),
                "episode_seed": int(result.final_info["episode_seed"]),
            }
        )
        return result

    evaluation_started = time.perf_counter()
    evaluation = evaluate_track_batch(
        project_config=project,
        level_id=config.level_id,
        batch=validation.pool.batch,
        generator_version=validation.pool.generator_version,
        controller_directory=controller_directory,
        backend=config.backend,
        reset_seeds=reset_seeds,
        track_pool=validation.pool,
        env_factory=recorder.evaluation_factory,
        run_episode=bounded_episode,
    )
    evaluation_wall_s = time.perf_counter() - evaluation_started
    memory.sample("after_controller_evaluation")
    if tuple(episode.track_id for episode in evaluation.episodes) != expected_track_ids:
        raise RuntimeError("ordinary Controller evaluation changed fixed Validation order")
    if len(episode_identities) != config.validation_track_count:
        raise RuntimeError("ordinary Controller evaluation omitted public episode identities")

    rows = [
        episode_to_report_row(episode, **identity)
        for episode, identity in zip(evaluation.episodes, episode_identities, strict=True)
    ]
    selected_replay_index = replay_track_index(rows)
    replay_episode = evaluation.episodes[selected_replay_index]
    replay_environment = recorder.create(
        "replay",
        project_config=project,
        level_id=config.level_id,
        track_pool=validation.pool,
        backend=config.backend,
    )
    replay_started = time.perf_counter()
    try:
        recorded = record_controller_episode(
            replay_environment,
            controller_directory,
            selected_replay_index,
            max_steps=config.max_episode_steps,
            reset_options={"track_index": selected_replay_index},
        )
    finally:
        replay_environment.close()
    replay_wall_s = time.perf_counter() - replay_started
    if not _outcome_matches(
        recorded,
        replay_episode,
        episode_identities[selected_replay_index],
    ):
        raise RuntimeError("deterministic replay outcome differs from its evaluation row")
    memory.sample("after_replay")

    input_paths = {
        **non_validation_input_paths,
        "validation_asset": validation_asset_path,
        "validation_manifest": validation_manifest_path,
    }
    input_sha256_after = _input_hashes(input_paths)
    if input_sha256_before != input_sha256_after:
        raise RuntimeError("formal Controller evaluation inputs changed during the workload")
    selection_artifact = selection_report["artifacts"]["selection_config"]
    if selection_artifact["sha256"] != input_sha256_before["selection_config"]:
        raise RuntimeError("selection report is not bound to the frozen selection config")
    if input_sha256_before["controller_policy"] != runtime_controller.policy_evidence.sha256:
        raise RuntimeError("Controller policy digest changed after strict runtime loading")
    # Render into an external staging directory so report validation cannot dirty the worktree.
    with tempfile.TemporaryDirectory(prefix="controller-learning-m7-replay-") as temporary:
        temporary_root = Path(temporary)
        temporary_trajectory = temporary_root / "trajectory.json"
        temporary_overview = temporary_root / "overview.png"
        trajectory_artifact = write_trajectory_json(
            recorded.trajectory,
            temporary_trajectory,
        )
        overview_artifact = write_trajectory_overview_png(
            recorded.trajectory,
            temporary_overview,
        )
        trajectory_bytes = temporary_trajectory.read_bytes()
        overview_bytes = temporary_overview.read_bytes()
        if (
            hashlib.sha256(trajectory_bytes).hexdigest() != trajectory_artifact.sha256
            or len(trajectory_bytes) != trajectory_artifact.size_bytes
            or hashlib.sha256(overview_bytes).hexdigest() != overview_artifact.sha256
            or len(overview_bytes) != overview_artifact.size_bytes
        ):
            raise RuntimeError("staged replay artifact identity changed before publication")
    trajectory_record = _artifact_record_from_bytes(
        root,
        config.trajectory_path,
        trajectory_bytes,
    )
    overview_record = _artifact_record_from_bytes(
        root,
        config.overview_path,
        overview_bytes,
    )
    memory.sample("after_artifact_render")
    if overview_artifact.rendered_frame_indices != tuple(range(recorded.trajectory.frame_count)):
        raise RuntimeError("overview renderer did not consume every public trajectory frame")

    summary = evaluation_summary(evaluation.episodes)
    evaluation_artifacts = {
        name: _artifact_record(root, path) for name, path in input_paths.items()
    }
    final_access = access_guard.evidence(validation_loaded=True)
    if final_access["denied_event_count"] != 0:
        raise RuntimeError("formal Controller evaluation attempted forbidden asset access")
    allowed_outputs = list(FORMAL_OUTPUT_PATHS)
    payload_outputs = {
        config.trajectory_path: trajectory_bytes,
        config.overview_path: overview_bytes,
    }
    output_snapshots = _capture_output_snapshots(root, allowed_outputs)
    prior_output_content = {
        snapshot.relative_path: snapshot.content for snapshot in output_snapshots
    }
    observed_payload_changes = sorted(
        relative
        for relative, payload in payload_outputs.items()
        if prior_output_content[relative] != payload
    )
    post_output = {
        "allowed_payload_output_paths": sorted(payload_outputs),
        "observed_payload_changed_paths": observed_payload_changes,
        "only_allowed_payload_outputs_before_report_write": True,
        "published_output_bytes_verified": True,
        "report_change_excluded_from_payload_observation": True,
        "report_output_path": config.report_path,
        "revision": preflight["revision"],
        "unexpected_changed_paths": [],
    }
    report: dict[str, Any] = {
        "artifacts": evaluation_artifacts,
        "asset_access": final_access,
        "configuration": config.to_dict(),
        "controller": {
            "checkpoint": checkpoint.to_dict(),
            "config_sha256": input_sha256_before["controller_config"],
            "directory": config.controller_directory,
            "finalized": True,
            "fresh_instance_count": 101,
            "inference_runtime": "numpy",
            "metadata_sha256": input_sha256_before["controller_metadata"],
            "name": "ppo",
            "policy_schema_version": runtime_controller.policy_evidence.schema_version,
            "policy_sha256": runtime_controller.policy_evidence.sha256,
            "policy_size_bytes": runtime_controller.policy_evidence.size_bytes,
            "torch_imported": "torch" in sys.modules,
        },
        "export": {
            "controller_artifacts": exported_artifacts,
            "controller_checkpoint": exported_controller["checkpoint"],
            "report_schema_version": export_report["schema_version"],
            "report_status": export_report["status"],
            "selected_candidate": exported_candidate,
        },
        "evaluation": {"episodes": rows, "summary": summary},
        "execution": {
            "environment_instances": 1,
            "environment_steps": summary["environment_steps"],
            "evaluation_wall_s": evaluation_wall_s,
            "first_use_timing": recorder.first_use_timing(),
            "physics_substeps": project.vehicle.simulation.physics_steps_per_control
            * (summary["environment_steps"] + recorded.result.steps),
            "replay_environment_instances": 1,
            "replay_steps": recorded.result.steps,
            "replay_wall_s": replay_wall_s,
            "transitions_per_second": summary["environment_steps"] / evaluation_wall_s,
        },
        "memory": memory.report(),
        "protocol": {
            "backend": config.backend,
            "benchmark_version": config.benchmark_version,
            "controller_execution_model": FORMAL_CONTROLLER_EXECUTION_MODEL,
            "environment_instances": 1,
            "fresh_controller_per_episode": True,
            "level_id": config.level_id,
            "max_episode_steps": config.max_episode_steps,
            "no_gradient_updates": True,
            "ordinary_controller_plugin": True,
            "output_crash_recovery_method": FORMAL_OUTPUT_CRASH_RECOVERY_METHOD,
            "replay_environment_instances": 1,
            "replay_selection_rule": config.replay_selection_rule,
            "reset_seed_rule": config.reset_seed_rule,
            "test_accessed": False,
            "train_assets_accessed": False,
            "validation_track_count": config.validation_track_count,
        },
        "replay": {
            "evaluation_outcome_matched": True,
            "overview": {
                "all_source_frames_rendered": True,
                "artifact": overview_record,
                "rendered_frame_count": len(overview_artifact.rendered_frame_indices),
                "source_frame_count": overview_artifact.source_frame_count,
            },
            "reset_seed": selected_replay_index,
            "selection_rule": config.replay_selection_rule,
            "track_id": replay_episode.track_id,
            "track_index": selected_replay_index,
            "trajectory": {
                "artifact": trajectory_record,
                "final_lap_completed": bool(recorded.trajectory.final_info["lap_completed"]),
                "final_termination_reason": int(
                    recorded.trajectory.final_info["termination_reason"]
                ),
                "frame_count": recorded.trajectory.frame_count,
                "schema_version": recorded.trajectory.schema_version,
                "step_count": recorded.trajectory.step_count,
            },
        },
        "runtime": runtime,
        "schema_version": CONTROLLER_EVALUATION_REPORT_SCHEMA_VERSION,
        "selection": {
            "gate_passed": True,
            "report_status": "passed",
            "selected_checkpoint_sha256": selected_checkpoint_sha256,
            "selected_inference_policy_schema_version": selected_inference_policy["schema_version"],
            "selected_inference_policy_sha256": selected_inference_policy["sha256"],
            "selected_inference_policy_size_bytes": selected_inference_policy["size_bytes"],
            "selected_success_count": selection_report["selection"]["selected_success_count"],
            "selected_success_rate": selection_report["selection"]["selected_success_rate"],
            "selected_update": selected_update,
            "training_configuration_sha256": training_identity["configuration_sha256"],
        },
        "source": {
            "input_sha256_after": input_sha256_after,
            "input_sha256_before": input_sha256_before,
            "post_output_worktree": post_output,
            "preflight": preflight,
        },
        "status": "passed",
        "validation_assets": _json_value(validation.evidence),
    }
    report = _json_value(report)
    validate_controller_evaluation_report(report, config=config)
    expected_report_bytes = canonical_json_bytes(report)
    report_record = _artifact_record_from_bytes(root, config.report_path, expected_report_bytes)
    with publication_transaction as transaction:
        if transaction.ready_snapshots() != output_snapshots:
            raise RuntimeError("formal outputs changed before durable transaction entry")
        transaction.publish_bytes(config.trajectory_path, trajectory_bytes)
        transaction.publish_bytes(config.overview_path, overview_bytes)
        transaction.publish_bytes(config.report_path, expected_report_bytes)
        final_output = _source_snapshot_allowing_outputs(
            root,
            expected_revision=preflight["revision"],
            allowed_paths=allowed_outputs,
        )
        actual_payload_changes = sorted(
            set(final_output["observed_changed_paths"]) & set(payload_outputs)
        )
        if actual_payload_changes != observed_payload_changes:
            raise RuntimeError("payload-output Git evidence changed during publication")
        published_bytes = {
            config.trajectory_path: (root / config.trajectory_path).read_bytes(),
            config.overview_path: (root / config.overview_path).read_bytes(),
            config.report_path: (root / config.report_path).read_bytes(),
        }
        expected_bytes = {**payload_outputs, config.report_path: expected_report_bytes}
        if published_bytes != expected_bytes:
            raise RuntimeError("published output bytes differ from prevalidated artifacts")
        transaction.commit()
    return {
        "report": report_record["relative_path"],
        "replay_track_id": replay_episode.track_id,
        "selected_update": selected_update,
        "status": "passed",
        "success_count": evaluation.success_count,
    }


def main(argv: Sequence[str] | None = None) -> None:
    options = _parse_args(argv)
    guard = OfficialValidationAssetAccessGuard(
        official_track_root=PROJECT_ROOT / "controller_learning/assets/tracks",
        validation_manifest=(
            PROJECT_ROOT / "controller_learning/assets/tracks/v0.1/validation.json"
        ),
        validation_asset=PROJECT_ROOT / "controller_learning/assets/tracks/v0.1/validation.npz",
        track_cache_root=PROJECT_ROOT / ".track-cache",
    )
    # This is the first stateful action in the dedicated formal subprocess.
    guard.install()
    print(json.dumps(run_benchmark(options, access_guard=guard), allow_nan=False, sort_keys=True))


if __name__ == "__main__":  # pragma: no cover - exercised through the Pixi task.
    main(sys.argv[1:])
