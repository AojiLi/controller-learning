"""Train the M7 PPO policy on the exact official 1,024-world GPU Challenge.

This entrypoint is deliberately narrower than a general RL launcher. It accepts one versioned
PPO configuration, loads only the verified Level 1 Train pool, constructs the locked public
wrapper stack, and persists enough identity and runtime evidence to audit or safely resume the
local run. Validation and Test assets are never loaded here.
"""

from __future__ import annotations

import os

# These settings must exist before JAX or PyTorch is imported by any project module.
os.environ.setdefault("CUDA_DEVICE_ORDER", "PCI_BUS_ID")
os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")

import argparse
import dataclasses
import gc
import json
import math
import platform
import subprocess
import sys
import time
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from datetime import UTC, datetime
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from typing import Any, Final

from controller_learning.config import ProjectConfig, load_project_config
from controller_learning.rl.artifacts import (
    RESUME_SEMANTICS,
    ArtifactRecord,
    ArtifactValidationError,
    LoadedTrainingCheckpoint,
    TrainingCheckpointMetadata,
    TrainingContinuationState,
    TrainingRunIdentity,
    atomic_write_bytes,
    atomic_write_json,
    load_training_checkpoint,
    read_latest_checkpoint_pointer,
    read_strict_json,
    save_training_checkpoint,
    sha256_file,
)
from controller_learning.rl.assets import TrainPoolAccessEvidence, VerifiedTrainPool
from controller_learning.rl.configuration import PpoTrainingConfig, load_ppo_config
from controller_learning.rl.schema import (
    LOCAL_TRACK_FEATURE_SCHEMA_VERSION,
    PUBLIC_REWARD_SCHEMA_VERSION,
)

PROJECT_ROOT: Final = Path(__file__).resolve().parents[1]
DEFAULT_PPO_CONFIG: Final = Path("configs/ppo.toml")
FORMAL_TORCH_DEVICE: Final = "cuda:0"
TRAINING_MANIFEST_SCHEMA_VERSION: Final = "controller-learning.m7-ppo-training-run.v1"
FORMAL_WRAPPER_ORDER: Final = (
    "VecCarRacingEnv",
    "PublicRewardShapingVecEnv",
    "LocalTrackObservationVecEnv",
    "JaxToTorchVecEnv",
)
PHYSICS_BACKEND: Final = "mjx_warp"
TRAIN_SPLIT: Final = "train"
CHECKPOINT_DIRECTORY: Final = "checkpoints"


class ForbiddenOfficialAssetAccessError(RuntimeError):
    """Raised before a non-Train official Track asset can be opened."""


@dataclass(slots=True)
class OfficialTrainAssetAccessGuard:
    """Process-wide audit guard that permits only the two formal Train inputs.

    Python audit hooks cannot be removed. The CLI therefore installs exactly one guard in its
    dedicated process immediately before the Train loader and keeps it active through shutdown.
    Evidence contains categories and counts only, never machine-specific paths.
    """

    official_track_root: Path
    train_manifest: Path
    track_cache_root: Path
    train_cache: Path
    _installed: bool = False
    _allowed_event_counts: dict[str, int] = field(default_factory=dict)
    _allowed_event_sequence: list[dict[str, str | int | None]] = field(default_factory=list)
    _denied_event_count: int = 0

    def __post_init__(self) -> None:
        for name in (
            "official_track_root",
            "train_manifest",
            "track_cache_root",
            "train_cache",
        ):
            value = Path(getattr(self, name)).resolve(strict=False)
            object.__setattr__(self, name, value)
        if not self.train_manifest.is_relative_to(self.official_track_root):
            raise ValueError("Train manifest must be inside the official Track asset root")
        if not self.train_cache.is_relative_to(self.track_cache_root):
            raise ValueError("Train cache must be inside the configured Track cache root")

    def _category(self, candidate: Path) -> str | None:
        if candidate == self.train_manifest:
            return "official_train_manifest"
        if candidate == self.train_cache:
            return "configured_train_cache"
        return None

    def _is_protected(self, candidate: Path) -> bool:
        if candidate.suffix not in {".json", ".npz"}:
            return False
        return candidate.is_relative_to(self.official_track_root) or candidate.is_relative_to(
            self.track_cache_root
        )

    def _audit(self, event: str, arguments: tuple[Any, ...]) -> None:
        if event != "open" or not arguments:
            return
        source = arguments[0]
        if not isinstance(source, (str, bytes, os.PathLike)):
            return
        candidate = Path(os.fsdecode(os.fspath(source))).resolve(strict=False)
        if not self._is_protected(candidate):
            return
        category = self._category(candidate)
        if category is None:
            self._denied_event_count += 1
            raise ForbiddenOfficialAssetAccessError(
                "PPO optimization forbids non-Train official Track asset access"
            )
        mode = arguments[1] if len(arguments) > 1 else None
        flags = arguments[2] if len(arguments) > 2 else None
        write_mode = isinstance(mode, str) and any(token in mode for token in "wax+")
        write_flags = type(flags) is int and bool(
            flags & (os.O_WRONLY | os.O_RDWR | os.O_CREAT | os.O_TRUNC | os.O_APPEND)
        )
        if write_mode or write_flags:
            self._denied_event_count += 1
            raise ForbiddenOfficialAssetAccessError(
                "PPO optimization permits Train Track assets only as read-only inputs"
            )
        self._allowed_event_counts[category] = self._allowed_event_counts.get(category, 0) + 1
        self._allowed_event_sequence.append(
            {
                "category": category,
                "mode": mode if isinstance(mode, str) else None,
                "flags": flags if type(flags) is int else None,
            }
        )

    def install(self) -> None:
        if self._installed:
            raise RuntimeError("official asset access guard is already installed")
        sys.addaudithook(self._audit)
        self._installed = True

    def evidence(self, *, loader_succeeded: bool) -> dict[str, Any]:
        if type(loader_succeeded) is not bool:
            raise TypeError("loader_succeeded must be a boolean")
        expected = {"official_train_manifest", "configured_train_cache"}
        observed = set(self._allowed_event_counts)
        if loader_succeeded and observed != expected:
            raise RuntimeError(
                "successful Train loading did not audit both expected Train asset categories"
            )
        return {
            "audit_hook_installed_before_asset_loader": self._installed,
            "loader_succeeded": loader_succeeded,
            "opened_splits": [TRAIN_SPLIT] if loader_succeeded else [],
            "opened_path_categories": sorted(observed),
            "open_event_counts": dict(sorted(self._allowed_event_counts.items())),
            "open_event_sequence": list(self._allowed_event_sequence),
            "denied_event_count": self._denied_event_count,
            "validation_opened": False,
            "test_opened": False,
        }


def _positive_integer(value: str) -> int:
    try:
        result = int(value)
    except ValueError as error:
        raise argparse.ArgumentTypeError("value must be a positive integer") from error
    if result < 1:
        raise argparse.ArgumentTypeError("value must be a positive integer")
    return result


@dataclass(frozen=True, slots=True)
class TrainingOptions:
    """Strict command-line inputs for one formal run or explicit smoke prefix."""

    run_id: str
    config: Path = DEFAULT_PPO_CONFIG
    device: str = FORMAL_TORCH_DEVICE
    smoke_updates: int | None = None
    resume: bool = False

    def __post_init__(self) -> None:
        # TrainingRunIdentity owns the canonical run-id validation contract.
        TrainingRunIdentity(
            run_id=self.run_id,
            benchmark_version="0.1",
            source_revision="0" * 40,
            configuration_sha256="0" * 64,
            lock_sha256="0" * 64,
            train_manifest_sha256="0" * 64,
            train_cache_sha256="0" * 64,
            feature_schema_version=LOCAL_TRACK_FEATURE_SCHEMA_VERSION,
            reward_schema_version=PUBLIC_REWARD_SCHEMA_VERSION,
            environment_seed=0,
            policy_seed=1,
            minibatch_seed=2,
        )
        config = Path(self.config)
        if config.suffix != ".toml":
            raise ValueError("config must use the .toml suffix")
        if self.device != FORMAL_TORCH_DEVICE:
            raise ValueError(f"formal M7 training requires device {FORMAL_TORCH_DEVICE!r}")
        if self.smoke_updates is not None and (
            type(self.smoke_updates) is not int or self.smoke_updates < 1
        ):
            raise ValueError("smoke_updates must be a positive integer or None")
        if type(self.resume) is not bool:
            raise TypeError("resume must be a boolean")
        object.__setattr__(self, "config", config)


@dataclass(frozen=True, slots=True)
class SourceSnapshot:
    """Git identity checked before any expensive GPU or Track-pool work."""

    revision: str
    worktree_clean: bool
    status: str

    def __post_init__(self) -> None:
        invalid_character = any(character not in "0123456789abcdef" for character in self.revision)
        if len(self.revision) != 40 or invalid_character:
            raise ValueError("source revision must be a full lowercase Git SHA-1")
        if type(self.worktree_clean) is not bool:
            raise TypeError("worktree_clean must be a boolean")
        if not isinstance(self.status, str):
            raise TypeError("status must be a string")
        if self.worktree_clean != (self.status == ""):
            raise ValueError("worktree_clean and status disagree")

    def to_dict(self) -> dict[str, Any]:
        return {
            "revision": self.revision,
            "worktree_clean": self.worktree_clean,
        }


@dataclass(slots=True)
class FormalTrainingStack:
    """The exact official environment, public wrappers, policy, collector, and updater."""

    base_environment: Any
    environment: Any
    policy: Any
    collector: Any
    updater: Any
    wrapper_order: tuple[str, ...] = FORMAL_WRAPPER_ORDER
    _closed: bool = False

    def close(self) -> None:
        if self._closed:
            return
        self.environment.close()
        self._closed = True

    def __enter__(self) -> FormalTrainingStack:
        return self

    def __exit__(self, *_exception: object) -> None:
        self.close()


def _run_command(
    command: Sequence[str],
    *,
    cwd: Path | None = None,
) -> tuple[str | None, str | None]:
    try:
        completed = subprocess.run(
            tuple(command),
            cwd=cwd,
            check=True,
            capture_output=True,
            text=True,
            timeout=15,
        )
    except (OSError, subprocess.SubprocessError) as error:
        return None, f"{type(error).__name__}: {error}"
    return completed.stdout.strip(), None


def capture_source_snapshot(project_root: Path = PROJECT_ROOT) -> SourceSnapshot:
    """Capture one full Git revision and ignored-output-aware worktree status."""

    revision, revision_error = _run_command(
        ("git", "rev-parse", "--verify", "HEAD"),
        cwd=project_root,
    )
    if revision is None:
        raise RuntimeError(
            f"formal PPO training requires a readable Git revision: {revision_error}"
        )
    status, status_error = _run_command(
        ("git", "status", "--porcelain", "--untracked-files=normal"),
        cwd=project_root,
    )
    if status is None:
        raise RuntimeError(f"formal PPO training requires readable Git status: {status_error}")
    return SourceSnapshot(
        revision=revision,
        worktree_clean=not bool(status),
        status=status,
    )


def require_formal_source(snapshot: SourceSnapshot) -> None:
    """Reject source state that cannot be reproduced from the recorded revision."""

    if not isinstance(snapshot, SourceSnapshot):
        raise TypeError("snapshot must be a SourceSnapshot")
    if not snapshot.worktree_clean:
        raise RuntimeError(
            "formal PPO training requires a clean worktree; commit or stash source changes first"
        )


def _absolute_project_file(project_root: Path, path: Path, *, label: str) -> Path:
    root = project_root.resolve(strict=True)
    source = path if path.is_absolute() else root / path
    if source.is_symlink():
        raise ValueError(f"{label} must be a regular non-symlink file")
    try:
        candidate = source.resolve(strict=True)
    except FileNotFoundError as error:
        raise ValueError(f"{label} does not exist") from error
    try:
        relative = candidate.relative_to(root)
    except ValueError as error:
        raise ValueError(f"{label} must be inside the project root") from error
    if not candidate.is_file():
        raise ValueError(f"{label} must be a regular non-symlink file: {relative.as_posix()}")
    return candidate


def build_run_identity(
    *,
    run_id: str,
    config: PpoTrainingConfig,
    config_path: Path,
    lock_path: Path,
    source: SourceSnapshot,
    train_evidence: TrainPoolAccessEvidence,
) -> TrainingRunIdentity:
    """Bind every immutable optimization input used by checkpoint continuity."""

    if not isinstance(config, PpoTrainingConfig):
        raise TypeError("config must be a PpoTrainingConfig")
    if not isinstance(source, SourceSnapshot):
        raise TypeError("source must be a SourceSnapshot")
    if not isinstance(train_evidence, TrainPoolAccessEvidence):
        raise TypeError("train_evidence must be TrainPoolAccessEvidence")
    if train_evidence.loaded_splits != (TRAIN_SPLIT,):
        raise ValueError("PPO optimization identity may bind only the Train split")
    return TrainingRunIdentity(
        run_id=run_id,
        benchmark_version=config.environment.benchmark_version,
        source_revision=source.revision,
        configuration_sha256=sha256_file(config_path),
        lock_sha256=sha256_file(lock_path),
        train_manifest_sha256=train_evidence.manifest_sha256,
        train_cache_sha256=train_evidence.cache_file_sha256,
        feature_schema_version=LOCAL_TRACK_FEATURE_SCHEMA_VERSION,
        reward_schema_version=PUBLIC_REWARD_SCHEMA_VERSION,
        environment_seed=config.environment.environment_seed,
        policy_seed=config.ppo.policy_seed,
        minibatch_seed=config.ppo.minibatch_seed,
    )


def build_official_training_stack(
    project: ProjectConfig,
    config: PpoTrainingConfig,
    train_pool: VerifiedTrainPool,
    *,
    device: str = FORMAL_TORCH_DEVICE,
) -> FormalTrainingStack:
    """Construct the one supported M7 optimization stack without a fallback backend."""

    if not isinstance(project, ProjectConfig):
        raise TypeError("project must be a ProjectConfig")
    if not isinstance(config, PpoTrainingConfig):
        raise TypeError("config must be a PpoTrainingConfig")
    if not isinstance(train_pool, VerifiedTrainPool):
        raise TypeError("train_pool must be a VerifiedTrainPool")
    if device != FORMAL_TORCH_DEVICE:
        raise ValueError(f"formal M7 training requires device {FORMAL_TORCH_DEVICE!r}")
    if train_pool.evidence.loaded_splits != (TRAIN_SPLIT,):
        raise ValueError("formal PPO optimization must receive only the Train pool")
    if train_pool.pool.split != TRAIN_SPLIT:
        raise ValueError("formal PPO optimization requires a Train TrackPool")

    # GPU-only imports stay below source/config/asset preflight and after allocator environment
    # variables have been set at module entry.
    from controller_learning.envs.vector_racing import VecCarRacingEnv
    from controller_learning.rl.collector import TorchRolloutCollector
    from controller_learning.rl.features import (
        LOCAL_TRACK_FEATURE_DIM,
        LocalTrackObservationVecEnv,
    )
    from controller_learning.rl.policy import PpoActorCritic
    from controller_learning.rl.ppo import PpoUpdater
    from controller_learning.rl.reward import PublicRewardShapingVecEnv
    from controller_learning.rl.torch_bridge import JaxToTorchVecEnv

    base: Any | None = None
    outer: Any | None = None
    try:
        base = VecCarRacingEnv(
            num_envs=config.environment.num_envs,
            project_config=project,
            level_id=config.environment.level_id,
            backend=config.environment.backend,
            track_pool=train_pool.pool,
        )
        shaped = PublicRewardShapingVecEnv(base, config.reward)
        featured = LocalTrackObservationVecEnv(shaped, config=config.observation)
        outer = JaxToTorchVecEnv(featured, device=device)
        policy = PpoActorCritic(
            LOCAL_TRACK_FEATURE_DIM,
            action_low=base.single_action_space.low,
            action_high=base.single_action_space.high,
            policy_seed=config.ppo.policy_seed,
            initial_log_std=config.ppo.initial_log_std,
            hidden_sizes=config.ppo.hidden_sizes,
            device=outer.device,
        )
        collector = TorchRolloutCollector(
            outer,
            policy,
            rollout_steps=config.rollout.steps_per_update,
        )
        updater = PpoUpdater(policy, config.ppo)
        if (
            base.backend != PHYSICS_BACKEND
            or base.level_id != 1
            or base.num_envs != 1024
            or outer.env is not featured
            or featured.env is not shaped
            or shaped.env is not base
        ):
            raise RuntimeError("constructed PPO stack differs from the locked formal wrapper chain")
        return FormalTrainingStack(
            base_environment=base,
            environment=outer,
            policy=policy,
            collector=collector,
            updater=updater,
        )
    except BaseException:
        if outer is not None:
            outer.close()
        elif base is not None:
            base.close()
        raise


def _package_version(name: str) -> str | None:
    try:
        return version(name)
    except PackageNotFoundError:
        return None


def _nvidia_inventory() -> tuple[list[dict[str, Any]], str | None]:
    fields = ("index", "name", "uuid", "driver_version", "memory.total")
    output, error = _run_command(
        (
            "nvidia-smi",
            f"--query-gpu={','.join(fields)}",
            "--format=csv,noheader,nounits",
        )
    )
    if output is None:
        return [], error
    inventory: list[dict[str, Any]] = []
    for line in output.splitlines():
        values = tuple(value.strip() for value in line.split(","))
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


def _selected_physical_gpu(
    inventory: Sequence[Mapping[str, Any]],
) -> tuple[dict[str, Any] | None, str | None]:
    visible = os.environ.get("CUDA_VISIBLE_DEVICES")
    token = None
    if visible:
        tokens = tuple(item.strip() for item in visible.split(",") if item.strip())
        if not tokens:
            return None, "CUDA_VISIBLE_DEVICES contains no usable device"
        token = tokens[0]
    if token is None or token.isdigit():
        index = 0 if token is None else int(token)
        selected = next((dict(item) for item in inventory if item.get("index") == index), None)
    else:
        selected = next(
            (
                dict(item)
                for item in inventory
                if item.get("uuid") == token or str(item.get("uuid", "")).startswith(token)
            ),
            None,
        )
    if selected is None:
        return None, "cannot map logical cuda:0 to nvidia-smi inventory"
    return selected, None


def _process_vram_mib(gpu_uuid: str | None) -> tuple[float | None, str | None]:
    if gpu_uuid is None:
        return None, "selected physical GPU is unavailable"
    output, error = _run_command(
        (
            "nvidia-smi",
            "--query-compute-apps=gpu_uuid,pid,used_gpu_memory",
            "--format=csv,noheader,nounits",
        )
    )
    if output is None:
        return None, error
    total = 0.0
    matched = False
    for line in output.splitlines():
        values = tuple(value.strip() for value in line.split(","))
        if len(values) != 3:
            continue
        try:
            pid = int(values[1])
            used = float(values[2])
        except ValueError:
            continue
        if values[0] == gpu_uuid and pid == os.getpid():
            matched = True
            total += used
    return (total if matched else None), None


def runtime_evidence() -> tuple[dict[str, Any], str | None]:
    """Return path/UUID-safe local runtime identity and the private GPU UUID for sampling."""

    inventory, inventory_error = _nvidia_inventory()
    selected, selection_error = _selected_physical_gpu(inventory)
    gpu_uuid = None if selected is None else str(selected.pop("uuid"))
    evidence = {
        "python": platform.python_version(),
        "platform": platform.system(),
        "kernel": platform.release(),
        "machine": platform.machine(),
        "packages": {
            name: _package_version(name)
            for name in (
                "controller-learning",
                "jax",
                "jaxlib",
                "mujoco",
                "mujoco-mjx",
                "numpy",
                "torch",
                "warp-lang",
            )
        },
        "torch_device": FORMAL_TORCH_DEVICE,
        "selected_gpu": selected,
        "nvidia_smi_error": inventory_error,
        "gpu_selection_error": selection_error,
        "cuda_device_order": os.environ.get("CUDA_DEVICE_ORDER"),
        "cuda_visible_devices_configured": "CUDA_VISIBLE_DEVICES" in os.environ,
        "xla_python_client_preallocate": os.environ.get("XLA_PYTHON_CLIENT_PREALLOCATE"),
    }
    return evidence, gpu_uuid


@dataclass(slots=True)
class MemoryEvidenceRecorder:
    """Capture synchronized process, Torch allocator, and JAX allocator samples."""

    torch: Any
    torch_device: Any
    jax_device: Any
    gpu_uuid: str | None
    started: float = field(default_factory=time.perf_counter)
    samples: list[dict[str, Any]] = field(default_factory=list)

    def sample(self, phase: str) -> dict[str, Any]:
        if not isinstance(phase, str) or not phase:
            raise ValueError("memory sample phase must be a non-empty string")
        self.torch.cuda.synchronize(self.torch_device)
        process_vram, process_error = _process_vram_mib(self.gpu_uuid)
        if process_vram is None:
            raise RuntimeError(f"selected-process VRAM sampling failed: {process_error}")
        try:
            raw_jax = self.jax_device.memory_stats() or {}
            jax_allocator = {
                str(key): int(value)
                for key, value in raw_jax.items()
                if isinstance(value, int) and value >= 0
            }
            if not {"bytes_in_use", "peak_bytes_in_use"} <= set(jax_allocator):
                raise RuntimeError("JAX allocator statistics omit required byte counters")
            jax_error = None
        except (TypeError, ValueError) as error:
            raise RuntimeError("JAX allocator statistics are unavailable") from error
        sample = {
            "phase": phase,
            "elapsed_seconds": time.perf_counter() - self.started,
            "synchronized": True,
            "process_vram_mib": process_vram,
            "process_vram_error": process_error,
            "torch_cuda_allocated_bytes": self.torch.cuda.memory_allocated(self.torch_device),
            "torch_cuda_reserved_bytes": self.torch.cuda.memory_reserved(self.torch_device),
            "torch_cuda_max_allocated_bytes": self.torch.cuda.max_memory_allocated(
                self.torch_device
            ),
            "jax_allocator": jax_allocator,
            "jax_allocator_error": jax_error,
        }
        self.samples.append(sample)
        return sample

    def report(self) -> dict[str, Any]:
        process = tuple(
            float(value)
            for sample in self.samples
            if isinstance((value := sample["process_vram_mib"]), (int, float))
        )
        torch_allocated = tuple(
            int(sample["torch_cuda_allocated_bytes"]) for sample in self.samples
        )
        torch_reserved = tuple(int(sample["torch_cuda_reserved_bytes"]) for sample in self.samples)
        jax_values = tuple(
            int(value)
            for sample in self.samples
            for key, value in sample["jax_allocator"].items()
            if key in {"bytes_in_use", "peak_bytes_in_use"}
        )
        post_close = next(
            (
                sample["jax_allocator"].get("bytes_in_use")
                for sample in reversed(self.samples)
                if sample["phase"] == "after_environment_close"
            ),
            None,
        )
        return {
            "sampling_method": (
                "synchronized selected-process nvidia-smi plus Torch CUDA and JAX allocator "
                "statistics at preflight and configured update boundaries"
            ),
            "samples": list(self.samples),
            "sample_count": len(self.samples),
            "peak_sampled_process_vram_mib": max(process, default=None),
            "peak_sampled_torch_cuda_allocated_bytes": max(torch_allocated, default=None),
            "peak_sampled_torch_cuda_reserved_bytes": max(torch_reserved, default=None),
            "peak_sampled_jax_allocator_bytes": max(jax_values, default=None),
            "post_environment_close_jax_bytes_in_use": post_close,
        }


def warm_up_official_stack(stack: FormalTrainingStack, *, seed: int) -> dict[str, Any]:
    """Compile one reset and deterministic public step, excluded from optimization counts."""

    import torch

    reset_started = time.perf_counter()
    observation, _info = stack.environment.reset(seed=seed)
    torch.cuda.synchronize(stack.policy.device)
    reset_seconds = time.perf_counter() - reset_started

    step_started = time.perf_counter()
    with torch.no_grad():
        action = stack.policy.deterministic(observation).action
    next_observation, reward, terminated, truncated, _info = stack.environment.step(action)
    # Synchronize every numerical output used by the warm-up evidence.
    torch.cuda.synchronize(stack.policy.device)
    checks = torch.stack(
        (
            torch.all(torch.isfinite(next_observation)),
            torch.all(torch.isfinite(reward)),
            torch.logical_not(torch.any(terminated & truncated)),
        )
    ).to(device="cpu")
    if not all(checks.tolist()):
        raise RuntimeError("official wrapper warm-up produced invalid public outputs")
    step_seconds = time.perf_counter() - step_started
    return {
        "method": (
            "wall clock around one full public reset and one deterministic public step with "
            "synchronization; both calls are excluded from optimization and followed by the "
            "trainer's fresh seeded environment reset"
        ),
        "reset_compile_and_execute_seconds": reset_seconds,
        "step_compile_and_execute_seconds": step_seconds,
        "excluded_environment_step_calls": 1,
        "excluded_world_slots": stack.base_environment.num_envs,
    }


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
            raise ValueError("training evidence cannot contain NaN or Infinity")
        return value
    raise TypeError(f"unsupported training evidence type {type(value).__name__}")


def _training_accounting(
    *,
    config: PpoTrainingConfig,
    project: ProjectConfig,
    summary: Any,
) -> dict[str, int | bool]:
    """Recompute the fixed-budget and NEXT_STEP conservation identities."""

    counts = summary.counts
    expected_calls = summary.completed_updates * config.rollout.steps_per_update
    expected_raw = expected_calls * config.environment.num_envs
    if counts.environment_step_calls != expected_calls:
        raise RuntimeError("training summary environment-step count differs from update budget")
    if counts.raw_transitions != expected_raw:
        raise RuntimeError("training summary raw world slots differ from update budget")
    if counts.raw_transitions != counts.valid_transitions + counts.dummy_reset_transitions:
        raise RuntimeError("raw world slots do not equal valid plus reset-only slots")
    if counts.autoreset_slots != counts.dummy_reset_transitions:
        raise RuntimeError("autoreset slots do not equal reset-only slots")
    if counts.terminal_events != counts.terminated_events + counts.truncated_events:
        raise RuntimeError("terminal events do not equal terminated plus truncated events")
    final_pending = counts.terminal_events - counts.autoreset_slots
    if not 0 <= final_pending <= config.environment.num_envs:
        raise RuntimeError("NEXT_STEP terminal/autoreset conservation is invalid")
    if summary.episodes.episodes != counts.terminal_events:
        raise RuntimeError("public episode count differs from terminal events")
    if summary.episodes.invalid_action_episodes != 0:
        raise RuntimeError("formal PPO produced an invalid-action episode")
    return {
        "configured_updates": summary.configured_updates,
        "starting_update": summary.starting_update,
        "completed_updates": summary.completed_updates,
        "invocation_updates": summary.completed_updates - summary.starting_update,
        "configured_budget_completed": summary.configured_budget_completed,
        "environment_step_calls": counts.environment_step_calls,
        "raw_world_slots": counts.raw_transitions,
        "valid_transitions": counts.valid_transitions,
        "dummy_reset_transitions": counts.dummy_reset_transitions,
        "autoreset_slots": counts.autoreset_slots,
        "terminal_events": counts.terminal_events,
        "terminated_events": counts.terminated_events,
        "truncated_events": counts.truncated_events,
        "final_pending_reset_slots": final_pending,
        "physics_substeps": (
            counts.raw_transitions * project.vehicle.simulation.physics_steps_per_control
        ),
        "invalid_action_episodes": summary.episodes.invalid_action_episodes,
    }


def _artifact_dict(record: ArtifactRecord) -> dict[str, Any]:
    return record.to_dict()


def _existing_artifact(root: Path, relative_path: str | Path) -> ArtifactRecord:
    """Hash one completed regular run artifact without following a final symlink."""

    relative = Path(relative_path)
    if relative.is_absolute() or ".." in relative.parts or relative == Path("."):
        raise ArtifactValidationError("run artifact path must be a safe relative path")
    candidate = root.joinpath(relative)
    if candidate.is_symlink() or not candidate.is_file():
        raise ArtifactValidationError("expected run artifact is missing or unsafe")
    return ArtifactRecord(
        relative_path=relative.as_posix(),
        sha256=sha256_file(candidate),
        size_bytes=candidate.stat().st_size,
    )


def _completed_run_artifacts(
    run_directory: Path,
    *,
    summary: Any,
    tensorboard_enabled: bool,
) -> dict[str, Any]:
    """Verify and content-bind CSV, TensorBoard, and the published final checkpoint."""

    expected_metrics = run_directory / "metrics.csv"
    if summary.metrics_path != expected_metrics:
        raise ArtifactValidationError("trainer metrics path differs from the formal run layout")
    pointer = read_latest_checkpoint_pointer(run_directory)
    if pointer is None or pointer.update_index != summary.completed_updates:
        raise ArtifactValidationError("latest checkpoint does not match the completed update")
    checkpoint = _existing_artifact(run_directory, pointer.checkpoint.relative_path)
    if checkpoint != pointer.checkpoint:
        raise ArtifactValidationError("published checkpoint bytes differ from latest pointer")
    latest_pointer = _existing_artifact(
        run_directory,
        f"{CHECKPOINT_DIRECTORY}/latest.json",
    )
    event_records = [
        _existing_artifact(run_directory, path.relative_to(run_directory))
        for path in sorted(run_directory.glob("events.out.tfevents.*"))
    ]
    if tensorboard_enabled and not event_records:
        raise ArtifactValidationError("TensorBoard is enabled but no event artifact was produced")
    if not tensorboard_enabled and event_records:
        raise ArtifactValidationError("TensorBoard artifacts exist while logging is disabled")
    return {
        "config": _artifact_dict(_existing_artifact(run_directory, "config.toml")),
        "metrics_csv": _artifact_dict(_existing_artifact(run_directory, "metrics.csv")),
        "tensorboard_events": [_artifact_dict(record) for record in event_records],
        "final_checkpoint": _artifact_dict(checkpoint),
        "latest_checkpoint_pointer": _artifact_dict(latest_pointer),
    }


def _train_evidence_dict(evidence: TrainPoolAccessEvidence) -> dict[str, Any]:
    return {
        "schema_version": evidence.schema_version,
        "loaded_splits": list(evidence.loaded_splits),
        "benchmark_version": evidence.benchmark_version,
        "generator_version": evidence.generator_version,
        "level_id": evidence.level_id,
        "split": evidence.split,
        "manifest_file": evidence.manifest_file,
        "manifest_sha256": evidence.manifest_sha256,
        "cache_file": evidence.cache_file,
        "manifest_asset_sha256": evidence.manifest_asset_sha256,
        "cache_file_sha256": evidence.cache_file_sha256,
        "track_count": evidence.track_count,
        "capacity": dataclasses.asdict(evidence.capacity),
        "first_track_id": evidence.first_track_id,
        "last_track_id": evidence.last_track_id,
        "track_ids_sha256": evidence.track_ids_sha256,
        "geometry_hashes_sha256": evidence.geometry_hashes_sha256,
        "loader_accessed_validation": evidence.loader_accessed_validation,
        "loader_accessed_test": evidence.loader_accessed_test,
    }


def _initial_manifest(
    *,
    identity: TrainingRunIdentity,
    source: SourceSnapshot,
    config_artifact: ArtifactRecord,
    train_evidence: TrainPoolAccessEvidence,
    asset_access: Mapping[str, Any],
    runtime: Mapping[str, Any],
    smoke_updates: int | None,
    resume_requested: bool,
    resumed_from_update: int | None,
) -> dict[str, Any]:
    return {
        "schema_version": TRAINING_MANIFEST_SCHEMA_VERSION,
        "status": "running",
        "started_at_utc": datetime.now(UTC).isoformat(),
        "run_identity": identity.to_dict(),
        "source": source.to_dict(),
        "protocol": {
            "benchmark_version": identity.benchmark_version,
            "level_id": 1,
            "backend": PHYSICS_BACKEND,
            "num_envs": 1024,
            "environment_construction_count": 1,
            "one_long_lived_environment": True,
            "track_split": TRAIN_SPLIT,
            "wrapper_order": list(FORMAL_WRAPPER_ORDER),
            "autoreset_mode": "NEXT_STEP",
            "torch_device": FORMAL_TORCH_DEVICE,
            "optimization_uses_public_observation_and_reward_wrappers_only": True,
            "cpu_fallback": False,
            "cpu_multiprocessing": False,
            "validation_accessed": False,
            "test_accessed": False,
        },
        "requested_smoke_updates": smoke_updates,
        "resume": {
            "requested": resume_requested,
            "resumed_from_update": resumed_from_update,
            "environment_state_restored": False,
            "fresh_environment_reset_required": True,
            "semantics": RESUME_SEMANTICS,
        },
        "resume_history": [],
        "assets": {
            "train_pool": _train_evidence_dict(train_evidence),
            "runtime_access": dict(asset_access),
        },
        "runtime": dict(runtime),
        "artifacts": {"config": _artifact_dict(config_artifact)},
    }


def _verify_existing_run(
    run_directory: Path,
    *,
    identity: TrainingRunIdentity,
    config_bytes: bytes,
) -> dict[str, Any]:
    manifest = read_strict_json(run_directory, "manifest.json")
    if manifest.get("schema_version") != TRAINING_MANIFEST_SCHEMA_VERSION:
        raise ArtifactValidationError("existing run manifest schema differs from this trainer")
    if manifest.get("run_identity") != identity.to_dict():
        raise ArtifactValidationError("existing run manifest identity differs from current inputs")
    config_path = run_directory / "config.toml"
    if config_path.is_symlink() or not config_path.is_file():
        raise ArtifactValidationError("existing run config snapshot is missing or unsafe")
    if config_path.read_bytes() != config_bytes:
        raise ArtifactValidationError("existing run config snapshot differs from current config")
    return manifest


def _load_resume_checkpoint(
    run_directory: Path,
    *,
    identity: TrainingRunIdentity,
    torch_module: Any,
) -> LoadedTrainingCheckpoint:
    return load_training_checkpoint(
        run_directory,
        expected_identity=identity,
        checkpoint_directory=CHECKPOINT_DIRECTORY,
        torch_module=torch_module,
    )


def _checkpoint_callback(
    run_directory: Path,
    identity: TrainingRunIdentity,
    *,
    keep_last: int,
    torch_module: Any,
) -> Callable[[Any], object]:
    def persist(request: Any) -> object:
        if request.optimizer_state_dict is None:
            raise RuntimeError("formal resume checkpoints require optimizer state")
        resume = request.resume_state
        metadata = TrainingCheckpointMetadata(
            run_identity=identity,
            update_index=request.update_index,
            vector_steps=request.vector_steps,
            valid_transitions=request.counts.valid_transitions,
            elapsed_seconds=request.elapsed_seconds,
        )
        counts = resume.counts
        episodes = resume.episodes
        continuation = TrainingContinuationState(
            starting_update=resume.starting_update,
            num_envs=counts.num_envs,
            environment_step_calls=counts.environment_step_calls,
            raw_transitions=counts.raw_transitions,
            valid_transitions=counts.valid_transitions,
            dummy_reset_transitions=counts.dummy_reset_transitions,
            autoreset_slots=counts.autoreset_slots,
            terminal_events=counts.terminal_events,
            terminated_events=counts.terminated_events,
            truncated_events=counts.truncated_events,
            episodes=episodes.episodes,
            successful_episodes=episodes.successful_episodes,
            offtrack_episodes=episodes.offtrack_episodes,
            invalid_action_episodes=episodes.invalid_action_episodes,
            timeout_episodes=episodes.timeout_episodes,
            successful_lap_time_sum_s=episodes.successful_lap_time_sum_s,
            episode_length_sum_steps=episodes.episode_length_sum_steps,
            cumulative_reward_sum=resume.cumulative_reward_sum,
            cumulative_compute_update_seconds=(resume.cumulative_compute_update_seconds),
            wall_elapsed_before_persistence_seconds=(
                resume.wall_elapsed_before_persistence_seconds
            ),
        )
        return save_training_checkpoint(
            run_directory,
            metadata=metadata,
            continuation_state=continuation,
            model_state_dict=request.model_state_dict,
            optimizer_state_dict=request.optimizer_state_dict,
            policy_rng_state=request.policy_rng_state,
            minibatch_rng_state=request.minibatch_rng_state,
            keep_last=keep_last,
            checkpoint_directory=CHECKPOINT_DIRECTORY,
            torch_module=torch_module,
        )

    return persist


def _restore_model_and_optimizer(
    stack: FormalTrainingStack,
    loaded: LoadedTrainingCheckpoint,
) -> None:
    """Restore trusted local policy/optimizer bytes before constructing trainer resume state."""

    stack.policy.load_state_dict(loaded.payload["model_state_dict"], strict=True)
    stack.updater.optimizer.load_state_dict(loaded.payload["optimizer_state_dict"])
    stack.policy.project_log_std_()


def _trainer_resume_state(loaded: LoadedTrainingCheckpoint | None) -> Any:
    """Adapt a verified checkpoint to the trainer's explicit fresh-environment resume contract."""

    if loaded is None:
        return None
    from controller_learning.rl.rollout import TransitionCounts
    from controller_learning.rl.trainer import EpisodeMetrics, TrainingResumeState

    continuation = loaded.continuation_state
    return TrainingResumeState(
        starting_update=continuation.starting_update,
        counts=TransitionCounts(
            num_envs=continuation.num_envs,
            environment_step_calls=continuation.environment_step_calls,
            raw_transitions=continuation.raw_transitions,
            valid_transitions=continuation.valid_transitions,
            dummy_reset_transitions=continuation.dummy_reset_transitions,
            autoreset_slots=continuation.autoreset_slots,
            terminal_events=continuation.terminal_events,
            terminated_events=continuation.terminated_events,
            truncated_events=continuation.truncated_events,
        ),
        episodes=EpisodeMetrics(
            episodes=continuation.episodes,
            successful_episodes=continuation.successful_episodes,
            offtrack_episodes=continuation.offtrack_episodes,
            invalid_action_episodes=continuation.invalid_action_episodes,
            timeout_episodes=continuation.timeout_episodes,
            successful_lap_time_sum_s=continuation.successful_lap_time_sum_s,
            episode_length_sum_steps=continuation.episode_length_sum_steps,
        ),
        cumulative_reward_sum=continuation.cumulative_reward_sum,
        cumulative_compute_update_seconds=continuation.cumulative_compute_update_seconds,
        wall_elapsed_before_persistence_seconds=(
            continuation.wall_elapsed_before_persistence_seconds
        ),
        policy_rng_state=loaded.payload["policy_rng_state"],
        minibatch_rng_state=loaded.payload["minibatch_rng_state"],
    )


def run_training(options: TrainingOptions, *, project_root: Path = PROJECT_ROOT) -> dict[str, Any]:
    """Execute one full configured run or explicit smoke prefix and return compact evidence."""

    if not isinstance(options, TrainingOptions):
        raise TypeError("options must be TrainingOptions")
    root = Path(project_root)
    config_path = _absolute_project_file(root, options.config, label="PPO config")
    lock_path = _absolute_project_file(root, Path("pixi.lock"), label="Pixi lock")
    config = load_ppo_config(config_path)
    if options.smoke_updates is not None and options.smoke_updates > config.update_count:
        raise ValueError(
            f"smoke_updates cannot exceed configured update count {config.update_count}"
        )
    source = capture_source_snapshot(root)
    require_formal_source(source)

    # This is the sole asset loader used by optimization. It cannot touch Validation or Test.
    from controller_learning.rl.assets import load_verified_train_pool

    project = load_project_config(root)
    train_cache = root / config.environment.train_cache
    asset_guard = OfficialTrainAssetAccessGuard(
        official_track_root=root / "controller_learning/assets/tracks",
        train_manifest=(
            root
            / "controller_learning/assets/tracks"
            / config.environment.benchmark_version
            / "train.json"
        ),
        track_cache_root=root / ".track-cache",
        train_cache=train_cache,
    )
    asset_guard.install()
    verified_train = load_verified_train_pool(
        project,
        train_cache_path=train_cache,
    )
    asset_access = asset_guard.evidence(loader_succeeded=True)
    identity = build_run_identity(
        run_id=options.run_id,
        config=config,
        config_path=config_path,
        lock_path=lock_path,
        source=source,
        train_evidence=verified_train.evidence,
    )
    run_directory = root / config.logging.run_directory / identity.run_id
    config_bytes = config_path.read_bytes()
    runtime, gpu_uuid = runtime_evidence()
    run_wall_started = time.perf_counter()

    if options.resume:
        existing_manifest = _verify_existing_run(
            run_directory,
            identity=identity,
            config_bytes=config_bytes,
        )
        config_artifact = ArtifactRecord(
            relative_path="config.toml",
            sha256=identity.configuration_sha256,
            size_bytes=len(config_bytes),
        )
    else:
        if run_directory.exists() or run_directory.is_symlink():
            relative_run = run_directory.relative_to(root)
            raise FileExistsError(f"run directory already exists: {relative_run}")
        config_artifact = atomic_write_bytes(
            run_directory,
            "config.toml",
            config_bytes,
            overwrite=False,
        )
        existing_manifest = None

    manifest = _initial_manifest(
        identity=identity,
        source=source,
        config_artifact=config_artifact,
        train_evidence=verified_train.evidence,
        asset_access=asset_access,
        runtime=runtime,
        smoke_updates=options.smoke_updates,
        resume_requested=options.resume,
        resumed_from_update=None,
    )
    if existing_manifest is not None:
        manifest["started_at_utc"] = existing_manifest["started_at_utc"]
        manifest["resume"]["prior_manifest_status"] = existing_manifest.get("status")
        prior_history = existing_manifest.get("resume_history", [])
        if not isinstance(prior_history, list) or any(
            not isinstance(item, Mapping) for item in prior_history
        ):
            raise ArtifactValidationError("existing resume_history is not a list of objects")
        manifest["resume_history"] = [dict(item) for item in prior_history]
        manifest["resume_history"].append(
            {
                "requested_at_utc": datetime.now(UTC).isoformat(),
                "prior_manifest_status": existing_manifest.get("status"),
                "checkpoint_update": None,
                "completed_update": None,
            }
        )
    atomic_write_json(run_directory, "manifest.json", manifest)

    stack: FormalTrainingStack | None = None
    memory: MemoryEvidenceRecorder | None = None
    loaded: LoadedTrainingCheckpoint | None = None
    try:
        import jax
        import torch

        if not torch.cuda.is_available():
            raise RuntimeError("formal M7 PPO training requires an available CUDA device")
        jax_gpus = jax.devices("gpu")
        if not jax_gpus:
            raise RuntimeError("formal M7 PPO training requires a visible JAX GPU")
        if runtime.get("selected_gpu") is None or gpu_uuid is None:
            raise RuntimeError("formal M7 PPO training requires mapped nvidia-smi GPU evidence")
        runtime["jax_visible_gpu_count"] = len(jax_gpus)
        runtime["jax_device"] = {
            "logical_id": int(getattr(jax_gpus[0], "id", 0)),
            "platform": str(jax_gpus[0].platform),
            "device_kind": str(jax_gpus[0].device_kind),
        }
        runtime["torch_cuda_version"] = torch.version.cuda

        stack_build_started = time.perf_counter()
        stack = build_official_training_stack(
            project,
            config,
            verified_train,
            device=options.device,
        )
        stack_build_seconds = time.perf_counter() - stack_build_started
        memory = MemoryEvidenceRecorder(
            torch=torch,
            torch_device=stack.policy.device,
            jax_device=jax_gpus[0],
            gpu_uuid=gpu_uuid,
        )
        memory.sample("after_stack_build")
        if options.resume:
            loaded = _load_resume_checkpoint(
                run_directory,
                identity=identity,
                torch_module=torch,
            )
            _restore_model_and_optimizer(stack, loaded)
            manifest["resume"]["resumed_from_update"] = loaded.metadata.update_index
            manifest["resume"]["checkpoint_sha256"] = loaded.pointer.checkpoint.sha256
            manifest["resume_history"][-1]["checkpoint_update"] = loaded.metadata.update_index
        warmup = warm_up_official_stack(
            stack,
            seed=config.environment.environment_seed,
        )
        memory.sample("after_compile_warmup")
        manifest["runtime"] = dict(runtime)
        atomic_write_json(run_directory, "manifest.json", manifest)

        from controller_learning.rl.trainer import TorchCudaMemoryMetrics, train_ppo

        memory_boundary = 0

        def sample_trainer_memory(device: Any) -> Any:
            nonlocal memory_boundary
            memory_boundary += 1
            sample = memory.sample(f"training_boundary_{memory_boundary:04d}")
            return TorchCudaMemoryMetrics(
                allocated_bytes=sample["torch_cuda_allocated_bytes"],
                reserved_bytes=sample["torch_cuda_reserved_bytes"],
                max_allocated_bytes=sample["torch_cuda_max_allocated_bytes"],
            )

        resume_state = _trainer_resume_state(loaded)
        summary = train_ppo(
            stack.collector,
            stack.updater,
            config,
            run_directory=run_directory,
            update_limit=options.smoke_updates,
            checkpoint_callback=_checkpoint_callback(
                run_directory,
                identity,
                keep_last=config.checkpoint.keep_last,
                torch_module=torch,
            ),
            memory_sampler=sample_trainer_memory,
            resume_state=resume_state,
        )
        memory.sample("after_training")
        stack.close()
        gc.collect()
        memory.sample("after_environment_close")
        source_after = capture_source_snapshot(root)
        require_formal_source(source_after)
        input_stability = {
            "source_revision_unchanged": source_after.revision == source.revision,
            "configuration_sha256_unchanged": (
                sha256_file(config_path) == identity.configuration_sha256
            ),
            "lock_sha256_unchanged": sha256_file(lock_path) == identity.lock_sha256,
            "train_manifest_sha256_unchanged": (
                sha256_file(asset_guard.train_manifest) == identity.train_manifest_sha256
            ),
            "train_cache_sha256_unchanged": (
                sha256_file(asset_guard.train_cache) == identity.train_cache_sha256
            ),
        }
        if not all(input_stability.values()):
            raise RuntimeError("formal PPO run inputs changed during optimization")
        final_asset_access = asset_guard.evidence(loader_succeeded=True)
        final_record = summary.records[-1]
        accounting = _training_accounting(
            config=config,
            project=project,
            summary=summary,
        )
        completed_artifacts = _completed_run_artifacts(
            run_directory,
            summary=summary,
            tensorboard_enabled=config.logging.tensorboard_enabled,
        )
        invocation_raw_world_slots = (
            accounting["invocation_updates"] * config.nominal_world_slots_per_update
        )
        manifest.update(
            {
                "status": "complete" if summary.configured_budget_completed else "smoke_complete",
                "completed_at_utc": datetime.now(UTC).isoformat(),
                "timing": {
                    "stack_build_seconds": stack_build_seconds,
                    "compile_warmup": warmup,
                    "training_invocation_compute_update_seconds": (summary.compute_update_seconds),
                    "training_invocation_end_to_end_seconds": (summary.end_to_end_elapsed_seconds),
                    "cumulative_compute_update_seconds": (
                        final_record.cumulative_compute_update_seconds
                    ),
                    "cumulative_wall_before_final_checkpoint_persistence_seconds": (
                        final_record.wall_elapsed_before_persistence_seconds
                    ),
                    "cli_wall_before_final_manifest_seconds": (
                        time.perf_counter() - run_wall_started
                    ),
                },
                "source_after": source_after.to_dict(),
                "input_stability": input_stability,
                "counts": accounting,
                "throughput": {
                    "compute_valid_transitions_per_second": (
                        summary.compute_valid_transitions_per_second
                    ),
                    "end_to_end_valid_transitions_per_second": (
                        summary.end_to_end_valid_transitions_per_second
                    ),
                    "end_to_end_raw_world_slots_per_second": (
                        invocation_raw_world_slots / summary.end_to_end_elapsed_seconds
                    ),
                    "cumulative_compute_valid_transitions_per_second": (
                        summary.counts.valid_transitions
                        / final_record.cumulative_compute_update_seconds
                    ),
                    "cumulative_pre_persistence_valid_transitions_per_second": (
                        summary.counts.valid_transitions
                        / final_record.wall_elapsed_before_persistence_seconds
                    ),
                    "scope": (
                        "trainer end-to-end wall time including configured durable CSV, "
                        "TensorBoard, and checkpoint boundaries"
                    ),
                },
                "episodes": _json_value(summary.episodes),
                "final_optimization": _json_value(final_record.optimization),
                "numerical": {
                    "collector_or_optimizer_exception": False,
                    "nonfinite_boundary_failures": 0,
                    "invalid_action_episodes": summary.episodes.invalid_action_episodes,
                    "final_optimization_metrics_strict_json_finite": True,
                },
                "memory": memory.report(),
                "assets": {
                    "train_pool": _train_evidence_dict(verified_train.evidence),
                    "runtime_access": final_asset_access,
                },
                "artifacts": completed_artifacts,
            }
        )
        if options.resume:
            manifest["resume_history"][-1]["completed_update"] = summary.completed_updates
        atomic_write_json(run_directory, "manifest.json", _json_value(manifest))
        return {
            "run_id": identity.run_id,
            "status": manifest["status"],
            "run_directory": run_directory.relative_to(root).as_posix(),
            "completed_updates": summary.completed_updates,
            "configured_updates": summary.configured_updates,
            "valid_transitions": summary.counts.valid_transitions,
            "valid_transitions_per_second": (summary.end_to_end_valid_transitions_per_second),
        }
    except BaseException as error:
        cleanup_errors: list[BaseException] = []
        if stack is not None:
            try:
                stack.close()
                gc.collect()
            except BaseException as cleanup_error:
                cleanup_errors.append(cleanup_error)
        if memory is not None and stack is not None and stack._closed:
            try:
                memory.sample("after_failure_environment_close")
            except BaseException as cleanup_error:
                cleanup_errors.append(cleanup_error)
        try:
            if run_directory.is_dir() and (run_directory / "manifest.json").is_file():
                failed = read_strict_json(run_directory, "manifest.json")
                failed.update(
                    {
                        "status": "failed",
                        "failed_at_utc": datetime.now(UTC).isoformat(),
                        "failure": {
                            "type": type(error).__name__,
                            "message": str(error),
                        },
                        "memory": None if memory is None else memory.report(),
                    }
                )
                atomic_write_json(run_directory, "manifest.json", _json_value(failed))
        except BaseException as cleanup_error:
            cleanup_errors.append(cleanup_error)
        if cleanup_errors:
            raise BaseExceptionGroup(
                "formal PPO training failed and cleanup also failed",
                [error, *cleanup_errors],
            ) from None
        raise error.with_traceback(error.__traceback__) from None


def _parse_args(argv: Sequence[str] | None = None) -> TrainingOptions:
    parser = argparse.ArgumentParser(
        description=("Train PPO on the exact 1,024-world benchmark 0.1 Level 1 MJX-Warp Train pool")
    )
    parser.add_argument("--run-id", required=True, help="Stable lowercase local run identifier")
    parser.add_argument(
        "--config",
        type=Path,
        default=DEFAULT_PPO_CONFIG,
        help="Strict PPO TOML path inside the repository",
    )
    parser.add_argument(
        "--device",
        default=FORMAL_TORCH_DEVICE,
        choices=(FORMAL_TORCH_DEVICE,),
        help="Locked formal Torch device",
    )
    parser.add_argument(
        "--smoke-updates",
        type=_positive_integer,
        default=None,
        help="Run an explicit update prefix; omit for the full configured budget",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Verify and continue the latest checkpoint with a freshly reset environment",
    )
    values = parser.parse_args(argv)
    return TrainingOptions(
        run_id=values.run_id,
        config=values.config,
        device=values.device,
        smoke_updates=values.smoke_updates,
        resume=values.resume,
    )


def main(argv: Sequence[str] | None = None) -> None:
    result = run_training(_parse_args(argv))
    print(json.dumps(result, allow_nan=False, sort_keys=True))


if __name__ == "__main__":  # pragma: no cover - exercised through Pixi/CLI
    main(sys.argv[1:])
