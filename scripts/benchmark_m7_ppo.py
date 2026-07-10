"""Select the frozen M7 PPO checkpoint on Validation without touching Test or Train assets."""

from __future__ import annotations

import os

# Allocator policy must be fixed before JAX or Torch can be imported by the formal subprocess.
os.environ.setdefault("CUDA_DEVICE_ORDER", "PCI_BUS_ID")
os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")

import argparse
import csv
import dataclasses
import hashlib
import io
import json
import math
import platform
import subprocess
import sys
import time
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from typing import Any, Final

PROJECT_ROOT: Final = Path(__file__).resolve().parents[1]
DEFAULT_SELECTION_CONFIG: Final = Path("configs/ppo_selection.toml")
TRAINING_MANIFEST_SCHEMA: Final = "controller-learning.m7-ppo-training-run.v1"
FORMAL_DEVICE: Final = "cuda:0"


class ForbiddenSelectionAssetAccessError(RuntimeError):
    """Raised before any non-Validation official Track asset can be opened."""


@dataclass(slots=True)
class OfficialValidationAssetAccessGuard:
    """Process-wide read-only guard for exactly validation.json and validation.npz."""

    official_track_root: Path
    validation_manifest: Path
    validation_asset: Path
    track_cache_root: Path
    _installed: bool = False
    _allowed_event_counts: dict[str, int] = field(default_factory=dict)
    _allowed_event_sequence: list[dict[str, str | int | None]] = field(default_factory=list)
    _denied_event_count: int = 0
    _validation_reads_enabled: bool = False

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

    def _audit(self, event: str, arguments: tuple[Any, ...]) -> None:
        if event != "open" or not arguments:
            return
        source = arguments[0]
        if not isinstance(source, (str, bytes, os.PathLike)):
            return
        candidate = Path(os.fsdecode(os.fspath(source))).resolve(strict=False)
        protected = candidate.is_relative_to(self.official_track_root) or candidate.is_relative_to(
            self.track_cache_root
        )
        if not protected:
            return
        category = self._category(candidate)
        if category is None:
            self._denied_event_count += 1
            raise ForbiddenSelectionAssetAccessError(
                "M7 checkpoint selection forbids Train, Test, and Track-cache access"
            )
        if not self._validation_reads_enabled:
            self._denied_event_count += 1
            raise ForbiddenSelectionAssetAccessError(
                "Validation reads remain disabled until candidate preflight completes"
            )
        mode = arguments[1] if len(arguments) > 1 else None
        flags = arguments[2] if len(arguments) > 2 else None
        write_mode = isinstance(mode, str) and any(token in mode for token in "wax+")
        write_flags = type(flags) is int and bool(
            flags & (os.O_WRONLY | os.O_RDWR | os.O_CREAT | os.O_TRUNC | os.O_APPEND)
        )
        if write_mode or write_flags:
            self._denied_event_count += 1
            raise ForbiddenSelectionAssetAccessError("Validation assets are read-only")
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
        self._installed = True

    def enable_validation_reads(self) -> None:
        """Open the one-way Validation phase only after all candidate preflight succeeds."""

        if not self._installed:
            raise RuntimeError("Validation asset guard must be installed before phase transition")
        if self._validation_reads_enabled:
            raise RuntimeError("Validation reads are already enabled")
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
    config: Path = DEFAULT_SELECTION_CONFIG

    def __post_init__(self) -> None:
        path = Path(self.config)
        if path.suffix != ".toml":
            raise ValueError("selection config must use the .toml suffix")
        object.__setattr__(self, "config", path)


def _parse_args(argv: Sequence[str] | None = None) -> BenchmarkOptions:
    parser = argparse.ArgumentParser(
        description="Select the frozen M7 PPO checkpoint on the fixed Validation Track set"
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=DEFAULT_SELECTION_CONFIG,
        help="Frozen selection TOML inside the repository",
    )
    values = parser.parse_args(argv)
    return BenchmarkOptions(config=values.config)


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
        ("git", "status", "--porcelain", "--untracked-files=normal"),
        cwd=project_root,
    )
    if len(revision) != 40 or any(character not in "0123456789abcdef" for character in revision):
        raise RuntimeError("selection requires a full lowercase Git revision")
    if status:
        raise RuntimeError("formal Validation selection requires a clean worktree")
    return {"revision": revision, "worktree_clean": True}


def _source_snapshot_allowing_outputs(
    project_root: Path,
    *,
    expected_revision: str,
    allowed_paths: Sequence[str],
) -> dict[str, Any]:
    """Prove that only the two declared generated artifacts may dirty the worktree."""

    revision = _run_command(("git", "rev-parse", "--verify", "HEAD"), cwd=project_root)
    if revision != expected_revision:
        raise RuntimeError("source revision changed during Validation selection")
    status = _run_command(
        ("git", "status", "--porcelain", "--untracked-files=all"),
        cwd=project_root,
    )
    observed: set[str] = set()
    for line in status.splitlines():
        if len(line) < 4:
            raise RuntimeError("Git worktree status output is malformed")
        path = line[3:]
        if " -> " in path:
            path = path.rsplit(" -> ", maxsplit=1)[1]
        observed.add(path)
    allowed = set(allowed_paths)
    unexpected = observed - allowed
    if unexpected:
        raise RuntimeError(
            "unexpected worktree changes appeared during Validation selection: "
            + ", ".join(sorted(unexpected))
        )
    return {
        "allowed_generated_output_paths": sorted(allowed),
        "observed_changed_paths": sorted(observed),
        "only_allowed_generated_outputs": True,
        "revision": revision,
        "unexpected_changed_paths": [],
    }


def _project_file(project_root: Path, relative: Path, *, label: str) -> Path:
    root = project_root.resolve(strict=True)
    source = relative if relative.is_absolute() else root / relative
    if source.is_symlink():
        raise ValueError(f"{label} must be a regular non-symlink file")
    try:
        candidate = source.resolve(strict=True)
        candidate.relative_to(root)
    except (FileNotFoundError, ValueError) as error:
        raise ValueError(f"{label} must be an existing file inside the project root") from error
    if not candidate.is_file():
        raise ValueError(f"{label} must be a regular file")
    return candidate


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
            raise ValueError("selection evidence cannot contain NaN or Infinity")
        return value
    raise TypeError(f"unsupported selection evidence type {type(value).__name__}")


def _package_version(name: str) -> str | None:
    try:
        return version(name)
    except PackageNotFoundError:
        return None


def _runtime_evidence(torch_module: Any) -> tuple[dict[str, Any], str]:
    inventory = _run_command(
        (
            "nvidia-smi",
            "--query-gpu=index,uuid,name,driver_version,memory.total",
            "--format=csv,noheader,nounits",
        )
    )
    selected = inventory.splitlines()[0].split(",")
    if len(selected) != 5:
        raise RuntimeError("nvidia-smi GPU evidence is malformed")
    cuda_runtime = torch_module.version.cuda
    if not isinstance(cuda_runtime, str) or not cuda_runtime:
        raise RuntimeError("Torch does not report a CUDA runtime")
    return (
        {
            "cuda_device_order": os.environ.get("CUDA_DEVICE_ORDER"),
            "cuda_visible_devices_configured": "CUDA_VISIBLE_DEVICES" in os.environ,
            "kernel": platform.release(),
            "machine": platform.machine(),
            "packages": {
                name: _package_version(name)
                for name in (
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
            },
            "platform": platform.system(),
            "python": platform.python_version(),
            "selected_gpu": {
                "driver_version": selected[3].strip(),
                "index": int(selected[0].strip()),
                "memory_total_mib": float(selected[4].strip()),
                "name": selected[2].strip(),
                "uuid": selected[1].strip(),
            },
            "torch_cuda_runtime": cuda_runtime,
            "torch_device": FORMAL_DEVICE,
            "xla_python_client_preallocate": os.environ.get("XLA_PYTHON_CLIENT_PREALLOCATE"),
        },
        selected[1].strip(),
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
        fields = [field.strip() for field in line.split(",")]
        if len(fields) != 3:
            continue
        try:
            pid = int(fields[1])
            memory = float(fields[2])
        except ValueError:
            continue
        if fields[0] == gpu_uuid and pid == os.getpid():
            total += memory
    return total, None


@dataclass(slots=True)
class MemoryRecorder:
    torch: Any
    device: Any
    jax_device: Any
    gpu_uuid: str
    samples: list[dict[str, Any]] = field(default_factory=list)

    def sample(self, phase: str) -> None:
        self.torch.cuda.synchronize(self.device)
        statistics = self.jax_device.memory_stats() or {}
        process_vram_mib, process_vram_error = _process_vram_mib(self.gpu_uuid)
        self.samples.append(
            {
                "jax_bytes_in_use": int(statistics.get("bytes_in_use", 0)),
                "jax_peak_bytes_in_use": int(statistics.get("peak_bytes_in_use", 0)),
                "phase": phase,
                "process_vram_error": process_vram_error,
                "process_vram_mib": process_vram_mib,
                "synchronized": True,
                "torch_allocated_bytes": self.torch.cuda.memory_allocated(self.device),
                "torch_max_allocated_bytes": self.torch.cuda.max_memory_allocated(self.device),
                "torch_reserved_bytes": self.torch.cuda.memory_reserved(self.device),
            }
        )

    def report(self) -> dict[str, Any]:
        return {
            "sample_count": len(self.samples),
            "samples": list(self.samples),
            "peak_jax_bytes_in_use": max(
                (sample["jax_peak_bytes_in_use"] for sample in self.samples), default=0
            ),
            "peak_torch_allocated_bytes": max(
                (sample["torch_max_allocated_bytes"] for sample in self.samples), default=0
            ),
            "peak_sampled_process_vram_mib": max(
                (sample["process_vram_mib"] for sample in self.samples), default=0.0
            ),
        }


def _read_curve_metrics(
    path: Path, *, expected_updates: int
) -> tuple[list[int], list[float], list[float]]:
    if path.is_symlink() or not path.is_file():
        raise ValueError("formal metrics CSV must be a regular non-symlink file")
    with path.open(encoding="utf-8", newline="") as file:
        reader = csv.DictReader(file)
        required = {"update_index", "cumulative_success_rate", "cumulative_mean_valid_reward"}
        if not required <= set(reader.fieldnames or ()):
            raise ValueError("training metrics CSV omits required curve columns")
        rows = list(reader)
    if len(rows) != expected_updates:
        raise ValueError("training metrics CSV does not contain the complete update budget")
    updates: list[int] = []
    success: list[float] = []
    reward: list[float] = []
    for expected, row in enumerate(rows, start=1):
        try:
            update = int(row["update_index"])
            success_value = float(row["cumulative_success_rate"])
            reward_value = float(row["cumulative_mean_valid_reward"])
        except (KeyError, TypeError, ValueError) as error:
            raise ValueError("training metrics CSV contains an invalid curve value") from error
        if update != expected or not all(map(math.isfinite, (success_value, reward_value))):
            raise ValueError("training metrics CSV curve rows are not a finite ordered prefix")
        if not 0.0 <= success_value <= 1.0:
            raise ValueError("cumulative success rate must be in [0, 1]")
        updates.append(update)
        success.append(success_value)
        reward.append(reward_value)
    return updates, success, reward


def deterministic_training_curve_png(
    updates: Sequence[int],
    success_rate: Sequence[float],
    mean_reward: Sequence[float],
    *,
    width_px: int,
    height_px: int,
    dpi: int,
) -> bytes:
    """Render a fixed English two-panel PNG with stable dimensions and metadata."""

    if not (len(updates) == len(success_rate) == len(mean_reward)) or not updates:
        raise ValueError("training curve series must have one common non-zero length")
    import matplotlib

    matplotlib.use("Agg", force=True)
    from matplotlib import pyplot as plt

    with plt.rc_context(
        {
            "axes.grid": True,
            "font.family": "DejaVu Sans",
            "font.size": 11,
            "figure.facecolor": "white",
            "savefig.facecolor": "white",
        }
    ):
        figure, axes = plt.subplots(
            2,
            1,
            figsize=(width_px / dpi, height_px / dpi),
            dpi=dpi,
            constrained_layout=True,
        )
        axes[0].plot(updates, success_rate, color="#1f77b4", linewidth=2.0)
        axes[0].set_title("M7 PPO Training Curve")
        axes[0].set_ylabel("Cumulative success rate")
        axes[0].set_ylim(0.0, 1.0)
        axes[1].plot(updates, mean_reward, color="#d62728", linewidth=2.0)
        axes[1].set_xlabel("PPO update")
        axes[1].set_ylabel("Cumulative mean valid reward")
        output = io.BytesIO()
        figure.savefig(
            output,
            format="png",
            dpi=dpi,
            metadata={"Software": "controller-learning"},
            pil_kwargs={"compress_level": 9},
        )
        plt.close(figure)
    payload = output.getvalue()
    if not payload.startswith(b"\x89PNG\r\n\x1a\n"):
        raise RuntimeError("training curve renderer did not produce PNG bytes")
    return payload


def evaluate_first_terminal_rows(
    environment: Any,
    action_provider: Callable[[Any], Any],
    *,
    expected_track_ids: Sequence[int],
    reset_seed: int,
    max_vector_steps: int,
    control_dt_s: float,
    torch_module: Any,
) -> tuple[Any, ...]:
    """Collect exactly the first terminal event for every fixed world under NEXT_STEP."""

    import numpy as np

    from controller_learning.rl.selection import SelectionTrackResult

    track_ids = tuple(int(value) for value in expected_track_ids)
    num_envs = len(track_ids)
    if num_envs < 1 or getattr(environment, "num_envs", None) != num_envs:
        raise ValueError("expected_track_ids must match environment.num_envs")
    if (
        isinstance(control_dt_s, bool)
        or not isinstance(control_dt_s, (int, float))
        or not math.isfinite(float(control_dt_s))
        or float(control_dt_s) <= 0.0
    ):
        raise ValueError("control_dt_s must be finite and positive")
    observation, initial_info = environment.reset(
        seed=reset_seed,
        options={"track_indices": np.arange(num_envs, dtype=np.int32)},
    )
    if not isinstance(observation, Mapping) or set(observation) != {"features", "track_progress"}:
        raise ValueError("selection observation must contain only features and track_progress")
    device = observation["features"].device
    expected_ids = torch_module.tensor(track_ids, dtype=torch_module.uint32, device=device)
    if not torch_module.equal(initial_info["track_id"], expected_ids):
        raise ValueError("Validation reset did not preserve manifest Track order")
    done = torch_module.zeros(num_envs, dtype=torch_module.bool, device=device)
    reasons = torch_module.zeros(num_envs, dtype=torch_module.int32, device=device)
    lap_times = torch_module.zeros(num_envs, dtype=torch_module.float32, device=device)
    steps = torch_module.zeros(num_envs, dtype=torch_module.int32, device=device)
    # CUDA does not implement ``where`` for uint32. Track IDs remain exact after widening.
    terminal_track_ids = torch_module.zeros(num_envs, dtype=torch_module.int64, device=device)
    initial_progress = observation["track_progress"]
    if (
        not isinstance(initial_progress, torch_module.Tensor)
        or initial_progress.shape != (num_envs,)
        or initial_progress.dtype is not torch_module.float32
        or initial_progress.device != device
        or not bool(torch_module.all(torch_module.isfinite(initial_progress)))
        or not bool(torch_module.all((initial_progress >= 0.0) & (initial_progress <= 1.0)))
    ):
        raise ValueError("selection track_progress must be a finite float32 tensor in [0, 1]")
    max_progress = initial_progress.clone()
    action_low = torch_module.as_tensor(
        environment.single_action_space.low,
        dtype=torch_module.float32,
        device=device,
    )
    action_high = torch_module.as_tensor(
        environment.single_action_space.high,
        dtype=torch_module.float32,
        device=device,
    )
    if action_low.shape != (2,) or action_high.shape != (2,):
        raise ValueError("selection environment must expose two physical action bounds")
    zero_action: Any | None = None

    with torch_module.inference_mode():
        for step_index in range(1, max_vector_steps + 1):
            action = action_provider(observation["features"])
            if (
                not isinstance(action, torch_module.Tensor)
                or action.shape != (num_envs, 2)
                or action.dtype is not torch_module.float32
                or action.device != device
            ):
                raise ValueError("selection action provider returned an invalid tensor")
            if not bool(torch_module.all(torch_module.isfinite(action))) or not bool(
                torch_module.all((action >= action_low) & (action <= action_high))
            ):
                raise ValueError("selection actions must be finite and inside physical bounds")
            if zero_action is None:
                zero_action = torch_module.zeros_like(action)
            action = torch_module.where(done[:, None], zero_action, action)
            next_observation, _reward, terminated, truncated, info = environment.step(action)
            if (
                not isinstance(terminated, torch_module.Tensor)
                or not isinstance(truncated, torch_module.Tensor)
                or terminated.shape != (num_envs,)
                or truncated.shape != (num_envs,)
                or terminated.dtype is not torch_module.bool
                or truncated.dtype is not torch_module.bool
                or terminated.device != device
                or truncated.device != device
                or not isinstance(info, Mapping)
            ):
                raise ValueError("selection terminal outputs violate the public tensor schema")
            required_info = {
                "track_id": torch_module.uint32,
                "termination_reason": torch_module.int32,
                "lap_completed": torch_module.bool,
                "lap_time_s": torch_module.float32,
            }
            for name, dtype in required_info.items():
                value = info.get(name)
                if (
                    not isinstance(value, torch_module.Tensor)
                    or value.shape != (num_envs,)
                    or value.dtype is not dtype
                    or value.device != device
                ):
                    raise ValueError(
                        f"selection info field {name!r} violates the public tensor schema"
                    )
            if torch_module.any(terminated & truncated):
                raise ValueError("a world cannot terminate and truncate together")
            active = torch_module.logical_not(done)
            terminal = terminated | truncated
            reason = info["termination_reason"]
            success = reason == 1
            terminated_reason = (reason >= 1) & (reason <= 3)
            semantic_valid = (
                torch_module.where(terminal, reason != 0, reason == 0)
                & torch_module.where(terminated, terminated_reason, True)
                & torch_module.where(truncated, reason == 4, True)
                & (info["lap_completed"] == success)
                & (
                    info["track_id"].to(dtype=torch_module.int64)
                    == expected_ids.to(dtype=torch_module.int64)
                )
            )
            expected_lap_time = torch_module.full_like(
                info["lap_time_s"],
                step_index,
            ) * torch_module.as_tensor(
                control_dt_s,
                dtype=info["lap_time_s"].dtype,
                device=device,
            )
            lap_time_valid = torch_module.where(
                success,
                torch_module.isclose(
                    info["lap_time_s"],
                    expected_lap_time,
                    rtol=0.0,
                    atol=1.0e-6,
                ),
                info["lap_time_s"] == 0.0,
            )
            if not bool(
                torch_module.all(torch_module.where(active, semantic_valid, True))
            ) or not bool(torch_module.all(torch_module.where(active, lap_time_valid, True))):
                raise ValueError(
                    "selection terminal reason, flags, success, Track identity, or lap time differ"
                )
            next_progress = next_observation["track_progress"]
            if (
                not isinstance(next_progress, torch_module.Tensor)
                or next_progress.shape != (num_envs,)
                or next_progress.dtype is not torch_module.float32
                or next_progress.device != device
                or not bool(torch_module.all(torch_module.isfinite(next_progress)))
                or not bool(torch_module.all((next_progress >= 0.0) & (next_progress <= 1.0)))
            ):
                raise ValueError(
                    "selection track_progress must be a finite float32 tensor in [0, 1]"
                )
            max_progress.copy_(
                torch_module.where(
                    active, torch_module.maximum(max_progress, next_progress), max_progress
                )
            )
            first_terminal = active & (terminated | truncated)
            reasons.copy_(torch_module.where(first_terminal, info["termination_reason"], reasons))
            lap_times.copy_(torch_module.where(first_terminal, info["lap_time_s"], lap_times))
            steps.copy_(
                torch_module.where(
                    first_terminal,
                    torch_module.full_like(steps, step_index),
                    steps,
                )
            )
            terminal_track_ids.copy_(
                torch_module.where(
                    first_terminal,
                    info["track_id"].to(dtype=torch_module.int64),
                    terminal_track_ids,
                )
            )
            done.logical_or_(first_terminal)
            observation = next_observation
            if bool(torch_module.all(done)):
                break
        else:
            raise RuntimeError("not every Validation world reached a first terminal event")

    host_reason = reasons.to(device="cpu").tolist()
    host_lap = lap_times.to(device="cpu").tolist()
    host_progress = max_progress.to(device="cpu").tolist()
    host_steps = steps.to(device="cpu").tolist()
    host_terminal_ids = terminal_track_ids.to(device="cpu").tolist()
    rows = tuple(
        SelectionTrackResult(
            track_index=index,
            track_id=track_id,
            termination_reason=int(host_reason[index]),
            success=int(host_reason[index]) == 1,
            lap_time_s=float(host_lap[index]),
            max_progress=float(host_progress[index]),
            steps=int(host_steps[index]),
        )
        for index, track_id in enumerate(track_ids)
    )
    if tuple(int(value) for value in host_terminal_ids) != track_ids:
        raise ValueError("first terminal rows do not retain initial Validation Track IDs")
    return rows


def _validation_evidence(evidence: Any) -> dict[str, Any]:
    return _json_value(evidence)


def _build_stack(project: Any, training_config: Any, validation: Any) -> Any:
    from controller_learning.envs.vector_racing import VecCarRacingEnv
    from controller_learning.rl.selection_observation import SelectionPublicObservationVecEnv
    from controller_learning.rl.torch_bridge import JaxToTorchVecEnv

    base = VecCarRacingEnv(
        num_envs=100,
        project_config=project,
        level_id=1,
        backend="mjx_warp",
        track_pool=validation.pool,
    )
    try:
        featured = SelectionPublicObservationVecEnv(base, config=training_config.observation)
        bridge = JaxToTorchVecEnv(featured, device=FORMAL_DEVICE)
    except BaseException:
        base.close()
        raise
    return dataclasses.make_dataclass(
        "SelectionStack",
        (("base", Any), ("environment", Any)),
        frozen=True,
        slots=True,
    )(base, bridge)


def _artifact_from_mapping(value: object, *, artifact_type: Any) -> Any:
    if not isinstance(value, Mapping):
        raise ValueError("training manifest artifact entry must be an object")
    return artifact_type(**dict(value))


def run_benchmark(
    options: BenchmarkOptions,
    *,
    access_guard: OfficialValidationAssetAccessGuard,
    project_root: Path = PROJECT_ROOT,
) -> dict[str, Any]:
    """Run one frozen selection pass; all preflight work precedes Validation asset loading."""

    if not isinstance(options, BenchmarkOptions):
        raise TypeError("options must be BenchmarkOptions")
    if (
        not isinstance(access_guard, OfficialValidationAssetAccessGuard)
        or not access_guard._installed
    ):
        raise RuntimeError("the Validation audit hook must be installed before run_benchmark")

    # Project imports happen only after the formal subprocess has installed its audit hook.
    import jax
    import torch

    from controller_learning.config import load_project_config
    from controller_learning.rl.artifacts import (
        ArtifactRecord,
        TrainingRunIdentity,
        atomic_write_bytes,
        atomic_write_json,
        load_published_training_checkpoint,
        read_latest_checkpoint_pointer,
        read_strict_json,
        sha256_file,
    )
    from controller_learning.rl.configuration import load_ppo_config
    from controller_learning.rl.numpy_actor import (
        NUMPY_ACTOR_SCHEMA_VERSION,
        canonical_numpy_actor_bytes,
        numpy_actor_from_ppo_state_dict,
    )
    from controller_learning.rl.policy import PpoActorCritic
    from controller_learning.rl.selection import (
        FROZEN_CANDIDATE_UPDATES,
        FROZEN_WRAPPER_ORDER,
        SELECTION_REPORT_SCHEMA_VERSION,
        PolicySelectionResult,
        load_ppo_selection_config,
        rank_candidate_results,
        selection_gate_values,
        torch_state_dict_sha256,
        validate_selection_report,
    )
    from controller_learning.rl.validation_assets import load_verified_validation_pool

    root = Path(project_root).resolve(strict=True)
    config_path = _project_file(root, options.config, label="selection config")
    expected_config_path = (root / DEFAULT_SELECTION_CONFIG).resolve(strict=True)
    if config_path != expected_config_path:
        raise RuntimeError("formal selection requires configs/ppo_selection.toml exactly")
    selection_config = load_ppo_selection_config(config_path)
    lock_path = _project_file(root, Path("pixi.lock"), label="Pixi lock")
    training_config_path = _project_file(
        root,
        Path(selection_config.training_config),
        label="training config",
    )
    source = _source_snapshot(root)
    runtime, gpu_uuid = _runtime_evidence(torch)
    project = load_project_config(root)
    training_config = load_ppo_config(training_config_path)
    if (
        project.benchmark.version != selection_config.benchmark_version
        or training_config.environment.backend != selection_config.backend
        or training_config.environment.level_id != selection_config.level_id
        or training_config.update_count != FROZEN_CANDIDATE_UPDATES[-1]
        or training_config.checkpoint.interval_updates != 10
        or training_config.checkpoint.keep_last != len(FROZEN_CANDIDATE_UPDATES)
    ):
        raise RuntimeError("training/project configuration differs from the frozen selector")

    run_directory = root / selection_config.run_directory
    manifest = read_strict_json(run_directory, "manifest.json")
    if (
        manifest.get("schema_version") != TRAINING_MANIFEST_SCHEMA
        or manifest.get("status") != "complete"
    ):
        raise RuntimeError("selection requires one complete formal M7 training run")
    identity_value = manifest.get("run_identity")
    if not isinstance(identity_value, Mapping):
        raise RuntimeError("training manifest omits run_identity")
    identity = TrainingRunIdentity.from_dict(identity_value)
    if identity.run_id != selection_config.run_id:
        raise RuntimeError("training run identity differs from frozen selection config")
    if identity.source_revision != source["revision"]:
        raise RuntimeError("formal training run was not produced from the selected clean revision")
    if manifest.get("source", {}).get("revision") != identity.source_revision:
        raise RuntimeError("training manifest source differs from run identity")
    if sha256_file(training_config_path) != identity.configuration_sha256:
        raise RuntimeError("repository training config differs from formal run identity")
    if sha256_file(lock_path) != identity.lock_sha256:
        raise RuntimeError("Pixi lock differs from formal training run identity")
    run_config = run_directory / "config.toml"
    if run_config.is_symlink() or not run_config.is_file():
        raise RuntimeError("formal run config snapshot must be a regular non-symlink file")
    if run_config.read_bytes() != training_config_path.read_bytes():
        raise RuntimeError("formal run config snapshot differs from repository config")
    counts = manifest.get("counts")
    if not isinstance(counts, Mapping) or (
        counts.get("completed_updates") != FROZEN_CANDIDATE_UPDATES[-1]
        or counts.get("configured_budget_completed") is not True
    ):
        raise RuntimeError("formal training manifest did not complete the frozen update budget")
    training_assets = manifest.get("assets", {}).get("train_pool")
    if not isinstance(training_assets, Mapping) or (
        training_assets.get("manifest_sha256") != identity.train_manifest_sha256
        or training_assets.get("cache_file_sha256") != identity.train_cache_sha256
    ):
        raise RuntimeError("training manifest Track evidence differs from run identity")

    manifest_artifacts = manifest.get("artifacts")
    if not isinstance(manifest_artifacts, Mapping) or set(manifest_artifacts) != {
        "config",
        "final_checkpoint",
        "latest_checkpoint_pointer",
        "metrics_csv",
        "tensorboard_events",
    }:
        raise RuntimeError("training manifest omits artifacts")
    run_config_record = _artifact_from_mapping(
        manifest_artifacts.get("config"), artifact_type=ArtifactRecord
    )
    final_checkpoint_record = _artifact_from_mapping(
        manifest_artifacts.get("final_checkpoint"), artifact_type=ArtifactRecord
    )
    metrics_record = _artifact_from_mapping(
        manifest_artifacts.get("metrics_csv"), artifact_type=ArtifactRecord
    )
    if (
        run_config_record.relative_path != "config.toml"
        or metrics_record.relative_path != "metrics.csv"
    ):
        raise RuntimeError("training manifest uses noncanonical config or metrics paths")
    if (
        sha256_file(run_config) != run_config_record.sha256
        or run_config.stat().st_size != run_config_record.size_bytes
    ):
        raise RuntimeError("training config snapshot differs from the completed manifest")
    metrics_path = run_directory / metrics_record.relative_path
    if sha256_file(metrics_path) != metrics_record.sha256:
        raise RuntimeError("training metrics bytes differ from the completed manifest")
    pointer_record = _artifact_from_mapping(
        manifest_artifacts.get("latest_checkpoint_pointer"), artifact_type=ArtifactRecord
    )
    if pointer_record.relative_path != f"{selection_config.checkpoint_directory}/latest.json":
        raise RuntimeError("training manifest uses a noncanonical checkpoint pointer path")
    if sha256_file(run_directory / pointer_record.relative_path) != pointer_record.sha256:
        raise RuntimeError("latest checkpoint pointer differs from the completed manifest")
    pointer = read_latest_checkpoint_pointer(
        run_directory,
        checkpoint_directory=selection_config.checkpoint_directory,
    )
    if pointer is None or pointer.schema_version != 2:
        raise RuntimeError("selection requires the strict v2 retained-checkpoint ledger")
    if pointer.published_updates != selection_config.candidate_updates:
        raise RuntimeError("retained checkpoint ledger differs from the exact eight candidates")
    if final_checkpoint_record != pointer.checkpoint:
        raise RuntimeError("training manifest final checkpoint differs from the v2 ledger")

    loaded_candidates: list[Any] = []
    candidate_preflight: list[dict[str, Any]] = []
    for update in selection_config.candidate_updates:
        loaded = load_published_training_checkpoint(
            run_directory,
            expected_identity=identity,
            update_index=update,
            checkpoint_directory=selection_config.checkpoint_directory,
            torch_module=torch,
        )
        parameter_sha = torch_state_dict_sha256(
            loaded.payload["model_state_dict"],
            torch_module=torch,
        )
        inference_policy_bytes = canonical_numpy_actor_bytes(
            numpy_actor_from_ppo_state_dict(loaded.payload["model_state_dict"])
        )
        loaded_candidates.append(loaded)
        candidate_preflight.append(
            {
                "checkpoint": loaded.record.to_dict(),
                "inference_policy": {
                    "schema_version": NUMPY_ACTOR_SCHEMA_VERSION,
                    "sha256": hashlib.sha256(inference_policy_bytes).hexdigest(),
                    "size_bytes": len(inference_policy_bytes),
                },
                "parameter_sha256": parameter_sha,
                "update_index": update,
                "valid_transitions": loaded.metadata.valid_transitions,
                "vector_steps": loaded.metadata.vector_steps,
            }
        )

    manifest_path = run_directory / "manifest.json"
    pointer_path = run_directory / pointer_record.relative_path
    pre_evaluation_sha256 = {
        "latest_checkpoint_pointer": sha256_file(pointer_path),
        "pixi_lock": sha256_file(lock_path),
        "selection_config": sha256_file(config_path),
        "training_config": sha256_file(training_config_path),
        "training_manifest": sha256_file(manifest_path),
        "training_metrics": sha256_file(metrics_path),
        "training_run_config": sha256_file(run_config),
    }
    for preflight in candidate_preflight:
        update = preflight["update_index"]
        checkpoint_path = run_directory / preflight["checkpoint"]["relative_path"]
        digest = sha256_file(checkpoint_path)
        if digest != preflight["checkpoint"]["sha256"]:
            raise RuntimeError("candidate checkpoint changed after strict loading")
        pre_evaluation_sha256[f"checkpoint_update_{update:08d}"] = digest

    # Curve generation is based only on the already-bound Train metrics, before Validation opens.
    curve_series = _read_curve_metrics(
        metrics_path,
        expected_updates=FROZEN_CANDIDATE_UPDATES[-1],
    )
    curve_bytes = deterministic_training_curve_png(
        *curve_series,
        width_px=selection_config.training_curve_width_px,
        height_px=selection_config.training_curve_height_px,
        dpi=selection_config.training_curve_dpi,
    )
    pre_validation_access = access_guard.evidence(validation_loaded=False)
    # This is the sole point at which official Validation bytes may first be opened.
    access_guard.enable_validation_reads()
    validation = load_verified_validation_pool(project)
    expected_track_ids = tuple(int(value) for value in validation.pool.batch.seed)
    if len(expected_track_ids) != selection_config.validation_track_count:
        raise RuntimeError("Validation pool width differs from the frozen selector")

    stack = _build_stack(project, training_config, validation)
    memory = MemoryRecorder(
        torch=torch,
        device=stack.environment.device,
        jax_device=jax.devices("gpu")[0],
        gpu_uuid=gpu_uuid,
    )
    memory.sample("after_stack_build")
    candidate_results: list[Any] = []
    timings: list[dict[str, Any]] = []
    try:
        for loaded, preflight in zip(loaded_candidates, candidate_preflight, strict=True):
            update = loaded.metadata.update_index
            policy = PpoActorCritic(
                100,
                action_low=stack.base.single_action_space.low,
                action_high=stack.base.single_action_space.high,
                policy_seed=identity.policy_seed,
                initial_log_std=training_config.ppo.initial_log_std,
                hidden_sizes=training_config.ppo.hidden_sizes,
                device=stack.environment.device,
            )
            policy.load_state_dict(loaded.payload["model_state_dict"], strict=True)
            policy.requires_grad_(False)
            policy.eval()
            before = torch_state_dict_sha256(policy.state_dict(), torch_module=torch)
            if before != preflight["parameter_sha256"]:
                raise RuntimeError("loaded CUDA policy differs from checkpoint parameter SHA")
            started = time.perf_counter()
            rows = evaluate_first_terminal_rows(
                stack.environment,
                lambda features, selected=policy: selected.deterministic(features).action,
                expected_track_ids=expected_track_ids,
                reset_seed=selection_config.validation_reset_seed,
                max_vector_steps=selection_config.max_vector_steps,
                control_dt_s=project.vehicle.simulation.control_dt_s,
                torch_module=torch,
            )
            elapsed = time.perf_counter() - started
            after = torch_state_dict_sha256(policy.state_dict(), torch_module=torch)
            if any(parameter.grad is not None for parameter in policy.parameters()):
                raise RuntimeError(
                    "deterministic Validation selection created a parameter gradient"
                )
            candidate_results.append(
                PolicySelectionResult(
                    policy_kind="candidate",
                    policy_id=f"checkpoint_update_{update:08d}",
                    update_index=update,
                    parameter_sha256_before=before,
                    parameter_sha256_after=after,
                    rows=rows,
                )
            )
            timings.append(
                {"elapsed_seconds": elapsed, "policy_id": f"checkpoint_update_{update:08d}"}
            )
            del policy
        low = torch.as_tensor(
            stack.base.single_action_space.low,
            dtype=torch.float32,
            device=stack.environment.device,
        )
        high = torch.as_tensor(
            stack.base.single_action_space.high,
            dtype=torch.float32,
            device=stack.environment.device,
        )
        random_generator = torch.Generator(device=stack.environment.device).manual_seed(
            selection_config.random_baseline_seed
        )

        def random_actions(features: Any) -> Any:
            unit = torch.rand(
                (features.shape[0], 2),
                dtype=torch.float32,
                device=features.device,
                generator=random_generator,
            )
            return low + (high - low) * unit

        started = time.perf_counter()
        random_rows = evaluate_first_terminal_rows(
            stack.environment,
            random_actions,
            expected_track_ids=expected_track_ids,
            reset_seed=selection_config.validation_reset_seed,
            max_vector_steps=selection_config.max_vector_steps,
            control_dt_s=project.vehicle.simulation.control_dt_s,
            torch_module=torch,
        )
        timings.append(
            {"elapsed_seconds": time.perf_counter() - started, "policy_id": "random_seed_17"}
        )
        random_result = PolicySelectionResult(
            policy_kind="random_baseline",
            policy_id="random_seed_17",
            update_index=None,
            parameter_sha256_before=None,
            parameter_sha256_after=None,
            rows=random_rows,
        )
        memory.sample("after_evaluations")
    finally:
        stack.environment.close()
    memory.sample("after_environment_close")

    validation_evidence = _validation_evidence(validation.evidence)
    expected_post_sha256 = {
        **pre_evaluation_sha256,
        "validation_asset": validation_evidence["asset_file_sha256"],
        "validation_manifest": validation_evidence["manifest_sha256"],
    }
    post_evaluation_sha256 = {
        "latest_checkpoint_pointer": sha256_file(pointer_path),
        "pixi_lock": sha256_file(lock_path),
        "selection_config": sha256_file(config_path),
        "training_config": sha256_file(training_config_path),
        "training_manifest": sha256_file(manifest_path),
        "training_metrics": sha256_file(metrics_path),
        "training_run_config": sha256_file(run_config),
        "validation_asset": sha256_file(access_guard.validation_asset),
        "validation_manifest": sha256_file(access_guard.validation_manifest),
    }
    for preflight in candidate_preflight:
        update = preflight["update_index"]
        checkpoint_path = run_directory / preflight["checkpoint"]["relative_path"]
        post_evaluation_sha256[f"checkpoint_update_{update:08d}"] = sha256_file(checkpoint_path)
    if post_evaluation_sha256 != expected_post_sha256:
        changed = sorted(
            key
            for key in expected_post_sha256.keys() | post_evaluation_sha256.keys()
            if expected_post_sha256.get(key) != post_evaluation_sha256.get(key)
        )
        raise RuntimeError(
            "formal selection inputs changed during evaluation: " + ", ".join(changed)
        )
    post_input_source = _source_snapshot(root)
    if post_input_source != source:
        raise RuntimeError("source state changed during Validation evaluation")
    final_access = access_guard.evidence(validation_loaded=True)

    ranked = rank_candidate_results(candidate_results)
    selected = ranked[0]
    gates = selection_gate_values(selected, random_result, config=selection_config)
    selection_payload = {
        "candidate_updates_in_rank_order": [result.update_index for result in ranked],
        "mean_successful_lap_time_no_success_policy": "positive_infinity_worst",
        "ranking": selection_config.ranking,
        "selected_mean_successful_lap_time_s": selected.mean_successful_lap_time_s,
        "selected_success_count": selected.success_count,
        "selected_success_rate": selected.success_rate,
        "selected_update": selected.update_index,
    }

    # Generated artifacts remain in memory until every input is rehashed and the report validates.
    curve_record = ArtifactRecord(
        relative_path=selection_config.training_curve_path,
        sha256=hashlib.sha256(curve_bytes).hexdigest(),
        size_bytes=len(curve_bytes),
    )
    project_artifacts = {
        "pixi_lock": ArtifactRecord(
            relative_path=lock_path.relative_to(root).as_posix(),
            sha256=post_evaluation_sha256["pixi_lock"],
            size_bytes=lock_path.stat().st_size,
        ).to_dict(),
        "selection_config": ArtifactRecord(
            relative_path=config_path.relative_to(root).as_posix(),
            sha256=post_evaluation_sha256["selection_config"],
            size_bytes=config_path.stat().st_size,
        ).to_dict(),
        "training_config": ArtifactRecord(
            relative_path=training_config_path.relative_to(root).as_posix(),
            sha256=post_evaluation_sha256["training_config"],
            size_bytes=training_config_path.stat().st_size,
        ).to_dict(),
        "training_manifest": ArtifactRecord(
            relative_path=manifest_path.relative_to(root).as_posix(),
            sha256=post_evaluation_sha256["training_manifest"],
            size_bytes=manifest_path.stat().st_size,
        ).to_dict(),
        "training_run_config": ArtifactRecord(
            relative_path=run_config.relative_to(root).as_posix(),
            sha256=post_evaluation_sha256["training_run_config"],
            size_bytes=run_config.stat().st_size,
        ).to_dict(),
        "validation_asset": ArtifactRecord(
            relative_path=access_guard.validation_asset.relative_to(root).as_posix(),
            sha256=post_evaluation_sha256["validation_asset"],
            size_bytes=access_guard.validation_asset.stat().st_size,
        ).to_dict(),
        "validation_manifest": ArtifactRecord(
            relative_path=access_guard.validation_manifest.relative_to(root).as_posix(),
            sha256=post_evaluation_sha256["validation_manifest"],
            size_bytes=access_guard.validation_manifest.stat().st_size,
        ).to_dict(),
    }
    allowed_output_paths = [selection_config.report_path, selection_config.training_curve_path]
    pending_output_source = {
        "allowed_generated_output_paths": sorted(allowed_output_paths),
        "observed_changed_paths": [],
        "only_allowed_generated_outputs": True,
        "revision": source["revision"],
        "unexpected_changed_paths": [],
    }
    report = {
        "artifacts": {
            "latest_checkpoint_pointer": pointer_record.to_dict(),
            "metrics_csv": metrics_record.to_dict(),
            **project_artifacts,
            "training_curve": curve_record.to_dict(),
        },
        "asset_access": final_access,
        "configuration": _json_value(selection_config),
        "evaluations": {
            "candidates": [result.to_dict() for result in candidate_results],
            "random_baseline": random_result.to_dict(),
        },
        "gates": gates,
        "memory": memory.report(),
        "post_selection": {
            "controller_evaluation_status": "not_run",
            "export_status": "not_run",
        },
        "protocol": {
            "autoreset_mode": "NEXT_STEP",
            "backend": selection_config.backend,
            "benchmark_version": selection_config.benchmark_version,
            "candidate_count": len(selection_config.candidate_updates),
            "candidate_updates": list(selection_config.candidate_updates),
            "control_dt_s": project.vehicle.simulation.control_dt_s,
            "deterministic_candidate_actions": True,
            "first_terminal_event_only": True,
            "level_id": selection_config.level_id,
            "max_vector_steps": selection_config.max_vector_steps,
            "no_gradient_updates": True,
            "num_envs": selection_config.num_envs,
            "one_long_lived_environment": True,
            "random_baseline_seed": selection_config.random_baseline_seed,
            "reset_options_track_indices": "numpy.arange(100,dtype=int32)",
            "reward_wrapper_used": False,
            "same_reset_seed_and_track_order_for_every_policy": True,
            "test_accessed": False,
            "train_assets_accessed": False,
            "validation_reset_seed": selection_config.validation_reset_seed,
            "validation_track_count": selection_config.validation_track_count,
            "wrapper_order": list(FROZEN_WRAPPER_ORDER),
        },
        "runtime": {**runtime, "evaluation_timings": timings},
        "schema_version": SELECTION_REPORT_SCHEMA_VERSION,
        "selection": selection_payload,
        "source": {
            "input_stability": {
                "all_inputs_unchanged": True,
                "expected_post_sha256": dict(sorted(expected_post_sha256.items())),
                "post_evaluation_sha256": dict(sorted(post_evaluation_sha256.items())),
            },
            "post_input_check": post_input_source,
            "post_output_worktree": pending_output_source,
            "preflight": source,
        },
        "status": "passed" if gates["passed"] else "gate_failed",
        "training_run": {
            "candidate_checkpoints": candidate_preflight,
            "identity": identity.to_dict(),
            "manifest_sha256": post_evaluation_sha256["training_manifest"],
            "pre_validation_access": pre_validation_access,
            "run_directory": selection_config.run_directory,
        },
        "validation_assets": validation_evidence,
    }
    report = _json_value(report)
    validate_selection_report(report, config=selection_config)
    output_backups: dict[str, bytes | None] = {}
    for relative_path in allowed_output_paths:
        destination = root / relative_path
        if destination.is_symlink() or (destination.exists() and not destination.is_file()):
            raise RuntimeError("generated output destination is not a regular local file")
        output_backups[relative_path] = destination.read_bytes() if destination.exists() else None
    try:
        committed_curve = atomic_write_bytes(
            root,
            selection_config.training_curve_path,
            curve_bytes,
        )
        if committed_curve != curve_record:
            raise RuntimeError("training curve publication differs from its validated record")
        report_record = atomic_write_json(root, selection_config.report_path, report)
        report["source"]["post_output_worktree"] = _source_snapshot_allowing_outputs(
            root,
            expected_revision=source["revision"],
            allowed_paths=allowed_output_paths,
        )
        validate_selection_report(report, config=selection_config)
        report_record = atomic_write_json(root, selection_config.report_path, report)
        final_output_source = _source_snapshot_allowing_outputs(
            root,
            expected_revision=source["revision"],
            allowed_paths=allowed_output_paths,
        )
        if final_output_source != report["source"]["post_output_worktree"]:
            raise RuntimeError(
                "generated-output worktree evidence changed after report publication"
            )
    except BaseException:
        for relative_path, previous in output_backups.items():
            destination = root / relative_path
            if previous is None:
                destination.unlink(missing_ok=True)
            else:
                atomic_write_bytes(root, relative_path, previous)
        raise
    return {
        "gate_passed": gates["passed"],
        "random_success_count": random_result.success_count,
        "report": report_record.relative_path,
        "selected_success_count": selected.success_count,
        "selected_update": selected.update_index,
    }


def main(argv: Sequence[str] | None = None) -> None:
    guard = OfficialValidationAssetAccessGuard(
        official_track_root=PROJECT_ROOT / "controller_learning/assets/tracks",
        validation_manifest=(
            PROJECT_ROOT / "controller_learning/assets/tracks/v0.1/validation.json"
        ),
        validation_asset=(PROJECT_ROOT / "controller_learning/assets/tracks/v0.1/validation.npz"),
        track_cache_root=PROJECT_ROOT / ".track-cache",
    )
    # Install before argument/config parsing or any project import in the dedicated subprocess.
    guard.install()
    options = _parse_args(argv)
    result = run_benchmark(options, access_guard=guard)
    print(json.dumps(result, allow_nan=False, sort_keys=True))


if __name__ == "__main__":  # pragma: no cover - exercised through the Pixi task.
    main(sys.argv[1:])
