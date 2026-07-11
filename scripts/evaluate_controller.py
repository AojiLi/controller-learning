"""Run an informal Controller evaluation on Level 0 or Validation Tracks."""

from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import os
import re
import shutil
import subprocess
import sys
from collections.abc import Callable, Iterator, Mapping, Sequence
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal, TypeAlias

import numpy as np

from controller_learning.config import ConfigError, ProjectConfig, load_project_config
from controller_learning.control import EpisodeRunResult, run_controller_episode
from controller_learning.control.identity import (
    ControllerDirectoryIdentity,
    capture_controller_directory_identity,
)
from controller_learning.evaluation.controller import (
    ControllerEvaluation,
    EpisodeEvaluation,
    EvaluationBackend,
    TimingSummary,
    evaluate_track_batch,
)
from controller_learning.evaluation.trajectory import (
    EpisodeTrajectory,
    RecordedControllerEpisode,
    record_controller_episode,
    write_trajectory_json,
)
from controller_learning.rl.validation_assets import load_verified_validation_pool
from controller_learning.tracks.assets import (
    TrackAssetError,
    load_track_asset_manifest,
    sha256_file,
)
from controller_learning.tracks.official_assets import (
    load_verified_manifest_batch,
    official_track_asset_directory,
    official_track_split_spec,
    validate_official_manifest,
)
from controller_learning.tracks.pool import TrackPool
from controller_learning.tracks.types import TrackBatch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEVELOPMENT_EVALUATION_SCHEMA_VERSION = "controller-learning.development-evaluation.v1"
DEVELOPMENT_EPISODE_SCHEMA_VERSION = "controller-learning.development-episode.v1"
DEVELOPMENT_TRACK_SOURCE_SCHEMA_VERSION = "controller-learning.development-track-source.v1"
DEVELOPMENT_EVALUATION_KIND = "informal_development_evaluation"
DEFAULT_OUTPUT_ROOT = Path("runs/evaluations")

DevelopmentSplit: TypeAlias = Literal["level0", "validation"]
ControllerEvaluator: TypeAlias = Callable[..., ControllerEvaluation]
EpisodeRunner: TypeAlias = Callable[..., EpisodeRunResult]
TrajectoryRecorder: TypeAlias = Callable[..., RecordedControllerEpisode]

_RUN_ID_PATTERN = re.compile(r"[a-z0-9](?:[a-z0-9._-]{0,126}[a-z0-9])?")
_EPISODE_COLUMNS = (
    "schema_version",
    "run_id",
    "evaluation_kind",
    "benchmark_version",
    "split",
    "backend",
    "row_index",
    "track_id",
    "reset_seed",
    "success",
    "lap_time_s",
    "environment_steps",
    "total_reward",
    "terminated",
    "truncated",
    "termination_reason",
    "controller_import_time_s",
    "controller_init_time_s",
    "compute_sample_count",
    "compute_deadline_s",
    "compute_p50_s",
    "compute_p95_s",
    "compute_p99_s",
    "compute_max_s",
    "compute_deadline_miss_count",
    "compute_deadline_miss_rate",
)


class DevelopmentEvaluationError(RuntimeError):
    """A user-facing development-evaluation contract was violated."""


@dataclass(frozen=True, slots=True)
class DevelopmentEvaluationOptions:
    """Validated choices for one explicitly informal Controller evaluation."""

    controller_directory: Path
    run_id: str
    split: DevelopmentSplit
    backend: EvaluationBackend = "cpu_reference"
    count: int | None = None
    capture_row: int | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "controller_directory", Path(self.controller_directory))
        if not isinstance(self.run_id, str) or _RUN_ID_PATTERN.fullmatch(self.run_id) is None:
            raise ValueError("run_id must be lowercase, path-safe, and at most 128 characters")
        if self.split not in ("level0", "validation"):
            raise ValueError("split must be 'level0' or 'validation'")
        if self.backend not in ("cpu_reference", "mjx_warp"):
            raise ValueError("backend must be 'cpu_reference' or 'mjx_warp'")
        for name, value in (("count", self.count), ("capture_row", self.capture_row)):
            if value is not None and (
                type(value) is not int or value < (1 if name == "count" else 0)
            ):
                qualifier = "positive" if name == "count" else "non-negative"
                raise ValueError(f"{name} must be a {qualifier} integer or None")


@dataclass(frozen=True, slots=True)
class SourceSnapshot:
    """Path-sanitized Git identity for an informal local run."""

    revision: str
    worktree_clean: bool
    dirty_file_count: int
    status_sha256: str

    def to_dict(self) -> dict[str, str | int | bool]:
        """Return the public, path-sanitized source evidence."""

        return {
            "revision": self.revision,
            "worktree_clean": self.worktree_clean,
            "dirty_file_count": self.dirty_file_count,
            "status_sha256": self.status_sha256,
        }


@dataclass(frozen=True, slots=True)
class PreparedTracks:
    """Selected Track rows plus split-specific access evidence."""

    split: DevelopmentSplit
    level_id: int
    batch: TrackBatch
    generator_version: str
    track_pool: TrackPool | None
    available_track_count: int
    evidence: Mapping[str, object]


class SelectedTrajectoryRunner:
    """Capture one preselected row without executing a second rollout."""

    def __init__(
        self,
        capture_row: int,
        *,
        runner: EpisodeRunner = run_controller_episode,
        recorder: TrajectoryRecorder = record_controller_episode,
    ) -> None:
        if type(capture_row) is not int or capture_row < 0:
            raise ValueError("capture_row must be a non-negative integer")
        self.capture_row = capture_row
        self._runner = runner
        self._recorder = recorder
        self._next_row = 0
        self.trajectory: EpisodeTrajectory | None = None

    def __call__(
        self,
        env: object,
        controller_directory: str,
        reset_seed: int,
        *,
        reset_options: Mapping[str, Any] | None = None,
    ) -> EpisodeRunResult:
        row = self._next_row
        self._next_row += 1
        if reset_options is not None and int(reset_options.get("track_index", -1)) != row:
            raise RuntimeError("evaluation row order differs from reset_options.track_index")
        kwargs = {} if reset_options is None else {"reset_options": reset_options}
        if row == self.capture_row:
            recorded = self._recorder(
                env,
                controller_directory,
                reset_seed,
                **kwargs,
            )
            if self.trajectory is not None:
                raise RuntimeError("selected trajectory row was captured more than once")
            self.trajectory = recorded.trajectory
            return recorded.result
        return self._runner(env, controller_directory, reset_seed, **kwargs)


def _positive_integer(value: str) -> int:
    try:
        result = int(value)
    except ValueError as error:
        raise argparse.ArgumentTypeError("must be an integer") from error
    if result < 1:
        raise argparse.ArgumentTypeError("must be positive")
    return result


def _nonnegative_integer(value: str) -> int:
    try:
        result = int(value)
    except ValueError as error:
        raise argparse.ArgumentTypeError("must be an integer") from error
    if result < 0:
        raise argparse.ArgumentTypeError("must be non-negative")
    return result


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--controller",
        dest="controller_directory",
        type=Path,
        required=True,
        help="Trusted Controller plugin directory",
    )
    parser.add_argument(
        "--run-id",
        required=True,
        help="Lowercase identifier used under runs/evaluations/",
    )
    parser.add_argument(
        "--split",
        choices=("level0", "validation"),
        required=True,
        help="Development split; Test is intentionally unavailable",
    )
    parser.add_argument(
        "--backend",
        choices=("cpu_reference", "mjx_warp"),
        default="cpu_reference",
        help="Explicit vehicle backend (default: cpu_reference)",
    )
    parser.add_argument(
        "--count",
        type=_positive_integer,
        default=None,
        help="Evaluate the ordered manifest prefix (default: the complete selected split)",
    )
    parser.add_argument(
        "--capture-row",
        type=_nonnegative_integer,
        default=None,
        help="Capture one selected row trajectory from the measured rollout",
    )
    return parser


def _parse_args(argv: Sequence[str] | None = None) -> DevelopmentEvaluationOptions:
    values = _build_parser().parse_args(argv)
    return DevelopmentEvaluationOptions(
        controller_directory=values.controller_directory,
        run_id=values.run_id,
        split=values.split,
        backend=values.backend,
        count=values.count,
        capture_row=values.capture_row,
    )


def _run_git(project_root: Path, *arguments: str) -> str:
    try:
        completed = subprocess.run(
            ("git", *arguments),
            cwd=project_root,
            check=True,
            capture_output=True,
            text=True,
            timeout=15,
        )
    except (OSError, subprocess.SubprocessError) as error:
        raise DevelopmentEvaluationError(
            f"development evaluation requires readable Git source identity: {error}"
        ) from error
    return completed.stdout.strip()


def _source_snapshot(project_root: Path) -> SourceSnapshot:
    revision = _run_git(project_root, "rev-parse", "--verify", "HEAD")
    if re.fullmatch(r"[0-9a-f]{40}", revision) is None:
        raise DevelopmentEvaluationError("Git HEAD must be a full lowercase revision")
    status = _run_git(project_root, "status", "--porcelain", "--untracked-files=normal")
    dirty_count = 0 if not status else len(status.splitlines())
    return SourceSnapshot(
        revision=revision,
        worktree_clean=dirty_count == 0,
        dirty_file_count=dirty_count,
        status_sha256=hashlib.sha256(status.encode("utf-8")).hexdigest(),
    )


def _resolve_controller_directory(value: Path, *, project_root: Path) -> tuple[Path, str, bool]:
    source = value.expanduser()
    if not source.is_absolute():
        source = project_root / source
    if source.is_symlink():
        raise DevelopmentEvaluationError("Controller directory cannot be a symbolic link")
    try:
        directory = source.resolve(strict=True)
    except FileNotFoundError as error:
        raise DevelopmentEvaluationError(f"Controller directory does not exist: {value}") from error
    if not directory.is_dir():
        raise DevelopmentEvaluationError("Controller path must identify a directory")
    missing = tuple(
        name for name in ("controller.py", "config.toml") if not (directory / name).is_file()
    )
    if missing:
        raise DevelopmentEvaluationError(
            "Controller directory is missing required file(s): " + ", ".join(missing)
        )
    try:
        display = directory.relative_to(project_root).as_posix()
        external = False
    except ValueError:
        display = f"external:{directory.name}"
        external = True
    return directory, display, external


def _prefix_batch(batch: TrackBatch, count: int) -> TrackBatch:
    return TrackBatch(*(np.array(value[:count], copy=True) for value in batch))


def _stable_sha256(path: Path, *, label: str) -> str:
    try:
        before = sha256_file(path)
    except FileNotFoundError as error:
        raise TrackAssetError(f"{label} does not exist: {path.name}") from error
    after = sha256_file(path)
    if before != after:
        raise TrackAssetError(f"{label} changed while it was read")
    return before


def _load_level0_tracks(project_config: ProjectConfig) -> PreparedTracks:
    directory = official_track_asset_directory(project_config.benchmark.version)
    spec = official_track_split_spec("level0")
    if (
        spec.level_id != 0
        or spec.track_count != 1
        or spec.manifest_file != "level0.json"
        or spec.asset_file != "level0.npz"
        or spec.package_asset is not True
    ):
        raise TrackAssetError("Level 0 split contract differs from benchmark 0.1")
    manifest_path = directory / spec.manifest_file
    asset_path = directory / spec.asset_file
    manifest_sha256 = _stable_sha256(manifest_path, label="Level 0 manifest")
    manifest = load_track_asset_manifest(manifest_path)
    if _stable_sha256(manifest_path, label="Level 0 manifest") != manifest_sha256:
        raise TrackAssetError("Level 0 manifest changed while loading")
    validate_official_manifest(project_config, manifest)
    if manifest.split != "level0" or manifest.level_id != 0 or manifest.track_count != 1:
        raise TrackAssetError("Level 0 manifest differs from the fixed singleton contract")
    asset_sha256 = _stable_sha256(asset_path, label="Level 0 asset")
    if asset_sha256 != manifest.asset_sha256:
        raise TrackAssetError("Level 0 asset SHA-256 differs from its manifest")
    batch = load_verified_manifest_batch(manifest, asset_path)
    if _stable_sha256(asset_path, label="Level 0 asset") != asset_sha256:
        raise TrackAssetError("Level 0 asset changed while loading")
    evidence = {
        "schema_version": DEVELOPMENT_TRACK_SOURCE_SCHEMA_VERSION,
        "loaded_splits": ["level0"],
        "loader_accessed_train": False,
        "loader_accessed_validation": False,
        "loader_accessed_test": False,
        "manifest_file": manifest_path.name,
        "manifest_sha256": manifest_sha256,
        "asset_file": asset_path.name,
        "asset_sha256": asset_sha256,
        "available_track_count": 1,
    }
    return PreparedTracks(
        split="level0",
        level_id=0,
        batch=batch,
        generator_version=manifest.generator_version,
        track_pool=None,
        available_track_count=1,
        evidence=evidence,
    )


def _load_validation_tracks(
    project_config: ProjectConfig,
    *,
    count: int | None,
) -> PreparedTracks:
    verified = load_verified_validation_pool(project_config)
    available = verified.pool.size
    selected_count = available if count is None else count
    if selected_count > available:
        raise DevelopmentEvaluationError(
            f"count cannot exceed the {available}-Track Validation split"
        )
    batch = _prefix_batch(verified.pool.batch, selected_count)
    pool = TrackPool(
        benchmark_version=verified.pool.benchmark_version,
        generator_version=verified.pool.generator_version,
        split="validation",
        batch=batch,
    )
    source = verified.evidence
    evidence = {
        "schema_version": DEVELOPMENT_TRACK_SOURCE_SCHEMA_VERSION,
        "loaded_splits": list(source.loaded_splits),
        "loader_accessed_train": source.loader_accessed_train,
        "loader_accessed_validation": True,
        "loader_accessed_test": source.loader_accessed_test,
        "manifest_file": source.manifest_file,
        "manifest_sha256": source.manifest_sha256,
        "asset_file": source.asset_file,
        "asset_sha256": source.asset_file_sha256,
        "available_track_count": source.track_count,
        "track_ids_sha256": source.track_ids_sha256,
        "geometry_hashes_sha256": source.geometry_hashes_sha256,
    }
    return PreparedTracks(
        split="validation",
        level_id=1,
        batch=batch,
        generator_version=verified.pool.generator_version,
        track_pool=pool,
        available_track_count=available,
        evidence=evidence,
    )


def _prepare_tracks(
    project_config: ProjectConfig,
    options: DevelopmentEvaluationOptions,
) -> PreparedTracks:
    if options.split == "level0":
        if options.count not in (None, 1):
            raise DevelopmentEvaluationError("Level 0 contains exactly one Track")
        return _load_level0_tracks(project_config)
    return _load_validation_tracks(project_config, count=options.count)


def _timing_payload(timing: TimingSummary) -> dict[str, int | float]:
    return {
        "sample_count": timing.sample_count,
        "deadline_s": timing.deadline_s,
        "p50_s": timing.p50_s,
        "p95_s": timing.p95_s,
        "p99_s": timing.p99_s,
        "max_s": timing.max_s,
        "deadline_miss_count": timing.deadline_miss_count,
        "deadline_miss_rate": timing.deadline_miss_rate,
    }


def _episode_row(
    episode: EpisodeEvaluation,
    *,
    options: DevelopmentEvaluationOptions,
    benchmark_version: str,
) -> dict[str, object]:
    timing = episode.compute_timing
    return {
        "schema_version": DEVELOPMENT_EPISODE_SCHEMA_VERSION,
        "run_id": options.run_id,
        "evaluation_kind": DEVELOPMENT_EVALUATION_KIND,
        "benchmark_version": benchmark_version,
        "split": options.split,
        "backend": options.backend,
        "row_index": episode.track_index,
        "track_id": episode.track_id,
        "reset_seed": episode.reset_seed,
        "success": episode.success,
        "lap_time_s": "" if episode.lap_time_s is None else episode.lap_time_s,
        "environment_steps": episode.steps,
        "total_reward": episode.total_reward,
        "terminated": episode.terminated,
        "truncated": episode.truncated,
        "termination_reason": episode.termination_reason,
        "controller_import_time_s": episode.controller_import_time_s,
        "controller_init_time_s": episode.controller_init_time_s,
        "compute_sample_count": timing.sample_count,
        "compute_deadline_s": timing.deadline_s,
        "compute_p50_s": timing.p50_s,
        "compute_p95_s": timing.p95_s,
        "compute_p99_s": timing.p99_s,
        "compute_max_s": timing.max_s,
        "compute_deadline_miss_count": timing.deadline_miss_count,
        "compute_deadline_miss_rate": timing.deadline_miss_rate,
    }


def _episodes_csv_bytes(
    evaluation: ControllerEvaluation,
    *,
    options: DevelopmentEvaluationOptions,
    benchmark_version: str,
) -> bytes:
    output = io.StringIO(newline="")
    writer = csv.DictWriter(output, fieldnames=_EPISODE_COLUMNS, lineterminator="\n")
    writer.writeheader()
    for episode in evaluation.episodes:
        writer.writerow(_episode_row(episode, options=options, benchmark_version=benchmark_version))
    return output.getvalue().encode("utf-8")


def _write_new_file(path: Path, content: bytes) -> dict[str, str | int]:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("xb") as stream:
        stream.write(content)
        stream.flush()
        os.fsync(stream.fileno())
    path.chmod(0o644)
    return {
        "path": path.name,
        "sha256": hashlib.sha256(content).hexdigest(),
        "size_bytes": len(content),
    }


def _canonical_json_bytes(value: object) -> bytes:
    return (
        json.dumps(
            value,
            allow_nan=False,
            ensure_ascii=True,
            separators=(",", ":"),
            sort_keys=True,
        )
        + "\n"
    ).encode("ascii")


@contextmanager
def _output_transaction(
    project_root: Path,
    run_id: str,
) -> Iterator[tuple[Path, Path]]:
    runs = project_root / "runs"
    if runs.is_symlink():
        raise DevelopmentEvaluationError("runs output root cannot be a symbolic link")
    runs.mkdir(exist_ok=True)
    base = runs / "evaluations"
    if base.is_symlink():
        raise DevelopmentEvaluationError("evaluation output root cannot be a symbolic link")
    base.mkdir(exist_ok=True)
    final = base / run_id
    staging = base / f".{run_id}.staging"
    if final.exists() or final.is_symlink():
        raise DevelopmentEvaluationError(
            f"evaluation run already exists: {final.relative_to(project_root)}"
        )
    if staging.exists() or staging.is_symlink():
        raise DevelopmentEvaluationError(
            f"staging directory already exists: {staging.relative_to(project_root)}"
        )
    staging.mkdir(mode=0o700)
    published = False
    try:
        yield staging, final
        if final.exists() or final.is_symlink():
            raise DevelopmentEvaluationError("evaluation destination appeared before publication")
        staging.rename(final)
        published = True
    finally:
        if not published and staging.exists() and not staging.is_symlink():
            shutil.rmtree(staging)


def _progress_callback(total: int) -> Callable[[EpisodeEvaluation], None]:
    def report(episode: EpisodeEvaluation) -> None:
        print(
            json.dumps(
                {
                    "completed": episode.track_index + 1,
                    "success": episode.success,
                    "total": total,
                    "track_id": episode.track_id,
                },
                allow_nan=False,
                sort_keys=True,
                separators=(",", ":"),
            ),
            file=sys.stderr,
            flush=True,
        )

    return report


def _summary_payload(
    *,
    options: DevelopmentEvaluationOptions,
    project_config: ProjectConfig,
    prepared: PreparedTracks,
    evaluation: ControllerEvaluation,
    controller_display: str,
    controller_external: bool,
    controller_identity: ControllerDirectoryIdentity,
    source: SourceSnapshot,
    episode_artifact: Mapping[str, object],
    trajectory_artifact: Mapping[str, object] | None,
) -> dict[str, object]:
    track_ids = [int(value) for value in prepared.batch.seed]
    outputs: dict[str, object] = {"episodes_csv": dict(episode_artifact)}
    if trajectory_artifact is not None:
        outputs["selected_trajectory"] = dict(trajectory_artifact)
    return {
        "schema_version": DEVELOPMENT_EVALUATION_SCHEMA_VERSION,
        "status": "completed",
        "evaluation_kind": DEVELOPMENT_EVALUATION_KIND,
        "formal_benchmark_result": False,
        "comparable_to_accepted_test_result": False,
        "run_id": options.run_id,
        "benchmark_version": project_config.benchmark.version,
        "split": options.split,
        "level_id": prepared.level_id,
        "backend": options.backend,
        "backend_scope": (
            "cpu_development_reference"
            if options.backend == "cpu_reference"
            else "mjx_warp_informal_development"
        ),
        "source": source.to_dict(),
        "controller": {
            "directory": controller_display,
            "external_directory": controller_external,
            **controller_identity.to_dict(),
        },
        "track_source": dict(prepared.evidence),
        "track_selection": {
            "selection_rule": "complete_split"
            if len(track_ids) == prepared.available_track_count
            else "manifest_order_prefix",
            "available_track_count": prepared.available_track_count,
            "selected_track_count": len(track_ids),
            "track_ids": track_ids,
            "reset_seed_rule": "row_index_uint32",
            "reset_seeds": list(range(len(track_ids))),
            "capture_row": options.capture_row,
        },
        "result": {
            "track_count": evaluation.track_count,
            "success_count": evaluation.success_count,
            "success_rate": evaluation.success_rate,
            "mean_successful_lap_time_s": evaluation.mean_successful_lap_time_s,
            "compute_timing": _timing_payload(evaluation.compute_timing),
        },
        "outputs": outputs,
    }


def run_development_evaluation(
    options: DevelopmentEvaluationOptions,
    *,
    project_root: str | Path = PROJECT_ROOT,
    evaluator: ControllerEvaluator = evaluate_track_batch,
) -> dict[str, object]:
    """Execute and transactionally publish one informal development evaluation."""

    if not isinstance(options, DevelopmentEvaluationOptions):
        raise TypeError("options must be DevelopmentEvaluationOptions")
    root = Path(project_root).expanduser().resolve(strict=True)
    if not root.is_dir():
        raise DevelopmentEvaluationError("project_root must identify a directory")

    with _output_transaction(root, options.run_id) as (staging, final):
        project_config = load_project_config(root)
        if project_config.benchmark.version != "0.1":
            raise DevelopmentEvaluationError(
                "development evaluation requires benchmark version 0.1"
            )
        controller, controller_display, controller_external = _resolve_controller_directory(
            options.controller_directory,
            project_root=root,
        )
        controller_identity_before = capture_controller_directory_identity(controller)
        source_before = _source_snapshot(root)
        prepared = _prepare_tracks(project_config, options)
        selected_count = int(prepared.batch.seed.shape[0])
        if options.capture_row is not None and options.capture_row >= selected_count:
            raise DevelopmentEvaluationError(
                f"capture_row must be smaller than the selected Track count {selected_count}"
            )

        selected_runner = (
            None if options.capture_row is None else SelectedTrajectoryRunner(options.capture_row)
        )
        evaluator_kwargs: dict[str, object] = {
            "track_pool": prepared.track_pool,
            "progress_callback": _progress_callback(selected_count),
        }
        if selected_runner is not None:
            evaluator_kwargs["run_episode"] = selected_runner
        evaluation = evaluator(
            project_config,
            prepared.level_id,
            prepared.batch,
            prepared.generator_version,
            controller,
            options.backend,
            reset_seeds=np.arange(selected_count, dtype=np.uint32),
            **evaluator_kwargs,
        )
        if (
            evaluation.track_count != selected_count
            or evaluation.level_id != prepared.level_id
            or evaluation.backend != options.backend
        ):
            raise DevelopmentEvaluationError("evaluator result differs from the requested workload")

        controller_identity_after = capture_controller_directory_identity(controller)
        if controller_identity_after != controller_identity_before:
            raise DevelopmentEvaluationError("Controller directory changed during evaluation")
        source_after = _source_snapshot(root)
        if source_after != source_before:
            raise DevelopmentEvaluationError("Git source state changed during evaluation")

        episode_bytes = _episodes_csv_bytes(
            evaluation,
            options=options,
            benchmark_version=project_config.benchmark.version,
        )
        episode_artifact = _write_new_file(staging / "episodes.csv", episode_bytes)
        episode_artifact["path"] = "episodes.csv"

        trajectory_artifact: dict[str, object] | None = None
        if selected_runner is not None:
            if selected_runner.trajectory is None:
                raise DevelopmentEvaluationError("the selected trajectory row was not captured")
            trajectory_path = (
                staging / "selected_replays" / (f"row_{options.capture_row:03d}_trajectory.json")
            )
            artifact = write_trajectory_json(selected_runner.trajectory, trajectory_path)
            trajectory_artifact = {
                "path": trajectory_path.relative_to(staging).as_posix(),
                "sha256": artifact.sha256,
                "size_bytes": artifact.size_bytes,
                "row_index": options.capture_row,
                "captured_from_evaluation_rollout": True,
            }

        summary = _summary_payload(
            options=options,
            project_config=project_config,
            prepared=prepared,
            evaluation=evaluation,
            controller_display=controller_display,
            controller_external=controller_external,
            controller_identity=controller_identity_before,
            source=source_before,
            episode_artifact=episode_artifact,
            trajectory_artifact=trajectory_artifact,
        )
        _write_new_file(staging / "summary.json", _canonical_json_bytes(summary))

    return {
        "status": "completed",
        "evaluation_kind": DEVELOPMENT_EVALUATION_KIND,
        "run_id": options.run_id,
        "output": final.relative_to(root).as_posix(),
        "success_count": evaluation.success_count,
        "track_count": evaluation.track_count,
        "success_rate": evaluation.success_rate,
    }


def main(argv: Sequence[str] | None = None) -> None:
    """Parse arguments, execute the informal workload, and print its result location."""

    parser = _build_parser()
    values = parser.parse_args(argv)
    try:
        options = DevelopmentEvaluationOptions(
            controller_directory=values.controller_directory,
            run_id=values.run_id,
            split=values.split,
            backend=values.backend,
            count=values.count,
            capture_row=values.capture_row,
        )
        result = run_development_evaluation(options)
    except (ConfigError, DevelopmentEvaluationError, TrackAssetError, ValueError) as error:
        parser.exit(2, f"evaluate-controller: error: {error}\n")
    print(json.dumps(result, allow_nan=False, sort_keys=True, separators=(",", ":")))


if __name__ == "__main__":
    main()
