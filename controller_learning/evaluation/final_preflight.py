"""Asset-free immutable input and source preflight for the M8 formal run."""

from __future__ import annotations

import hashlib
import json
import os
import secrets
import stat
import subprocess
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from types import MappingProxyType
from typing import Any, Final

from controller_learning.evaluation.controller_identity import (
    FrozenControllerIdentity,
    capture_frozen_controller_identity,
)
from controller_learning.evaluation.final_benchmark import (
    M8_CONTROLLER_ORDER,
    M8FinalEvaluationConfig,
)

M8_CONTROLLER_SNAPSHOT_RELATIVE_PATH: Final = "runs/m8_final_controller_snapshot"
M8_COMMITTED_CONTROLLER_SNAPSHOT_RELATIVE_PATH: Final = (
    "runs/m8_final_controller_snapshot.committed"
)
M8_ABORTED_CONTROLLER_SNAPSHOT_PREFIX: Final = "m8_final_controller_snapshot.abort."

CommandRunner = Callable[[Sequence[str], Path], str]


@dataclass(frozen=True, slots=True)
class CleanSourceSnapshot:
    """One full clean Git revision used by formal M8."""

    revision: str
    worktree_clean: bool

    def __post_init__(self) -> None:
        if (
            len(self.revision) != 40
            or any(character not in "0123456789abcdef" for character in self.revision)
            or self.worktree_clean is not True
        ):
            raise ValueError("formal source must be one clean lowercase full Git revision")

    def to_dict(self) -> dict[str, object]:
        return {"revision": self.revision, "worktree_clean": self.worktree_clean}


@dataclass(frozen=True, slots=True)
class FrozenInputReport:
    """Stable bytes and parsed payload for one required historical report."""

    name: str
    relative_path: str
    sha256: str
    size_bytes: int
    payload: Mapping[str, Any]


@dataclass(frozen=True, slots=True)
class FrozenControllerSnapshot:
    """Read-only whole-plugin copies used by all 60 formal Runner invocations."""

    root: Path
    directories: Mapping[str, Path]
    identities: Mapping[str, FrozenControllerIdentity]

    def __post_init__(self) -> None:
        if set(self.directories) != set(M8_CONTROLLER_ORDER) or set(self.identities) != set(
            M8_CONTROLLER_ORDER
        ):
            raise ValueError("Controller snapshot must cover exactly pid, mpc, and ppo")
        object.__setattr__(self, "directories", MappingProxyType(dict(self.directories)))
        object.__setattr__(self, "identities", MappingProxyType(dict(self.identities)))


def _default_command_runner(command: Sequence[str], cwd: Path) -> str:
    try:
        completed = subprocess.run(
            tuple(command),
            cwd=cwd,
            check=True,
            capture_output=True,
            text=True,
            timeout=30,
        )
    except (OSError, subprocess.SubprocessError) as error:
        raise RuntimeError(f"preflight command failed: {' '.join(command)}") from error
    return completed.stdout.rstrip("\r\n")


def capture_clean_source(
    project_root: str | Path,
    *,
    command_runner: CommandRunner = _default_command_runner,
) -> CleanSourceSnapshot:
    """Require a clean repository before any Test-bound state can be created."""

    root = Path(project_root).resolve(strict=True)
    revision = command_runner(("git", "rev-parse", "--verify", "HEAD"), root)
    status = command_runner(
        ("git", "status", "--porcelain", "--untracked-files=normal"),
        root,
    )
    if status:
        raise RuntimeError("formal M8 evaluation requires a clean Git worktree")
    return CleanSourceSnapshot(revision=revision, worktree_clean=True)


def sha256_regular_file(path: str | Path) -> tuple[str, int, bytes]:
    """Read one non-symlink regular file only when metadata remains stable."""

    source = Path(path)
    before = source.lstat()
    if stat.S_ISLNK(before.st_mode) or not stat.S_ISREG(before.st_mode):
        raise ValueError("formal input must be a non-symlink regular file")
    descriptor = os.open(source, os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0))
    try:
        opened = os.fstat(descriptor)
        with os.fdopen(descriptor, "rb", closefd=False) as stream:
            content = stream.read()
    finally:
        os.close(descriptor)
    after = source.lstat()
    fields = ("st_dev", "st_ino", "st_mode", "st_size", "st_mtime_ns")
    if any(getattr(before, name) != getattr(opened, name) for name in fields) or any(
        getattr(opened, name) != getattr(after, name) for name in fields
    ):
        raise RuntimeError("formal input changed while it was read")
    if len(content) != opened.st_size:
        raise RuntimeError("formal input size changed while it was read")
    return hashlib.sha256(content).hexdigest(), len(content), content


def _strict_json_object(content: bytes, *, name: str) -> Mapping[str, Any]:
    def reject_constant(value: str) -> None:
        raise ValueError(f"{name} contains forbidden JSON constant {value}")

    def reject_duplicates(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in pairs:
            if key in result:
                raise ValueError(f"{name} contains duplicate JSON key {key!r}")
            result[key] = value
        return result

    try:
        value = json.loads(
            content,
            parse_constant=reject_constant,
            object_pairs_hook=reject_duplicates,
        )
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError(f"{name} is not strict JSON") from error
    if not isinstance(value, Mapping):
        raise ValueError(f"{name} must contain a JSON object")
    return value


def _all_checks_passed(report: Mapping[str, Any], *, name: str) -> None:
    checks = report.get("checks")
    if not isinstance(checks, list) or not checks:
        raise ValueError(f"{name} must contain non-empty formal checks")
    if any(
        not isinstance(check, Mapping)
        or check.get("passed") is not True
        or not isinstance(check.get("id"), str)
        for check in checks
    ):
        raise ValueError(f"{name} contains a failed or malformed formal check")


def load_frozen_input_reports(
    project_root: str | Path,
    config: M8FinalEvaluationConfig,
) -> Mapping[str, FrozenInputReport]:
    """Load and semantically validate the five pre-Test M5/M6/M7 reports."""

    if not isinstance(config, M8FinalEvaluationConfig):
        raise TypeError("config must be an M8FinalEvaluationConfig")
    root = Path(project_root).resolve(strict=True)
    reports: dict[str, FrozenInputReport] = {}
    payloads: dict[str, Mapping[str, Any]] = {}
    for name, relative in config.input_paths.items():
        path = root / relative
        digest, size, content = sha256_regular_file(path)
        payload = _strict_json_object(content, name=name)
        reports[name] = FrozenInputReport(name, relative, digest, size, payload)
        payloads[name] = payload

    m5 = payloads["m5_track_admission_report"]
    if m5.get("schema_version") != 1 or m5.get("status") != "pass":
        raise ValueError("M5 Track admission report did not pass its frozen schema")
    _all_checks_passed(m5, name="M5 Track admission report")
    artifacts = m5.get("artifacts")
    readback = m5.get("artifact_readback")
    if not isinstance(artifacts, Mapping) or not isinstance(readback, Mapping):
        raise ValueError("M5 Track admission report lacks artifact identity")
    test_artifact = artifacts.get("test")
    if not isinstance(test_artifact, Mapping) or (
        test_artifact.get("manifest_file") != "test.json"
        or test_artifact.get("asset_file") != "test.npz"
        or test_artifact.get("manifest_sha256") != config.test_manifest_sha256
        or test_artifact.get("asset_sha256") != config.test_asset_sha256
        or test_artifact.get("storage") != "repository_asset"
    ):
        raise ValueError("M5 report does not bind the frozen official Test assets")
    if (
        not isinstance(readback.get("manifest_files_sha256"), Mapping)
        or not isinstance(readback.get("asset_files_sha256"), Mapping)
        or readback["manifest_files_sha256"].get("test") != config.test_manifest_sha256
        or readback["asset_files_sha256"].get("test") != config.test_asset_sha256
        or readback.get("passed") is not True
    ):
        raise ValueError("M5 Test artifact readback differs from the frozen identity")

    m6 = payloads["m6_report"]
    if (
        m6.get("schema_version") != "controller-learning.m6-controllers.v1"
        or m6.get("status") != "pass"
    ):
        raise ValueError("M6 Controller report did not pass its frozen schema")
    _all_checks_passed(m6, name="M6 Controller report")
    m6_configs = m6.get("controller_configs")
    if not isinstance(m6_configs, Mapping):
        raise ValueError("M6 report lacks frozen Controller config identities")
    for name in ("pid", "mpc"):
        value = m6_configs.get(name)
        if (
            not isinstance(value, Mapping)
            or value.get("directory") != f"controllers/{name}"
            or value.get("config_file") != f"controllers/{name}/config.toml"
            or value.get("config_sha256") != config.controller_config_sha256[name]
        ):
            raise ValueError(f"M6 {name.upper()} config identity differs from M8")

    from controller_learning.rl.controller_benchmark import (
        load_ppo_controller_evaluation_config,
        validate_controller_evaluation_report,
    )
    from controller_learning.rl.export_protocol import validate_export_report
    from controller_learning.rl.selection import (
        load_ppo_selection_config,
        validate_selection_report,
    )

    validate_selection_report(
        payloads["m7_selection_report"],
        config=load_ppo_selection_config(root / "configs/ppo_selection.toml"),
    )
    validate_export_report(payloads["m7_export_report"])
    export_controller = payloads["m7_export_report"].get("controller")
    if not isinstance(export_controller, Mapping) or not isinstance(
        export_controller.get("artifacts"), Mapping
    ):
        raise ValueError("M7 export report lacks PPO plugin artifacts")
    export_config = export_controller["artifacts"].get("config")
    if (
        export_controller.get("plugin_directory") != "controllers/ppo"
        or not isinstance(export_config, Mapping)
        or export_config.get("sha256") != config.controller_config_sha256["ppo"]
    ):
        raise ValueError("M7 PPO config identity differs from M8")
    validate_controller_evaluation_report(
        payloads["m7_controller_report"],
        config=load_ppo_controller_evaluation_config(
            root / "configs/ppo_controller_evaluation.toml"
        ),
    )
    return MappingProxyType(reports)


def frozen_input_digest(reports: Mapping[str, FrozenInputReport]) -> str:
    """Bind the complete ordered report path/hash/size set into one transaction identity."""

    if set(reports) != {
        "m5_track_admission_report",
        "m6_report",
        "m7_selection_report",
        "m7_export_report",
        "m7_controller_report",
    }:
        raise ValueError("frozen input reports are incomplete")
    rows = [
        {
            "name": name,
            "relative_path": reports[name].relative_path,
            "sha256": reports[name].sha256,
            "size_bytes": reports[name].size_bytes,
        }
        for name in sorted(reports)
    ]
    content = json.dumps(rows, ensure_ascii=True, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(content).hexdigest()


def _write_snapshot_file_at(parent_descriptor: int, name: str, content: bytes) -> None:
    flags = (
        os.O_WRONLY
        | os.O_CREAT
        | os.O_EXCL
        | getattr(os, "O_CLOEXEC", 0)
        | getattr(os, "O_NOFOLLOW", 0)
    )
    descriptor = os.open(name, flags, 0o444, dir_fd=parent_descriptor)
    try:
        os.fchmod(descriptor, 0o444)
        view = memoryview(content)
        while view:
            written = os.write(descriptor, view)
            if written < 1:
                raise OSError("Controller snapshot write made no progress")
            view = view[written:]
        os.fsync(descriptor)
        metadata = os.fstat(descriptor)
        if not stat.S_ISREG(metadata.st_mode) or metadata.st_size != len(content):
            raise RuntimeError("Controller snapshot file failed exact descriptor readback")
    finally:
        os.close(descriptor)
    os.fsync(parent_descriptor)


def _ensure_runs_directory_descriptor(project_root: Path) -> int:
    root_metadata = project_root.lstat()
    if stat.S_ISLNK(root_metadata.st_mode) or not stat.S_ISDIR(root_metadata.st_mode):
        raise RuntimeError("project root must be a real directory")
    flags = (
        os.O_RDONLY
        | getattr(os, "O_CLOEXEC", 0)
        | getattr(os, "O_DIRECTORY", 0)
        | getattr(os, "O_NOFOLLOW", 0)
    )
    root_descriptor = os.open(project_root, flags)
    try:
        if not _same_directory_inode(root_metadata, os.fstat(root_descriptor)):
            raise RuntimeError("project root changed while it was opened")
        try:
            runs_metadata = os.stat("runs", dir_fd=root_descriptor, follow_symlinks=False)
        except FileNotFoundError:
            os.mkdir("runs", 0o700, dir_fd=root_descriptor)
            os.fsync(root_descriptor)
        else:
            if stat.S_ISLNK(runs_metadata.st_mode) or not stat.S_ISDIR(runs_metadata.st_mode):
                raise RuntimeError("the formal runs path must be a real directory")
        return _open_real_child_directory(root_descriptor, "runs")
    finally:
        os.close(root_descriptor)


def create_frozen_controller_snapshot(
    project_root: str | Path,
    config: M8FinalEvaluationConfig,
    *,
    relative_path: str = M8_CONTROLLER_SNAPSHOT_RELATIVE_PATH,
) -> FrozenControllerSnapshot:
    """Copy exact precommitted plugins to one read-only ignored runtime tree."""

    if not isinstance(config, M8FinalEvaluationConfig):
        raise TypeError("config must be an M8FinalEvaluationConfig")
    root = Path(project_root).resolve(strict=True)
    relative = PurePosixPath(relative_path)
    if (
        not isinstance(relative_path, str)
        or relative.is_absolute()
        or relative.as_posix() != relative_path
        or len(relative.parts) != 2
        or relative.parts[0] != "runs"
        or any(part in {"", ".", ".."} for part in relative.parts)
    ):
        raise ValueError("Controller snapshot must be a normalized child of ignored runs/")
    snapshot_root = root / relative_path
    runs_descriptor = _ensure_runs_directory_descriptor(root)
    directory_descriptors: dict[tuple[str, ...], int] = {}
    try:
        os.mkdir(relative.parts[1], 0o700, dir_fd=runs_descriptor)
        os.fsync(runs_descriptor)
        active_descriptor = _open_real_child_directory(
            runs_descriptor,
            relative.parts[1],
        )
        directory_descriptors[()] = active_descriptor
        for name in M8_CONTROLLER_ORDER:
            identity = capture_frozen_controller_identity(root, name)
            if (
                identity.aggregate_sha256 != config.controller_aggregate_sha256[name]
                or identity.config_sha256 != config.controller_config_sha256[name]
            ):
                raise RuntimeError(f"Controller {name!r} differs from the committed M8 identity")
            for file in identity.files:
                source = root / identity.directory / file.path
                digest, size, content = sha256_regular_file(source)
                if digest != file.sha256 or size != file.size_bytes:
                    raise RuntimeError("Controller changed while its snapshot was created")
                destination = PurePosixPath(identity.directory) / file.path
                parent_parts = destination.parent.parts
                current_parts: tuple[str, ...] = ()
                for part in parent_parts:
                    next_parts = (*current_parts, part)
                    if next_parts not in directory_descriptors:
                        os.mkdir(
                            part,
                            0o700,
                            dir_fd=directory_descriptors[current_parts],
                        )
                        os.fsync(directory_descriptors[current_parts])
                        directory_descriptors[next_parts] = _open_real_child_directory(
                            directory_descriptors[current_parts],
                            part,
                        )
                    current_parts = next_parts
                _write_snapshot_file_at(
                    directory_descriptors[current_parts],
                    destination.name,
                    content,
                )
        for parts in sorted(directory_descriptors, key=len, reverse=True):
            descriptor = directory_descriptors[parts]
            os.fchmod(descriptor, 0o555)
            os.fsync(descriptor)
        os.fsync(runs_descriptor)
    finally:
        for descriptor in reversed(tuple(directory_descriptors.values())):
            os.close(descriptor)
        os.close(runs_descriptor)

    directories = {name: snapshot_root / "controllers" / name for name in M8_CONTROLLER_ORDER}
    identities: dict[str, FrozenControllerIdentity] = {}
    for name in M8_CONTROLLER_ORDER:
        snapshotted = capture_frozen_controller_identity(snapshot_root, name)
        if snapshotted != capture_frozen_controller_identity(root, name):
            raise RuntimeError("Controller snapshot content identity differs from source")
        identities[name] = snapshotted
    return FrozenControllerSnapshot(snapshot_root, directories, identities)


def validate_frozen_controller_snapshot(
    snapshot: FrozenControllerSnapshot,
    config: M8FinalEvaluationConfig,
) -> None:
    """Fail closed if any runtime Controller snapshot byte changed."""

    if not isinstance(snapshot, FrozenControllerSnapshot):
        raise TypeError("snapshot must be a FrozenControllerSnapshot")
    for name in M8_CONTROLLER_ORDER:
        observed = capture_frozen_controller_identity(snapshot.root, name)
        if (
            observed != snapshot.identities[name]
            or observed.aggregate_sha256 != config.controller_aggregate_sha256[name]
            or observed.config_sha256 != config.controller_config_sha256[name]
        ):
            raise RuntimeError(f"Controller {name!r} runtime snapshot changed")


def _same_directory_inode(left: os.stat_result, right: os.stat_result) -> bool:
    return (
        left.st_dev == right.st_dev
        and left.st_ino == right.st_ino
        and stat.S_IFMT(left.st_mode) == stat.S_IFMT(right.st_mode)
    )


def _open_runs_directory(project_root: Path) -> int | None:
    runs_directory = project_root / "runs"
    try:
        metadata = runs_directory.lstat()
    except FileNotFoundError:
        return None
    if stat.S_ISLNK(metadata.st_mode) or not stat.S_ISDIR(metadata.st_mode):
        raise RuntimeError("the formal runs path must be a real directory")
    directory_flags = (
        os.O_RDONLY
        | getattr(os, "O_CLOEXEC", 0)
        | getattr(os, "O_DIRECTORY", 0)
        | getattr(os, "O_NOFOLLOW", 0)
    )
    descriptor = os.open(runs_directory, directory_flags)
    if not _same_directory_inode(metadata, os.fstat(descriptor)):
        os.close(descriptor)
        raise RuntimeError("the formal runs directory changed while it was opened")
    return descriptor


def _directory_entry_exists(descriptor: int, name: str) -> bool:
    try:
        os.stat(name, dir_fd=descriptor, follow_symlinks=False)
    except FileNotFoundError:
        return False
    return True


def _open_real_child_directory(parent_descriptor: int, name: str) -> int:
    metadata = os.stat(name, dir_fd=parent_descriptor, follow_symlinks=False)
    if stat.S_ISLNK(metadata.st_mode) or not stat.S_ISDIR(metadata.st_mode):
        raise RuntimeError("snapshot directory entry must be a real directory")
    flags = (
        os.O_RDONLY
        | getattr(os, "O_CLOEXEC", 0)
        | getattr(os, "O_DIRECTORY", 0)
        | getattr(os, "O_NOFOLLOW", 0)
    )
    descriptor = os.open(name, flags, dir_fd=parent_descriptor)
    if not _same_directory_inode(metadata, os.fstat(descriptor)):
        os.close(descriptor)
        raise RuntimeError("snapshot directory entry changed while it was opened")
    return descriptor


def _fsync_aborted_snapshot_quarantines(runs_descriptor: int) -> None:
    # Abort quarantines are same-parent renames of already durable snapshot entries.  Synchronizing
    # the runs namespace is sufficient and never requires opening an arbitrary quarantined node.
    os.fsync(runs_descriptor)


def _unique_aborted_snapshot_name(runs_descriptor: int) -> str:
    for _attempt in range(128):
        name = M8_ABORTED_CONTROLLER_SNAPSHOT_PREFIX + secrets.token_hex(16)
        if not _directory_entry_exists(runs_descriptor, name):
            return name
    raise RuntimeError("could not allocate a unique aborted snapshot quarantine")


def isolate_aborted_controller_snapshot(project_root: str | Path) -> Path | None:
    """Atomically isolate the fixed active snapshot before retiring a PREPARED attempt."""

    root = Path(project_root).resolve(strict=True)
    runs_descriptor = _open_runs_directory(root)
    if runs_descriptor is None:
        return None
    active_name = Path(M8_CONTROLLER_SNAPSHOT_RELATIVE_PATH).name
    try:
        if not _directory_entry_exists(runs_descriptor, active_name):
            _fsync_aborted_snapshot_quarantines(runs_descriptor)
            return None
        active_metadata = os.stat(
            active_name,
            dir_fd=runs_descriptor,
            follow_symlinks=False,
        )
        quarantine_name = _unique_aborted_snapshot_name(runs_descriptor)
        os.rename(
            active_name,
            quarantine_name,
            src_dir_fd=runs_descriptor,
            dst_dir_fd=runs_descriptor,
        )
        quarantine_metadata = os.stat(
            quarantine_name,
            dir_fd=runs_descriptor,
            follow_symlinks=False,
        )
        if _directory_entry_exists(runs_descriptor, active_name) or not _same_directory_inode(
            active_metadata, quarantine_metadata
        ):
            raise RuntimeError("aborted Controller snapshot quarantine rename differed")
        os.fsync(runs_descriptor)
        return root / "runs" / quarantine_name
    finally:
        os.close(runs_descriptor)


def require_controller_snapshot_quarantine_absent(project_root: str | Path) -> None:
    """Fail closed if a prior COMMITTED snapshot quarantine exists before Test."""

    root = Path(project_root).resolve(strict=True)
    descriptor = _open_runs_directory(root)
    if descriptor is None:
        return
    try:
        quarantine_name = Path(M8_COMMITTED_CONTROLLER_SNAPSHOT_RELATIVE_PATH).name
        if _directory_entry_exists(descriptor, quarantine_name):
            raise RuntimeError("a COMMITTED Controller snapshot quarantine already exists")
    finally:
        os.close(descriptor)


def retire_committed_controller_snapshot(project_root: str | Path) -> None:
    """Atomically quarantine the fixed snapshot after a durable ``COMMITTED`` transition."""

    root = Path(project_root).resolve(strict=True)
    descriptor = _open_runs_directory(root)
    if descriptor is None:
        raise RuntimeError("COMMITTED snapshot retirement requires exactly one snapshot state")
    active_name = Path(M8_CONTROLLER_SNAPSHOT_RELATIVE_PATH).name
    quarantine_name = Path(M8_COMMITTED_CONTROLLER_SNAPSHOT_RELATIVE_PATH).name
    try:
        active_exists = _directory_entry_exists(descriptor, active_name)
        quarantine_exists = _directory_entry_exists(descriptor, quarantine_name)
        if active_exists == quarantine_exists:
            raise RuntimeError(
                "COMMITTED snapshot retirement requires exactly one active or quarantine state"
            )
        if active_exists:
            active_metadata = os.stat(
                active_name,
                dir_fd=descriptor,
                follow_symlinks=False,
            )
            if stat.S_ISLNK(active_metadata.st_mode) or not stat.S_ISDIR(active_metadata.st_mode):
                raise RuntimeError("active COMMITTED Controller snapshot must be a real directory")
            os.rename(
                active_name,
                quarantine_name,
                src_dir_fd=descriptor,
                dst_dir_fd=descriptor,
            )
            if _directory_entry_exists(descriptor, active_name) or not _directory_entry_exists(
                descriptor, quarantine_name
            ):
                raise RuntimeError("COMMITTED Controller snapshot quarantine rename differed")
            quarantine_metadata = os.stat(
                quarantine_name,
                dir_fd=descriptor,
                follow_symlinks=False,
            )
            if (
                stat.S_ISLNK(quarantine_metadata.st_mode)
                or not stat.S_ISDIR(quarantine_metadata.st_mode)
                or not _same_directory_inode(active_metadata, quarantine_metadata)
            ):
                raise RuntimeError("COMMITTED snapshot quarantine inode differs after rename")
        else:
            quarantine_metadata = os.stat(
                quarantine_name,
                dir_fd=descriptor,
                follow_symlinks=False,
            )
            if stat.S_ISLNK(quarantine_metadata.st_mode) or not stat.S_ISDIR(
                quarantine_metadata.st_mode
            ):
                raise RuntimeError(
                    "COMMITTED Controller snapshot quarantine must be a real directory"
                )
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def validate_committed_controller_snapshot_quarantine(project_root: str | Path) -> None:
    """Require the post-publication active-absent, real-quarantine state."""

    root = Path(project_root).resolve(strict=True)
    descriptor = _open_runs_directory(root)
    if descriptor is None:
        raise RuntimeError("COMMITTED Controller snapshot quarantine is missing")
    active_name = Path(M8_CONTROLLER_SNAPSHOT_RELATIVE_PATH).name
    quarantine_name = Path(M8_COMMITTED_CONTROLLER_SNAPSHOT_RELATIVE_PATH).name
    try:
        if _directory_entry_exists(descriptor, active_name):
            raise RuntimeError("active Controller snapshot remains after COMMITTED retirement")
        if not _directory_entry_exists(descriptor, quarantine_name):
            raise RuntimeError("COMMITTED Controller snapshot quarantine is missing")
        metadata = os.stat(
            quarantine_name,
            dir_fd=descriptor,
            follow_symlinks=False,
        )
        if stat.S_ISLNK(metadata.st_mode) or not stat.S_ISDIR(metadata.st_mode):
            raise RuntimeError("COMMITTED Controller snapshot quarantine must be a real directory")
    finally:
        os.close(descriptor)


__all__ = [
    "M8_ABORTED_CONTROLLER_SNAPSHOT_PREFIX",
    "M8_COMMITTED_CONTROLLER_SNAPSHOT_RELATIVE_PATH",
    "M8_CONTROLLER_SNAPSHOT_RELATIVE_PATH",
    "CleanSourceSnapshot",
    "FrozenControllerSnapshot",
    "FrozenInputReport",
    "capture_clean_source",
    "create_frozen_controller_snapshot",
    "frozen_input_digest",
    "isolate_aborted_controller_snapshot",
    "load_frozen_input_reports",
    "require_controller_snapshot_quarantine_absent",
    "retire_committed_controller_snapshot",
    "sha256_regular_file",
    "validate_committed_controller_snapshot_quarantine",
    "validate_frozen_controller_snapshot",
]
