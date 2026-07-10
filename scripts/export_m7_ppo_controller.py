"""Export the passed M7 Validation selection as one inference-only PPO Controller.

This process reads no official Track asset. It accepts only the frozen canonical selection report,
loads its exact retained checkpoint through the v2 publication ledger, and activates the checked-in
unfinalized Controller template once.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import stat
import subprocess
import sys
import tempfile
import tomllib
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Final

PROJECT_ROOT: Final = Path(__file__).resolve().parents[1]
DEFAULT_SELECTION_CONFIG: Final = Path("configs/ppo_selection.toml")
EXPORT_REPORT_PATH: Final = "benchmarks/v0.1/m7_ppo_export_report.json"
PPO_CONTROLLER_DIRECTORY: Final = "controllers/ppo"
PPO_CONTROLLER_POLICY_FILE: Final = "policy.npz"
PPO_CONTROLLER_METADATA_FILE: Final = "metadata.json"
EXPORT_TRANSACTION_DIRECTORY: Final = "runs/ppo/.m7-controller-export-transaction"
_EXPORT_TRANSACTION_SCHEMA: Final = "controller-learning.m7-export-transaction.v1"
_EXPORT_TRANSACTION_ORIGINAL_CONFIG: Final = "original_config.bin"
_EXPORT_TRANSACTION_METADATA: Final = "metadata.json"
_EXPORT_TRANSACTION_READY: Final = "READY"
_EXPORT_TRANSACTION_COMMITTED: Final = "COMMITTED"
_EXPORT_TRANSACTION_STAGING: Final = "staging"
_EXPORT_TRANSACTION_READY_BYTES: Final = b"controller-learning.m7-export-ready.v1\n"
_MUTATION_PATH_ARGUMENTS: Final[dict[str, tuple[tuple[int, int | None], ...]]] = {
    "os.chmod": ((0, 2),),
    "os.chown": ((0, 3),),
    "os.link": ((0, 2), (1, 3)),
    "os.mkdir": ((0, 2),),
    "os.mkfifo": ((0, 2),),
    "os.mknod": ((0, 3),),
    "os.remove": ((0, 1),),
    "os.removexattr": ((0, None),),
    "os.rename": ((0, 2), (1, 3)),
    "os.replace": ((0, 2), (1, 3)),
    "os.rmdir": ((0, 1),),
    "os.setxattr": ((0, None),),
    # A symlink's source is only link text; the destination is the mutated path.
    "os.symlink": ((1, 2),),
    "os.truncate": ((0, None),),
    "os.unlink": ((0, 1),),
    "os.utime": ((0, 3),),
    "shutil.rmtree": ((0, 1),),
}
_ORIGINAL_OS_MKFIFO: Final = os.mkfifo
_ORIGINAL_OS_MKNOD: Final = os.mknod
_INSTALLED_ASSET_GUARDS: list[Any] = []
_UNAUDITED_MUTATION_WRAPPERS_INSTALLED = False


class ForbiddenExportAssetAccessError(RuntimeError):
    """Raised before any official Track or materialized Train-cache file can open."""


@dataclass(slots=True)
class ExportAssetAccessGuard:
    """Process-wide deny-all audit guard for environment assets during export."""

    official_track_root: Path
    track_cache_root: Path
    _installed: bool = False
    _denied_event_count: int = 0
    _denied_open_event_count: int = 0
    _denied_mutation_event_count: int = 0
    _official_track_open_count: int = 0
    _track_cache_open_count: int = 0
    _official_track_mutation_count: int = 0
    _track_cache_mutation_count: int = 0
    _mutation_event_counts: dict[str, int] | None = None
    _unaudited_mutation_wrappers_installed: bool = False

    def __post_init__(self) -> None:
        self.official_track_root = self.official_track_root.resolve(strict=False)
        self.track_cache_root = self.track_cache_root.resolve(strict=False)
        self._mutation_event_counts = {}

    @staticmethod
    def _directory_from_fd(value: object) -> Path | None:
        if type(value) is not int or value < 0:
            return None
        try:
            return Path(os.readlink(f"/proc/self/fd/{value}")).resolve(strict=False)
        except OSError:
            return None

    def _candidate(
        self,
        arguments: tuple[Any, ...],
        *,
        path_index: int,
        dir_fd_index: int | None,
    ) -> Path | None:
        if path_index >= len(arguments):
            return None
        source = arguments[path_index]
        if type(source) is int:
            return self._directory_from_fd(source)
        if not isinstance(source, (str, bytes, os.PathLike)):
            return None
        path = Path(os.fsdecode(os.fspath(source)))
        if not path.is_absolute() and dir_fd_index is not None and dir_fd_index < len(arguments):
            directory = self._directory_from_fd(arguments[dir_fd_index])
            if directory is not None:
                path = directory / path
        return path.resolve(strict=False)

    def _category(self, candidate: Path) -> str | None:
        if candidate.is_relative_to(self.official_track_root):
            return "official_track"
        if candidate.is_relative_to(self.track_cache_root):
            return "track_cache"
        return None

    def _record_mutation(self, event: str, category: str) -> None:
        self._denied_event_count += 1
        self._denied_mutation_event_count += 1
        if category == "official_track":
            self._official_track_mutation_count += 1
        else:
            self._track_cache_mutation_count += 1
        assert self._mutation_event_counts is not None
        self._mutation_event_counts[event] = self._mutation_event_counts.get(event, 0) + 1

    def _audit(self, event: str, arguments: tuple[Any, ...]) -> None:
        if not arguments:
            return
        if event == "open":
            candidate = self._candidate(arguments, path_index=0, dir_fd_index=None)
            category = self._category(candidate) if candidate is not None else None
            if category is None:
                return
            self._denied_event_count += 1
            self._denied_open_event_count += 1
            if category == "official_track":
                self._official_track_open_count += 1
            else:
                self._track_cache_open_count += 1
            raise ForbiddenExportAssetAccessError(
                f"M7 Controller export forbids every {category} file open"
            )

        path_arguments = _MUTATION_PATH_ARGUMENTS.get(event)
        if path_arguments is None:
            return
        for path_index, dir_fd_index in path_arguments:
            candidate = self._candidate(
                arguments,
                path_index=path_index,
                dir_fd_index=dir_fd_index,
            )
            category = self._category(candidate) if candidate is not None else None
            if category is not None:
                self._record_mutation(event, category)
                raise ForbiddenExportAssetAccessError(
                    f"M7 Controller export forbids {event} mutation of {category}"
                )

    def install(self) -> None:
        if self._installed:
            raise RuntimeError("export asset guard is already installed")
        sys.addaudithook(self._audit)
        _install_unaudited_mutation_wrappers(self)
        self._installed = True
        self._unaudited_mutation_wrappers_installed = True

    def evidence(self) -> dict[str, Any]:
        if not self._installed:
            raise RuntimeError("export asset guard was not installed")
        categories = []
        if self._official_track_open_count:
            categories.append("official_track")
        if self._track_cache_open_count:
            categories.append("track_cache")
        return {
            "audit_hook_installed_before_project_imports": True,
            "denied_event_count": self._denied_event_count,
            "denied_mutation_event_count": self._denied_mutation_event_count,
            "denied_open_event_count": self._denied_open_event_count,
            "official_track_open_count": self._official_track_open_count,
            "official_track_mutation_count": self._official_track_mutation_count,
            "opened_path_categories": categories,
            "track_cache_open_count": self._track_cache_open_count,
            "track_cache_mutation_count": self._track_cache_mutation_count,
            "mutation_event_counts": dict(sorted((self._mutation_event_counts or {}).items())),
            "unaudited_mutation_wrappers": (
                ["os.mkfifo", "os.mknod"] if self._unaudited_mutation_wrappers_installed else []
            ),
        }


def _guarded_mkfifo(
    path: str | bytes | os.PathLike[str] | os.PathLike[bytes],
    mode: int = 0o666,
    *,
    dir_fd: int | None = None,
) -> None:
    arguments = (path, mode, -1 if dir_fd is None else dir_fd)
    for guard in tuple(_INSTALLED_ASSET_GUARDS):
        guard._audit("os.mkfifo", arguments)
    if dir_fd is None:
        _ORIGINAL_OS_MKFIFO(path, mode)
    else:
        _ORIGINAL_OS_MKFIFO(path, mode, dir_fd=dir_fd)


def _guarded_mknod(
    path: str | bytes | os.PathLike[str] | os.PathLike[bytes],
    mode: int = 0o600,
    device: int = 0,
    *,
    dir_fd: int | None = None,
) -> None:
    arguments = (path, mode, device, -1 if dir_fd is None else dir_fd)
    for guard in tuple(_INSTALLED_ASSET_GUARDS):
        guard._audit("os.mknod", arguments)
    if dir_fd is None:
        _ORIGINAL_OS_MKNOD(path, mode, device)
    else:
        _ORIGINAL_OS_MKNOD(path, mode, device, dir_fd=dir_fd)


def _install_unaudited_mutation_wrappers(guard: ExportAssetAccessGuard) -> None:
    """Patch the two CPython 3.11 filesystem calls that emit no audit event."""

    global _UNAUDITED_MUTATION_WRAPPERS_INSTALLED
    _INSTALLED_ASSET_GUARDS.append(guard)
    if _UNAUDITED_MUTATION_WRAPPERS_INSTALLED:
        return
    os.mkfifo = _guarded_mkfifo
    os.mknod = _guarded_mknod
    _UNAUDITED_MUTATION_WRAPPERS_INSTALLED = True


@dataclass(frozen=True, slots=True)
class ExportOptions:
    """The frozen selection config is the only command-line input."""

    config: Path = DEFAULT_SELECTION_CONFIG

    def __post_init__(self) -> None:
        path = Path(self.config)
        if path.suffix != ".toml":
            raise ValueError("selection config must use the .toml suffix")
        object.__setattr__(self, "config", path)


def _fsync_directory(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _ensure_transaction_parent(project_root: Path) -> Path:
    current = project_root
    for part in ("runs", "ppo"):
        candidate = current / part
        try:
            metadata = candidate.lstat()
        except FileNotFoundError:
            os.mkdir(candidate, 0o700)
            _fsync_directory(candidate)
            _fsync_directory(current)
            metadata = candidate.lstat()
        if stat.S_ISLNK(metadata.st_mode) or not stat.S_ISDIR(metadata.st_mode):
            raise RuntimeError("persistent export transaction parents must be real directories")
        current = candidate
    return current


def _create_durable_file(path: Path, payload: bytes, *, mode: int = 0o600) -> None:
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_CLOEXEC", 0)
    flags |= getattr(os, "O_NOFOLLOW", 0)
    descriptor = os.open(path, flags, mode)
    try:
        with os.fdopen(descriptor, "wb") as file:
            descriptor = -1
            file.write(payload)
            file.flush()
            os.fsync(file.fileno())
    finally:
        if descriptor >= 0:
            os.close(descriptor)
    if path.read_bytes() != payload:
        raise RuntimeError(f"persistent transaction file {path.name} failed exact readback")


def _canonical_transaction_metadata(*, config: bytes, mode: int) -> bytes:
    document = {
        "config_mode": mode,
        "config_relative_path": f"{PPO_CONTROLLER_DIRECTORY}/config.toml",
        "config_sha256": hashlib.sha256(config).hexdigest(),
        "config_size_bytes": len(config),
        "schema_version": _EXPORT_TRANSACTION_SCHEMA,
    }
    return (
        json.dumps(
            document,
            allow_nan=False,
            ensure_ascii=True,
            separators=(",", ":"),
            sort_keys=True,
        )
        + "\n"
    ).encode("utf-8")


def _read_regular_file(path: Path, *, label: str) -> bytes:
    try:
        metadata = path.lstat()
    except FileNotFoundError as error:
        raise RuntimeError(f"persistent transaction is missing {label}") from error
    if stat.S_ISLNK(metadata.st_mode) or not stat.S_ISREG(metadata.st_mode):
        raise RuntimeError(f"persistent transaction {label} must be a regular file")
    return path.read_bytes()


def _load_transaction_snapshot(directory: Path) -> tuple[bytes, int]:
    original = _read_regular_file(
        directory / _EXPORT_TRANSACTION_ORIGINAL_CONFIG,
        label=_EXPORT_TRANSACTION_ORIGINAL_CONFIG,
    )
    metadata_bytes = _read_regular_file(
        directory / _EXPORT_TRANSACTION_METADATA,
        label=_EXPORT_TRANSACTION_METADATA,
    )
    try:
        value = json.loads(metadata_bytes)
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise RuntimeError("persistent transaction metadata is invalid JSON") from error
    expected_keys = {
        "config_mode",
        "config_relative_path",
        "config_sha256",
        "config_size_bytes",
        "schema_version",
    }
    if not isinstance(value, dict) or set(value) != expected_keys:
        raise RuntimeError("persistent transaction metadata keys differ")
    mode = value["config_mode"]
    if (
        value["schema_version"] != _EXPORT_TRANSACTION_SCHEMA
        or value["config_relative_path"] != f"{PPO_CONTROLLER_DIRECTORY}/config.toml"
        or type(mode) is not int
        or not 0 <= mode <= 0o777
        or value["config_size_bytes"] != len(original)
        or value["config_sha256"] != hashlib.sha256(original).hexdigest()
        or _canonical_transaction_metadata(config=original, mode=mode) != metadata_bytes
    ):
        raise RuntimeError("persistent transaction metadata differs from the original config")
    return original, mode


def _atomic_restore_file(
    path: Path,
    payload: bytes,
    *,
    mode: int,
    staging_directory: Path,
) -> None:
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.",
        suffix=".recovery",
        dir=staging_directory,
    )
    temporary = Path(temporary_name)
    try:
        os.fchmod(descriptor, mode)
        with os.fdopen(descriptor, "wb") as file:
            descriptor = -1
            file.write(payload)
            file.flush()
            os.fsync(file.fileno())
        os.replace(temporary, path)
        _fsync_directory(staging_directory)
        _fsync_directory(path.parent)
        if path.read_bytes() != payload or stat.S_IMODE(path.stat().st_mode) != mode:
            raise RuntimeError("restored Controller config differs from the persistent snapshot")
    finally:
        if descriptor >= 0:
            os.close(descriptor)
        temporary.unlink(missing_ok=True)


def _publish_staged_artifact(
    project_root: Path,
    relative_path: str,
    payload: bytes,
    *,
    staging_directory: Path,
) -> Any:
    """Publish immutable bytes from transaction-local staging with exact readback."""

    from controller_learning.rl.artifacts import ArtifactRecord

    destination = project_root / relative_path
    if destination.exists() or destination.is_symlink():
        raise RuntimeError(f"refusing to overwrite formal export artifact {relative_path}")
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{destination.name}.",
        suffix=".tmp",
        dir=staging_directory,
    )
    temporary = Path(temporary_name)
    try:
        os.fchmod(descriptor, 0o600)
        with os.fdopen(descriptor, "wb") as file:
            descriptor = -1
            file.write(payload)
            file.flush()
            os.fsync(file.fileno())
        if destination.exists() or destination.is_symlink():
            raise RuntimeError(f"formal export artifact appeared concurrently: {relative_path}")
        os.replace(temporary, destination)
        _fsync_directory(staging_directory)
        _fsync_directory(destination.parent)
        if destination.read_bytes() != payload:
            raise RuntimeError(f"formal export artifact failed exact readback: {relative_path}")
    finally:
        if descriptor >= 0:
            os.close(descriptor)
        temporary.unlink(missing_ok=True)
    return ArtifactRecord(
        relative_path=relative_path,
        sha256=hashlib.sha256(payload).hexdigest(),
        size_bytes=len(payload),
    )


def _remove_export_outputs(*, plugin_directory: Path, report_path: Path) -> None:
    touched_directories: set[Path] = set()
    for path in (
        plugin_directory / PPO_CONTROLLER_POLICY_FILE,
        plugin_directory / PPO_CONTROLLER_METADATA_FILE,
        report_path,
    ):
        try:
            metadata = path.lstat()
        except FileNotFoundError:
            continue
        if stat.S_ISDIR(metadata.st_mode):
            raise RuntimeError(f"cannot safely remove directory at export output {path}")
        path.unlink()
        touched_directories.add(path.parent)
    for directory in sorted(touched_directories):
        _fsync_directory(directory)


def _remove_transaction_directory(directory: Path) -> None:
    try:
        metadata = directory.lstat()
    except FileNotFoundError:
        return
    if stat.S_ISLNK(metadata.st_mode) or not stat.S_ISDIR(metadata.st_mode):
        raise RuntimeError("persistent export transaction must be a real directory")
    allowed = {
        _EXPORT_TRANSACTION_COMMITTED,
        _EXPORT_TRANSACTION_ORIGINAL_CONFIG,
        _EXPORT_TRANSACTION_METADATA,
        _EXPORT_TRANSACTION_READY,
        _EXPORT_TRANSACTION_STAGING,
    }
    entries = tuple(directory.iterdir())
    unexpected = sorted(entry.name for entry in entries if entry.name not in allowed)
    if unexpected:
        raise RuntimeError(
            "persistent export transaction contains unexpected entries: " + ", ".join(unexpected)
        )
    staging = directory / _EXPORT_TRANSACTION_STAGING
    for entry in entries:
        entry_metadata = entry.lstat()
        if (
            entry == staging
            and stat.S_ISDIR(entry_metadata.st_mode)
            and not stat.S_ISLNK(entry_metadata.st_mode)
        ):
            continue
        if stat.S_ISLNK(entry_metadata.st_mode) or not stat.S_ISREG(entry_metadata.st_mode):
            raise RuntimeError("persistent export transaction entries must be regular files")
    state_entries = tuple(
        directory / name
        for name in (_EXPORT_TRANSACTION_READY, _EXPORT_TRANSACTION_COMMITTED)
        if (directory / name).exists()
    )
    if len(state_entries) > 1:
        raise RuntimeError("persistent export transaction has conflicting state markers")
    for entry in state_entries:
        entry.unlink()
    if state_entries:
        _fsync_directory(directory)
    if staging.exists():
        staging_entries = tuple(staging.iterdir())
        for entry in staging_entries:
            entry_metadata = entry.lstat()
            if stat.S_ISLNK(entry_metadata.st_mode) or not stat.S_ISREG(entry_metadata.st_mode):
                raise RuntimeError("persistent transaction staging entries must be regular files")
            entry.unlink()
        _fsync_directory(staging)
        os.rmdir(staging)
        _fsync_directory(directory)
    for name in (_EXPORT_TRANSACTION_ORIGINAL_CONFIG, _EXPORT_TRANSACTION_METADATA):
        entry = directory / name
        entry.unlink(missing_ok=True)
    _fsync_directory(directory)
    os.rmdir(directory)
    _fsync_directory(directory.parent)


def _restore_ready_transaction(
    directory: Path,
    *,
    plugin_directory: Path,
    report_path: Path,
) -> None:
    ready = _read_regular_file(directory / _EXPORT_TRANSACTION_READY, label="READY")
    if ready != _EXPORT_TRANSACTION_READY_BYTES:
        raise RuntimeError("persistent export transaction READY marker is invalid")
    original, mode = _load_transaction_snapshot(directory)
    _atomic_restore_file(
        plugin_directory / "config.toml",
        original,
        mode=mode,
        staging_directory=directory / _EXPORT_TRANSACTION_STAGING,
    )
    _remove_export_outputs(plugin_directory=plugin_directory, report_path=report_path)


def _recover_persistent_export_transaction(project_root: Path) -> str:
    """Recover an interrupted export before the clean-worktree and one-time gates."""

    _ensure_transaction_parent(project_root)
    active = project_root / EXPORT_TRANSACTION_DIRECTORY
    try:
        active_metadata = active.lstat()
    except FileNotFoundError:
        return "none"
    if stat.S_ISLNK(active_metadata.st_mode) or not stat.S_ISDIR(active_metadata.st_mode):
        raise RuntimeError("persistent export transaction must be a real directory")
    committed_path = active / _EXPORT_TRANSACTION_COMMITTED
    if committed_path.exists() or committed_path.is_symlink():
        committed = _read_regular_file(committed_path, label="COMMITTED")
        if committed != _EXPORT_TRANSACTION_READY_BYTES:
            raise RuntimeError("persistent export transaction COMMITTED marker is invalid")
        _remove_transaction_directory(active)
        return "committed_cleanup_completed"
    ready_path = active / _EXPORT_TRANSACTION_READY
    if ready_path.exists() or ready_path.is_symlink():
        _restore_ready_transaction(
            active,
            plugin_directory=project_root / PPO_CONTROLLER_DIRECTORY,
            report_path=project_root / EXPORT_REPORT_PATH,
        )
        _remove_transaction_directory(active)
        return "ready_rolled_back"
    # The exporter is called only after READY has been durably published, so an unready staging
    # directory cannot have produced policy, metadata, report, or finalized-config writes.
    _remove_transaction_directory(active)
    return "unready_staging_cleaned"


@dataclass(slots=True)
class _ExportOutputTransaction:
    """Durably restore the source template unless every post-export proof commits."""

    project_root: Path
    plugin_directory: Path
    report_path: Path
    _committed: bool = False

    @property
    def directory(self) -> Path:
        return self.project_root / EXPORT_TRANSACTION_DIRECTORY

    @property
    def staging_directory(self) -> Path:
        return self.directory / _EXPORT_TRANSACTION_STAGING

    def __enter__(self) -> _ExportOutputTransaction:
        parent = _ensure_transaction_parent(self.project_root)
        if self.directory.exists() or self.directory.is_symlink():
            raise RuntimeError("persistent export transaction was not recovered before staging")
        config_path = self.plugin_directory / "config.toml"
        if config_path.is_symlink() or not config_path.is_file():
            raise RuntimeError(
                "Controller config must be a regular file before transaction staging"
            )
        original = config_path.read_bytes()
        mode = stat.S_IMODE(config_path.stat().st_mode)
        os.mkdir(self.directory, 0o700)
        _fsync_directory(parent)
        os.mkdir(self.staging_directory, 0o700)
        _fsync_directory(self.directory)
        _create_durable_file(
            self.directory / _EXPORT_TRANSACTION_ORIGINAL_CONFIG,
            original,
        )
        _create_durable_file(
            self.directory / _EXPORT_TRANSACTION_METADATA,
            _canonical_transaction_metadata(config=original, mode=mode),
        )
        _fsync_directory(self.directory)
        # READY is the durable boundary: no exporter/output mutation may happen before this file.
        _create_durable_file(
            self.directory / _EXPORT_TRANSACTION_READY,
            _EXPORT_TRANSACTION_READY_BYTES,
        )
        _fsync_directory(self.directory)
        return self

    def _rollback(self) -> None:
        _restore_ready_transaction(
            self.directory,
            plugin_directory=self.plugin_directory,
            report_path=self.report_path,
        )
        _remove_transaction_directory(self.directory)

    def commit(self) -> None:
        ready = self.directory / _EXPORT_TRANSACTION_READY
        committed = self.directory / _EXPORT_TRANSACTION_COMMITTED
        os.rename(ready, committed)
        _fsync_directory(self.directory)
        # The atomic marker transition is the commit point. A crash afterward only needs cleanup.
        self._committed = True
        _remove_transaction_directory(self.directory)

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
            self._rollback()
        except BaseException as rollback_error:
            if exception is None:
                raise
            raise RuntimeError(
                "Controller export failed and its persistent rollback also failed"
            ) from rollback_error
        if exception is None:
            raise RuntimeError("Controller export transaction exited without commit")
        return False


def _parse_args(argv: Sequence[str] | None = None) -> ExportOptions:
    parser = argparse.ArgumentParser(
        description="Export the passed M7 checkpoint as the inference-only PPO Controller"
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=DEFAULT_SELECTION_CONFIG,
        help="Frozen M7 selection TOML inside the repository",
    )
    return ExportOptions(config=parser.parse_args(argv).config)


def _run_command(command: Sequence[str], *, cwd: Path) -> str:
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
        raise RuntimeError("formal Controller export requires a full lowercase Git revision")
    if status:
        raise RuntimeError("formal Controller export requires a clean worktree")
    return {"revision": revision, "worktree_clean": True}


def _source_snapshot_allowing_outputs(
    project_root: Path,
    *,
    expected_revision: str,
    allowed_paths: Sequence[str],
) -> dict[str, Any]:
    revision = _run_command(("git", "rev-parse", "--verify", "HEAD"), cwd=project_root)
    if revision != expected_revision:
        raise RuntimeError("source revision changed during Controller export")
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
    allowed = set(allowed_paths)
    unexpected = observed - allowed
    missing = allowed - observed
    if unexpected:
        raise RuntimeError(
            "unexpected worktree changes appeared during Controller export: "
            + ", ".join(sorted(unexpected))
        )
    if missing:
        raise RuntimeError(
            "formal Controller export did not produce every declared output: "
            + ", ".join(sorted(missing))
        )
    return {
        "allowed_generated_output_paths": sorted(allowed),
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


def _record(project_root: Path, path: Path) -> Any:
    from controller_learning.rl.artifacts import ArtifactRecord, sha256_file

    resolved = _project_file(project_root, path, label="artifact")
    return ArtifactRecord(
        relative_path=resolved.relative_to(project_root.resolve(strict=True)).as_posix(),
        sha256=sha256_file(resolved),
        size_bytes=resolved.stat().st_size,
    )


def _mapping(value: object, *, field: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise RuntimeError(f"{field} must be an object")
    return value


def _artifact_from_report(value: object, *, field: str) -> Any:
    from controller_learning.rl.artifacts import ArtifactRecord

    mapping = _mapping(value, field=field)
    try:
        return ArtifactRecord(**dict(mapping))
    except (TypeError, ValueError) as error:
        raise RuntimeError(f"{field} is not a valid artifact record") from error


def _require_unfinalized_template(plugin_directory: Path, *, report_path: Path) -> None:
    controller_path = plugin_directory / "controller.py"
    config_path = plugin_directory / "config.toml"
    for path in (controller_path, config_path):
        if path.is_symlink() or not path.is_file():
            raise RuntimeError(f"PPO template requires regular {path.name}")
    try:
        with config_path.open("rb") as file:
            config = tomllib.load(file)
    except (OSError, tomllib.TOMLDecodeError) as error:
        raise RuntimeError("PPO template config.toml is invalid") from error
    if config.get("finalized") is not False:
        raise RuntimeError("formal export requires the unfinalized one-time PPO template")
    for path in (
        plugin_directory / PPO_CONTROLLER_POLICY_FILE,
        plugin_directory / PPO_CONTROLLER_METADATA_FILE,
        report_path,
    ):
        if path.exists() or path.is_symlink():
            raise RuntimeError(f"one-time export output already exists: {path.name}")


def _input_hashes(
    *,
    selection_config_path: Path,
    selection_report_path: Path,
    training_config_path: Path,
    checkpoint_path: Path,
) -> dict[str, str]:
    from controller_learning.rl.artifacts import sha256_file

    return {
        "selected_checkpoint": sha256_file(checkpoint_path),
        "selection_config": sha256_file(selection_config_path),
        "selection_report": sha256_file(selection_report_path),
        "training_config": sha256_file(training_config_path),
    }


def _load_torch() -> Any:
    import torch

    return torch


def _verify_exported_controller(
    *,
    project_root: Path,
    plugin_directory: Path,
    result: Any,
    selected: Any,
    canonical_policy_bytes: bytes,
) -> dict[str, Any]:
    from controller_learning.rl.artifacts import read_strict_json, sha256_file
    from controller_learning.rl.controller_export import (
        PpoControllerExportResult,
        load_ppo_controller_runtime,
    )
    from controller_learning.rl.numpy_actor import canonical_numpy_actor_bytes

    if not isinstance(result, PpoControllerExportResult):
        raise TypeError("formal exporter must return PpoControllerExportResult")
    if result.plugin_directory != plugin_directory:
        raise RuntimeError("exporter returned a different plugin directory")
    if (
        result.policy.schema_version != selected.inference_policy["schema_version"]
        or result.policy.sha256 != selected.inference_policy["sha256"]
        or result.policy.size_bytes != selected.inference_policy["size_bytes"]
    ):
        raise RuntimeError("export result policy differs from Validation selection evidence")

    config_path = plugin_directory / "config.toml"
    metadata_path = plugin_directory / PPO_CONTROLLER_METADATA_FILE
    policy_path = plugin_directory / PPO_CONTROLLER_POLICY_FILE
    if policy_path.read_bytes() != canonical_policy_bytes:
        raise RuntimeError("exported policy bytes differ from the selected canonical actor")
    if (
        sha256_file(config_path) != result.config_sha256
        or config_path.stat().st_size != result.config_size_bytes
        or sha256_file(metadata_path) != result.metadata_sha256
        or metadata_path.stat().st_size != result.metadata_size_bytes
    ):
        raise RuntimeError("exported config or metadata differs from exporter evidence")
    with config_path.open("rb") as file:
        config = tomllib.load(file)
    runtime = load_ppo_controller_runtime(
        {"controller": config},
        plugin_directory=plugin_directory,
    )
    if runtime.checkpoint != result.checkpoint or runtime.policy_evidence != result.policy:
        raise RuntimeError("finalized Controller runtime differs from exporter evidence")
    if canonical_numpy_actor_bytes(runtime.actor) != canonical_policy_bytes:
        raise RuntimeError("finalized Controller runtime reconstructs different actor bytes")
    metadata = read_strict_json(plugin_directory, PPO_CONTROLLER_METADATA_FILE)
    if metadata.get("checkpoint") != result.checkpoint.to_dict() or metadata.get("policy") != {
        "file": PPO_CONTROLLER_POLICY_FILE,
        "schema_version": result.policy.schema_version,
        "sha256": result.policy.sha256,
        "size_bytes": result.policy.size_bytes,
    }:
        raise RuntimeError("finalized metadata differs from selected checkpoint or policy")
    inference = metadata.get("inference_only")
    if inference != {
        "contains_environment_state": False,
        "contains_optimizer_state": False,
        "contains_value_network": False,
        "runtime": "numpy",
    }:
        raise RuntimeError("finalized metadata is not inference-only")
    return {
        "artifacts": {
            "config": _record(project_root, config_path).to_dict(),
            "metadata": _record(project_root, metadata_path).to_dict(),
            "policy": _record(project_root, policy_path).to_dict(),
        },
        "checkpoint": result.checkpoint.to_dict(),
        "inference_only": {
            "contains_environment_state": False,
            "contains_optimizer_state": False,
            "contains_value_network": False,
        },
        "plugin_directory": PPO_CONTROLLER_DIRECTORY,
        "runtime": "numpy",
    }


def run_export(
    options: ExportOptions,
    *,
    access_guard: ExportAssetAccessGuard,
    project_root: Path = PROJECT_ROOT,
    torch_module: Any | None = None,
) -> dict[str, Any]:
    """Perform the clean-source, passed-selection, exact-checkpoint export once."""

    if not isinstance(options, ExportOptions):
        raise TypeError("options must be ExportOptions")
    if not isinstance(access_guard, ExportAssetAccessGuard) or not access_guard._installed:
        raise RuntimeError("the deny-all asset guard must be installed before project imports")

    # All project imports are deliberately below the installed process-wide asset audit hook.
    from controller_learning.rl.artifacts import (
        TrainingRunIdentity,
        canonical_json_bytes,
        load_published_training_checkpoint,
        read_strict_json,
    )
    from controller_learning.rl.configuration import load_ppo_config
    from controller_learning.rl.controller_export import export_ppo_controller
    from controller_learning.rl.export_protocol import (
        EXPORT_REPORT_SCHEMA_VERSION,
        selected_export_candidate,
        validate_export_report,
    )
    from controller_learning.rl.numpy_actor import (
        canonical_numpy_actor_bytes,
        numpy_actor_from_ppo_state_dict,
    )
    from controller_learning.rl.selection import (
        SELECTION_REPORT_SCHEMA_VERSION,
        load_ppo_selection_config,
        torch_state_dict_sha256,
        validate_selection_report,
    )

    root = Path(project_root).resolve(strict=True)
    _recover_persistent_export_transaction(root)
    config_path = _project_file(root, options.config, label="selection config")
    if config_path.relative_to(root).as_posix() != DEFAULT_SELECTION_CONFIG.as_posix():
        raise RuntimeError("formal export requires configs/ppo_selection.toml")
    source = _source_snapshot(root)
    selection_config = load_ppo_selection_config(config_path)
    selection_report_path = _project_file(
        root, selection_config.report_path, label="selection report"
    )
    selection_report = read_strict_json(root, selection_config.report_path)
    validate_selection_report(selection_report, config=selection_config)
    selected = selected_export_candidate(selection_report)

    selection_config_record = _record(root, config_path)
    selection_report_record = _record(root, selection_report_path)
    reported_selection_config = _artifact_from_report(
        _mapping(selection_report.get("artifacts"), field="selection artifacts").get(
            "selection_config"
        ),
        field="selection config artifact",
    )
    if reported_selection_config != selection_config_record:
        raise RuntimeError("selection report does not bind the current frozen selection config")

    training = _mapping(selection_report.get("training_run"), field="training_run")
    identity = TrainingRunIdentity.from_dict(
        _mapping(training.get("identity"), field="training_run.identity")
    )
    run_directory = _project_directory(root, selection_config.run_directory, label="training run")
    training_config_path = _project_file(
        root, selection_config.training_config, label="training config"
    )
    training_config_record = _record(root, training_config_path)
    reported_training_config = _artifact_from_report(
        _mapping(selection_report.get("artifacts"), field="selection artifacts").get(
            "training_config"
        ),
        field="training config artifact",
    )
    if (
        reported_training_config != training_config_record
        or training_config_record.sha256 != identity.configuration_sha256
    ):
        raise RuntimeError("selection report, training config, and run identity differ")
    training_config = load_ppo_config(training_config_path)

    plugin_directory = _project_directory(root, PPO_CONTROLLER_DIRECTORY, label="PPO Controller")
    export_report_path = root / EXPORT_REPORT_PATH
    _require_unfinalized_template(plugin_directory, report_path=export_report_path)

    torch = _load_torch() if torch_module is None else torch_module
    loaded = load_published_training_checkpoint(
        run_directory,
        expected_identity=identity,
        update_index=selected.update_index,
        checkpoint_directory=selection_config.checkpoint_directory,
        torch_module=torch,
    )
    if (
        loaded.record != selected.checkpoint
        or loaded.metadata.run_identity != identity
        or loaded.metadata.update_index != selected.update_index
        or loaded.metadata.vector_steps != selected.vector_steps
        or loaded.metadata.valid_transitions != selected.valid_transitions
    ):
        raise RuntimeError("strictly loaded checkpoint differs from selection evidence")
    parameter_sha256 = torch_state_dict_sha256(
        loaded.payload["model_state_dict"], torch_module=torch
    )
    if parameter_sha256 != selected.parameter_sha256:
        raise RuntimeError("selected checkpoint parameter SHA-256 differs from Validation evidence")
    canonical_policy = canonical_numpy_actor_bytes(
        numpy_actor_from_ppo_state_dict(loaded.payload["model_state_dict"])
    )
    policy_identity = {
        "schema_version": selected.inference_policy["schema_version"],
        "sha256": hashlib.sha256(canonical_policy).hexdigest(),
        "size_bytes": len(canonical_policy),
    }
    if policy_identity != dict(selected.inference_policy):
        raise RuntimeError("selected canonical NumPy actor differs from Validation evidence")

    checkpoint_path = _project_file(
        run_directory, selected.checkpoint.relative_path, label="selected checkpoint"
    )
    pre_export_hashes = _input_hashes(
        selection_config_path=config_path,
        selection_report_path=selection_report_path,
        training_config_path=training_config_path,
        checkpoint_path=checkpoint_path,
    )
    if pre_export_hashes["selected_checkpoint"] != selected.checkpoint.sha256:
        raise RuntimeError("selected checkpoint changed after strict loading")

    allowed_outputs = [
        EXPORT_REPORT_PATH,
        f"{PPO_CONTROLLER_DIRECTORY}/config.toml",
        f"{PPO_CONTROLLER_DIRECTORY}/{PPO_CONTROLLER_METADATA_FILE}",
        f"{PPO_CONTROLLER_DIRECTORY}/{PPO_CONTROLLER_POLICY_FILE}",
    ]
    with _ExportOutputTransaction(
        project_root=root,
        plugin_directory=plugin_directory,
        report_path=export_report_path,
    ) as transaction:
        result = export_ppo_controller(
            plugin_directory,
            loaded_checkpoint=loaded,
            training_config_path=training_config_path,
            public_policy_max_bytes=training_config.checkpoint.public_checkpoint_max_bytes,
            staging_directory=transaction.staging_directory,
        )
        controller_evidence = _verify_exported_controller(
            project_root=root,
            plugin_directory=plugin_directory,
            result=result,
            selected=selected,
            canonical_policy_bytes=canonical_policy,
        )
        post_export_hashes = _input_hashes(
            selection_config_path=config_path,
            selection_report_path=selection_report_path,
            training_config_path=training_config_path,
            checkpoint_path=checkpoint_path,
        )
        if post_export_hashes != pre_export_hashes:
            raise RuntimeError("a formal export input changed during Controller finalization")
        asset_access = access_guard.evidence()
        if asset_access != {
            "audit_hook_installed_before_project_imports": True,
            "denied_event_count": 0,
            "denied_mutation_event_count": 0,
            "denied_open_event_count": 0,
            "official_track_open_count": 0,
            "official_track_mutation_count": 0,
            "opened_path_categories": [],
            "track_cache_open_count": 0,
            "track_cache_mutation_count": 0,
            "mutation_event_counts": {},
            "unaudited_mutation_wrappers": ["os.mkfifo", "os.mknod"],
        }:
            raise RuntimeError("formal export attempted to open an environment asset")

        report = {
            "asset_access": asset_access,
            "controller": controller_evidence,
            "input_stability": {
                "all_inputs_unchanged": True,
                "post_export_sha256": post_export_hashes,
                "pre_export_sha256": pre_export_hashes,
            },
            "protocol": {
                "canonical_inference_policy_verified": True,
                "canonical_selection_report_required": True,
                "exact_published_checkpoint_loader": "v2_explicit_update",
                "formal_export_function": (
                    "controller_learning.rl.controller_export.export_ppo_controller"
                ),
                "full_parameter_sha256_verified": True,
                "no_gradient_or_optimizer_operations": True,
                "one_time_unfinalized_template_activation": True,
                "passed_selection_gate_required": True,
                "persistent_crash_recovery": {
                    "commit_transition": "READY_to_COMMITTED_then_cleanup",
                    "exporter_starts_only_after_ready": True,
                    "original_config_bytes_and_mode_fsynced": True,
                    "startup_ready_action": "restore_config_delete_outputs_then_cleanup",
                    "startup_unready_action": "cleanup_staging_only",
                    "temporary_file_location": "transaction_staging_only",
                    "transaction_directory": EXPORT_TRANSACTION_DIRECTORY,
                },
                "selection_outputs_committed_before_export": True,
            },
            "schema_version": EXPORT_REPORT_SCHEMA_VERSION,
            "selection": {
                "config": selection_config_record.to_dict(),
                "gate_passed": True,
                "report": selection_report_record.to_dict(),
                "report_schema_version": SELECTION_REPORT_SCHEMA_VERSION,
                "report_status": "passed",
                "selected_candidate": selected.to_dict(),
            },
            "source": {
                "post_export_worktree": {
                    "allowed_generated_output_paths": sorted(allowed_outputs),
                    "observed_changed_paths": sorted(allowed_outputs),
                    "only_allowed_generated_outputs": True,
                    "revision": source["revision"],
                    "unexpected_changed_paths": [],
                },
                "preflight": source,
            },
            "status": "passed",
            "training": {
                "checkpoint_directory": selection_config.checkpoint_directory,
                "identity": identity.to_dict(),
                "run_directory": selection_config.run_directory,
                "training_config": training_config_record.to_dict(),
            },
        }
        validate_export_report(report)
        report_record = _publish_staged_artifact(
            root,
            EXPORT_REPORT_PATH,
            canonical_json_bytes(report),
            staging_directory=transaction.staging_directory,
        )
        post_source = _source_snapshot_allowing_outputs(
            root,
            expected_revision=source["revision"],
            allowed_paths=allowed_outputs,
        )
        if post_source != report["source"]["post_export_worktree"]:
            raise RuntimeError("post-export worktree evidence differs from declared outputs")
        # Re-read the canonical report to prove the final bytes satisfy the public validator.
        validate_export_report(read_strict_json(root, EXPORT_REPORT_PATH))
        transaction.commit()
    return {
        "checkpoint_sha256": selected.checkpoint.sha256,
        "export_report": report_record.relative_path,
        "policy_sha256": result.policy.sha256,
        "selected_update": selected.update_index,
    }


def main(argv: Sequence[str] | None = None) -> None:
    guard = ExportAssetAccessGuard(
        official_track_root=PROJECT_ROOT / "controller_learning/assets/tracks",
        track_cache_root=PROJECT_ROOT / ".track-cache",
    )
    # Keep this install before run_export: that function performs every project import lazily.
    guard.install()
    result = run_export(_parse_args(argv), access_guard=guard)
    print(
        "M7 PPO Controller export passed: "
        f"update={result['selected_update']}, policy_sha256={result['policy_sha256']}, "
        f"report={result['export_report']}"
    )


if __name__ == "__main__":
    main()
