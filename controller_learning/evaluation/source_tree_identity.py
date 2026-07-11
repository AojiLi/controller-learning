"""Deterministic, Test-content-free identity of the formal M8 source tree."""

from __future__ import annotations

import hashlib
import json
import os
import stat
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Final, Literal

SOURCE_TREE_IDENTITY_SCHEMA_VERSION: Final = "controller-learning.m8-source-tree-identity.v1"
PROTECTED_TRACK_TREE_RELATIVE_PATH: Final = "controller_learning/assets/tracks/v0.1"
PROTECTED_TRACK_METADATA_RELATIVE_PATHS: Final = (
    PROTECTED_TRACK_TREE_RELATIVE_PATH,
    *(
        f"{PROTECTED_TRACK_TREE_RELATIVE_PATH}/{name}"
        for name in (
            "level0.json",
            "level0.npz",
            "train.json",
            "train.npz",
            "validation.json",
            "validation.npz",
            "test.json",
            "test.npz",
        )
    ),
)
M8_CENTRAL_OUTPUT_RELATIVE_PATHS: Final = frozenset(
    {
        "benchmarks/v0.1/m8_final_evaluation_report.json",
        "benchmarks/v0.1/m8_final_results.csv",
        "benchmarks/v0.1/m8_test_row_000_comparison.png",
    }
)

# These selectors are part of the evidence contract. They are intentionally fixed here instead of
# being inferred from Git or .gitignore, both of which are unavailable after the formal bind.
SOURCE_TREE_EXCLUSION_POLICY: Final = (
    "root-directories:.git,.pixi,.venv,venv,reference,runs,results,dist,site,build",
    "dynamic-root-directories:.track-cache,wandb,videos",
    "cache-directories:__pycache__,.cache,.hypothesis,.pytest_cache,.ruff_cache,.mypy_cache,.tox,.nox",
    "coverage:coverage,htmlcov,.coverage,.coverage.*,coverage.xml",
    "editor:.idea,.vscode,.DS_Store,Thumbs.db,*~",
    "build-metadata:*.egg-info",
    "unsafe-import-artifacts:reject-*.pyc,*.pyd,*.pyo,*.so-outside-excluded-directories",
    "local-secrets:.env,.env.*,*.pem,*.key",
    "local-benchmarks:benchmarks/local",
    "materialized-track-pools:controller_learning/assets/tracks/**/train_pool.npz",
    "m8-central-outputs:" + ",".join(sorted(M8_CENTRAL_OUTPUT_RELATIVE_PATHS)),
)

_ROOT_EXCLUDED_DIRECTORIES: Final = frozenset(
    {
        ".git",
        ".pixi",
        ".track-cache",
        ".venv",
        "build",
        "dist",
        "reference",
        "results",
        "runs",
        "site",
        "venv",
        "videos",
        "wandb",
    }
)
_EXCLUDED_DIRECTORY_NAMES: Final = frozenset(
    {
        ".cache",
        ".idea",
        ".hypothesis",
        ".mypy_cache",
        ".nox",
        ".pytest_cache",
        ".ruff_cache",
        ".tox",
        ".vscode",
        "__pycache__",
        "coverage",
        "htmlcov",
    }
)
_EXCLUDED_FILE_NAMES: Final = frozenset({".DS_Store", "Thumbs.db", "coverage.xml"})
_PROTECTED_TRACK_PARTS: Final = PurePosixPath(PROTECTED_TRACK_TREE_RELATIVE_PATH).parts
_STABLE_STAT_FIELDS: Final = (
    "st_dev",
    "st_ino",
    "st_mode",
    "st_size",
    "st_mtime_ns",
    "st_ctime_ns",
)
_READ_CHUNK_SIZE: Final = 1024 * 1024

NodeType = Literal["directory", "regular_file"]


@dataclass(frozen=True, slots=True)
class SourceTreeEntry:
    """Canonical identity row for one included source-tree node."""

    path: str
    node_type: NodeType
    mode: int
    size_bytes: int | None
    content_sha256: str | None
    mtime_ns: int | None
    protected_metadata_only: bool

    def __post_init__(self) -> None:
        relative = PurePosixPath(self.path)
        if (
            self.path != relative.as_posix()
            or relative.is_absolute()
            or ".." in relative.parts
            or self.path == ""
        ):
            raise ValueError("source-tree entry path must be canonical and relative")
        if self.node_type not in ("directory", "regular_file"):
            raise ValueError("source-tree entry has an unsupported node type")
        if type(self.mode) is not int or not 0 <= self.mode <= 0o7777:
            raise ValueError("source-tree entry mode must contain Unix permission bits")
        if self.protected_metadata_only:
            if (
                type(self.size_bytes) is not int
                or self.size_bytes < 0
                or type(self.mtime_ns) is not int
                or self.mtime_ns < 0
                or self.content_sha256 is not None
            ):
                raise ValueError("protected source-tree entries must contain lstat metadata only")
        elif self.node_type == "regular_file":
            if (
                type(self.size_bytes) is not int
                or self.size_bytes < 0
                or not _is_sha256(self.content_sha256)
                or self.mtime_ns is not None
            ):
                raise ValueError("ordinary files must contain size and content SHA-256")
        elif any(
            value is not None for value in (self.size_bytes, self.content_sha256, self.mtime_ns)
        ):
            raise ValueError("ordinary directory entries contain path, type, and mode only")

    def to_dict(self) -> dict[str, object]:
        """Return the canonical JSON-compatible representation."""

        return {
            "content_sha256": self.content_sha256,
            "mode": self.mode,
            "mtime_ns": self.mtime_ns,
            "node_type": self.node_type,
            "path": self.path,
            "protected_metadata_only": self.protected_metadata_only,
            "size_bytes": self.size_bytes,
        }


@dataclass(frozen=True, slots=True)
class SourceTreeIdentity:
    """Machine-independent source-tree evidence and its canonical aggregate digest."""

    entries: tuple[SourceTreeEntry, ...]
    aggregate_sha256: str
    schema_version: str = SOURCE_TREE_IDENTITY_SCHEMA_VERSION
    protected_track_tree: str = PROTECTED_TRACK_TREE_RELATIVE_PATH
    protected_track_paths: tuple[str, ...] = PROTECTED_TRACK_METADATA_RELATIVE_PATHS
    exclusion_policy: tuple[str, ...] = SOURCE_TREE_EXCLUSION_POLICY

    def __post_init__(self) -> None:
        if self.schema_version != SOURCE_TREE_IDENTITY_SCHEMA_VERSION:
            raise ValueError("unexpected source-tree identity schema version")
        if self.protected_track_tree != PROTECTED_TRACK_TREE_RELATIVE_PATH:
            raise ValueError("unexpected protected Track tree")
        if self.protected_track_paths != PROTECTED_TRACK_METADATA_RELATIVE_PATHS:
            raise ValueError("unexpected protected Track metadata paths")
        if self.exclusion_policy != SOURCE_TREE_EXCLUSION_POLICY:
            raise ValueError("unexpected source-tree exclusion policy")
        if tuple(entry.path for entry in self.entries) != tuple(
            sorted(entry.path for entry in self.entries)
        ) or len({entry.path for entry in self.entries}) != len(self.entries):
            raise ValueError("source-tree entries must have unique canonical path order")
        if not _is_sha256(self.aggregate_sha256):
            raise ValueError("aggregate_sha256 must be a lowercase SHA-256 digest")
        expected = hashlib.sha256(_canonical_identity_payload_bytes(self.entries)).hexdigest()
        if self.aggregate_sha256 != expected:
            raise ValueError("aggregate_sha256 does not match the canonical source-tree evidence")

    def to_dict(self) -> dict[str, object]:
        """Return the complete canonical JSON-compatible evidence object."""

        return {
            "aggregate_sha256": self.aggregate_sha256,
            "entries": [entry.to_dict() for entry in self.entries],
            "exclusion_policy": list(self.exclusion_policy),
            "protected_track_paths": list(self.protected_track_paths),
            "protected_track_tree": self.protected_track_tree,
            "schema_version": self.schema_version,
        }


def _is_sha256(value: object) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    )


def _canonical_identity_payload(entries: tuple[SourceTreeEntry, ...]) -> dict[str, object]:
    return {
        "entries": [entry.to_dict() for entry in entries],
        "exclusion_policy": list(SOURCE_TREE_EXCLUSION_POLICY),
        "protected_track_paths": list(PROTECTED_TRACK_METADATA_RELATIVE_PATHS),
        "protected_track_tree": PROTECTED_TRACK_TREE_RELATIVE_PATH,
        "schema_version": SOURCE_TREE_IDENTITY_SCHEMA_VERSION,
    }


def _canonical_identity_payload_bytes(entries: tuple[SourceTreeEntry, ...]) -> bytes:
    return json.dumps(
        _canonical_identity_payload(entries),
        allow_nan=False,
        ensure_ascii=True,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("ascii")


def canonical_source_tree_identity_bytes(identity: SourceTreeIdentity) -> bytes:
    """Serialize complete source-tree evidence with stable JSON settings."""

    if not isinstance(identity, SourceTreeIdentity):
        raise TypeError("identity must be a SourceTreeIdentity")
    return json.dumps(
        identity.to_dict(),
        allow_nan=False,
        ensure_ascii=True,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("ascii")


def _is_excluded(relative: PurePosixPath, *, is_directory_hint: bool) -> bool:
    parts = relative.parts
    if not parts:
        return False
    if len(parts) == 1 and parts[0] in _ROOT_EXCLUDED_DIRECTORIES:
        return True
    if is_directory_hint and (
        parts[-1] in _EXCLUDED_DIRECTORY_NAMES or parts[-1].endswith(".egg-info")
    ):
        return True
    path = relative.as_posix()
    if path == "benchmarks/local" or path.startswith("benchmarks/local/"):
        return True
    if path in M8_CENTRAL_OUTPUT_RELATIVE_PATHS:
        return True
    if (
        len(parts) >= 4
        and parts[:3] == ("controller_learning", "assets", "tracks")
        and parts[-1] == "train_pool.npz"
    ):
        return True
    name = parts[-1]
    if name in _EXCLUDED_FILE_NAMES or name.endswith(("~", ".egg-info", ".key", ".pem")):
        return True
    if name != ".env.example" and (name == ".env" or name.startswith(".env.")):
        return True
    return len(parts) == 1 and (name == ".coverage" or name.startswith(".coverage."))


def _is_protected(relative: PurePosixPath) -> bool:
    parts = relative.parts
    return len(parts) >= len(_PROTECTED_TRACK_PARTS) and (
        parts[: len(_PROTECTED_TRACK_PARTS)] == _PROTECTED_TRACK_PARTS
    )


def _is_unsafe_import_artifact(relative: PurePosixPath) -> bool:
    return relative.name.endswith((".pyc", ".pyd", ".pyo", ".so"))


def _same_stat(left: os.stat_result, right: os.stat_result) -> bool:
    return all(getattr(left, field) == getattr(right, field) for field in _STABLE_STAT_FIELDS)


def _node_type(metadata: os.stat_result, relative: PurePosixPath) -> NodeType:
    if stat.S_ISDIR(metadata.st_mode):
        return "directory"
    if stat.S_ISREG(metadata.st_mode):
        return "regular_file"
    if stat.S_ISLNK(metadata.st_mode):
        raise ValueError(f"source tree cannot contain symlinks: {relative.as_posix()}")
    raise ValueError(f"source tree cannot contain special files: {relative.as_posix()}")


def _ordinary_file_entry(
    parent_descriptor: int,
    name: str,
    relative: PurePosixPath,
    before: os.stat_result,
) -> SourceTreeEntry:
    flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0)
    descriptor = os.open(name, flags, dir_fd=parent_descriptor)
    try:
        opened = os.fstat(descriptor)
        if not stat.S_ISREG(opened.st_mode):
            raise ValueError(
                f"source file ceased to be regular while being hashed: {relative.as_posix()}"
            )
        digest = hashlib.sha256()
        size = 0
        while chunk := os.read(descriptor, _READ_CHUNK_SIZE):
            digest.update(chunk)
            size += len(chunk)
        opened_after = os.fstat(descriptor)
    finally:
        os.close(descriptor)
    after = os.stat(name, dir_fd=parent_descriptor, follow_symlinks=False)
    if (
        not _same_stat(before, opened)
        or not _same_stat(opened, opened_after)
        or not _same_stat(opened_after, after)
    ):
        raise RuntimeError(f"source file changed while it was hashed: {relative.as_posix()}")
    if size != opened.st_size:
        raise RuntimeError(f"source file size changed while it was read: {relative.as_posix()}")
    return SourceTreeEntry(
        path=relative.as_posix(),
        node_type="regular_file",
        mode=stat.S_IMODE(opened.st_mode),
        size_bytes=size,
        content_sha256=digest.hexdigest(),
        mtime_ns=None,
        protected_metadata_only=False,
    )


def _metadata_only_entry(
    path: Path,
    relative: PurePosixPath,
    before: os.stat_result,
) -> SourceTreeEntry:
    after = path.lstat()
    if not _same_stat(before, after):
        raise RuntimeError(
            f"protected Track node changed while it was inspected: {relative.as_posix()}"
        )
    return _protected_entry(relative, before)


def _protected_entry(
    relative: PurePosixPath,
    metadata: os.stat_result,
) -> SourceTreeEntry:
    return SourceTreeEntry(
        path=relative.as_posix(),
        node_type=_node_type(metadata, relative),
        mode=stat.S_IMODE(metadata.st_mode),
        size_bytes=metadata.st_size,
        content_sha256=None,
        mtime_ns=metadata.st_mtime_ns,
        protected_metadata_only=True,
    )


def _capture_protected_entries(
    root: Path,
    entries: list[SourceTreeEntry],
) -> None:
    """Capture only declared protected paths using lstat, without directory enumeration."""

    for relative_path in PROTECTED_TRACK_METADATA_RELATIVE_PATHS:
        relative = PurePosixPath(relative_path)
        path = root.joinpath(*relative.parts)
        try:
            metadata = path.lstat()
        except (FileNotFoundError, NotADirectoryError):
            continue
        entries.append(_metadata_only_entry(path, relative, metadata))


def _walk_directory(
    root: Path,
    descriptor: int,
    relative: PurePosixPath,
    opened_metadata: os.stat_result,
    entries: list[SourceTreeEntry],
) -> None:
    if relative.parts:
        entries.append(
            SourceTreeEntry(
                path=relative.as_posix(),
                node_type="directory",
                mode=stat.S_IMODE(opened_metadata.st_mode),
                size_bytes=None,
                content_sha256=None,
                mtime_ns=None,
                protected_metadata_only=False,
            )
        )

    with os.scandir(descriptor) as iterator:
        children = sorted(iterator, key=lambda child: child.name)
    for child in children:
        child_relative = relative / child.name
        is_directory_hint = child.is_dir(follow_symlinks=False)
        if _is_excluded(child_relative, is_directory_hint=is_directory_hint):
            continue
        metadata = child.stat(follow_symlinks=False)
        node_type = _node_type(metadata, child_relative)
        if node_type == "regular_file" and _is_unsafe_import_artifact(child_relative):
            raise ValueError(
                "source tree cannot contain import-capable artifacts outside excluded dynamic "
                f"directories: {child_relative.as_posix()}"
            )
        if _is_protected(child_relative):
            _capture_protected_entries(root, entries)
            continue
        if node_type == "regular_file":
            entries.append(_ordinary_file_entry(descriptor, child.name, child_relative, metadata))
            continue

        flags = (
            os.O_RDONLY
            | getattr(os, "O_CLOEXEC", 0)
            | getattr(os, "O_DIRECTORY", 0)
            | getattr(os, "O_NOFOLLOW", 0)
        )
        child_descriptor = os.open(child.name, flags, dir_fd=descriptor)
        try:
            child_opened = os.fstat(child_descriptor)
            if not stat.S_ISDIR(child_opened.st_mode) or not _same_stat(metadata, child_opened):
                raise RuntimeError(
                    f"source directory changed while it was opened: {child_relative.as_posix()}"
                )
            _walk_directory(
                root,
                child_descriptor,
                child_relative,
                child_opened,
                entries,
            )
            child_after = os.fstat(child_descriptor)
            if not _same_stat(child_opened, child_after):
                raise RuntimeError(
                    f"source directory changed while it was traversed: {child_relative.as_posix()}"
                )
        finally:
            os.close(child_descriptor)


def capture_source_tree_identity(project_root: str | Path) -> SourceTreeIdentity:
    """Capture the non-dynamic tree without opening or enumerating protected Track paths."""

    supplied_root = Path(project_root)
    supplied_metadata = supplied_root.lstat()
    if stat.S_ISLNK(supplied_metadata.st_mode) or not stat.S_ISDIR(supplied_metadata.st_mode):
        raise ValueError("project_root must be a non-symlink directory")
    root = supplied_root.resolve(strict=True)
    flags = (
        os.O_RDONLY
        | getattr(os, "O_CLOEXEC", 0)
        | getattr(os, "O_DIRECTORY", 0)
        | getattr(os, "O_NOFOLLOW", 0)
    )
    descriptor = os.open(root, flags)
    entries: list[SourceTreeEntry] = []
    try:
        opened = os.fstat(descriptor)
        if not stat.S_ISDIR(opened.st_mode) or not _same_stat(supplied_metadata, opened):
            raise RuntimeError("project_root changed while it was opened")
        _walk_directory(
            root,
            descriptor,
            PurePosixPath(),
            opened,
            entries,
        )
        after = os.fstat(descriptor)
        if not _same_stat(opened, after):
            raise RuntimeError("project_root changed while it was traversed")
    finally:
        os.close(descriptor)

    ordered = tuple(sorted(entries, key=lambda entry: entry.path))
    digest = hashlib.sha256(_canonical_identity_payload_bytes(ordered)).hexdigest()
    return SourceTreeIdentity(entries=ordered, aggregate_sha256=digest)


__all__ = [
    "M8_CENTRAL_OUTPUT_RELATIVE_PATHS",
    "PROTECTED_TRACK_METADATA_RELATIVE_PATHS",
    "PROTECTED_TRACK_TREE_RELATIVE_PATH",
    "SOURCE_TREE_EXCLUSION_POLICY",
    "SOURCE_TREE_IDENTITY_SCHEMA_VERSION",
    "SourceTreeEntry",
    "SourceTreeIdentity",
    "canonical_source_tree_identity_bytes",
    "capture_source_tree_identity",
]
