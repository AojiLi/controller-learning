"""Content identities for trusted local Controller directories."""

from __future__ import annotations

import hashlib
import json
import os
import stat
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True, slots=True)
class ControllerFileIdentity:
    """Stable content identity for one regular file in a Controller plugin."""

    path: str
    sha256: str
    size_bytes: int

    def to_dict(self) -> dict[str, str | int]:
        """Return a JSON-compatible representation."""

        return {"path": self.path, "sha256": self.sha256, "size_bytes": self.size_bytes}


@dataclass(frozen=True, slots=True)
class ControllerDirectoryIdentity:
    """Whole-directory identity independent of the Controller's filesystem location."""

    files: tuple[ControllerFileIdentity, ...]
    aggregate_sha256: str

    def __post_init__(self) -> None:
        if not self.files:
            raise ValueError("Controller identity must contain at least one file")
        paths = tuple(item.path for item in self.files)
        if paths != tuple(sorted(paths)) or len(paths) != len(set(paths)):
            raise ValueError("Controller identity files must be uniquely sorted by path")
        if len(self.aggregate_sha256) != 64 or any(
            character not in "0123456789abcdef" for character in self.aggregate_sha256
        ):
            raise ValueError("aggregate_sha256 must be a lowercase SHA-256 digest")

    def to_dict(self) -> dict[str, object]:
        """Return a canonical JSON-compatible representation."""

        return {
            "aggregate_sha256": self.aggregate_sha256,
            "files": [item.to_dict() for item in self.files],
        }


def _stable_regular_file_bytes(path: Path) -> bytes:
    before = path.lstat()
    if stat.S_ISLNK(before.st_mode) or not stat.S_ISREG(before.st_mode):
        raise ValueError(f"Controller input must be a non-symlink regular file: {path.name}")
    descriptor = os.open(path, os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0))
    try:
        opened = os.fstat(descriptor)
        if not stat.S_ISREG(opened.st_mode):
            raise ValueError(f"Controller input ceased to be a regular file: {path.name}")
        with os.fdopen(descriptor, "rb", closefd=False) as stream:
            content = stream.read()
    finally:
        os.close(descriptor)
    after = path.lstat()
    stable_fields = ("st_dev", "st_ino", "st_mode", "st_size", "st_mtime_ns")
    if any(getattr(before, field) != getattr(opened, field) for field in stable_fields) or any(
        getattr(opened, field) != getattr(after, field) for field in stable_fields
    ):
        raise RuntimeError(f"Controller file changed while it was hashed: {path.name}")
    if len(content) != opened.st_size:
        raise RuntimeError(f"Controller file size changed while it was read: {path.name}")
    return content


def capture_controller_directory_identity(
    controller_directory: str | Path,
) -> ControllerDirectoryIdentity:
    """Hash every visible regular file in one trusted Controller directory.

    Bytecode caches are runtime products and do not contribute to the identity. Symlinks and
    special files are rejected so the identity cannot silently depend on content outside the
    declared plugin directory.
    """

    source = Path(controller_directory).expanduser()
    if source.is_symlink():
        raise ValueError("Controller directory must be a non-symlink directory")
    directory = source.resolve(strict=True)
    if not directory.is_dir():
        raise ValueError("Controller directory must be a non-symlink directory")

    paths: list[Path] = []
    for path in directory.rglob("*"):
        relative = path.relative_to(directory)
        if "__pycache__" in relative.parts or path.suffix == ".pyc":
            continue
        metadata = path.lstat()
        if stat.S_ISLNK(metadata.st_mode):
            raise ValueError(f"Controller directory cannot contain symlinks: {relative.as_posix()}")
        if stat.S_ISDIR(metadata.st_mode):
            continue
        if not stat.S_ISREG(metadata.st_mode):
            raise ValueError(
                f"Controller directory cannot contain special files: {relative.as_posix()}"
            )
        paths.append(path)
    if not paths:
        raise ValueError("Controller directory cannot be empty")

    files: list[ControllerFileIdentity] = []
    for path in sorted(paths, key=lambda item: item.relative_to(directory).as_posix()):
        relative = path.relative_to(directory).as_posix()
        content = _stable_regular_file_bytes(path)
        files.append(
            ControllerFileIdentity(
                path=relative,
                sha256=hashlib.sha256(content).hexdigest(),
                size_bytes=len(content),
            )
        )
    canonical = json.dumps(
        [item.to_dict() for item in files],
        allow_nan=False,
        ensure_ascii=True,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("ascii")
    return ControllerDirectoryIdentity(
        files=tuple(files),
        aggregate_sha256=hashlib.sha256(canonical).hexdigest(),
    )


__all__ = [
    "ControllerDirectoryIdentity",
    "ControllerFileIdentity",
    "capture_controller_directory_identity",
]
