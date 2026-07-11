"""Frozen whole-plugin identities for the M8 PID/MPC/PPO comparison."""

from __future__ import annotations

import hashlib
import json
import os
import stat
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType
from typing import Final

M8_CONTROLLER_FILE_MANIFEST: Final = MappingProxyType(
    {
        "pid": ("README.md", "config.toml", "controller.py", "helpers.py"),
        "mpc": ("README.md", "config.toml", "controller.py", "helpers.py", "solver.py"),
        "ppo": ("README.md", "config.toml", "controller.py", "metadata.json", "policy.npz"),
    }
)


@dataclass(frozen=True, slots=True)
class ControllerFileIdentity:
    """Content identity of one required regular file inside a Controller plugin."""

    path: str
    sha256: str
    size_bytes: int

    def to_dict(self) -> dict[str, object]:
        return {"path": self.path, "sha256": self.sha256, "size_bytes": self.size_bytes}


@dataclass(frozen=True, slots=True)
class FrozenControllerIdentity:
    """Complete content identity of one frozen M8 Controller directory."""

    controller: str
    directory: str
    files: tuple[ControllerFileIdentity, ...]
    aggregate_sha256: str

    def __post_init__(self) -> None:
        expected = M8_CONTROLLER_FILE_MANIFEST.get(self.controller)
        if expected is None:
            raise ValueError("controller must be one of pid, mpc, or ppo")
        if self.directory != f"controllers/{self.controller}":
            raise ValueError("Controller directory does not match its frozen public path")
        if tuple(item.path for item in self.files) != expected:
            raise ValueError("Controller files differ from the frozen whole-plugin manifest")
        if len(self.aggregate_sha256) != 64 or any(
            character not in "0123456789abcdef" for character in self.aggregate_sha256
        ):
            raise ValueError("aggregate_sha256 must be a lowercase SHA-256 digest")

    @property
    def config_sha256(self) -> str:
        return next(item.sha256 for item in self.files if item.path == "config.toml")

    def to_dict(self) -> dict[str, object]:
        return {
            "aggregate_sha256": self.aggregate_sha256,
            "config_sha256": self.config_sha256,
            "controller": self.controller,
            "directory": self.directory,
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


def _visible_controller_files(directory: Path) -> tuple[str, ...]:
    observed: list[str] = []
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
        observed.append(relative.as_posix())
    return tuple(sorted(observed))


def capture_frozen_controller_identity(
    project_root: str | Path,
    controller: str,
) -> FrozenControllerIdentity:
    """Hash every required file and reject whole-plugin manifest drift."""

    expected = M8_CONTROLLER_FILE_MANIFEST.get(controller)
    if expected is None:
        raise ValueError("controller must be one of pid, mpc, or ppo")
    root = Path(project_root).resolve(strict=True)
    directory = root / "controllers" / controller
    if directory.is_symlink() or not directory.is_dir():
        raise ValueError("frozen Controller directory must be a non-symlink directory")
    if _visible_controller_files(directory) != expected:
        raise ValueError(
            "Controller directory contents differ from the frozen whole-plugin manifest"
        )

    files: list[ControllerFileIdentity] = []
    for relative in expected:
        content = _stable_regular_file_bytes(directory / relative)
        files.append(
            ControllerFileIdentity(
                path=relative,
                sha256=hashlib.sha256(content).hexdigest(),
                size_bytes=len(content),
            )
        )
    canonical = json.dumps(
        [item.to_dict() for item in files],
        ensure_ascii=True,
        allow_nan=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("ascii")
    return FrozenControllerIdentity(
        controller=controller,
        directory=f"controllers/{controller}",
        files=tuple(files),
        aggregate_sha256=hashlib.sha256(canonical).hexdigest(),
    )


__all__ = [
    "M8_CONTROLLER_FILE_MANIFEST",
    "ControllerFileIdentity",
    "FrozenControllerIdentity",
    "capture_frozen_controller_identity",
]
