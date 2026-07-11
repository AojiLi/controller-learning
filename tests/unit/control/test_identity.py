"""Tests for trusted Controller directory content identities."""

from __future__ import annotations

from pathlib import Path

import pytest

from controller_learning.control.identity import capture_controller_directory_identity


def _controller(directory: Path) -> Path:
    directory.mkdir()
    (directory / "controller.py").write_text("VALUE = 1\n", encoding="utf-8")
    (directory / "config.toml").write_text('name = "example"\n', encoding="utf-8")
    helpers = directory / "helpers"
    helpers.mkdir()
    (helpers / "math.py").write_text("GAIN = 2.0\n", encoding="utf-8")
    return directory


def test_identity_hashes_nested_files_and_ignores_bytecode_cache(tmp_path: Path) -> None:
    directory = _controller(tmp_path / "controller")
    cache = directory / "__pycache__"
    cache.mkdir()
    (cache / "controller.cpython-311.pyc").write_bytes(b"runtime bytecode")

    identity = capture_controller_directory_identity(directory)

    assert tuple(item.path for item in identity.files) == (
        "config.toml",
        "controller.py",
        "helpers/math.py",
    )
    assert identity.to_dict()["aggregate_sha256"] == identity.aggregate_sha256


def test_identity_changes_when_controller_content_changes(tmp_path: Path) -> None:
    directory = _controller(tmp_path / "controller")
    before = capture_controller_directory_identity(directory)

    (directory / "controller.py").write_text("VALUE = 2\n", encoding="utf-8")
    after = capture_controller_directory_identity(directory)

    assert before.aggregate_sha256 != after.aggregate_sha256


def test_identity_rejects_symlinks_and_empty_directories(tmp_path: Path) -> None:
    empty = tmp_path / "empty"
    empty.mkdir()
    with pytest.raises(ValueError, match="cannot be empty"):
        capture_controller_directory_identity(empty)

    directory = _controller(tmp_path / "controller")
    directory_link = tmp_path / "controller-link"
    directory_link.symlink_to(directory, target_is_directory=True)
    with pytest.raises(ValueError, match="non-symlink directory"):
        capture_controller_directory_identity(directory_link)

    (directory / "external.py").symlink_to(directory / "controller.py")
    with pytest.raises(ValueError, match="cannot contain symlinks"):
        capture_controller_directory_identity(directory)
