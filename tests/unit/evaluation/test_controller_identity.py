"""Tests for frozen whole-plugin M8 identities."""

from __future__ import annotations

import shutil
from pathlib import Path

import pytest

from controller_learning.evaluation.controller_identity import (
    M8_CONTROLLER_FILE_MANIFEST,
    capture_frozen_controller_identity,
)

PROJECT_ROOT = Path(__file__).resolve().parents[3]


@pytest.mark.parametrize("controller", tuple(M8_CONTROLLER_FILE_MANIFEST))
def test_capture_frozen_controller_identity_hashes_every_required_file(controller: str) -> None:
    identity = capture_frozen_controller_identity(PROJECT_ROOT, controller)

    assert tuple(item.path for item in identity.files) == M8_CONTROLLER_FILE_MANIFEST[controller]
    assert identity.directory == f"controllers/{controller}"
    assert identity.config_sha256 == next(
        item.sha256 for item in identity.files if item.path == "config.toml"
    )


def _copy_controller(tmp_path: Path, controller: str) -> Path:
    target_root = tmp_path / "project"
    target = target_root / "controllers" / controller
    target.parent.mkdir(parents=True)
    shutil.copytree(PROJECT_ROOT / "controllers" / controller, target)
    return target_root


def test_identity_rejects_an_unhashed_helper(tmp_path: Path) -> None:
    root = _copy_controller(tmp_path, "pid")
    (root / "controllers" / "pid" / "unfrozen_helper.py").write_text(
        "VALUE = 1\n", encoding="utf-8"
    )

    with pytest.raises(ValueError, match="whole-plugin manifest"):
        capture_frozen_controller_identity(root, "pid")


def test_identity_changes_when_a_controller_file_mutates(tmp_path: Path) -> None:
    root = _copy_controller(tmp_path, "ppo")
    before = capture_frozen_controller_identity(root, "ppo")
    config = root / "controllers" / "ppo" / "config.toml"
    config.write_bytes(config.read_bytes() + b"\n")
    after = capture_frozen_controller_identity(root, "ppo")

    assert before.aggregate_sha256 != after.aggregate_sha256
    assert before.config_sha256 != after.config_sha256


def test_identity_rejects_a_symlinked_required_file(tmp_path: Path) -> None:
    root = _copy_controller(tmp_path, "pid")
    helper = root / "controllers" / "pid" / "helpers.py"
    helper.unlink()
    helper.symlink_to("controller.py")

    with pytest.raises(ValueError, match="symlinks"):
        capture_frozen_controller_identity(root, "pid")
