"""Tests for deterministic, Test-content-free M8 source-tree identities."""

from __future__ import annotations

import json
import os
import stat
from pathlib import Path

import pytest

import controller_learning.evaluation.source_tree_identity as source_tree_identity
from controller_learning.evaluation.source_tree_identity import (
    M8_CENTRAL_OUTPUT_RELATIVE_PATHS,
    PROTECTED_TRACK_TREE_RELATIVE_PATH,
    canonical_source_tree_identity_bytes,
    capture_source_tree_identity,
)


def _project(tmp_path: Path) -> Path:
    root = tmp_path / "project"
    root.mkdir()
    (root / "source.py").write_text("VALUE = 1\n", encoding="utf-8")
    protected = root / PROTECTED_TRACK_TREE_RELATIVE_PATH
    protected.mkdir(parents=True)
    (protected / "test.npz").write_bytes(b"official Test bytes must not be opened")
    return root


def _digest(root: Path) -> str:
    return capture_source_tree_identity(root).aggregate_sha256


def test_identity_is_canonical_deterministic_and_path_ordered(tmp_path: Path) -> None:
    root = _project(tmp_path)
    (root / "z-last.txt").write_bytes(b"z")
    (root / "a-first.txt").write_bytes(b"a")

    first = capture_source_tree_identity(root)
    second = capture_source_tree_identity(root)
    canonical = canonical_source_tree_identity_bytes(first)

    assert first == second
    assert tuple(entry.path for entry in first.entries) == tuple(
        sorted(entry.path for entry in first.entries)
    )
    assert json.loads(canonical)["aggregate_sha256"] == first.aggregate_sha256
    assert canonical == canonical_source_tree_identity_bytes(second)


def test_ordinary_source_content_and_metadata_are_bound(tmp_path: Path) -> None:
    root = _project(tmp_path)
    source = root / "source.py"
    before = capture_source_tree_identity(root)
    source_entry = next(entry for entry in before.entries if entry.path == "source.py")

    source.write_text("VALUE = 2\n", encoding="utf-8")
    after_content = capture_source_tree_identity(root)
    source.chmod(stat.S_IMODE(source.stat().st_mode) ^ stat.S_IXUSR)
    after_mode = capture_source_tree_identity(root)

    assert source_entry.node_type == "regular_file"
    assert source_entry.size_bytes == len(b"VALUE = 1\n")
    assert source_entry.content_sha256 is not None
    assert source_entry.mtime_ns is None
    assert before.aggregate_sha256 != after_content.aggregate_sha256
    assert after_content.aggregate_sha256 != after_mode.aggregate_sha256


def test_new_empty_file_and_empty_directory_each_change_identity(tmp_path: Path) -> None:
    root = _project(tmp_path)
    before = _digest(root)
    (root / "empty.txt").touch()
    with_empty_file = capture_source_tree_identity(root)
    (root / "empty-directory").mkdir()
    with_empty_directory = capture_source_tree_identity(root)

    empty_file = next(entry for entry in with_empty_file.entries if entry.path == "empty.txt")
    empty_directory = next(
        entry for entry in with_empty_directory.entries if entry.path == "empty-directory"
    )
    assert before != with_empty_file.aggregate_sha256
    assert with_empty_file.aggregate_sha256 != with_empty_directory.aggregate_sha256
    assert empty_file.size_bytes == 0
    assert empty_directory.node_type == "directory"
    assert empty_directory.size_bytes is None


def test_deleting_an_included_file_changes_identity(tmp_path: Path) -> None:
    root = _project(tmp_path)
    before = _digest(root)
    (root / "source.py").unlink()

    assert _digest(root) != before


def test_protected_track_identity_uses_lstat_metadata_not_content(tmp_path: Path) -> None:
    root = _project(tmp_path)
    protected = root / PROTECTED_TRACK_TREE_RELATIVE_PATH / "test.npz"
    before = capture_source_tree_identity(root)
    entry = next(
        candidate
        for candidate in before.entries
        if candidate.path == f"{PROTECTED_TRACK_TREE_RELATIVE_PATH}/test.npz"
    )
    protected_root_entry = next(
        candidate
        for candidate in before.entries
        if candidate.path == PROTECTED_TRACK_TREE_RELATIVE_PATH
    )
    original_stat = protected.stat()
    replacement = b"x" * original_stat.st_size
    protected.write_bytes(replacement)
    os.utime(protected, ns=(original_stat.st_atime_ns, original_stat.st_mtime_ns))
    same_metadata = capture_source_tree_identity(root)
    os.utime(protected, ns=(original_stat.st_atime_ns, original_stat.st_mtime_ns + 1))
    changed_stat = capture_source_tree_identity(root)

    assert entry.protected_metadata_only is True
    assert entry.content_sha256 is None
    assert entry.size_bytes == original_stat.st_size
    assert entry.mtime_ns == original_stat.st_mtime_ns
    assert protected_root_entry.protected_metadata_only is True
    assert protected_root_entry.size_bytes is not None
    assert protected_root_entry.mtime_ns is not None
    assert same_metadata.aggregate_sha256 == before.aggregate_sha256
    assert changed_stat.aggregate_sha256 != before.aggregate_sha256


def test_capture_never_opens_a_protected_track_file(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root = _project(tmp_path)
    original_open = os.open
    opened_names: list[str] = []

    def audited_open(path: str | bytes | os.PathLike[str] | os.PathLike[bytes], *args, **kwargs):
        name = os.fsdecode(path)
        opened_names.append(name)
        if name in {"v0.1", "test.npz"}:
            raise AssertionError("protected Track node was opened")
        return original_open(path, *args, **kwargs)

    original_scandir = os.scandir

    def audited_scandir(path: int | str | bytes | os.PathLike[str] | os.PathLike[bytes]):
        if not isinstance(path, int) and Path(path) == root / PROTECTED_TRACK_TREE_RELATIVE_PATH:
            raise AssertionError("protected Track tree was enumerated")
        return original_scandir(path)

    monkeypatch.setattr(source_tree_identity.os, "open", audited_open)
    monkeypatch.setattr(source_tree_identity.os, "scandir", audited_scandir)
    first = capture_source_tree_identity(root)

    second = capture_source_tree_identity(root)

    assert second == first
    assert "source.py" in opened_names
    assert "v0.1" not in opened_names
    assert "test.npz" not in opened_names


def test_protected_baseline_directory_metadata_detects_new_paths(tmp_path: Path) -> None:
    root = _project(tmp_path)
    baseline = capture_source_tree_identity(root)
    (root / PROTECTED_TRACK_TREE_RELATIVE_PATH / "new-track.npz").write_bytes(b"new")

    recaptured = capture_source_tree_identity(root)

    assert recaptured.aggregate_sha256 != baseline.aggregate_sha256


@pytest.mark.parametrize(
    "relative,is_directory",
    [
        (".git", True),
        (".pixi", True),
        (".venv", True),
        ("venv", True),
        ("reference", True),
        ("runs", True),
        ("results", True),
        ("dist", True),
        ("site", True),
        ("build", True),
        ("package/__pycache__", True),
        (".pytest_cache", True),
        (".ruff_cache", True),
        (".mypy_cache", True),
        (".hypothesis", True),
        ("htmlcov", True),
        (".coverage", False),
        (".coverage.worker", False),
        ("coverage.xml", False),
        (".idea", True),
        (".vscode", True),
        ("notes.txt~", False),
        (".env", False),
        (".env.local", False),
        ("certificate.pem", False),
        ("private.key", False),
        ("benchmarks/local", True),
        (f"{PROTECTED_TRACK_TREE_RELATIVE_PATH}/train_pool.npz", False),
    ],
)
def test_explicit_dynamic_exclusions_do_not_change_identity(
    tmp_path: Path,
    relative: str,
    is_directory: bool,
) -> None:
    root = _project(tmp_path)
    first_part = Path(relative).parts[0]
    if len(Path(relative).parts) > 1 and first_part not in {
        "controller_learning",
        "runs",
        "results",
    }:
        (root / first_part).mkdir(exist_ok=True)
    path = root / relative
    protected_train_pool = relative == f"{PROTECTED_TRACK_TREE_RELATIVE_PATH}/train_pool.npz"
    if protected_train_pool:
        path.write_bytes(b"initial dynamic pool")
    before = _digest(root)
    if is_directory:
        path.mkdir(parents=True, exist_ok=True)
        (path / "dynamic.bin").write_bytes(b"dynamic")
    else:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"changed dynamic" if protected_train_pool else b"dynamic")

    assert _digest(root) == before


def test_env_example_remains_part_of_the_source_tree(tmp_path: Path) -> None:
    root = _project(tmp_path)
    before = _digest(root)
    (root / ".env.example").write_text("SETTING=example\n", encoding="utf-8")

    assert _digest(root) != before


def test_central_outputs_and_snapshot_quarantine_do_not_change_identity(tmp_path: Path) -> None:
    root = _project(tmp_path)
    (root / "benchmarks/v0.1").mkdir(parents=True)
    before = _digest(root)
    for relative in M8_CENTRAL_OUTPUT_RELATIVE_PATHS:
        output = root / relative
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_bytes(relative.encode("ascii"))
    active = root / "runs/m8_final_controller_snapshot/controller.py"
    quarantine = root / "runs/m8_final_controller_snapshot.committed/controller.py"
    active.parent.mkdir(parents=True)
    quarantine.parent.mkdir(parents=True)
    active.write_bytes(b"active")
    quarantine.write_bytes(b"committed")

    assert _digest(root) == before


@pytest.mark.parametrize("node_type", ["symlink", "fifo"])
def test_included_symlink_and_special_file_are_rejected(
    tmp_path: Path,
    node_type: str,
) -> None:
    root = _project(tmp_path)
    node = root / "forbidden"
    if node_type == "symlink":
        node.symlink_to("source.py")
        expected = "symlinks"
    else:
        os.mkfifo(node)
        expected = "special files"

    with pytest.raises(ValueError, match=expected):
        capture_source_tree_identity(root)


@pytest.mark.parametrize(
    "name",
    [
        "shadow.cpython-311-x86_64-linux-gnu.so",
        "shadow.pyd",
        "shadow.pyc",
        "shadow.pyo",
    ],
)
def test_root_import_capable_artifacts_are_rejected(tmp_path: Path, name: str) -> None:
    root = _project(tmp_path)
    (root / name).write_bytes(b"unsafe import artifact")

    with pytest.raises(ValueError, match="import-capable artifacts"):
        capture_source_tree_identity(root)


def test_import_artifacts_inside_excluded_dynamic_directories_are_ignored(
    tmp_path: Path,
) -> None:
    root = _project(tmp_path)
    (root / "package").mkdir()
    before = _digest(root)
    for directory in (".pixi", "build", "package/__pycache__"):
        target = root / directory
        target.mkdir(parents=True)
        for name in ("extension.so", "extension.pyd", "module.pyc", "module.pyo"):
            (target / name).write_bytes(b"dynamic")

    assert _digest(root) == before


def test_directory_mode_is_bound(tmp_path: Path) -> None:
    root = _project(tmp_path)
    directory = root / "ordinary-directory"
    directory.mkdir()
    before = _digest(root)
    directory.chmod(stat.S_IMODE(directory.stat().st_mode) ^ stat.S_IXGRP)

    assert _digest(root) != before
