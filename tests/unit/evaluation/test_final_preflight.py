"""Tests for asset-free M8 source, report, and Controller snapshot preflight."""

from __future__ import annotations

import os
import shutil
import socket
import stat
from pathlib import Path

import pytest

import controller_learning.evaluation.final_preflight as final_preflight
from controller_learning.evaluation.final_benchmark import load_m8_final_evaluation_config
from controller_learning.evaluation.final_preflight import (
    M8_ABORTED_CONTROLLER_SNAPSHOT_PREFIX,
    M8_COMMITTED_CONTROLLER_SNAPSHOT_RELATIVE_PATH,
    M8_CONTROLLER_SNAPSHOT_RELATIVE_PATH,
    capture_clean_source,
    create_frozen_controller_snapshot,
    frozen_input_digest,
    isolate_aborted_controller_snapshot,
    load_frozen_input_reports,
    require_controller_snapshot_quarantine_absent,
    retire_committed_controller_snapshot,
    validate_committed_controller_snapshot_quarantine,
    validate_frozen_controller_snapshot,
)

PROJECT_ROOT = Path(__file__).resolve().parents[3]
CONFIG = load_m8_final_evaluation_config(PROJECT_ROOT / "configs/final_evaluation.toml")


def test_clean_source_requires_full_revision_and_empty_status(tmp_path: Path) -> None:
    calls: list[tuple[str, ...]] = []

    def command(command: tuple[str, ...], cwd: Path) -> str:
        assert cwd == tmp_path
        calls.append(command)
        return "1" * 40 if "rev-parse" in command else ""

    snapshot = capture_clean_source(tmp_path, command_runner=command)
    assert snapshot.revision == "1" * 40
    assert snapshot.worktree_clean is True
    assert len(calls) == 2

    def dirty(command: tuple[str, ...], _cwd: Path) -> str:
        return "1" * 40 if "rev-parse" in command else " M tracked.py"

    with pytest.raises(RuntimeError, match="clean Git worktree"):
        capture_clean_source(tmp_path, command_runner=dirty)


def test_frozen_input_reports_validate_current_m5_m6_m7_and_predecessor_chain() -> None:
    reports = load_frozen_input_reports(PROJECT_ROOT, CONFIG)

    assert tuple(reports) == tuple(CONFIG.input_paths)
    assert "m8_attempt_001_failure_report" in reports
    assert reports["m8_attempt_001_failure_report"].sha256 == (
        CONFIG.replacement_failure_report_sha256
    )
    assert len(frozen_input_digest(reports)) == 64
    assert reports["m5_track_admission_report"].payload["status"] == "pass"
    assert reports["m7_controller_report"].payload["runtime"]["selected_gpu"]["uuid"] == (
        "redacted"
    )


def test_controller_snapshot_is_read_only_hash_bound_and_abort_isolatable(tmp_path: Path) -> None:
    shutil.copytree(PROJECT_ROOT / "controllers", tmp_path / "controllers")
    snapshot = create_frozen_controller_snapshot(tmp_path, CONFIG)
    try:
        validate_frozen_controller_snapshot(snapshot, CONFIG)
        assert tuple(snapshot.directories) == ("pid", "mpc", "ppo")
        assert all(path.is_dir() for path in snapshot.directories.values())
        target = snapshot.directories["pid"] / "config.toml"
        target.chmod(0o644)
        target.write_bytes(target.read_bytes() + b"\n")
        with pytest.raises(RuntimeError, match="snapshot changed"):
            validate_frozen_controller_snapshot(snapshot, CONFIG)
    finally:
        quarantine = isolate_aborted_controller_snapshot(tmp_path)
        assert quarantine is not None
    assert not snapshot.root.exists()


def test_aborted_snapshot_isolation_moves_partial_tree_without_traversal(tmp_path: Path) -> None:
    active = tmp_path / M8_CONTROLLER_SNAPSHOT_RELATIVE_PATH
    partial = active / "controllers/pid/controller.py"
    partial.parent.mkdir(parents=True)
    partial.write_bytes(b"partial")

    quarantine = isolate_aborted_controller_snapshot(tmp_path)

    assert quarantine is not None
    assert quarantine.name.startswith(M8_ABORTED_CONTROLLER_SNAPSHOT_PREFIX)
    assert not active.exists()
    assert (quarantine / "controllers/pid/controller.py").read_bytes() == b"partial"
    assert isolate_aborted_controller_snapshot(tmp_path) is None


def test_snapshot_creation_failure_leaves_partial_tree_for_caller_isolation(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    shutil.copytree(PROJECT_ROOT / "controllers", tmp_path / "controllers")
    original_write = final_preflight._write_snapshot_file_at
    write_count = 0

    def fail_after_first_write(parent_descriptor: int, name: str, content: bytes) -> None:
        nonlocal write_count
        original_write(parent_descriptor, name, content)
        write_count += 1
        if write_count == 1:
            raise OSError("simulated snapshot creation crash")

    monkeypatch.setattr(final_preflight, "_write_snapshot_file_at", fail_after_first_write)
    with pytest.raises(OSError, match="creation crash"):
        create_frozen_controller_snapshot(tmp_path, CONFIG)
    active = tmp_path / M8_CONTROLLER_SNAPSHOT_RELATIVE_PATH
    assert active.is_dir()
    assert any(active.rglob("*"))

    quarantine = isolate_aborted_controller_snapshot(tmp_path)
    assert quarantine is not None
    assert quarantine.is_dir()
    assert not active.exists()


def test_snapshot_writer_does_not_follow_replaced_parent_symlink(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    shutil.copytree(PROJECT_ROOT / "controllers", tmp_path / "controllers")
    active = tmp_path / M8_CONTROLLER_SNAPSHOT_RELATIVE_PATH
    displaced = tmp_path / "displaced-active-snapshot"
    external = tmp_path / "external"
    external.mkdir()
    marker = external / "marker.txt"
    marker.write_bytes(b"untouched")
    marker.chmod(0o440)
    original_write = final_preflight._write_snapshot_file_at
    replaced = False

    def replace_parent_then_write(parent_descriptor: int, name: str, content: bytes) -> None:
        nonlocal replaced
        if not replaced:
            replaced = True
            active.rename(displaced)
            active.symlink_to(external, target_is_directory=True)
        original_write(parent_descriptor, name, content)

    monkeypatch.setattr(
        final_preflight,
        "_write_snapshot_file_at",
        replace_parent_then_write,
    )
    with pytest.raises((FileNotFoundError, RuntimeError, ValueError)):
        create_frozen_controller_snapshot(tmp_path, CONFIG)

    assert replaced is True
    assert marker.read_bytes() == b"untouched"
    assert stat.S_IMODE(marker.stat().st_mode) == 0o440
    assert not (external / "controllers").exists()
    quarantine = isolate_aborted_controller_snapshot(tmp_path)
    assert quarantine is not None
    assert quarantine.is_symlink()


@pytest.mark.parametrize("node_type", ["symlink", "fifo", "socket"])
def test_aborted_snapshot_isolation_moves_special_entry_without_following_it(
    tmp_path: Path,
    node_type: str,
) -> None:
    active = tmp_path / M8_CONTROLLER_SNAPSHOT_RELATIVE_PATH
    active.parent.mkdir()
    external = tmp_path / "external"
    external.mkdir()
    target = external / "target.txt"
    target.write_bytes(b"untouched")
    target.chmod(0o440)
    if node_type == "symlink":
        active.symlink_to(external, target_is_directory=True)
    elif node_type == "fifo":
        os.mkfifo(active)
    else:
        socket_path = tmp_path / "abort.sock"
        endpoint = socket.socket(socket.AF_UNIX)
        try:
            endpoint.bind(str(socket_path))
        finally:
            endpoint.close()
        socket_path.rename(active)

    quarantine = isolate_aborted_controller_snapshot(tmp_path)

    assert quarantine is not None
    moved = quarantine
    metadata = moved.lstat()
    assert {
        "symlink": stat.S_ISLNK,
        "fifo": stat.S_ISFIFO,
        "socket": stat.S_ISSOCK,
    }[node_type](metadata.st_mode)
    assert target.read_bytes() == b"untouched"
    assert stat.S_IMODE(target.stat().st_mode) == 0o440


def test_aborted_snapshot_isolation_uses_unique_containers(tmp_path: Path) -> None:
    active = tmp_path / M8_CONTROLLER_SNAPSHOT_RELATIVE_PATH
    active.parent.mkdir()
    active.write_bytes(b"first")
    first = isolate_aborted_controller_snapshot(tmp_path)
    active.write_bytes(b"second")
    second = isolate_aborted_controller_snapshot(tmp_path)

    assert first is not None and second is not None and first != second
    assert first.read_bytes() == b"first"
    assert second.read_bytes() == b"second"


def test_aborted_snapshot_isolation_recovers_after_rename_before_fsync_crash(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    active = tmp_path / M8_CONTROLLER_SNAPSHOT_RELATIVE_PATH
    active.mkdir(parents=True)
    original_fsync = os.fsync
    crash_once = True

    def fail_first_fsync(descriptor: int) -> None:
        nonlocal crash_once
        if crash_once:
            crash_once = False
            raise OSError("simulated abort quarantine fsync crash")
        original_fsync(descriptor)

    monkeypatch.setattr(os, "fsync", fail_first_fsync)
    with pytest.raises(OSError, match="fsync crash"):
        isolate_aborted_controller_snapshot(tmp_path)
    assert not active.exists()
    quarantines = tuple(
        path
        for path in (tmp_path / "runs").iterdir()
        if path.name.startswith(M8_ABORTED_CONTROLLER_SNAPSHOT_PREFIX)
    )
    assert len(quarantines) == 1
    assert quarantines[0].is_dir()

    assert isolate_aborted_controller_snapshot(tmp_path) is None


def test_aborted_snapshot_isolation_rejects_runs_symlink(tmp_path: Path) -> None:
    external = tmp_path / "external"
    external.mkdir()
    (tmp_path / "runs").symlink_to(external, target_is_directory=True)

    with pytest.raises(RuntimeError, match="real directory"):
        isolate_aborted_controller_snapshot(tmp_path)


def test_committed_snapshot_retirement_atomically_quarantines_the_whole_tree(
    tmp_path: Path,
) -> None:
    snapshot_root = tmp_path / M8_CONTROLLER_SNAPSHOT_RELATIVE_PATH
    quarantine = tmp_path / M8_COMMITTED_CONTROLLER_SNAPSHOT_RELATIVE_PATH
    nested = snapshot_root / "controllers/pid"
    nested.mkdir(parents=True)
    partial = nested / "controller.py"
    partial.write_bytes(b"partial snapshot")
    external = tmp_path / "external-controller.py"
    external.write_bytes(b"shared inode")
    external.chmod(0o440)
    os.link(external, nested / "hard-linked.py")
    nested.chmod(0o500)
    snapshot_root.chmod(0o500)
    source_inode = snapshot_root.lstat().st_ino

    retire_committed_controller_snapshot(tmp_path)

    assert not snapshot_root.exists()
    assert quarantine.lstat().st_ino == source_inode
    assert (quarantine / "controllers/pid/controller.py").read_bytes() == b"partial snapshot"
    assert external.read_bytes() == b"shared inode"
    assert stat.S_IMODE(external.stat().st_mode) == 0o440
    retire_committed_controller_snapshot(tmp_path)


def test_committed_snapshot_retirement_rejects_when_both_states_are_absent(
    tmp_path: Path,
) -> None:
    with pytest.raises(RuntimeError, match="exactly one"):
        retire_committed_controller_snapshot(tmp_path)


def test_committed_snapshot_retirement_rejects_active_and_quarantine_together(
    tmp_path: Path,
) -> None:
    active = tmp_path / M8_CONTROLLER_SNAPSHOT_RELATIVE_PATH
    quarantine = tmp_path / M8_COMMITTED_CONTROLLER_SNAPSHOT_RELATIVE_PATH
    active.mkdir(parents=True)
    quarantine.mkdir()

    with pytest.raises(RuntimeError, match="exactly one"):
        retire_committed_controller_snapshot(tmp_path)
    assert active.is_dir()
    assert quarantine.is_dir()


def test_snapshot_quarantine_gates_reject_a_runs_symlink(
    tmp_path: Path,
) -> None:
    external = tmp_path / "external"
    external.mkdir()
    (external / "keep.txt").write_text("keep", encoding="utf-8")
    runs = tmp_path / "runs"
    runs.symlink_to(external, target_is_directory=True)

    with pytest.raises(RuntimeError, match="real directory"):
        retire_committed_controller_snapshot(tmp_path)
    with pytest.raises(RuntimeError, match="real directory"):
        require_controller_snapshot_quarantine_absent(tmp_path)
    assert (external / "keep.txt").read_text(encoding="utf-8") == "keep"


@pytest.mark.parametrize("node_type", ["symlink", "fifo", "socket"])
def test_committed_snapshot_retirement_rejects_special_active_source(
    tmp_path: Path,
    node_type: str,
) -> None:
    active = tmp_path / M8_CONTROLLER_SNAPSHOT_RELATIVE_PATH
    quarantine = tmp_path / M8_COMMITTED_CONTROLLER_SNAPSHOT_RELATIVE_PATH
    active.parent.mkdir()
    external = tmp_path / "external"
    external.mkdir()
    target = external / "target.txt"
    target.write_bytes(b"untouched")
    target.chmod(0o440)
    if node_type == "symlink":
        active.symlink_to(external, target_is_directory=True)
    elif node_type == "fifo":
        os.mkfifo(active)
    else:
        socket_path = tmp_path / "active.sock"
        endpoint = socket.socket(socket.AF_UNIX)
        try:
            endpoint.bind(str(socket_path))
        finally:
            endpoint.close()
        socket_path.rename(active)

    with pytest.raises(RuntimeError, match="must be a real directory"):
        retire_committed_controller_snapshot(tmp_path)

    assert active.exists() or active.is_symlink()
    assert not quarantine.exists() and not quarantine.is_symlink()
    assert target.read_bytes() == b"untouched"
    assert stat.S_IMODE(target.stat().st_mode) == 0o440


def test_committed_snapshot_retirement_does_not_follow_child_symlinks(tmp_path: Path) -> None:
    active = tmp_path / M8_CONTROLLER_SNAPSHOT_RELATIVE_PATH
    quarantine = tmp_path / M8_COMMITTED_CONTROLLER_SNAPSHOT_RELATIVE_PATH
    active.mkdir(parents=True)
    external = tmp_path / "external"
    external.mkdir()
    target = external / "outside.py"
    target.write_bytes(b"must survive")
    (active / "outside").symlink_to(external, target_is_directory=True)

    retire_committed_controller_snapshot(tmp_path)

    assert (quarantine / "outside").is_symlink()
    assert target.read_bytes() == b"must survive"


def test_snapshot_quarantine_absence_gate_rejects_prior_committed_state(tmp_path: Path) -> None:
    require_controller_snapshot_quarantine_absent(tmp_path)
    quarantine = tmp_path / M8_COMMITTED_CONTROLLER_SNAPSHOT_RELATIVE_PATH
    quarantine.parent.mkdir()
    quarantine.write_bytes(b"prior committed snapshot")

    with pytest.raises(RuntimeError, match="quarantine already exists"):
        require_controller_snapshot_quarantine_absent(tmp_path)


def test_committed_snapshot_retirement_recovers_after_rename_before_fsync_crash(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    active = tmp_path / M8_CONTROLLER_SNAPSHOT_RELATIVE_PATH
    quarantine = tmp_path / M8_COMMITTED_CONTROLLER_SNAPSHOT_RELATIVE_PATH
    active.mkdir(parents=True)
    original_fsync = os.fsync
    crash_once = True

    def fail_first_fsync(descriptor: int) -> None:
        nonlocal crash_once
        if crash_once:
            crash_once = False
            raise OSError("simulated crash before runs fsync")
        original_fsync(descriptor)

    monkeypatch.setattr(os, "fsync", fail_first_fsync)
    with pytest.raises(OSError, match="simulated crash"):
        retire_committed_controller_snapshot(tmp_path)
    assert not active.exists()
    assert quarantine.exists()

    retire_committed_controller_snapshot(tmp_path)
    assert quarantine.exists()


@pytest.mark.parametrize("node_type", ["symlink", "fifo", "socket"])
def test_committed_snapshot_retirement_rejects_special_quarantine_state(
    tmp_path: Path,
    node_type: str,
) -> None:
    quarantine = tmp_path / M8_COMMITTED_CONTROLLER_SNAPSHOT_RELATIVE_PATH
    quarantine.parent.mkdir()
    external = tmp_path / "external"
    external.mkdir()
    if node_type == "symlink":
        quarantine.symlink_to(external, target_is_directory=True)
    elif node_type == "fifo":
        os.mkfifo(quarantine)
    else:
        socket_path = tmp_path / "quarantine.sock"
        endpoint = socket.socket(socket.AF_UNIX)
        try:
            endpoint.bind(str(socket_path))
        finally:
            endpoint.close()
        socket_path.rename(quarantine)

    with pytest.raises(RuntimeError, match="must be a real directory"):
        retire_committed_controller_snapshot(tmp_path)


def test_committed_snapshot_outer_validation_requires_only_real_quarantine(
    tmp_path: Path,
) -> None:
    active = tmp_path / M8_CONTROLLER_SNAPSHOT_RELATIVE_PATH
    active.mkdir(parents=True)
    with pytest.raises(RuntimeError, match="active Controller snapshot"):
        validate_committed_controller_snapshot_quarantine(tmp_path)

    retire_committed_controller_snapshot(tmp_path)
    validate_committed_controller_snapshot_quarantine(tmp_path)
    active.mkdir()
    with pytest.raises(RuntimeError, match="active Controller snapshot"):
        validate_committed_controller_snapshot_quarantine(tmp_path)


def test_parameterized_attempt_002_snapshot_operations_preserve_predecessor_bytes(
    tmp_path: Path,
) -> None:
    predecessor = tmp_path / "runs/m8_final_controller_snapshot"
    predecessor_file = predecessor / "controllers/pid/controller.py"
    predecessor_file.parent.mkdir(parents=True)
    predecessor_file.write_bytes(b"attempt-001-controller\x00\xff")
    predecessor_committed = tmp_path / "runs/m8_final_controller_snapshot.committed"
    predecessor_committed.mkdir()
    (predecessor_committed / "evidence.bin").write_bytes(b"attempt-001-committed")
    predecessor_before = predecessor_file.read_bytes()
    predecessor_committed_before = (predecessor_committed / "evidence.bin").read_bytes()

    active_relative_path = "runs/custom_controller_snapshot_002"
    committed_relative_path = active_relative_path + ".committed"
    active = tmp_path / active_relative_path
    active.mkdir()
    (active / "partial.bin").write_bytes(b"attempt-002-partial")

    require_controller_snapshot_quarantine_absent(
        tmp_path,
        committed_relative_path=committed_relative_path,
    )
    aborted = isolate_aborted_controller_snapshot(
        tmp_path,
        relative_path=active_relative_path,
    )
    assert aborted is not None
    assert aborted.name.startswith("custom_controller_snapshot_002.abort.")
    assert (aborted / "partial.bin").read_bytes() == b"attempt-002-partial"

    active.mkdir()
    (active / "complete.bin").write_bytes(b"attempt-002-complete")
    retire_committed_controller_snapshot(
        tmp_path,
        relative_path=active_relative_path,
        committed_relative_path=committed_relative_path,
    )
    validate_committed_controller_snapshot_quarantine(
        tmp_path,
        relative_path=active_relative_path,
        committed_relative_path=committed_relative_path,
    )

    assert predecessor_file.read_bytes() == predecessor_before
    assert (predecessor_committed / "evidence.bin").read_bytes() == (predecessor_committed_before)


def test_parameterized_snapshot_rejects_a_mismatched_committed_path(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="active path plus"):
        retire_committed_controller_snapshot(
            tmp_path,
            relative_path="runs/custom_controller_snapshot_002",
            committed_relative_path="runs/unrelated.committed",
        )
