"""Crash-recovery tests for the asset-free M8 attempt transaction."""

from __future__ import annotations

import hashlib
import json
import os
import stat
from pathlib import Path

import pytest

from controller_learning.evaluation.attempt_transaction import (
    FORMAL_CONTROLLER_ORDER,
    FORMAL_EPISODE_COUNT,
    FORMAL_EXECUTION_EVIDENCE_BLOB_PATH,
    M8_EXECUTION_EVIDENCE_SCHEMA_VERSION,
    AttemptIdentity,
    AttemptPhase,
    AttemptTransactionError,
    AttemptTransactionTamperError,
    EpisodeJournalRecord,
    IncompleteTestAttemptError,
    M8AttemptTransaction,
    canonical_execution_evidence_bytes,
)
from controller_learning.evaluation.final_benchmark import (
    M8_CONTROLLER_EXECUTION_MODEL,
    M8_ENVIRONMENT_LIFECYCLE,
)
from controller_learning.evaluation.final_report import (
    M8_TEST_ACCESS_AUDIT_SCHEMA_VERSION,
)
from controller_learning.evaluation.final_runtime import (
    FINAL_ENVIRONMENT_LIFECYCLE_SCHEMA_VERSION,
    FINAL_MEMORY_EVIDENCE_SCHEMA_VERSION,
    FINAL_RUNTIME_EVIDENCE_SCHEMA_VERSION,
    FINAL_RUNTIME_PACKAGE_NAMES,
)
from controller_learning.evaluation.test_assets import M8_TEST_POOL_ACCESS_SCHEMA_VERSION

IDENTITY = AttemptIdentity(
    source_revision="1" * 40,
    source_tree_sha256="5" * 64,
    config_sha256="2" * 64,
    pixi_lock_sha256="3" * 64,
    input_sha256="4" * 64,
)
OUTPUTS = ("artifacts/a.bin", "artifacts/nested/b.json")
TRANSACTION_PATH = "runs/m8-test-transaction"


def _transaction(root: Path, *, identity: AttemptIdentity = IDENTITY) -> M8AttemptTransaction:
    return M8AttemptTransaction(
        root,
        transaction_relative_path=TRANSACTION_PATH,
        output_allowlist=OUTPUTS,
        identity=identity,
    )


def _record(controller: str, row: int, *, outcome: str = "success") -> EpisodeJournalRecord:
    trajectory = _trajectory_payload(controller, row)
    return EpisodeJournalRecord(
        controller=controller,
        row_index=row,
        track_id=2_000_000 + row,
        reset_seed=row,
        episode_seed=10_000 + row,
        controller_seed=20_000 + row,
        outcome=outcome,
        steps=100 + row,
        trajectory_blob_path=f"episodes/{controller}/row_{row:03d}_trajectory.json",
        trajectory_blob_sha256=hashlib.sha256(trajectory).hexdigest(),
        trajectory_blob_size_bytes=len(trajectory),
        data={
            "compute_times_s": [0.001] * (100 + row),
            "controller_import_time_s": 0.01,
            "controller_init_time_s": 0.02,
        },
    )


def _trajectory_payload(controller: str, row: int) -> bytes:
    return f'{{"controller":"{controller}","row":{row}}}\n'.encode()


def _append(transaction: M8AttemptTransaction, record: EpisodeJournalRecord) -> None:
    transaction.append_episode_bundle(
        record,
        _trajectory_payload(record.controller, record.row_index),
    )


def _append_all(
    transaction: M8AttemptTransaction,
    *,
    outcome: str = "success",
) -> None:
    for controller in FORMAL_CONTROLLER_ORDER:
        for row in range(20):
            _append(transaction, _record(controller, row, outcome=outcome))


def _formal_outputs(prefix: bytes = b"formal") -> dict[str, bytes]:
    return {path: prefix + b":" + path.encode() for path in OUTPUTS}


def _execution_evidence_payload() -> bytes:
    track_ids = tuple(2_000_000 + row for row in range(20))
    track_id_lines = b"".join(f"{track_id}\n".encode("ascii") for track_id in track_ids)
    return canonical_execution_evidence_bytes(
        {
            "asset_access": {
                "all_track_reads_forbidden": True,
                "audit_hook_installed_before_preflight": True,
                "denied_event_count": 0,
                "denied_mutation_event_count": 0,
                "denied_mutation_event_types": {},
                "open_event_counts": {
                    "official_test_asset": 1,
                    "official_test_manifest": 1,
                },
                "open_event_sequence": [
                    {"category": "official_test_manifest", "flags": 0, "mode": "r"},
                    {"category": "official_test_asset", "flags": 0, "mode": "r"},
                ],
                "opened_path_categories": [
                    "official_test_asset",
                    "official_test_manifest",
                ],
                "opened_splits": ["test"],
                "pre_test_open_event_count": 0,
                "schema_version": M8_TEST_ACCESS_AUDIT_SCHEMA_VERSION,
                "test_loaded": True,
                "test_reads_enabled": True,
                "track_cache_opened": False,
                "train_opened": False,
                "validation_opened": False,
            },
            "execution": {
                "automatic_retry_after_test_bound": False,
                "controller_execution_model": M8_CONTROLLER_EXECUTION_MODEL,
                "controller_init_soft_limit_s": 30.0,
                "controller_order": ["pid", "mpc", "ppo"],
                "controller_wall_time_s": {"pid": 3.0, "mpc": 3.0, "ppo": 3.0},
                "environment_instance_count": 1,
                "environment_lifecycle": M8_ENVIRONMENT_LIFECYCLE,
                "environment_steps_by_controller": {"pid": 100, "mpc": 100, "ppo": 100},
                "environment_steps_per_second": 30.0,
                "episode_count": 60,
                "fresh_controller_instance_count": 60,
                "fresh_controller_per_episode": True,
                "initialization_over_soft_limit_rows": {"pid": [], "mpc": [], "ppo": []},
                "measured_environment_lifecycle": {
                    "close_count": 1,
                    "environment_create_wall_time_s": 0.1,
                    "environment_instance_count": 1,
                    "expected_reset_count": 60,
                    "expected_step_count": 300,
                    "first_reset_wall_time_including_lazy_compilation_s": 0.2,
                    "first_step_wall_time_including_lazy_compilation_s": 0.3,
                    "method": "wall clock including lazy compilation",
                    "reset_count": 60,
                    "schema_version": FINAL_ENVIRONMENT_LIFECYCLE_SCHEMA_VERSION,
                    "step_count": 300,
                },
                "numerical_failure_count": 0,
                "replay_captured_from_same_rollout": True,
                "replay_environment_instance_count": 0,
                "replay_row_index": 0,
                "retry_count": 0,
                "row_order": list(range(20)),
                "schema_version": M8_EXECUTION_EVIDENCE_SCHEMA_VERSION,
                "total_environment_steps": 300,
                "wall_time_s": 10.0,
            },
            "memory": {
                "final_jax_live_bytes": 0,
                "peak_jax_allocator_bytes": 1024,
                "peak_sampled_process_vram_mib": 100.0,
                "sample_count": 1,
                "samples": [
                    {
                        "jax_bytes_in_use": 0,
                        "jax_peak_bytes_in_use": 1024,
                        "label": "after_environment_close",
                        "process_vram_mib": 100.0,
                        "synchronized": True,
                    }
                ],
                "sampling_method": "JAX synchronized process memory sampling",
                "schema_version": FINAL_MEMORY_EVIDENCE_SCHEMA_VERSION,
            },
            "runtime": {
                "cpu_model": "Synthetic CPU",
                "cuda_device_order": "PCI_BUS_ID",
                "cuda_driver": "570.00",
                "cuda_runtime": "CUDA 12.8",
                "cuda_visible_devices_configured": True,
                "jax_device": {
                    "device_kind": "NVIDIA Synthetic GPU",
                    "id": 0,
                    "platform": "gpu",
                },
                "kernel": "6.8.0-synthetic",
                "machine": "x86_64",
                "packages": {name: "1.0.0" for name in FINAL_RUNTIME_PACKAGE_NAMES},
                "platform": "Linux",
                "python": "3.11.9",
                "schema_version": FINAL_RUNTIME_EVIDENCE_SCHEMA_VERSION,
                "selected_gpu": {
                    "driver_version": "570.00",
                    "index": 0,
                    "memory_total_mib": 24000.0,
                    "name": "NVIDIA Synthetic GPU",
                    "uuid": "redacted",
                },
                "xla_python_client_preallocate": "false",
            },
            "schema_version": M8_EXECUTION_EVIDENCE_SCHEMA_VERSION,
            "test_assets": {
                "asset_file": "test.npz",
                "asset_file_sha256": "a" * 64,
                "benchmark_version": "0.1",
                "capacity": {"max_checkpoints": 48, "max_track_points": 640},
                "generator_version": "synthetic-v1",
                "geometry_hashes_sha256": "b" * 64,
                "level_id": 1,
                "loaded_splits": ["test"],
                "loader_accessed_train": False,
                "loader_accessed_validation": False,
                "manifest_asset_sha256": "a" * 64,
                "manifest_file": "test.json",
                "manifest_sha256": "c" * 64,
                "schema_version": M8_TEST_POOL_ACCESS_SCHEMA_VERSION,
                "split": "test",
                "track_count": 20,
                "track_ids": list(track_ids),
                "track_ids_sha256": hashlib.sha256(track_id_lines).hexdigest(),
            },
        }
    )


def _seal(transaction: M8AttemptTransaction) -> None:
    record = transaction.write_execution_evidence(_execution_evidence_payload())
    assert record.relative_path == FORMAL_EXECUTION_EVIDENCE_BLOB_PATH
    assert transaction.read_execution_evidence()["schema_version"] == (
        M8_EXECUTION_EVIDENCE_SCHEMA_VERSION
    )


def test_episode_record_requires_row_reset_seed_and_domain_separation() -> None:
    with pytest.raises(ValueError, match="row index"):
        EpisodeJournalRecord(
            controller="pid",
            row_index=2,
            track_id=2_000_002,
            reset_seed=3,
            episode_seed=10,
            controller_seed=11,
            outcome="success",
            steps=1,
            trajectory_blob_path="episodes/pid/row_002_trajectory.json",
            trajectory_blob_sha256="0" * 64,
            trajectory_blob_size_bytes=1,
        )
    with pytest.raises(ValueError, match="domain-separated"):
        EpisodeJournalRecord(
            controller="pid",
            row_index=2,
            track_id=2_000_002,
            reset_seed=2,
            episode_seed=10,
            controller_seed=10,
            outcome="success",
            steps=1,
            trajectory_blob_path="episodes/pid/row_002_trajectory.json",
            trajectory_blob_sha256="0" * 64,
            trajectory_blob_size_bytes=1,
        )


def test_journal_rejects_cross_controller_track_or_seed_drift(tmp_path: Path) -> None:
    transaction = _transaction(tmp_path)
    transaction.prepare()
    transaction.bind_test()
    for row in range(20):
        _append(transaction, _record("pid", row))

    drifted = EpisodeJournalRecord(
        controller="mpc",
        row_index=0,
        track_id=2_000_001,
        reset_seed=0,
        episode_seed=10_000,
        controller_seed=20_000,
        outcome="success",
        steps=100,
        trajectory_blob_path="episodes/mpc/row_000_trajectory.json",
        trajectory_blob_sha256=hashlib.sha256(_trajectory_payload("mpc", 0)).hexdigest(),
        trajectory_blob_size_bytes=len(_trajectory_payload("mpc", 0)),
        data={
            "compute_times_s": [0.001] * 100,
            "controller_import_time_s": 0.01,
            "controller_init_time_s": 0.02,
        },
    )
    with pytest.raises(AttemptTransactionError, match="reuse the PID row"):
        _append(transaction, drifted)


def _complete_and_validate(
    transaction: M8AttemptTransaction,
    *,
    outputs: dict[str, bytes] | None = None,
) -> dict[str, bytes]:
    values = _formal_outputs() if outputs is None else outputs
    _seal(transaction)
    transaction.complete_evaluation(values)
    transaction.mark_artifacts_validated(semantic_validation_sha256="a" * 64)
    return values


def _advance_to_phase(
    transaction: M8AttemptTransaction,
    phase: AttemptPhase,
) -> None:
    transaction.prepare()
    if phase is AttemptPhase.PREPARED:
        return
    transaction.bind_test()
    if phase is AttemptPhase.TEST_BOUND:
        return
    _append_all(transaction)
    _seal(transaction)
    transaction.complete_evaluation(_formal_outputs())
    if phase is AttemptPhase.EVALUATION_COMPLETE:
        return
    transaction.mark_artifacts_validated(semantic_validation_sha256="a" * 64)
    if phase is AttemptPhase.ARTIFACTS_VALIDATED:
        return
    transaction.publish_and_commit()


def test_constructor_rejects_unsafe_or_noncanonical_paths(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="child of ignored runs"):
        M8AttemptTransaction(
            tmp_path,
            transaction_relative_path="transactions/m8",
            output_allowlist=OUTPUTS,
            identity=IDENTITY,
        )
    with pytest.raises(ValueError, match="normalized relative"):
        M8AttemptTransaction(
            tmp_path,
            transaction_relative_path="runs/m8",
            output_allowlist=("../escape",),
            identity=IDENTITY,
        )
    with pytest.raises(ValueError, match="sorted"):
        M8AttemptTransaction(
            tmp_path,
            transaction_relative_path="runs/m8",
            output_allowlist=tuple(reversed(OUTPUTS)),
            identity=IDENTITY,
        )


def test_prepare_is_canonical_and_pre_test_recovery_removes_absent_outputs(
    tmp_path: Path,
) -> None:
    transaction = _transaction(tmp_path)

    inspection = transaction.prepare()

    assert inspection.phase is AttemptPhase.PREPARED
    assert inspection.journal_record_count == 0
    assert inspection.output_state == "original"
    manifest = transaction.transaction_directory / "manifest.json"
    payload = manifest.read_bytes()
    assert payload.endswith(b"\n")
    assert (
        json.dumps(
            json.loads(payload),
            allow_nan=False,
            ensure_ascii=False,
            separators=(",", ":"),
            sort_keys=True,
        ).encode()
        + b"\n"
        == payload
    )

    recovery = _transaction(tmp_path).recover()

    assert recovery.action == "pre_test_restored"
    assert not transaction.transaction_directory.exists()
    assert all(not (tmp_path / path).exists() for path in OUTPUTS)
    assert not (tmp_path / "artifacts").exists()


def test_pre_test_recovery_preserves_existing_bytes_and_modes(tmp_path: Path) -> None:
    original = {
        OUTPUTS[0]: (b"old-a", 0o640),
        OUTPUTS[1]: (b"old-b", 0o600),
    }
    for path, (payload, mode) in original.items():
        destination = tmp_path / path
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_bytes(payload)
        destination.chmod(mode)
    transaction = _transaction(tmp_path)
    transaction.prepare()

    assert _transaction(tmp_path).recover().action == "pre_test_restored"

    for path, (payload, mode) in original.items():
        destination = tmp_path / path
        assert destination.read_bytes() == payload
        assert stat.S_IMODE(destination.stat().st_mode) == mode


def test_prepared_recovery_accepts_restore_scratch_without_rewriting_outputs(
    tmp_path: Path,
) -> None:
    destination = tmp_path / OUTPUTS[0]
    destination.parent.mkdir(parents=True)
    destination.write_bytes(b"original")
    destination.chmod(0o640)
    transaction = _transaction(tmp_path)
    transaction.prepare()
    original_identity = destination.stat()
    publication = transaction.transaction_directory / "publication"
    publication.mkdir()
    (publication / "001.restore").write_bytes(b"safe crash residue")

    recovery = _transaction(tmp_path).recover()

    assert recovery.action == "pre_test_restored"
    assert destination.read_bytes() == b"original"
    recovered_identity = destination.stat()
    assert recovered_identity.st_ino == original_identity.st_ino
    assert recovered_identity.st_mtime_ns == original_identity.st_mtime_ns
    assert stat.S_IMODE(recovered_identity.st_mode) == 0o640
    assert not (tmp_path / OUTPUTS[1]).exists()
    assert not (tmp_path / "artifacts/nested").exists()
    assert not transaction.transaction_directory.exists()


@pytest.mark.parametrize("residue_kind", ("publish", "out_of_range", "symlink", "directory"))
def test_prepared_recovery_rejects_unsafe_publication_residue(
    tmp_path: Path,
    residue_kind: str,
) -> None:
    transaction = _transaction(tmp_path)
    transaction.prepare()
    publication = transaction.transaction_directory / "publication"
    publication.mkdir()
    if residue_kind == "publish":
        (publication / "000.publish").write_bytes(b"residue")
    elif residue_kind == "out_of_range":
        (publication / "999.restore").write_bytes(b"residue")
    elif residue_kind == "symlink":
        (publication / "000.restore").symlink_to(tmp_path / "outside")
    else:
        (publication / "000.restore").mkdir()

    with pytest.raises(AttemptTransactionTamperError):
        transaction.recover()


def test_prepared_recovery_resumes_after_created_directory_cleanup_crash(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from controller_learning.evaluation import attempt_transaction as module

    transaction = _transaction(tmp_path)
    transaction.prepare()
    real_fsync_directory = module._fsync_directory
    failed = False

    def fail_after_nested_removal(path: Path) -> None:
        nonlocal failed
        if path == tmp_path / "artifacts" and not failed:
            failed = True
            raise OSError("simulated cleanup crash")
        real_fsync_directory(path)

    monkeypatch.setattr(module, "_fsync_directory", fail_after_nested_removal)
    with pytest.raises(OSError, match="cleanup crash"):
        transaction.recover()
    assert not (tmp_path / "artifacts/nested").exists()
    assert transaction.inspect().phase is AttemptPhase.PREPARED

    monkeypatch.setattr(module, "_fsync_directory", real_fsync_directory)
    assert _transaction(tmp_path).recover().action == "pre_test_restored"
    assert not transaction.transaction_directory.exists()


def test_prepared_recovery_resumes_after_atomic_retirement_cleanup_crash(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    transaction = _transaction(tmp_path)
    transaction.prepare()

    def fail_retired_tree_cleanup(path: Path) -> None:
        assert path == transaction.cleanup_directory
        raise OSError("simulated retired-tree cleanup crash")

    monkeypatch.setattr(transaction, "_remove_tree", fail_retired_tree_cleanup)
    with pytest.raises(OSError, match="retired-tree"):
        transaction.recover()
    assert not transaction.transaction_directory.exists()
    assert transaction.cleanup_directory.exists()

    assert _transaction(tmp_path).recover().action == "none"
    assert not transaction.cleanup_directory.exists()


def test_prepare_crash_before_atomic_rename_leaves_no_active_attempt(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from controller_learning.evaluation import attempt_transaction as module

    transaction = _transaction(tmp_path)
    real_replace = module.os.replace
    staged: list[Path] = []

    def fail_activation(source: Path | str, destination: Path | str) -> None:
        if Path(destination) == transaction.transaction_directory:
            staged.append(Path(source))
            raise OSError("simulated pre-rename crash")
        real_replace(source, destination)

    monkeypatch.setattr(module.os, "replace", fail_activation)
    with pytest.raises(OSError, match="pre-rename"):
        transaction.prepare()
    assert not transaction.transaction_directory.exists()
    assert len(staged) == 1
    assert staged[0].name.startswith(transaction.transaction_directory.name + ".prepare.")
    (staged[0] / "untrusted-link").symlink_to(tmp_path / "outside")

    monkeypatch.setattr(module.os, "replace", real_replace)
    assert _transaction(tmp_path).prepare().phase is AttemptPhase.PREPARED
    assert (staged[0] / "untrusted-link").is_symlink()


@pytest.mark.parametrize(
    "phase",
    (
        AttemptPhase.PREPARED,
        AttemptPhase.TEST_BOUND,
        AttemptPhase.EVALUATION_COMPLETE,
        AttemptPhase.COMMITTED,
    ),
)
def test_load_removes_only_exact_regular_atomic_state_residue(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    phase: AttemptPhase,
) -> None:
    from controller_learning.evaluation import attempt_transaction as module

    transaction = _transaction(tmp_path)
    _advance_to_phase(transaction, phase)
    residue = transaction.transaction_directory / ".state.json.sigkill_01.tmp"
    residue.write_bytes(b"incomplete replacement")
    real_fsync_directory = module._fsync_directory
    fsynced: list[Path] = []

    def track_fsync(path: Path) -> None:
        fsynced.append(path)
        real_fsync_directory(path)

    monkeypatch.setattr(module, "_fsync_directory", track_fsync)

    assert transaction.inspect().phase is phase
    assert not residue.exists()
    assert transaction.transaction_directory in fsynced


@pytest.mark.parametrize("residue_kind", ("other_dot", "symlink", "directory", "fifo"))
def test_load_rejects_unsafe_atomic_state_residue(
    tmp_path: Path,
    residue_kind: str,
) -> None:
    transaction = _transaction(tmp_path)
    transaction.prepare()
    name = ".unexpected" if residue_kind == "other_dot" else ".state.json.crash.tmp"
    residue = transaction.transaction_directory / name
    if residue_kind == "symlink":
        residue.symlink_to(tmp_path / "outside")
    elif residue_kind == "directory":
        residue.mkdir()
    elif residue_kind == "fifo":
        os.mkfifo(residue)
    else:
        residue.write_bytes(b"unsafe")

    with pytest.raises(AttemptTransactionTamperError):
        transaction.inspect()


def test_prepare_rejects_symlink_output_parent_and_destination(tmp_path: Path) -> None:
    outside = tmp_path / "outside"
    outside.mkdir()
    (tmp_path / "artifacts").symlink_to(outside, target_is_directory=True)

    with pytest.raises(AttemptTransactionTamperError, match="non-symlink"):
        _transaction(tmp_path).prepare()

    (tmp_path / "artifacts").unlink()
    (tmp_path / "artifacts/nested").mkdir(parents=True)
    (tmp_path / OUTPUTS[0]).symlink_to(outside / "target")
    with pytest.raises(AttemptTransactionTamperError, match="regular"):
        _transaction(tmp_path).prepare()


def test_incomplete_test_attempt_refuses_recovery_and_new_process_append(
    tmp_path: Path,
) -> None:
    transaction = _transaction(tmp_path)
    transaction.prepare()
    transaction.bind_test()
    _append(transaction, _record("pid", 0))
    journal = transaction.transaction_directory / "episode-journal.jsonl"
    durable_bytes = journal.read_bytes()

    resumed = _transaction(tmp_path)
    with pytest.raises(IncompleteTestAttemptError) as captured:
        resumed.recover()
    assert captured.value.inspection.journal_record_count == 1
    assert captured.value.inspection.next_episode == ("pid", 1)
    assert journal.read_bytes() == durable_bytes

    with pytest.raises(IncompleteTestAttemptError):
        _append(resumed, _record("pid", 1))
    assert journal.read_bytes() == durable_bytes


def test_uncommitted_final_episode_blob_cannot_make_attempt_complete(tmp_path: Path) -> None:
    transaction = _transaction(tmp_path)
    transaction.prepare()
    transaction.bind_test()
    for controller in FORMAL_CONTROLLER_ORDER:
        row_limit = 19 if controller == "ppo" else 20
        for row in range(row_limit):
            _append(transaction, _record(controller, row))
    pending = _record("ppo", 19)
    transaction.write_blob(
        pending.trajectory_blob_path,
        _trajectory_payload("ppo", 19),
    )

    with pytest.raises(IncompleteTestAttemptError) as captured:
        _transaction(tmp_path).recover()

    assert captured.value.inspection.journal_record_count == 59
    assert captured.value.inspection.next_episode == ("ppo", 19)


def test_journal_rejects_duplicate_and_out_of_order_records(tmp_path: Path) -> None:
    transaction = _transaction(tmp_path)
    transaction.prepare()
    transaction.bind_test()

    with pytest.raises(AttemptTransactionError, match="out of order"):
        _append(transaction, _record("mpc", 0))
    _append(transaction, _record("pid", 0))
    with pytest.raises(AttemptTransactionError, match="out of order"):
        _append(transaction, _record("pid", 0))
    with pytest.raises(AttemptTransactionError, match="out of order"):
        _append(transaction, _record("pid", 2))
    assert transaction.inspect().journal_record_count == 1


def test_binary_blobs_are_fsynced_immutable_and_tamper_evident(tmp_path: Path) -> None:
    transaction = _transaction(tmp_path)
    transaction.prepare()
    transaction.bind_test()

    record = transaction.write_blob("pid/row_000_metrics.bin", b"\x00metric\xff")

    assert record.size_bytes == 8
    assert transaction.inspect().blob_records == (record,)
    assert transaction.read_blob("pid/row_000_metrics.bin") == b"\x00metric\xff"
    verified_path = transaction.verified_blob_path("pid/row_000_metrics.bin")
    assert verified_path.read_bytes() == b"\x00metric\xff"
    assert verified_path.is_relative_to(transaction.transaction_directory)
    with pytest.raises(AttemptTransactionError, match="not present exactly once"):
        transaction.read_blob("pid/missing.bin")
    with pytest.raises(AttemptTransactionError, match="immutable"):
        transaction.write_blob("pid/row_000_metrics.bin", b"different")
    blob = transaction.transaction_directory / "blobs/pid/row_000_metrics.bin"
    blob.write_bytes(b"tampered")
    with pytest.raises(AttemptTransactionTamperError, match="blob bytes changed"):
        transaction.inspect()


def test_evaluation_requires_exactly_60_records_and_exact_output_allowlist(
    tmp_path: Path,
) -> None:
    transaction = _transaction(tmp_path)
    transaction.prepare()
    transaction.bind_test()
    _append(transaction, _record("pid", 0))
    with pytest.raises(AttemptTransactionError, match="exactly 60"):
        transaction.complete_evaluation(_formal_outputs())

    # Start a fresh fixture because a Test-bound attempt is intentionally non-abortable.
    other_root = tmp_path / "other"
    other_root.mkdir()
    complete = _transaction(other_root)
    complete.prepare()
    complete.bind_test()
    _append_all(complete)
    _seal(complete)
    with pytest.raises(AttemptTransactionError, match="exactly match"):
        complete.complete_evaluation({OUTPUTS[0]: b"missing second output"})
    assert complete.inspect().phase is AttemptPhase.TEST_BOUND
    assert complete.inspect().journal_record_count == FORMAL_EPISODE_COUNT


def test_completed_attempt_resumes_without_workload_and_publishes_absent_outputs(
    tmp_path: Path,
) -> None:
    transaction = _transaction(tmp_path)
    transaction.prepare()
    transaction.bind_test()
    transaction.write_blob("shared/finalization-input.bin", b"durable-finalization-input")
    _append_all(transaction)
    _seal(transaction)
    values = _formal_outputs()
    completed = transaction.complete_evaluation(values)
    assert completed.phase is AttemptPhase.EVALUATION_COMPLETE
    assert dict(transaction.read_staged_outputs()) == values
    assert len(transaction.episode_records()) == FORMAL_EPISODE_COUNT
    journal_before = (transaction.transaction_directory / "episode-journal.jsonl").read_bytes()

    resumed = _transaction(tmp_path)
    recovery = resumed.recover()
    assert recovery.action == "evaluation_complete_ready"
    assert recovery.journal_record_count == 60
    assert resumed.read_blob("shared/finalization-input.bin") == b"durable-finalization-input"
    resumed.mark_artifacts_validated(semantic_validation_sha256="a" * 64)
    assert _transaction(tmp_path).recover().action == "artifacts_validated_ready"
    published = _transaction(tmp_path).publish_and_commit()

    assert len(published) == len(OUTPUTS)
    assert _transaction(tmp_path).inspect().phase is AttemptPhase.COMMITTED
    _transaction(tmp_path).retire_committed()
    assert not transaction.transaction_directory.exists()
    assert all((tmp_path / path).read_bytes() == values[path] for path in OUTPUTS)
    assert journal_before.count(b"\n") == FORMAL_EPISODE_COUNT


def test_complete_journal_without_post_close_evidence_is_not_recoverable(
    tmp_path: Path,
) -> None:
    transaction = _transaction(tmp_path)
    transaction.prepare()
    transaction.bind_test()
    _append_all(transaction)

    resumed = _transaction(tmp_path)
    with pytest.raises(IncompleteTestAttemptError, match="execution evidence sealed=false"):
        resumed.recover()
    with pytest.raises(AttemptTransactionError, match="post-close execution evidence"):
        transaction.complete_evaluation(_formal_outputs())
    with pytest.raises(IncompleteTestAttemptError):
        resumed.write_execution_evidence(_execution_evidence_payload())

    assert transaction.inspect().phase is AttemptPhase.TEST_BOUND
    assert transaction.inspect().journal_record_count == FORMAL_EPISODE_COUNT


def test_different_process_id_cannot_inherit_test_bound_write_authority(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from controller_learning.evaluation import attempt_transaction as module

    transaction = _transaction(tmp_path)
    transaction.prepare()
    transaction.bind_test()
    _append_all(transaction)
    binder_pid = module.os.getpid()
    original_getpid = module.os.getpid
    monkeypatch.setattr(module.os, "getpid", lambda: binder_pid + 1)
    with pytest.raises(IncompleteTestAttemptError):
        transaction.write_execution_evidence(_execution_evidence_payload())
    assert all(
        record.relative_path != FORMAL_EXECUTION_EVIDENCE_BLOB_PATH
        for record in transaction.inspect().blob_records
    )
    monkeypatch.setattr(module.os, "getpid", original_getpid)
    _seal(transaction)
    assert _transaction(tmp_path).recover().action == "evaluation_complete_ready"


def test_execution_evidence_rejects_empty_or_false_post_close_claims() -> None:
    valid = json.loads(_execution_evidence_payload())
    valid["execution"] = {}
    with pytest.raises(ValueError):
        canonical_execution_evidence_bytes(valid)

    invalid_lifecycle = json.loads(_execution_evidence_payload())
    invalid_lifecycle["execution"]["measured_environment_lifecycle"]["close_count"] = 0
    with pytest.raises(ValueError, match="close_count"):
        canonical_execution_evidence_bytes(invalid_lifecycle)


def test_test_bound_complete_journal_can_seal_surviving_staged_bytes(
    tmp_path: Path,
) -> None:
    transaction = _transaction(tmp_path)
    transaction.prepare()
    transaction.bind_test()
    _append_all(transaction)
    _seal(transaction)
    values = _formal_outputs(b"survived")
    normalized = transaction._normalize_outputs(values, modes=None)
    transaction._build_staged_outputs(normalized)

    resumed = _transaction(tmp_path)
    assert resumed.recover().action == "evaluation_complete_ready"
    inspection = resumed.complete_evaluation()

    assert inspection.phase is AttemptPhase.EVALUATION_COMPLETE
    assert inspection.journal_record_count == FORMAL_EPISODE_COUNT


def test_journal_reload_rejects_cross_controller_identity_drift(tmp_path: Path) -> None:
    transaction = _transaction(tmp_path)
    transaction.prepare()
    transaction.bind_test()
    for row in range(20):
        _append(transaction, _record("pid", row))
    _append(transaction, _record("mpc", 0))
    journal = transaction.transaction_directory / "episode-journal.jsonl"
    lines = journal.read_bytes().splitlines(keepends=True)
    changed = json.loads(lines[20])
    changed["track_id"] += 1
    lines[20] = (
        json.dumps(
            changed,
            allow_nan=False,
            ensure_ascii=False,
            separators=(",", ":"),
            sort_keys=True,
        ).encode("utf-8")
        + b"\n"
    )
    journal.write_bytes(b"".join(lines))

    with pytest.raises(AttemptTransactionTamperError, match="Track or seed sequence"):
        _transaction(tmp_path).inspect()


def test_execution_evidence_blob_tamper_is_detected(tmp_path: Path) -> None:
    transaction = _transaction(tmp_path)
    transaction.prepare()
    transaction.bind_test()
    _append_all(transaction)
    _seal(transaction)
    seal_path = transaction.transaction_directory / "blobs" / FORMAL_EXECUTION_EVIDENCE_BLOB_PATH
    seal_path.write_bytes(seal_path.read_bytes() + b" ")

    with pytest.raises(AttemptTransactionTamperError, match="blob bytes changed"):
        transaction.inspect()


def test_partial_publication_restores_absent_outputs_then_republishes_identically(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from controller_learning.evaluation import attempt_transaction as module

    transaction = _transaction(tmp_path)
    transaction.prepare()
    transaction.bind_test()
    _append_all(transaction)
    values = _complete_and_validate(transaction)
    real_replace = module.os.replace
    failed = False

    def fail_second_publish(source: Path | str, destination: Path | str) -> None:
        nonlocal failed
        destination_path = Path(destination)
        if destination_path == tmp_path / OUTPUTS[1] and not failed:
            failed = True
            raise OSError("simulated publication loss")
        real_replace(source, destination)

    monkeypatch.setattr(module.os, "replace", fail_second_publish)
    with pytest.raises(OSError, match="simulated"):
        transaction.publish_and_commit(retain_committed_transaction=False)
    monkeypatch.setattr(module.os, "replace", real_replace)
    assert transaction.inspect().output_state == "partial_publication"

    recovery = _transaction(tmp_path).recover()
    assert recovery.action == "partial_publication_restored"
    assert all(not (tmp_path / path).exists() for path in OUTPUTS)
    assert _transaction(tmp_path).inspect().output_state == "original"

    _transaction(tmp_path).publish_and_commit()
    assert all((tmp_path / path).read_bytes() == values[path] for path in OUTPUTS)


def test_partial_publication_restores_existing_bytes_modes_then_republishes(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from controller_learning.evaluation import attempt_transaction as module

    originals = {OUTPUTS[0]: (b"old-a", 0o640), OUTPUTS[1]: (b"old-b", 0o600)}
    for path, (payload, mode) in originals.items():
        destination = tmp_path / path
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_bytes(payload)
        destination.chmod(mode)
    transaction = _transaction(tmp_path)
    transaction.prepare()
    transaction.bind_test()
    _append_all(transaction)
    values = _complete_and_validate(transaction)
    real_replace = module.os.replace
    failed = False

    def fail_second(source: Path | str, destination: Path | str) -> None:
        nonlocal failed
        if Path(destination) == tmp_path / OUTPUTS[1] and not failed:
            failed = True
            raise OSError("stop after one replacement")
        real_replace(source, destination)

    monkeypatch.setattr(module.os, "replace", fail_second)
    with pytest.raises(OSError):
        transaction.publish_and_commit()
    monkeypatch.setattr(module.os, "replace", real_replace)

    assert _transaction(tmp_path).recover().action == "partial_publication_restored"
    for path, (payload, mode) in originals.items():
        assert (tmp_path / path).read_bytes() == payload
        assert stat.S_IMODE((tmp_path / path).stat().st_mode) == mode
    _transaction(tmp_path).publish_and_commit()
    assert all((tmp_path / path).read_bytes() == values[path] for path in OUTPUTS)


def test_publish_revalidates_original_immediately_before_replace(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from controller_learning.evaluation import attempt_transaction as module

    transaction = _transaction(tmp_path)
    transaction.prepare()
    transaction.bind_test()
    _append_all(transaction)
    _complete_and_validate(transaction)
    real_write = module._write_fsynced_file
    changed = False

    def change_destination_after_scratch(
        path: Path,
        payload: bytes,
        *,
        mode: int,
    ) -> None:
        nonlocal changed
        real_write(path, payload, mode=mode)
        if path.name == "000.publish" and not changed:
            changed = True
            (tmp_path / OUTPUTS[0]).write_bytes(b"concurrent publication change")

    monkeypatch.setattr(module, "_write_fsynced_file", change_destination_after_scratch)

    with pytest.raises(AttemptTransactionTamperError, match="immediately before mutation"):
        transaction.publish_and_commit()
    assert (tmp_path / OUTPUTS[0]).read_bytes() == b"concurrent publication change"


def test_restore_revalidates_staged_output_immediately_before_replace(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from controller_learning.evaluation import attempt_transaction as module

    originals = {OUTPUTS[0]: b"old-a", OUTPUTS[1]: b"old-b"}
    for relative, payload in originals.items():
        destination = tmp_path / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_bytes(payload)
    transaction = _transaction(tmp_path)
    transaction.prepare()
    transaction.bind_test()
    _append_all(transaction)
    _complete_and_validate(transaction)
    real_replace = module.os.replace

    def stop_after_first_publish(source: Path | str, destination: Path | str) -> None:
        if Path(destination) == tmp_path / OUTPUTS[1]:
            raise OSError("stop publication")
        real_replace(source, destination)

    monkeypatch.setattr(module.os, "replace", stop_after_first_publish)
    with pytest.raises(OSError, match="stop publication"):
        transaction.publish_and_commit()
    monkeypatch.setattr(module.os, "replace", real_replace)
    real_write = module._write_fsynced_file

    def change_destination_after_restore_scratch(
        path: Path,
        payload: bytes,
        *,
        mode: int,
    ) -> None:
        real_write(path, payload, mode=mode)
        if path.name == "000.restore":
            (tmp_path / OUTPUTS[0]).write_bytes(b"concurrent restore change")

    monkeypatch.setattr(module, "_write_fsynced_file", change_destination_after_restore_scratch)

    with pytest.raises(AttemptTransactionTamperError, match="immediately before mutation"):
        _transaction(tmp_path).recover()
    assert (tmp_path / OUTPUTS[0]).read_bytes() == b"concurrent restore change"


def test_restore_revalidates_staged_output_immediately_before_unlink(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from controller_learning.evaluation import attempt_transaction as module

    transaction = _transaction(tmp_path)
    transaction.prepare()
    transaction.bind_test()
    _append_all(transaction)
    _complete_and_validate(transaction)
    real_replace = module.os.replace

    def stop_after_first_publish(source: Path | str, destination: Path | str) -> None:
        if Path(destination) == tmp_path / OUTPUTS[1]:
            raise OSError("stop publication")
        real_replace(source, destination)

    monkeypatch.setattr(module.os, "replace", stop_after_first_publish)
    with pytest.raises(OSError, match="stop publication"):
        transaction.publish_and_commit()
    monkeypatch.setattr(module.os, "replace", real_replace)
    resumed = _transaction(tmp_path)
    real_require = resumed._require_output_identity
    first_output_checks = 0

    def change_before_second_identity_check(*args, **kwargs):
        nonlocal first_output_checks
        snapshot = args[0]
        if snapshot.relative_path == OUTPUTS[0]:
            first_output_checks += 1
            if first_output_checks == 2:
                (tmp_path / OUTPUTS[0]).write_bytes(b"concurrent unlink change")
        return real_require(*args, **kwargs)

    monkeypatch.setattr(resumed, "_require_output_identity", change_before_second_identity_check)

    with pytest.raises(AttemptTransactionTamperError, match="immediately before mutation"):
        resumed.recover()
    assert (tmp_path / OUTPUTS[0]).read_bytes() == b"concurrent unlink change"


def test_committed_state_survives_cleanup_crash_and_recovers_without_republication(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    transaction = _transaction(tmp_path)
    transaction.prepare()
    transaction.bind_test()
    _append_all(transaction)
    values = _complete_and_validate(transaction)

    def fail_cleanup() -> None:
        raise OSError("simulated cleanup loss")

    monkeypatch.setattr(transaction, "_retire_transaction", fail_cleanup)
    with pytest.raises(OSError, match="cleanup loss"):
        transaction.publish_and_commit(retain_committed_transaction=False)
    inspection = _transaction(tmp_path).inspect()
    assert inspection.phase is AttemptPhase.COMMITTED
    assert inspection.output_state == "published"

    recovery = _transaction(tmp_path).recover()
    assert recovery.action == "committed_retained"
    assert all((tmp_path / path).read_bytes() == values[path] for path in OUTPUTS)
    assert transaction.transaction_directory.exists()


def test_formal_publication_can_retain_committed_until_outer_gates_pass(
    tmp_path: Path,
) -> None:
    transaction = _transaction(tmp_path)
    transaction.prepare()
    transaction.bind_test()
    _append_all(transaction)
    values = _complete_and_validate(transaction)

    published = transaction.publish_and_commit()

    inspection = _transaction(tmp_path).inspect()
    assert inspection.phase is AttemptPhase.COMMITTED
    assert inspection.output_state == "published"
    assert transaction.transaction_directory.exists()
    assert all((tmp_path / path).read_bytes() == values[path] for path in OUTPUTS)
    assert tuple(record.relative_path for record in published) == OUTPUTS

    retired = _transaction(tmp_path).retire_committed()
    assert tuple(record.relative_path for record in retired) == OUTPUTS
    assert not transaction.transaction_directory.exists()


def test_retained_committed_rejects_tamper_before_explicit_retirement(tmp_path: Path) -> None:
    transaction = _transaction(tmp_path)
    transaction.prepare()
    transaction.bind_test()
    _append_all(transaction)
    _complete_and_validate(transaction)
    transaction.publish_and_commit(retain_committed_transaction=True)
    (tmp_path / OUTPUTS[0]).write_bytes(b"changed-after-commit")

    with pytest.raises(AttemptTransactionTamperError, match="output"):
        _transaction(tmp_path).retire_committed()


def test_publish_retain_flag_and_retire_phase_are_strict(tmp_path: Path) -> None:
    transaction = _transaction(tmp_path)
    transaction.prepare()
    with pytest.raises(AttemptTransactionError, match="COMMITTED"):
        transaction.retire_committed()

    transaction.bind_test()
    _append_all(transaction)
    _complete_and_validate(transaction)
    with pytest.raises(TypeError, match="boolean"):
        transaction.publish_and_commit(retain_committed_transaction=1)  # type: ignore[arg-type]


def test_low_performance_never_controls_completion_or_retry(tmp_path: Path) -> None:
    transaction = _transaction(tmp_path)
    transaction.prepare()
    transaction.bind_test()
    _append_all(transaction, outcome="off_track")
    _seal(transaction)

    inspection = transaction.complete_evaluation(_formal_outputs(b"all-failed"))

    assert inspection.phase is AttemptPhase.EVALUATION_COMPLETE
    assert inspection.journal_record_count == 60


@pytest.mark.parametrize("target", ["manifest", "backup", "journal", "staged"])
def test_tampered_durable_bytes_are_rejected(tmp_path: Path, target: str) -> None:
    destination = tmp_path / OUTPUTS[0]
    destination.parent.mkdir(parents=True)
    destination.write_bytes(b"original")
    transaction = _transaction(tmp_path)
    transaction.prepare()
    if target in {"journal", "staged"}:
        transaction.bind_test()
    if target == "journal":
        _append(transaction, _record("pid", 0))
        (transaction.transaction_directory / "episode-journal.jsonl").write_bytes(b"not-json\n")
    elif target == "manifest":
        path = transaction.transaction_directory / "manifest.json"
        path.write_bytes(path.read_bytes() + b" ")
    elif target == "backup":
        (transaction.transaction_directory / "backups/000.bin").write_bytes(b"changed")
    else:
        _append_all(transaction)
        _seal(transaction)
        transaction.complete_evaluation(_formal_outputs())
        (transaction.transaction_directory / "final-staged/000.bin").write_bytes(b"changed")

    with pytest.raises(AttemptTransactionTamperError):
        _transaction(tmp_path).inspect()
    assert transaction.transaction_directory.exists()


def test_symlink_and_unknown_transaction_residue_are_rejected(tmp_path: Path) -> None:
    transaction = _transaction(tmp_path)
    transaction.prepare()
    residue = transaction.transaction_directory / "unexpected"
    residue.write_bytes(b"residue")
    with pytest.raises(AttemptTransactionTamperError, match="residue"):
        transaction.inspect()
    residue.unlink()
    (transaction.transaction_directory / "blobs/link").symlink_to(tmp_path / "outside")
    with pytest.raises(AttemptTransactionTamperError):
        transaction.inspect()


def test_identity_drift_cannot_recover_existing_attempt(tmp_path: Path) -> None:
    transaction = _transaction(tmp_path)
    transaction.prepare()
    changed = AttemptIdentity(
        source_revision=IDENTITY.source_revision,
        source_tree_sha256=IDENTITY.source_tree_sha256,
        config_sha256="9" * 64,
        pixi_lock_sha256=IDENTITY.pixi_lock_sha256,
        input_sha256=IDENTITY.input_sha256,
    )

    with pytest.raises(AttemptTransactionTamperError, match="identity"):
        _transaction(tmp_path, identity=changed).recover()
    assert transaction.transaction_directory.exists()

    changed_source_tree = AttemptIdentity(
        source_revision=IDENTITY.source_revision,
        source_tree_sha256="8" * 64,
        config_sha256=IDENTITY.config_sha256,
        pixi_lock_sha256=IDENTITY.pixi_lock_sha256,
        input_sha256=IDENTITY.input_sha256,
    )
    with pytest.raises(AttemptTransactionTamperError, match="identity"):
        _transaction(tmp_path, identity=changed_source_tree).recover()
    assert transaction.transaction_directory.exists()


def test_output_changed_to_unrecognized_bytes_is_never_overwritten_by_recovery(
    tmp_path: Path,
) -> None:
    destination = tmp_path / OUTPUTS[0]
    destination.parent.mkdir(parents=True)
    destination.write_bytes(b"original")
    transaction = _transaction(tmp_path)
    transaction.prepare()
    destination.write_bytes(b"concurrent-user-change")

    with pytest.raises(AttemptTransactionTamperError, match="neither original nor staged"):
        transaction.recover()
    assert destination.read_bytes() == b"concurrent-user-change"
    assert transaction.transaction_directory.exists()


def test_root_symlink_is_rejected(tmp_path: Path) -> None:
    real = tmp_path / "real"
    real.mkdir()
    link = tmp_path / "link"
    link.symlink_to(real, target_is_directory=True)

    with pytest.raises(ValueError, match="non-symlink"):
        _transaction(link)


def test_journal_is_fsynced_after_each_append(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from controller_learning.evaluation import attempt_transaction as module

    transaction = _transaction(tmp_path)
    transaction.prepare()
    transaction.bind_test()
    real_fsync = module.os.fsync
    fsynced: list[int] = []

    def track_fsync(descriptor: int) -> None:
        fsynced.append(descriptor)
        real_fsync(descriptor)

    monkeypatch.setattr(module.os, "fsync", track_fsync)
    _append(transaction, _record("pid", 0))

    assert fsynced
    assert os.stat(transaction.transaction_directory / "episode-journal.jsonl").st_size > 0
