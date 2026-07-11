"""Tests for the read-only M8 attempt-001 replacement lineage gate."""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import stat
from pathlib import Path

import pytest

from controller_learning.evaluation.replacement import (
    M8_ATTEMPT_001_FAILURE_REPORT_RELATIVE_PATH,
    M8_ATTEMPT_001_TRANSACTION_RELATIVE_PATH,
    M8_REPLACEMENT_RUN_ID,
    ReplacementEligibilityError,
    build_failure_report,
    canonical_failure_report_bytes,
    validate_failure_report_bytes,
    validate_failure_report_file,
    validate_local_predecessor,
)

PROJECT_ROOT = Path(__file__).resolve().parents[3]
REPORT_PATH = PROJECT_ROOT / M8_ATTEMPT_001_FAILURE_REPORT_RELATIVE_PATH
REPORT_SHA256 = "60bdb6d038b27867b13e1a12455b46e6717d1840bff65f1e072de06692645235"


def _copy_tree(source: Path, destination: Path, *, ignore_pycache: bool = False) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    ignore = shutil.ignore_patterns("__pycache__") if ignore_pycache else None
    shutil.copytree(source, destination, copy_function=shutil.copy2, ignore=ignore)


def _project_fixture(tmp_path: Path) -> Path:
    root = tmp_path / "project"
    root.mkdir(parents=True)
    _copy_tree(
        PROJECT_ROOT / M8_ATTEMPT_001_TRANSACTION_RELATIVE_PATH,
        root / M8_ATTEMPT_001_TRANSACTION_RELATIVE_PATH,
    )
    _copy_tree(
        PROJECT_ROOT / "runs/m8_final_controller_snapshot",
        root / "runs/m8_final_controller_snapshot",
    )
    for controller in ("pid", "mpc", "ppo"):
        _copy_tree(
            PROJECT_ROOT / "controllers" / controller,
            root / "controllers" / controller,
            ignore_pycache=True,
        )
    config = root / "configs/final_evaluation.toml"
    config.parent.mkdir(parents=True)
    shutil.copy2(PROJECT_ROOT / "configs/final_evaluation.toml", config)
    report = root / M8_ATTEMPT_001_FAILURE_REPORT_RELATIVE_PATH
    report.parent.mkdir(parents=True)
    shutil.copy2(REPORT_PATH, report)
    return root


def _tree_identity(path: Path) -> tuple[tuple[object, ...], ...]:
    records: list[tuple[object, ...]] = []
    for directory, directories, files in os.walk(path, topdown=True, followlinks=False):
        current = Path(directory)
        relative = current.relative_to(path).as_posix()
        metadata = current.lstat()
        records.append(("d", relative, stat.S_IMODE(metadata.st_mode)))
        for name in sorted((*directories, *files)):
            candidate = current / name
            child = candidate.relative_to(path).as_posix()
            metadata = candidate.lstat()
            if stat.S_ISLNK(metadata.st_mode):
                records.append(("l", child, stat.S_IMODE(metadata.st_mode), os.readlink(candidate)))
            elif stat.S_ISREG(metadata.st_mode):
                payload = candidate.read_bytes()
                records.append(
                    (
                        "f",
                        child,
                        stat.S_IMODE(metadata.st_mode),
                        len(payload),
                        hashlib.sha256(payload).hexdigest(),
                    )
                )
    return tuple(sorted(records))


def _rewrite_canonical_json(path: Path, mutate: object) -> None:
    value = json.loads(path.read_bytes())
    mutate(value)
    path.write_bytes(canonical_failure_report_bytes(value))


def test_public_report_is_canonical_and_has_the_frozen_digest() -> None:
    payload = REPORT_PATH.read_bytes()
    report = validate_failure_report_bytes(payload, expected_sha256=REPORT_SHA256)

    assert hashlib.sha256(payload).hexdigest() == REPORT_SHA256
    assert report["predecessor"]["journal_record_count"] == 0
    assert report["predecessor"]["execution_evidence"] is None
    assert report["failure"]["workload"] is None
    assert report["failure"]["infrastructure_phase"] == "environment_create"
    assert report["transaction"]["output_count"] == 24


def test_tmp_predecessor_validates_read_only(tmp_path: Path) -> None:
    root = _project_fixture(tmp_path)
    transaction = root / M8_ATTEMPT_001_TRANSACTION_RELATIVE_PATH
    snapshot = root / "runs/m8_final_controller_snapshot"
    before = (_tree_identity(transaction), _tree_identity(snapshot))

    validation = validate_local_predecessor(
        root,
        M8_ATTEMPT_001_FAILURE_REPORT_RELATIVE_PATH,
        expected_sha256=REPORT_SHA256,
    )

    assert validation.eligible is True
    assert validation.report_sha256 == REPORT_SHA256
    assert validation.transaction_tree_sha256 == (
        "746787724df04c0fcc741cb797c8a18affc34f6cab10ebd44cc34d3bacfd304f"
    )
    assert validation.predecessor_source_revision == "fa26064bbdccf5433ed578384c6a115ae8c489cc"
    assert validation.successor_run_id == M8_REPLACEMENT_RUN_ID
    assert (_tree_identity(transaction), _tree_identity(snapshot)) == before


def test_tmp_builder_reproduces_the_public_report(tmp_path: Path) -> None:
    root = _project_fixture(tmp_path)

    assert canonical_failure_report_bytes(build_failure_report(root)) == REPORT_PATH.read_bytes()


@pytest.mark.parametrize(
    ("case", "expected_error"),
    (
        ("journal", "journal is not empty"),
        ("episode_blob", "transaction directories differ"),
        ("execution_seal", "transaction directories differ"),
        ("extra_blob", "transaction files differ"),
        ("state_phase", "state/manifest binding differs"),
        ("workload", "authorized failure"),
        ("output", "output exists"),
        ("final_staged", "transaction directories differ"),
        ("publication", "transaction directories differ"),
    ),
)
def test_local_gate_rejects_ineligible_predecessor_state(
    tmp_path: Path,
    case: str,
    expected_error: str,
) -> None:
    root = _project_fixture(tmp_path)
    transaction = root / M8_ATTEMPT_001_TRANSACTION_RELATIVE_PATH
    if case == "journal":
        (transaction / "episode-journal.jsonl").write_bytes(b"{}\n")
    elif case == "episode_blob":
        target = transaction / "blobs/episodes/pid"
        target.mkdir(parents=True)
        (target / "row_000_trajectory.json").write_bytes(b"{}\n")
    elif case == "execution_seal":
        target = transaction / "blobs/execution"
        target.mkdir()
        (target / "final_evidence.json").write_bytes(b"{}\n")
    elif case == "extra_blob":
        (transaction / "blobs/extra.bin").write_bytes(b"extra")
    elif case == "state_phase":
        _rewrite_canonical_json(
            transaction / "state.json",
            lambda value: value.update({"phase": "EVALUATION_COMPLETE", "phase_index": 2}),
        )
    elif case == "workload":
        _rewrite_canonical_json(
            transaction / "blobs/failures/final-workload.json",
            lambda value: value.update({"workload": "pid"}),
        )
    elif case == "output":
        output = root / "benchmarks/v0.1/m8_final_results.csv"
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text("performance\n", encoding="utf-8")
    elif case == "final_staged":
        (transaction / "final-staged").mkdir()
    elif case == "publication":
        (transaction / "publication").mkdir()
    else:  # pragma: no cover - fixed parametrization
        raise AssertionError(case)

    with pytest.raises(ReplacementEligibilityError, match=expected_error):
        validate_local_predecessor(
            root,
            M8_ATTEMPT_001_FAILURE_REPORT_RELATIVE_PATH,
            expected_sha256=REPORT_SHA256,
        )


def test_local_gate_rejects_controller_and_official_hash_drift(tmp_path: Path) -> None:
    root = _project_fixture(tmp_path)
    live_config = root / "controllers/pid/config.toml"
    live_config.write_bytes(live_config.read_bytes() + b"\n")

    with pytest.raises(ReplacementEligibilityError, match="live/snapshot identity differs"):
        build_failure_report(root)

    root = _project_fixture(tmp_path / "official")
    config = root / "configs/final_evaluation.toml"
    config.write_bytes(
        config.read_bytes().replace(
            b'asset_sha256 = "0d654395',
            b'asset_sha256 = "1d654395',
            1,
        )
    )
    with pytest.raises(ReplacementEligibilityError, match="official Test hash bindings changed"):
        build_failure_report(root)


def test_local_gate_requires_the_active_read_only_snapshot(tmp_path: Path) -> None:
    root = _project_fixture(tmp_path)
    snapshot = root / "runs/m8_final_controller_snapshot"
    snapshot.chmod(0o755)

    with pytest.raises(ReplacementEligibilityError, match="mode differs"):
        build_failure_report(root)


def test_report_validation_rejects_noncanonical_hash_and_symlink(tmp_path: Path) -> None:
    payload = REPORT_PATH.read_bytes()
    pretty = json.dumps(json.loads(payload), indent=2).encode("utf-8")
    with pytest.raises(ReplacementEligibilityError, match="canonical JSON"):
        validate_failure_report_bytes(pretty)
    with pytest.raises(ReplacementEligibilityError, match="SHA-256 differs"):
        validate_failure_report_bytes(payload, expected_sha256="0" * 64)

    link = tmp_path / "report.json"
    link.symlink_to(REPORT_PATH)
    with pytest.raises(ReplacementEligibilityError, match="non-symlink regular file"):
        validate_failure_report_file(link)


def test_report_validation_rejects_relaxed_replacement_or_output_claim() -> None:
    report = json.loads(REPORT_PATH.read_bytes())
    report["authorization"]["max_replacement_attempts"] = 2
    with pytest.raises(ReplacementEligibilityError, match="authorization differs"):
        validate_failure_report_bytes(canonical_failure_report_bytes(report))

    report = json.loads(REPORT_PATH.read_bytes())
    report["transaction"]["outputs"][0]["local_state"] = "present"
    report["transaction"]["tree_sha256"] = report["transaction"]["tree_sha256"]
    with pytest.raises(ReplacementEligibilityError, match="output identity differs"):
        validate_failure_report_bytes(canonical_failure_report_bytes(report))
