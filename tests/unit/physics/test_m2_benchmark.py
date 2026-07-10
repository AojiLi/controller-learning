"""Tests for the formal M2 benchmark protocol and subprocess orchestration."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from controller_learning.physics import m2_benchmark


def _worker_result(
    spec: m2_benchmark.ScaleSpec,
    *,
    process_id: int = 100,
    status: str = "pass",
) -> dict:
    result = {
        "schema_version": m2_benchmark.WORKER_SCHEMA_VERSION,
        "protocol_version": m2_benchmark.PROTOCOL_VERSION,
        "status": status,
        "process_id": process_id,
        "num_worlds": spec.num_worlds,
        "environment_steps": spec.environment_steps,
        "chunk_steps": spec.chunk_steps,
        "physics_substeps_per_environment_step": 10,
        "runtime": {},
        "capacities": {},
        "numerical": {},
        "timing": {},
        "memory": {},
        "checks": [],
    }
    if spec.num_worlds == 1:
        result["cpu_gpu_consistency"] = {"status": "pass"}
    return result


@pytest.mark.parametrize(
    "kwargs",
    [
        {"num_worlds": 0, "environment_steps": 100},
        {"num_worlds": 1, "environment_steps": 0},
        {"num_worlds": 1, "environment_steps": 101, "chunk_steps": 100},
        {"num_worlds": 1, "environment_steps": 100, "chunk_steps": 0},
    ],
)
def test_scale_spec_rejects_non_executable_shapes(kwargs) -> None:
    with pytest.raises(ValueError):
        m2_benchmark.ScaleSpec(**kwargs)


def test_extract_worker_json_uses_the_last_valid_sentinel() -> None:
    prefix = m2_benchmark.WORKER_JSON_PREFIX
    stdout = "\n".join(
        (
            "native runtime noise",
            f"{prefix}{{not-json}}",
            f"{prefix}{json.dumps({'status': 'pass', 'value': 2})}",
            "trailing noise",
        )
    )

    assert m2_benchmark.extract_worker_json(stdout) == {"status": "pass", "value": 2}


def test_strict_json_writer_is_atomic_and_rejects_nan(tmp_path) -> None:
    output = tmp_path / "nested" / "report.json"
    m2_benchmark.write_strict_json(output, {"status": "pass", "value": 1.25})

    assert json.loads(output.read_text()) == {"status": "pass", "value": 1.25}
    with pytest.raises(ValueError, match="Out of range float values"):
        m2_benchmark.write_strict_json(output, {"value": float("nan")})
    assert json.loads(output.read_text()) == {"status": "pass", "value": 1.25}


def test_worker_contract_reports_protocol_mismatches() -> None:
    spec = m2_benchmark.ScaleSpec(1, 100, 100)
    payload = _worker_result(spec)
    payload["num_worlds"] = 64
    payload.pop("cpu_gpu_consistency")

    violations = m2_benchmark.validate_worker_result(payload, spec)

    assert any("num_worlds" in violation for violation in violations)
    assert any("cpu_gpu_consistency" in violation for violation in violations)


def test_timeout_output_bytes_are_safe_for_strict_json() -> None:
    assert m2_benchmark._output_tail(b"prefix\xfftail", maximum_chars=5) == "�tail"


def test_formal_report_passes_only_with_clean_fresh_passing_workers(
    monkeypatch,
    tmp_path,
) -> None:
    process_ids = {
        spec.num_worlds: 10_000 + index
        for index, spec in enumerate(m2_benchmark.FORMAL_SCALE_SPECS)
    }

    def fake_git(project_root: Path, *args: str) -> str:
        if args == ("rev-parse", "HEAD"):
            return "a" * 40
        if args == ("status", "--porcelain"):
            return ""
        raise AssertionError(args)

    def fake_worker(project_root, worker_path, spec, **kwargs):
        return _worker_result(spec, process_id=process_ids[spec.num_worlds])

    monkeypatch.setattr(m2_benchmark, "_git", fake_git)
    monkeypatch.setattr(m2_benchmark, "_run_scale_worker", fake_worker)
    monkeypatch.setattr(
        m2_benchmark,
        "_source_hashes",
        lambda project_root, *, worker_path: {"source": "b" * 64},
    )

    report = m2_benchmark.run_m2_benchmark(tmp_path)

    assert report["status"] == "pass"
    assert report["selection"] == {
        "m2_passed": True,
        "ready_for_m3": True,
        "formal_protocol": True,
        "all_workers_passed": True,
        "evidence_valid": True,
    }
    assert report["cpu_gpu_consistency"] == {"status": "pass"}


def test_dirty_worktree_invalidates_otherwise_passing_evidence(monkeypatch, tmp_path) -> None:
    def fake_git(project_root: Path, *args: str) -> str:
        return "uncommitted.py" if args == ("status", "--porcelain") else "a" * 40

    monkeypatch.setattr(m2_benchmark, "_git", fake_git)
    monkeypatch.setattr(
        m2_benchmark,
        "_run_scale_worker",
        lambda project_root, worker_path, spec, **kwargs: _worker_result(
            spec,
            process_id=20_000 + spec.num_worlds,
        ),
    )
    monkeypatch.setattr(
        m2_benchmark,
        "_source_hashes",
        lambda project_root, *, worker_path: {},
    )

    report = m2_benchmark.run_m2_benchmark(tmp_path)

    assert report["status"] == "fail"
    assert report["selection"]["evidence_valid"] is False


def test_formal_evidence_rejects_a_noncanonical_worker(monkeypatch, tmp_path) -> None:
    monkeypatch.setattr(
        m2_benchmark,
        "_git",
        lambda project_root, *args: "a" * 40 if args == ("rev-parse", "HEAD") else "",
    )
    monkeypatch.setattr(
        m2_benchmark,
        "_run_scale_worker",
        lambda project_root, worker_path, spec, **kwargs: _worker_result(
            spec,
            process_id=30_000 + spec.num_worlds,
        ),
    )
    monkeypatch.setattr(
        m2_benchmark,
        "_source_hashes",
        lambda project_root, *, worker_path: {"worker": str(worker_path)},
    )

    report = m2_benchmark.run_m2_benchmark(
        tmp_path,
        worker_path=tmp_path / "substitute_worker.py",
    )

    assert report["status"] == "fail"
    assert report["selection"]["formal_protocol"] is False
    check = next(check for check in report["checks"] if check["id"] == "protocol.canonical_worker")
    assert check["passed"] is False


def test_formal_evidence_rejects_source_changes_during_workers(monkeypatch, tmp_path) -> None:
    hash_call = 0

    def changing_hashes(project_root, *, worker_path):
        nonlocal hash_call
        hash_call += 1
        return {"source": str(hash_call)}

    monkeypatch.setattr(
        m2_benchmark,
        "_git",
        lambda project_root, *args: "a" * 40 if args == ("rev-parse", "HEAD") else "",
    )
    monkeypatch.setattr(
        m2_benchmark,
        "_run_scale_worker",
        lambda project_root, worker_path, spec, **kwargs: _worker_result(
            spec,
            process_id=40_000 + spec.num_worlds,
        ),
    )
    monkeypatch.setattr(m2_benchmark, "_source_hashes", changing_hashes)

    report = m2_benchmark.run_m2_benchmark(tmp_path)

    assert report["status"] == "fail"
    assert report["selection"]["evidence_valid"] is False
