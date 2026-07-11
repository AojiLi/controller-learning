"""Synthetic-only tests for strict M8 Controller manifests and the global report."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping
from dataclasses import dataclass, replace
from pathlib import Path
from types import MappingProxyType

import numpy as np
import pytest

from controller_learning.control import EpisodeRunResult
from controller_learning.envs.episode import initialize_episode_identities
from controller_learning.evaluation.controller_identity import (
    FrozenControllerIdentity,
    capture_frozen_controller_identity,
)
from controller_learning.evaluation.final_benchmark import (
    M8_CONTROLLER_ORDER,
    M8_FINAL_REPORT_SCHEMA_VERSION,
    M8FinalEvaluationConfig,
    controller_output_paths,
    formal_output_paths,
    load_m8_final_evaluation_config,
)
from controller_learning.evaluation.final_metrics import (
    FINAL_METRICS_SCHEMA_VERSION,
    EpisodeMetricSamples,
    build_final_metrics_data,
    canonical_final_metrics_bytes,
    compute_episode_metric_samples,
)
from controller_learning.evaluation.final_report import (
    M8_CONTROLLER_RUN_MANIFEST_SCHEMA_VERSION,
    ArtifactRecord,
    DurableExecutionEvidenceSeal,
    EnvironmentLifecycleEvidence,
    ExecutionEvidence,
    FinalConfigEvidence,
    FinalReportArtifactError,
    MemoryEvidence,
    PrivacyEvidence,
    RuntimeEvidence,
    SourceEvidence,
    TransactionEvidence,
    canonical_controller_run_manifest_json_bytes,
    canonical_m8_final_report_json_bytes,
    validate_controller_run_manifest_json_bytes,
    validate_m8_final_report_json_bytes,
    validate_m8_publication,
)
from controller_learning.evaluation.final_report import (
    TestAccessAuditEvidence as M8AccessAuditEvidence,
)
from controller_learning.evaluation.final_results import (
    FINAL_COMPARISON_SCHEMA_VERSION,
    FINAL_CONTROLLER_RESULTS_SCHEMA_VERSION,
    FINAL_CONTROLLER_SUMMARY_SCHEMA_VERSION,
    FINAL_RESULTS_ACTION_LIMITS,
    FinalControllerResult,
    FinalEpisodeResult,
    canonical_controller_results_csv_bytes,
    canonical_controller_summary_json_bytes,
    canonical_final_comparison_csv_bytes,
)
from controller_learning.evaluation.final_runtime import (
    FINAL_ENVIRONMENT_LIFECYCLE_SCHEMA_VERSION,
    FINAL_MEMORY_EVIDENCE_SCHEMA_VERSION,
    FINAL_RUNTIME_EVIDENCE_SCHEMA_VERSION,
    FINAL_RUNTIME_PACKAGE_NAMES,
)
from controller_learning.evaluation.test_assets import (
    M8_TEST_POOL_ACCESS_SCHEMA_VERSION,
)
from controller_learning.evaluation.test_assets import (
    TestPoolAccessEvidence as M8PoolAccessEvidence,
)
from controller_learning.evaluation.trajectory import (
    TRAJECTORY_SCHEMA_VERSION,
    EpisodeTrajectory,
    RecordedControllerEpisode,
    canonical_trajectory_json_bytes,
)
from controller_learning.tracks.types import TrackCapacity
from controller_learning.visualization.final_results import (
    render_controller_telemetry_png,
    render_final_comparison_png,
)
from controller_learning.visualization.replay import render_trajectory_overview_png

PROJECT_ROOT = Path(__file__).resolve().parents[3]


def _lines_digest(values: tuple[int, ...]) -> str:
    digest = hashlib.sha256()
    for value in values:
        digest.update(str(value).encode("ascii"))
        digest.update(b"\n")
    return digest.hexdigest()


def _recorded_episode(*, row: int, reason: int, step_count: int) -> RecordedControllerEpisode:
    identity = initialize_episode_identities(row, 1)
    track_id = 2_000_000 + row
    reset_info = MappingProxyType(
        {
            "episode_seed": int(identity.episode_seed[0]),
            "controller_seed": int(identity.controller_seed[0]),
            "track_id": track_id,
            "benchmark_version": "0.1",
            "termination_reason": 0,
            "lap_completed": False,
            "lap_time_s": 0.0,
        }
    )
    success = reason == 1
    final_info = MappingProxyType(
        {
            **dict(reset_info),
            "termination_reason": reason,
            "lap_completed": success,
            "lap_time_s": step_count * 0.05 if success else 0.0,
        }
    )
    centerline = np.asarray(
        ((0.0, 0.0), (10.0, 0.0), (10.0, 10.0), (0.0, 10.0), (0.0, 0.0)),
        dtype=np.float32,
    )
    position = np.column_stack(
        (
            np.linspace(0.0, 0.5 * step_count, step_count + 1, dtype=np.float32),
            np.linspace(0.0, 0.1, step_count + 1, dtype=np.float32),
        )
    )
    velocity = np.column_stack(
        (
            np.linspace(0.0, 2.0, step_count + 1, dtype=np.float32),
            np.zeros(step_count + 1, dtype=np.float32),
        )
    )
    action = np.column_stack(
        (
            np.linspace(-0.1, 0.1, step_count, dtype=np.float32),
            np.linspace(1.0, 2.0, step_count, dtype=np.float32),
        )
    )
    reward = np.linspace(0.1, 0.2, step_count, dtype=np.float32)
    terminated = np.zeros(step_count, dtype=np.bool_)
    truncated = np.zeros(step_count, dtype=np.bool_)
    (truncated if reason == 4 else terminated)[-1] = True
    trajectory = EpisodeTrajectory(
        reset_info=reset_info,
        final_info=final_info,
        centerline_m=centerline,
        left_boundary_m=centerline + np.asarray((0.0, 1.0), dtype=np.float32),
        right_boundary_m=centerline - np.asarray((0.0, 1.0), dtype=np.float32),
        track_mask=np.ones(centerline.shape[0], dtype=np.bool_),
        track_length_m=40.0,
        position_m=position,
        yaw_rad=np.zeros(step_count + 1, dtype=np.float32),
        velocity_body_mps=velocity,
        yaw_rate_rad_s=np.zeros(step_count + 1, dtype=np.float32),
        steering_angle_rad=np.zeros(step_count + 1, dtype=np.float32),
        track_progress=np.linspace(0.0, 1.0, step_count + 1, dtype=np.float32),
        action=action,
        reward=reward,
        terminated=terminated,
        truncated=truncated,
    )
    compute_times = tuple(0.001 + index * 0.00001 for index in range(step_count))
    result = EpisodeRunResult(
        steps=step_count,
        total_reward=float(np.sum(reward, dtype=np.float64)),
        terminated=reason != 4,
        truncated=reason == 4,
        final_info=final_info,
        debug_commands=(),
        controller_import_time_s=0.001,
        controller_init_time_s=0.002,
        compute_times_s=compute_times,
    )
    return RecordedControllerEpisode(result=result, trajectory=trajectory)


def _controller_result(
    controller: str,
    *,
    success_count: int,
) -> tuple[FinalControllerResult, RecordedControllerEpisode]:
    rows: list[FinalEpisodeResult] = []
    samples: list[EpisodeMetricSamples] = []
    row_zero: RecordedControllerEpisode | None = None
    for row in range(20):
        reason = 1 if row < success_count else (2, 3, 4)[row % 3]
        recorded = _recorded_episode(row=row, reason=reason, step_count=2 + row % 2)
        metric = compute_episode_metric_samples(
            recorded,
            reset_seed=row,
            action_limits=FINAL_RESULTS_ACTION_LIMITS,
        )
        rows.append(FinalEpisodeResult(controller, row, recorded, metric))
        samples.append(metric)
        if row == 0:
            row_zero = recorded
    assert row_zero is not None
    return (
        FinalControllerResult(
            controller_name=controller,
            episodes=tuple(rows),
            metrics=build_final_metrics_data(controller, samples),
        ),
        row_zero,
    )


def _runtime() -> RuntimeEvidence:
    return RuntimeEvidence.from_mapping(
        {
            "schema_version": FINAL_RUNTIME_EVIDENCE_SCHEMA_VERSION,
            "platform": "Linux",
            "machine": "x86_64",
            "kernel": "6.8.0-synthetic",
            "python": "3.11.9",
            "cpu_model": "Synthetic x86 CPU",
            "cuda_runtime": "CUDA 12.8",
            "cuda_driver": "570.00",
            "cuda_device_order": "PCI_BUS_ID",
            "cuda_visible_devices_configured": True,
            "xla_python_client_preallocate": "false",
            "jax_device": {
                "id": 0,
                "platform": "gpu",
                "device_kind": "NVIDIA Synthetic GPU",
            },
            "packages": {name: "1.0.0" for name in FINAL_RUNTIME_PACKAGE_NAMES},
            "selected_gpu": {
                "index": 0,
                "uuid": "redacted",
                "name": "NVIDIA Synthetic GPU",
                "driver_version": "570.00",
                "memory_total_mib": 24576.0,
            },
        }
    )


def _memory() -> MemoryEvidence:
    return MemoryEvidence.from_mapping(
        {
            "schema_version": FINAL_MEMORY_EVIDENCE_SCHEMA_VERSION,
            "sampling_method": "JAX synchronized labelled phase boundary sampling",
            "sample_count": 2,
            "samples": [
                {
                    "label": "before_environment",
                    "synchronized": True,
                    "process_vram_mib": 256.0,
                    "jax_bytes_in_use": 0,
                    "jax_peak_bytes_in_use": 0,
                },
                {
                    "label": "after_environment_close",
                    "synchronized": True,
                    "process_vram_mib": 512.0,
                    "jax_bytes_in_use": 0,
                    "jax_peak_bytes_in_use": 128 * 1024**2,
                },
            ],
            "peak_sampled_process_vram_mib": 512.0,
            "peak_jax_allocator_bytes": 128 * 1024**2,
            "final_jax_live_bytes": 0,
        }
    )


def _lifecycle(step_count: int) -> dict[str, object]:
    return {
        "schema_version": FINAL_ENVIRONMENT_LIFECYCLE_SCHEMA_VERSION,
        "environment_instance_count": 1,
        "environment_create_wall_time_s": 0.1,
        "first_reset_wall_time_including_lazy_compilation_s": 2.0,
        "first_step_wall_time_including_lazy_compilation_s": 3.0,
        "reset_count": 60,
        "expected_reset_count": 60,
        "step_count": step_count,
        "expected_step_count": step_count,
        "close_count": 1,
        "method": (
            "wall clock around environment construction and the first public reset and step; "
            "the first-call timings include any lazy compilation"
        ),
    }


def _access_audit() -> M8AccessAuditEvidence:
    return M8AccessAuditEvidence(
        open_event_counts={"official_test_manifest": 2, "official_test_asset": 1},
        open_event_sequence=(
            {"category": "official_test_manifest", "flags": 0, "mode": "r"},
            {"category": "official_test_manifest", "flags": 0, "mode": "r"},
            {"category": "official_test_asset", "flags": 0, "mode": "r"},
        ),
    )


def _artifact_record(path: str, content: bytes) -> ArtifactRecord:
    if path.endswith("/metrics.npz"):
        metadata = ("application/x-npz", FINAL_METRICS_SCHEMA_VERSION)
    elif path.endswith("/results.csv"):
        metadata = ("text/csv", FINAL_CONTROLLER_RESULTS_SCHEMA_VERSION)
    elif path.endswith("/summary.json"):
        metadata = ("application/json", FINAL_CONTROLLER_SUMMARY_SCHEMA_VERSION)
    elif path.endswith("/run_manifest.json"):
        metadata = ("application/json", M8_CONTROLLER_RUN_MANIFEST_SCHEMA_VERSION)
    elif path.endswith("test_row_000_trajectory.json"):
        metadata = ("application/json", TRAJECTORY_SCHEMA_VERSION)
    elif path.endswith("m8_final_evaluation_report.json"):
        metadata = ("application/json", M8_FINAL_REPORT_SCHEMA_VERSION)
    elif path.endswith("m8_final_results.csv"):
        metadata = ("text/csv", FINAL_COMPARISON_SCHEMA_VERSION)
    elif path.endswith(".png"):
        metadata = ("image/png", None)
    else:  # pragma: no cover - fixture only receives frozen paths
        raise AssertionError(path)
    return ArtifactRecord.from_bytes(path, content, *metadata)


@dataclass(frozen=True, slots=True)
class SyntheticPublication:
    config: M8FinalEvaluationConfig
    results: Mapping[str, FinalControllerResult]
    outputs: Mapping[str, bytes]
    evidence: Mapping[str, object]
    controller_manifest_evidence: Mapping[str, Mapping[str, object]]


@pytest.fixture(scope="module")
def publication() -> SyntheticPublication:
    config = load_m8_final_evaluation_config(PROJECT_ROOT / "configs/final_evaluation.toml")
    results: dict[str, FinalControllerResult] = {}
    row_zero: dict[str, RecordedControllerEpisode] = {}
    for name, successes in zip(M8_CONTROLLER_ORDER, (8, 12, 10), strict=True):
        results[name], row_zero[name] = _controller_result(name, success_count=successes)
    track_ids = tuple(row.track_id for row in results["pid"].episodes)
    test_pool_access = M8PoolAccessEvidence(
        schema_version=M8_TEST_POOL_ACCESS_SCHEMA_VERSION,
        loaded_splits=("test",),
        benchmark_version="0.1",
        generator_version="synthetic-closed-track-v1",
        level_id=1,
        split="test",
        manifest_file="test.json",
        manifest_sha256=config.test_manifest_sha256,
        asset_file="test.npz",
        manifest_asset_sha256=config.test_asset_sha256,
        asset_file_sha256=config.test_asset_sha256,
        track_count=20,
        capacity=TrackCapacity(max_track_points=640, max_checkpoints=48),
        track_ids=track_ids,
        track_ids_sha256=_lines_digest(track_ids),
        geometry_hashes_sha256="b" * 64,
        loader_accessed_train=False,
        loader_accessed_validation=False,
    )
    identities = {
        name: capture_frozen_controller_identity(PROJECT_ROOT, name) for name in M8_CONTROLLER_ORDER
    }
    source = SourceEvidence("a" * 40)
    config_evidence = FinalConfigEvidence.from_bytes(
        config,
        (PROJECT_ROOT / "configs/final_evaluation.toml").read_bytes(),
    )
    pixi_lock = ArtifactRecord.from_bytes(
        "pixi.lock",
        (PROJECT_ROOT / "pixi.lock").read_bytes(),
        "application/yaml",
    )
    input_reports: dict[str, ArtifactRecord] = {}
    replacement_failure_payload: Mapping[str, object] | None = None
    for name, path in config.input_paths.items():
        if name == "m8_attempt_001_failure_report":
            content = (PROJECT_ROOT / path).read_bytes()
            replacement_failure_payload = json.loads(content)
            schema = replacement_failure_payload["schema_version"]
        else:
            content = f'{{"synthetic_input":"{name}"}}\n'.encode("ascii")
            schema = f"controller-learning.synthetic-{name}.v1"
        input_reports[name] = ArtifactRecord.from_bytes(
            path,
            content,
            "application/json",
            schema,
        )
    assert replacement_failure_payload is not None
    execution = ExecutionEvidence.from_results(
        results,
        wall_time_s=10.0,
        controller_wall_time_s={name: 3.0 for name in M8_CONTROLLER_ORDER},
        initialization_over_soft_limit_rows={name: () for name in M8_CONTROLLER_ORDER},
        measured_environment_lifecycle=_lifecycle(
            sum(result.summary.environment_steps for result in results.values())
        ),
    )
    test_access_audit = _access_audit()
    runtime = _runtime()
    memory = _memory()
    durable_execution_evidence = DurableExecutionEvidenceSeal.from_evidence(
        test_access_audit=test_access_audit,
        execution=execution,
        memory=memory,
        runtime=runtime,
        test_pool_access=test_pool_access,
    )
    evidence: dict[str, object] = {
        "source": source,
        "protocol_config": config,
        "config_evidence": config_evidence,
        "pixi_lock": pixi_lock,
        "input_reports": input_reports,
        "replacement_failure_report": replacement_failure_payload,
        "controller_identities_before": identities,
        "controller_identities_after": identities,
        "test_pool_access": test_pool_access,
        "test_access_audit": test_access_audit,
        "runtime": runtime,
        "memory": memory,
        "execution": execution,
        "durable_execution_evidence": durable_execution_evidence,
        "transaction": TransactionEvidence(),
        "privacy": PrivacyEvidence(),
    }
    outputs: dict[str, bytes] = {}
    for name in M8_CONTROLLER_ORDER:
        paths = controller_output_paths(config, name)
        result = results[name]
        trajectory = row_zero[name].trajectory
        samples = result.metrics.episode(0)
        outputs[paths["metrics"]] = canonical_final_metrics_bytes(result.metrics)
        outputs[paths["replay_trajectory"]] = canonical_trajectory_json_bytes(trajectory)
        outputs[paths["results"]] = canonical_controller_results_csv_bytes(result)
        outputs[paths["summary"]] = canonical_controller_summary_json_bytes(result)
        outputs[paths["telemetry"]] = render_controller_telemetry_png(
            controller_name=name,
            control_dt_s=config.control_dt_s,
            speed_mps=samples.speed_mps,
            lateral_error_m=samples.lateral_error_m,
            requested_action=samples.requested_action,
            steering_saturated=samples.steering_saturated,
            longitudinal_saturated=samples.longitudinal_saturated,
        )
        outputs[paths["trajectory"]] = render_trajectory_overview_png(trajectory)
    outputs[config.comparison_csv_path] = canonical_final_comparison_csv_bytes(results)
    first = row_zero["pid"].trajectory
    outputs[config.comparison_png_path] = render_final_comparison_png(
        benchmark_version="0.1",
        track_id=track_ids[0],
        centerline_m=first.centerline_m,
        left_boundary_m=first.left_boundary_m,
        right_boundary_m=first.right_boundary_m,
        track_mask=first.track_mask,
        trajectories_m={name: row_zero[name].trajectory.position_m for name in M8_CONTROLLER_ORDER},
    )

    controller_manifest_evidence: dict[str, Mapping[str, object]] = {}
    for name in M8_CONTROLLER_ORDER:
        paths = controller_output_paths(config, name)
        controller_outputs = {
            key: _artifact_record(path, outputs[path])
            for key, path in paths.items()
            if key != "run_manifest"
        }
        manifest_evidence = {
            "source": source,
            "protocol_config": config,
            "config_evidence": config_evidence,
            "pixi_lock": pixi_lock,
            "input_reports": input_reports,
            "controller_identity": identities[name],
            "test_pool_access": test_pool_access,
            "test_access_audit": evidence["test_access_audit"],
            "runtime": evidence["runtime"],
            "memory": evidence["memory"],
            "execution": execution,
            "durable_execution_evidence": durable_execution_evidence,
            "output_artifacts": controller_outputs,
        }
        controller_manifest_evidence[name] = MappingProxyType(manifest_evidence)
        outputs[paths["run_manifest"]] = canonical_controller_run_manifest_json_bytes(
            results[name],
            **manifest_evidence,  # type: ignore[arg-type]
        )

    records = {
        path: _artifact_record(path, outputs[path])
        for path in formal_output_paths(config)
        if path != config.report_path
    }
    outputs[config.report_path] = canonical_m8_final_report_json_bytes(
        results,
        **evidence,  # type: ignore[arg-type]
        output_artifacts=records,
    )
    assert set(outputs) == set(formal_output_paths(config))
    return SyntheticPublication(
        config=config,
        results=MappingProxyType(results),
        outputs=MappingProxyType(outputs),
        evidence=MappingProxyType(evidence),
        controller_manifest_evidence=MappingProxyType(controller_manifest_evidence),
    )


def test_artifact_record_from_bytes_binds_safe_path_hash_size_and_metadata() -> None:
    payload = b"artifact\n"
    record = ArtifactRecord.from_bytes(
        "results/0.1/pid/file.json",
        payload,
        "application/json",
        "synthetic.v1",
    )

    assert record.sha256 == hashlib.sha256(payload).hexdigest()
    assert record.size_bytes == len(payload)
    assert record.to_dict()["relative_path"] == "results/0.1/pid/file.json"
    with pytest.raises(FinalReportArtifactError, match="relative POSIX"):
        ArtifactRecord.from_bytes("../escape.json", payload, "application/json")
    with pytest.raises(FinalReportArtifactError, match="relative POSIX"):
        ArtifactRecord.from_bytes("/home/user/result.json", payload, "application/json")
    with pytest.raises(TypeError, match="immutable bytes"):
        ArtifactRecord.from_bytes("safe.json", bytearray(payload), "application/json")  # type: ignore[arg-type]


def test_typed_evidence_rejects_uuid_failures_retries_and_denied_access(
    publication: SyntheticPublication,
) -> None:
    runtime = publication.evidence["runtime"].to_dict()
    runtime["selected_gpu"]["uuid"] = "GPU-12345678-1234-1234-1234-123456789abc"
    with pytest.raises(FinalReportArtifactError, match="runtime evidence"):
        RuntimeEvidence.from_mapping(runtime)
    execution = publication.evidence["execution"]
    assert ExecutionEvidence.from_mapping(execution.to_dict()) == execution
    lifecycle = execution.measured_environment_lifecycle
    assert EnvironmentLifecycleEvidence.from_mapping(lifecycle.to_dict()) == lifecycle
    seal = publication.evidence["durable_execution_evidence"]
    assert DurableExecutionEvidenceSeal.from_mapping(seal.to_dict()).sha256 == seal.sha256
    with pytest.raises(FinalReportArtifactError, match="numerical_failure_count"):
        replace(execution, numerical_failure_count=1)
    with pytest.raises(FinalReportArtifactError, match="retry_count"):
        replace(execution, retry_count=1)
    with pytest.raises(FinalReportArtifactError, match="attempt count scope"):
        replace(
            publication.evidence["transaction"],
            attempt_count_scope="all_formal_attempts",
        )
    with pytest.raises(FinalReportArtifactError, match="denied_event_count"):
        replace(publication.evidence["test_access_audit"], denied_event_count=1)
    with pytest.raises(FinalReportArtifactError, match="guarded load"):
        replace(
            publication.evidence["test_access_audit"],
            all_track_reads_forbidden=False,
        )

    drifted_seal = seal.to_dict()
    drifted_seal["runtime"]["selected_gpu"]["name"] = "Different GPU"
    with pytest.raises(FinalReportArtifactError, match="durable execution seal differs"):
        DurableExecutionEvidenceSeal.from_mapping(drifted_seal).cross_check(
            test_access_audit=publication.evidence["test_access_audit"],
            execution=execution,
            memory=publication.evidence["memory"],
            runtime=publication.evidence["runtime"],
            test_pool_access=publication.evidence["test_pool_access"],
        )


def test_controller_manifest_is_canonical_and_exactly_recomputed(
    publication: SyntheticPublication,
) -> None:
    for name in M8_CONTROLLER_ORDER:
        path = controller_output_paths(publication.config, name)["run_manifest"]
        payload = publication.outputs[path]
        parsed = validate_controller_run_manifest_json_bytes(
            payload,
            publication.results[name],
            **publication.controller_manifest_evidence[name],  # type: ignore[arg-type]
        )
        assert parsed["schema_version"] == M8_CONTROLLER_RUN_MANIFEST_SCHEMA_VERSION
        assert parsed["output_artifact_count"] == 6
        assert len(parsed["frozen_input_reports"]) == 6
        assert parsed["runtime"]["selected_gpu"]["uuid"] == "redacted"
        assert parsed["runtime"]["pixi"]["lock_sha256"] == publication.evidence["pixi_lock"].sha256
        assert parsed["reset_seeds"] == list(range(20))
        assert payload.endswith(b"\n") and b"\r" not in payload


def test_global_report_has_protocol_status_rank_rows_and_complete_artifacts(
    publication: SyntheticPublication,
) -> None:
    report_path = publication.config.report_path
    records = {
        path: _artifact_record(path, publication.outputs[path])
        for path in formal_output_paths(publication.config)
        if path != report_path
    }
    parsed = validate_m8_final_report_json_bytes(
        publication.outputs[report_path],
        publication.results,
        **publication.evidence,  # type: ignore[arg-type]
        output_artifacts=records,
    )

    assert parsed["status"] == "passed"
    assert parsed["protocol_result"]["status_basis"] == ("protocol_and_artifact_validation_only")
    assert parsed["protocol_result"]["performance_pass_gate"] is False
    assert parsed["protocol_result"]["combined_score_present"] is False
    assert parsed["rank_order"] == ["mpc", "ppo", "pid"]
    assert [len(value["episode_rows"]) for value in parsed["controllers"]] == [20, 20, 20]
    assert parsed["output_artifact_count"] == len(parsed["output_artifacts"]) == 23
    assert parsed["controller_identities"]["unchanged"] is True
    assert parsed["test_pool_access"]["loaded_splits"] == ["test"]
    lineage = parsed["replacement_lineage"]
    assert lineage["schema_version"] == "controller-learning.m8-replacement-lineage.v1"
    assert lineage["predecessor"]["run_id"] == "m8-final-v0-1-001"
    assert lineage["predecessor"]["transaction_phase"] == "TEST_BOUND"
    assert lineage["predecessor"]["test_pool_load_completed"] is True
    assert lineage["predecessor"]["durable_episode_record_count"] == 0
    assert lineage["predecessor"]["environment_create_completed"] is False
    assert lineage["predecessor"]["environment_reset_count"] == 0
    assert lineage["predecessor"]["environment_step_count"] == 0
    assert lineage["predecessor"]["plugin_controller_instance_count"] == 0
    assert lineage["predecessor"]["performance_observed"] is False
    assert lineage["predecessor"]["failure"]["workload"] is None
    assert lineage["successor"]["run_id"] == "m8-final-v0-1-002"
    assert lineage["successor"]["current_transaction_attempt_count"] == 1
    assert lineage["successor"]["current_transaction_retry_count"] == 0
    assert lineage["authorization"]["third_attempt_allowed"] is False


def test_global_report_rejects_replacement_failure_payload_drift(
    publication: SyntheticPublication,
) -> None:
    report_path = publication.config.report_path
    records = {
        path: _artifact_record(path, publication.outputs[path])
        for path in formal_output_paths(publication.config)
        if path != report_path
    }
    replacement_report = json.loads(json.dumps(publication.evidence["replacement_failure_report"]))
    replacement_report["predecessor"]["performance_observed"] = True

    with pytest.raises(FinalReportArtifactError, match="failure report payload differs"):
        canonical_m8_final_report_json_bytes(
            publication.results,
            **{
                **publication.evidence,
                "replacement_failure_report": replacement_report,
            },  # type: ignore[arg-type]
            output_artifacts=records,
        )


@pytest.mark.parametrize(
    "mutation",
    ("missing", "reordered", "absolute", "secret", "nonfinite", "duplicate"),
)
def test_global_validator_rejects_partial_reordered_or_private_payloads(
    publication: SyntheticPublication,
    mutation: str,
) -> None:
    report_path = publication.config.report_path
    records = {
        path: _artifact_record(path, publication.outputs[path])
        for path in formal_output_paths(publication.config)
        if path != report_path
    }
    parsed = json.loads(publication.outputs[report_path])
    if mutation == "missing":
        parsed["output_artifacts"].pop()
        payload = (json.dumps(parsed, indent=2, sort_keys=True) + "\n").encode("ascii")
    elif mutation == "reordered":
        parsed["output_artifacts"].reverse()
        payload = (json.dumps(parsed, indent=2, sort_keys=True) + "\n").encode("ascii")
    elif mutation == "absolute":
        parsed["runtime"]["selected_gpu"]["name"] = "/home/user/private-project"
        payload = (json.dumps(parsed, indent=2, sort_keys=True) + "\n").encode("ascii")
    elif mutation == "secret":
        parsed["runtime"]["selected_gpu"]["name"] = "ghp_ABCDEFGHIJKLMNOPQRSTUVWXYZ"
        payload = (json.dumps(parsed, indent=2, sort_keys=True) + "\n").encode("ascii")
    elif mutation == "nonfinite":
        payload = publication.outputs[report_path].replace(
            b'"status": "passed"',
            b'"unexpected": NaN,\n  "status": "passed"',
        )
    else:
        payload = publication.outputs[report_path].replace(
            b"{\n",
            b'{\n  "status": "passed",\n',
            1,
        )

    with pytest.raises(FinalReportArtifactError):
        validate_m8_final_report_json_bytes(
            payload,
            publication.results,
            **publication.evidence,  # type: ignore[arg-type]
            output_artifacts=records,
        )


def test_builders_reject_identity_and_artifact_coverage_drift(
    publication: SyntheticPublication,
) -> None:
    name = "pid"
    evidence = dict(publication.controller_manifest_evidence[name])
    identity = evidence["controller_identity"]
    drifted = FrozenControllerIdentity(
        controller=identity.controller,
        directory=identity.directory,
        files=identity.files,
        aggregate_sha256="f" * 64,
    )
    with pytest.raises(FinalReportArtifactError, match="aggregate"):
        canonical_controller_run_manifest_json_bytes(
            publication.results[name],
            **{**evidence, "controller_identity": drifted},  # type: ignore[arg-type]
        )

    outputs = dict(evidence["output_artifacts"])
    outputs.pop("metrics")
    with pytest.raises(FinalReportArtifactError, match="six outputs"):
        canonical_controller_run_manifest_json_bytes(
            publication.results[name],
            **{**evidence, "output_artifacts": outputs},  # type: ignore[arg-type]
        )


def test_validate_m8_publication_attests_all_24_outputs_and_rejects_tamper(
    publication: SyntheticPublication,
) -> None:
    digest = validate_m8_publication(
        publication.outputs,
        publication.results,
        **publication.evidence,  # type: ignore[arg-type]
    )
    assert len(digest) == 64 and set(digest) <= set("0123456789abcdef")

    changed = dict(publication.outputs)
    path = controller_output_paths(publication.config, "ppo")["summary"]
    changed[path] += b"\n"
    with pytest.raises(Exception, match=r"canonical|recomputation|differs"):
        validate_m8_publication(
            changed,
            publication.results,
            **publication.evidence,  # type: ignore[arg-type]
        )
