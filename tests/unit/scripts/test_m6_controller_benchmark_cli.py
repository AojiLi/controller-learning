"""CPU-only protocol tests for the formal M6 PID/MPC benchmark CLI."""

from __future__ import annotations

import copy
import json
from dataclasses import asdict, replace
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from controller_learning.config import load_project_config
from controller_learning.evaluation import (
    ControllerEvaluation,
    EpisodeEvaluation,
    summarize_compute_times,
)
from scripts import benchmark_m6_controllers as benchmark

PROJECT_ROOT = Path(__file__).parents[3]


def _fake_runtime() -> dict[str, Any]:
    return {
        "python": "3.11.0",
        "platform": "Linux-test",
        "kernel": "test-kernel",
        "machine": "x86_64",
        "cpu": {"model": "test CPU", "logical_count": 8},
        "packages": {
            name: "test"
            for name in (
                "casadi",
                "controller-learning",
                "ipopt",
                "jax",
                "jax-cuda12-plugin",
                "jaxlib",
                "mujoco",
                "mujoco-mjx",
                "nvidia-cuda-nvcc-cu12",
                "nvidia-cuda-runtime-cu12",
                "numpy",
                "warp-lang",
            )
        },
        "casadi_ipopt_available": True,
        "jax_device": {"id": 0, "platform": "gpu", "device_kind": "test GPU"},
        "jax_gpu_error": None,
        "selected_nvidia_gpu": {
            "index": 0,
            "name": "test GPU",
            "driver_version": "1.0",
            "memory_total_mib": 16384.0,
        },
        "nvidia_smi_error": None,
        "gpu_selection_error": None,
        "xla_python_client_preallocate": "false",
        "cuda_device_order": "PCI_BUS_ID",
        "cuda_visible_devices_configured": False,
    }


def _fake_snapshot(_root: Path) -> dict[str, Any]:
    hashes = {relative: "a" * 64 for relative in benchmark.RELEVANT_SOURCE_PATHS}
    hashes[benchmark.M2_EVIDENCE_PATH.as_posix()] = benchmark.M2_EVIDENCE_SHA256
    hashes[benchmark.M5_EVIDENCE_PATH.as_posix()] = benchmark.M5_EVIDENCE_SHA256
    return {
        "git_revision": "0123456789abcdef",
        "relevant_source_clean": True,
        "source_files_sha256": hashes,
    }


def _fake_execution_evidence(
    evaluations: dict[str, Any],
    config,
) -> dict[str, Any]:
    groups: dict[str, dict[str, Any]] = {}
    instances: list[dict[str, Any]] = []
    for controller in ("pid", "mpc"):
        for split in ("level0", "validation"):
            label = f"{controller}.{split}"
            episodes = evaluations[controller][split]["episodes"]
            steps = sum(episode["steps"] for episode in episodes)
            wall_s = len(episodes) * 0.1
            groups[label] = {
                "episode_count": len(episodes),
                "environment_steps": steps,
                "wall_s": wall_s,
                "end_to_end_transitions_per_second": steps / wall_s,
            }
            instances.extend(
                {
                    "group": label,
                    "create_s": 0.001,
                    "reset_count": 1,
                    "reset_wall_s": 0.002,
                    "first_reset_s": 0.002,
                    "step_count": episode["steps"],
                    "step_wall_s": episode["steps"] * 0.004,
                    "first_step_s": 0.004,
                    "closed": True,
                }
                for episode in episodes
            )
    environment_steps = sum(instance["step_count"] for instance in instances)
    evaluation_wall_s = sum(group["wall_s"] for group in groups.values())
    step_wall_s = sum(instance["step_wall_s"] for instance in instances)
    memory_samples = [
        {
            "phase": phase,
            "process_vram_mib": 100.0 + index,
            "process_vram_error": None,
            "jax_allocator": {"peak_bytes_in_use": 1000 + index},
            "jax_allocator_error": None,
        }
        for index, phase in enumerate(benchmark.MEMORY_SAMPLE_PHASES)
    ]
    return {
        "controller_evaluation": {
            "execution_model": benchmark.CONTROLLER_EXECUTION_MODEL,
            "throughput_scope": benchmark.THROUGHPUT_SCOPE,
            "num_envs_per_environment": 1,
            "maximum_concurrent_worlds": 1,
            "environment_instances": len(instances),
            "episode_count": len(instances),
            "environment_steps": environment_steps,
            "transitions": environment_steps,
            "physics_substeps_per_environment_step": (
                config.vehicle.simulation.physics_steps_per_control
            ),
            "world_physics_steps": (
                environment_steps * config.vehicle.simulation.physics_steps_per_control
            ),
            "per_step_host_synchronization": True,
            "evaluation_wall_s": evaluation_wall_s,
            "end_to_end_transitions_per_second": environment_steps / evaluation_wall_s,
            "environment_step_call_wall_s": step_wall_s,
            "environment_step_call_transitions_per_second": environment_steps / step_wall_s,
            "groups": groups,
            "instances": instances,
        },
        "first_use_timing": {
            "method": benchmark.FIRST_USE_TIMING_METHOD,
            "first_environment_create_and_backend_initialization_s": 0.001,
            "first_reset_compile_and_execute_s": 0.002,
            "first_step_compile_and_execute_s": 0.004,
            "combined_first_create_reset_step_s": 0.007,
        },
        "memory": {
            "method": benchmark.MEMORY_SAMPLING_METHOD,
            "gpu_selection_error": None,
            "required_phases": list(benchmark.MEMORY_SAMPLE_PHASES),
            "sample_count": len(memory_samples),
            "samples": memory_samples,
            "peak_sampled_process_vram_mib": 108.0,
            "peak_jax_allocator_bytes": 1008.0,
        },
        "numerical": {
            "scope": [
                "all numeric public observation fields",
                "reward",
                "info.lap_time_s",
            ],
            "checked_transition_count": environment_steps,
            "failure_event_count": 0,
            "failure_field_counts": {},
            "invalid_action_count": 0,
            "internal_physics_diagnostics_claimed": False,
        },
    }


def _replace_compute_samples(report: dict[str, Any], value: float) -> None:
    for controller in ("pid", "mpc"):
        controller_result = report["evaluations"][controller]
        combined: list[float] = []
        for split in ("level0", "validation"):
            evaluation = controller_result[split]
            aggregate: list[float] = []
            for episode in evaluation["episodes"]:
                samples = [value] * episode["steps"]
                episode["compute_times_s"] = samples
                episode["compute_timing"] = asdict(summarize_compute_times(samples))
                aggregate.extend(samples)
            evaluation["compute_timing"] = asdict(summarize_compute_times(aggregate))
            combined.extend(aggregate)
        controller_result["combined_timing"] = asdict(summarize_compute_times(combined))
        controller_result["realtime_qualification"] = benchmark._realtime_qualification(
            controller_result["combined_timing"]
        )


def _evaluation(
    batch,
    directory: Path,
    level_id: int,
    reset_seeds: np.ndarray,
) -> ControllerEvaluation:
    track_count = int(batch.seed.shape[0])
    name = directory.name
    successes = track_count
    if name == "pid" and level_id == 1:
        successes = 0
    elif name == "mpc" and level_id == 1:
        successes = 80

    episodes = []
    for index in range(track_count):
        success = index < successes
        samples = (0.01,)
        episodes.append(
            EpisodeEvaluation(
                track_index=index,
                track_id=int(batch.seed[index]),
                reset_seed=int(reset_seeds[index]),
                success=success,
                lap_time_s=10.0 + index if success else None,
                steps=1,
                total_reward=1.0 if success else -1.0,
                terminated=success,
                truncated=not success,
                termination_reason=1 if success else 4,
                controller_import_time_s=0.001,
                controller_init_time_s=0.002,
                compute_times_s=samples,
                compute_timing=summarize_compute_times(samples),
            )
        )
    successful_laps = [episode.lap_time_s for episode in episodes if episode.success]
    return ControllerEvaluation(
        controller_directory=str(directory),
        level_id=level_id,
        backend="mjx_warp",
        episodes=tuple(episodes),
        track_count=track_count,
        success_count=successes,
        success_rate=successes / track_count,
        mean_successful_lap_time_s=(
            float(np.mean(successful_laps, dtype=np.float64)) if successful_laps else None
        ),
        compute_timing=summarize_compute_times(tuple(0.01 for _ in episodes)),
    )


@pytest.fixture(scope="module")
def official_assets():
    config = load_project_config(PROJECT_ROOT)
    return benchmark._load_evaluation_assets(config, PROJECT_ROOT)


@pytest.fixture()
def passing_report(official_assets):
    calls: list[dict[str, Any]] = []

    def evaluator(config, level_id, batch, generator_version, directory, backend, **kwargs):
        del config, generator_version
        calls.append(
            {
                "level_id": level_id,
                "track_count": int(batch.seed.shape[0]),
                "directory": Path(directory).name,
                "backend": backend,
                "reset_seeds": tuple(int(value) for value in kwargs["reset_seeds"]),
            }
        )
        return _evaluation(batch, Path(directory), level_id, kwargs["reset_seeds"])

    report = benchmark.run_benchmark(
        benchmark.BenchmarkOptions(),
        project_root=PROJECT_ROOT,
        asset_loader=lambda _config, _root: official_assets,
        evaluator=evaluator,
        snapshot_loader=_fake_snapshot,
        runtime_loader=_fake_runtime,
        execution_evidence_loader=_fake_execution_evidence,
    )
    return report, calls


def test_cli_exposes_only_the_report_path() -> None:
    assert benchmark._parse_args([]) == benchmark.BenchmarkOptions()
    assert benchmark._parse_args(["--output", "results/m6.json"]).output == Path("results/m6.json")

    with pytest.raises(SystemExit) as backend_error:
        benchmark._parse_args(["--backend", "cpu_reference"])
    assert backend_error.value.code == 2

    with pytest.raises(SystemExit) as suffix_error:
        benchmark._parse_args(["--output", "results/m6.txt"])
    assert suffix_error.value.code == 2


@pytest.mark.parametrize(
    ("snapshot_update", "message"),
    [
        ({"git_revision": None}, "non-empty Git revision"),
        ({"relevant_source_clean": False}, "clean relevant source"),
        ({"source_files_sha256": {}}, "does not cover every input"),
    ],
)
def test_source_preflight_rejects_before_expensive_work(
    snapshot_update: dict[str, Any],
    message: str,
) -> None:
    snapshot = _fake_snapshot(PROJECT_ROOT)
    snapshot.update(snapshot_update)
    calls: list[str] = []

    def forbidden_asset_loader(*_args):
        calls.append("assets")
        raise AssertionError("asset loading must not start after a failed source preflight")

    with pytest.raises(RuntimeError, match=message):
        benchmark.run_benchmark(
            benchmark.BenchmarkOptions(),
            project_root=PROJECT_ROOT,
            asset_loader=forbidden_asset_loader,
            snapshot_loader=lambda _root: snapshot,
            runtime_loader=_fake_runtime,
        )

    assert calls == []


def test_official_loader_reads_only_level0_and_validation(monkeypatch) -> None:
    config = load_project_config(PROJECT_ROOT)
    original = benchmark.load_manifest_track_batch
    accessed: list[str] = []

    def tracked(path):
        accessed.append(Path(path).name)
        return original(path)

    monkeypatch.setattr(benchmark, "load_manifest_track_batch", tracked)
    assets = benchmark._load_evaluation_assets(config, PROJECT_ROOT)

    assert accessed == ["level0.json", "validation.json"]
    assert assets.evidence["loaded_splits"] == ["level0", "validation"]
    assert assets.evidence["test_split_accessed"] is False
    assert "test" not in assets.evidence
    assert not any("test.json" in path for path in benchmark.RELEVANT_SOURCE_PATHS)
    assert not any("test.npz" in path for path in benchmark.RELEVANT_SOURCE_PATHS)


def test_formal_workload_and_row_index_seeds_are_fixed(passing_report) -> None:
    report, calls = passing_report

    assert calls == [
        {
            "level_id": 0,
            "track_count": 1,
            "directory": "pid",
            "backend": "mjx_warp",
            "reset_seeds": (0,),
        },
        {
            "level_id": 1,
            "track_count": 10,
            "directory": "pid",
            "backend": "mjx_warp",
            "reset_seeds": tuple(range(10)),
        },
        {
            "level_id": 0,
            "track_count": 1,
            "directory": "mpc",
            "backend": "mjx_warp",
            "reset_seeds": (0,),
        },
        {
            "level_id": 1,
            "track_count": 100,
            "directory": "mpc",
            "backend": "mjx_warp",
            "reset_seeds": tuple(range(100)),
        },
    ]
    assert report["status"] == "pass"
    assert all(check["passed"] for check in report["checks"])
    assert report["evaluations"]["pid"]["validation"]["success_rate"] == 0.0
    assert report["evaluations"]["mpc"]["validation"]["success_rate"] == 0.8
    execution = report["execution"]["controller_evaluation"]
    assert execution["num_envs_per_environment"] == 1
    assert execution["maximum_concurrent_worlds"] == 1
    assert execution["environment_instances"] == 112
    assert execution["episode_count"] == 112
    assert execution["environment_steps"] == 112
    assert execution["transitions"] == 112
    assert execution["world_physics_steps"] == 1120
    assert report["execution"]["memory"]["peak_sampled_process_vram_mib"] == 108.0
    assert report["execution"]["numerical"]["failure_event_count"] == 0


@pytest.mark.parametrize(
    ("mutation", "expected_gate"),
    [
        (
            lambda report: report["execution"]["controller_evaluation"].__setitem__(
                "transitions", 113
            ),
            "execution.counts_and_throughput",
        ),
        (
            lambda report: report["execution"]["first_use_timing"].__setitem__(
                "first_reset_compile_and_execute_s", 1.0
            ),
            "execution.first_use_timing",
        ),
        (
            lambda report: report["execution"]["memory"]["samples"].pop(),
            "execution.memory",
        ),
        (
            lambda report: report["execution"]["numerical"].__setitem__("failure_event_count", 1),
            "execution.public_numerical_health",
        ),
        (
            lambda report: report["runtime"]["packages"].__setitem__("ipopt", None),
            "runtime.hardware_software_versions",
        ),
        (
            lambda report: report["historical_gpu_evidence"]["m2_physics"].__setitem__(
                "sha256", "0" * 64
            ),
            "evidence.historical_gpu_reports",
        ),
        (
            lambda report: report["historical_gpu_evidence"]["m2_physics"].__setitem__(
                "compilation_s", 99.0
            ),
            "evidence.historical_gpu_reports",
        ),
    ],
)
def test_gpu_execution_evidence_mutation_gates(passing_report, mutation, expected_gate) -> None:
    report, _calls = passing_report
    failing = copy.deepcopy(report)
    mutation(failing)

    failed = {
        check["id"] for check in benchmark.evaluate_report_gates(failing) if not check["passed"]
    }

    assert failed == {expected_gate}


def test_historical_gpu_gate_cross_checks_source_snapshot_hashes(passing_report) -> None:
    report, _calls = passing_report
    failing = copy.deepcopy(report)
    path = benchmark.M5_EVIDENCE_PATH.as_posix()
    for phase in ("before", "after"):
        failing["source_evidence"][phase]["source_files_sha256"][path] = "0" * 64

    failed = {
        check["id"] for check in benchmark.evaluate_report_gates(failing) if not check["passed"]
    }

    assert failed == {"evidence.historical_gpu_reports"}


def test_gpu_execution_gates_have_no_performance_threshold(passing_report) -> None:
    report, _calls = passing_report
    slow = copy.deepcopy(report)
    evaluation = slow["execution"]["controller_evaluation"]
    for group in evaluation["groups"].values():
        group["wall_s"] *= 1_000_000.0
        group["end_to_end_transitions_per_second"] = group["environment_steps"] / group["wall_s"]
    evaluation["evaluation_wall_s"] = sum(
        group["wall_s"] for group in evaluation["groups"].values()
    )
    evaluation["end_to_end_transitions_per_second"] = (
        evaluation["environment_steps"] / evaluation["evaluation_wall_s"]
    )
    for instance in evaluation["instances"]:
        instance["step_wall_s"] *= 1_000_000.0
    evaluation["environment_step_call_wall_s"] = sum(
        instance["step_wall_s"] for instance in evaluation["instances"]
    )
    evaluation["environment_step_call_transitions_per_second"] = (
        evaluation["environment_steps"] / evaluation["environment_step_call_wall_s"]
    )

    assert all(check["passed"] for check in benchmark.evaluate_report_gates(slow))


def test_execution_recorder_measures_calls_and_public_numerical_health() -> None:
    class FakeEnvironment:
        unwrapped = None

        def reset(self, **_kwargs):
            return {"position": np.zeros(2)}, {"lap_time_s": 0.0}

        def step(self, action):
            position = np.asarray(action, dtype=np.float64)
            return (
                {"position": position},
                0.0,
                False,
                False,
                {"lap_time_s": 0.0},
            )

        def render(self):
            return None

        def close(self):
            return None

    memory_index = 0

    def memory_sampler(_device, phase, _gpu_uuid):
        nonlocal memory_index
        memory_index += 1
        return {
            "phase": phase,
            "process_vram_mib": 100.0 + memory_index,
            "process_vram_error": None,
            "jax_allocator": {"peak_bytes_in_use": memory_index},
            "jax_allocator_error": None,
        }

    progress_events: list[dict[str, Any]] = []
    recorder = benchmark._ExecutionRecorder(
        device=object(),
        gpu_uuid="private-test-id",
        gpu_selection_error=None,
        environment_factory=lambda **_kwargs: FakeEnvironment(),
        memory_sampler=memory_sampler,
        progress_sink=lambda payload: progress_events.append(dict(payload)),
    )
    recorder.begin_group("pid.level0")
    env = recorder.create_environment()
    env.reset(seed=0)
    env.step([0.0, 0.0])
    env.step([float("nan"), 0.0])
    env.close()

    record = recorder.instances[0]
    assert record.reset_count == 1
    assert record.step_count == 2
    assert record.create_s > 0.0
    assert record.first_reset_s is not None and record.first_reset_s > 0.0
    assert record.first_step_s is not None and record.first_step_s > 0.0
    assert record.closed is True
    assert recorder.checked_transitions == 2
    assert recorder.numerical_failure_count == 1
    assert recorder.numerical_fields == {"observation.position": 1}
    assert progress_events == [
        {
            "event": "m6_environment_closed",
            "group": "pid.level0",
            "group_episode_completed": 1,
            "group_episode_total": 1,
            "environment_steps": 2,
        }
    ]
    assert [sample["phase"] for sample in recorder.memory_samples] == [
        "before_evaluation",
        "after_first_environment_create",
        "after_first_reset",
        "after_first_step",
    ]


def test_default_execution_path_builds_evidence_from_instrumented_environment(
    official_assets,
    monkeypatch,
) -> None:
    class FakeEnvironment:
        unwrapped = None

        def reset(self, **_kwargs):
            return {"position": np.zeros(2)}, {"lap_time_s": 0.0}

        def step(self, _action):
            return (
                {"position": np.zeros(2)},
                0.0,
                False,
                False,
                {"lap_time_s": 0.0},
            )

        def close(self):
            return None

    sample_index = 0

    def memory_sampler(_device, phase, _gpu_uuid):
        nonlocal sample_index
        sample_index += 1
        return {
            "phase": phase,
            "process_vram_mib": 200.0 + sample_index,
            "process_vram_error": None,
            "jax_allocator": {"peak_bytes_in_use": 2000 + sample_index},
            "jax_allocator_error": None,
        }

    recorder = benchmark._ExecutionRecorder(
        device=object(),
        gpu_uuid="private-test-id",
        gpu_selection_error=None,
        environment_factory=lambda **_kwargs: FakeEnvironment(),
        memory_sampler=memory_sampler,
    )
    monkeypatch.setattr(benchmark, "_formal_execution_recorder", lambda: recorder)

    def evaluator(config, level_id, batch, generator_version, directory, backend, **kwargs):
        del config, generator_version, backend
        for _ in range(int(batch.seed.shape[0])):
            env = kwargs["env_factory"]()
            env.reset(seed=0)
            env.step([0.0, 0.0])
            env.close()
        return _evaluation(batch, Path(directory), level_id, kwargs["reset_seeds"])

    report = benchmark.run_benchmark(
        benchmark.BenchmarkOptions(),
        project_root=PROJECT_ROOT,
        asset_loader=lambda _config, _root: official_assets,
        evaluator=evaluator,
        snapshot_loader=_fake_snapshot,
        runtime_loader=_fake_runtime,
    )

    assert report["status"] == "pass"
    assert report["execution"]["controller_evaluation"]["environment_instances"] == 112
    assert report["execution"]["numerical"]["checked_transition_count"] == 112
    assert [sample["phase"] for sample in report["execution"]["memory"]["samples"]] == list(
        benchmark.MEMORY_SAMPLE_PHASES
    )


def test_historical_gpu_evidence_is_bound_to_reviewed_reports() -> None:
    evidence = benchmark._historical_gpu_evidence(PROJECT_ROOT)

    assert evidence["m2_physics"]["sha256"] == benchmark.M2_EVIDENCE_SHA256
    assert evidence["m2_physics"]["num_worlds"] == 1024
    assert evidence["m2_physics"]["numerical_failure_count"] == 0
    assert evidence["m5_vector_environment"]["sha256"] == benchmark.M5_EVIDENCE_SHA256
    assert evidence["m5_vector_environment"]["transitions"] == 10_240_000
    assert evidence["m5_vector_environment"]["numerical_failure_count"] == 0


def test_report_gates_recompute_and_realtime_is_diagnostic(passing_report) -> None:
    report, _calls = passing_report

    assert benchmark.evaluate_report_gates(report) == report["checks"]
    _replace_compute_samples(report, 0.06)

    required_checks = benchmark.evaluate_report_gates(report)

    assert all(check["passed"] for check in required_checks)
    assert report["evaluations"]["pid"]["realtime_qualification"]["eligible"] is False
    assert report["evaluations"]["mpc"]["realtime_qualification"]["eligible"] is False


def test_mpc_validation_below_eighty_percent_fails_exact_threshold_gate(
    passing_report,
) -> None:
    report, _calls = passing_report
    failing = copy.deepcopy(report)
    validation = failing["evaluations"]["mpc"]["validation"]
    episode = validation["episodes"][79]
    episode["success"] = False
    episode["lap_time_s"] = None
    episode["terminated"] = False
    episode["truncated"] = True
    episode["termination_reason"] = 4
    validation["success_count"] = 79
    validation["success_rate"] = 0.79
    validation["mean_successful_lap_time_s"] = float(
        np.mean([item["lap_time_s"] for item in validation["episodes"] if item["success"]])
    )

    failed = {
        check["id"] for check in benchmark.evaluate_report_gates(failing) if not check["passed"]
    }

    assert failed == {"controllers.mpc.validation_success_rate"}


def test_gate_rejects_invalid_action_and_incomplete_timing(passing_report) -> None:
    report, _calls = passing_report
    failing = copy.deepcopy(report)
    episode = failing["evaluations"]["pid"]["validation"]["episodes"][0]
    episode["termination_reason"] = 3
    episode["compute_times_s"] = []

    failed = {
        check["id"] for check in benchmark.evaluate_report_gates(failing) if not check["passed"]
    }

    assert failed == {"controllers.no_invalid_action", "controllers.timing_consistency"}


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("backend", "cpu_reference"),
        ("level_id", 9),
        ("controller_directory", "controllers/pid"),
    ],
)
def test_gate_rejects_evaluation_identity_mutations(passing_report, field, value) -> None:
    report, _calls = passing_report
    failing = copy.deepcopy(report)
    failing["evaluations"]["mpc"]["validation"][field] = value

    failed = {
        check["id"] for check in benchmark.evaluate_report_gates(failing) if not check["passed"]
    }

    assert failed == {"protocol.evaluation_identity"}


def test_runtime_rejects_evaluator_identity_mismatch(official_assets) -> None:
    def evaluator(config, level_id, batch, generator_version, directory, backend, **kwargs):
        del config, generator_version, backend
        result = _evaluation(batch, Path(directory), level_id, kwargs["reset_seeds"])
        return replace(result, backend="cpu_reference")

    with pytest.raises(ValueError, match="non-formal backend"):
        benchmark.run_benchmark(
            benchmark.BenchmarkOptions(),
            project_root=PROJECT_ROOT,
            asset_loader=lambda _config, _root: official_assets,
            evaluator=evaluator,
            snapshot_loader=_fake_snapshot,
            runtime_loader=_fake_runtime,
            execution_evidence_loader=_fake_execution_evidence,
        )


def test_gate_recomputes_success_and_lap_aggregates(passing_report) -> None:
    report, _calls = passing_report
    failing = copy.deepcopy(report)
    episode = failing["evaluations"]["mpc"]["validation"]["episodes"][0]
    episode["success"] = False
    episode["lap_time_s"] = None
    episode["terminated"] = False
    episode["truncated"] = True
    episode["termination_reason"] = 4

    failed = {
        check["id"] for check in benchmark.evaluate_report_gates(failing) if not check["passed"]
    }

    assert failed == {"controllers.aggregate_consistency"}


def test_gate_recomputes_timing_summaries_and_realtime_qualification(passing_report) -> None:
    report, _calls = passing_report
    bad_timing = copy.deepcopy(report)
    bad_timing["evaluations"]["pid"]["level0"]["episodes"][0]["compute_timing"]["p99_s"] = 0.02

    failed_timing = {
        check["id"] for check in benchmark.evaluate_report_gates(bad_timing) if not check["passed"]
    }
    assert failed_timing == {"controllers.timing_consistency"}

    bad_realtime = copy.deepcopy(report)
    bad_realtime["evaluations"]["mpc"]["realtime_qualification"]["eligible"] = False
    failed_realtime = {
        check["id"]
        for check in benchmark.evaluate_report_gates(bad_realtime)
        if not check["passed"]
    }
    assert failed_realtime == {"controllers.realtime_qualification_consistency"}


def test_gate_rejects_controller_init_over_thirty_seconds(passing_report) -> None:
    report, _calls = passing_report
    failing = copy.deepcopy(report)
    failing["evaluations"]["pid"]["validation"]["episodes"][0]["controller_init_time_s"] = 30.000001

    failed = {
        check["id"] for check in benchmark.evaluate_report_gates(failing) if not check["passed"]
    }

    assert failed == {"controllers.init_timeout"}


def test_strict_json_is_atomic_and_report_is_private(
    passing_report,
    tmp_path: Path,
) -> None:
    report, _calls = passing_report
    output = tmp_path / "report.json"

    benchmark.write_strict_json(output, report)
    assert json.loads(output.read_text(encoding="utf-8"))["status"] == "pass"
    assert benchmark._privacy_findings(report) == {"absolute_paths": [], "gpu_uuids": []}

    bad_output = tmp_path / "bad.json"
    with pytest.raises(ValueError, match="Out of range float values"):
        benchmark.write_strict_json(bad_output, {"bad": float("nan")})
    assert not bad_output.exists()


def test_privacy_gate_rejects_absolute_paths_and_gpu_uuids(passing_report) -> None:
    report, _calls = passing_report
    failing = copy.deepcopy(report)
    failing["runtime"]["leak"] = [
        "RuntimeError: cache at /home/user/controller_learning/cache.bin",
        r"failure under C:\Users\name\controller-learning\cache.bin",
        "GPU-12345678-1234-1234-1234-123456789abc",
    ]

    failed = {
        check["id"] for check in benchmark.evaluate_report_gates(failing) if not check["passed"]
    }

    assert failed == {"report.privacy"}
    findings = benchmark._privacy_findings(failing)
    assert "/home/user/controller_learning/cache.bin" in findings["absolute_paths"]
    assert r"C:\Users\name\controller-learning\cache.bin" in findings["absolute_paths"]


def test_main_writes_the_report_and_exits_nonzero_after_a_failed_gate(
    passing_report,
    monkeypatch,
    tmp_path: Path,
) -> None:
    report, _calls = passing_report
    passing_output = tmp_path / "passing.json"
    monkeypatch.setattr(benchmark, "run_benchmark", lambda _options: report)

    benchmark.main(["--output", str(passing_output)])

    assert json.loads(passing_output.read_text(encoding="utf-8"))["status"] == "pass"

    failing = copy.deepcopy(report)
    failing["status"] = "fail"
    failing["checks"][0]["passed"] = False
    failing_output = tmp_path / "failing.json"
    monkeypatch.setattr(benchmark, "run_benchmark", lambda _options: failing)

    with pytest.raises(SystemExit, match="failed one or more gates"):
        benchmark.main(["--output", str(failing_output)])
    assert json.loads(failing_output.read_text(encoding="utf-8"))["status"] == "fail"


def test_relevant_sources_are_unique_relative_and_present() -> None:
    assert len(benchmark.RELEVANT_SOURCE_PATHS) == len(set(benchmark.RELEVANT_SOURCE_PATHS))
    for relative in benchmark.RELEVANT_SOURCE_PATHS:
        assert not Path(relative).is_absolute()
        assert (PROJECT_ROOT / relative).is_file(), relative

    package_sources = {
        path.relative_to(PROJECT_ROOT).as_posix()
        for path in (PROJECT_ROOT / "controller_learning").rglob("*.py")
    }
    assert package_sources.issubset(benchmark.RELEVANT_SOURCE_PATHS)
    assert {
        "benchmarks/v0.1/gpu_report.json",
        "benchmarks/v0.1/m5_track_pool_report.json",
        "controller_learning/__init__.py",
        "controller_learning/config/__init__.py",
        "controller_learning/control/__init__.py",
        "controller_learning/control/debug_draw.py",
        "controller_learning/evaluation/__init__.py",
        "controller_learning/physics/__init__.py",
        "controller_learning/tracks/hashing.py",
        "controller_learning/tracks/pool.py",
        "controller_learning/tracks/specs.py",
        "scripts/benchmark_racing_env.py",
    }.issubset(benchmark.RELEVANT_SOURCE_PATHS)
