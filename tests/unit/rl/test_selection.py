"""CPU-only tests for the frozen M7 Validation checkpoint-selection protocol."""

from __future__ import annotations

import copy
import hashlib
from dataclasses import asdict, replace
from pathlib import Path
from typing import Any

import pytest

from controller_learning.rl.selection import (
    FROZEN_CANDIDATE_UPDATES,
    FROZEN_WRAPPER_ORDER,
    SELECTION_REPORT_SCHEMA_VERSION,
    PolicySelectionResult,
    SelectionProtocolError,
    SelectionTrackResult,
    load_ppo_selection_config,
    rank_candidate_results,
    selection_gate_values,
    validate_selection_report,
)

PROJECT_ROOT = Path(__file__).parents[3]


def _rows(success_count: int, *, lap_time_s: float = 10.0) -> tuple[SelectionTrackResult, ...]:
    return tuple(
        SelectionTrackResult(
            track_index=index,
            track_id=1_000 + index,
            termination_reason=1 if index < success_count else 2,
            success=index < success_count,
            lap_time_s=lap_time_s if index < success_count else 0.0,
            max_progress=1.0 if index < success_count else 0.5,
            steps=round(lap_time_s / 0.05) if index < success_count else index + 1,
        )
        for index in range(100)
    )


def _candidate(
    update: int, success_count: int, *, lap_time_s: float = 10.0
) -> PolicySelectionResult:
    parameter_sha = f"{update // 10:x}" * 64
    return PolicySelectionResult(
        policy_kind="candidate",
        policy_id=f"checkpoint_update_{update:08d}",
        update_index=update,
        parameter_sha256_before=parameter_sha,
        parameter_sha256_after=parameter_sha,
        rows=_rows(success_count, lap_time_s=lap_time_s),
    )


def _baseline(success_count: int) -> PolicySelectionResult:
    return PolicySelectionResult(
        policy_kind="random_baseline",
        policy_id="random_seed_17",
        update_index=None,
        parameter_sha256_before=None,
        parameter_sha256_after=None,
        rows=_rows(success_count, lap_time_s=20.0),
    )


def _artifact(relative_path: str, digest: str = "a" * 64) -> dict[str, Any]:
    return {
        "relative_path": relative_path,
        "schema_version": 1,
        "sha256": digest,
        "size_bytes": 123,
    }


def _access(*, loaded: bool) -> dict[str, Any]:
    categories = ["official_validation_asset", "official_validation_manifest"] if loaded else []
    sequence = (
        [{"category": category, "flags": 0, "mode": "rb"} for category in categories]
        if loaded
        else []
    )
    return {
        "audit_hook_installed_before_preflight": True,
        "denied_event_count": 0,
        "open_event_counts": {category: 1 for category in categories},
        "open_event_sequence": sequence,
        "opened_path_categories": categories,
        "opened_splits": ["validation"] if loaded else [],
        "pre_validation_open_event_count": 0,
        "test_opened": False,
        "track_cache_opened": False,
        "train_opened": False,
        "validation_loaded": loaded,
        "validation_reads_enabled": loaded,
    }


def _valid_report() -> tuple[dict[str, Any], Any]:
    config = load_ppo_selection_config(PROJECT_ROOT / "configs/ppo_selection.toml")
    candidates = tuple(
        _candidate(update, 10 + update // 10, lap_time_s=20.0 - update / 10.0)
        for update in FROZEN_CANDIDATE_UPDATES
    )
    baseline = _baseline(2)
    ranked = rank_candidate_results(candidates)
    selected = ranked[0]
    gates = selection_gate_values(selected, baseline, config=config)
    revision = "1" * 40
    stable_hashes = {
        "latest_checkpoint_pointer": "a" * 64,
        "pixi_lock": "a" * 64,
        "selection_config": "a" * 64,
        "training_config": "a" * 64,
        "training_manifest": "a" * 64,
        "training_metrics": "a" * 64,
        "training_run_config": "a" * 64,
        "validation_asset": "a" * 64,
        "validation_manifest": "a" * 64,
        **{f"checkpoint_update_{update:08d}": "a" * 64 for update in FROZEN_CANDIDATE_UPDATES},
    }
    track_digest = hashlib.sha256()
    for track_id in range(1_000, 1_100):
        track_digest.update(str(track_id).encode("ascii"))
        track_digest.update(b"\n")
    configuration = asdict(config)
    configuration["candidate_updates"] = list(config.candidate_updates)
    memory_samples = [
        {
            "jax_bytes_in_use": 100 + index,
            "jax_peak_bytes_in_use": 200 + index,
            "phase": phase,
            "process_vram_error": None,
            "process_vram_mib": 600.0 + index,
            "synchronized": True,
            "torch_allocated_bytes": 300 + index,
            "torch_max_allocated_bytes": 400 + index,
            "torch_reserved_bytes": 500 + index,
        }
        for index, phase in enumerate(
            ("after_stack_build", "after_evaluations", "after_environment_close")
        )
    ]
    policy_ids = [candidate.policy_id for candidate in candidates] + [baseline.policy_id]
    report = {
        "artifacts": {
            "latest_checkpoint_pointer": _artifact("checkpoints/latest.json"),
            "metrics_csv": _artifact("metrics.csv"),
            "pixi_lock": _artifact("pixi.lock"),
            "selection_config": _artifact("configs/ppo_selection.toml"),
            "training_config": _artifact(config.training_config),
            "training_curve": _artifact(config.training_curve_path, "b" * 64),
            "training_manifest": _artifact(f"{config.run_directory}/manifest.json"),
            "training_run_config": _artifact(f"{config.run_directory}/config.toml"),
            "validation_asset": _artifact("controller_learning/assets/tracks/v0.1/validation.npz"),
            "validation_manifest": _artifact(
                "controller_learning/assets/tracks/v0.1/validation.json"
            ),
        },
        "asset_access": _access(loaded=True),
        "configuration": configuration,
        "evaluations": {
            "candidates": [candidate.to_dict() for candidate in candidates],
            "random_baseline": baseline.to_dict(),
        },
        "gates": gates,
        "memory": {
            "peak_jax_bytes_in_use": 202,
            "peak_sampled_process_vram_mib": 602.0,
            "peak_torch_allocated_bytes": 402,
            "sample_count": 3,
            "samples": memory_samples,
        },
        "post_selection": {
            "controller_evaluation_status": "not_run",
            "export_status": "not_run",
        },
        "protocol": {
            "autoreset_mode": "NEXT_STEP",
            "backend": config.backend,
            "benchmark_version": config.benchmark_version,
            "candidate_count": 8,
            "candidate_updates": list(FROZEN_CANDIDATE_UPDATES),
            "control_dt_s": 0.05,
            "deterministic_candidate_actions": True,
            "first_terminal_event_only": True,
            "level_id": 1,
            "max_vector_steps": 4000,
            "no_gradient_updates": True,
            "num_envs": 100,
            "one_long_lived_environment": True,
            "random_baseline_seed": 17,
            "reset_options_track_indices": "numpy.arange(100,dtype=int32)",
            "reward_wrapper_used": False,
            "same_reset_seed_and_track_order_for_every_policy": True,
            "test_accessed": False,
            "train_assets_accessed": False,
            "validation_reset_seed": 7,
            "validation_track_count": 100,
            "wrapper_order": list(FROZEN_WRAPPER_ORDER),
        },
        "runtime": {
            "cuda_device_order": "PCI_BUS_ID",
            "cuda_visible_devices_configured": False,
            "evaluation_timings": [
                {"elapsed_seconds": 1.0, "policy_id": policy_id} for policy_id in policy_ids
            ],
            "kernel": "test-kernel",
            "machine": "x86_64",
            "packages": {
                name: "test"
                for name in (
                    "controller-learning",
                    "jax",
                    "jaxlib",
                    "matplotlib",
                    "mujoco",
                    "mujoco-mjx",
                    "numpy",
                    "torch",
                    "warp-lang",
                )
            },
            "platform": "Linux",
            "python": "3.11.15",
            "selected_gpu": {
                "driver_version": "1.0",
                "index": 0,
                "memory_total_mib": 24_576.0,
                "name": "test GPU",
                "uuid": "redacted",
            },
            "torch_cuda_runtime": "12.8",
            "torch_device": "cuda:0",
            "xla_python_client_preallocate": "false",
        },
        "schema_version": SELECTION_REPORT_SCHEMA_VERSION,
        "selection": {
            "candidate_updates_in_rank_order": [result.update_index for result in ranked],
            "mean_successful_lap_time_no_success_policy": "positive_infinity_worst",
            "ranking": config.ranking,
            "selected_mean_successful_lap_time_s": selected.mean_successful_lap_time_s,
            "selected_success_count": selected.success_count,
            "selected_success_rate": selected.success_rate,
            "selected_update": selected.update_index,
        },
        "source": {
            "input_stability": {
                "all_inputs_unchanged": True,
                "expected_post_sha256": stable_hashes,
                "post_evaluation_sha256": dict(stable_hashes),
            },
            "post_input_check": {"revision": revision, "worktree_clean": True},
            "post_output_worktree": {
                "allowed_generated_output_paths": sorted(
                    [config.report_path, config.training_curve_path]
                ),
                "observed_changed_paths": [config.training_curve_path],
                "only_allowed_generated_outputs": True,
                "revision": revision,
                "unexpected_changed_paths": [],
            },
            "preflight": {"revision": revision, "worktree_clean": True},
        },
        "status": "passed" if gates["passed"] else "gate_failed",
        "training_run": {
            "candidate_checkpoints": [
                {
                    "checkpoint": _artifact(f"checkpoints/update_{update:08d}.pt"),
                    "inference_policy": {
                        "schema_version": 1,
                        "sha256": f"{update:064x}",
                        "size_bytes": 100_000 + update,
                    },
                    "parameter_sha256": candidate.parameter_sha256_before,
                    "update_index": update,
                    "valid_transitions": update * 1_000,
                    "vector_steps": update * 10,
                }
                for update, candidate in zip(FROZEN_CANDIDATE_UPDATES, candidates, strict=True)
            ],
            "identity": {
                "benchmark_version": "0.1",
                "configuration_sha256": "a" * 64,
                "environment_seed": 7,
                "feature_schema_version": 1,
                "lock_sha256": "a" * 64,
                "minibatch_seed": 13,
                "policy_seed": 11,
                "reward_schema_version": "controller-learning.m7-public-reward.v1",
                "run_id": config.run_id,
                "schema_version": 1,
                "source_revision": revision,
                "train_cache_sha256": "c" * 64,
                "train_manifest_sha256": "d" * 64,
            },
            "manifest_sha256": "a" * 64,
            "pre_validation_access": _access(loaded=False),
            "run_directory": config.run_directory,
        },
        "validation_assets": {
            "asset_file": "validation.npz",
            "asset_file_sha256": "a" * 64,
            "benchmark_version": "0.1",
            "capacity": {"max_checkpoints": 48, "max_track_points": 640},
            "first_track_id": 1_000,
            "generator_version": "test-v1",
            "geometry_hashes_sha256": "e" * 64,
            "last_track_id": 1_099,
            "level_id": 1,
            "loaded_splits": ["validation"],
            "loader_accessed_test": False,
            "loader_accessed_train": False,
            "manifest_asset_sha256": "a" * 64,
            "manifest_file": "validation.json",
            "manifest_sha256": "a" * 64,
            "schema_version": "controller-learning.m7-validation-pool-access.v1",
            "split": "validation",
            "track_count": 100,
            "track_ids_sha256": track_digest.hexdigest(),
        },
    }
    return report, config


def test_frozen_config_loads_and_rejects_protocol_mutations(tmp_path: Path) -> None:
    config = load_ppo_selection_config(PROJECT_ROOT / "configs/ppo_selection.toml")
    assert config.candidate_updates == FROZEN_CANDIDATE_UPDATES
    assert config.random_baseline_seed == 17
    assert config.minimum_success_rate_margin == 0.10

    with pytest.raises(SelectionProtocolError, match="candidate_updates"):
        replace(config, candidate_updates=(10, 20))
    with pytest.raises(SelectionProtocolError, match="minimum_success_rate_margin"):
        replace(config, minimum_success_rate_margin=0.09)
    with pytest.raises(SelectionProtocolError, match="report_path"):
        replace(config, report_path="benchmarks/selection.json")

    source = (PROJECT_ROOT / "configs/ppo_selection.toml").read_text(encoding="utf-8")
    unknown = tmp_path / "unknown.toml"
    unknown.write_text(source + "\nunknown_artifact = true\n", encoding="utf-8")
    with pytest.raises(SelectionProtocolError, match="artifacts keys differ"):
        load_ppo_selection_config(unknown)
    changed = tmp_path / "changed.toml"
    changed.write_text(
        source.replace(
            "candidate_updates = [10, 20, 30, 40, 50, 60, 70, 80]",
            "candidate_updates = [10, 20]",
        ),
        encoding="utf-8",
    )
    with pytest.raises(SelectionProtocolError, match="candidate_updates"):
        load_ppo_selection_config(changed)


def test_ranking_and_learning_gates_recompute_from_raw_rows() -> None:
    config = load_ppo_selection_config(PROJECT_ROOT / "configs/ppo_selection.toml")
    success_counts = {10: 15, 20: 15, 30: 14, 40: 14, 50: 13, 60: 12, 70: 0, 80: 0}
    lap_times = {10: 9.0, 20: 8.0, 30: 7.0, 40: 7.0, 50: 6.0, 60: 5.0, 70: 0.0, 80: 0.0}
    candidates = tuple(
        _candidate(update, success_counts[update], lap_time_s=lap_times[update])
        for update in FROZEN_CANDIDATE_UPDATES
    )
    ranked = rank_candidate_results(candidates)
    assert [result.update_index for result in ranked[:4]] == [20, 10, 30, 40]
    assert [result.update_index for result in ranked[-2:]] == [70, 80]

    gates = selection_gate_values(ranked[0], _baseline(5), config=config)
    assert gates["success_rate_margin"] == pytest.approx(0.10)
    assert gates["passed"] is True


def test_strict_report_validator_binds_raw_rows_checkpoints_and_runtime_evidence() -> None:
    report, config = _valid_report()
    validate_selection_report(report, config=config)

    mutations = []
    aggregate = copy.deepcopy(report)
    aggregate["evaluations"]["candidates"][0]["success_count"] += 1
    mutations.append(aggregate)
    checkpoint = copy.deepcopy(report)
    checkpoint["training_run"]["candidate_checkpoints"][0]["parameter_sha256"] = "f" * 64
    mutations.append(checkpoint)
    inference_policy = copy.deepcopy(report)
    inference_policy["training_run"]["candidate_checkpoints"][0]["inference_policy"][
        "schema_version"
    ] = True
    mutations.append(inference_policy)
    artifact_schema = copy.deepcopy(report)
    artifact_schema["artifacts"]["selection_config"]["schema_version"] = True
    mutations.append(artifact_schema)
    protocol = copy.deepcopy(report)
    protocol["protocol"]["reward_wrapper_used"] = True
    mutations.append(protocol)
    source = copy.deepcopy(report)
    source["source"]["input_stability"]["post_evaluation_sha256"]["training_config"] = "f" * 64
    mutations.append(source)
    runtime = copy.deepcopy(report)
    runtime["runtime"]["evaluation_timings"].pop()
    mutations.append(runtime)

    for mutated in mutations:
        with pytest.raises(SelectionProtocolError):
            validate_selection_report(mutated, config=config)
