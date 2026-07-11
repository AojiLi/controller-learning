"""CPU tests for the frozen post-selection PPO Controller protocol."""

from __future__ import annotations

import copy
from dataclasses import asdict, replace
from pathlib import Path

import pytest

from controller_learning.evaluation import EpisodeEvaluation, summarize_compute_times
from controller_learning.rl.controller_benchmark import (
    CONTROLLER_EVALUATION_REPORT_SCHEMA_VERSION,
    FORMAL_OUTPUT_CRASH_RECOVERY_METHOD,
    FORMAL_REPLAY_CAPTURE_METHOD,
    ControllerBenchmarkProtocolError,
    episode_to_report_row,
    evaluation_summary,
    load_ppo_controller_evaluation_config,
    replay_track_index,
    validate_controller_evaluation_report,
)

PROJECT_ROOT = Path(__file__).parents[3]
CONFIG_PATH = PROJECT_ROOT / "configs/ppo_controller_evaluation.toml"
_DIGEST = "a" * 64


def _episode(index: int) -> EpisodeEvaluation:
    success = index in {2, 8}
    samples = (0.001 + index * 1.0e-7,)
    return EpisodeEvaluation(
        track_index=index,
        track_id=1_000_000 + index,
        reset_seed=index,
        success=success,
        lap_time_s=0.05 if success else None,
        steps=1,
        total_reward=1.0 if success else -1.0,
        terminated=True,
        truncated=False,
        termination_reason=1 if success else 2,
        controller_import_time_s=0.002,
        controller_init_time_s=0.003,
        compute_times_s=samples,
        compute_timing=summarize_compute_times(samples),
    )


def _artifact(path: str, *, digest: str = _DIGEST, size: int = 10) -> dict[str, object]:
    return {
        "relative_path": path,
        "schema_version": 1,
        "sha256": digest,
        "size_bytes": size,
    }


def _track_ids_sha256(episodes: list[EpisodeEvaluation]) -> str:
    import hashlib

    digest = hashlib.sha256()
    for episode in episodes:
        digest.update(str(episode.track_id).encode("ascii"))
        digest.update(b"\n")
    return digest.hexdigest()


def _report() -> tuple[object, dict[str, object]]:
    from controller_learning.envs.episode import initialize_episode_identities

    config = load_ppo_controller_evaluation_config(CONFIG_PATH)
    episodes = [_episode(index) for index in range(100)]
    identities = [initialize_episode_identities(index, 1) for index in range(100)]
    rows = [
        episode_to_report_row(
            episode,
            episode_seed=int(identity.episode_seed[0]),
            controller_seed=int(identity.controller_seed[0]),
            benchmark_version="0.1",
        )
        for episode, identity in zip(episodes, identities, strict=True)
    ]
    summary = evaluation_summary(episodes)
    policy_digest = "b" * 64
    metadata_digest = "c" * 64
    config_digest = "d" * 64
    artifacts = {
        "controller_config": _artifact("controllers/ppo/config.toml", digest=config_digest),
        "controller_metadata": _artifact("controllers/ppo/metadata.json", digest=metadata_digest),
        "controller_policy": _artifact(
            "controllers/ppo/policy.npz", digest=policy_digest, size=1234
        ),
        "controller_source": _artifact("controllers/ppo/controller.py"),
        "evaluation_config": _artifact("configs/ppo_controller_evaluation.toml"),
        "export_report": _artifact(config.export_report),
        "pixi_lock": _artifact("pixi.lock"),
        "selection_config": _artifact(config.selection_config),
        "selection_report": _artifact(config.selection_report),
        "training_config": _artifact("configs/ppo.toml"),
        "validation_asset": _artifact("controller_learning/assets/tracks/v0.1/validation.npz"),
        "validation_manifest": _artifact("controller_learning/assets/tracks/v0.1/validation.json"),
    }
    memory_samples = [
        {
            "jax_bytes_in_use": 100 + index,
            "jax_peak_bytes_in_use": 200 + index,
            "phase": phase,
            "process_vram_error": None,
            "process_vram_mib": 300.0 + index,
        }
        for index, phase in enumerate(
            (
                "before_environment_create",
                "after_controller_evaluation",
                "after_replay_capture_validation",
                "after_artifact_render",
            )
        )
    ]
    input_hashes = {name: str(record["sha256"]) for name, record in artifacts.items()}
    checkpoint = {
        "checkpoint_sha256": _DIGEST,
        "run_id": "m7-formal-v0-1-001",
        "source_revision": "0" * 40,
        "training_configuration_sha256": _DIGEST,
        "update_index": 80,
        "valid_transitions": 10_000,
        "vector_steps": 10_000,
    }
    report: dict[str, object] = {
        "artifacts": artifacts,
        "asset_access": {
            "audit_hook_installed_before_preflight": True,
            "denied_event_count": 0,
            "denied_mutation_event_count": 0,
            "denied_mutation_event_types": {},
            "open_event_counts": {
                "official_validation_asset": 1,
                "official_validation_manifest": 1,
            },
            "open_event_sequence": [
                {
                    "category": "official_validation_manifest",
                    "flags": 0,
                    "mode": "r",
                },
                {"category": "official_validation_asset", "flags": 0, "mode": "r"},
            ],
            "opened_path_categories": [
                "official_validation_asset",
                "official_validation_manifest",
            ],
            "opened_splits": ["validation"],
            "pre_validation_open_event_count": 0,
            "test_opened": False,
            "track_cache_opened": False,
            "train_opened": False,
            "validation_loaded": True,
            "validation_reads_enabled": True,
        },
        "configuration": config.to_dict(),
        "controller": {
            "checkpoint": checkpoint,
            "config_sha256": config_digest,
            "directory": config.controller_directory,
            "finalized": True,
            "fresh_instance_count": 100,
            "inference_runtime": "numpy",
            "metadata_sha256": metadata_digest,
            "name": "ppo",
            "policy_schema_version": 1,
            "policy_sha256": policy_digest,
            "policy_size_bytes": 1234,
            "torch_imported": False,
        },
        "evaluation": {"episodes": rows, "summary": summary},
        "execution": {
            "captured_replay_episode_wall_s": 0.5,
            "captured_replay_steps": 1,
            "environment_instances": 1,
            "environment_steps": 100,
            "evaluation_wall_s": 10.0,
            "first_use_timing": {
                "first_environment_create_s": 0.1,
                "first_reset_s": 0.2,
                "first_step_s": 0.3,
                "method": "measured around real calls",
            },
            "physics_substeps": 1000,
            "recorded_episode_count": 3,
            "transitions_per_second": 10.0,
        },
        "memory": {
            "peak_jax_allocator_bytes": 203,
            "peak_sampled_process_vram_mib": 303.0,
            "sample_count": 4,
            "samples": memory_samples,
        },
        "export": {
            "controller_artifacts": {
                "config": dict(artifacts["controller_config"]),
                "metadata": dict(artifacts["controller_metadata"]),
                "policy": dict(artifacts["controller_policy"]),
            },
            "controller_checkpoint": dict(checkpoint),
            "report_schema_version": "controller-learning.m7-ppo-controller-export.v1",
            "report_status": "passed",
            "selected_candidate": {
                "checkpoint": _artifact(
                    "checkpoints/update_00000080.pt",
                    digest=_DIGEST,
                ),
                "inference_policy": {
                    "schema_version": 1,
                    "sha256": policy_digest,
                    "size_bytes": 1234,
                },
                "parameter_sha256": "e" * 64,
                "update_index": 80,
                "valid_transitions": 10_000,
                "vector_steps": 10_000,
            },
        },
        "protocol": {
            "backend": "mjx_warp",
            "benchmark_version": "0.1",
            "controller_execution_model": (
                "one reusable batch-one MJX-Warp environment with one fresh ordinary "
                "Controller per episode"
            ),
            "environment_instances": 1,
            "fresh_controller_per_episode": True,
            "level_id": 1,
            "max_episode_steps": 4000,
            "no_gradient_updates": True,
            "ordinary_controller_plugin": True,
            "output_crash_recovery_method": FORMAL_OUTPUT_CRASH_RECOVERY_METHOD,
            "replay_capture_method": FORMAL_REPLAY_CAPTURE_METHOD,
            "replay_environment_instances": 0,
            "replay_selection_rule": ("first_successful_track_in_fixed_order_else_first_track"),
            "reset_seed_rule": "validation_row_index_uint32",
            "test_accessed": False,
            "train_assets_accessed": False,
            "validation_track_count": 100,
        },
        "replay": {
            "captured_from_evaluation_row": True,
            "overview": {
                "all_source_frames_rendered": True,
                "artifact": _artifact(config.overview_path),
                "rendered_frame_count": 2,
                "source_frame_count": 2,
            },
            "reset_seed": 2,
            "selection_rule": config.replay_selection_rule,
            "track_id": episodes[2].track_id,
            "track_index": 2,
            "trajectory": {
                "artifact": _artifact(config.trajectory_path),
                "final_lap_completed": True,
                "final_termination_reason": 1,
                "frame_count": 2,
                "schema_version": "controller-learning-trajectory-v1",
                "step_count": 1,
            },
        },
        "runtime": {
            "cuda_device_order": "PCI_BUS_ID",
            "cuda_visible_devices_configured": False,
            "jax_device": {"device_kind": "fake GPU", "id": 0, "platform": "gpu"},
            "kernel": "test",
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
            "python": "3.11.0",
            "selected_gpu": {
                "driver_version": "test",
                "index": 0,
                "memory_total_mib": 16_384.0,
                "name": "fake GPU",
                "uuid": "redacted",
            },
            "xla_python_client_preallocate": "false",
        },
        "schema_version": CONTROLLER_EVALUATION_REPORT_SCHEMA_VERSION,
        "selection": {
            "gate_passed": True,
            "report_status": "passed",
            "selected_checkpoint_sha256": _DIGEST,
            "selected_inference_policy_schema_version": 1,
            "selected_inference_policy_sha256": policy_digest,
            "selected_inference_policy_size_bytes": 1234,
            "selected_success_count": 20,
            "selected_success_rate": 0.2,
            "selected_update": 80,
            "training_configuration_sha256": _DIGEST,
        },
        "source": {
            "input_sha256_after": dict(input_hashes),
            "input_sha256_before": dict(input_hashes),
            "post_output_worktree": {
                "allowed_payload_output_paths": sorted(
                    [config.trajectory_path, config.overview_path]
                ),
                "observed_payload_changed_paths": sorted(
                    [config.trajectory_path, config.overview_path]
                ),
                "only_allowed_payload_outputs_before_report_write": True,
                "published_output_bytes_verified": True,
                "report_change_excluded_from_payload_observation": True,
                "report_output_path": config.report_path,
                "revision": "0" * 40,
                "unexpected_changed_paths": [],
            },
            "preflight": {"revision": "0" * 40, "worktree_clean": True},
        },
        "status": "passed",
        "validation_assets": {
            "asset_file": "validation.npz",
            "asset_file_sha256": _DIGEST,
            "benchmark_version": "0.1",
            "capacity": {"max_checkpoints": 48, "max_track_points": 640},
            "first_track_id": episodes[0].track_id,
            "generator_version": "v0.1",
            "geometry_hashes_sha256": _DIGEST,
            "last_track_id": episodes[-1].track_id,
            "level_id": 1,
            "loaded_splits": ["validation"],
            "loader_accessed_test": False,
            "loader_accessed_train": False,
            "manifest_asset_sha256": _DIGEST,
            "manifest_file": "validation.json",
            "manifest_sha256": _DIGEST,
            "schema_version": "controller-learning.m7-validation-pool-access.v1",
            "split": "validation",
            "track_count": 100,
            "track_ids_sha256": _track_ids_sha256(episodes),
        },
    }
    return config, report


def test_frozen_config_loads_and_rejects_type_aliases_and_unknown_keys(tmp_path: Path) -> None:
    config = load_ppo_controller_evaluation_config(CONFIG_PATH)
    assert config.validation_track_count == 100
    assert config.max_episode_steps == 4000
    assert config.schema_version == 2
    assert config.replay_selection_rule.endswith("else_first_track")
    assert config.replay_capture_method == FORMAL_REPLAY_CAPTURE_METHOD

    with pytest.raises(ControllerBenchmarkProtocolError, match="schema_version"):
        replace(config, schema_version=True)
    with pytest.raises(ControllerBenchmarkProtocolError, match="replay_capture_method"):
        replace(config, replay_capture_method="rerun_selected_episode")

    mutated = CONFIG_PATH.read_text(encoding="utf-8") + "\nunknown = 1\n"
    path = tmp_path / "evaluation.toml"
    path.write_text(mutated, encoding="utf-8")
    with pytest.raises(ControllerBenchmarkProtocolError, match="keys differ"):
        load_ppo_controller_evaluation_config(path)


def test_checked_in_formal_report_and_replay_are_strictly_bound() -> None:
    from controller_learning.evaluation import load_trajectory_json
    from controller_learning.rl.artifacts import read_strict_json, sha256_file

    config = load_ppo_controller_evaluation_config(CONFIG_PATH)
    report = read_strict_json(PROJECT_ROOT, config.report_path)
    validate_controller_evaluation_report(report, config=config)

    assert report["status"] == "passed"
    assert report["source"]["preflight"]["revision"] == ("1b434f4128043001722e33899a0a767b8e5cdba7")
    assert report["evaluation"]["summary"]["success_count"] == 99
    assert report["controller"]["fresh_instance_count"] == 100
    assert report["protocol"]["replay_environment_instances"] == 0

    trajectory_record = report["replay"]["trajectory"]["artifact"]
    trajectory = load_trajectory_json(
        PROJECT_ROOT / trajectory_record["relative_path"],
        expected_sha256=trajectory_record["sha256"],
    )
    assert trajectory.step_count == report["replay"]["trajectory"]["step_count"]

    overview_record = report["replay"]["overview"]["artifact"]
    overview_path = PROJECT_ROOT / overview_record["relative_path"]
    assert overview_path.stat().st_size == overview_record["size_bytes"]
    assert sha256_file(overview_path) == overview_record["sha256"]


def test_replay_rule_is_first_success_else_first_row() -> None:
    rows = [{"track_index": index, "success": index == 7} for index in range(10)]
    assert replay_track_index(rows) == 7
    assert (
        replay_track_index([{"track_index": index, "success": False} for index in range(10)]) == 0
    )


def test_strict_report_recomputes_all_rows_timing_and_identity_links() -> None:
    config, report = _report()
    validate_controller_evaluation_report(report, config=config)
    clean_rerun = copy.deepcopy(report)
    clean_rerun["source"]["post_output_worktree"]["observed_payload_changed_paths"] = []
    validate_controller_evaluation_report(clean_rerun, config=config)

    mutations = []
    changed = copy.deepcopy(report)
    changed["evaluation"]["summary"]["success_count"] += 1
    mutations.append(changed)
    changed = copy.deepcopy(report)
    changed["evaluation"]["episodes"][4]["reset_seed"] = 99
    mutations.append(changed)
    changed = copy.deepcopy(report)
    changed["replay"]["track_index"] = 0
    mutations.append(changed)
    changed = copy.deepcopy(report)
    changed["controller"]["policy_sha256"] = "e" * 64
    mutations.append(changed)
    changed = copy.deepcopy(report)
    changed["source"]["input_sha256_after"]["pixi_lock"] = "f" * 64
    mutations.append(changed)
    changed = copy.deepcopy(report)
    changed["runtime"]["platform"] = "Darwin"
    mutations.append(changed)
    changed = copy.deepcopy(report)
    changed["runtime"]["packages"]["jax"] = None
    mutations.append(changed)
    changed = copy.deepcopy(report)
    changed["runtime"]["selected_gpu"]["uuid"] = ""
    mutations.append(changed)
    changed = copy.deepcopy(report)
    changed["runtime"]["jax_device"]["platform"] = "cpu"
    mutations.append(changed)
    changed = copy.deepcopy(report)
    changed["memory"]["samples"][0]["process_vram_mib"] = 0.0
    mutations.append(changed)
    changed = copy.deepcopy(report)
    changed["evaluation"]["episodes"][2]["lap_time_s"] = 0.1
    mutations.append(changed)
    changed = copy.deepcopy(report)
    oversized_samples = [0.001] * 4001
    changed["evaluation"]["episodes"][4]["steps"] = len(oversized_samples)
    changed["evaluation"]["episodes"][4]["compute_times_s"] = oversized_samples
    changed["evaluation"]["episodes"][4]["compute_timing"] = asdict(
        summarize_compute_times(oversized_samples)
    )
    mutations.append(changed)
    changed = copy.deepcopy(report)
    changed["source"]["post_output_worktree"]["observed_payload_changed_paths"] = ["unexpected.txt"]
    mutations.append(changed)
    changed = copy.deepcopy(report)
    changed["source"]["post_output_worktree"]["report_output_path"] = config.trajectory_path
    mutations.append(changed)
    changed = copy.deepcopy(report)
    changed["export"]["selected_candidate"]["update_index"] = 70
    mutations.append(changed)
    changed = copy.deepcopy(report)
    changed["export"]["controller_artifacts"]["policy"]["sha256"] = "f" * 64
    mutations.append(changed)
    changed = copy.deepcopy(report)
    changed["export"]["controller_checkpoint"]["checkpoint_sha256"] = "f" * 64
    mutations.append(changed)
    changed = copy.deepcopy(report)
    changed["selection"]["selected_success_count"] = 101
    changed["selection"]["selected_success_rate"] = 1.01
    mutations.append(changed)
    changed = copy.deepcopy(report)
    changed["artifacts"]["training_config"]["sha256"] = "f" * 64
    changed["source"]["input_sha256_before"]["training_config"] = "f" * 64
    changed["source"]["input_sha256_after"]["training_config"] = "f" * 64
    mutations.append(changed)
    changed = copy.deepcopy(report)
    changed["protocol"]["output_crash_recovery_method"] = "memory_only_rollback"
    mutations.append(changed)
    changed = copy.deepcopy(report)
    changed["protocol"]["replay_capture_method"] = "rerun_selected_episode"
    mutations.append(changed)
    changed = copy.deepcopy(report)
    changed["replay"]["captured_from_evaluation_row"] = False
    mutations.append(changed)
    changed = copy.deepcopy(report)
    changed["execution"]["recorded_episode_count"] = 4
    mutations.append(changed)
    changed = copy.deepcopy(report)
    changed["execution"]["physics_substeps"] += 10
    mutations.append(changed)
    changed = copy.deepcopy(report)
    changed["execution"]["captured_replay_steps"] = True
    mutations.append(changed)
    changed = copy.deepcopy(report)
    changed["execution"]["captured_replay_episode_wall_s"] = 10.1
    mutations.append(changed)

    for mutation in mutations:
        with pytest.raises(ControllerBenchmarkProtocolError):
            validate_controller_evaluation_report(mutation, config=config)


def test_episode_report_row_keeps_raw_compute_samples() -> None:
    from controller_learning.envs.episode import initialize_episode_identities

    episode = _episode(2)
    identity = initialize_episode_identities(episode.reset_seed, 1)
    row = episode_to_report_row(
        episode,
        episode_seed=int(identity.episode_seed[0]),
        controller_seed=int(identity.controller_seed[0]),
        benchmark_version="0.1",
    )
    assert row["compute_times_s"] == list(episode.compute_times_s)
    assert row["compute_timing"] == asdict(episode.compute_timing)
