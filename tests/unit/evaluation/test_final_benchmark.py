"""Tests for the frozen M8 final-evaluation configuration and ranking rules."""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import pytest

from controller_learning.evaluation.controller_identity import capture_frozen_controller_identity
from controller_learning.evaluation.final_benchmark import (
    M8_ACCEPTED_RESULT_RULE,
    M8_CONTROLLER_ORDER,
    M8_FINAL_RUN_ID,
    M8_REPLAY_CAPTURE_METHOD,
    M8_RESET_SEED_RULE,
    FinalBenchmarkProtocolError,
    controller_output_paths,
    formal_output_paths,
    load_m8_final_evaluation_config,
    rank_controller_summaries,
    validate_formal_output_tree,
)

PROJECT_ROOT = Path(__file__).parents[3]
CONFIG_PATH = PROJECT_ROOT / "configs/final_evaluation.toml"


def test_frozen_final_config_and_output_allowlist() -> None:
    config = load_m8_final_evaluation_config(CONFIG_PATH)

    assert config.run_id == M8_FINAL_RUN_ID
    assert config.controller_order == M8_CONTROLLER_ORDER
    assert config.reset_seed_rule == M8_RESET_SEED_RULE
    assert config.replay_capture_method == M8_REPLAY_CAPTURE_METHOD
    assert config.accepted_result == M8_ACCEPTED_RESULT_RULE
    assert config.test_track_count == 20
    assert config.environment_instances == 1
    assert config.replay_environment_instances == 0
    assert config.to_dict()["controller_order"] == ["pid", "mpc", "ppo"]

    outputs = formal_output_paths(config)
    assert len(outputs) == len(set(outputs)) == 24
    assert tuple(outputs) == tuple(sorted(outputs))
    for name in M8_CONTROLLER_ORDER:
        identity = capture_frozen_controller_identity(PROJECT_ROOT, name)
        assert identity.aggregate_sha256 == config.controller_aggregate_sha256[name]
        assert identity.config_sha256 == config.controller_config_sha256[name]
        paths = controller_output_paths(config, name)
        assert set(paths) == {
            "metrics",
            "replay_trajectory",
            "results",
            "run_manifest",
            "summary",
            "telemetry",
            "trajectory",
        }
        assert all(f"/{name}/{M8_FINAL_RUN_ID}/" in path for path in paths.values())


def test_final_config_rejects_aliases_drift_and_unknown_keys(tmp_path: Path) -> None:
    config = load_m8_final_evaluation_config(CONFIG_PATH)

    constructors = (
        lambda: replace(config, schema_version=True),
        lambda: replace(config, controller_order=("ppo", "mpc", "pid")),
        lambda: replace(config, replay_test_row_index=1),
        lambda: replace(config, automatic_retry_after_test_bound=True),
        lambda: replace(config, replay_capture_method="rerun_selected_episode"),
        lambda: replace(
            config,
            controller_directories={**config.controller_directories, "pid": "other"},
        ),
    )
    for construct in constructors:
        with pytest.raises(FinalBenchmarkProtocolError):
            construct()

    modified = CONFIG_PATH.read_text(encoding="utf-8") + "\nunknown = 1\n"
    path = tmp_path / "final.toml"
    path.write_text(modified, encoding="utf-8")
    with pytest.raises(FinalBenchmarkProtocolError, match="keys differ"):
        load_m8_final_evaluation_config(path)


def test_final_ranking_uses_success_then_lap_without_combined_score() -> None:
    summaries = {
        "pid": {"success_rate": 0.9, "mean_successful_lap_time_s": 30.0},
        "mpc": {"success_rate": 1.0, "mean_successful_lap_time_s": 28.0},
        "ppo": {"success_rate": 1.0, "mean_successful_lap_time_s": 25.0},
    }
    assert rank_controller_summaries(summaries) == ("ppo", "mpc", "pid")

    summaries["pid"] = {"success_rate": 0.0, "mean_successful_lap_time_s": None}
    assert rank_controller_summaries(summaries)[-1] == "pid"
    with pytest.raises(FinalBenchmarkProtocolError):
        rank_controller_summaries({"pid": summaries["pid"]})


def test_formal_output_tree_rejects_partial_publication_and_residue(tmp_path: Path) -> None:
    config = load_m8_final_evaluation_config(CONFIG_PATH)
    assert validate_formal_output_tree(tmp_path, config, expected_present=False) == ()

    outputs = formal_output_paths(config)
    for relative in outputs:
        path = tmp_path / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(relative.encode())
    assert validate_formal_output_tree(tmp_path, config, expected_present=True) == outputs

    missing = tmp_path / outputs[0]
    missing.unlink()
    with pytest.raises(FinalBenchmarkProtocolError, match="presence differs"):
        validate_formal_output_tree(tmp_path, config, expected_present=True)
    missing.write_bytes(b"restored")

    residue = tmp_path / config.results_root / "pid" / config.run_id / "unexpected.log"
    residue.write_text("residue", encoding="utf-8")
    with pytest.raises(FinalBenchmarkProtocolError, match="unallowlisted residue"):
        validate_formal_output_tree(tmp_path, config, expected_present=True)
