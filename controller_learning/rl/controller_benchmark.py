"""Frozen post-selection PPO Controller evaluation and replay protocol."""

from __future__ import annotations

import hashlib
import math
import os
import re
import tomllib
from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
from itertools import pairwise
from pathlib import Path, PurePosixPath
from typing import Any, Final

import numpy as np

from controller_learning.evaluation.controller import (
    EpisodeEvaluation,
    summarize_compute_times,
)

CONTROLLER_EVALUATION_CONFIG_SCHEMA_VERSION: Final = 2
CONTROLLER_EVALUATION_REPORT_SCHEMA_VERSION: Final = (
    "controller-learning.m7-ppo-controller-evaluation.v2"
)
FORMAL_VALIDATION_TRACK_COUNT: Final = 100
FORMAL_MAX_EPISODE_STEPS: Final = 4000
FORMAL_RESET_SEED_RULE: Final = "validation_row_index_uint32"
FORMAL_REPLAY_SELECTION_RULE: Final = "first_successful_track_in_fixed_order_else_first_track"
FORMAL_REPLAY_CAPTURE_METHOD: Final = (
    "record_fixed_order_episodes_until_first_success_and_retain_selected_evaluation_trajectory"
)
FORMAL_CONTROLLER_EXECUTION_MODEL: Final = (
    "one reusable batch-one MJX-Warp environment with one fresh ordinary Controller per episode"
)
FORMAL_OUTPUT_CRASH_RECOVERY_METHOD: Final = (
    "fixed_runs_transaction_with_fsynced_backups_ready_and_transaction_staging_v1"
)

_SHA256_PATTERN = re.compile(r"^[0-9a-f]{64}$")
_SOURCE_REVISION_PATTERN = re.compile(r"^[0-9a-f]{40}$")
_TERMINATION_REASONS: Final = {1: "success", 2: "off_track", 3: "invalid_action", 4: "timeout"}


class ControllerBenchmarkProtocolError(ValueError):
    """A frozen config or published report violates the M7 Controller protocol."""


def _exact_keys(value: Mapping[str, Any], expected: set[str], *, field: str) -> None:
    if any(type(key) is not str for key in value) or set(value) != expected:
        raise ControllerBenchmarkProtocolError(
            f"{field} keys differ; missing={sorted(expected - set(value))}, "
            f"extra={sorted(set(value) - expected)}"
        )


def _table(value: Mapping[str, Any], key: str) -> Mapping[str, Any]:
    result = value.get(key)
    if not isinstance(result, Mapping):
        raise ControllerBenchmarkProtocolError(f"{key} must be a TOML table")
    return result


def _plain_integer(value: object, *, field: str, minimum: int = 0) -> int:
    if type(value) is not int or value < minimum:
        raise ControllerBenchmarkProtocolError(
            f"{field} must be an integer greater than or equal to {minimum}"
        )
    return value


def _safe_relative_path(value: object, *, field: str, suffix: str | None = None) -> str:
    if not isinstance(value, str) or not value or "\\" in value:
        raise ControllerBenchmarkProtocolError(f"{field} must be a normalized relative POSIX path")
    path = PurePosixPath(value)
    if (
        path.is_absolute()
        or path.as_posix() != value
        or any(part in {"", ".", ".."} for part in path.parts)
    ):
        raise ControllerBenchmarkProtocolError(f"{field} must be a normalized relative POSIX path")
    if suffix is not None and path.suffix != suffix:
        raise ControllerBenchmarkProtocolError(f"{field} must use the {suffix} suffix")
    return value


def _sha256(value: object, *, field: str) -> str:
    if not isinstance(value, str) or _SHA256_PATTERN.fullmatch(value) is None:
        raise ControllerBenchmarkProtocolError(f"{field} must be a lowercase SHA-256 digest")
    return value


def _finite_number(value: object, *, field: str, minimum: float = 0.0) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ControllerBenchmarkProtocolError(f"{field} must be a finite number")
    result = float(value)
    if not math.isfinite(result) or result < minimum:
        raise ControllerBenchmarkProtocolError(
            f"{field} must be finite and greater than or equal to {minimum}"
        )
    return result


@dataclass(frozen=True, slots=True)
class PpoControllerEvaluationConfig:
    """Exact inputs and outputs for the single formal post-selection run."""

    benchmark_version: str
    level_id: int
    backend: str
    validation_track_count: int
    reset_seed_rule: str
    max_episode_steps: int
    environment_instances: int
    fresh_controller_per_episode: bool
    selection_config: str
    selection_report: str
    export_report: str
    controller_directory: str
    replay_selection_rule: str
    replay_capture_method: str
    trajectory_path: str
    overview_path: str
    report_path: str
    schema_version: int = CONTROLLER_EVALUATION_CONFIG_SCHEMA_VERSION

    def __post_init__(self) -> None:
        if (
            type(self.schema_version) is not int
            or self.schema_version != CONTROLLER_EVALUATION_CONFIG_SCHEMA_VERSION
        ):
            raise ControllerBenchmarkProtocolError("config schema_version must be exactly 2")
        expected_scalars = {
            "benchmark_version": (self.benchmark_version, "0.1"),
            "backend": (self.backend, "mjx_warp"),
            "reset_seed_rule": (self.reset_seed_rule, FORMAL_RESET_SEED_RULE),
            "replay_selection_rule": (
                self.replay_selection_rule,
                FORMAL_REPLAY_SELECTION_RULE,
            ),
            "replay_capture_method": (
                self.replay_capture_method,
                FORMAL_REPLAY_CAPTURE_METHOD,
            ),
        }
        for field, (actual, expected) in expected_scalars.items():
            if actual != expected:
                raise ControllerBenchmarkProtocolError(f"{field} must be exactly {expected!r}")
        if type(self.level_id) is not int or self.level_id != 1:
            raise ControllerBenchmarkProtocolError("level_id must be exactly 1")
        if (
            type(self.validation_track_count) is not int
            or self.validation_track_count != FORMAL_VALIDATION_TRACK_COUNT
        ):
            raise ControllerBenchmarkProtocolError("validation_track_count must be exactly 100")
        if (
            type(self.max_episode_steps) is not int
            or self.max_episode_steps != FORMAL_MAX_EPISODE_STEPS
        ):
            raise ControllerBenchmarkProtocolError("max_episode_steps must be exactly 4000")
        if type(self.environment_instances) is not int or self.environment_instances != 1:
            raise ControllerBenchmarkProtocolError("environment_instances must be exactly 1")
        if self.fresh_controller_per_episode is not True:
            raise ControllerBenchmarkProtocolError("fresh_controller_per_episode must be true")
        expected_paths = {
            "selection_config": "configs/ppo_selection.toml",
            "selection_report": "benchmarks/v0.1/m7_ppo_selection_report.json",
            "export_report": "benchmarks/v0.1/m7_ppo_export_report.json",
            "controller_directory": "controllers/ppo",
            "trajectory_path": "benchmarks/v0.1/m7_ppo_replay_trajectory.json",
            "overview_path": "benchmarks/v0.1/m7_ppo_replay_overview.png",
            "report_path": "benchmarks/v0.1/m7_ppo_controller_evaluation_report.json",
        }
        for field, expected in expected_paths.items():
            actual = _safe_relative_path(
                getattr(self, field),
                field=field,
                suffix=Path(expected).suffix or None,
            )
            if actual != expected:
                raise ControllerBenchmarkProtocolError(f"{field} must be exactly {expected!r}")

    def to_dict(self) -> dict[str, Any]:
        """Return the stable JSON representation used in the report."""

        return asdict(self)


def load_ppo_controller_evaluation_config(
    path: str | Path,
) -> PpoControllerEvaluationConfig:
    """Load the frozen TOML without accepting aliases or unknown keys."""

    source = Path(path)
    if source.suffix != ".toml" or source.is_symlink() or not source.is_file():
        raise ControllerBenchmarkProtocolError(
            "Controller evaluation config must be a regular non-symlink TOML file"
        )
    try:
        with source.open("rb") as file:
            data = tomllib.load(file)
    except (OSError, tomllib.TOMLDecodeError) as error:
        raise ControllerBenchmarkProtocolError("Controller evaluation config is invalid") from error
    _exact_keys(
        data,
        {"schema_version", "protocol", "inputs", "replay", "artifacts"},
        field="config",
    )
    protocol = _table(data, "protocol")
    inputs = _table(data, "inputs")
    replay = _table(data, "replay")
    artifacts = _table(data, "artifacts")
    _exact_keys(
        protocol,
        {
            "benchmark_version",
            "level_id",
            "backend",
            "validation_track_count",
            "reset_seed_rule",
            "max_episode_steps",
            "environment_instances",
            "fresh_controller_per_episode",
        },
        field="protocol",
    )
    _exact_keys(
        inputs,
        {"selection_config", "selection_report", "export_report", "controller_directory"},
        field="inputs",
    )
    _exact_keys(
        replay,
        {"selection_rule", "capture_method", "trajectory_path", "overview_path"},
        field="replay",
    )
    _exact_keys(artifacts, {"report_path"}, field="artifacts")
    fresh = protocol["fresh_controller_per_episode"]
    if type(fresh) is not bool:
        raise ControllerBenchmarkProtocolError("fresh_controller_per_episode must be boolean")
    return PpoControllerEvaluationConfig(
        schema_version=_plain_integer(data["schema_version"], field="schema_version"),
        benchmark_version=protocol["benchmark_version"],
        level_id=_plain_integer(protocol["level_id"], field="protocol.level_id"),
        backend=protocol["backend"],
        validation_track_count=_plain_integer(
            protocol["validation_track_count"],
            field="protocol.validation_track_count",
            minimum=1,
        ),
        reset_seed_rule=protocol["reset_seed_rule"],
        max_episode_steps=_plain_integer(
            protocol["max_episode_steps"], field="protocol.max_episode_steps", minimum=1
        ),
        environment_instances=_plain_integer(
            protocol["environment_instances"],
            field="protocol.environment_instances",
            minimum=1,
        ),
        fresh_controller_per_episode=fresh,
        selection_config=inputs["selection_config"],
        selection_report=inputs["selection_report"],
        export_report=inputs["export_report"],
        controller_directory=inputs["controller_directory"],
        replay_selection_rule=replay["selection_rule"],
        replay_capture_method=replay["capture_method"],
        trajectory_path=replay["trajectory_path"],
        overview_path=replay["overview_path"],
        report_path=artifacts["report_path"],
    )


def replay_track_index(episodes: Sequence[Mapping[str, Any]]) -> int:
    """Apply the predeclared replay rule to fixed-order raw episode rows."""

    if not isinstance(episodes, Sequence) or isinstance(episodes, (str, bytes)) or not episodes:
        raise ControllerBenchmarkProtocolError("episodes must be a non-empty sequence")
    for expected_index, row in enumerate(episodes):
        if not isinstance(row, Mapping) or row.get("track_index") != expected_index:
            raise ControllerBenchmarkProtocolError("episodes must preserve contiguous fixed order")
        if type(row.get("success")) is not bool:
            raise ControllerBenchmarkProtocolError("episode success must be boolean")
        if row["success"]:
            return expected_index
    return 0


def episode_to_report_row(
    episode: EpisodeEvaluation,
    *,
    episode_seed: int,
    controller_seed: int,
    benchmark_version: str,
) -> dict[str, Any]:
    """Serialize every outcome and timing sample needed for exact recomputation."""

    if not isinstance(episode, EpisodeEvaluation):
        raise TypeError("episode must be an EpisodeEvaluation")
    for field, value in (("episode_seed", episode_seed), ("controller_seed", controller_seed)):
        if type(value) is not int or not 0 <= value < 2**32:
            raise ValueError(f"{field} must fit in uint32")
    if episode_seed == controller_seed:
        raise ValueError("episode_seed and controller_seed must be distinct")
    if benchmark_version != "0.1":
        raise ValueError("benchmark_version must be exactly '0.1'")
    return {
        "benchmark_version": benchmark_version,
        "compute_times_s": list(episode.compute_times_s),
        "compute_timing": asdict(episode.compute_timing),
        "controller_seed": controller_seed,
        "controller_import_time_s": episode.controller_import_time_s,
        "controller_init_time_s": episode.controller_init_time_s,
        "episode_seed": episode_seed,
        "lap_time_s": episode.lap_time_s,
        "reset_seed": episode.reset_seed,
        "steps": episode.steps,
        "success": episode.success,
        "terminated": episode.terminated,
        "termination_reason": episode.termination_reason,
        "total_reward": episode.total_reward,
        "track_id": episode.track_id,
        "track_index": episode.track_index,
        "truncated": episode.truncated,
    }


def evaluation_summary(episodes: Sequence[EpisodeEvaluation]) -> dict[str, Any]:
    """Compute the canonical 100-row aggregate from typed episode outcomes."""

    rows = tuple(episodes)
    if not rows:
        raise ControllerBenchmarkProtocolError("evaluation episodes cannot be empty")
    successful_laps = tuple(row.lap_time_s for row in rows if row.success)
    samples = tuple(sample for row in rows for sample in row.compute_times_s)
    reasons = Counter(_TERMINATION_REASONS[row.termination_reason] for row in rows)
    return {
        "compute_timing": asdict(summarize_compute_times(samples)),
        "environment_steps": sum(row.steps for row in rows),
        "mean_successful_lap_time_s": (
            float(np.mean(successful_laps, dtype=np.float64)) if successful_laps else None
        ),
        "success_count": len(successful_laps),
        "success_rate": len(successful_laps) / len(rows),
        "termination_counts": {
            label: reasons.get(label, 0) for label in _TERMINATION_REASONS.values()
        },
        "track_count": len(rows),
    }


def _episode_from_report_row(value: object, *, expected_index: int) -> EpisodeEvaluation:
    if not isinstance(value, Mapping):
        raise ControllerBenchmarkProtocolError(f"episodes[{expected_index}] must be an object")
    _exact_keys(
        value,
        {
            "benchmark_version",
            "compute_times_s",
            "compute_timing",
            "controller_seed",
            "controller_import_time_s",
            "controller_init_time_s",
            "episode_seed",
            "lap_time_s",
            "reset_seed",
            "steps",
            "success",
            "terminated",
            "termination_reason",
            "total_reward",
            "track_id",
            "track_index",
            "truncated",
        },
        field=f"episodes[{expected_index}]",
    )
    if value["track_index"] != expected_index or value["reset_seed"] != expected_index:
        raise ControllerBenchmarkProtocolError(
            "episode row index and uint32 reset seed must equal fixed Validation row index"
        )
    if value["benchmark_version"] != "0.1":
        raise ControllerBenchmarkProtocolError("episode benchmark_version must be exactly '0.1'")
    from controller_learning.envs.episode import initialize_episode_identities

    expected_identity = initialize_episode_identities(expected_index, 1)
    expected_episode_seed = int(expected_identity.episode_seed[0])
    expected_controller_seed = int(expected_identity.controller_seed[0])
    if (
        type(value["episode_seed"]) is not int
        or value["episode_seed"] != expected_episode_seed
        or type(value["controller_seed"]) is not int
        or value["controller_seed"] != expected_controller_seed
    ):
        raise ControllerBenchmarkProtocolError(
            "episode and Controller seeds differ from the fixed row-index root seed"
        )
    samples = value["compute_times_s"]
    if not isinstance(samples, list):
        raise ControllerBenchmarkProtocolError("compute_times_s must be a JSON array")
    typed = EpisodeEvaluation(
        track_index=value["track_index"],
        track_id=value["track_id"],
        reset_seed=value["reset_seed"],
        success=value["success"],
        lap_time_s=value["lap_time_s"],
        steps=value["steps"],
        total_reward=value["total_reward"],
        terminated=value["terminated"],
        truncated=value["truncated"],
        termination_reason=value["termination_reason"],
        controller_import_time_s=value["controller_import_time_s"],
        controller_init_time_s=value["controller_init_time_s"],
        compute_times_s=tuple(samples),
        compute_timing=summarize_compute_times(samples),
    )
    if value["compute_timing"] != asdict(typed.compute_timing):
        raise ControllerBenchmarkProtocolError("episode compute timing differs from raw samples")
    if typed.steps > FORMAL_MAX_EPISODE_STEPS:
        raise ControllerBenchmarkProtocolError("episode steps exceed the formal safety bound")
    if typed.success and not math.isclose(
        float(typed.lap_time_s),
        typed.steps * 0.05,
        rel_tol=0.0,
        abs_tol=2.0e-5,
    ):
        raise ControllerBenchmarkProtocolError(
            "successful lap_time_s must match float32 Challenge control time"
        )
    if (
        typed.termination_reason not in _TERMINATION_REASONS
        or (typed.termination_reason == 1) != typed.success
        or (typed.termination_reason == 4) != typed.truncated
        or (typed.termination_reason in {1, 2, 3}) != typed.terminated
    ):
        raise ControllerBenchmarkProtocolError("episode terminal flags and reason differ")
    return typed


def _artifact_record(
    value: object,
    *,
    field: str,
    expected_path: str | None = None,
) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise ControllerBenchmarkProtocolError(f"{field} must be an artifact object")
    _exact_keys(
        value,
        {"relative_path", "schema_version", "sha256", "size_bytes"},
        field=field,
    )
    path = _safe_relative_path(value["relative_path"], field=f"{field}.relative_path")
    if expected_path is not None and path != expected_path:
        raise ControllerBenchmarkProtocolError(f"{field} path differs from the frozen protocol")
    if type(value["schema_version"]) is not int or value["schema_version"] != 1:
        raise ControllerBenchmarkProtocolError(f"{field}.schema_version must be 1")
    _sha256(value["sha256"], field=f"{field}.sha256")
    _plain_integer(value["size_bytes"], field=f"{field}.size_bytes", minimum=1)
    return dict(value)


def _validate_access_evidence(value: object) -> None:
    if not isinstance(value, Mapping):
        raise ControllerBenchmarkProtocolError("asset_access must be an object")
    _exact_keys(
        value,
        {
            "audit_hook_installed_before_preflight",
            "denied_event_count",
            "denied_mutation_event_count",
            "denied_mutation_event_types",
            "open_event_counts",
            "open_event_sequence",
            "opened_path_categories",
            "opened_splits",
            "pre_validation_open_event_count",
            "test_opened",
            "track_cache_opened",
            "train_opened",
            "validation_loaded",
            "validation_reads_enabled",
        },
        field="asset_access",
    )
    if (
        value["audit_hook_installed_before_preflight"] is not True
        or value["validation_reads_enabled"] is not True
        or value["denied_mutation_event_count"] != 0
        or value["denied_mutation_event_types"] != {}
        or value["validation_loaded"] is not True
        or value["opened_splits"] != ["validation"]
        or value["opened_path_categories"]
        != ["official_validation_asset", "official_validation_manifest"]
        or value["test_opened"] is not False
        or value["train_opened"] is not False
        or value["track_cache_opened"] is not False
        or value["pre_validation_open_event_count"] != 0
        or value["denied_event_count"] != 0
    ):
        raise ControllerBenchmarkProtocolError("asset access is not Validation-only and read-only")
    counts = value["open_event_counts"]
    sequence = value["open_event_sequence"]
    if (
        not isinstance(counts, Mapping)
        or set(counts) != {"official_validation_asset", "official_validation_manifest"}
        or any(type(count) is not int or count < 1 for count in counts.values())
        or not isinstance(sequence, list)
        or len(sequence) != sum(counts.values())
    ):
        raise ControllerBenchmarkProtocolError("Validation open-event evidence is incomplete")
    observed_counts: Counter[str] = Counter()
    for index, event in enumerate(sequence):
        if not isinstance(event, Mapping):
            raise ControllerBenchmarkProtocolError(
                f"open_event_sequence[{index}] must be an object"
            )
        _exact_keys(event, {"category", "flags", "mode"}, field=f"open_event_sequence[{index}]")
        category = event["category"]
        if category not in counts:
            raise ControllerBenchmarkProtocolError("open-event category is not Validation-only")
        if event["mode"] is not None and not isinstance(event["mode"], str):
            raise ControllerBenchmarkProtocolError("open-event mode must be string or null")
        if event["flags"] is not None and type(event["flags"]) is not int:
            raise ControllerBenchmarkProtocolError("open-event flags must be integer or null")
        write_mode = isinstance(event["mode"], str) and any(
            token in event["mode"] for token in "wax+"
        )
        write_flags = type(event["flags"]) is int and bool(
            event["flags"] & (os.O_WRONLY | os.O_RDWR | os.O_CREAT | os.O_TRUNC | os.O_APPEND)
        )
        if write_mode or write_flags:
            raise ControllerBenchmarkProtocolError("Validation open-event was not read-only")
        observed_counts[category] += 1
    if dict(observed_counts) != dict(counts):
        raise ControllerBenchmarkProtocolError("open-event counts differ from the event sequence")


def _validate_validation_assets(value: object, *, track_ids: Sequence[int]) -> None:
    if not isinstance(value, Mapping):
        raise ControllerBenchmarkProtocolError("validation_assets must be an object")
    required = {
        "schema_version",
        "loaded_splits",
        "benchmark_version",
        "generator_version",
        "level_id",
        "split",
        "manifest_file",
        "manifest_sha256",
        "asset_file",
        "manifest_asset_sha256",
        "asset_file_sha256",
        "track_count",
        "capacity",
        "first_track_id",
        "last_track_id",
        "track_ids_sha256",
        "geometry_hashes_sha256",
        "loader_accessed_train",
        "loader_accessed_test",
    }
    _exact_keys(value, required, field="validation_assets")
    if (
        value["schema_version"] != "controller-learning.m7-validation-pool-access.v1"
        or value["loaded_splits"] != ["validation"]
        or value["benchmark_version"] != "0.1"
        or value["level_id"] != 1
        or value["split"] != "validation"
        or value["manifest_file"] != "validation.json"
        or value["asset_file"] != "validation.npz"
        or value["track_count"] != FORMAL_VALIDATION_TRACK_COUNT
        or value["loader_accessed_train"] is not False
        or value["loader_accessed_test"] is not False
        or value["first_track_id"] != track_ids[0]
        or value["last_track_id"] != track_ids[-1]
        or value["manifest_asset_sha256"] != value["asset_file_sha256"]
    ):
        raise ControllerBenchmarkProtocolError("Validation asset identity differs")
    capacity = value["capacity"]
    if not isinstance(capacity, Mapping):
        raise ControllerBenchmarkProtocolError("Validation capacity must be an object")
    _exact_keys(capacity, {"max_checkpoints", "max_track_points"}, field="capacity")
    if dict(capacity) != {"max_checkpoints": 48, "max_track_points": 640}:
        raise ControllerBenchmarkProtocolError("Validation capacity differs from benchmark 0.1")
    if any(previous >= current for previous, current in pairwise(track_ids)):
        raise ControllerBenchmarkProtocolError("Validation Track IDs must be strictly increasing")
    for field in (
        "manifest_sha256",
        "manifest_asset_sha256",
        "asset_file_sha256",
        "track_ids_sha256",
        "geometry_hashes_sha256",
    ):
        _sha256(value[field], field=f"validation_assets.{field}")
    digest = hashlib.sha256()
    for track_id in track_ids:
        digest.update(str(track_id).encode("ascii"))
        digest.update(b"\n")
    if digest.hexdigest() != value["track_ids_sha256"]:
        raise ControllerBenchmarkProtocolError("Validation Track-ID digest differs from rows")


def controller_evaluation_report_findings(
    report: object,
    *,
    config: PpoControllerEvaluationConfig,
) -> tuple[str, ...]:
    """Return one concise structural or recomputation failure for a strict report."""

    findings: list[str] = []
    try:
        if not isinstance(config, PpoControllerEvaluationConfig):
            raise ControllerBenchmarkProtocolError("config type differs")
        if not isinstance(report, Mapping):
            raise ControllerBenchmarkProtocolError("report must be an object")
        _exact_keys(
            report,
            {
                "artifacts",
                "asset_access",
                "configuration",
                "controller",
                "evaluation",
                "execution",
                "export",
                "memory",
                "protocol",
                "replay",
                "runtime",
                "schema_version",
                "selection",
                "source",
                "status",
                "validation_assets",
            },
            field="report",
        )
        if report["schema_version"] != CONTROLLER_EVALUATION_REPORT_SCHEMA_VERSION:
            raise ControllerBenchmarkProtocolError("report schema_version differs")
        if report["configuration"] != config.to_dict():
            raise ControllerBenchmarkProtocolError("reported configuration differs")
        expected_protocol = {
            "backend": "mjx_warp",
            "benchmark_version": "0.1",
            "controller_execution_model": FORMAL_CONTROLLER_EXECUTION_MODEL,
            "environment_instances": 1,
            "fresh_controller_per_episode": True,
            "level_id": 1,
            "max_episode_steps": FORMAL_MAX_EPISODE_STEPS,
            "no_gradient_updates": True,
            "ordinary_controller_plugin": True,
            "output_crash_recovery_method": FORMAL_OUTPUT_CRASH_RECOVERY_METHOD,
            "replay_capture_method": FORMAL_REPLAY_CAPTURE_METHOD,
            "replay_environment_instances": 0,
            "replay_selection_rule": FORMAL_REPLAY_SELECTION_RULE,
            "reset_seed_rule": FORMAL_RESET_SEED_RULE,
            "test_accessed": False,
            "train_assets_accessed": False,
            "validation_track_count": FORMAL_VALIDATION_TRACK_COUNT,
        }
        if report["protocol"] != expected_protocol:
            raise ControllerBenchmarkProtocolError("protocol differs from frozen values")

        evaluation = report["evaluation"]
        if not isinstance(evaluation, Mapping):
            raise ControllerBenchmarkProtocolError("evaluation must be an object")
        _exact_keys(evaluation, {"episodes", "summary"}, field="evaluation")
        episode_values = evaluation["episodes"]
        if not isinstance(episode_values, list) or len(episode_values) != 100:
            raise ControllerBenchmarkProtocolError("evaluation must contain exactly 100 rows")
        episodes = tuple(
            _episode_from_report_row(value, expected_index=index)
            for index, value in enumerate(episode_values)
        )
        expected_summary = evaluation_summary(episodes)
        if evaluation["summary"] != expected_summary:
            raise ControllerBenchmarkProtocolError("evaluation summary differs from raw rows")
        track_ids = tuple(episode.track_id for episode in episodes)
        if len(set(track_ids)) != 100:
            raise ControllerBenchmarkProtocolError("Validation Track IDs must be unique")
        _validate_validation_assets(report["validation_assets"], track_ids=track_ids)
        _validate_access_evidence(report["asset_access"])

        selection = report["selection"]
        if not isinstance(selection, Mapping):
            raise ControllerBenchmarkProtocolError("selection must be an object")
        _exact_keys(
            selection,
            {
                "gate_passed",
                "report_status",
                "selected_checkpoint_sha256",
                "selected_inference_policy_schema_version",
                "selected_inference_policy_sha256",
                "selected_inference_policy_size_bytes",
                "selected_success_count",
                "selected_success_rate",
                "selected_update",
                "training_configuration_sha256",
            },
            field="selection",
        )
        if (
            selection["gate_passed"] is not True
            or selection["report_status"] != "passed"
            or type(selection["selected_update"]) is not int
            or selection["selected_update"] not in {10, 20, 30, 40, 50, 60, 70, 80}
            or type(selection["selected_success_count"]) is not int
            or not 1 <= selection["selected_success_count"] <= FORMAL_VALIDATION_TRACK_COUNT
            or _finite_number(
                selection["selected_success_rate"],
                field="selection.selected_success_rate",
            )
            > 1.0
            or not math.isclose(
                _finite_number(
                    selection["selected_success_rate"],
                    field="selection.selected_success_rate",
                ),
                selection["selected_success_count"] / 100,
                rel_tol=0.0,
                abs_tol=1.0e-15,
            )
        ):
            raise ControllerBenchmarkProtocolError("selection did not pass its frozen gate")
        _sha256(
            selection["selected_checkpoint_sha256"],
            field="selection.selected_checkpoint_sha256",
        )
        _sha256(
            selection["selected_inference_policy_sha256"],
            field="selection.selected_inference_policy_sha256",
        )
        if (
            type(selection["selected_inference_policy_schema_version"]) is not int
            or selection["selected_inference_policy_schema_version"] != 1
            or type(selection["selected_inference_policy_size_bytes"]) is not int
            or selection["selected_inference_policy_size_bytes"] < 1
        ):
            raise ControllerBenchmarkProtocolError("selected inference policy identity is invalid")
        _sha256(
            selection["training_configuration_sha256"],
            field="selection.training_configuration_sha256",
        )

        controller = report["controller"]
        if not isinstance(controller, Mapping):
            raise ControllerBenchmarkProtocolError("controller must be an object")
        _exact_keys(
            controller,
            {
                "checkpoint",
                "config_sha256",
                "directory",
                "finalized",
                "fresh_instance_count",
                "inference_runtime",
                "metadata_sha256",
                "name",
                "policy_schema_version",
                "policy_sha256",
                "policy_size_bytes",
                "torch_imported",
            },
            field="controller",
        )
        checkpoint = controller["checkpoint"]
        if not isinstance(checkpoint, Mapping):
            raise ControllerBenchmarkProtocolError("controller.checkpoint must be an object")
        _exact_keys(
            checkpoint,
            {
                "checkpoint_sha256",
                "run_id",
                "source_revision",
                "training_configuration_sha256",
                "update_index",
                "valid_transitions",
                "vector_steps",
            },
            field="controller.checkpoint",
        )
        if (
            controller["directory"] != config.controller_directory
            or controller["name"] != "ppo"
            or controller["finalized"] is not True
            or controller["fresh_instance_count"] != 100
            or controller["inference_runtime"] != "numpy"
            or controller["policy_schema_version"] != 1
            or type(controller["policy_size_bytes"]) is not int
            or controller["policy_size_bytes"] < 1
            or controller["torch_imported"] is not False
            or checkpoint["update_index"] != selection["selected_update"]
            or checkpoint["checkpoint_sha256"] != selection["selected_checkpoint_sha256"]
            or checkpoint["training_configuration_sha256"]
            != selection["training_configuration_sha256"]
            or checkpoint["run_id"] != "m7-formal-v0-1-001"
            or type(checkpoint["vector_steps"]) is not int
            or checkpoint["vector_steps"] < 1
            or type(checkpoint["valid_transitions"]) is not int
            or checkpoint["valid_transitions"] < 1
            or not isinstance(checkpoint["source_revision"], str)
            or _SOURCE_REVISION_PATTERN.fullmatch(checkpoint["source_revision"]) is None
        ):
            raise ControllerBenchmarkProtocolError("finalized Controller identity differs")

        export = report["export"]
        if not isinstance(export, Mapping):
            raise ControllerBenchmarkProtocolError("export must be an object")
        _exact_keys(
            export,
            {
                "controller_artifacts",
                "controller_checkpoint",
                "report_schema_version",
                "report_status",
                "selected_candidate",
            },
            field="export",
        )
        if (
            export["report_schema_version"] != "controller-learning.m7-ppo-controller-export.v1"
            or export["report_status"] != "passed"
            or export["controller_checkpoint"] != checkpoint
        ):
            raise ControllerBenchmarkProtocolError("canonical export report identity differs")
        exported_candidate = export["selected_candidate"]
        if not isinstance(exported_candidate, Mapping):
            raise ControllerBenchmarkProtocolError("export.selected_candidate must be an object")
        _exact_keys(
            exported_candidate,
            {
                "checkpoint",
                "inference_policy",
                "parameter_sha256",
                "update_index",
                "valid_transitions",
                "vector_steps",
            },
            field="export.selected_candidate",
        )
        exported_checkpoint = _artifact_record(
            exported_candidate["checkpoint"],
            field="export.selected_candidate.checkpoint",
            expected_path=f"checkpoints/update_{selection['selected_update']:08d}.pt",
        )
        exported_policy = exported_candidate["inference_policy"]
        if not isinstance(exported_policy, Mapping):
            raise ControllerBenchmarkProtocolError("export inference_policy must be an object")
        _exact_keys(
            exported_policy,
            {"schema_version", "sha256", "size_bytes"},
            field="export.selected_candidate.inference_policy",
        )
        _sha256(
            exported_candidate["parameter_sha256"],
            field="export.selected_candidate.parameter_sha256",
        )
        if (
            exported_checkpoint["sha256"] != selection["selected_checkpoint_sha256"]
            or exported_candidate["update_index"] != selection["selected_update"]
            or exported_candidate["vector_steps"] != checkpoint["vector_steps"]
            or exported_candidate["valid_transitions"] != checkpoint["valid_transitions"]
            or exported_policy
            != {
                "schema_version": selection["selected_inference_policy_schema_version"],
                "sha256": selection["selected_inference_policy_sha256"],
                "size_bytes": selection["selected_inference_policy_size_bytes"],
            }
        ):
            raise ControllerBenchmarkProtocolError("exported candidate differs from selection")
        exported_artifacts = export["controller_artifacts"]
        if not isinstance(exported_artifacts, Mapping):
            raise ControllerBenchmarkProtocolError("export controller_artifacts must be an object")
        _exact_keys(
            exported_artifacts,
            {"config", "metadata", "policy"},
            field="export.controller_artifacts",
        )
        exported_artifact_paths = {
            "config": "controllers/ppo/config.toml",
            "metadata": "controllers/ppo/metadata.json",
            "policy": "controllers/ppo/policy.npz",
        }
        exported_controller_records = {
            name: _artifact_record(
                exported_artifacts[name],
                field=f"export.controller_artifacts.{name}",
                expected_path=expected_path,
            )
            for name, expected_path in exported_artifact_paths.items()
        }

        replay = report["replay"]
        if not isinstance(replay, Mapping):
            raise ControllerBenchmarkProtocolError("replay must be an object")
        _exact_keys(
            replay,
            {
                "captured_from_evaluation_row",
                "overview",
                "reset_seed",
                "selection_rule",
                "track_id",
                "track_index",
                "trajectory",
            },
            field="replay",
        )
        selected_index = replay_track_index(episode_values)
        selected_row = episodes[selected_index]
        if (
            replay["selection_rule"] != FORMAL_REPLAY_SELECTION_RULE
            or replay["track_index"] != selected_index
            or replay["track_id"] != selected_row.track_id
            or replay["reset_seed"] != selected_index
            or replay["captured_from_evaluation_row"] is not True
        ):
            raise ControllerBenchmarkProtocolError("replay row differs from the predeclared rule")
        trajectory = replay["trajectory"]
        overview = replay["overview"]
        if not isinstance(trajectory, Mapping) or not isinstance(overview, Mapping):
            raise ControllerBenchmarkProtocolError("replay artifacts must be objects")
        _exact_keys(
            trajectory,
            {
                "artifact",
                "final_lap_completed",
                "final_termination_reason",
                "frame_count",
                "schema_version",
                "step_count",
            },
            field="replay.trajectory",
        )
        _exact_keys(
            overview,
            {
                "artifact",
                "all_source_frames_rendered",
                "rendered_frame_count",
                "source_frame_count",
            },
            field="replay.overview",
        )
        _artifact_record(
            trajectory["artifact"],
            field="replay.trajectory.artifact",
            expected_path=config.trajectory_path,
        )
        _artifact_record(
            overview["artifact"],
            field="replay.overview.artifact",
            expected_path=config.overview_path,
        )
        if (
            trajectory["schema_version"] != "controller-learning-trajectory-v1"
            or trajectory["step_count"] != selected_row.steps
            or trajectory["frame_count"] != selected_row.steps + 1
            or trajectory["final_lap_completed"] is not selected_row.success
            or trajectory["final_termination_reason"] != selected_row.termination_reason
            or overview["source_frame_count"] != trajectory["frame_count"]
            or overview["rendered_frame_count"] != trajectory["frame_count"]
            or overview["all_source_frames_rendered"] is not True
        ):
            raise ControllerBenchmarkProtocolError("replay contents differ from evaluation row")

        artifacts = report["artifacts"]
        if not isinstance(artifacts, Mapping):
            raise ControllerBenchmarkProtocolError("artifacts must be an object")
        expected_artifact_paths = {
            "controller_config": "controllers/ppo/config.toml",
            "controller_metadata": "controllers/ppo/metadata.json",
            "controller_policy": "controllers/ppo/policy.npz",
            "controller_source": "controllers/ppo/controller.py",
            "evaluation_config": "configs/ppo_controller_evaluation.toml",
            "export_report": config.export_report,
            "pixi_lock": "pixi.lock",
            "selection_config": config.selection_config,
            "selection_report": config.selection_report,
            "training_config": "configs/ppo.toml",
            "validation_asset": "controller_learning/assets/tracks/v0.1/validation.npz",
            "validation_manifest": "controller_learning/assets/tracks/v0.1/validation.json",
        }
        _exact_keys(artifacts, set(expected_artifact_paths), field="artifacts")
        records = {
            name: _artifact_record(value, field=f"artifacts.{name}", expected_path=path)
            for name, (value, path) in (
                (name, (artifacts[name], expected_path))
                for name, expected_path in expected_artifact_paths.items()
            )
        }
        if (
            records["controller_policy"]["sha256"] != controller["policy_sha256"]
            or records["controller_policy"]["size_bytes"] != controller["policy_size_bytes"]
            or controller["policy_sha256"] != selection["selected_inference_policy_sha256"]
            or controller["policy_size_bytes"] != selection["selected_inference_policy_size_bytes"]
            or controller["policy_schema_version"]
            != selection["selected_inference_policy_schema_version"]
            or records["controller_metadata"]["sha256"] != controller["metadata_sha256"]
            or records["controller_config"]["sha256"] != controller["config_sha256"]
            or records["training_config"]["sha256"] != selection["training_configuration_sha256"]
            or exported_controller_records["config"] != records["controller_config"]
            or exported_controller_records["metadata"] != records["controller_metadata"]
            or exported_controller_records["policy"] != records["controller_policy"]
            or records["validation_asset"]["sha256"]
            != report["validation_assets"]["asset_file_sha256"]
            or records["validation_manifest"]["sha256"]
            != report["validation_assets"]["manifest_sha256"]
        ):
            raise ControllerBenchmarkProtocolError("artifact hash links differ")

        execution = report["execution"]
        if not isinstance(execution, Mapping):
            raise ControllerBenchmarkProtocolError("execution must be an object")
        _exact_keys(
            execution,
            {
                "environment_instances",
                "environment_steps",
                "evaluation_wall_s",
                "first_use_timing",
                "physics_substeps",
                "captured_replay_episode_wall_s",
                "captured_replay_steps",
                "recorded_episode_count",
                "transitions_per_second",
            },
            field="execution",
        )
        environment_steps = expected_summary["environment_steps"]
        wall = _finite_number(execution["evaluation_wall_s"], field="evaluation_wall_s")
        environment_instances = _plain_integer(
            execution["environment_instances"],
            field="execution.environment_instances",
            minimum=1,
        )
        reported_environment_steps = _plain_integer(
            execution["environment_steps"],
            field="execution.environment_steps",
            minimum=1,
        )
        captured_replay_steps = _plain_integer(
            execution["captured_replay_steps"],
            field="execution.captured_replay_steps",
            minimum=1,
        )
        recorded_episode_count = _plain_integer(
            execution["recorded_episode_count"],
            field="execution.recorded_episode_count",
            minimum=1,
        )
        physics_substeps = _plain_integer(
            execution["physics_substeps"],
            field="execution.physics_substeps",
            minimum=1,
        )
        captured_wall = _finite_number(
            execution["captured_replay_episode_wall_s"],
            field="captured_replay_episode_wall_s",
        )
        first_success_index = next(
            (index for index, episode in enumerate(episodes) if episode.success),
            None,
        )
        expected_recorded_episode_count = (
            first_success_index + 1
            if first_success_index is not None
            else FORMAL_VALIDATION_TRACK_COUNT
        )
        if (
            environment_instances != 1
            or reported_environment_steps != environment_steps
            or captured_replay_steps != selected_row.steps
            or recorded_episode_count != expected_recorded_episode_count
            or physics_substeps != 10 * environment_steps
            or wall <= 0.0
            or captured_wall <= 0.0
            or captured_wall > wall
            or not math.isclose(
                execution["transitions_per_second"],
                environment_steps / wall,
                rel_tol=1.0e-12,
                abs_tol=0.0,
            )
        ):
            raise ControllerBenchmarkProtocolError("execution totals differ from raw rows")
        first_use = execution["first_use_timing"]
        if not isinstance(first_use, Mapping):
            raise ControllerBenchmarkProtocolError("first_use_timing must be an object")
        _exact_keys(
            first_use,
            {
                "first_environment_create_s",
                "first_reset_s",
                "first_step_s",
                "method",
            },
            field="first_use_timing",
        )
        for field in ("first_environment_create_s", "first_reset_s", "first_step_s"):
            _finite_number(first_use[field], field=f"first_use_timing.{field}")
        if not isinstance(first_use["method"], str) or not first_use["method"]:
            raise ControllerBenchmarkProtocolError("first-use timing method is missing")

        memory = report["memory"]
        if not isinstance(memory, Mapping):
            raise ControllerBenchmarkProtocolError("memory must be an object")
        _exact_keys(
            memory,
            {
                "peak_jax_allocator_bytes",
                "peak_sampled_process_vram_mib",
                "sample_count",
                "samples",
            },
            field="memory",
        )
        samples = memory["samples"]
        if (
            not isinstance(samples, list)
            or memory["sample_count"] != len(samples)
            or len(samples) != 4
        ):
            raise ControllerBenchmarkProtocolError("memory sample coverage is incomplete")
        process_values: list[float] = []
        allocator_values: list[int] = []
        expected_memory_phases = (
            "before_environment_create",
            "after_controller_evaluation",
            "after_replay_capture_validation",
            "after_artifact_render",
        )
        if tuple(sample.get("phase") for sample in samples if isinstance(sample, Mapping)) != (
            expected_memory_phases
        ):
            raise ControllerBenchmarkProtocolError("memory sample phases differ from the protocol")
        for index, sample in enumerate(samples):
            if not isinstance(sample, Mapping):
                raise ControllerBenchmarkProtocolError(f"memory.samples[{index}] must be an object")
            _exact_keys(
                sample,
                {
                    "jax_bytes_in_use",
                    "jax_peak_bytes_in_use",
                    "phase",
                    "process_vram_error",
                    "process_vram_mib",
                },
                field=f"memory.samples[{index}]",
            )
            if not isinstance(sample["phase"], str) or not sample["phase"]:
                raise ControllerBenchmarkProtocolError("memory sample phase is missing")
            if sample["process_vram_error"] is not None:
                raise ControllerBenchmarkProtocolError("process VRAM sampling failed")
            process = _finite_number(sample["process_vram_mib"], field="process_vram_mib")
            if process <= 0.0:
                raise ControllerBenchmarkProtocolError("process VRAM must be positive")
            process_values.append(process)
            allocator_values.append(
                _plain_integer(sample["jax_peak_bytes_in_use"], field="jax_peak_bytes_in_use")
            )
            _plain_integer(sample["jax_bytes_in_use"], field="jax_bytes_in_use")
        if memory["peak_sampled_process_vram_mib"] != max(process_values) or memory[
            "peak_jax_allocator_bytes"
        ] != max(allocator_values):
            raise ControllerBenchmarkProtocolError("memory peaks differ from samples")

        runtime = report["runtime"]
        if not isinstance(runtime, Mapping):
            raise ControllerBenchmarkProtocolError("runtime must be an object")
        _exact_keys(
            runtime,
            {
                "cuda_device_order",
                "cuda_visible_devices_configured",
                "jax_device",
                "kernel",
                "machine",
                "packages",
                "platform",
                "python",
                "selected_gpu",
                "xla_python_client_preallocate",
            },
            field="runtime",
        )
        if (
            runtime["cuda_device_order"] != "PCI_BUS_ID"
            or runtime["xla_python_client_preallocate"] != "false"
            or type(runtime["cuda_visible_devices_configured"]) is not bool
            or runtime["platform"] != "Linux"
            or runtime["machine"] != "x86_64"
            or not isinstance(runtime["python"], str)
            or re.fullmatch(r"3\.11(?:\.[0-9]+)?", runtime["python"]) is None
            or not isinstance(runtime["kernel"], str)
            or not runtime["kernel"]
        ):
            raise ControllerBenchmarkProtocolError("runtime does not identify the formal GPU path")
        packages = runtime["packages"]
        expected_packages = {
            "controller-learning",
            "jax",
            "jaxlib",
            "matplotlib",
            "mujoco",
            "mujoco-mjx",
            "numpy",
            "torch",
            "warp-lang",
        }
        if (
            not isinstance(packages, Mapping)
            or set(packages) != expected_packages
            or any(not isinstance(value, str) or not value for value in packages.values())
        ):
            raise ControllerBenchmarkProtocolError("runtime package inventory is incomplete")
        gpu = runtime["selected_gpu"]
        if not isinstance(gpu, Mapping):
            raise ControllerBenchmarkProtocolError("selected_gpu must be an object")
        _exact_keys(
            gpu,
            {"driver_version", "index", "memory_total_mib", "name", "uuid"},
            field="runtime.selected_gpu",
        )
        if (
            type(gpu["index"]) is not int
            or gpu["index"] != 0
            or not isinstance(gpu["name"], str)
            or not gpu["name"]
            or not isinstance(gpu["driver_version"], str)
            or not gpu["driver_version"]
            or not isinstance(gpu["uuid"], str)
            or re.fullmatch(
                r"GPU-[0-9a-fA-F]{8}(?:-[0-9a-fA-F]{4}){3}-[0-9a-fA-F]{12}",
                gpu["uuid"],
            )
            is None
            or _finite_number(gpu["memory_total_mib"], field="memory_total_mib") <= 0.0
        ):
            raise ControllerBenchmarkProtocolError("selected GPU evidence is invalid")
        device = runtime["jax_device"]
        if not isinstance(device, Mapping):
            raise ControllerBenchmarkProtocolError("jax_device must be an object")
        _exact_keys(device, {"device_kind", "id", "platform"}, field="runtime.jax_device")
        if (
            type(device["id"]) is not int
            or device["id"] != 0
            or device["platform"] != "gpu"
            or not isinstance(device["device_kind"], str)
            or not device["device_kind"]
        ):
            raise ControllerBenchmarkProtocolError("JAX GPU device evidence is invalid")

        source = report["source"]
        if not isinstance(source, Mapping):
            raise ControllerBenchmarkProtocolError("source must be an object")
        _exact_keys(
            source,
            {"input_sha256_after", "input_sha256_before", "post_output_worktree", "preflight"},
            field="source",
        )
        before = source["input_sha256_before"]
        after = source["input_sha256_after"]
        if (
            not isinstance(before, Mapping)
            or not isinstance(after, Mapping)
            or dict(before) != dict(after)
        ):
            raise ControllerBenchmarkProtocolError("formal inputs changed during evaluation")
        if not before or any(_SHA256_PATTERN.fullmatch(value) is None for value in before.values()):
            raise ControllerBenchmarkProtocolError("input hash coverage is invalid")
        if set(before) != set(records) or any(
            before[name] != record["sha256"] for name, record in records.items()
        ):
            raise ControllerBenchmarkProtocolError("input hashes differ from artifact records")
        preflight = source["preflight"]
        if (
            not isinstance(preflight, Mapping)
            or preflight.get("worktree_clean") is not True
            or not isinstance(preflight.get("revision"), str)
            or _SOURCE_REVISION_PATTERN.fullmatch(preflight["revision"]) is None
        ):
            raise ControllerBenchmarkProtocolError("source preflight is invalid")
        _exact_keys(
            preflight,
            {"revision", "worktree_clean"},
            field="source.preflight",
        )
        output = source["post_output_worktree"]
        allowed_payloads = sorted([config.trajectory_path, config.overview_path])
        observed_payloads = (
            output.get("observed_payload_changed_paths") if isinstance(output, Mapping) else None
        )
        if (
            not isinstance(output, Mapping)
            or set(output)
            != {
                "allowed_payload_output_paths",
                "observed_payload_changed_paths",
                "only_allowed_payload_outputs_before_report_write",
                "published_output_bytes_verified",
                "report_change_excluded_from_payload_observation",
                "report_output_path",
                "revision",
                "unexpected_changed_paths",
            }
            or output.get("allowed_payload_output_paths") != allowed_payloads
            or not isinstance(observed_payloads, list)
            or observed_payloads != sorted(set(observed_payloads))
            or not set(observed_payloads) <= set(allowed_payloads)
            or output.get("only_allowed_payload_outputs_before_report_write") is not True
            or output.get("published_output_bytes_verified") is not True
            or output.get("report_change_excluded_from_payload_observation") is not True
            or output.get("report_output_path") != config.report_path
            or output.get("unexpected_changed_paths") != []
            or output.get("revision") != preflight["revision"]
        ):
            raise ControllerBenchmarkProtocolError("post-output worktree evidence differs")

        if report["status"] != "passed":
            raise ControllerBenchmarkProtocolError("report status must be passed")
    except (ControllerBenchmarkProtocolError, KeyError, TypeError, ValueError) as error:
        findings.append(str(error))
    return tuple(findings)


def validate_controller_evaluation_report(
    report: object,
    *,
    config: PpoControllerEvaluationConfig,
) -> None:
    """Raise unless every report aggregate and identity recomputes exactly."""

    findings = controller_evaluation_report_findings(report, config=config)
    if findings:
        raise ControllerBenchmarkProtocolError("; ".join(findings))


__all__ = [
    "CONTROLLER_EVALUATION_CONFIG_SCHEMA_VERSION",
    "CONTROLLER_EVALUATION_REPORT_SCHEMA_VERSION",
    "FORMAL_CONTROLLER_EXECUTION_MODEL",
    "FORMAL_MAX_EPISODE_STEPS",
    "FORMAL_OUTPUT_CRASH_RECOVERY_METHOD",
    "FORMAL_REPLAY_CAPTURE_METHOD",
    "FORMAL_REPLAY_SELECTION_RULE",
    "FORMAL_RESET_SEED_RULE",
    "FORMAL_VALIDATION_TRACK_COUNT",
    "ControllerBenchmarkProtocolError",
    "PpoControllerEvaluationConfig",
    "controller_evaluation_report_findings",
    "episode_to_report_row",
    "evaluation_summary",
    "load_ppo_controller_evaluation_config",
    "replay_track_index",
    "validate_controller_evaluation_report",
]
