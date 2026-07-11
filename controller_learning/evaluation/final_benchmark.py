"""Frozen M8 Test-only comparison protocol and strict report schema."""

from __future__ import annotations

import math
import re
import tomllib
from collections.abc import Mapping
from dataclasses import dataclass, fields
from pathlib import Path, PurePosixPath
from types import MappingProxyType
from typing import Any, Final

M8_FINAL_CONFIG_SCHEMA_VERSION: Final = 2
M8_FINAL_REPORT_SCHEMA_VERSION: Final = "controller-learning.m8-final-evaluation.v2"
M8_FINAL_RUN_ID: Final = "m8-final-v0-1-002"
M8_PREDECESSOR_RUN_ID: Final = "m8-final-v0-1-001"
M8_CONTROLLER_ORDER: Final = ("pid", "mpc", "ppo")
M8_TEST_TRACK_COUNT: Final = 20
M8_TOTAL_EPISODES: Final = len(M8_CONTROLLER_ORDER) * M8_TEST_TRACK_COUNT
M8_RESET_SEED_RULE: Final = "test_row_index_uint32_reused_for_each_controller"
M8_CONTROLLER_SEED_RULE: Final = "numpy_seedsequence_root_reset_seed_world_0_episode_0_domain_1"
M8_TRACK_ORDER_RULE: Final = "official_test_manifest_rows_0_through_19"
M8_CONTROLLER_EXECUTION_MODEL: Final = "ordinary_controller_plugin_runner"
M8_ENVIRONMENT_LIFECYCLE: Final = "one_shared_batch_one_environment_all_60_episodes"
M8_RANKING_RULE: Final = "success_rate_desc_then_mean_successful_lap_time_asc"
M8_CONTROLLER_EXCEPTION_POLICY: Final = (
    "invalidate_attempt_preserve_sanitized_evidence_no_automatic_retry"
)
M8_CONTROLLER_INIT_LIMIT_POLICY: Final = "record_soft_diagnostic_and_continue"
M8_REPLAY_CAPTURE_METHOD: Final = "record_all_canonical_episodes_retain_predeclared_row_zero"
M8_ACCEPTED_RESULT_RULE: Final = "first_complete_protocol_passing_replacement_attempt"
M8_REPLACEMENT_ELIGIBILITY_RULE: Final = (
    "predecessor_test_bound_zero_journal_null_execution_evidence_only_canonical_"
    "environment_create_workload_null_failure_no_outputs_seal_or_staged_artifacts"
)
M8_REPLACEMENT_FAILURE_REPORT_PATH: Final = "benchmarks/v0.1/m8_attempt_001_failure_report.json"
M8_ATTEMPT_001_FAILURE_REPORT_SHA256: Final = (
    "60bdb6d038b27867b13e1a12455b46e6717d1840bff65f1e072de06692645235"
)
M8_PRE_TEST_INITIALIZATION: Final = "warp.init_before_test_bind"
M8_METRIC_SAMPLE_RULES: Final = MappingProxyType(
    {
        "aggregate_weighting": "transition_weighted",
        "lateral_error_sample": "post_step_topology_local_centerline_projection",
        "saturation_sample": "requested_action_strictly_outside_physical_bounds",
        "smoothness_sample": (
            "successive_requested_action_delta_per_second_excluding_first_action"
        ),
        "speed_sample": "post_step_body_velocity_norm",
    }
)

_SHA256_PATTERN = re.compile(r"^[0-9a-f]{64}$")
_REVISION_PATTERN = re.compile(r"^[0-9a-f]{40}$")


class FinalBenchmarkProtocolError(ValueError):
    """The frozen M8 configuration or a final report violates its strict schema."""


def _exact_keys(value: Mapping[str, Any], expected: set[str], *, field: str) -> None:
    if any(type(key) is not str for key in value) or set(value) != expected:
        raise FinalBenchmarkProtocolError(
            f"{field} keys differ; missing={sorted(expected - set(value))}, "
            f"extra={sorted(set(value) - expected)}"
        )


def _table(value: Mapping[str, Any], key: str) -> Mapping[str, Any]:
    result = value.get(key)
    if not isinstance(result, Mapping):
        raise FinalBenchmarkProtocolError(f"{key} must be a TOML table")
    return result


def _plain_integer(value: object, *, field: str, minimum: int = 0) -> int:
    if type(value) is not int or value < minimum:
        raise FinalBenchmarkProtocolError(
            f"{field} must be an integer greater than or equal to {minimum}"
        )
    return value


def _plain_boolean(value: object, *, field: str) -> bool:
    if type(value) is not bool:
        raise FinalBenchmarkProtocolError(f"{field} must be a boolean")
    return value


def _plain_float(value: object, *, field: str) -> float:
    if type(value) is not float or not math.isfinite(value):
        raise FinalBenchmarkProtocolError(f"{field} must be a finite TOML float")
    return value


def _safe_relative_path(value: object, *, field: str, suffix: str | None = None) -> str:
    if not isinstance(value, str) or not value or "\\" in value:
        raise FinalBenchmarkProtocolError(f"{field} must be a normalized relative POSIX path")
    path = PurePosixPath(value)
    if (
        path.is_absolute()
        or path.as_posix() != value
        or any(part in {"", ".", ".."} for part in path.parts)
    ):
        raise FinalBenchmarkProtocolError(f"{field} must be a normalized relative POSIX path")
    if suffix is not None and path.suffix != suffix:
        raise FinalBenchmarkProtocolError(f"{field} must use the {suffix} suffix")
    return value


def _expected_controller_directory(name: str) -> str:
    if name not in M8_CONTROLLER_ORDER:
        raise FinalBenchmarkProtocolError(f"unknown final Controller {name!r}")
    return f"controllers/{name}"


@dataclass(frozen=True, slots=True)
class M8FinalEvaluationConfig:
    """Exact immutable inputs, rules, and outputs for the authorized replacement Test run."""

    run_id: str
    benchmark_version: str
    level_id: int
    backend: str
    test_track_count: int
    controller_order: tuple[str, ...]
    reset_seed_rule: str
    controller_seed_rule: str
    track_order_rule: str
    controller_execution_model: str
    environment_lifecycle: str
    max_episode_steps: int
    environment_instances: int
    fresh_controller_per_episode: bool
    fresh_controller_instance_count: int
    same_track_and_reset_seed_for_each_controller: bool
    controller_frequency_hz: int
    compute_deadline_s: float
    controller_init_soft_limit_s: float
    realtime_p99_limit_s: float
    realtime_miss_rate_limit: float
    realtime_qualification_required_for_pass: bool
    controller_exception_policy: str
    controller_init_limit_exceeded_policy: str
    ranking_rule: str
    test_informed_tuning_allowed: bool
    success_rate_pass_gate: bool
    projection_backward_segments: int
    projection_forward_segments: int
    speed_sample: str
    lateral_error_sample: str
    saturation_sample: str
    smoothness_sample: str
    aggregate_weighting: str
    control_dt_s: float
    steering_lower_rad: float
    steering_upper_rad: float
    longitudinal_lower_mps2: float
    longitudinal_upper_mps2: float
    replay_test_row_index: int
    replay_capture_method: str
    replay_environment_instances: int
    accepted_result: str
    automatic_retry_after_test_bound: bool
    performance_outcome_can_trigger_retry: bool
    completed_workload_can_only_finalize_from_durable_journal_and_execution_evidence: bool
    replacement_authorized: bool
    replacement_of_run_id: str
    replacement_attempt_limit: int
    replacement_eligibility_rule: str
    replacement_failure_report_path: str
    replacement_failure_report_sha256: str
    pre_test_initialization: str
    third_attempt_allowed: bool
    controller_directories: Mapping[str, str]
    controller_aggregate_sha256: Mapping[str, str]
    controller_config_sha256: Mapping[str, str]
    test_manifest_sha256: str
    test_asset_sha256: str
    input_paths: Mapping[str, str]
    results_root: str
    report_path: str
    comparison_csv_path: str
    comparison_png_path: str
    schema_version: int = M8_FINAL_CONFIG_SCHEMA_VERSION

    def __post_init__(self) -> None:
        if (
            type(self.schema_version) is not int
            or self.schema_version != M8_FINAL_CONFIG_SCHEMA_VERSION
        ):
            raise FinalBenchmarkProtocolError(
                f"final config schema_version must be exactly {M8_FINAL_CONFIG_SCHEMA_VERSION}"
            )
        if self.run_id != M8_FINAL_RUN_ID:
            raise FinalBenchmarkProtocolError(f"run_id must be exactly {M8_FINAL_RUN_ID!r}")
        expected_scalars = {
            "benchmark_version": (self.benchmark_version, "0.1"),
            "backend": (self.backend, "mjx_warp"),
            "reset_seed_rule": (self.reset_seed_rule, M8_RESET_SEED_RULE),
            "controller_seed_rule": (self.controller_seed_rule, M8_CONTROLLER_SEED_RULE),
            "track_order_rule": (self.track_order_rule, M8_TRACK_ORDER_RULE),
            "controller_execution_model": (
                self.controller_execution_model,
                M8_CONTROLLER_EXECUTION_MODEL,
            ),
            "environment_lifecycle": (
                self.environment_lifecycle,
                M8_ENVIRONMENT_LIFECYCLE,
            ),
            "ranking_rule": (self.ranking_rule, M8_RANKING_RULE),
            "controller_exception_policy": (
                self.controller_exception_policy,
                M8_CONTROLLER_EXCEPTION_POLICY,
            ),
            "controller_init_limit_exceeded_policy": (
                self.controller_init_limit_exceeded_policy,
                M8_CONTROLLER_INIT_LIMIT_POLICY,
            ),
            "replay_capture_method": (
                self.replay_capture_method,
                M8_REPLAY_CAPTURE_METHOD,
            ),
            "accepted_result": (self.accepted_result, M8_ACCEPTED_RESULT_RULE),
            "replacement_of_run_id": (
                self.replacement_of_run_id,
                M8_PREDECESSOR_RUN_ID,
            ),
            "replacement_eligibility_rule": (
                self.replacement_eligibility_rule,
                M8_REPLACEMENT_ELIGIBILITY_RULE,
            ),
            "pre_test_initialization": (
                self.pre_test_initialization,
                M8_PRE_TEST_INITIALIZATION,
            ),
            **{
                field: (getattr(self, field), expected)
                for field, expected in M8_METRIC_SAMPLE_RULES.items()
            },
        }
        for field, (actual, expected) in expected_scalars.items():
            if actual != expected:
                raise FinalBenchmarkProtocolError(f"{field} must be exactly {expected!r}")
        exact_integers = {
            "level_id": (self.level_id, 1),
            "test_track_count": (self.test_track_count, M8_TEST_TRACK_COUNT),
            "max_episode_steps": (self.max_episode_steps, 4000),
            "environment_instances": (self.environment_instances, 1),
            "controller_frequency_hz": (self.controller_frequency_hz, 20),
            "fresh_controller_instance_count": (
                self.fresh_controller_instance_count,
                M8_TOTAL_EPISODES,
            ),
            "projection_backward_segments": (self.projection_backward_segments, 4),
            "projection_forward_segments": (self.projection_forward_segments, 12),
            "replay_test_row_index": (self.replay_test_row_index, 0),
            "replay_environment_instances": (self.replay_environment_instances, 0),
            "replacement_attempt_limit": (self.replacement_attempt_limit, 1),
        }
        for field, (actual, expected) in exact_integers.items():
            if type(actual) is not int or actual != expected:
                raise FinalBenchmarkProtocolError(f"{field} must be exactly {expected}")
        if tuple(self.controller_order) != M8_CONTROLLER_ORDER:
            raise FinalBenchmarkProtocolError(
                f"controller_order must be exactly {list(M8_CONTROLLER_ORDER)!r}"
            )
        for field in (
            "fresh_controller_per_episode",
            "same_track_and_reset_seed_for_each_controller",
            "completed_workload_can_only_finalize_from_durable_journal_and_execution_evidence",
            "replacement_authorized",
        ):
            if getattr(self, field) is not True:
                raise FinalBenchmarkProtocolError(f"{field} must be true")
        for field in (
            "automatic_retry_after_test_bound",
            "performance_outcome_can_trigger_retry",
            "test_informed_tuning_allowed",
            "success_rate_pass_gate",
            "realtime_qualification_required_for_pass",
            "third_attempt_allowed",
        ):
            if getattr(self, field) is not False:
                raise FinalBenchmarkProtocolError(f"{field} must be false")

        if (
            _safe_relative_path(
                self.replacement_failure_report_path,
                field="replacement_failure_report_path",
                suffix=".json",
            )
            != M8_REPLACEMENT_FAILURE_REPORT_PATH
        ):
            raise FinalBenchmarkProtocolError(
                "replacement_failure_report_path must be exactly "
                f"{M8_REPLACEMENT_FAILURE_REPORT_PATH!r}"
            )
        if (
            _sha256(
                self.replacement_failure_report_sha256,
                field="replacement_failure_report_sha256",
            )
            != M8_ATTEMPT_001_FAILURE_REPORT_SHA256
        ):
            raise FinalBenchmarkProtocolError("attempt 001 failure report hash differs")
        exact_floats = {
            "compute_deadline_s": (self.compute_deadline_s, 0.05),
            "controller_init_soft_limit_s": (self.controller_init_soft_limit_s, 30.0),
            "realtime_p99_limit_s": (self.realtime_p99_limit_s, 0.05),
            "realtime_miss_rate_limit": (self.realtime_miss_rate_limit, 0.01),
            "control_dt_s": (self.control_dt_s, 0.05),
            "steering_lower_rad": (self.steering_lower_rad, -0.6),
            "steering_upper_rad": (self.steering_upper_rad, 0.6),
            "longitudinal_lower_mps2": (self.longitudinal_lower_mps2, -8.0),
            "longitudinal_upper_mps2": (self.longitudinal_upper_mps2, 4.0),
        }
        for field, (actual, expected) in exact_floats.items():
            if type(actual) is not float or actual != expected:
                raise FinalBenchmarkProtocolError(f"{field} must be exactly {expected}")

        directories = dict(self.controller_directories)
        if set(directories) != set(M8_CONTROLLER_ORDER):
            raise FinalBenchmarkProtocolError("controller_directories must cover pid, mpc, and ppo")
        for name in M8_CONTROLLER_ORDER:
            expected = _expected_controller_directory(name)
            actual = _safe_relative_path(directories[name], field=f"controller_directories.{name}")
            if actual != expected:
                raise FinalBenchmarkProtocolError(
                    f"Controller {name!r} directory must be exactly {expected!r}"
                )
        object.__setattr__(self, "controller_directories", MappingProxyType(directories))
        aggregate_hashes = dict(self.controller_aggregate_sha256)
        config_hashes = dict(self.controller_config_sha256)
        if set(aggregate_hashes) != set(M8_CONTROLLER_ORDER) or set(config_hashes) != set(
            M8_CONTROLLER_ORDER
        ):
            raise FinalBenchmarkProtocolError("Controller hashes must cover pid, mpc, and ppo")
        expected_controller_hashes = {
            "pid": (
                "4f9f63eb2b6c0862fcf3c584f73ae25e9a721fac4cf916e738657b3f6c9c0d71",
                "10d661604ad1cab25bb2073d29aafb16003df3cae59026baef10a10e5e737e47",
            ),
            "mpc": (
                "f0a288515b48ec360e72939e65184d58234b491e917c55bb5dd4e9466150c9bb",
                "0aef6eacb4f9882adf0a97d728b210d9c99b09bb694f6792a2ad53d7802281fd",
            ),
            "ppo": (
                "55720b360d6780704135da4670a1ac35cc13045bb11cb4866e256c494be14f2e",
                "ee9f09deb5b55f21df90f234d251b79d6dfcdfaf80f7fbf2b7b488c489acf5dc",
            ),
        }
        for name, (expected_aggregate, expected_config) in expected_controller_hashes.items():
            if _sha256(aggregate_hashes[name], field=f"controllers.{name}.aggregate_sha256") != (
                expected_aggregate
            ):
                raise FinalBenchmarkProtocolError(f"Controller {name!r} aggregate hash differs")
            if _sha256(config_hashes[name], field=f"controllers.{name}.config_sha256") != (
                expected_config
            ):
                raise FinalBenchmarkProtocolError(f"Controller {name!r} config hash differs")
        object.__setattr__(self, "controller_aggregate_sha256", MappingProxyType(aggregate_hashes))
        object.__setattr__(self, "controller_config_sha256", MappingProxyType(config_hashes))

        if _sha256(self.test_manifest_sha256, field="test_manifest_sha256") != (
            "2230e29f3e13029d4ca09de32a703e9a80c070e654386563b9ef4f7a2d197f8b"
        ):
            raise FinalBenchmarkProtocolError("official Test manifest hash differs")
        if _sha256(self.test_asset_sha256, field="test_asset_sha256") != (
            "0d654395630ec0b64952b076a2595de96f3926ea208fac3796a50be37df29c71"
        ):
            raise FinalBenchmarkProtocolError("official Test asset hash differs")

        expected_inputs = {
            "m5_track_admission_report": "benchmarks/v0.1/m5_track_admission_report.json",
            "m6_report": "benchmarks/v0.1/m6_controller_report.json",
            "m7_selection_report": "benchmarks/v0.1/m7_ppo_selection_report.json",
            "m7_export_report": "benchmarks/v0.1/m7_ppo_export_report.json",
            "m7_controller_report": ("benchmarks/v0.1/m7_ppo_controller_evaluation_report.json"),
            "m8_attempt_001_failure_report": M8_REPLACEMENT_FAILURE_REPORT_PATH,
        }
        inputs = dict(self.input_paths)
        if set(inputs) != set(expected_inputs):
            raise FinalBenchmarkProtocolError("input_paths differ from the six frozen reports")
        for name, expected in expected_inputs.items():
            actual = _safe_relative_path(
                inputs[name],
                field=f"input_paths.{name}",
                suffix=".json",
            )
            if actual != expected:
                raise FinalBenchmarkProtocolError(f"input path {name!r} differs")
        if inputs["m8_attempt_001_failure_report"] != self.replacement_failure_report_path:
            raise FinalBenchmarkProtocolError(
                "replacement failure report must be the frozen M8 input report"
            )
        object.__setattr__(self, "input_paths", MappingProxyType(inputs))

        expected_outputs = {
            "results_root": (self.results_root, "results/0.1", None),
            "report_path": (
                self.report_path,
                "benchmarks/v0.1/m8_final_evaluation_report.json",
                ".json",
            ),
            "comparison_csv_path": (
                self.comparison_csv_path,
                "benchmarks/v0.1/m8_final_results.csv",
                ".csv",
            ),
            "comparison_png_path": (
                self.comparison_png_path,
                "benchmarks/v0.1/m8_test_row_000_comparison.png",
                ".png",
            ),
        }
        for field, (actual, expected, suffix) in expected_outputs.items():
            if _safe_relative_path(actual, field=field, suffix=suffix) != expected:
                raise FinalBenchmarkProtocolError(f"{field} must be exactly {expected!r}")

    def to_dict(self) -> dict[str, Any]:
        """Return a stable JSON-compatible configuration mapping."""

        value = {field.name: getattr(self, field.name) for field in fields(self)}
        value["controller_order"] = list(self.controller_order)
        value["controller_directories"] = dict(self.controller_directories)
        value["controller_aggregate_sha256"] = dict(self.controller_aggregate_sha256)
        value["controller_config_sha256"] = dict(self.controller_config_sha256)
        value["input_paths"] = dict(self.input_paths)
        return value


def load_m8_final_evaluation_config(path: str | Path) -> M8FinalEvaluationConfig:
    """Load the only accepted M8 TOML with exact tables, keys, and scalar types."""

    source = Path(path)
    if source.suffix != ".toml" or source.is_symlink() or not source.is_file():
        raise FinalBenchmarkProtocolError("final evaluation config must be a regular TOML file")
    try:
        with source.open("rb") as file:
            data = tomllib.load(file)
    except (OSError, tomllib.TOMLDecodeError) as error:
        raise FinalBenchmarkProtocolError("final evaluation config is invalid") from error
    _exact_keys(
        data,
        {
            "schema_version",
            "run_id",
            "protocol",
            "metrics",
            "replay",
            "retry",
            "replacement",
            "controllers",
            "test_assets",
            "inputs",
            "outputs",
        },
        field="config",
    )
    protocol = _table(data, "protocol")
    metrics = _table(data, "metrics")
    replay = _table(data, "replay")
    retry = _table(data, "retry")
    replacement = _table(data, "replacement")
    controllers = _table(data, "controllers")
    test_assets = _table(data, "test_assets")
    inputs = _table(data, "inputs")
    outputs = _table(data, "outputs")
    _exact_keys(
        protocol,
        {
            "benchmark_version",
            "level_id",
            "backend",
            "test_track_count",
            "controller_order",
            "reset_seed_rule",
            "controller_seed_rule",
            "track_order_rule",
            "controller_execution_model",
            "environment_lifecycle",
            "max_episode_steps",
            "environment_instances",
            "fresh_controller_per_episode",
            "fresh_controller_instance_count",
            "same_track_and_reset_seed_for_each_controller",
            "controller_frequency_hz",
            "compute_deadline_s",
            "controller_init_soft_limit_s",
            "realtime_p99_limit_s",
            "realtime_miss_rate_limit",
            "realtime_qualification_required_for_pass",
            "controller_exception_policy",
            "controller_init_limit_exceeded_policy",
            "ranking_rule",
            "test_informed_tuning_allowed",
            "success_rate_pass_gate",
        },
        field="protocol",
    )
    _exact_keys(
        metrics,
        set(M8_METRIC_SAMPLE_RULES)
        | {
            "projection_backward_segments",
            "projection_forward_segments",
            "control_dt_s",
            "steering_lower_rad",
            "steering_upper_rad",
            "longitudinal_lower_mps2",
            "longitudinal_upper_mps2",
        },
        field="metrics",
    )
    _exact_keys(
        replay,
        {"test_row_index", "capture_method", "replay_environment_instances"},
        field="replay",
    )
    _exact_keys(
        retry,
        {
            "accepted_result",
            "automatic_retry_after_test_bound",
            "performance_outcome_can_trigger_retry",
            "completed_workload_can_only_finalize_from_durable_journal_and_execution_evidence",
        },
        field="retry",
    )
    _exact_keys(
        replacement,
        {
            "authorized",
            "replacement_of_run_id",
            "replacement_attempt_limit",
            "eligibility_rule",
            "failure_report_path",
            "failure_report_sha256",
            "pre_test_initialization",
            "third_attempt_allowed",
        },
        field="replacement",
    )
    _exact_keys(controllers, set(M8_CONTROLLER_ORDER), field="controllers")
    controller_directories: dict[str, str] = {}
    for name in M8_CONTROLLER_ORDER:
        table = _table(controllers, name)
        _exact_keys(
            table,
            {"directory", "aggregate_sha256", "config_sha256"},
            field=f"controllers.{name}",
        )
        controller_directories[name] = table["directory"]
    controller_aggregate_sha256 = {
        name: _table(controllers, name)["aggregate_sha256"] for name in M8_CONTROLLER_ORDER
    }
    controller_config_sha256 = {
        name: _table(controllers, name)["config_sha256"] for name in M8_CONTROLLER_ORDER
    }
    _exact_keys(test_assets, {"manifest_sha256", "asset_sha256"}, field="test_assets")
    _exact_keys(
        inputs,
        {
            "m5_track_admission_report",
            "m6_report",
            "m7_selection_report",
            "m7_export_report",
            "m7_controller_report",
            "m8_attempt_001_failure_report",
        },
        field="inputs",
    )
    _exact_keys(
        outputs,
        {"results_root", "report_path", "comparison_csv_path", "comparison_png_path"},
        field="outputs",
    )
    order = protocol["controller_order"]
    if not isinstance(order, list) or any(type(value) is not str for value in order):
        raise FinalBenchmarkProtocolError("protocol.controller_order must be an array of strings")
    return M8FinalEvaluationConfig(
        schema_version=_plain_integer(data["schema_version"], field="schema_version"),
        run_id=data["run_id"],
        benchmark_version=protocol["benchmark_version"],
        level_id=_plain_integer(protocol["level_id"], field="protocol.level_id"),
        backend=protocol["backend"],
        test_track_count=_plain_integer(
            protocol["test_track_count"], field="protocol.test_track_count", minimum=1
        ),
        controller_order=tuple(order),
        reset_seed_rule=protocol["reset_seed_rule"],
        controller_seed_rule=protocol["controller_seed_rule"],
        track_order_rule=protocol["track_order_rule"],
        controller_execution_model=protocol["controller_execution_model"],
        environment_lifecycle=protocol["environment_lifecycle"],
        max_episode_steps=_plain_integer(
            protocol["max_episode_steps"], field="protocol.max_episode_steps", minimum=1
        ),
        environment_instances=_plain_integer(
            protocol["environment_instances"],
            field="protocol.environment_instances",
            minimum=1,
        ),
        fresh_controller_per_episode=_plain_boolean(
            protocol["fresh_controller_per_episode"],
            field="protocol.fresh_controller_per_episode",
        ),
        fresh_controller_instance_count=_plain_integer(
            protocol["fresh_controller_instance_count"],
            field="protocol.fresh_controller_instance_count",
            minimum=1,
        ),
        same_track_and_reset_seed_for_each_controller=_plain_boolean(
            protocol["same_track_and_reset_seed_for_each_controller"],
            field="protocol.same_track_and_reset_seed_for_each_controller",
        ),
        controller_frequency_hz=_plain_integer(
            protocol["controller_frequency_hz"],
            field="protocol.controller_frequency_hz",
            minimum=1,
        ),
        compute_deadline_s=_plain_float(
            protocol["compute_deadline_s"], field="protocol.compute_deadline_s"
        ),
        controller_init_soft_limit_s=_plain_float(
            protocol["controller_init_soft_limit_s"],
            field="protocol.controller_init_soft_limit_s",
        ),
        realtime_p99_limit_s=_plain_float(
            protocol["realtime_p99_limit_s"], field="protocol.realtime_p99_limit_s"
        ),
        realtime_miss_rate_limit=_plain_float(
            protocol["realtime_miss_rate_limit"],
            field="protocol.realtime_miss_rate_limit",
        ),
        realtime_qualification_required_for_pass=_plain_boolean(
            protocol["realtime_qualification_required_for_pass"],
            field="protocol.realtime_qualification_required_for_pass",
        ),
        controller_exception_policy=protocol["controller_exception_policy"],
        controller_init_limit_exceeded_policy=protocol["controller_init_limit_exceeded_policy"],
        ranking_rule=protocol["ranking_rule"],
        test_informed_tuning_allowed=_plain_boolean(
            protocol["test_informed_tuning_allowed"],
            field="protocol.test_informed_tuning_allowed",
        ),
        success_rate_pass_gate=_plain_boolean(
            protocol["success_rate_pass_gate"], field="protocol.success_rate_pass_gate"
        ),
        projection_backward_segments=_plain_integer(
            metrics["projection_backward_segments"],
            field="metrics.projection_backward_segments",
        ),
        projection_forward_segments=_plain_integer(
            metrics["projection_forward_segments"],
            field="metrics.projection_forward_segments",
        ),
        speed_sample=metrics["speed_sample"],
        lateral_error_sample=metrics["lateral_error_sample"],
        saturation_sample=metrics["saturation_sample"],
        smoothness_sample=metrics["smoothness_sample"],
        aggregate_weighting=metrics["aggregate_weighting"],
        control_dt_s=_plain_float(metrics["control_dt_s"], field="metrics.control_dt_s"),
        steering_lower_rad=_plain_float(
            metrics["steering_lower_rad"], field="metrics.steering_lower_rad"
        ),
        steering_upper_rad=_plain_float(
            metrics["steering_upper_rad"], field="metrics.steering_upper_rad"
        ),
        longitudinal_lower_mps2=_plain_float(
            metrics["longitudinal_lower_mps2"], field="metrics.longitudinal_lower_mps2"
        ),
        longitudinal_upper_mps2=_plain_float(
            metrics["longitudinal_upper_mps2"], field="metrics.longitudinal_upper_mps2"
        ),
        replay_test_row_index=_plain_integer(
            replay["test_row_index"], field="replay.test_row_index"
        ),
        replay_capture_method=replay["capture_method"],
        replay_environment_instances=_plain_integer(
            replay["replay_environment_instances"],
            field="replay.replay_environment_instances",
        ),
        accepted_result=retry["accepted_result"],
        automatic_retry_after_test_bound=_plain_boolean(
            retry["automatic_retry_after_test_bound"],
            field="retry.automatic_retry_after_test_bound",
        ),
        performance_outcome_can_trigger_retry=_plain_boolean(
            retry["performance_outcome_can_trigger_retry"],
            field="retry.performance_outcome_can_trigger_retry",
        ),
        completed_workload_can_only_finalize_from_durable_journal_and_execution_evidence=(
            _plain_boolean(
                retry[
                    "completed_workload_can_only_finalize_from_durable_journal_and_execution_evidence"
                ],
                field=(
                    "retry."
                    "completed_workload_can_only_finalize_from_durable_journal_and_execution_evidence"
                ),
            )
        ),
        replacement_authorized=_plain_boolean(
            replacement["authorized"],
            field="replacement.authorized",
        ),
        replacement_of_run_id=replacement["replacement_of_run_id"],
        replacement_attempt_limit=_plain_integer(
            replacement["replacement_attempt_limit"],
            field="replacement.replacement_attempt_limit",
            minimum=1,
        ),
        replacement_eligibility_rule=replacement["eligibility_rule"],
        replacement_failure_report_path=replacement["failure_report_path"],
        replacement_failure_report_sha256=replacement["failure_report_sha256"],
        pre_test_initialization=replacement["pre_test_initialization"],
        third_attempt_allowed=_plain_boolean(
            replacement["third_attempt_allowed"],
            field="replacement.third_attempt_allowed",
        ),
        controller_directories=controller_directories,
        controller_aggregate_sha256=controller_aggregate_sha256,
        controller_config_sha256=controller_config_sha256,
        test_manifest_sha256=test_assets["manifest_sha256"],
        test_asset_sha256=test_assets["asset_sha256"],
        input_paths=dict(inputs),
        results_root=outputs["results_root"],
        report_path=outputs["report_path"],
        comparison_csv_path=outputs["comparison_csv_path"],
        comparison_png_path=outputs["comparison_png_path"],
    )


def controller_output_paths(
    config: M8FinalEvaluationConfig,
    controller: str,
) -> Mapping[str, str]:
    """Return the seven frozen output paths for one Controller."""

    if not isinstance(config, M8FinalEvaluationConfig):
        raise TypeError("config must be an M8FinalEvaluationConfig")
    if controller not in M8_CONTROLLER_ORDER:
        raise FinalBenchmarkProtocolError(f"unknown final Controller {controller!r}")
    base = f"{config.results_root}/{controller}/{config.run_id}"
    return MappingProxyType(
        {
            "metrics": f"{base}/metrics.npz",
            "replay_trajectory": f"{base}/selected_replays/test_row_000_trajectory.json",
            "results": f"{base}/results.csv",
            "run_manifest": f"{base}/run_manifest.json",
            "summary": f"{base}/summary.json",
            "telemetry": f"{base}/telemetry.png",
            "trajectory": f"{base}/trajectory.png",
        }
    )


def formal_output_paths(config: M8FinalEvaluationConfig) -> tuple[str, ...]:
    """Return the exact sorted allowlist for one formal M8 publication."""

    if not isinstance(config, M8FinalEvaluationConfig):
        raise TypeError("config must be an M8FinalEvaluationConfig")
    paths = {config.report_path, config.comparison_csv_path, config.comparison_png_path}
    for controller in M8_CONTROLLER_ORDER:
        paths.update(controller_output_paths(config, controller).values())
    if len(paths) != 24:
        raise RuntimeError("formal M8 output allowlist must contain exactly 24 unique paths")
    return tuple(sorted(paths))


def validate_formal_output_tree(
    project_root: str | Path,
    config: M8FinalEvaluationConfig,
    *,
    expected_present: bool,
) -> tuple[str, ...]:
    """Reject symlinks, missing outputs, and residue outside the exact 24-file allowlist."""

    if not isinstance(config, M8FinalEvaluationConfig):
        raise TypeError("config must be an M8FinalEvaluationConfig")
    if type(expected_present) is not bool:
        raise TypeError("expected_present must be a boolean")
    root = Path(project_root).resolve(strict=True)
    allowlist = formal_output_paths(config)
    observed: set[str] = set()
    for relative in allowlist:
        candidate = root / relative
        if candidate.is_symlink():
            raise FinalBenchmarkProtocolError(f"formal output cannot be a symlink: {relative}")
        if candidate.exists():
            if not candidate.is_file():
                raise FinalBenchmarkProtocolError(
                    f"formal output must be a regular file: {relative}"
                )
            observed.add(relative)

    for controller in M8_CONTROLLER_ORDER:
        run_directory = root / config.results_root / controller / config.run_id
        if run_directory.is_symlink():
            raise FinalBenchmarkProtocolError("formal result run directory cannot be a symlink")
        if not run_directory.exists():
            continue
        if not run_directory.is_dir():
            raise FinalBenchmarkProtocolError("formal result run path must be a directory")
        for candidate in run_directory.rglob("*"):
            relative = candidate.relative_to(root).as_posix()
            if candidate.is_symlink():
                raise FinalBenchmarkProtocolError(
                    f"formal result tree contains symlink: {relative}"
                )
            if candidate.is_dir():
                continue
            if not candidate.is_file() or relative not in allowlist:
                raise FinalBenchmarkProtocolError(
                    f"formal result tree contains unallowlisted residue: {relative}"
                )

    expected = set(allowlist) if expected_present else set()
    if observed != expected:
        raise FinalBenchmarkProtocolError(
            f"formal output presence differs; missing={sorted(expected - observed)}, "
            f"unexpected={sorted(observed - expected)}"
        )
    return tuple(sorted(observed))


def rank_controller_summaries(values: Mapping[str, Mapping[str, Any]]) -> tuple[str, ...]:
    """Apply success-rate then mean-lap ranking without inventing a combined score."""

    if not isinstance(values, Mapping) or set(values) != set(M8_CONTROLLER_ORDER):
        raise FinalBenchmarkProtocolError("summaries must contain exactly pid, mpc, and ppo")
    rows: list[tuple[str, float, float]] = []
    for name in M8_CONTROLLER_ORDER:
        summary = values[name]
        if not isinstance(summary, Mapping):
            raise FinalBenchmarkProtocolError(f"summary {name!r} must be an object")
        rate = summary.get("success_rate")
        lap = summary.get("mean_successful_lap_time_s")
        if isinstance(rate, bool) or not isinstance(rate, (int, float)):
            raise FinalBenchmarkProtocolError(f"summary {name!r} success_rate must be numeric")
        rate_value = float(rate)
        if not math.isfinite(rate_value) or not 0.0 <= rate_value <= 1.0:
            raise FinalBenchmarkProtocolError(f"summary {name!r} success_rate is invalid")
        if lap is None:
            lap_value = math.inf
        elif isinstance(lap, bool) or not isinstance(lap, (int, float)):
            raise FinalBenchmarkProtocolError(
                f"summary {name!r} mean_successful_lap_time_s must be numeric or null"
            )
        else:
            lap_value = float(lap)
            if not math.isfinite(lap_value) or lap_value <= 0.0:
                raise FinalBenchmarkProtocolError(f"summary {name!r} mean lap is invalid")
        rows.append((name, rate_value, lap_value))
    return tuple(
        name
        for name, _rate, _lap in sorted(
            rows,
            key=lambda value: (-value[1], value[2], M8_CONTROLLER_ORDER.index(value[0])),
        )
    )


def _sha256(value: object, *, field: str) -> str:
    if not isinstance(value, str) or _SHA256_PATTERN.fullmatch(value) is None:
        raise FinalBenchmarkProtocolError(f"{field} must be a lowercase SHA-256 digest")
    return value


def _source_snapshot(value: object, *, field: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise FinalBenchmarkProtocolError(f"{field} must be an object")
    _exact_keys(value, {"revision", "worktree_clean"}, field=field)
    revision = value["revision"]
    if (
        not isinstance(revision, str)
        or _REVISION_PATTERN.fullmatch(revision) is None
        or value["worktree_clean"] is not True
    ):
        raise FinalBenchmarkProtocolError(f"{field} must identify one clean full revision")
    return value


__all__ = [
    "M8_ACCEPTED_RESULT_RULE",
    "M8_ATTEMPT_001_FAILURE_REPORT_SHA256",
    "M8_CONTROLLER_EXCEPTION_POLICY",
    "M8_CONTROLLER_EXECUTION_MODEL",
    "M8_CONTROLLER_INIT_LIMIT_POLICY",
    "M8_CONTROLLER_ORDER",
    "M8_CONTROLLER_SEED_RULE",
    "M8_ENVIRONMENT_LIFECYCLE",
    "M8_FINAL_CONFIG_SCHEMA_VERSION",
    "M8_FINAL_REPORT_SCHEMA_VERSION",
    "M8_FINAL_RUN_ID",
    "M8_METRIC_SAMPLE_RULES",
    "M8_PREDECESSOR_RUN_ID",
    "M8_PRE_TEST_INITIALIZATION",
    "M8_RANKING_RULE",
    "M8_REPLACEMENT_ELIGIBILITY_RULE",
    "M8_REPLACEMENT_FAILURE_REPORT_PATH",
    "M8_REPLAY_CAPTURE_METHOD",
    "M8_RESET_SEED_RULE",
    "M8_TEST_TRACK_COUNT",
    "M8_TOTAL_EPISODES",
    "M8_TRACK_ORDER_RULE",
    "FinalBenchmarkProtocolError",
    "M8FinalEvaluationConfig",
    "controller_output_paths",
    "formal_output_paths",
    "load_m8_final_evaluation_config",
    "rank_controller_summaries",
    "validate_formal_output_tree",
]
