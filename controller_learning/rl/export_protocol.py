"""Strict, asset-free protocol for the one-time M7 PPO Controller export."""

from __future__ import annotations

import re
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any, Final

from controller_learning.rl.artifacts import ArtifactRecord, TrainingRunIdentity
from controller_learning.rl.numpy_actor import NUMPY_ACTOR_SCHEMA_VERSION
from controller_learning.rl.selection import SELECTION_REPORT_SCHEMA_VERSION

EXPORT_REPORT_SCHEMA_VERSION: Final = "controller-learning.m7-ppo-controller-export.v1"
EXPORT_REPORT_PATH: Final = "benchmarks/v0.1/m7_ppo_export_report.json"
PPO_CONTROLLER_DIRECTORY: Final = "controllers/ppo"

_SHA256_PATTERN = re.compile(r"^[0-9a-f]{64}$")
_SOURCE_REVISION_PATTERN = re.compile(r"^[0-9a-f]{40}$")


class ExportProtocolError(ValueError):
    """An M7 selection/export document violates the frozen binding protocol."""


def _exact_keys(value: Mapping[str, Any], expected: set[str], *, field: str) -> None:
    if any(type(key) is not str for key in value) or set(value) != expected:
        raise ExportProtocolError(
            f"{field} keys differ; missing={sorted(expected - set(value))}, "
            f"extra={sorted(set(value) - expected)}"
        )


def _mapping(value: object, *, field: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ExportProtocolError(f"{field} must be an object")
    return value


def _plain_positive_integer(value: object, *, field: str) -> int:
    if type(value) is not int or value < 1:
        raise ExportProtocolError(f"{field} must be a positive integer")
    return value


def _sha256(value: object, *, field: str) -> str:
    if not isinstance(value, str) or _SHA256_PATTERN.fullmatch(value) is None:
        raise ExportProtocolError(f"{field} must be a lowercase SHA-256 digest")
    return value


def _artifact(value: object, *, field: str, expected_path: str) -> ArtifactRecord:
    mapping = _mapping(value, field=field)
    _exact_keys(
        mapping,
        {"relative_path", "schema_version", "sha256", "size_bytes"},
        field=field,
    )
    try:
        record = ArtifactRecord(**dict(mapping))
    except (TypeError, ValueError) as error:
        raise ExportProtocolError(f"{field} is not a valid artifact record") from error
    if record.relative_path != expected_path:
        raise ExportProtocolError(f"{field} path must be exactly {expected_path!r}")
    return record


def _inference_policy(value: object, *, field: str) -> dict[str, str | int]:
    mapping = _mapping(value, field=field)
    _exact_keys(mapping, {"schema_version", "sha256", "size_bytes"}, field=field)
    schema_version = mapping["schema_version"]
    if type(schema_version) is not int or schema_version != NUMPY_ACTOR_SCHEMA_VERSION:
        raise ExportProtocolError(f"{field}.schema_version must be {NUMPY_ACTOR_SCHEMA_VERSION}")
    return {
        "schema_version": schema_version,
        "sha256": _sha256(mapping["sha256"], field=f"{field}.sha256"),
        "size_bytes": _plain_positive_integer(mapping["size_bytes"], field=f"{field}.size_bytes"),
    }


@dataclass(frozen=True, slots=True)
class SelectedExportCandidate:
    """The exact selected checkpoint and inference bytes declared by Validation."""

    checkpoint: ArtifactRecord
    inference_policy: Mapping[str, str | int]
    parameter_sha256: str
    update_index: int
    valid_transitions: int
    vector_steps: int

    def __post_init__(self) -> None:
        if not isinstance(self.checkpoint, ArtifactRecord):
            raise ExportProtocolError("selected checkpoint must be an ArtifactRecord")
        policy = _inference_policy(self.inference_policy, field="selected inference_policy")
        object.__setattr__(self, "inference_policy", policy)
        object.__setattr__(
            self,
            "parameter_sha256",
            _sha256(self.parameter_sha256, field="selected parameter_sha256"),
        )
        for field in ("update_index", "valid_transitions", "vector_steps"):
            object.__setattr__(
                self,
                field,
                _plain_positive_integer(getattr(self, field), field=f"selected {field}"),
            )

    def to_dict(self) -> dict[str, Any]:
        return {
            "checkpoint": self.checkpoint.to_dict(),
            "inference_policy": dict(self.inference_policy),
            "parameter_sha256": self.parameter_sha256,
            "update_index": self.update_index,
            "valid_transitions": self.valid_transitions,
            "vector_steps": self.vector_steps,
        }


def selected_export_candidate(selection_report: object) -> SelectedExportCandidate:
    """Extract one selected candidate only from a passed, internally bound report.

    The caller must first run :func:`controller_learning.rl.selection.validate_selection_report`.
    This function deliberately performs the selected-candidate joins again so the export cannot
    accidentally pair the ranking outcome, evaluated parameters, and retained checkpoint from
    different candidates.
    """

    report = _mapping(selection_report, field="selection report")
    if report.get("schema_version") != SELECTION_REPORT_SCHEMA_VERSION:
        raise ExportProtocolError("selection report schema_version differs")
    gates = _mapping(report.get("gates"), field="selection report gates")
    if report.get("status") != "passed" or gates.get("passed") is not True:
        raise ExportProtocolError("Controller export requires a passed selection gate")
    selection = _mapping(report.get("selection"), field="selection result")
    update = _plain_positive_integer(
        selection.get("selected_update"), field="selection.selected_update"
    )

    training_run = _mapping(report.get("training_run"), field="training_run")
    checkpoints = training_run.get("candidate_checkpoints")
    if not isinstance(checkpoints, list):
        raise ExportProtocolError("training_run.candidate_checkpoints must be an array")
    matching_checkpoints = [item for item in checkpoints if item.get("update_index") == update]
    if len(matching_checkpoints) != 1:
        raise ExportProtocolError("selected update must identify exactly one checkpoint")
    checkpoint_value = _mapping(matching_checkpoints[0], field="selected checkpoint evidence")
    _exact_keys(
        checkpoint_value,
        {
            "checkpoint",
            "inference_policy",
            "parameter_sha256",
            "update_index",
            "valid_transitions",
            "vector_steps",
        },
        field="selected checkpoint evidence",
    )
    configuration = _mapping(report.get("configuration"), field="selection configuration")
    checkpoint_directory = configuration.get("checkpoint_directory")
    if not isinstance(checkpoint_directory, str):
        raise ExportProtocolError("selection checkpoint_directory must be a string")
    checkpoint = _artifact(
        checkpoint_value["checkpoint"],
        field="selected checkpoint record",
        expected_path=f"{checkpoint_directory}/update_{update:08d}.pt",
    )

    evaluations = _mapping(report.get("evaluations"), field="evaluations")
    candidate_evaluations = evaluations.get("candidates")
    if not isinstance(candidate_evaluations, list):
        raise ExportProtocolError("candidate evaluations must be an array")
    matching_evaluations = [
        item for item in candidate_evaluations if item.get("update_index") == update
    ]
    if len(matching_evaluations) != 1:
        raise ExportProtocolError("selected update must identify exactly one evaluation")
    evaluation = _mapping(matching_evaluations[0], field="selected evaluation")
    parameter_sha256 = _sha256(
        checkpoint_value["parameter_sha256"],
        field="selected checkpoint parameter_sha256",
    )
    if (
        evaluation.get("policy_id") != f"checkpoint_update_{update:08d}"
        or evaluation.get("parameter_unchanged") is not True
        or evaluation.get("parameter_sha256_before") != parameter_sha256
        or evaluation.get("parameter_sha256_after") != parameter_sha256
    ):
        raise ExportProtocolError("selected evaluation does not bind the checkpoint parameters")

    return SelectedExportCandidate(
        checkpoint=checkpoint,
        inference_policy=_inference_policy(
            checkpoint_value["inference_policy"], field="selected checkpoint inference_policy"
        ),
        parameter_sha256=parameter_sha256,
        update_index=update,
        valid_transitions=checkpoint_value["valid_transitions"],
        vector_steps=checkpoint_value["vector_steps"],
    )


def _source_snapshot(value: object, *, field: str) -> Mapping[str, Any]:
    mapping = _mapping(value, field=field)
    _exact_keys(mapping, {"revision", "worktree_clean"}, field=field)
    revision = mapping["revision"]
    if (
        not isinstance(revision, str)
        or _SOURCE_REVISION_PATTERN.fullmatch(revision) is None
        or mapping["worktree_clean"] is not True
    ):
        raise ExportProtocolError(f"{field} is not a clean full-revision snapshot")
    return mapping


def validate_export_report(report: object) -> None:
    """Recompute all cross-document and output bindings in one canonical export report."""

    root = _mapping(report, field="export report")
    _exact_keys(
        root,
        {
            "asset_access",
            "controller",
            "input_stability",
            "protocol",
            "schema_version",
            "selection",
            "source",
            "status",
            "training",
        },
        field="export report",
    )
    if root["schema_version"] != EXPORT_REPORT_SCHEMA_VERSION or root["status"] != "passed":
        raise ExportProtocolError("export report must use the frozen schema and passed status")

    protocol = _mapping(root["protocol"], field="protocol")
    expected_protocol = {
        "canonical_inference_policy_verified": True,
        "canonical_selection_report_required": True,
        "exact_published_checkpoint_loader": "v2_explicit_update",
        "formal_export_function": "controller_learning.rl.controller_export.export_ppo_controller",
        "full_parameter_sha256_verified": True,
        "no_gradient_or_optimizer_operations": True,
        "one_time_unfinalized_template_activation": True,
        "passed_selection_gate_required": True,
        "persistent_crash_recovery": {
            "commit_transition": "READY_to_COMMITTED_then_cleanup",
            "exporter_starts_only_after_ready": True,
            "original_config_bytes_and_mode_fsynced": True,
            "startup_ready_action": "restore_config_delete_outputs_then_cleanup",
            "startup_unready_action": "cleanup_staging_only",
            "temporary_file_location": "transaction_staging_only",
            "transaction_directory": "runs/ppo/.m7-controller-export-transaction",
        },
        "selection_outputs_committed_before_export": True,
    }
    if dict(protocol) != expected_protocol:
        raise ExportProtocolError("export protocol differs from the frozen one-time workflow")

    asset_access = _mapping(root["asset_access"], field="asset_access")
    expected_asset_access = {
        "audit_hook_installed_before_project_imports": True,
        "denied_event_count": 0,
        "denied_mutation_event_count": 0,
        "denied_open_event_count": 0,
        "official_track_open_count": 0,
        "official_track_mutation_count": 0,
        "opened_path_categories": [],
        "track_cache_open_count": 0,
        "track_cache_mutation_count": 0,
        "mutation_event_counts": {},
        "unaudited_mutation_wrappers": ["os.mkfifo", "os.mknod"],
    }
    if dict(asset_access) != expected_asset_access:
        raise ExportProtocolError("export asset-access audit evidence differs")

    source = _mapping(root["source"], field="source")
    _exact_keys(source, {"post_export_worktree", "preflight"}, field="source")
    preflight = _source_snapshot(source["preflight"], field="source.preflight")
    post = _mapping(source["post_export_worktree"], field="source.post_export_worktree")
    _exact_keys(
        post,
        {
            "allowed_generated_output_paths",
            "observed_changed_paths",
            "only_allowed_generated_outputs",
            "revision",
            "unexpected_changed_paths",
        },
        field="source.post_export_worktree",
    )
    allowed_outputs = sorted(
        [
            EXPORT_REPORT_PATH,
            f"{PPO_CONTROLLER_DIRECTORY}/config.toml",
            f"{PPO_CONTROLLER_DIRECTORY}/metadata.json",
            f"{PPO_CONTROLLER_DIRECTORY}/policy.npz",
        ]
    )
    if (
        post["allowed_generated_output_paths"] != allowed_outputs
        or post["observed_changed_paths"] != allowed_outputs
        or post["only_allowed_generated_outputs"] is not True
        or post["revision"] != preflight["revision"]
        or post["unexpected_changed_paths"] != []
    ):
        raise ExportProtocolError("post-export worktree evidence differs")

    selection = _mapping(root["selection"], field="selection")
    _exact_keys(
        selection,
        {
            "config",
            "gate_passed",
            "report",
            "report_schema_version",
            "report_status",
            "selected_candidate",
        },
        field="selection",
    )
    selection_config = _artifact(
        selection["config"],
        field="selection.config",
        expected_path="configs/ppo_selection.toml",
    )
    selection_report = _artifact(
        selection["report"],
        field="selection.report",
        expected_path="benchmarks/v0.1/m7_ppo_selection_report.json",
    )
    if (
        selection["gate_passed"] is not True
        or selection["report_schema_version"] != SELECTION_REPORT_SCHEMA_VERSION
        or selection["report_status"] != "passed"
    ):
        raise ExportProtocolError("selection evidence is not a passed frozen report")
    selected_mapping = _mapping(selection["selected_candidate"], field="selected_candidate")
    _exact_keys(
        selected_mapping,
        {
            "checkpoint",
            "inference_policy",
            "parameter_sha256",
            "update_index",
            "valid_transitions",
            "vector_steps",
        },
        field="selected_candidate",
    )
    update = _plain_positive_integer(
        selected_mapping["update_index"], field="selected_candidate.update_index"
    )
    selected = SelectedExportCandidate(
        checkpoint=_artifact(
            selected_mapping["checkpoint"],
            field="selected_candidate.checkpoint",
            expected_path=f"checkpoints/update_{update:08d}.pt",
        ),
        inference_policy=selected_mapping["inference_policy"],
        parameter_sha256=selected_mapping["parameter_sha256"],
        update_index=update,
        valid_transitions=selected_mapping["valid_transitions"],
        vector_steps=selected_mapping["vector_steps"],
    )

    training = _mapping(root["training"], field="training")
    _exact_keys(
        training,
        {"checkpoint_directory", "identity", "run_directory", "training_config"},
        field="training",
    )
    if (
        training["checkpoint_directory"] != "checkpoints"
        or training["run_directory"] != "runs/ppo/m7-formal-v0-1-001"
    ):
        raise ExportProtocolError("training paths differ from the frozen M7 run")
    try:
        identity = TrainingRunIdentity.from_dict(
            _mapping(training["identity"], field="training.identity")
        )
    except (TypeError, ValueError) as error:
        raise ExportProtocolError("training.identity is invalid") from error
    training_config = _artifact(
        training["training_config"],
        field="training.training_config",
        expected_path="configs/ppo.toml",
    )
    if training_config.sha256 != identity.configuration_sha256:
        raise ExportProtocolError("training config does not bind the run identity")

    controller = _mapping(root["controller"], field="controller")
    _exact_keys(
        controller,
        {"artifacts", "checkpoint", "inference_only", "plugin_directory", "runtime"},
        field="controller",
    )
    if (
        controller["plugin_directory"] != PPO_CONTROLLER_DIRECTORY
        or controller["runtime"] != "numpy"
    ):
        raise ExportProtocolError("Controller deployment identity differs")
    if controller["inference_only"] != {
        "contains_environment_state": False,
        "contains_optimizer_state": False,
        "contains_value_network": False,
    }:
        raise ExportProtocolError("Controller inference-only evidence differs")
    artifacts = _mapping(controller["artifacts"], field="controller.artifacts")
    _exact_keys(artifacts, {"config", "metadata", "policy"}, field="controller.artifacts")
    _artifact(
        artifacts["config"],
        field="controller.artifacts.config",
        expected_path=f"{PPO_CONTROLLER_DIRECTORY}/config.toml",
    )
    _artifact(
        artifacts["metadata"],
        field="controller.artifacts.metadata",
        expected_path=f"{PPO_CONTROLLER_DIRECTORY}/metadata.json",
    )
    policy = _artifact(
        artifacts["policy"],
        field="controller.artifacts.policy",
        expected_path=f"{PPO_CONTROLLER_DIRECTORY}/policy.npz",
    )
    if (
        policy.sha256 != selected.inference_policy["sha256"]
        or policy.size_bytes != selected.inference_policy["size_bytes"]
    ):
        raise ExportProtocolError("exported policy differs from selected inference bytes")

    checkpoint = _mapping(controller["checkpoint"], field="controller.checkpoint")
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
    expected_checkpoint = {
        "checkpoint_sha256": selected.checkpoint.sha256,
        "run_id": identity.run_id,
        "source_revision": identity.source_revision,
        "training_configuration_sha256": identity.configuration_sha256,
        "update_index": selected.update_index,
        "valid_transitions": selected.valid_transitions,
        "vector_steps": selected.vector_steps,
    }
    if dict(checkpoint) != expected_checkpoint:
        raise ExportProtocolError("Controller checkpoint identity differs from selection/training")

    stability = _mapping(root["input_stability"], field="input_stability")
    _exact_keys(
        stability,
        {"all_inputs_unchanged", "post_export_sha256", "pre_export_sha256"},
        field="input_stability",
    )
    pre_hashes = _mapping(stability["pre_export_sha256"], field="pre_export_sha256")
    post_hashes = _mapping(stability["post_export_sha256"], field="post_export_sha256")
    expected_hashes = {
        "selected_checkpoint": selected.checkpoint.sha256,
        "selection_config": selection_config.sha256,
        "selection_report": selection_report.sha256,
        "training_config": training_config.sha256,
    }
    if (
        stability["all_inputs_unchanged"] is not True
        or dict(pre_hashes) != expected_hashes
        or dict(post_hashes) != expected_hashes
    ):
        raise ExportProtocolError("formal export inputs changed or differ from artifact records")


__all__ = [
    "EXPORT_REPORT_PATH",
    "EXPORT_REPORT_SCHEMA_VERSION",
    "PPO_CONTROLLER_DIRECTORY",
    "ExportProtocolError",
    "SelectedExportCandidate",
    "selected_export_candidate",
    "validate_export_report",
]
