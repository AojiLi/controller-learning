"""Strict Torch-free runtime loading and one-way PPO Controller finalization."""

from __future__ import annotations

import hashlib
import json
import os
import re
import tempfile
import tomllib
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Final

from controller_learning.rl.configuration import PpoObservationConfig
from controller_learning.rl.numpy_actor import (
    NUMPY_ACTOR_MAX_BYTES,
    NUMPY_ACTOR_SCHEMA_VERSION,
    NumpyActorArtifactError,
    NumpyActorFileEvidence,
    NumpyDeterministicActor,
    load_numpy_actor_npz,
    numpy_actor_from_ppo_state_dict,
    save_numpy_actor_npz,
)
from controller_learning.rl.schema import LOCAL_TRACK_FEATURE_SCHEMA_VERSION

PPO_CONTROLLER_SCHEMA_VERSION: Final = 1
PPO_CONTROLLER_METADATA_SCHEMA_VERSION: Final = 1
PPO_CONTROLLER_ARTIFACT_TYPE: Final = "controller-learning.ppo-controller.v1"
PPO_CONTROLLER_POLICY_FILE: Final = "policy.npz"
PPO_CONTROLLER_METADATA_FILE: Final = "metadata.json"
PPO_CONTROLLER_METADATA_MAX_BYTES: Final = 64 * 1024

_DESCRIPTION = "Torch-free deterministic PPO actor over public local-track features"
_SHA256_PATTERN = re.compile(r"^[0-9a-f]{64}$")
_SOURCE_REVISION_PATTERN = re.compile(r"^(?:[0-9a-f]{40}|[0-9a-f]{64})$")
_RUN_ID_PATTERN = re.compile(r"^[a-z0-9](?:[a-z0-9._-]{0,126}[a-z0-9])?$")


class PpoControllerExportError(RuntimeError):
    """A PPO Controller runtime artifact is unsafe, inconsistent, or incomplete."""


class PpoControllerNotFinalizedError(PpoControllerExportError):
    """The committed source template has not received selected inference weights."""


def _plain_positive_integer(value: object, *, field: str) -> int:
    if type(value) is not int or value < 1:
        raise ValueError(f"{field} must be a positive integer")
    return value


def _schema_version(value: object, *, field: str, expected: int) -> int:
    if type(value) is not int or value != expected:
        raise ValueError(f"{field} must be integer schema version {expected}")
    return value


def _sha256(value: object, *, field: str) -> str:
    if not isinstance(value, str) or _SHA256_PATTERN.fullmatch(value) is None:
        raise ValueError(f"{field} must be a lowercase SHA-256 digest")
    return value


def _exact_keys(value: Mapping[str, Any], expected: set[str], *, field: str) -> None:
    if any(type(key) is not str for key in value):
        raise ValueError(f"{field} must use string keys")
    actual = set(value)
    if actual != expected:
        raise ValueError(
            f"{field} keys differ; missing={sorted(expected - actual)}, "
            f"extra={sorted(actual - expected)}"
        )


def _table(value: Mapping[str, Any], key: str, expected: set[str]) -> Mapping[str, Any]:
    result = value.get(key)
    if not isinstance(result, Mapping):
        raise ValueError(f"PPO Controller field {key!r} must be a table")
    _exact_keys(result, expected, field=f"controller.{key}")
    return result


def _plain_file_name(value: object, *, field: str, expected: str) -> str:
    if not isinstance(value, str) or value != expected or Path(value).name != value:
        raise ValueError(f"{field} must be the local filename {expected!r}")
    return value


@dataclass(frozen=True, slots=True)
class SelectedCheckpointIdentity:
    """Training checkpoint identity retained by the inference-only plugin."""

    run_id: str
    update_index: int
    vector_steps: int
    valid_transitions: int
    checkpoint_sha256: str
    source_revision: str
    training_configuration_sha256: str

    def __post_init__(self) -> None:
        if not isinstance(self.run_id, str) or _RUN_ID_PATTERN.fullmatch(self.run_id) is None:
            raise ValueError("run_id has an invalid format")
        for field in ("update_index", "vector_steps", "valid_transitions"):
            object.__setattr__(
                self,
                field,
                _plain_positive_integer(getattr(self, field), field=field),
            )
        object.__setattr__(
            self,
            "checkpoint_sha256",
            _sha256(self.checkpoint_sha256, field="checkpoint_sha256"),
        )
        if (
            not isinstance(self.source_revision, str)
            or _SOURCE_REVISION_PATTERN.fullmatch(self.source_revision) is None
        ):
            raise ValueError("source_revision must be a full lowercase Git revision")
        object.__setattr__(
            self,
            "training_configuration_sha256",
            _sha256(
                self.training_configuration_sha256,
                field="training_configuration_sha256",
            ),
        )

    def to_dict(self) -> dict[str, str | int]:
        return {
            "checkpoint_sha256": self.checkpoint_sha256,
            "run_id": self.run_id,
            "source_revision": self.source_revision,
            "training_configuration_sha256": self.training_configuration_sha256,
            "update_index": self.update_index,
            "valid_transitions": self.valid_transitions,
            "vector_steps": self.vector_steps,
        }


@dataclass(frozen=True, slots=True)
class PpoControllerRuntime:
    """Validated actor and feature contract loaded by one fresh Controller instance."""

    actor: NumpyDeterministicActor
    observation: PpoObservationConfig
    checkpoint: SelectedCheckpointIdentity
    policy_evidence: NumpyActorFileEvidence


@dataclass(frozen=True, slots=True)
class PpoControllerExportResult:
    """Content identities produced when an unfinalized plugin is activated."""

    plugin_directory: Path
    policy: NumpyActorFileEvidence
    metadata_sha256: str
    metadata_size_bytes: int
    config_sha256: str
    config_size_bytes: int
    checkpoint: SelectedCheckpointIdentity

    def __post_init__(self) -> None:
        object.__setattr__(self, "plugin_directory", Path(self.plugin_directory))
        if not isinstance(self.policy, NumpyActorFileEvidence):
            raise TypeError("policy must be NumpyActorFileEvidence")
        for field in ("metadata_sha256", "config_sha256"):
            object.__setattr__(self, field, _sha256(getattr(self, field), field=field))
        for field in ("metadata_size_bytes", "config_size_bytes"):
            _plain_positive_integer(getattr(self, field), field=field)
        if not isinstance(self.checkpoint, SelectedCheckpointIdentity):
            raise TypeError("checkpoint must be SelectedCheckpointIdentity")


def selected_checkpoint_identity(
    checkpoint_metadata: object,
    *,
    checkpoint_sha256: str,
) -> SelectedCheckpointIdentity:
    """Lazily extract the inference identity from strict training checkpoint metadata."""

    from controller_learning.rl.artifacts import TrainingCheckpointMetadata

    if not isinstance(checkpoint_metadata, TrainingCheckpointMetadata):
        raise TypeError("checkpoint_metadata must be TrainingCheckpointMetadata")
    run = checkpoint_metadata.run_identity
    if run.benchmark_version != "0.1":
        raise ValueError("PPO Controller export requires benchmark version 0.1")
    if run.feature_schema_version != LOCAL_TRACK_FEATURE_SCHEMA_VERSION:
        raise ValueError("checkpoint feature schema differs from the Controller feature schema")
    return SelectedCheckpointIdentity(
        run_id=run.run_id,
        update_index=checkpoint_metadata.update_index,
        vector_steps=checkpoint_metadata.vector_steps,
        valid_transitions=checkpoint_metadata.valid_transitions,
        checkpoint_sha256=checkpoint_sha256,
        source_revision=run.source_revision,
        training_configuration_sha256=run.configuration_sha256,
    )


def _feature_dict(config: PpoObservationConfig) -> dict[str, int | float]:
    if not isinstance(config, PpoObservationConfig):
        raise TypeError("observation_config must be PpoObservationConfig")
    return {
        "max_speed_mps": config.max_speed_mps,
        "preview_distance_m": config.preview_distance_m,
        "preview_points": config.preview_points,
        "schema_version": LOCAL_TRACK_FEATURE_SCHEMA_VERSION,
    }


def _metadata_document(
    *,
    policy: NumpyActorFileEvidence,
    observation: PpoObservationConfig,
    checkpoint: SelectedCheckpointIdentity,
) -> dict[str, Any]:
    return {
        "artifact_type": PPO_CONTROLLER_ARTIFACT_TYPE,
        "checkpoint": checkpoint.to_dict(),
        "feature": _feature_dict(observation),
        "inference_only": {
            "contains_environment_state": False,
            "contains_optimizer_state": False,
            "contains_value_network": False,
            "runtime": "numpy",
        },
        "policy": {
            "file": PPO_CONTROLLER_POLICY_FILE,
            "schema_version": policy.schema_version,
            "sha256": policy.sha256,
            "size_bytes": policy.size_bytes,
        },
        "schema_version": PPO_CONTROLLER_METADATA_SCHEMA_VERSION,
    }


def _canonical_json_bytes(value: Mapping[str, Any]) -> bytes:
    return (
        json.dumps(
            value,
            allow_nan=False,
            ensure_ascii=True,
            separators=(",", ":"),
            sort_keys=True,
        )
        + "\n"
    ).encode("utf-8")


def _toml_string(value: str) -> str:
    return json.dumps(value, ensure_ascii=True)


def _config_bytes(
    *,
    policy: NumpyActorFileEvidence,
    metadata_sha256: str,
    metadata_size_bytes: int,
    observation: PpoObservationConfig,
    checkpoint: SelectedCheckpointIdentity,
    public_policy_max_bytes: int,
) -> bytes:
    feature = _feature_dict(observation)
    values = f'''name = "ppo"
description = {_toml_string(_DESCRIPTION)}
schema_version = {PPO_CONTROLLER_SCHEMA_VERSION}
finalized = true

[policy]
file = "{PPO_CONTROLLER_POLICY_FILE}"
sha256 = "{policy.sha256}"
size_bytes = {policy.size_bytes}
schema_version = {policy.schema_version}
max_size_bytes = {public_policy_max_bytes}

[metadata]
file = "{PPO_CONTROLLER_METADATA_FILE}"
sha256 = "{metadata_sha256}"
size_bytes = {metadata_size_bytes}
schema_version = {PPO_CONTROLLER_METADATA_SCHEMA_VERSION}

[feature]
schema_version = {feature["schema_version"]}
preview_points = {feature["preview_points"]}
preview_distance_m = {feature["preview_distance_m"]!r}
max_speed_mps = {feature["max_speed_mps"]!r}

[checkpoint]
run_id = {_toml_string(checkpoint.run_id)}
update_index = {checkpoint.update_index}
vector_steps = {checkpoint.vector_steps}
valid_transitions = {checkpoint.valid_transitions}
checkpoint_sha256 = "{checkpoint.checkpoint_sha256}"
source_revision = "{checkpoint.source_revision}"
training_configuration_sha256 = "{checkpoint.training_configuration_sha256}"
'''
    return values.encode("utf-8")


def _fsync_directory(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _atomic_write(path: Path, data: bytes) -> None:
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.",
        suffix=".tmp",
        dir=path.parent,
    )
    temporary = Path(temporary_name)
    try:
        os.fchmod(descriptor, 0o600)
        with os.fdopen(descriptor, "wb") as file:
            descriptor = -1
            file.write(data)
            file.flush()
            os.fsync(file.fileno())
        os.replace(temporary, path)
        _fsync_directory(path.parent)
        if path.read_bytes() != data:
            raise OSError("atomic artifact failed exact readback")
    except OSError as error:
        if descriptor >= 0:
            os.close(descriptor)
        raise PpoControllerExportError(f"failed to write {path.name} atomically") from error
    finally:
        temporary.unlink(missing_ok=True)


def _unfinalized_plugin_directory(directory: str | Path) -> Path:
    root = Path(directory).expanduser()
    if root.is_symlink() or not root.is_dir():
        raise PpoControllerExportError("plugin_directory must be a regular existing directory")
    resolved = root.resolve(strict=True)
    for required in ("controller.py", "config.toml"):
        candidate = resolved / required
        if candidate.is_symlink() or not candidate.is_file():
            raise PpoControllerExportError(f"plugin template requires regular {required}")
    try:
        with (resolved / "config.toml").open("rb") as file:
            current = tomllib.load(file)
    except (OSError, tomllib.TOMLDecodeError) as error:
        raise PpoControllerExportError("plugin template config.toml is unreadable") from error
    if current.get("finalized") is not False:
        raise PpoControllerExportError("refusing to overwrite a finalized PPO Controller")
    return resolved


def export_numpy_actor_controller(
    plugin_directory: str | Path,
    *,
    actor: NumpyDeterministicActor,
    checkpoint: SelectedCheckpointIdentity,
    observation_config: PpoObservationConfig,
    public_policy_max_bytes: int,
) -> PpoControllerExportResult:
    """Activate an unfinalized plugin, committing its finalized config last."""

    root = _unfinalized_plugin_directory(plugin_directory)
    if not isinstance(actor, NumpyDeterministicActor):
        raise TypeError("actor must be NumpyDeterministicActor")
    if not isinstance(checkpoint, SelectedCheckpointIdentity):
        raise TypeError("checkpoint must be SelectedCheckpointIdentity")
    maximum = _plain_positive_integer(
        public_policy_max_bytes,
        field="public_policy_max_bytes",
    )
    observation_values = _feature_dict(observation_config)
    del observation_values

    policy = save_numpy_actor_npz(actor, root / PPO_CONTROLLER_POLICY_FILE)
    if policy.size_bytes > maximum or policy.size_bytes > NUMPY_ACTOR_MAX_BYTES:
        raise PpoControllerExportError("canonical policy exceeds the configured public size limit")
    metadata_bytes = _canonical_json_bytes(
        _metadata_document(
            policy=policy,
            observation=observation_config,
            checkpoint=checkpoint,
        )
    )
    if len(metadata_bytes) > PPO_CONTROLLER_METADATA_MAX_BYTES:
        raise PpoControllerExportError("PPO Controller metadata exceeds its size limit")
    metadata_sha256 = hashlib.sha256(metadata_bytes).hexdigest()
    _atomic_write(root / PPO_CONTROLLER_METADATA_FILE, metadata_bytes)
    config_bytes = _config_bytes(
        policy=policy,
        metadata_sha256=metadata_sha256,
        metadata_size_bytes=len(metadata_bytes),
        observation=observation_config,
        checkpoint=checkpoint,
        public_policy_max_bytes=maximum,
    )
    # This final atomic replacement is the activation commit. Until it succeeds, the checked-in
    # config remains finalized=false and the Controller refuses to load any staged artifact.
    _atomic_write(root / "config.toml", config_bytes)
    return PpoControllerExportResult(
        plugin_directory=root,
        policy=policy,
        metadata_sha256=metadata_sha256,
        metadata_size_bytes=len(metadata_bytes),
        config_sha256=hashlib.sha256(config_bytes).hexdigest(),
        config_size_bytes=len(config_bytes),
        checkpoint=checkpoint,
    )


def export_ppo_controller(
    plugin_directory: str | Path,
    *,
    loaded_checkpoint: object,
    training_config_path: str | Path,
    public_policy_max_bytes: int,
) -> PpoControllerExportResult:
    """Convert one verified retained checkpoint into an inference-only Controller.

    Accepting the loader's compound result prevents callers from pairing weights from one
    candidate with metadata or a content digest from another candidate.
    """

    from controller_learning.rl.artifacts import LoadedCandidateCheckpoint
    from controller_learning.rl.configuration import load_ppo_config

    if not isinstance(loaded_checkpoint, LoadedCandidateCheckpoint):
        raise TypeError("loaded_checkpoint must be a verified LoadedCandidateCheckpoint")
    config_path = Path(training_config_path)
    if config_path.is_symlink() or not config_path.is_file():
        raise PpoControllerExportError(
            "training_config_path must be a regular non-symlink TOML file"
        )
    try:
        config_bytes_before = config_path.read_bytes()
    except OSError as error:
        raise PpoControllerExportError("training configuration could not be read") from error
    expected_config_sha256 = loaded_checkpoint.metadata.run_identity.configuration_sha256
    if hashlib.sha256(config_bytes_before).hexdigest() != expected_config_sha256:
        raise PpoControllerExportError(
            "training configuration SHA-256 differs from the selected checkpoint identity"
        )
    training_config = load_ppo_config(config_path)
    try:
        config_bytes_after = config_path.read_bytes()
    except OSError as error:
        raise PpoControllerExportError("training configuration changed while loading") from error
    if config_bytes_after != config_bytes_before:
        raise PpoControllerExportError("training configuration changed while loading")

    actor = numpy_actor_from_ppo_state_dict(loaded_checkpoint.payload["model_state_dict"])
    checkpoint = selected_checkpoint_identity(
        loaded_checkpoint.metadata,
        checkpoint_sha256=loaded_checkpoint.record.sha256,
    )
    return export_numpy_actor_controller(
        plugin_directory,
        actor=actor,
        checkpoint=checkpoint,
        observation_config=training_config.observation,
        public_policy_max_bytes=public_policy_max_bytes,
    )


def _strict_json(path: Path, *, expected_sha256: str, expected_size: int) -> dict[str, Any]:
    if path.is_symlink() or not path.is_file():
        raise PpoControllerExportError("metadata must be a regular local file")
    data = path.read_bytes()
    if len(data) != expected_size or len(data) > PPO_CONTROLLER_METADATA_MAX_BYTES:
        raise PpoControllerExportError("metadata size differs from config.toml")
    if hashlib.sha256(data).hexdigest() != expected_sha256:
        raise PpoControllerExportError("metadata SHA-256 differs from config.toml")

    def reject_constant(value: str) -> None:
        raise ValueError(f"strict JSON forbids {value}")

    def unique_object(items: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in items:
            if key in result:
                raise ValueError(f"duplicate JSON key {key!r}")
            result[key] = value
        return result

    try:
        value = json.loads(
            data,
            parse_constant=reject_constant,
            object_pairs_hook=unique_object,
        )
    except (UnicodeDecodeError, json.JSONDecodeError, ValueError) as error:
        raise PpoControllerExportError("metadata is not strict JSON") from error
    if not isinstance(value, dict) or _canonical_json_bytes(value) != data:
        raise PpoControllerExportError("metadata is not a canonical JSON object")
    return value


def _checkpoint_from_table(value: Mapping[str, Any]) -> SelectedCheckpointIdentity:
    return SelectedCheckpointIdentity(
        run_id=value["run_id"],
        update_index=value["update_index"],
        vector_steps=value["vector_steps"],
        valid_transitions=value["valid_transitions"],
        checkpoint_sha256=value["checkpoint_sha256"],
        source_revision=value["source_revision"],
        training_configuration_sha256=value["training_configuration_sha256"],
    )


def _validate_metadata_scalar_types(value: Mapping[str, Any]) -> None:
    """Reject bool/float aliases that compare equal to integer or boolean schema values."""

    _schema_version(
        value.get("schema_version"),
        field="metadata.schema_version",
        expected=PPO_CONTROLLER_METADATA_SCHEMA_VERSION,
    )
    policy = value.get("policy")
    feature = value.get("feature")
    checkpoint = value.get("checkpoint")
    inference = value.get("inference_only")
    if not all(isinstance(table, Mapping) for table in (policy, feature, checkpoint, inference)):
        raise PpoControllerExportError("metadata tables are invalid")
    _schema_version(
        policy.get("schema_version"),
        field="metadata.policy.schema_version",
        expected=NUMPY_ACTOR_SCHEMA_VERSION,
    )
    _schema_version(
        feature.get("schema_version"),
        field="metadata.feature.schema_version",
        expected=LOCAL_TRACK_FEATURE_SCHEMA_VERSION,
    )
    for field in ("size_bytes",):
        _plain_positive_integer(policy.get(field), field=f"metadata.policy.{field}")
    for field in ("preview_points",):
        _plain_positive_integer(feature.get(field), field=f"metadata.feature.{field}")
    for field in ("update_index", "vector_steps", "valid_transitions"):
        _plain_positive_integer(checkpoint.get(field), field=f"metadata.checkpoint.{field}")
    for field in (
        "contains_environment_state",
        "contains_optimizer_state",
        "contains_value_network",
    ):
        if type(inference.get(field)) is not bool:
            raise PpoControllerExportError(f"metadata.inference_only.{field} must be boolean")


def load_ppo_controller_runtime(
    public_config: Mapping[str, Any],
    *,
    plugin_directory: str | Path,
) -> PpoControllerRuntime:
    """Load one finalized local actor using only immutable public Controller config."""

    if not isinstance(public_config, Mapping):
        raise TypeError("public_config must be a mapping")
    controller = public_config.get("controller")
    if not isinstance(controller, Mapping):
        raise ValueError("public_config must contain the Controller-owned config table")
    _exact_keys(
        controller,
        {
            "name",
            "description",
            "schema_version",
            "finalized",
            "policy",
            "metadata",
            "feature",
            "checkpoint",
        },
        field="controller",
    )
    if controller["finalized"] is not True:
        raise PpoControllerNotFinalizedError(
            "PPO Controller template is not finalized; export selected weights first"
        )
    if controller["name"] != "ppo" or controller["description"] != _DESCRIPTION:
        raise ValueError("PPO Controller root identity is invalid")
    _schema_version(
        controller["schema_version"],
        field="controller.schema_version",
        expected=PPO_CONTROLLER_SCHEMA_VERSION,
    )
    policy = _table(
        controller,
        "policy",
        {"file", "sha256", "size_bytes", "schema_version", "max_size_bytes"},
    )
    metadata = _table(
        controller,
        "metadata",
        {"file", "sha256", "size_bytes", "schema_version"},
    )
    feature = _table(
        controller,
        "feature",
        {"schema_version", "preview_points", "preview_distance_m", "max_speed_mps"},
    )
    checkpoint_table = _table(
        controller,
        "checkpoint",
        {
            "run_id",
            "update_index",
            "vector_steps",
            "valid_transitions",
            "checkpoint_sha256",
            "source_revision",
            "training_configuration_sha256",
        },
    )
    _plain_file_name(
        policy["file"],
        field="controller.policy.file",
        expected=PPO_CONTROLLER_POLICY_FILE,
    )
    _plain_file_name(
        metadata["file"],
        field="controller.metadata.file",
        expected=PPO_CONTROLLER_METADATA_FILE,
    )
    _schema_version(
        policy["schema_version"],
        field="controller.policy.schema_version",
        expected=NUMPY_ACTOR_SCHEMA_VERSION,
    )
    _schema_version(
        metadata["schema_version"],
        field="controller.metadata.schema_version",
        expected=PPO_CONTROLLER_METADATA_SCHEMA_VERSION,
    )
    _schema_version(
        feature["schema_version"],
        field="controller.feature.schema_version",
        expected=LOCAL_TRACK_FEATURE_SCHEMA_VERSION,
    )
    policy_sha256 = _sha256(policy["sha256"], field="controller.policy.sha256")
    policy_size = _plain_positive_integer(
        policy["size_bytes"],
        field="controller.policy.size_bytes",
    )
    policy_maximum = _plain_positive_integer(
        policy["max_size_bytes"],
        field="controller.policy.max_size_bytes",
    )
    if policy_size > policy_maximum or policy_size > NUMPY_ACTOR_MAX_BYTES:
        raise PpoControllerExportError("configured policy size exceeds its safe local limit")
    metadata_sha256 = _sha256(metadata["sha256"], field="controller.metadata.sha256")
    metadata_size = _plain_positive_integer(
        metadata["size_bytes"],
        field="controller.metadata.size_bytes",
    )
    observation = PpoObservationConfig(
        preview_points=feature["preview_points"],
        preview_distance_m=feature["preview_distance_m"],
        max_speed_mps=feature["max_speed_mps"],
    )
    checkpoint = _checkpoint_from_table(checkpoint_table)

    root = Path(plugin_directory).expanduser()
    if root.is_symlink() or not root.is_dir():
        raise PpoControllerExportError("plugin_directory must be a regular local directory")
    root = root.resolve(strict=True)
    metadata_document = _strict_json(
        root / PPO_CONTROLLER_METADATA_FILE,
        expected_sha256=metadata_sha256,
        expected_size=metadata_size,
    )
    _validate_metadata_scalar_types(metadata_document)
    policy_path = root / PPO_CONTROLLER_POLICY_FILE
    if policy_path.is_symlink() or not policy_path.is_file():
        raise PpoControllerExportError("policy must be a regular local file")
    if policy_path.stat().st_size != policy_size:
        raise PpoControllerExportError("policy size differs from config.toml")
    try:
        loaded_policy = load_numpy_actor_npz(
            policy_path,
            expected_sha256=policy_sha256,
            expected_size_bytes=policy_size,
        )
    except NumpyActorArtifactError as error:
        raise PpoControllerExportError("policy.npz failed strict local verification") from error
    expected_metadata = _metadata_document(
        policy=loaded_policy.evidence,
        observation=observation,
        checkpoint=checkpoint,
    )
    if metadata_document != expected_metadata:
        raise PpoControllerExportError("metadata content differs from finalized config.toml")
    return PpoControllerRuntime(
        actor=loaded_policy.actor,
        observation=observation,
        checkpoint=checkpoint,
        policy_evidence=loaded_policy.evidence,
    )


__all__ = [
    "PPO_CONTROLLER_ARTIFACT_TYPE",
    "PPO_CONTROLLER_METADATA_FILE",
    "PPO_CONTROLLER_METADATA_SCHEMA_VERSION",
    "PPO_CONTROLLER_POLICY_FILE",
    "PPO_CONTROLLER_SCHEMA_VERSION",
    "PpoControllerExportError",
    "PpoControllerExportResult",
    "PpoControllerNotFinalizedError",
    "PpoControllerRuntime",
    "SelectedCheckpointIdentity",
    "export_numpy_actor_controller",
    "export_ppo_controller",
    "load_ppo_controller_runtime",
    "selected_checkpoint_identity",
]
