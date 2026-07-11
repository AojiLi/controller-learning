"""Durable one-shot transaction for the frozen M8 Test evaluation.

This module owns persistence only.  It does not know where official Track assets live and never
loads an environment or a Controller.  The caller must durably prepare the transaction before
opening Test, bind Test exactly once, append the fixed 60 episode records in protocol order, and
publish only the predeclared output allowlist.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import re
import shutil
import stat
import tempfile
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from enum import StrEnum
from pathlib import Path, PurePosixPath
from types import MappingProxyType
from typing import Any, Final, Literal

ATTEMPT_TRANSACTION_SCHEMA_VERSION: Final = "controller-learning.m8-attempt-transaction.v2"
ARTIFACT_VALIDATION_SCHEMA_VERSION: Final = "controller-learning.m8-artifact-validation.v1"
FORMAL_ARTIFACT_VALIDATOR_ID: Final = (
    "controller_learning.evaluation.final_report.validate_m8_publication"
)
M8_EXECUTION_EVIDENCE_SCHEMA_VERSION: Final = "controller-learning.m8-execution-evidence.v1"
FORMAL_EXECUTION_EVIDENCE_BLOB_PATH: Final = "execution/final_evidence.json"
FORMAL_CONTROLLER_ORDER: Final = ("pid", "mpc", "ppo")
FORMAL_ROWS_PER_CONTROLLER: Final = 20
FORMAL_EPISODE_COUNT: Final = 60

_UINT32_MAX: Final = 2**32 - 1
_SHA256_PATTERN = re.compile(r"^[0-9a-f]{64}$")
_REVISION_PATTERN = re.compile(r"^[0-9a-f]{40}$")
_PUBLICATION_SCRATCH_PATTERN = re.compile(r"^(\d{3})\.(publish|restore)$")
_STATE_TEMPORARY_PATTERN = re.compile(r"^\.state\.json\.[A-Za-z0-9_-]+\.tmp$")
_OUTCOMES: Final = frozenset({"success", "off_track", "invalid_action", "timeout"})
_PHASE_INDEX: Final = {
    "PREPARED": 0,
    "TEST_BOUND": 1,
    "EVALUATION_COMPLETE": 2,
    "ARTIFACTS_VALIDATED": 3,
    "COMMITTED": 4,
}


class AttemptTransactionError(RuntimeError):
    """A durable M8 attempt is unsafe, inconsistent, or used out of phase."""


class AttemptTransactionTamperError(AttemptTransactionError):
    """Durable transaction bytes or protected output state differ from their manifest."""


class IncompleteTestAttemptError(AttemptTransactionError):
    """A process-lost Test-bound attempt cannot be retried automatically."""

    def __init__(self, inspection: AttemptInspection) -> None:
        self.inspection = inspection
        execution_evidence_sealed = any(
            record.relative_path == FORMAL_EXECUTION_EVIDENCE_BLOB_PATH
            for record in inspection.blob_records
        )
        super().__init__(
            "the Test-bound attempt is incomplete; automatic retry is forbidden and the "
            f"durable journal is preserved ({inspection.journal_record_count}/"
            f"{FORMAL_EPISODE_COUNT} records; execution evidence sealed="
            f"{str(execution_evidence_sealed).lower()})"
        )


class AttemptPhase(StrEnum):
    """Monotonic durable phases of one formal Test attempt."""

    PREPARED = "PREPARED"
    TEST_BOUND = "TEST_BOUND"
    EVALUATION_COMPLETE = "EVALUATION_COMPLETE"
    ARTIFACTS_VALIDATED = "ARTIFACTS_VALIDATED"
    COMMITTED = "COMMITTED"


@dataclass(frozen=True, slots=True)
class AttemptIdentity:
    """Immutable source, configuration, dependency, and input identity."""

    source_revision: str
    source_tree_sha256: str
    config_sha256: str
    pixi_lock_sha256: str
    input_sha256: str

    def __post_init__(self) -> None:
        if (
            not isinstance(self.source_revision, str)
            or _REVISION_PATTERN.fullmatch(self.source_revision) is None
        ):
            raise ValueError("source_revision must be a lowercase 40-character Git revision")
        for name in (
            "source_tree_sha256",
            "config_sha256",
            "pixi_lock_sha256",
            "input_sha256",
        ):
            value = getattr(self, name)
            if not isinstance(value, str) or _SHA256_PATTERN.fullmatch(value) is None:
                raise ValueError(f"{name} must be a lowercase SHA-256 digest")

    def to_dict(self) -> dict[str, str]:
        """Return the canonical JSON-compatible identity."""

        return {
            "config_sha256": self.config_sha256,
            "input_sha256": self.input_sha256,
            "pixi_lock_sha256": self.pixi_lock_sha256,
            "source_revision": self.source_revision,
            "source_tree_sha256": self.source_tree_sha256,
        }


@dataclass(frozen=True, slots=True)
class EpisodeJournalRecord:
    """One canonical result in the fixed Controller-major Test order."""

    controller: str
    row_index: int
    track_id: int
    reset_seed: int
    episode_seed: int
    controller_seed: int
    outcome: str
    steps: int
    trajectory_blob_path: str
    trajectory_blob_sha256: str
    trajectory_blob_size_bytes: int
    data: Mapping[str, Any] = field(default_factory=dict)
    schema_version: int = 1

    def __post_init__(self) -> None:
        if type(self.schema_version) is not int or self.schema_version != 1:
            raise ValueError("episode journal schema_version must be exactly 1")
        if self.controller not in FORMAL_CONTROLLER_ORDER:
            raise ValueError("controller must be one of pid, mpc, or ppo")
        if type(self.row_index) is not int or not 0 <= self.row_index < 20:
            raise ValueError("row_index must be an integer in [0, 20)")
        for name in ("track_id", "reset_seed", "episode_seed", "controller_seed"):
            value = getattr(self, name)
            if type(value) is not int or not 0 <= value <= _UINT32_MAX:
                raise ValueError(f"{name} must be a uint32 integer")
        if self.reset_seed != self.row_index:
            raise ValueError("reset_seed must equal the fixed Test row index")
        if self.episode_seed == self.controller_seed:
            raise ValueError("episode_seed and controller_seed must remain domain-separated")
        if self.outcome not in _OUTCOMES:
            raise ValueError("outcome must be success, off_track, invalid_action, or timeout")
        if type(self.steps) is not int or self.steps < 1:
            raise ValueError("steps must be a positive integer")
        expected_blob_path = f"episodes/{self.controller}/row_{self.row_index:03d}_trajectory.json"
        if self.trajectory_blob_path != expected_blob_path:
            raise ValueError("trajectory_blob_path differs from the fixed episode evidence path")
        if (
            not isinstance(self.trajectory_blob_sha256, str)
            or _SHA256_PATTERN.fullmatch(self.trajectory_blob_sha256) is None
        ):
            raise ValueError("trajectory_blob_sha256 must be a lowercase SHA-256 digest")
        if type(self.trajectory_blob_size_bytes) is not int or self.trajectory_blob_size_bytes < 1:
            raise ValueError("trajectory_blob_size_bytes must be a positive integer")
        if not isinstance(self.data, Mapping):
            raise TypeError("data must be a mapping")
        data = _json_snapshot(self.data, field="episode.data")
        if not isinstance(data, dict):  # pragma: no cover - guarded by Mapping above
            raise TypeError("data must be a mapping")
        expected_data_keys = {
            "compute_times_s",
            "controller_import_time_s",
            "controller_init_time_s",
        }
        if set(data) != expected_data_keys:
            raise ValueError("episode data must contain the exact Runner timing evidence")
        compute_times = data["compute_times_s"]
        if (
            not isinstance(compute_times, list)
            or len(compute_times) != self.steps
            or any(
                type(value) is not float or not math.isfinite(value) or value < 0.0
                for value in compute_times
            )
        ):
            raise ValueError("compute_times_s must contain one finite non-negative float per step")
        for name in ("controller_import_time_s", "controller_init_time_s"):
            value = data[name]
            if type(value) is not float or not math.isfinite(value) or value < 0.0:
                raise ValueError(f"{name} must be a finite non-negative float")
        object.__setattr__(self, "data", MappingProxyType(data))

    def to_dict(self) -> dict[str, Any]:
        """Return the exact canonical journal object."""

        return {
            "controller": self.controller,
            "controller_seed": self.controller_seed,
            "data": dict(self.data),
            "episode_seed": self.episode_seed,
            "outcome": self.outcome,
            "reset_seed": self.reset_seed,
            "row_index": self.row_index,
            "schema_version": self.schema_version,
            "steps": self.steps,
            "track_id": self.track_id,
            "trajectory_blob_path": self.trajectory_blob_path,
            "trajectory_blob_sha256": self.trajectory_blob_sha256,
            "trajectory_blob_size_bytes": self.trajectory_blob_size_bytes,
        }


@dataclass(frozen=True, slots=True)
class BlobRecord:
    """Content identity of one immutable Test-bound binary evidence blob."""

    relative_path: str
    sha256: str
    size_bytes: int
    mode: int


@dataclass(frozen=True, slots=True)
class PublishedOutput:
    """Content identity of one committed allowlisted output."""

    relative_path: str
    sha256: str
    size_bytes: int
    mode: int


@dataclass(frozen=True, slots=True)
class AttemptInspection:
    """Read-only, fully verified state of one existing durable attempt."""

    exists: bool
    phase: AttemptPhase | None
    journal_record_count: int
    next_episode: tuple[str, int] | None
    blob_records: tuple[BlobRecord, ...]
    staged_outputs: tuple[PublishedOutput, ...]
    output_state: Literal["absent", "original", "partial_publication", "published"]


@dataclass(frozen=True, slots=True)
class AttemptRecovery:
    """Action taken while inspecting a transaction left by an earlier process."""

    action: Literal[
        "none",
        "pre_test_restored",
        "evaluation_complete_ready",
        "artifacts_validated_ready",
        "partial_publication_restored",
        "committed_retained",
    ]
    phase_before: AttemptPhase | None
    journal_record_count: int


@dataclass(frozen=True, slots=True)
class _OutputSnapshot:
    relative_path: str
    content: bytes | None
    mode: int | None


@dataclass(frozen=True, slots=True)
class _LoadedTransaction:
    phase: AttemptPhase
    state: Mapping[str, Any]
    manifest: Mapping[str, Any]
    snapshots: tuple[_OutputSnapshot, ...]
    journal: tuple[EpisodeJournalRecord, ...]
    blobs: tuple[BlobRecord, ...]
    staged: tuple[PublishedOutput, ...]
    output_state: Literal["original", "partial_publication", "published"]


def _json_snapshot(value: Any, *, field: str, active: set[int] | None = None) -> Any:
    if value is None or type(value) in {bool, int, str}:
        if isinstance(value, str):
            try:
                value.encode("utf-8")
            except UnicodeEncodeError as error:
                raise ValueError(f"{field} is not valid UTF-8") from error
        return value
    if type(value) is float:
        if not math.isfinite(value):
            raise ValueError(f"{field} must not contain NaN or infinity")
        return value
    active = set() if active is None else active
    identity = id(value)
    if identity in active:
        raise ValueError(f"{field} contains a reference cycle")
    if isinstance(value, Mapping):
        active.add(identity)
        try:
            result: dict[str, Any] = {}
            for key, item in value.items():
                if type(key) is not str:
                    raise ValueError(f"{field} must use string object keys")
                result[key] = _json_snapshot(item, field=f"{field}.{key}", active=active)
            return result
        finally:
            active.remove(identity)
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        active.add(identity)
        try:
            return [
                _json_snapshot(item, field=f"{field}[{index}]", active=active)
                for index, item in enumerate(value)
            ]
        finally:
            active.remove(identity)
    raise ValueError(f"{field} contains unsupported value type {type(value).__name__}")


def _canonical_json_bytes(value: Mapping[str, Any]) -> bytes:
    snapshot = _json_snapshot(value, field="$")
    if not isinstance(snapshot, dict):  # pragma: no cover - guarded by Mapping annotation
        raise ValueError("canonical JSON root must be an object")
    return (
        json.dumps(
            snapshot,
            allow_nan=False,
            ensure_ascii=False,
            separators=(",", ":"),
            sort_keys=True,
        )
        + "\n"
    ).encode("utf-8")


def canonical_execution_evidence_bytes(value: Mapping[str, Any]) -> bytes:
    """Validate and serialize the durable post-close execution evidence seal."""

    if not isinstance(value, Mapping):
        raise TypeError("execution evidence must be a mapping")
    _exact_keys(
        value,
        {
            "asset_access",
            "execution",
            "memory",
            "runtime",
            "schema_version",
            "test_assets",
        },
        field_name="execution evidence",
    )
    if value["schema_version"] != M8_EXECUTION_EVIDENCE_SCHEMA_VERSION:
        raise ValueError("execution evidence schema_version differs from the frozen protocol")
    for field_name in ("asset_access", "execution", "memory", "runtime", "test_assets"):
        if not isinstance(value[field_name], Mapping):
            raise TypeError(f"execution evidence {field_name} must be a mapping")
    from controller_learning.evaluation.final_report import (
        validate_durable_execution_evidence_mapping,
    )

    validate_durable_execution_evidence_mapping(value)
    return _canonical_json_bytes(value)


def _sha256(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _safe_relative_path(value: str, *, field_name: str) -> str:
    if not isinstance(value, str) or not value or "\\" in value:
        raise ValueError(f"{field_name} must be a normalized relative POSIX path")
    path = PurePosixPath(value)
    if (
        path.is_absolute()
        or path.as_posix() != value
        or any(part in {"", ".", ".."} for part in path.parts)
    ):
        raise ValueError(f"{field_name} must be a normalized relative POSIX path")
    return value


def _mode(value: int, *, field_name: str) -> int:
    if type(value) is not int or not 0 <= value <= 0o777:
        raise ValueError(f"{field_name} must be a permission value from 0o000 to 0o777")
    return value


def _exact_keys(value: Mapping[str, Any], expected: set[str], *, field_name: str) -> None:
    if any(type(key) is not str for key in value) or set(value) != expected:
        raise AttemptTransactionTamperError(f"{field_name} keys differ")


def _read_canonical_json(path: Path, *, field_name: str) -> Mapping[str, Any]:
    _require_regular_file(path, label=field_name)
    payload = path.read_bytes()
    try:
        value = json.loads(payload)
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise AttemptTransactionTamperError(f"{field_name} is not strict JSON") from error
    if not isinstance(value, Mapping) or _canonical_json_bytes(value) != payload:
        raise AttemptTransactionTamperError(f"{field_name} is not canonical JSON")
    return value


def _fsync_directory(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _require_directory(path: Path, *, label: str) -> None:
    try:
        metadata = path.lstat()
    except FileNotFoundError as error:
        raise AttemptTransactionTamperError(f"{label} is missing") from error
    if stat.S_ISLNK(metadata.st_mode) or not stat.S_ISDIR(metadata.st_mode):
        raise AttemptTransactionTamperError(f"{label} must be a non-symlink directory")


def _require_regular_file(path: Path, *, label: str) -> None:
    try:
        metadata = path.lstat()
    except FileNotFoundError as error:
        raise AttemptTransactionTamperError(f"{label} is missing") from error
    if stat.S_ISLNK(metadata.st_mode) or not stat.S_ISREG(metadata.st_mode):
        raise AttemptTransactionTamperError(f"{label} must be a non-symlink regular file")


def _write_fsynced_file(path: Path, payload: bytes, *, mode: int) -> None:
    if not isinstance(payload, bytes):
        raise TypeError("durable payload must be bytes")
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_CLOEXEC", 0)
    flags |= getattr(os, "O_NOFOLLOW", 0)
    descriptor = os.open(path, flags, mode)
    try:
        os.fchmod(descriptor, mode)
        with os.fdopen(descriptor, "wb") as file:
            descriptor = -1
            file.write(payload)
            file.flush()
            os.fsync(file.fileno())
    finally:
        if descriptor >= 0:
            os.close(descriptor)
    _fsync_directory(path.parent)
    _require_regular_file(path, label=path.name)
    if path.read_bytes() != payload or stat.S_IMODE(path.stat().st_mode) != mode:
        raise AttemptTransactionTamperError("durable file failed exact readback")


def _atomic_write(path: Path, payload: bytes, *, mode: int = 0o600) -> None:
    if path.exists() or path.is_symlink():
        _require_regular_file(path, label=path.name)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    temporary = Path(temporary_name)
    try:
        os.fchmod(descriptor, mode)
        with os.fdopen(descriptor, "wb") as file:
            descriptor = -1
            file.write(payload)
            file.flush()
            os.fsync(file.fileno())
        os.replace(temporary, path)
        _fsync_directory(path.parent)
        _require_regular_file(path, label=path.name)
        if path.read_bytes() != payload or stat.S_IMODE(path.stat().st_mode) != mode:
            raise AttemptTransactionTamperError("atomic file failed exact readback")
    finally:
        if descriptor >= 0:
            os.close(descriptor)
        temporary.unlink(missing_ok=True)


def _walk_without_symlinks(path: Path) -> None:
    _require_directory(path, label="cleanup tree")
    for root, directories, files in os.walk(path, topdown=True, followlinks=False):
        root_path = Path(root)
        for name in (*directories, *files):
            candidate = root_path / name
            metadata = candidate.lstat()
            if stat.S_ISLNK(metadata.st_mode):
                raise AttemptTransactionTamperError("cleanup trees cannot contain symlinks")
            if name in directories and not stat.S_ISDIR(metadata.st_mode):
                raise AttemptTransactionTamperError("cleanup directory entry has the wrong type")
            if name in files and not stat.S_ISREG(metadata.st_mode):
                raise AttemptTransactionTamperError("cleanup file entry has the wrong type")


class M8AttemptTransaction:
    """Persist and publish one non-retryable M8 Test comparison attempt."""

    def __init__(
        self,
        project_root: str | Path,
        *,
        transaction_relative_path: str,
        output_allowlist: Sequence[str],
        identity: AttemptIdentity,
    ) -> None:
        supplied_root = Path(project_root)
        try:
            root_metadata = supplied_root.lstat()
        except FileNotFoundError as error:
            raise ValueError("project_root must already exist") from error
        if stat.S_ISLNK(root_metadata.st_mode) or not stat.S_ISDIR(root_metadata.st_mode):
            raise ValueError("project_root must be a non-symlink directory")
        self.project_root = supplied_root.resolve(strict=True)
        transaction = _safe_relative_path(
            transaction_relative_path, field_name="transaction_relative_path"
        )
        transaction_parts = PurePosixPath(transaction).parts
        if len(transaction_parts) < 2 or transaction_parts[0] != "runs":
            raise ValueError("transaction_relative_path must be a child of ignored runs/")
        self.transaction_relative_path = transaction
        self.transaction_directory = self.project_root / transaction
        self.cleanup_directory = self.transaction_directory.with_name(
            self.transaction_directory.name + ".cleanup"
        )
        if not isinstance(identity, AttemptIdentity):
            raise TypeError("identity must be an AttemptIdentity")
        self.identity = identity
        if isinstance(output_allowlist, (str, bytes)) or not isinstance(output_allowlist, Sequence):
            raise TypeError("output_allowlist must be a sequence of paths")
        outputs = tuple(
            _safe_relative_path(value, field_name=f"output_allowlist[{index}]")
            for index, value in enumerate(output_allowlist)
        )
        if not outputs or len(set(outputs)) != len(outputs):
            raise ValueError("output_allowlist must contain unique paths")
        if outputs != tuple(sorted(outputs)):
            raise ValueError("output_allowlist must be sorted for canonical publication")
        transaction_prefix = f"{transaction}/"
        if any(path == transaction or path.startswith(transaction_prefix) for path in outputs):
            raise ValueError("formal outputs cannot be inside the transaction directory")
        self.output_allowlist = outputs
        self._bound_in_process = False
        self._bound_process_id: int | None = None

    def _require_bound_process(self) -> None:
        """Permit Test evidence writes only from the PID that crossed ``bind_test``."""

        if not self._bound_in_process or self._bound_process_id != os.getpid():
            raise IncompleteTestAttemptError(self.inspect())

    @property
    def _manifest_path(self) -> Path:
        return self.transaction_directory / "manifest.json"

    @property
    def _state_path(self) -> Path:
        return self.transaction_directory / "state.json"

    @property
    def _journal_path(self) -> Path:
        return self.transaction_directory / "episode-journal.jsonl"

    @property
    def _blob_index_path(self) -> Path:
        return self.transaction_directory / "blob-index.jsonl"

    @property
    def _artifact_validation_path(self) -> Path:
        return self.transaction_directory / "artifact-validation.json"

    @property
    def _blobs_directory(self) -> Path:
        return self.transaction_directory / "blobs"

    @property
    def _staged_directory(self) -> Path:
        return self.transaction_directory / "final-staged"

    @property
    def _staged_build_directory(self) -> Path:
        return self.transaction_directory / "final-staged.build"

    @property
    def _publication_directory(self) -> Path:
        return self.transaction_directory / "publication"

    @staticmethod
    def _expected_episode_keys() -> tuple[tuple[str, int], ...]:
        keys = tuple(
            (controller, row)
            for controller in FORMAL_CONTROLLER_ORDER
            for row in range(FORMAL_ROWS_PER_CONTROLLER)
        )
        if len(keys) != FORMAL_EPISODE_COUNT:  # pragma: no cover - constant invariant
            raise RuntimeError("formal episode protocol does not contain exactly 60 records")
        return keys

    def _ensure_real_directory_chain(self, relative_parent: PurePosixPath) -> tuple[str, ...]:
        current = self.project_root
        created: list[str] = []
        for part in relative_parent.parts:
            candidate = current / part
            try:
                metadata = candidate.lstat()
            except FileNotFoundError:
                os.mkdir(candidate, 0o755)
                _fsync_directory(current)
                created.append(candidate.relative_to(self.project_root).as_posix())
            else:
                if stat.S_ISLNK(metadata.st_mode) or not stat.S_ISDIR(metadata.st_mode):
                    raise AttemptTransactionTamperError(
                        "managed path parents must be non-symlink directories"
                    )
            current = candidate
        return tuple(created)

    def _output_path(self, relative_path: str) -> Path:
        if relative_path not in self.output_allowlist:
            raise AttemptTransactionError("path is outside the fixed output allowlist")
        destination = self.project_root / relative_path
        current = self.project_root
        for part in PurePosixPath(relative_path).parts[:-1]:
            current /= part
            _require_directory(current, label="formal output parent")
        if destination.exists() or destination.is_symlink():
            _require_regular_file(destination, label="formal output")
        return destination

    def _classification_output_path(
        self,
        relative_path: str,
        *,
        removable_directories: frozenset[str],
    ) -> Path | None:
        """Resolve an output while tolerating only PREPARED cleanup directories already removed."""

        current = self.project_root
        relative_parent = PurePosixPath(relative_path).parent
        for part in relative_parent.parts:
            current /= part
            try:
                metadata = current.lstat()
            except FileNotFoundError:
                relative = current.relative_to(self.project_root).as_posix()
                if relative in removable_directories:
                    return None
                raise AttemptTransactionTamperError("formal output parent is missing") from None
            if stat.S_ISLNK(metadata.st_mode) or not stat.S_ISDIR(metadata.st_mode):
                raise AttemptTransactionTamperError(
                    "formal output parent must be a non-symlink directory"
                )
        destination = self.project_root / relative_path
        try:
            metadata = destination.lstat()
        except FileNotFoundError:
            return destination
        if stat.S_ISLNK(metadata.st_mode) or not stat.S_ISREG(metadata.st_mode):
            raise AttemptTransactionTamperError("formal output must be a non-symlink regular file")
        return destination

    @staticmethod
    def _read_stable_output(path: Path) -> tuple[bytes, int]:
        try:
            before = path.lstat()
            payload = path.read_bytes()
            after = path.lstat()
        except FileNotFoundError as error:
            raise AttemptTransactionTamperError(
                "formal output changed while its identity was checked"
            ) from error
        identity_fields = (
            "st_dev",
            "st_ino",
            "st_mode",
            "st_uid",
            "st_gid",
            "st_size",
            "st_ctime_ns",
            "st_mtime_ns",
        )
        if (
            stat.S_ISLNK(before.st_mode)
            or not stat.S_ISREG(before.st_mode)
            or any(getattr(before, name) != getattr(after, name) for name in identity_fields)
        ):
            raise AttemptTransactionTamperError(
                "formal output changed while its identity was checked"
            )
        return payload, stat.S_IMODE(before.st_mode)

    def _output_identities(
        self,
        snapshot: _OutputSnapshot,
        staged: PublishedOutput | None,
        *,
        removable_directories: frozenset[str] = frozenset(),
    ) -> frozenset[str]:
        destination = self._classification_output_path(
            snapshot.relative_path,
            removable_directories=removable_directories,
        )
        if destination is None:
            return frozenset({"original"}) if snapshot.content is None else frozenset()
        try:
            metadata = destination.lstat()
        except FileNotFoundError:
            return frozenset({"original"}) if snapshot.content is None else frozenset()
        if stat.S_ISLNK(metadata.st_mode) or not stat.S_ISREG(metadata.st_mode):
            raise AttemptTransactionTamperError("formal output must be a non-symlink regular file")
        payload, mode = self._read_stable_output(destination)
        identities: set[str] = set()
        if snapshot.content is not None and payload == snapshot.content and mode == snapshot.mode:
            identities.add("original")
        if staged is not None and (
            _sha256(payload) == staged.sha256
            and len(payload) == staged.size_bytes
            and mode == staged.mode
        ):
            identities.add("staged")
        return frozenset(identities)

    def _require_output_identity(
        self,
        snapshot: _OutputSnapshot,
        staged: PublishedOutput | None,
        *,
        permitted: frozenset[str],
    ) -> str:
        identities = self._output_identities(snapshot, staged)
        matches = identities & permitted
        if not matches:
            raise AttemptTransactionTamperError(
                f"formal output {snapshot.relative_path!r} changed immediately before mutation"
            )
        return "original" if "original" in matches else "staged"

    def _remove_tree(self, path: Path) -> None:
        if not path.exists() and not path.is_symlink():
            return
        _walk_without_symlinks(path)
        shutil.rmtree(path)
        _fsync_directory(path.parent)

    def _retire_transaction(self) -> None:
        _require_directory(self.transaction_directory, label="attempt transaction")
        if self.cleanup_directory.exists() or self.cleanup_directory.is_symlink():
            self._remove_tree(self.cleanup_directory)
        os.replace(self.transaction_directory, self.cleanup_directory)
        _fsync_directory(self.cleanup_directory.parent)
        self._remove_tree(self.cleanup_directory)

    def _manifest(
        self,
        snapshots: Sequence[_OutputSnapshot],
        created_directories: Sequence[str],
    ) -> dict[str, Any]:
        outputs = []
        for index, snapshot in enumerate(snapshots):
            existed = snapshot.content is not None
            outputs.append(
                {
                    "backup_relative_path": f"backups/{index:03d}.bin" if existed else None,
                    "existed": existed,
                    "mode": snapshot.mode,
                    "relative_path": snapshot.relative_path,
                    "sha256": _sha256(snapshot.content) if existed else None,
                    "size_bytes": len(snapshot.content) if existed else 0,
                }
            )
        return {
            "created_output_directories": list(created_directories),
            "episode_protocol": {
                "controller_order": list(FORMAL_CONTROLLER_ORDER),
                "expected_record_count": FORMAL_EPISODE_COUNT,
                "ordering": "controller_major_then_row_index",
                "rows_per_controller": FORMAL_ROWS_PER_CONTROLLER,
            },
            "identity": self.identity.to_dict(),
            "output_allowlist": list(self.output_allowlist),
            "outputs": outputs,
            "recovery_policy": {
                "accepted_result": "first_complete_protocol_passing_attempt",
                "automatic_retry_after_test_bound": False,
                "completed_attempt_finalizes_from_durable_bytes_only": True,
                "low_performance_can_trigger_retry": False,
                "partial_publication_restores_originals_before_republish": True,
            },
            "schema_version": ATTEMPT_TRANSACTION_SCHEMA_VERSION,
            "transaction_relative_path": self.transaction_relative_path,
        }

    def _state(
        self,
        phase: AttemptPhase,
        *,
        manifest_sha256: str,
        evidence: Mapping[str, str] | None = None,
    ) -> dict[str, Any]:
        return {
            "evidence": dict(evidence) if evidence is not None else None,
            "identity": self.identity.to_dict(),
            "manifest_sha256": manifest_sha256,
            "phase": phase.value,
            "phase_index": _PHASE_INDEX[phase.value],
            "schema_version": ATTEMPT_TRANSACTION_SCHEMA_VERSION,
        }

    def _transition(
        self,
        phase: AttemptPhase,
        *,
        manifest_sha256: str,
        evidence: Mapping[str, str] | None,
    ) -> None:
        _atomic_write(
            self._state_path,
            _canonical_json_bytes(
                self._state(
                    phase,
                    manifest_sha256=manifest_sha256,
                    evidence=evidence,
                )
            ),
        )

    def prepare(self) -> AttemptInspection:
        """Capture and fsync every output backup before Test may be opened."""

        parent_relative = PurePosixPath(self.transaction_relative_path).parent
        self._ensure_real_directory_chain(parent_relative)
        if self.cleanup_directory.exists() or self.cleanup_directory.is_symlink():
            self._remove_tree(self.cleanup_directory)
        if self.transaction_directory.exists() or self.transaction_directory.is_symlink():
            raise AttemptTransactionError("an existing attempt requires inspect() or recover()")

        created_output_directories: list[str] = []
        snapshots: list[_OutputSnapshot] = []
        for relative_path in self.output_allowlist:
            parent = PurePosixPath(relative_path).parent
            created_output_directories.extend(self._ensure_real_directory_chain(parent))
            destination = self._output_path(relative_path)
            if destination.exists():
                payload, mode = self._read_stable_output(destination)
                snapshots.append(
                    _OutputSnapshot(
                        relative_path=relative_path,
                        content=payload,
                        mode=mode,
                    )
                )
            else:
                snapshots.append(
                    _OutputSnapshot(relative_path=relative_path, content=None, mode=None)
                )

        staging = Path(
            tempfile.mkdtemp(
                prefix=f"{self.transaction_directory.name}.prepare.",
                dir=self.transaction_directory.parent,
            )
        )
        _fsync_directory(staging.parent)
        for name in ("backups", "blobs"):
            os.mkdir(staging / name, 0o700)
            _fsync_directory(staging)
        _write_fsynced_file(staging / "episode-journal.jsonl", b"", mode=0o600)
        _write_fsynced_file(staging / "blob-index.jsonl", b"", mode=0o600)
        manifest = self._manifest(
            snapshots,
            tuple(dict.fromkeys(created_output_directories)),
        )
        for index, snapshot in enumerate(snapshots):
            if snapshot.content is not None:
                _write_fsynced_file(
                    staging / "backups" / f"{index:03d}.bin",
                    snapshot.content,
                    mode=0o600,
                )
        manifest_bytes = _canonical_json_bytes(manifest)
        _write_fsynced_file(staging / "manifest.json", manifest_bytes, mode=0o600)
        _write_fsynced_file(
            staging / "state.json",
            _canonical_json_bytes(
                self._state(
                    AttemptPhase.PREPARED,
                    manifest_sha256=_sha256(manifest_bytes),
                )
            ),
            mode=0o600,
        )
        _fsync_directory(staging)
        for snapshot in snapshots:
            self._require_output_identity(
                snapshot,
                None,
                permitted=frozenset({"original"}),
            )
        if self.transaction_directory.exists() or self.transaction_directory.is_symlink():
            raise AttemptTransactionError("an attempt appeared while PREPARED bytes were staged")
        os.replace(staging, self.transaction_directory)
        _fsync_directory(self.transaction_directory.parent)
        return self.inspect()

    def bind_test(self) -> AttemptInspection:
        """Cross the one-way durable boundary immediately before the first Test open."""

        loaded = self._load()
        if loaded.phase is not AttemptPhase.PREPARED:
            raise AttemptTransactionError("Test can be bound only from PREPARED")
        self._transition(
            AttemptPhase.TEST_BOUND,
            manifest_sha256=loaded.state["manifest_sha256"],
            evidence=None,
        )
        self._bound_in_process = True
        self._bound_process_id = os.getpid()
        return self.inspect()

    def append_episode_bundle(
        self,
        record: EpisodeJournalRecord,
        trajectory_payload: bytes,
    ) -> AttemptInspection:
        """Persist one canonical trajectory, then commit its next journal record."""

        if not isinstance(record, EpisodeJournalRecord):
            raise TypeError("record must be an EpisodeJournalRecord")
        if not isinstance(trajectory_payload, bytes):
            raise TypeError("trajectory_payload must be bytes")
        if (
            len(trajectory_payload) != record.trajectory_blob_size_bytes
            or _sha256(trajectory_payload) != record.trajectory_blob_sha256
        ):
            raise AttemptTransactionError("trajectory payload differs from its journal identity")
        self._require_bound_process()
        loaded = self._load()
        if loaded.phase is not AttemptPhase.TEST_BOUND:
            raise AttemptTransactionError("episode bundles may be appended only while TEST_BOUND")
        self._validate_next_episode_identity(record, loaded.journal)
        self.write_blob(record.trajectory_blob_path, trajectory_payload)
        return self._append_episode_record(record)

    def _validate_next_episode_identity(
        self,
        record: EpisodeJournalRecord,
        journal: Sequence[EpisodeJournalRecord],
    ) -> None:
        expected_keys = self._expected_episode_keys()
        if len(journal) >= len(expected_keys):
            raise AttemptTransactionError("the formal journal already contains all 60 records")
        expected = expected_keys[len(journal)]
        actual = (record.controller, record.row_index)
        if actual != expected:
            raise AttemptTransactionError(
                f"episode record is out of order; expected {expected!r}, got {actual!r}"
            )
        if record.controller == "pid":
            prior_track_ids = {item.track_id for item in journal}
            if record.track_id in prior_track_ids:
                raise AttemptTransactionError("PID Test rows must use 20 distinct Track IDs")
        else:
            reference = journal[record.row_index]
            shared_identity = (
                record.track_id,
                record.reset_seed,
                record.episode_seed,
                record.controller_seed,
            )
            expected_identity = (
                reference.track_id,
                reference.reset_seed,
                reference.episode_seed,
                reference.controller_seed,
            )
            if shared_identity != expected_identity:
                raise AttemptTransactionError(
                    "all Controllers must reuse the PID row's Track and seed identity"
                )

    def _append_episode_record(self, record: EpisodeJournalRecord) -> AttemptInspection:
        """Append the commit record only after its immutable trajectory blob exists."""

        if not isinstance(record, EpisodeJournalRecord):
            raise TypeError("record must be an EpisodeJournalRecord")
        self._require_bound_process()
        loaded = self._load()
        if loaded.phase is not AttemptPhase.TEST_BOUND:
            raise AttemptTransactionError("episode records may be appended only while TEST_BOUND")
        self._validate_next_episode_identity(record, loaded.journal)
        matching_blob = [
            item for item in loaded.blobs if item.relative_path == record.trajectory_blob_path
        ]
        if len(matching_blob) != 1:
            raise AttemptTransactionError("episode journal requires one indexed trajectory blob")
        blob = matching_blob[0]
        if (
            blob.sha256 != record.trajectory_blob_sha256
            or blob.size_bytes != record.trajectory_blob_size_bytes
            or blob.mode != 0o600
        ):
            raise AttemptTransactionError("episode trajectory blob identity differs")
        payload = _canonical_json_bytes(record.to_dict())
        _require_regular_file(self._journal_path, label="episode journal")
        descriptor = os.open(
            self._journal_path,
            os.O_WRONLY | os.O_APPEND | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0),
        )
        try:
            metadata = os.fstat(descriptor)
            if not stat.S_ISREG(metadata.st_mode):
                raise AttemptTransactionTamperError("episode journal descriptor is not regular")
            view = memoryview(payload)
            while view:
                written = os.write(descriptor, view)
                if written < 1:
                    raise OSError("episode journal append made no progress")
                view = view[written:]
            os.fsync(descriptor)
        finally:
            os.close(descriptor)
        _fsync_directory(self.transaction_directory)
        inspection = self.inspect()
        if inspection.journal_record_count != len(loaded.journal) + 1:
            raise AttemptTransactionTamperError("episode journal append failed exact readback")
        return inspection

    def write_blob(
        self,
        relative_path: str,
        payload: bytes,
        *,
        mode: int = 0o600,
    ) -> BlobRecord:
        """Write one immutable deterministic binary evidence blob and fsync its index."""

        self._require_bound_process()
        loaded = self._load()
        if loaded.phase is not AttemptPhase.TEST_BOUND:
            raise AttemptTransactionError("binary blobs may be written only while TEST_BOUND")
        relative = _safe_relative_path(relative_path, field_name="blob relative_path")
        permissions = _mode(mode, field_name="blob mode")
        if not isinstance(payload, bytes):
            raise TypeError("blob payload must be bytes")
        existing = {record.relative_path for record in loaded.blobs}
        if relative in existing:
            raise AttemptTransactionError("binary evidence blobs are immutable")
        parent_relative = PurePosixPath(relative).parent
        current = self._blobs_directory
        for part in parent_relative.parts:
            candidate = current / part
            try:
                metadata = candidate.lstat()
            except FileNotFoundError:
                os.mkdir(candidate, 0o700)
                _fsync_directory(current)
            else:
                if stat.S_ISLNK(metadata.st_mode) or not stat.S_ISDIR(metadata.st_mode):
                    raise AttemptTransactionTamperError(
                        "blob parents must be non-symlink directories"
                    )
            current = candidate
        destination = self._blobs_directory / relative
        if destination.exists() or destination.is_symlink():
            raise AttemptTransactionTamperError("unindexed blob path already exists")
        _write_fsynced_file(destination, payload, mode=permissions)
        record = BlobRecord(relative, _sha256(payload), len(payload), permissions)
        index_payload = _canonical_json_bytes(
            {
                "mode": record.mode,
                "relative_path": record.relative_path,
                "sha256": record.sha256,
                "size_bytes": record.size_bytes,
            }
        )
        descriptor = os.open(
            self._blob_index_path,
            os.O_WRONLY | os.O_APPEND | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0),
        )
        try:
            view = memoryview(index_payload)
            while view:
                written = os.write(descriptor, view)
                if written < 1:
                    raise OSError("blob index append made no progress")
                view = view[written:]
            os.fsync(descriptor)
        finally:
            os.close(descriptor)
        _fsync_directory(self.transaction_directory)
        verified = self._load_blobs()
        if verified[-1] != record:
            raise AttemptTransactionTamperError("blob index append failed exact readback")
        return record

    def write_execution_evidence(self, payload: bytes) -> BlobRecord:
        """Seal canonical post-close runtime evidence after all 60 episodes are durable.

        Only the process that crossed :meth:`bind_test` may create this seal.  A replacement
        process can therefore finalize a fully sealed attempt, but cannot manufacture missing
        lifecycle, asset-access, runtime, or memory evidence after losing the Test process.
        """

        if not isinstance(payload, bytes):
            raise TypeError("execution evidence payload must be bytes")
        self._require_bound_process()
        loaded = self._load()
        if loaded.phase is not AttemptPhase.TEST_BOUND:
            raise AttemptTransactionError("execution evidence may be sealed only while TEST_BOUND")
        if len(loaded.journal) != FORMAL_EPISODE_COUNT:
            raise AttemptTransactionError(
                "execution evidence requires exactly 60 durable episode records"
            )
        try:
            value = json.loads(payload)
        except (UnicodeDecodeError, json.JSONDecodeError) as error:
            raise AttemptTransactionError("execution evidence is not strict JSON") from error
        if not isinstance(value, Mapping):
            raise AttemptTransactionError("execution evidence root must be an object")
        try:
            canonical = canonical_execution_evidence_bytes(value)
        except (TypeError, ValueError, AttemptTransactionTamperError) as error:
            raise AttemptTransactionError(
                "execution evidence differs from the frozen schema"
            ) from error
        if canonical != payload:
            raise AttemptTransactionError("execution evidence must use canonical JSON bytes")
        return self.write_blob(FORMAL_EXECUTION_EVIDENCE_BLOB_PATH, payload, mode=0o600)

    def read_execution_evidence(self) -> Mapping[str, Any]:
        """Return the verified canonical execution-evidence object."""

        payload = self.read_blob(FORMAL_EXECUTION_EVIDENCE_BLOB_PATH)
        try:
            value = json.loads(payload)
        except (
            UnicodeDecodeError,
            json.JSONDecodeError,
        ) as error:  # pragma: no cover - loader gate
            raise AttemptTransactionTamperError("execution evidence is not strict JSON") from error
        if not isinstance(value, Mapping):  # pragma: no cover - loader gate
            raise AttemptTransactionTamperError("execution evidence root differs")
        return MappingProxyType(dict(value))

    def read_blob(self, relative_path: str) -> bytes:
        """Read one indexed blob only after verifying its durable identity and all indexes."""

        relative = _safe_relative_path(relative_path, field_name="blob relative_path")
        loaded = self._load()
        matching = [record for record in loaded.blobs if record.relative_path == relative]
        if len(matching) != 1:
            raise AttemptTransactionError("blob path is not present exactly once in the index")
        path = self._blobs_directory / relative
        payload = path.read_bytes()
        record = matching[0]
        if (
            len(payload) != record.size_bytes
            or _sha256(payload) != record.sha256
            or stat.S_IMODE(path.stat().st_mode) != record.mode
        ):
            raise AttemptTransactionTamperError("binary evidence blob failed verified readback")
        return payload

    def episode_records(self) -> tuple[EpisodeJournalRecord, ...]:
        """Return the fully verified durable journal prefix without exposing private paths."""

        return self._load().journal

    def verified_blob_path(self, relative_path: str) -> Path:
        """Return an indexed regular-file path only after exact durable readback."""

        relative = _safe_relative_path(relative_path, field_name="blob relative_path")
        self.read_blob(relative)
        path = self._blobs_directory / relative
        _require_regular_file(path, label="verified binary evidence blob")
        return path

    def read_staged_outputs(self) -> Mapping[str, bytes]:
        """Return every verified staged output byte after durable staging exists."""

        loaded = self._load()
        if loaded.phase not in {
            AttemptPhase.EVALUATION_COMPLETE,
            AttemptPhase.ARTIFACTS_VALIDATED,
            AttemptPhase.COMMITTED,
        }:
            raise AttemptTransactionError("staged outputs are unavailable before completion")
        values: dict[str, bytes] = {}
        for index, record in enumerate(loaded.staged):
            path = self._staged_directory / f"{index:03d}.bin"
            payload = path.read_bytes()
            if len(payload) != record.size_bytes or _sha256(payload) != record.sha256:
                raise AttemptTransactionTamperError("staged output failed verified readback")
            values[record.relative_path] = payload
        return MappingProxyType(values)

    def complete_evaluation(
        self,
        outputs: Mapping[str, bytes] | None = None,
        *,
        modes: Mapping[str, int] | None = None,
    ) -> AttemptInspection:
        """Durably stage the complete allowlist, then seal the 60-record workload.

        A resumed process may pass ``outputs=None`` only when a complete staged directory already
        survived.  It cannot append or rerun episodes after process loss.
        """

        loaded = self._load()
        if loaded.phase is not AttemptPhase.TEST_BOUND:
            raise AttemptTransactionError("evaluation can complete only from TEST_BOUND")
        if len(loaded.journal) != FORMAL_EPISODE_COUNT:
            raise AttemptTransactionError("evaluation requires exactly 60 durable episode records")
        execution_evidence = self._execution_evidence_record(loaded.blobs)
        if execution_evidence is None:
            raise AttemptTransactionError(
                "evaluation requires the durable post-close execution evidence seal"
            )
        if self._staged_build_directory.exists() or self._staged_build_directory.is_symlink():
            self._remove_tree(self._staged_build_directory)

        if self._staged_directory.exists() or self._staged_directory.is_symlink():
            staged = self._load_staged()
            if outputs is not None:
                self._verify_supplied_outputs(outputs, modes=modes, staged=staged)
        else:
            if outputs is None:
                raise AttemptTransactionError("no complete staged output set survived")
            normalized = self._normalize_outputs(outputs, modes=modes)
            self._build_staged_outputs(normalized)
            staged = self._load_staged()
        evidence = {
            "blob_index_sha256": _sha256(self._blob_index_path.read_bytes()),
            "episode_journal_sha256": _sha256(self._journal_path.read_bytes()),
            "execution_evidence_sha256": execution_evidence.sha256,
            "staged_manifest_sha256": _sha256(
                (self._staged_directory / "manifest.json").read_bytes()
            ),
        }
        self._transition(
            AttemptPhase.EVALUATION_COMPLETE,
            manifest_sha256=loaded.state["manifest_sha256"],
            evidence=evidence,
        )
        self._bound_in_process = False
        self._bound_process_id = None
        if len(staged) != len(self.output_allowlist):  # pragma: no cover - guarded by loader
            raise AttemptTransactionTamperError("staged output coverage differs")
        return self.inspect()

    def mark_artifacts_validated(
        self,
        *,
        semantic_validation_sha256: str,
        validator_id: str = FORMAL_ARTIFACT_VALIDATOR_ID,
    ) -> AttemptInspection:
        """Bind a named semantic-validator digest before publication is permitted."""

        loaded = self._load()
        if loaded.phase is not AttemptPhase.EVALUATION_COMPLETE:
            raise AttemptTransactionError(
                "artifacts can be validated only after EVALUATION_COMPLETE"
            )
        if (
            not isinstance(semantic_validation_sha256, str)
            or _SHA256_PATTERN.fullmatch(semantic_validation_sha256) is None
        ):
            raise ValueError("semantic_validation_sha256 must be a lowercase SHA-256 digest")
        if validator_id != FORMAL_ARTIFACT_VALIDATOR_ID:
            raise ValueError("validator_id must identify the frozen M8 semantic validator")
        staged_manifest_sha256 = _sha256((self._staged_directory / "manifest.json").read_bytes())
        attestation = _canonical_json_bytes(
            {
                "schema_version": ARTIFACT_VALIDATION_SCHEMA_VERSION,
                "semantic_validation_sha256": semantic_validation_sha256,
                "staged_manifest_sha256": staged_manifest_sha256,
                "status": "passed",
                "validated_output_count": len(loaded.staged),
                "validator_id": validator_id,
            }
        )
        if self._artifact_validation_path.exists() or self._artifact_validation_path.is_symlink():
            _require_regular_file(
                self._artifact_validation_path,
                label="artifact validation attestation",
            )
            if self._artifact_validation_path.read_bytes() != attestation:
                raise AttemptTransactionTamperError(
                    "surviving artifact validation attestation differs"
                )
        else:
            _write_fsynced_file(self._artifact_validation_path, attestation, mode=0o600)
        evidence = dict(loaded.state["evidence"])
        evidence["artifact_validation_sha256"] = _sha256(attestation)
        self._transition(
            AttemptPhase.ARTIFACTS_VALIDATED,
            manifest_sha256=loaded.state["manifest_sha256"],
            evidence=evidence,
        )
        return self.inspect()

    def publish_and_commit(
        self,
        *,
        retain_committed_transaction: bool = True,
    ) -> tuple[PublishedOutput, ...]:
        """Publish every staged byte and durably enter ``COMMITTED``.

        The safe default retains the completed transaction for permanent formal readback.  A
        maintenance caller may explicitly pass ``retain_committed_transaction=False`` or later
        call :meth:`retire_committed`; the formal CLI never does so.
        """

        if type(retain_committed_transaction) is not bool:
            raise TypeError("retain_committed_transaction must be a boolean")

        loaded = self._load()
        if loaded.phase is not AttemptPhase.ARTIFACTS_VALIDATED:
            raise AttemptTransactionError("publication requires ARTIFACTS_VALIDATED")
        if loaded.output_state != "original":
            raise AttemptTransactionError("recover() must restore partial publication first")
        if self._publication_directory.exists() or self._publication_directory.is_symlink():
            self._remove_tree(self._publication_directory)
        os.mkdir(self._publication_directory, 0o700)
        _fsync_directory(self.transaction_directory)
        try:
            for index, record in enumerate(loaded.staged):
                staged_path = self._staged_directory / f"{index:03d}.bin"
                payload = staged_path.read_bytes()
                temporary = self._publication_directory / f"{index:03d}.publish"
                _write_fsynced_file(temporary, payload, mode=record.mode)
                destination = self._output_path(record.relative_path)
                self._require_output_identity(
                    loaded.snapshots[index],
                    record,
                    permitted=frozenset({"original"}),
                )
                os.replace(temporary, destination)
                _fsync_directory(self._publication_directory)
                _fsync_directory(destination.parent)
                _require_regular_file(destination, label="published output")
                if (
                    destination.read_bytes() != payload
                    or stat.S_IMODE(destination.stat().st_mode) != record.mode
                ):
                    raise AttemptTransactionTamperError("published output failed exact readback")
            completed = self._load()
            if completed.output_state != "published":
                raise AttemptTransactionTamperError("not every formal output was published")
            self._transition(
                AttemptPhase.COMMITTED,
                manifest_sha256=loaded.state["manifest_sha256"],
                evidence=loaded.state["evidence"],
            )
            committed = self._load()
            if committed.output_state != "published":
                raise AttemptTransactionTamperError("COMMITTED output readback differs")
            published = committed.staged
            if not retain_committed_transaction:
                self._retire_transaction()
            return published
        except BaseException:
            # The ARTIFACTS_VALIDATED state and immutable final-staged bytes intentionally remain.
            raise

    def retire_committed(self) -> tuple[PublishedOutput, ...]:
        """Retire a verified COMMITTED transaction after all caller-owned gates pass."""

        loaded = self._load()
        if loaded.phase is not AttemptPhase.COMMITTED:
            raise AttemptTransactionError("only a COMMITTED transaction can be retired")
        if loaded.output_state != "published":
            raise AttemptTransactionTamperError("COMMITTED outputs differ from staged bytes")
        published = loaded.staged
        self._retire_transaction()
        return published

    def inspect(self) -> AttemptInspection:
        """Validate all transaction bytes and report the safe next boundary."""

        if not self.transaction_directory.exists() and not self.transaction_directory.is_symlink():
            return AttemptInspection(False, None, 0, None, (), (), "absent")
        loaded = self._load()
        keys = self._expected_episode_keys()
        next_episode = keys[len(loaded.journal)] if len(loaded.journal) < len(keys) else None
        return AttemptInspection(
            exists=True,
            phase=loaded.phase,
            journal_record_count=len(loaded.journal),
            next_episode=next_episode,
            blob_records=loaded.blobs,
            staged_outputs=loaded.staged,
            output_state=loaded.output_state,
        )

    def recover(self) -> AttemptRecovery:
        """Apply only recovery actions permitted by the durable phase.

        Incomplete Test-bound attempts raise :class:`IncompleteTestAttemptError` and are preserved.
        Complete attempts return to validation/publication from durable bytes without executing a
        workload.  Performance values are deliberately absent from every recovery decision.
        """

        transaction_exists = (
            self.transaction_directory.exists() or self.transaction_directory.is_symlink()
        )
        if self.cleanup_directory.exists() or self.cleanup_directory.is_symlink():
            self._remove_tree(self.cleanup_directory)
        if not transaction_exists:
            return AttemptRecovery("none", None, 0)
        self._bound_in_process = False
        self._bound_process_id = None
        loaded = self._load()
        count = len(loaded.journal)
        if loaded.phase is AttemptPhase.PREPARED:
            self._remove_created_output_directories(loaded)
            self._retire_transaction()
            return AttemptRecovery("pre_test_restored", loaded.phase, count)
        if loaded.phase is AttemptPhase.TEST_BOUND:
            inspection = self.inspect()
            if (
                count < FORMAL_EPISODE_COUNT
                or self._execution_evidence_record(loaded.blobs) is None
            ):
                raise IncompleteTestAttemptError(inspection)
            # All workload records and post-close evidence are durable.  Only deterministic
            # artifact finalization remains.
            return AttemptRecovery("evaluation_complete_ready", loaded.phase, count)
        if loaded.phase is AttemptPhase.EVALUATION_COMPLETE:
            if loaded.output_state != "original":
                raise AttemptTransactionTamperError(
                    "outputs changed before artifact validation/publication"
                )
            return AttemptRecovery("evaluation_complete_ready", loaded.phase, count)
        if loaded.phase is AttemptPhase.ARTIFACTS_VALIDATED:
            if loaded.output_state == "partial_publication" or loaded.output_state == "published":
                self._restore_originals(loaded)
                recovered = self._load()
                if recovered.output_state != "original":
                    raise AttemptTransactionTamperError(
                        "partial publication recovery failed exact readback"
                    )
                if self._publication_directory.exists():
                    self._remove_tree(self._publication_directory)
                return AttemptRecovery("partial_publication_restored", loaded.phase, count)
            if self._publication_directory.exists():
                self._remove_tree(self._publication_directory)
            return AttemptRecovery("artifacts_validated_ready", loaded.phase, count)
        if loaded.phase is AttemptPhase.COMMITTED:
            if loaded.output_state != "published":
                raise AttemptTransactionTamperError("COMMITTED outputs differ from staged bytes")
            return AttemptRecovery("committed_retained", loaded.phase, count)
        raise AssertionError("unreachable attempt phase")

    def _remove_atomic_state_residue(self) -> None:
        candidates: list[tuple[Path, int, int, int]] = []
        for path in self.transaction_directory.iterdir():
            if not path.name.startswith("."):
                continue
            if _STATE_TEMPORARY_PATTERN.fullmatch(path.name) is None:
                raise AttemptTransactionTamperError(
                    "transaction directory contains unsafe dot residue"
                )
            try:
                metadata = path.lstat()
            except FileNotFoundError:
                continue
            if stat.S_ISLNK(metadata.st_mode) or not stat.S_ISREG(metadata.st_mode):
                raise AttemptTransactionTamperError(
                    "state temporary residue must be a non-symlink regular file"
                )
            candidates.append((path, metadata.st_dev, metadata.st_ino, metadata.st_mode))
        for path, device, inode, mode in candidates:
            try:
                metadata = path.lstat()
            except FileNotFoundError:
                continue
            if (
                stat.S_ISLNK(metadata.st_mode)
                or not stat.S_ISREG(metadata.st_mode)
                or (metadata.st_dev, metadata.st_ino, metadata.st_mode) != (device, inode, mode)
            ):
                raise AttemptTransactionTamperError(
                    "state temporary residue changed before cleanup"
                )
            path.unlink()
        if candidates:
            _fsync_directory(self.transaction_directory)

    def _load(self) -> _LoadedTransaction:
        _require_directory(self.transaction_directory, label="attempt transaction")
        self._remove_atomic_state_residue()
        manifest = _read_canonical_json(self._manifest_path, field_name="transaction manifest")
        state = _read_canonical_json(self._state_path, field_name="transaction state")
        snapshots = self._validate_manifest(manifest)
        phase = self._validate_state(state, manifest)
        self._validate_top_level_entries(phase)
        journal = self._load_journal()
        blobs = self._load_blobs()
        self._validate_episode_blob_coverage(journal, blobs)
        execution_evidence = self._execution_evidence_record(blobs)
        if execution_evidence is not None:
            self._validate_execution_evidence_blob(execution_evidence)
        staged = self._load_staged() if self._staged_directory.exists() else ()
        evidence = state["evidence"]
        if phase is AttemptPhase.PREPARED:
            if journal or blobs or staged:
                raise AttemptTransactionTamperError("PREPARED attempt contains Test evidence")
        elif phase is AttemptPhase.TEST_BOUND:
            if len(journal) > FORMAL_EPISODE_COUNT:
                raise AttemptTransactionTamperError("Test journal exceeds 60 records")
            if execution_evidence is not None and len(journal) != FORMAL_EPISODE_COUNT:
                raise AttemptTransactionTamperError(
                    "execution evidence was sealed before all 60 episode records"
                )
        else:
            if (
                len(journal) != FORMAL_EPISODE_COUNT
                or execution_evidence is None
                or len(staged) != len(self.output_allowlist)
            ):
                raise AttemptTransactionTamperError(
                    "completed attempt lacks exact journal, execution-evidence, or staged-output "
                    "coverage"
                )
            expected_evidence_keys = {
                "blob_index_sha256",
                "episode_journal_sha256",
                "execution_evidence_sha256",
                "staged_manifest_sha256",
            }
            if phase in {AttemptPhase.ARTIFACTS_VALIDATED, AttemptPhase.COMMITTED}:
                expected_evidence_keys.add("artifact_validation_sha256")
            if not isinstance(evidence, Mapping) or set(evidence) != expected_evidence_keys:
                raise AttemptTransactionTamperError("completed evidence identity differs")
            expected_evidence = {
                "blob_index_sha256": _sha256(self._blob_index_path.read_bytes()),
                "episode_journal_sha256": _sha256(self._journal_path.read_bytes()),
                "execution_evidence_sha256": execution_evidence.sha256,
                "staged_manifest_sha256": _sha256(
                    (self._staged_directory / "manifest.json").read_bytes()
                ),
            }
            if phase in {AttemptPhase.ARTIFACTS_VALIDATED, AttemptPhase.COMMITTED}:
                attestation = _read_canonical_json(
                    self._artifact_validation_path,
                    field_name="artifact validation attestation",
                )
                _exact_keys(
                    attestation,
                    {
                        "schema_version",
                        "semantic_validation_sha256",
                        "staged_manifest_sha256",
                        "status",
                        "validated_output_count",
                        "validator_id",
                    },
                    field_name="artifact validation attestation",
                )
                if (
                    attestation["schema_version"] != ARTIFACT_VALIDATION_SCHEMA_VERSION
                    or attestation["status"] != "passed"
                    or attestation["validator_id"] != FORMAL_ARTIFACT_VALIDATOR_ID
                    or not isinstance(attestation["semantic_validation_sha256"], str)
                    or _SHA256_PATTERN.fullmatch(attestation["semantic_validation_sha256"]) is None
                    or attestation["staged_manifest_sha256"]
                    != expected_evidence["staged_manifest_sha256"]
                    or attestation["validated_output_count"] != len(self.output_allowlist)
                ):
                    raise AttemptTransactionTamperError(
                        "artifact validation attestation differs from staged outputs"
                    )
                expected_evidence["artifact_validation_sha256"] = _sha256(
                    self._artifact_validation_path.read_bytes()
                )
            if dict(evidence) != expected_evidence:
                raise AttemptTransactionTamperError("completed evidence bytes changed")
        output_state = self._classify_outputs(
            snapshots,
            staged,
            removable_directories=(
                frozenset(manifest["created_output_directories"])
                if phase is AttemptPhase.PREPARED
                else frozenset()
            ),
        )
        if (
            phase
            in {
                AttemptPhase.PREPARED,
                AttemptPhase.TEST_BOUND,
                AttemptPhase.EVALUATION_COMPLETE,
            }
            and output_state != "original"
        ):
            raise AttemptTransactionTamperError("formal outputs changed before publication")
        return _LoadedTransaction(
            phase=phase,
            state=state,
            manifest=manifest,
            snapshots=snapshots,
            journal=journal,
            blobs=blobs,
            staged=staged,
            output_state=output_state,
        )

    def _validate_manifest(self, manifest: Mapping[str, Any]) -> tuple[_OutputSnapshot, ...]:
        _exact_keys(
            manifest,
            {
                "created_output_directories",
                "episode_protocol",
                "identity",
                "output_allowlist",
                "outputs",
                "recovery_policy",
                "schema_version",
                "transaction_relative_path",
            },
            field_name="transaction manifest",
        )
        expected_protocol = {
            "controller_order": list(FORMAL_CONTROLLER_ORDER),
            "expected_record_count": FORMAL_EPISODE_COUNT,
            "ordering": "controller_major_then_row_index",
            "rows_per_controller": FORMAL_ROWS_PER_CONTROLLER,
        }
        expected_recovery = {
            "accepted_result": "first_complete_protocol_passing_attempt",
            "automatic_retry_after_test_bound": False,
            "completed_attempt_finalizes_from_durable_bytes_only": True,
            "low_performance_can_trigger_retry": False,
            "partial_publication_restores_originals_before_republish": True,
        }
        if (
            manifest["schema_version"] != ATTEMPT_TRANSACTION_SCHEMA_VERSION
            or manifest["identity"] != self.identity.to_dict()
            or manifest["transaction_relative_path"] != self.transaction_relative_path
            or manifest["output_allowlist"] != list(self.output_allowlist)
            or manifest["episode_protocol"] != expected_protocol
            or manifest["recovery_policy"] != expected_recovery
        ):
            raise AttemptTransactionTamperError("transaction manifest identity differs")
        created = manifest["created_output_directories"]
        if not isinstance(created, list) or any(
            not isinstance(value, str)
            or _safe_relative_path(value, field_name="created directory") != value
            for value in created
        ):
            raise AttemptTransactionTamperError("created output directory list differs")
        permitted_created_directories: set[str] = set()
        for relative_path in self.output_allowlist:
            current = PurePosixPath()
            for part in PurePosixPath(relative_path).parts[:-1]:
                current /= part
                permitted_created_directories.add(current.as_posix())
        if len(created) != len(set(created)) or not set(created).issubset(
            permitted_created_directories
        ):
            raise AttemptTransactionTamperError("created output directory identity differs")
        outputs = manifest["outputs"]
        if not isinstance(outputs, list) or len(outputs) != len(self.output_allowlist):
            raise AttemptTransactionTamperError("output backup coverage differs")
        backup_directory = self.transaction_directory / "backups"
        _require_directory(backup_directory, label="backup directory")
        expected_backup_names: set[str] = set()
        snapshots: list[_OutputSnapshot] = []
        for index, (record, expected_path) in enumerate(
            zip(outputs, self.output_allowlist, strict=True)
        ):
            if not isinstance(record, Mapping):
                raise AttemptTransactionTamperError("output backup record is not an object")
            _exact_keys(
                record,
                {
                    "backup_relative_path",
                    "existed",
                    "mode",
                    "relative_path",
                    "sha256",
                    "size_bytes",
                },
                field_name="output backup record",
            )
            if record["relative_path"] != expected_path or type(record["existed"]) is not bool:
                raise AttemptTransactionTamperError("output backup identity differs")
            if not record["existed"]:
                if (
                    record["backup_relative_path"] is not None
                    or record["mode"] is not None
                    or record["sha256"] is not None
                    or record["size_bytes"] != 0
                ):
                    raise AttemptTransactionTamperError("absent output backup record differs")
                snapshots.append(_OutputSnapshot(expected_path, None, None))
                continue
            backup_name = f"{index:03d}.bin"
            expected_backup_names.add(backup_name)
            if (
                record["backup_relative_path"] != f"backups/{backup_name}"
                or type(record["mode"]) is not int
                or not 0 <= record["mode"] <= 0o777
                or not isinstance(record["sha256"], str)
                or _SHA256_PATTERN.fullmatch(record["sha256"]) is None
                or type(record["size_bytes"]) is not int
                or record["size_bytes"] < 0
            ):
                raise AttemptTransactionTamperError("existing output backup record differs")
            backup = backup_directory / backup_name
            _require_regular_file(backup, label="output backup")
            if stat.S_IMODE(backup.stat().st_mode) != 0o600:
                raise AttemptTransactionTamperError("output backup mode changed")
            content = backup.read_bytes()
            if len(content) != record["size_bytes"] or _sha256(content) != record["sha256"]:
                raise AttemptTransactionTamperError("output backup bytes changed")
            snapshots.append(_OutputSnapshot(expected_path, content, record["mode"]))
        actual_backup_names = {path.name for path in backup_directory.iterdir()}
        if actual_backup_names != expected_backup_names:
            raise AttemptTransactionTamperError("backup directory contains residue")
        return tuple(snapshots)

    def _validate_state(
        self, state: Mapping[str, Any], manifest: Mapping[str, Any]
    ) -> AttemptPhase:
        _exact_keys(
            state,
            {
                "evidence",
                "identity",
                "manifest_sha256",
                "phase",
                "phase_index",
                "schema_version",
            },
            field_name="transaction state",
        )
        try:
            phase = AttemptPhase(state["phase"])
        except (TypeError, ValueError) as error:
            raise AttemptTransactionTamperError("transaction phase is invalid") from error
        manifest_bytes = _canonical_json_bytes(manifest)
        if (
            state["schema_version"] != ATTEMPT_TRANSACTION_SCHEMA_VERSION
            or state["identity"] != self.identity.to_dict()
            or state["manifest_sha256"] != _sha256(manifest_bytes)
            or state["phase_index"] != _PHASE_INDEX[phase.value]
        ):
            raise AttemptTransactionTamperError("transaction state identity differs")
        if (
            phase in {AttemptPhase.PREPARED, AttemptPhase.TEST_BOUND}
            and state["evidence"] is not None
        ):
            raise AttemptTransactionTamperError("unfinished phase has completion evidence")
        return phase

    def _validate_top_level_entries(self, phase: AttemptPhase) -> None:
        expected = {
            "backups",
            "blob-index.jsonl",
            "blobs",
            "episode-journal.jsonl",
            "manifest.json",
            "state.json",
        }
        validation_exists = (
            self._artifact_validation_path.exists() or self._artifact_validation_path.is_symlink()
        )
        if validation_exists:
            if phase not in {
                AttemptPhase.EVALUATION_COMPLETE,
                AttemptPhase.ARTIFACTS_VALIDATED,
                AttemptPhase.COMMITTED,
            }:
                raise AttemptTransactionTamperError(
                    "artifact validation residue exists before evaluation completion"
                )
            expected.add("artifact-validation.json")
        elif phase in {AttemptPhase.ARTIFACTS_VALIDATED, AttemptPhase.COMMITTED}:
            expected.add("artifact-validation.json")
        if self._staged_directory.exists() or self._staged_directory.is_symlink():
            expected.add("final-staged")
        if self._staged_build_directory.exists() or self._staged_build_directory.is_symlink():
            if phase is not AttemptPhase.TEST_BOUND:
                raise AttemptTransactionTamperError("staged build residue is out of phase")
            expected.add("final-staged.build")
        if self._publication_directory.exists() or self._publication_directory.is_symlink():
            if phase not in {
                AttemptPhase.PREPARED,
                AttemptPhase.ARTIFACTS_VALIDATED,
                AttemptPhase.COMMITTED,
            }:
                raise AttemptTransactionTamperError("publication residue is out of phase")
            expected.add("publication")
        actual = {path.name for path in self.transaction_directory.iterdir()}
        if actual != expected:
            raise AttemptTransactionTamperError("transaction directory contains residue")
        for directory_name in expected & {
            "backups",
            "blobs",
            "final-staged",
            "final-staged.build",
            "publication",
        }:
            _require_directory(
                self.transaction_directory / directory_name,
                label=f"transaction {directory_name}",
            )
        if self._staged_build_directory.exists():
            _walk_without_symlinks(self._staged_build_directory)
            allowed_build_names = {"manifest.json"} | {
                f"{index:03d}.bin" for index in range(len(self.output_allowlist))
            }
            build_entries = tuple(self._staged_build_directory.iterdir())
            if any(path.is_dir() or path.name not in allowed_build_names for path in build_entries):
                raise AttemptTransactionTamperError("staged build directory contains residue")
        if self._publication_directory.exists():
            _walk_without_symlinks(self._publication_directory)
            for path in self._publication_directory.iterdir():
                match = _PUBLICATION_SCRATCH_PATTERN.fullmatch(path.name)
                if (
                    path.is_dir()
                    or match is None
                    or int(match.group(1)) >= len(self.output_allowlist)
                    or (phase is AttemptPhase.PREPARED and match.group(2) != "restore")
                ):
                    raise AttemptTransactionTamperError("publication directory contains residue")

    def _load_journal(self) -> tuple[EpisodeJournalRecord, ...]:
        _require_regular_file(self._journal_path, label="episode journal")
        payload = self._journal_path.read_bytes()
        if payload and not payload.endswith(b"\n"):
            raise AttemptTransactionTamperError("episode journal has a partial final record")
        records: list[EpisodeJournalRecord] = []
        expected_keys = self._expected_episode_keys()
        for index, line in enumerate(payload.splitlines(keepends=True)):
            try:
                value = json.loads(line)
            except (UnicodeDecodeError, json.JSONDecodeError) as error:
                raise AttemptTransactionTamperError("episode journal is not JSON Lines") from error
            if not isinstance(value, Mapping) or _canonical_json_bytes(value) != line:
                raise AttemptTransactionTamperError("episode journal record is not canonical")
            _exact_keys(
                value,
                {
                    "controller",
                    "controller_seed",
                    "data",
                    "episode_seed",
                    "outcome",
                    "reset_seed",
                    "row_index",
                    "schema_version",
                    "steps",
                    "track_id",
                    "trajectory_blob_path",
                    "trajectory_blob_sha256",
                    "trajectory_blob_size_bytes",
                },
                field_name="episode journal record",
            )
            try:
                record = EpisodeJournalRecord(**dict(value))
            except (TypeError, ValueError) as error:
                raise AttemptTransactionTamperError("episode journal record is invalid") from error
            if (
                index >= len(expected_keys)
                or (record.controller, record.row_index) != expected_keys[index]
            ):
                raise AttemptTransactionTamperError(
                    "episode journal is duplicated or out of protocol order"
                )
            try:
                self._validate_next_episode_identity(record, records)
            except AttemptTransactionError as error:
                raise AttemptTransactionTamperError(
                    "episode journal Track or seed sequence differs from the protocol"
                ) from error
            records.append(record)
        return tuple(records)

    def _validate_episode_blob_coverage(
        self,
        journal: Sequence[EpisodeJournalRecord],
        blobs: Sequence[BlobRecord],
    ) -> None:
        indexed = {record.relative_path: record for record in blobs}
        committed_paths: set[str] = set()
        for record in journal:
            blob = indexed.get(record.trajectory_blob_path)
            if (
                blob is None
                or blob.sha256 != record.trajectory_blob_sha256
                or blob.size_bytes != record.trajectory_blob_size_bytes
                or blob.mode != 0o600
            ):
                raise AttemptTransactionTamperError(
                    "episode journal is not bound to one exact trajectory blob"
                )
            committed_paths.add(record.trajectory_blob_path)
        episode_blob_paths = {
            path for path in indexed if path == "episodes" or path.startswith("episodes/")
        }
        if not committed_paths.issubset(episode_blob_paths):  # pragma: no cover - path invariant
            raise AttemptTransactionTamperError("episode trajectory path escaped its namespace")
        if len(journal) == FORMAL_EPISODE_COUNT and episode_blob_paths != committed_paths:
            raise AttemptTransactionTamperError(
                "complete journal has extra or missing episode trajectory blobs"
            )

    @staticmethod
    def _execution_evidence_record(blobs: Sequence[BlobRecord]) -> BlobRecord | None:
        matches = [
            record
            for record in blobs
            if record.relative_path == FORMAL_EXECUTION_EVIDENCE_BLOB_PATH
        ]
        if len(matches) > 1:  # pragma: no cover - duplicate paths rejected by the blob loader
            raise AttemptTransactionTamperError(
                "execution evidence appears more than once in the blob index"
            )
        return matches[0] if matches else None

    def _validate_execution_evidence_blob(self, record: BlobRecord) -> None:
        if record.mode != 0o600 or record.size_bytes < 1:
            raise AttemptTransactionTamperError("execution evidence blob identity differs")
        path = self._blobs_directory / FORMAL_EXECUTION_EVIDENCE_BLOB_PATH
        payload = path.read_bytes()
        try:
            value = json.loads(payload)
        except (UnicodeDecodeError, json.JSONDecodeError) as error:
            raise AttemptTransactionTamperError("execution evidence is not strict JSON") from error
        if not isinstance(value, Mapping):
            raise AttemptTransactionTamperError("execution evidence root differs")
        try:
            canonical = canonical_execution_evidence_bytes(value)
        except (TypeError, ValueError, AttemptTransactionTamperError) as error:
            raise AttemptTransactionTamperError(
                "execution evidence differs from the frozen schema"
            ) from error
        if canonical != payload:
            raise AttemptTransactionTamperError("execution evidence is not canonical")

    def _load_blobs(self) -> tuple[BlobRecord, ...]:
        _require_regular_file(self._blob_index_path, label="blob index")
        _require_directory(self._blobs_directory, label="blobs directory")
        payload = self._blob_index_path.read_bytes()
        if payload and not payload.endswith(b"\n"):
            raise AttemptTransactionTamperError("blob index has a partial final record")
        records: list[BlobRecord] = []
        indexed_paths: set[str] = set()
        for line in payload.splitlines(keepends=True):
            try:
                value = json.loads(line)
            except (UnicodeDecodeError, json.JSONDecodeError) as error:
                raise AttemptTransactionTamperError("blob index is not JSON Lines") from error
            if not isinstance(value, Mapping) or _canonical_json_bytes(value) != line:
                raise AttemptTransactionTamperError("blob index record is not canonical")
            _exact_keys(
                value,
                {"mode", "relative_path", "sha256", "size_bytes"},
                field_name="blob index record",
            )
            try:
                relative = _safe_relative_path(
                    value["relative_path"], field_name="blob relative_path"
                )
            except (TypeError, ValueError) as error:
                raise AttemptTransactionTamperError("blob relative path differs") from error
            if relative in indexed_paths:
                raise AttemptTransactionTamperError("blob index contains a duplicate path")
            if (
                not isinstance(value["sha256"], str)
                or _SHA256_PATTERN.fullmatch(value["sha256"]) is None
                or type(value["size_bytes"]) is not int
                or value["size_bytes"] < 0
                or type(value["mode"]) is not int
                or not 0 <= value["mode"] <= 0o777
            ):
                raise AttemptTransactionTamperError("blob index content identity differs")
            path = self._blobs_directory / relative
            _require_regular_file(path, label="binary evidence blob")
            content = path.read_bytes()
            if (
                len(content) != value["size_bytes"]
                or _sha256(content) != value["sha256"]
                or stat.S_IMODE(path.stat().st_mode) != value["mode"]
            ):
                raise AttemptTransactionTamperError("binary evidence blob bytes changed")
            indexed_paths.add(relative)
            records.append(
                BlobRecord(relative, value["sha256"], value["size_bytes"], value["mode"])
            )
        actual_files: set[str] = set()
        for root, directories, files in os.walk(
            self._blobs_directory, topdown=True, followlinks=False
        ):
            root_path = Path(root)
            for directory in directories:
                _require_directory(root_path / directory, label="blob subdirectory")
            for filename in files:
                path = root_path / filename
                _require_regular_file(path, label="blob file")
                actual_files.add(path.relative_to(self._blobs_directory).as_posix())
        if actual_files != indexed_paths:
            raise AttemptTransactionTamperError("blobs directory contains unindexed residue")
        return tuple(records)

    def _normalize_outputs(
        self,
        outputs: Mapping[str, bytes],
        *,
        modes: Mapping[str, int] | None,
    ) -> tuple[tuple[str, bytes, int], ...]:
        if not isinstance(outputs, Mapping) or set(outputs) != set(self.output_allowlist):
            raise AttemptTransactionError("staged outputs must exactly match the allowlist")
        if modes is not None and (
            not isinstance(modes, Mapping) or set(modes) != set(self.output_allowlist)
        ):
            raise AttemptTransactionError("output modes must exactly match the allowlist")
        normalized = []
        for relative in self.output_allowlist:
            payload = outputs[relative]
            if not isinstance(payload, bytes):
                raise TypeError("every staged output must be bytes")
            output_mode = (
                0o644 if modes is None else _mode(modes[relative], field_name="output mode")
            )
            normalized.append((relative, payload, output_mode))
        return tuple(normalized)

    def _build_staged_outputs(self, values: Sequence[tuple[str, bytes, int]]) -> None:
        if self._staged_directory.exists() or self._staged_directory.is_symlink():
            raise AttemptTransactionError("a complete staged output set already exists")
        os.mkdir(self._staged_build_directory, 0o700)
        _fsync_directory(self.transaction_directory)
        try:
            records = []
            for index, (relative, payload, output_mode) in enumerate(values):
                _write_fsynced_file(
                    self._staged_build_directory / f"{index:03d}.bin",
                    payload,
                    mode=output_mode,
                )
                records.append(
                    {
                        "mode": output_mode,
                        "relative_path": relative,
                        "sha256": _sha256(payload),
                        "size_bytes": len(payload),
                        "staged_file": f"{index:03d}.bin",
                    }
                )
            _write_fsynced_file(
                self._staged_build_directory / "manifest.json",
                _canonical_json_bytes(
                    {
                        "output_allowlist": list(self.output_allowlist),
                        "outputs": records,
                        "schema_version": ATTEMPT_TRANSACTION_SCHEMA_VERSION,
                    }
                ),
                mode=0o600,
            )
            _fsync_directory(self._staged_build_directory)
            os.replace(self._staged_build_directory, self._staged_directory)
            _fsync_directory(self.transaction_directory)
        except BaseException:
            if (
                self._staged_build_directory.exists()
                and not self._staged_build_directory.is_symlink()
            ):
                self._remove_tree(self._staged_build_directory)
            raise

    def _load_staged(self) -> tuple[PublishedOutput, ...]:
        _require_directory(self._staged_directory, label="final staged directory")
        manifest_path = self._staged_directory / "manifest.json"
        manifest = _read_canonical_json(manifest_path, field_name="staged manifest")
        _exact_keys(
            manifest,
            {"output_allowlist", "outputs", "schema_version"},
            field_name="staged manifest",
        )
        outputs = manifest["outputs"]
        if (
            manifest["schema_version"] != ATTEMPT_TRANSACTION_SCHEMA_VERSION
            or manifest["output_allowlist"] != list(self.output_allowlist)
            or not isinstance(outputs, list)
            or len(outputs) != len(self.output_allowlist)
        ):
            raise AttemptTransactionTamperError("staged output manifest identity differs")
        expected_names = {"manifest.json"}
        records: list[PublishedOutput] = []
        for index, (value, relative) in enumerate(zip(outputs, self.output_allowlist, strict=True)):
            if not isinstance(value, Mapping):
                raise AttemptTransactionTamperError("staged output record is not an object")
            _exact_keys(
                value,
                {"mode", "relative_path", "sha256", "size_bytes", "staged_file"},
                field_name="staged output record",
            )
            filename = f"{index:03d}.bin"
            expected_names.add(filename)
            if (
                value["relative_path"] != relative
                or value["staged_file"] != filename
                or type(value["mode"]) is not int
                or not 0 <= value["mode"] <= 0o777
                or not isinstance(value["sha256"], str)
                or _SHA256_PATTERN.fullmatch(value["sha256"]) is None
                or type(value["size_bytes"]) is not int
                or value["size_bytes"] < 0
            ):
                raise AttemptTransactionTamperError("staged output identity differs")
            path = self._staged_directory / filename
            _require_regular_file(path, label="staged output")
            payload = path.read_bytes()
            if (
                len(payload) != value["size_bytes"]
                or _sha256(payload) != value["sha256"]
                or stat.S_IMODE(path.stat().st_mode) != value["mode"]
            ):
                raise AttemptTransactionTamperError("staged output bytes changed")
            records.append(
                PublishedOutput(relative, value["sha256"], value["size_bytes"], value["mode"])
            )
        if {path.name for path in self._staged_directory.iterdir()} != expected_names:
            raise AttemptTransactionTamperError("staged output directory contains residue")
        return tuple(records)

    def _verify_supplied_outputs(
        self,
        outputs: Mapping[str, bytes],
        *,
        modes: Mapping[str, int] | None,
        staged: Sequence[PublishedOutput],
    ) -> None:
        normalized = self._normalize_outputs(outputs, modes=modes)
        for (relative, payload, output_mode), record in zip(normalized, staged, strict=True):
            if (
                relative != record.relative_path
                or _sha256(payload) != record.sha256
                or len(payload) != record.size_bytes
                or output_mode != record.mode
            ):
                raise AttemptTransactionError("supplied outputs differ from durable staged bytes")

    def _classify_outputs(
        self,
        snapshots: Sequence[_OutputSnapshot],
        staged: Sequence[PublishedOutput],
        *,
        removable_directories: frozenset[str],
    ) -> Literal["original", "partial_publication", "published"]:
        statuses: list[str] = []
        for index, snapshot in enumerate(snapshots):
            identities = self._output_identities(
                snapshot,
                staged[index] if staged else None,
                removable_directories=removable_directories,
            )
            if "original" in identities:
                statuses.append("original")
            elif "staged" in identities:
                statuses.append("published")
            else:
                raise AttemptTransactionTamperError(
                    f"formal output {snapshot.relative_path!r} is neither original nor staged"
                )
        if all(status == "original" for status in statuses):
            return "original"
        if all(status == "published" for status in statuses):
            return "published"
        return "partial_publication"

    def _restore_originals(
        self,
        loaded: _LoadedTransaction,
    ) -> None:
        publication = self._publication_directory
        if not publication.exists() and not publication.is_symlink():
            os.mkdir(publication, 0o700)
            _fsync_directory(self.transaction_directory)
        else:
            _require_directory(publication, label="publication recovery directory")
            for path in publication.iterdir():
                if path.is_symlink() or not path.is_file():
                    raise AttemptTransactionTamperError(
                        "publication recovery directory contains unsafe residue"
                    )
                path.unlink()
            _fsync_directory(publication)
        for index, snapshot in reversed(tuple(enumerate(loaded.snapshots))):
            staged = loaded.staged[index]
            destination = self._output_path(snapshot.relative_path)
            identity = self._require_output_identity(
                snapshot,
                staged,
                permitted=frozenset({"original", "staged"}),
            )
            if snapshot.content is None:
                if identity == "staged":
                    identity = self._require_output_identity(
                        snapshot,
                        staged,
                        permitted=frozenset({"original", "staged"}),
                    )
                if identity == "staged":
                    destination.unlink()
                    _fsync_directory(destination.parent)
                continue
            if identity == "original":
                continue
            temporary = publication / f"{index:03d}.restore"
            _write_fsynced_file(temporary, snapshot.content, mode=snapshot.mode or 0)
            identity = self._require_output_identity(
                snapshot,
                staged,
                permitted=frozenset({"original", "staged"}),
            )
            if identity == "original":
                temporary.unlink()
            else:
                os.replace(temporary, destination)
            _fsync_directory(publication)
            _fsync_directory(destination.parent)
            if (
                destination.read_bytes() != snapshot.content
                or stat.S_IMODE(destination.stat().st_mode) != snapshot.mode
            ):
                raise AttemptTransactionTamperError("restored output failed exact readback")
        if publication.exists():
            self._remove_tree(publication)

    def _remove_created_output_directories(self, loaded: _LoadedTransaction) -> None:
        for relative in sorted(
            loaded.manifest["created_output_directories"],
            key=lambda value: value.count("/"),
            reverse=True,
        ):
            directory = self.project_root / relative
            try:
                metadata = directory.lstat()
            except FileNotFoundError:
                continue
            if stat.S_ISLNK(metadata.st_mode) or not stat.S_ISDIR(metadata.st_mode):
                raise AttemptTransactionTamperError(
                    "created output path must remain a non-symlink directory"
                )
            if any(directory.iterdir()):
                continue
            directory.rmdir()
            _fsync_directory(directory.parent)


__all__ = [
    "ARTIFACT_VALIDATION_SCHEMA_VERSION",
    "ATTEMPT_TRANSACTION_SCHEMA_VERSION",
    "FORMAL_ARTIFACT_VALIDATOR_ID",
    "FORMAL_CONTROLLER_ORDER",
    "FORMAL_EPISODE_COUNT",
    "FORMAL_EXECUTION_EVIDENCE_BLOB_PATH",
    "FORMAL_ROWS_PER_CONTROLLER",
    "M8_EXECUTION_EVIDENCE_SCHEMA_VERSION",
    "AttemptIdentity",
    "AttemptInspection",
    "AttemptPhase",
    "AttemptRecovery",
    "AttemptTransactionError",
    "AttemptTransactionTamperError",
    "BlobRecord",
    "EpisodeJournalRecord",
    "IncompleteTestAttemptError",
    "M8AttemptTransaction",
    "PublishedOutput",
    "canonical_execution_evidence_bytes",
]
