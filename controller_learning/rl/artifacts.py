"""Crash-conscious local artifacts for M7 PPO training.

This module deliberately has no import-time dependency on PyTorch.  Training checkpoints are
local, trusted artifacts: callers may inject a Torch-compatible object in tests, while production
code imports :mod:`torch` only when a checkpoint is saved.

Checkpoint resume means continuing the policy and optimizer from a durable checkpoint while
creating a freshly reset official vector environment.  Environment state is never serialized, so
these primitives do not claim bit-exact continuation of an in-flight rollout.
"""

from __future__ import annotations

import hashlib
import importlib
import io
import json
import math
import os
import re
import stat
import tempfile
from collections.abc import Iterator, Mapping, Sequence
from contextlib import contextmanager
from dataclasses import dataclass
from fcntl import LOCK_EX, LOCK_SH, LOCK_UN, flock
from pathlib import Path, PurePosixPath
from types import MappingProxyType
from typing import Any, Final, Protocol

from controller_learning.rl.schema import (
    LOCAL_TRACK_FEATURE_SCHEMA_VERSION,
    PUBLIC_REWARD_SCHEMA_VERSION,
)

ARTIFACT_SCHEMA_VERSION: Final = 1
RUN_IDENTITY_SCHEMA_VERSION: Final = 1
TRAINING_CHECKPOINT_SCHEMA_VERSION: Final = 1
TRAINING_CONTINUATION_SCHEMA_VERSION: Final = 2
LATEST_CHECKPOINT_SCHEMA_VERSION: Final = 1
RESUME_SEMANTICS: Final = "optimizer_continuation_with_environment_reset"
M7_FEATURE_SCHEMA_VERSION: Final = LOCAL_TRACK_FEATURE_SCHEMA_VERSION
M7_REWARD_SCHEMA_VERSION: Final = PUBLIC_REWARD_SCHEMA_VERSION

_SHA256_PATTERN = re.compile(r"[0-9a-f]{64}")
_SOURCE_REVISION_PATTERN = re.compile(r"(?:[0-9a-f]{40}|[0-9a-f]{64})")
_RUN_ID_PATTERN = re.compile(r"[a-z0-9](?:[a-z0-9._-]{0,126}[a-z0-9])?")
_BENCHMARK_VERSION_PATTERN = re.compile(r"[0-9]+(?:\.[0-9]+)+")
_SAFE_PATH_PART_PATTERN = re.compile(r"[A-Za-z0-9](?:[A-Za-z0-9._-]{0,126}[A-Za-z0-9])?")
_CHECKPOINT_NAME_PATTERN = re.compile(r"update_(0*[1-9][0-9]*)\.pt")


class ArtifactError(RuntimeError):
    """Base class for durable-artifact failures."""


class ArtifactValidationError(ArtifactError, ValueError):
    """Raised before mutation when artifact input or metadata is invalid."""


class ArtifactWriteError(ArtifactError):
    """Raised when an artifact could not be committed and verified."""


class ArtifactPruneError(ArtifactError):
    """Raised after a checkpoint commit when retention cleanup is incomplete."""


class _TorchCompatible(Protocol):
    """The small lazy/injected Torch surface used by checkpoint persistence."""

    def save(self, obj: Any, file: Any) -> None: ...

    def load(
        self,
        file: Any,
        *,
        map_location: str,
        weights_only: bool,
    ) -> Any: ...


def _require_plain_integer(value: object, *, field: str, minimum: int = 0) -> int:
    if type(value) is not int or value < minimum:
        raise ArtifactValidationError(f"{field} must be an integer >= {minimum}")
    return value


def _require_uint32(value: object, *, field: str) -> int:
    result = _require_plain_integer(value, field=field)
    if result >= 2**32:
        raise ArtifactValidationError(f"{field} must fit in uint32")
    return result


def _require_string(value: object, *, field: str, pattern: re.Pattern[str]) -> str:
    if not isinstance(value, str) or pattern.fullmatch(value) is None:
        raise ArtifactValidationError(f"{field} has an invalid format")
    return value


def _require_sha256(value: object, *, field: str) -> str:
    return _require_string(value, field=field, pattern=_SHA256_PATTERN)


def _require_finite_nonnegative(value: object, *, field: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ArtifactValidationError(f"{field} must be a finite non-negative number")
    result = float(value)
    if not math.isfinite(result) or result < 0.0:
        raise ArtifactValidationError(f"{field} must be a finite non-negative number")
    return result


def _require_exact_keys(value: Mapping[str, Any], expected: set[str], *, field: str) -> None:
    if any(type(key) is not str for key in value):
        raise ArtifactValidationError(f"{field} must use string keys")
    actual = set(value)
    if actual != expected:
        missing = sorted(expected - actual)
        extra = sorted(actual - expected)
        raise ArtifactValidationError(f"{field} keys differ: missing={missing}, extra={extra}")


def _safe_relative_path(value: str | Path, *, field: str) -> PurePosixPath:
    raw = os.fspath(value)
    if not isinstance(raw, str) or not raw or "\\" in raw or len(raw) > 1024:
        raise ArtifactValidationError(f"{field} must be a safe relative POSIX path")
    path = PurePosixPath(raw)
    if path.is_absolute() or path == PurePosixPath(".") or path.as_posix() != raw:
        raise ArtifactValidationError(f"{field} must be a normalized relative POSIX path")
    if any(
        part in {"", ".", ".."} or _SAFE_PATH_PART_PATTERN.fullmatch(part) is None
        for part in path.parts
    ):
        raise ArtifactValidationError(f"{field} contains an unsafe path component")
    return path


def _absolute_path(path: Path) -> Path:
    return Path(os.path.abspath(os.fspath(path.expanduser())))


def _checked_existing_directory(path: Path) -> Path | None:
    """Resolve an existing directory tree without following symlinks or creating entries."""

    absolute = _absolute_path(path)
    current = Path(absolute.anchor)
    for part in absolute.parts[1:]:
        current /= part
        try:
            metadata = current.lstat()
        except FileNotFoundError:
            return None
        if stat.S_ISLNK(metadata.st_mode):
            raise ArtifactValidationError("artifact directory paths may not contain symlinks")
        if not stat.S_ISDIR(metadata.st_mode):
            raise ArtifactValidationError("artifact directory path must contain only directories")
    return current


def _ensure_directory_durable(path: Path) -> Path:
    """Create a no-symlink directory chain and fsync every newly linked directory entry."""

    absolute = _absolute_path(path)
    current = Path(absolute.anchor)
    for part in absolute.parts[1:]:
        candidate = current / part
        try:
            metadata = candidate.lstat()
        except FileNotFoundError:
            try:
                os.mkdir(candidate, 0o755)
            except FileExistsError:
                metadata = candidate.lstat()
            else:
                _fsync_directory(candidate)
                _fsync_directory(current)
                metadata = candidate.lstat()
        if stat.S_ISLNK(metadata.st_mode):
            raise ArtifactValidationError("artifact directory paths may not contain symlinks")
        if not stat.S_ISDIR(metadata.st_mode):
            raise ArtifactValidationError("artifact directory path must contain only directories")
        current = candidate
    return current


def _prepare_destination_for_write(
    root: Path,
    relative_path: str | Path,
) -> tuple[Path, str, Path]:
    relative = _safe_relative_path(relative_path, field="relative_path")
    root_directory = _ensure_directory_durable(root)
    parent = _ensure_directory_durable(root_directory.joinpath(*relative.parts[:-1]))
    destination = parent / relative.name
    if destination.is_symlink():
        raise ArtifactValidationError("artifact destination may not be a symbolic link")
    if destination.exists() and not destination.is_file():
        raise ArtifactValidationError("artifact destination must be a regular file")
    return destination, relative.as_posix(), root_directory


def _resolve_destination_for_read(
    root: Path,
    relative_path: str | Path,
) -> tuple[Path | None, str, Path | None]:
    """Resolve a safe artifact path without creating a directory or file."""

    relative = _safe_relative_path(relative_path, field="relative_path")
    root_directory = _checked_existing_directory(root)
    if root_directory is None:
        return None, relative.as_posix(), None
    parent = _checked_existing_directory(root_directory.joinpath(*relative.parts[:-1]))
    if parent is None:
        return None, relative.as_posix(), root_directory
    destination = parent / relative.name
    try:
        metadata = destination.lstat()
    except FileNotFoundError:
        return None, relative.as_posix(), root_directory
    if stat.S_ISLNK(metadata.st_mode):
        raise ArtifactValidationError("artifact destination may not be a symbolic link")
    if not stat.S_ISREG(metadata.st_mode):
        raise ArtifactValidationError("artifact destination must be a regular file")
    return destination, relative.as_posix(), root_directory


def _json_snapshot(value: Any, *, field: str = "$", active: set[int] | None = None) -> Any:
    """Copy one value into an unambiguous JSON-native tree."""

    if value is None or type(value) in {bool, int, str}:
        if isinstance(value, str):
            try:
                value.encode("utf-8")
            except UnicodeEncodeError as error:
                raise ArtifactValidationError(f"{field} is not valid UTF-8 text") from error
        return value
    if type(value) is float:
        if not math.isfinite(value):
            raise ArtifactValidationError(f"{field} must not contain NaN or infinity")
        return value

    active = set() if active is None else active
    identity = id(value)
    if identity in active:
        raise ArtifactValidationError(f"{field} contains a reference cycle")

    if isinstance(value, Mapping):
        active.add(identity)
        try:
            result: dict[str, Any] = {}
            for key, item in value.items():
                if type(key) is not str:
                    raise ArtifactValidationError(f"{field} must use string object keys")
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

    raise ArtifactValidationError(
        f"{field} contains unsupported JSON value type {type(value).__name__}"
    )


def canonical_json_bytes(value: Mapping[str, Any]) -> bytes:
    """Serialize a mapping to the project's deterministic, strict JSON representation."""

    if not isinstance(value, Mapping):
        raise ArtifactValidationError("the JSON document root must be an object")
    snapshot = _json_snapshot(value)
    try:
        text = json.dumps(
            snapshot,
            allow_nan=False,
            ensure_ascii=False,
            separators=(",", ":"),
            sort_keys=True,
        )
    except (TypeError, ValueError, UnicodeError) as error:
        raise ArtifactValidationError("payload is not strict JSON") from error
    return f"{text}\n".encode()


def sha256_bytes(value: bytes) -> str:
    """Return the lowercase SHA-256 digest of immutable bytes."""

    if not isinstance(value, bytes):
        raise ArtifactValidationError("sha256_bytes requires bytes")
    return hashlib.sha256(value).hexdigest()


def sha256_file(path: Path, *, chunk_size: int = 1024 * 1024) -> str:
    """Stream one regular file into SHA-256 without importing optional dependencies."""

    _require_plain_integer(chunk_size, field="chunk_size", minimum=1)
    if path.is_symlink() or not path.is_file():
        raise ArtifactValidationError("SHA-256 input must be a regular non-symlink file")
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for block in iter(lambda: file.read(chunk_size), b""):
            digest.update(block)
    return digest.hexdigest()


@dataclass(frozen=True, slots=True)
class ArtifactRecord:
    """Content identity of one verified file relative to an artifact root."""

    relative_path: str
    sha256: str
    size_bytes: int
    schema_version: int = ARTIFACT_SCHEMA_VERSION

    def __post_init__(self) -> None:
        relative = _safe_relative_path(self.relative_path, field="artifact.relative_path")
        digest = _require_sha256(self.sha256, field="artifact.sha256")
        size = _require_plain_integer(self.size_bytes, field="artifact.size_bytes")
        if self.schema_version != ARTIFACT_SCHEMA_VERSION:
            raise ArtifactValidationError(
                f"artifact.schema_version must be {ARTIFACT_SCHEMA_VERSION}"
            )
        object.__setattr__(self, "relative_path", relative.as_posix())
        object.__setattr__(self, "sha256", digest)
        object.__setattr__(self, "size_bytes", size)

    def to_dict(self) -> dict[str, Any]:
        """Return strict-JSON-compatible immutable record data."""

        return {
            "relative_path": self.relative_path,
            "schema_version": self.schema_version,
            "sha256": self.sha256,
            "size_bytes": self.size_bytes,
        }


def _readback_bytes(path: Path) -> bytes:
    """Indirection used by tests to inject post-replace readback faults."""

    return path.read_bytes()


def _fsync_directory(path: Path) -> None:
    flags = os.O_RDONLY | getattr(os, "O_DIRECTORY", 0)
    descriptor = os.open(path, flags)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _write_fsynced_temporary(parent: Path, name: str, payload: bytes, mode: int) -> Path:
    descriptor, temporary_name = tempfile.mkstemp(prefix=f".{name}.", suffix=".tmp", dir=parent)
    temporary = Path(temporary_name)
    try:
        os.fchmod(descriptor, mode)
        with os.fdopen(descriptor, "wb") as file:
            descriptor = -1
            file.write(payload)
            file.flush()
            os.fsync(file.fileno())
    except BaseException:
        if descriptor >= 0:
            os.close(descriptor)
        temporary.unlink(missing_ok=True)
        raise
    return temporary


def _restore_previous(
    destination: Path,
    previous: bytes | None,
    previous_mode: int,
) -> None:
    if previous is None:
        destination.unlink(missing_ok=True)
        _fsync_directory(destination.parent)
        return
    temporary = _write_fsynced_temporary(
        destination.parent,
        destination.name,
        previous,
        previous_mode,
    )
    try:
        os.replace(temporary, destination)
        _fsync_directory(destination.parent)
        if destination.read_bytes() != previous:
            raise OSError("restored artifact failed exact readback")
    finally:
        temporary.unlink(missing_ok=True)


def atomic_write_bytes(
    root: Path,
    relative_path: str | Path,
    payload: bytes,
    *,
    overwrite: bool = True,
    mode: int = 0o644,
) -> ArtifactRecord:
    """Atomically replace and exactly verify one same-directory local artifact.

    A caught post-replace failure attempts to restore the previous file (or remove a newly created
    file) before raising.  Power-loss guarantees still depend on the underlying filesystem.
    """

    if not isinstance(payload, bytes):
        raise ArtifactValidationError("artifact payload must be bytes")
    if type(overwrite) is not bool:
        raise ArtifactValidationError("overwrite must be a boolean")
    if type(mode) is not int or mode < 0 or mode > 0o777:
        raise ArtifactValidationError("mode must be a permission value from 0o000 to 0o777")
    destination, relative, _root_directory = _prepare_destination_for_write(
        root,
        relative_path,
    )
    if destination.exists() and not overwrite:
        raise ArtifactValidationError(f"refusing to overwrite immutable artifact {relative!r}")

    try:
        previous = destination.read_bytes() if destination.exists() else None
        previous_mode = stat.S_IMODE(destination.stat().st_mode) if destination.exists() else mode
        temporary = _write_fsynced_temporary(
            destination.parent,
            destination.name,
            payload,
            mode,
        )
    except OSError as error:
        raise ArtifactWriteError(f"failed to stage {relative!r}") from error
    replaced = False
    try:
        os.replace(temporary, destination)
        replaced = True
        _fsync_directory(destination.parent)
        readback = _readback_bytes(destination)
        if readback != payload:
            raise OSError("artifact failed exact post-replace readback")
    except BaseException as error:
        if replaced:
            try:
                _restore_previous(destination, previous, previous_mode)
            except BaseException as rollback_error:
                raise ArtifactWriteError(
                    f"failed to commit {relative!r}; rollback also failed: {rollback_error}"
                ) from error
        raise ArtifactWriteError(f"failed to commit and verify {relative!r}") from error
    finally:
        temporary.unlink(missing_ok=True)

    return ArtifactRecord(
        relative_path=relative,
        sha256=sha256_bytes(payload),
        size_bytes=len(payload),
    )


def atomic_write_json(
    root: Path,
    relative_path: str | Path,
    value: Mapping[str, Any],
    *,
    overwrite: bool = True,
) -> ArtifactRecord:
    """Atomically persist canonical strict JSON with exact readback verification."""

    return atomic_write_bytes(
        root,
        relative_path,
        canonical_json_bytes(value),
        overwrite=overwrite,
    )


def _reject_json_constant(value: str) -> None:
    raise ArtifactValidationError(f"strict JSON forbids {value}")


def _unique_json_object(items: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in items:
        if key in result:
            raise ArtifactValidationError(f"strict JSON contains duplicate key {key!r}")
        result[key] = value
    return result


def read_strict_json(
    root: Path,
    relative_path: str | Path,
    *,
    require_canonical: bool = True,
) -> dict[str, Any]:
    """Read a strict JSON object from a safe path and optionally require canonical bytes."""

    if type(require_canonical) is not bool:
        raise ArtifactValidationError("require_canonical must be a boolean")
    destination, _relative, _root_directory = _resolve_destination_for_read(
        root,
        relative_path,
    )
    if destination is None:
        raise ArtifactValidationError("JSON artifact does not exist as a regular file")
    payload = destination.read_bytes()
    try:
        decoded = json.loads(
            payload,
            parse_constant=_reject_json_constant,
            object_pairs_hook=_unique_json_object,
        )
    except ArtifactValidationError:
        raise
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ArtifactValidationError("artifact is not valid strict JSON") from error
    if not isinstance(decoded, dict):
        raise ArtifactValidationError("the JSON document root must be an object")
    snapshot = _json_snapshot(decoded)
    if require_canonical and canonical_json_bytes(snapshot) != payload:
        raise ArtifactValidationError("JSON artifact is not in canonical form")
    return snapshot


@dataclass(frozen=True, slots=True)
class TrainingRunIdentity:
    """Immutable inputs that identify one PPO optimization run."""

    run_id: str
    benchmark_version: str
    source_revision: str
    configuration_sha256: str
    lock_sha256: str
    train_manifest_sha256: str
    train_cache_sha256: str
    feature_schema_version: int
    reward_schema_version: str
    environment_seed: int
    policy_seed: int
    minibatch_seed: int
    schema_version: int = RUN_IDENTITY_SCHEMA_VERSION

    def __post_init__(self) -> None:
        if self.schema_version != RUN_IDENTITY_SCHEMA_VERSION:
            raise ArtifactValidationError(
                f"run_identity.schema_version must be {RUN_IDENTITY_SCHEMA_VERSION}"
            )
        object.__setattr__(
            self,
            "run_id",
            _require_string(self.run_id, field="run_identity.run_id", pattern=_RUN_ID_PATTERN),
        )
        object.__setattr__(
            self,
            "benchmark_version",
            _require_string(
                self.benchmark_version,
                field="run_identity.benchmark_version",
                pattern=_BENCHMARK_VERSION_PATTERN,
            ),
        )
        object.__setattr__(
            self,
            "source_revision",
            _require_string(
                self.source_revision,
                field="run_identity.source_revision",
                pattern=_SOURCE_REVISION_PATTERN,
            ),
        )
        object.__setattr__(
            self,
            "configuration_sha256",
            _require_sha256(
                self.configuration_sha256,
                field="run_identity.configuration_sha256",
            ),
        )
        object.__setattr__(
            self,
            "lock_sha256",
            _require_sha256(self.lock_sha256, field="run_identity.lock_sha256"),
        )
        object.__setattr__(
            self,
            "train_manifest_sha256",
            _require_sha256(
                self.train_manifest_sha256,
                field="run_identity.train_manifest_sha256",
            ),
        )
        object.__setattr__(
            self,
            "train_cache_sha256",
            _require_sha256(
                self.train_cache_sha256,
                field="run_identity.train_cache_sha256",
            ),
        )
        if type(self.feature_schema_version) is not int or (
            self.feature_schema_version != M7_FEATURE_SCHEMA_VERSION
        ):
            raise ArtifactValidationError(
                f"run_identity.feature_schema_version must be {M7_FEATURE_SCHEMA_VERSION}"
            )
        if self.reward_schema_version != M7_REWARD_SCHEMA_VERSION:
            raise ArtifactValidationError(
                f"run_identity.reward_schema_version must be {M7_REWARD_SCHEMA_VERSION!r}"
            )
        seeds = (
            _require_uint32(self.environment_seed, field="run_identity.environment_seed"),
            _require_uint32(self.policy_seed, field="run_identity.policy_seed"),
            _require_uint32(self.minibatch_seed, field="run_identity.minibatch_seed"),
        )
        if len(set(seeds)) != len(seeds):
            raise ArtifactValidationError("run identity seeds must be pairwise distinct")
        object.__setattr__(self, "environment_seed", seeds[0])
        object.__setattr__(self, "policy_seed", seeds[1])
        object.__setattr__(self, "minibatch_seed", seeds[2])

    def to_dict(self) -> dict[str, Any]:
        """Return the canonical JSON representation of this identity."""

        return {
            "benchmark_version": self.benchmark_version,
            "configuration_sha256": self.configuration_sha256,
            "environment_seed": self.environment_seed,
            "feature_schema_version": self.feature_schema_version,
            "lock_sha256": self.lock_sha256,
            "minibatch_seed": self.minibatch_seed,
            "policy_seed": self.policy_seed,
            "reward_schema_version": self.reward_schema_version,
            "run_id": self.run_id,
            "schema_version": self.schema_version,
            "source_revision": self.source_revision,
            "train_cache_sha256": self.train_cache_sha256,
            "train_manifest_sha256": self.train_manifest_sha256,
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> TrainingRunIdentity:
        """Validate and reconstruct a run identity from strict JSON data."""

        if not isinstance(value, Mapping):
            raise ArtifactValidationError("run_identity must be an object")
        expected = {
            "benchmark_version",
            "configuration_sha256",
            "environment_seed",
            "feature_schema_version",
            "lock_sha256",
            "minibatch_seed",
            "policy_seed",
            "reward_schema_version",
            "run_id",
            "schema_version",
            "source_revision",
            "train_cache_sha256",
            "train_manifest_sha256",
        }
        _require_exact_keys(value, expected, field="run_identity")
        return cls(**dict(value))


@dataclass(frozen=True, slots=True)
class TrainingCheckpointMetadata:
    """Strict metadata stored in every resumable local training checkpoint."""

    run_identity: TrainingRunIdentity
    update_index: int
    vector_steps: int
    valid_transitions: int
    elapsed_seconds: float
    resume_semantics: str = RESUME_SEMANTICS
    schema_version: int = TRAINING_CHECKPOINT_SCHEMA_VERSION

    def __post_init__(self) -> None:
        if not isinstance(self.run_identity, TrainingRunIdentity):
            raise ArtifactValidationError(
                "checkpoint_metadata.run_identity must be a TrainingRunIdentity"
            )
        if self.schema_version != TRAINING_CHECKPOINT_SCHEMA_VERSION:
            raise ArtifactValidationError(
                f"checkpoint_metadata.schema_version must be {TRAINING_CHECKPOINT_SCHEMA_VERSION}"
            )
        if self.resume_semantics != RESUME_SEMANTICS:
            raise ArtifactValidationError(
                f"checkpoint_metadata.resume_semantics must be {RESUME_SEMANTICS!r}"
            )
        object.__setattr__(
            self,
            "update_index",
            _require_plain_integer(
                self.update_index,
                field="checkpoint_metadata.update_index",
                minimum=1,
            ),
        )
        object.__setattr__(
            self,
            "vector_steps",
            _require_plain_integer(
                self.vector_steps,
                field="checkpoint_metadata.vector_steps",
                minimum=1,
            ),
        )
        object.__setattr__(
            self,
            "valid_transitions",
            _require_plain_integer(
                self.valid_transitions,
                field="checkpoint_metadata.valid_transitions",
            ),
        )
        object.__setattr__(
            self,
            "elapsed_seconds",
            _require_finite_nonnegative(
                self.elapsed_seconds,
                field="checkpoint_metadata.elapsed_seconds",
            ),
        )

    def to_dict(self) -> dict[str, Any]:
        """Return strict JSON metadata without any environment-state claim."""

        return {
            "elapsed_seconds": self.elapsed_seconds,
            "resume_semantics": self.resume_semantics,
            "run_identity": self.run_identity.to_dict(),
            "schema_version": self.schema_version,
            "update_index": self.update_index,
            "valid_transitions": self.valid_transitions,
            "vector_steps": self.vector_steps,
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> TrainingCheckpointMetadata:
        """Validate and reconstruct checkpoint metadata from a Torch payload."""

        if not isinstance(value, Mapping):
            raise ArtifactValidationError("checkpoint_metadata must be an object")
        expected = {
            "elapsed_seconds",
            "resume_semantics",
            "run_identity",
            "schema_version",
            "update_index",
            "valid_transitions",
            "vector_steps",
        }
        _require_exact_keys(value, expected, field="checkpoint_metadata")
        fields = dict(value)
        fields["run_identity"] = TrainingRunIdentity.from_dict(fields["run_identity"])
        return cls(**fields)


@dataclass(frozen=True, slots=True)
class TrainingContinuationState:
    """Torch-free cumulative counters needed to construct trainer ``TrainingResumeState``.

    RNG tensors remain separate top-level checkpoint fields.  Simulator, collector, and partial
    active-episode state are intentionally absent because resume starts from a fresh environment
    reset under :data:`RESUME_SEMANTICS`.
    """

    starting_update: int
    num_envs: int
    environment_step_calls: int
    raw_transitions: int
    valid_transitions: int
    dummy_reset_transitions: int
    autoreset_slots: int
    discarded_pending_reset_slots: int
    terminal_events: int
    terminated_events: int
    truncated_events: int
    episodes: int
    successful_episodes: int
    offtrack_episodes: int
    invalid_action_episodes: int
    timeout_episodes: int
    successful_lap_time_sum_s: float
    episode_length_sum_steps: int
    cumulative_reward_sum: float
    cumulative_compute_update_seconds: float
    wall_elapsed_before_persistence_seconds: float
    schema_version: int = TRAINING_CONTINUATION_SCHEMA_VERSION

    def __post_init__(self) -> None:
        if self.schema_version != TRAINING_CONTINUATION_SCHEMA_VERSION:
            raise ArtifactValidationError(
                f"continuation.schema_version must be {TRAINING_CONTINUATION_SCHEMA_VERSION}"
            )
        object.__setattr__(
            self,
            "starting_update",
            _require_plain_integer(
                self.starting_update,
                field="continuation.starting_update",
                minimum=1,
            ),
        )
        object.__setattr__(
            self,
            "num_envs",
            _require_plain_integer(self.num_envs, field="continuation.num_envs", minimum=1),
        )
        count_fields = (
            "environment_step_calls",
            "raw_transitions",
            "valid_transitions",
            "dummy_reset_transitions",
            "autoreset_slots",
            "discarded_pending_reset_slots",
            "terminal_events",
            "terminated_events",
            "truncated_events",
            "episodes",
            "successful_episodes",
            "offtrack_episodes",
            "invalid_action_episodes",
            "timeout_episodes",
            "episode_length_sum_steps",
        )
        for field in count_fields:
            object.__setattr__(
                self,
                field,
                _require_plain_integer(
                    getattr(self, field),
                    field=f"continuation.{field}",
                ),
            )
        if self.environment_step_calls < 1:
            raise ArtifactValidationError(
                "continuation.environment_step_calls must be positive after an update"
            )
        if self.raw_transitions != self.num_envs * self.environment_step_calls:
            raise ArtifactValidationError(
                "continuation.raw_transitions must equal num_envs * environment_step_calls"
            )
        if self.raw_transitions != self.valid_transitions + self.dummy_reset_transitions:
            raise ArtifactValidationError(
                "continuation.raw_transitions must equal valid + dummy-reset transitions"
            )
        if self.autoreset_slots != self.dummy_reset_transitions:
            raise ArtifactValidationError(
                "continuation.autoreset_slots must equal dummy_reset_transitions"
            )
        if self.terminal_events != self.terminated_events + self.truncated_events:
            raise ArtifactValidationError(
                "continuation.terminal_events must equal terminated + truncated events"
            )
        if self.terminal_events > self.valid_transitions:
            raise ArtifactValidationError(
                "continuation.terminal_events cannot exceed valid_transitions"
            )
        pending_after_compensation = (
            self.terminal_events - self.autoreset_slots - self.discarded_pending_reset_slots
        )
        if self.discarded_pending_reset_slots > (self.terminal_events - self.autoreset_slots):
            raise ArtifactValidationError(
                "continuation.discarded_pending_reset_slots cannot exceed terminal_events - "
                "autoreset_slots"
            )
        if not 0 <= pending_after_compensation <= self.num_envs:
            raise ArtifactValidationError(
                "continuation uncompensated pending-reset slots must be in [0, num_envs]"
            )
        categorized = (
            self.successful_episodes
            + self.offtrack_episodes
            + self.invalid_action_episodes
            + self.timeout_episodes
        )
        if self.episodes != categorized or self.episodes != self.terminal_events:
            raise ArtifactValidationError(
                "continuation episodes must equal terminal events and the four reason counts"
            )
        terminated_reasons = (
            self.successful_episodes + self.offtrack_episodes + self.invalid_action_episodes
        )
        if self.terminated_events != terminated_reasons:
            raise ArtifactValidationError(
                "continuation terminated events must equal success, off-track, and invalid-action"
            )
        if self.truncated_events != self.timeout_episodes:
            raise ArtifactValidationError(
                "continuation truncated events must equal timeout episodes"
            )
        if (self.episodes == 0) != (self.episode_length_sum_steps == 0):
            raise ArtifactValidationError(
                "continuation episode lengths must be positive exactly when episodes exist"
            )
        if self.episode_length_sum_steps > self.valid_transitions:
            raise ArtifactValidationError(
                "continuation completed-episode lengths cannot exceed valid transitions"
            )
        lap_sum = _require_finite_nonnegative(
            self.successful_lap_time_sum_s,
            field="continuation.successful_lap_time_sum_s",
        )
        if (self.successful_episodes == 0) != (lap_sum == 0.0):
            raise ArtifactValidationError(
                "continuation successful lap time must be positive exactly when successes exist"
            )
        object.__setattr__(self, "successful_lap_time_sum_s", lap_sum)
        reward = self.cumulative_reward_sum
        if isinstance(reward, bool) or not isinstance(reward, (int, float)):
            raise ArtifactValidationError(
                "continuation.cumulative_reward_sum must be a finite number"
            )
        reward = float(reward)
        if not math.isfinite(reward):
            raise ArtifactValidationError(
                "continuation.cumulative_reward_sum must be a finite number"
            )
        object.__setattr__(self, "cumulative_reward_sum", reward)
        compute = _require_finite_nonnegative(
            self.cumulative_compute_update_seconds,
            field="continuation.cumulative_compute_update_seconds",
        )
        elapsed = _require_finite_nonnegative(
            self.wall_elapsed_before_persistence_seconds,
            field="continuation.wall_elapsed_before_persistence_seconds",
        )
        if compute <= 0.0 or elapsed <= 0.0:
            raise ArtifactValidationError(
                "continuation compute and pre-persistence wall seconds must be positive"
            )
        if elapsed < compute:
            raise ArtifactValidationError(
                "continuation pre-persistence wall seconds cannot be below cumulative compute "
                "seconds"
            )
        object.__setattr__(self, "cumulative_compute_update_seconds", compute)
        object.__setattr__(self, "wall_elapsed_before_persistence_seconds", elapsed)

    def validate_checkpoint_metadata(self, metadata: TrainingCheckpointMetadata) -> None:
        """Require the duplicated durability-boundary counters to match exactly."""

        if not isinstance(metadata, TrainingCheckpointMetadata):
            raise ArtifactValidationError("metadata must be TrainingCheckpointMetadata")
        comparisons = (
            ("update_index", self.starting_update, metadata.update_index),
            ("vector_steps", self.environment_step_calls, metadata.vector_steps),
            ("valid_transitions", self.valid_transitions, metadata.valid_transitions),
            (
                "elapsed_seconds",
                self.wall_elapsed_before_persistence_seconds,
                metadata.elapsed_seconds,
            ),
        )
        for field, continuation_value, metadata_value in comparisons:
            if continuation_value != metadata_value:
                raise ArtifactValidationError(
                    f"continuation {field} differs from checkpoint metadata"
                )

    def to_dict(self) -> dict[str, Any]:
        """Return the exact strict-JSON-compatible continuation schema."""

        return {
            "autoreset_slots": self.autoreset_slots,
            "cumulative_compute_update_seconds": self.cumulative_compute_update_seconds,
            "cumulative_reward_sum": self.cumulative_reward_sum,
            "discarded_pending_reset_slots": self.discarded_pending_reset_slots,
            "dummy_reset_transitions": self.dummy_reset_transitions,
            "environment_step_calls": self.environment_step_calls,
            "episode_length_sum_steps": self.episode_length_sum_steps,
            "episodes": self.episodes,
            "invalid_action_episodes": self.invalid_action_episodes,
            "num_envs": self.num_envs,
            "offtrack_episodes": self.offtrack_episodes,
            "raw_transitions": self.raw_transitions,
            "schema_version": self.schema_version,
            "starting_update": self.starting_update,
            "successful_episodes": self.successful_episodes,
            "successful_lap_time_sum_s": self.successful_lap_time_sum_s,
            "terminal_events": self.terminal_events,
            "terminated_events": self.terminated_events,
            "timeout_episodes": self.timeout_episodes,
            "truncated_events": self.truncated_events,
            "valid_transitions": self.valid_transitions,
            "wall_elapsed_before_persistence_seconds": (
                self.wall_elapsed_before_persistence_seconds
            ),
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> TrainingContinuationState:
        """Validate and reconstruct continuation counters from a Torch payload."""

        if not isinstance(value, Mapping):
            raise ArtifactValidationError("continuation_state must be an object")
        expected = {
            "autoreset_slots",
            "cumulative_compute_update_seconds",
            "cumulative_reward_sum",
            "discarded_pending_reset_slots",
            "dummy_reset_transitions",
            "environment_step_calls",
            "episode_length_sum_steps",
            "episodes",
            "invalid_action_episodes",
            "num_envs",
            "offtrack_episodes",
            "raw_transitions",
            "schema_version",
            "starting_update",
            "successful_episodes",
            "successful_lap_time_sum_s",
            "terminal_events",
            "terminated_events",
            "timeout_episodes",
            "truncated_events",
            "valid_transitions",
            "wall_elapsed_before_persistence_seconds",
        }
        _require_exact_keys(value, expected, field="continuation_state")
        return cls(**dict(value))


def run_identity_sha256(identity: TrainingRunIdentity) -> str:
    """Return the canonical digest that binds checkpoint continuity to every run input."""

    if not isinstance(identity, TrainingRunIdentity):
        raise ArtifactValidationError("identity must be TrainingRunIdentity")
    return sha256_bytes(canonical_json_bytes(identity.to_dict()))


@dataclass(frozen=True, slots=True)
class LatestCheckpointPointer:
    """Verified small pointer updated only after a checkpoint is durable and readable."""

    run_id: str
    run_identity_sha256: str
    update_index: int
    published_updates: tuple[int, ...]
    checkpoint: ArtifactRecord
    resume_semantics: str = RESUME_SEMANTICS
    schema_version: int = LATEST_CHECKPOINT_SCHEMA_VERSION

    def __post_init__(self) -> None:
        if self.schema_version != LATEST_CHECKPOINT_SCHEMA_VERSION:
            raise ArtifactValidationError(
                f"latest.schema_version must be {LATEST_CHECKPOINT_SCHEMA_VERSION}"
            )
        object.__setattr__(
            self,
            "run_id",
            _require_string(self.run_id, field="latest.run_id", pattern=_RUN_ID_PATTERN),
        )
        object.__setattr__(
            self,
            "run_identity_sha256",
            _require_sha256(
                self.run_identity_sha256,
                field="latest.run_identity_sha256",
            ),
        )
        object.__setattr__(
            self,
            "update_index",
            _require_plain_integer(self.update_index, field="latest.update_index", minimum=1),
        )
        if not isinstance(self.published_updates, (tuple, list)):
            raise ArtifactValidationError("latest.published_updates must be a sequence")
        published_updates = tuple(
            _require_plain_integer(update, field="latest.published_updates", minimum=1)
            for update in self.published_updates
        )
        if (
            not published_updates
            or tuple(sorted(set(published_updates))) != published_updates
            or published_updates[-1] != self.update_index
        ):
            raise ArtifactValidationError(
                "latest.published_updates must be strictly increasing and end at update_index"
            )
        object.__setattr__(self, "published_updates", published_updates)
        if not isinstance(self.checkpoint, ArtifactRecord):
            raise ArtifactValidationError("latest.checkpoint must be an ArtifactRecord")
        checkpoint_name = PurePosixPath(self.checkpoint.relative_path).name
        if checkpoint_name != f"update_{self.update_index:08d}.pt":
            raise ArtifactValidationError(
                "latest checkpoint filename must match latest.update_index"
            )
        if self.resume_semantics != RESUME_SEMANTICS:
            raise ArtifactValidationError(f"latest.resume_semantics must be {RESUME_SEMANTICS!r}")

    def to_dict(self) -> dict[str, Any]:
        """Return strict JSON pointer data."""

        return {
            "checkpoint": self.checkpoint.to_dict(),
            "published_updates": list(self.published_updates),
            "resume_semantics": self.resume_semantics,
            "run_id": self.run_id,
            "run_identity_sha256": self.run_identity_sha256,
            "schema_version": self.schema_version,
            "update_index": self.update_index,
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> LatestCheckpointPointer:
        """Validate and reconstruct a latest pointer."""

        if not isinstance(value, Mapping):
            raise ArtifactValidationError("latest pointer must be an object")
        expected = {
            "checkpoint",
            "published_updates",
            "resume_semantics",
            "run_id",
            "run_identity_sha256",
            "schema_version",
            "update_index",
        }
        _require_exact_keys(value, expected, field="latest")
        fields = dict(value)
        checkpoint = fields["checkpoint"]
        if not isinstance(checkpoint, Mapping):
            raise ArtifactValidationError("latest.checkpoint must be an object")
        _require_exact_keys(
            checkpoint,
            {"relative_path", "schema_version", "sha256", "size_bytes"},
            field="latest.checkpoint",
        )
        fields["checkpoint"] = ArtifactRecord(**dict(checkpoint))
        return cls(**fields)


@dataclass(frozen=True, slots=True)
class TrainingCheckpointArtifact:
    """Result of one verified checkpoint, pointer update, and retention pass."""

    checkpoint: ArtifactRecord
    latest_pointer: ArtifactRecord
    metadata: TrainingCheckpointMetadata
    pruned_relative_paths: tuple[str, ...]

    def __post_init__(self) -> None:
        if not isinstance(self.checkpoint, ArtifactRecord):
            raise ArtifactValidationError("checkpoint result requires an ArtifactRecord")
        if not isinstance(self.latest_pointer, ArtifactRecord):
            raise ArtifactValidationError("checkpoint result requires a latest-pointer record")
        if not isinstance(self.metadata, TrainingCheckpointMetadata):
            raise ArtifactValidationError("checkpoint result requires validated metadata")
        normalized = tuple(
            _safe_relative_path(path, field="pruned_relative_paths").as_posix()
            for path in self.pruned_relative_paths
        )
        object.__setattr__(self, "pruned_relative_paths", normalized)


@dataclass(frozen=True, slots=True)
class LoadedTrainingCheckpoint:
    """A fully verified latest checkpoint payload ready for trusted local restoration."""

    pointer: LatestCheckpointPointer
    metadata: TrainingCheckpointMetadata
    continuation_state: TrainingContinuationState
    payload: Mapping[str, Any]

    def __post_init__(self) -> None:
        if not isinstance(self.pointer, LatestCheckpointPointer):
            raise ArtifactValidationError("loaded checkpoint requires a validated pointer")
        if not isinstance(self.metadata, TrainingCheckpointMetadata):
            raise ArtifactValidationError("loaded checkpoint requires validated metadata")
        if not isinstance(self.continuation_state, TrainingContinuationState):
            raise ArtifactValidationError("loaded checkpoint requires validated continuation state")
        self.continuation_state.validate_checkpoint_metadata(self.metadata)
        if not isinstance(self.payload, Mapping):
            raise ArtifactValidationError("loaded checkpoint payload must be a mapping")
        object.__setattr__(self, "payload", MappingProxyType(dict(self.payload)))


def _checkpoint_relative_paths(
    checkpoint_directory: str | Path,
    update_index: int,
) -> tuple[PurePosixPath, str, str]:
    directory = _safe_relative_path(checkpoint_directory, field="checkpoint_directory")
    checkpoint_relative = (directory / f"update_{update_index:08d}.pt").as_posix()
    latest_relative = (directory / "latest.json").as_posix()
    return directory, checkpoint_relative, latest_relative


def _checkpoint_layout_for_write(
    root: Path,
    checkpoint_directory: str | Path,
    update_index: int,
) -> tuple[Path, Path, Path, str, str]:
    directory, checkpoint_relative, latest_relative = _checkpoint_relative_paths(
        checkpoint_directory,
        update_index,
    )
    root_directory = _ensure_directory_durable(root)
    absolute_directory = _ensure_directory_durable(root_directory.joinpath(*directory.parts))
    destination = absolute_directory / PurePosixPath(checkpoint_relative).name
    return (
        root_directory,
        absolute_directory,
        destination,
        checkpoint_relative,
        latest_relative,
    )


@contextmanager
def _checkpoint_lock(
    directory: Path,
    *,
    exclusive: bool,
    create: bool,
) -> Iterator[None]:
    """Hold one Linux advisory lock for a complete checkpoint transaction."""

    lock_path = directory / "transaction.lock"
    access = os.O_RDWR if create or exclusive else os.O_RDONLY
    flags = os.O_CLOEXEC | os.O_NOFOLLOW | access
    created = False
    if create:
        try:
            descriptor = os.open(lock_path, flags | os.O_CREAT | os.O_EXCL, 0o600)
            created = True
        except FileExistsError:
            descriptor = os.open(lock_path, flags)
    else:
        try:
            descriptor = os.open(lock_path, flags)
        except FileNotFoundError as error:
            raise ArtifactValidationError("checkpoint transaction lock does not exist") from error
    try:
        metadata = os.fstat(descriptor)
        if not stat.S_ISREG(metadata.st_mode):
            raise ArtifactValidationError("checkpoint transaction lock must be a regular file")
        if created:
            os.fsync(descriptor)
            _fsync_directory(directory)
        flock(descriptor, LOCK_EX if exclusive else LOCK_SH)
        try:
            yield
        finally:
            flock(descriptor, LOCK_UN)
    finally:
        os.close(descriptor)


def read_latest_checkpoint_pointer(
    root: Path,
    *,
    checkpoint_directory: str | Path = "checkpoints",
) -> LatestCheckpointPointer | None:
    """Read and validate the canonical latest pointer, returning ``None`` when absent."""

    directory, _checkpoint_relative, latest_relative = _checkpoint_relative_paths(
        checkpoint_directory,
        1,
    )
    destination, _normalized, _root_directory = _resolve_destination_for_read(
        root,
        latest_relative,
    )
    if destination is None:
        return None
    pointer = LatestCheckpointPointer.from_dict(
        read_strict_json(root, latest_relative, require_canonical=True)
    )
    if PurePosixPath(pointer.checkpoint.relative_path).parent != directory:
        raise ArtifactValidationError("latest checkpoint must remain inside checkpoint_directory")
    return pointer


def _torch_backend(torch_module: _TorchCompatible | None) -> _TorchCompatible:
    backend = importlib.import_module("torch") if torch_module is None else torch_module
    if not callable(getattr(backend, "save", None)) or not callable(getattr(backend, "load", None)):
        raise ArtifactValidationError("torch_module must provide callable save and load methods")
    return backend


def _validate_state_mapping(value: object, *, field: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping) or not value:
        raise ArtifactValidationError(f"{field} must be a non-empty mapping")
    if any(type(key) is not str or not key for key in value):
        raise ArtifactValidationError(f"{field} must use non-empty string keys")
    return value


def _validated_loaded_checkpoint(
    value: object,
) -> tuple[TrainingCheckpointMetadata, TrainingContinuationState, Mapping[str, Any]]:
    if not isinstance(value, Mapping):
        raise ArtifactValidationError("Torch checkpoint readback is not a mapping")
    expected_keys = {
        "continuation_state",
        "metadata",
        "minibatch_rng_state",
        "model_state_dict",
        "optimizer_state_dict",
        "policy_rng_state",
        "schema_version",
    }
    _require_exact_keys(value, expected_keys, field="Torch checkpoint")
    if value["schema_version"] != TRAINING_CHECKPOINT_SCHEMA_VERSION:
        raise ArtifactValidationError("Torch checkpoint schema version differs")
    loaded_metadata = TrainingCheckpointMetadata.from_dict(value["metadata"])
    continuation_state = TrainingContinuationState.from_dict(value["continuation_state"])
    continuation_state.validate_checkpoint_metadata(loaded_metadata)
    _validate_state_mapping(value["model_state_dict"], field="model_state_dict")
    _validate_state_mapping(value["optimizer_state_dict"], field="optimizer_state_dict")
    if value["policy_rng_state"] is None or value["minibatch_rng_state"] is None:
        raise ArtifactValidationError("Torch checkpoint RNG states must not be null")
    return loaded_metadata, continuation_state, MappingProxyType(dict(value))


def _torch_load_bytes(backend: _TorchCompatible, payload: bytes) -> Any:
    return backend.load(io.BytesIO(payload), map_location="cpu", weights_only=False)


def _verify_checkpoint_bytes(
    backend: _TorchCompatible,
    payload: bytes,
    *,
    expected_metadata: TrainingCheckpointMetadata | None = None,
    expected_continuation: TrainingContinuationState | None = None,
) -> tuple[TrainingCheckpointMetadata, TrainingContinuationState, Mapping[str, Any]]:
    if not payload:
        raise ArtifactValidationError("Torch checkpoint must not be empty")
    try:
        loaded = _torch_load_bytes(backend, payload)
        metadata, continuation, validated_payload = _validated_loaded_checkpoint(loaded)
    except ArtifactValidationError:
        raise
    except Exception as error:
        raise ArtifactValidationError("Torch checkpoint is not readable") from error
    if expected_metadata is not None and metadata != expected_metadata:
        raise ArtifactValidationError("Torch checkpoint metadata differs after readback")
    if expected_continuation is not None and continuation != expected_continuation:
        raise ArtifactValidationError("Torch checkpoint continuation differs after readback")
    return metadata, continuation, validated_payload


def _remove_new_checkpoint(path: Path) -> None:
    path.unlink(missing_ok=True)
    _fsync_directory(path.parent)


def _read_regular_checkpoint(path: Path) -> bytes:
    try:
        metadata = path.lstat()
    except FileNotFoundError as error:
        raise ArtifactValidationError("checkpoint target does not exist") from error
    if stat.S_ISLNK(metadata.st_mode) or not stat.S_ISREG(metadata.st_mode):
        raise ArtifactValidationError("checkpoint target must be a regular non-symlink file")
    return path.read_bytes()


def _record_for_checkpoint(relative_path: str, payload: bytes) -> ArtifactRecord:
    return ArtifactRecord(
        relative_path=relative_path,
        sha256=sha256_bytes(payload),
        size_bytes=len(payload),
    )


def _load_published_checkpoint_locked(
    root: Path,
    pointer: LatestCheckpointPointer,
    backend: _TorchCompatible,
    *,
    expected_identity: TrainingRunIdentity | None,
) -> LoadedTrainingCheckpoint:
    destination, relative, _root_directory = _resolve_destination_for_read(
        root,
        pointer.checkpoint.relative_path,
    )
    if destination is None:
        raise ArtifactValidationError("latest checkpoint target does not exist")
    checkpoint_bytes = _read_regular_checkpoint(destination)
    if len(checkpoint_bytes) != pointer.checkpoint.size_bytes:
        raise ArtifactValidationError("latest checkpoint size differs from its pointer")
    if sha256_bytes(checkpoint_bytes) != pointer.checkpoint.sha256:
        raise ArtifactValidationError("latest checkpoint SHA-256 differs from its pointer")
    if relative != pointer.checkpoint.relative_path:
        raise ArtifactValidationError("latest checkpoint path is not canonical")

    metadata, continuation, payload = _verify_checkpoint_bytes(backend, checkpoint_bytes)
    identity = metadata.run_identity
    if metadata.update_index != pointer.update_index:
        raise ArtifactValidationError("checkpoint metadata update differs from latest pointer")
    if identity.run_id != pointer.run_id:
        raise ArtifactValidationError("checkpoint run_id differs from latest pointer")
    if run_identity_sha256(identity) != pointer.run_identity_sha256:
        raise ArtifactValidationError("checkpoint run identity differs from latest pointer")
    if expected_identity is not None and identity != expected_identity:
        raise ArtifactValidationError("checkpoint run identity differs from expected_identity")
    return LoadedTrainingCheckpoint(
        pointer=pointer,
        metadata=metadata,
        continuation_state=continuation,
        payload=payload,
    )


def _reuse_or_clean_same_update_orphan(
    destination: Path,
    checkpoint_relative: str,
    backend: _TorchCompatible,
    expected_metadata: TrainingCheckpointMetadata,
    expected_continuation: TrainingContinuationState,
) -> ArtifactRecord | None:
    try:
        destination.lstat()
    except FileNotFoundError:
        return None
    try:
        checkpoint_bytes = _read_regular_checkpoint(destination)
        _verify_checkpoint_bytes(
            backend,
            checkpoint_bytes,
            expected_metadata=expected_metadata,
            expected_continuation=expected_continuation,
        )
    except ArtifactValidationError:
        try:
            metadata = destination.lstat()
            if stat.S_ISDIR(metadata.st_mode):
                raise ArtifactValidationError(
                    "invalid checkpoint orphan is a directory and cannot be cleaned safely"
                )
            destination.unlink()
            _fsync_directory(destination.parent)
        except FileNotFoundError:
            return None
        return None
    return _record_for_checkpoint(checkpoint_relative, checkpoint_bytes)


def _checkpoint_candidates(directory: Path) -> list[tuple[int, Path]]:
    candidates: list[tuple[int, Path]] = []
    for path in directory.iterdir():
        if path.is_symlink() or not path.is_file():
            continue
        matched = _CHECKPOINT_NAME_PATTERN.fullmatch(path.name)
        if matched is not None:
            candidates.append((int(matched.group(1)), path))
    return sorted(candidates, key=lambda item: (item[0], item[1].name))


def _unlink_checkpoint(path: Path) -> None:
    """Indirection used to test retention ordering."""

    path.unlink()


def _prune_checkpoints(
    directory: Path,
    *,
    keep_last: int,
    root: Path,
    published_updates: tuple[int, ...],
    protected_checkpoint: Path,
) -> tuple[str, ...]:
    published = set(published_updates)
    candidates = [
        candidate for candidate in _checkpoint_candidates(directory) if candidate[0] in published
    ]
    retained = candidates[-keep_last:]
    retained_paths = {path for _update, path in retained}
    retained_paths.add(protected_checkpoint)
    victims = [candidate for candidate in candidates if candidate[1] not in retained_paths]
    pruned: list[str] = []
    try:
        for _update_index, path in victims:
            relative = path.relative_to(root).as_posix()
            _unlink_checkpoint(path)
            pruned.append(relative)
        if victims:
            _fsync_directory(directory)
    except OSError as error:
        raise ArtifactPruneError(
            "checkpoint and latest pointer are committed, but retention pruning failed"
        ) from error
    return tuple(pruned)


def load_training_checkpoint(
    root: Path,
    *,
    expected_identity: TrainingRunIdentity,
    checkpoint_directory: str | Path = "checkpoints",
    torch_module: _TorchCompatible | None = None,
) -> LoadedTrainingCheckpoint:
    """Load the latest fully verified checkpoint without creating filesystem entries.

    The advisory shared lock spans pointer resolution, content size/hash verification, Torch
    deserialization, schema validation, and the full expected run-identity comparison.
    """

    if not isinstance(expected_identity, TrainingRunIdentity):
        raise ArtifactValidationError("expected_identity must be TrainingRunIdentity")
    directory = _safe_relative_path(checkpoint_directory, field="checkpoint_directory")
    root_directory = _checked_existing_directory(root)
    if root_directory is None:
        raise ArtifactValidationError("artifact root does not exist")
    absolute_directory = _checked_existing_directory(root_directory.joinpath(*directory.parts))
    if absolute_directory is None:
        raise ArtifactValidationError("checkpoint directory does not exist")
    backend = _torch_backend(torch_module)
    with _checkpoint_lock(absolute_directory, exclusive=False, create=False):
        pointer = read_latest_checkpoint_pointer(
            root_directory,
            checkpoint_directory=checkpoint_directory,
        )
        if pointer is None:
            raise ArtifactValidationError("latest checkpoint pointer does not exist")
        if pointer.run_identity_sha256 != run_identity_sha256(expected_identity):
            raise ArtifactValidationError(
                "latest checkpoint run identity differs from expected_identity"
            )
        return _load_published_checkpoint_locked(
            root_directory,
            pointer,
            backend,
            expected_identity=expected_identity,
        )


def save_training_checkpoint(
    root: Path,
    *,
    metadata: TrainingCheckpointMetadata,
    continuation_state: TrainingContinuationState,
    model_state_dict: Mapping[str, Any],
    optimizer_state_dict: Mapping[str, Any],
    policy_rng_state: Any,
    minibatch_rng_state: Any,
    keep_last: int,
    checkpoint_directory: str | Path = "checkpoints",
    torch_module: _TorchCompatible | None = None,
) -> TrainingCheckpointArtifact:
    """Commit, verify, publish, then prune one local optimizer-continuation checkpoint.

    The serialized state intentionally has no environment-state field.  A resumed process restores
    the model, optimizer, and dedicated policy/minibatch RNG states, then resets a new official
    environment according to :data:`RESUME_SEMANTICS`.
    """

    if not isinstance(metadata, TrainingCheckpointMetadata):
        raise ArtifactValidationError("metadata must be TrainingCheckpointMetadata")
    if not isinstance(continuation_state, TrainingContinuationState):
        raise ArtifactValidationError("continuation_state must be TrainingContinuationState")
    continuation_state.validate_checkpoint_metadata(metadata)
    keep_last = _require_plain_integer(keep_last, field="keep_last", minimum=1)
    model_state_dict = _validate_state_mapping(model_state_dict, field="model_state_dict")
    optimizer_state_dict = _validate_state_mapping(
        optimizer_state_dict,
        field="optimizer_state_dict",
    )
    if policy_rng_state is None or minibatch_rng_state is None:
        raise ArtifactValidationError("policy and minibatch RNG states are required")

    backend = _torch_backend(torch_module)
    (
        root_directory,
        absolute_directory,
        destination,
        checkpoint_relative,
        latest_relative,
    ) = _checkpoint_layout_for_write(
        root,
        checkpoint_directory,
        metadata.update_index,
    )
    payload = {
        "continuation_state": continuation_state.to_dict(),
        "metadata": metadata.to_dict(),
        "minibatch_rng_state": minibatch_rng_state,
        "model_state_dict": model_state_dict,
        "optimizer_state_dict": optimizer_state_dict,
        "policy_rng_state": policy_rng_state,
        "schema_version": TRAINING_CHECKPOINT_SCHEMA_VERSION,
    }
    with _checkpoint_lock(absolute_directory, exclusive=True, create=True):
        prior_pointer = read_latest_checkpoint_pointer(
            root_directory,
            checkpoint_directory=checkpoint_directory,
        )
        if prior_pointer is not None:
            prior = _load_published_checkpoint_locked(
                root_directory,
                prior_pointer,
                backend,
                expected_identity=None,
            )
            if prior.metadata.run_identity != metadata.run_identity:
                raise ArtifactValidationError(
                    "checkpoint root already belongs to a different full run identity"
                )
            if metadata.update_index < prior_pointer.update_index:
                raise ArtifactValidationError("checkpoint updates must increase monotonically")
            if metadata.update_index == prior_pointer.update_index:
                if prior.metadata != metadata:
                    raise ArtifactValidationError(
                        "same-update checkpoint retry metadata differs from published metadata"
                    )
                if prior.continuation_state != continuation_state:
                    raise ArtifactValidationError(
                        "same-update checkpoint retry continuation differs from published state"
                    )
                latest_path, _relative, _root = _resolve_destination_for_read(
                    root_directory,
                    latest_relative,
                )
                if latest_path is None:
                    raise ArtifactValidationError("latest checkpoint pointer disappeared")
                latest_bytes = latest_path.read_bytes()
                pointer_record = _record_for_checkpoint(latest_relative, latest_bytes)
                pruned = _prune_checkpoints(
                    absolute_directory,
                    keep_last=keep_last,
                    root=root_directory,
                    published_updates=prior_pointer.published_updates,
                    protected_checkpoint=destination,
                )
                return TrainingCheckpointArtifact(
                    checkpoint=prior_pointer.checkpoint,
                    latest_pointer=pointer_record,
                    metadata=metadata,
                    pruned_relative_paths=pruned,
                )

        checkpoint_record = _reuse_or_clean_same_update_orphan(
            destination,
            checkpoint_relative,
            backend,
            metadata,
            continuation_state,
        )
        if checkpoint_record is None:
            descriptor, temporary_name = tempfile.mkstemp(
                prefix=f".{destination.name}.",
                suffix=".tmp",
                dir=destination.parent,
            )
            temporary = Path(temporary_name)
            replaced = False
            try:
                os.fchmod(descriptor, 0o600)
                with os.fdopen(descriptor, "w+b") as file:
                    descriptor = -1
                    backend.save(payload, file)
                    file.flush()
                    os.fsync(file.fileno())
                expected_bytes = temporary.read_bytes()
                if not expected_bytes:
                    raise ArtifactWriteError("Torch produced an empty checkpoint")
                if destination.exists() or destination.is_symlink():
                    raise ArtifactWriteError("checkpoint destination appeared while locked")
                os.replace(temporary, destination)
                replaced = True
                _fsync_directory(destination.parent)
                readback = _readback_bytes(destination)
                if readback != expected_bytes:
                    raise ArtifactWriteError("checkpoint failed exact byte readback")
                try:
                    _verify_checkpoint_bytes(
                        backend,
                        readback,
                        expected_metadata=metadata,
                        expected_continuation=continuation_state,
                    )
                except ArtifactValidationError as error:
                    raise ArtifactWriteError(
                        "Torch checkpoint failed schema readback validation"
                    ) from error
            except BaseException as error:
                if descriptor >= 0:
                    os.close(descriptor)
                if replaced:
                    try:
                        _remove_new_checkpoint(destination)
                    except OSError as cleanup_error:
                        raise ArtifactWriteError(
                            "checkpoint verification failed and cleanup was unsuccessful"
                        ) from cleanup_error
                if isinstance(error, (ArtifactValidationError, ArtifactWriteError)):
                    raise
                raise ArtifactWriteError(
                    "failed to serialize and verify Torch checkpoint"
                ) from error
            finally:
                temporary.unlink(missing_ok=True)
            checkpoint_record = _record_for_checkpoint(
                checkpoint_relative,
                expected_bytes,
            )

        pointer = LatestCheckpointPointer(
            run_id=metadata.run_identity.run_id,
            run_identity_sha256=run_identity_sha256(metadata.run_identity),
            update_index=metadata.update_index,
            published_updates=(
                (prior_pointer.published_updates if prior_pointer is not None else ())
                + (metadata.update_index,)
            ),
            checkpoint=checkpoint_record,
        )
        # A verified checkpoint intentionally remains as a recognizable orphan if pointer
        # publication fails.  The prior pointer is preserved by atomic_write_json, and a retry of
        # this update verifies and reuses the orphan under the same lock.
        pointer_record = atomic_write_json(
            root_directory,
            latest_relative,
            pointer.to_dict(),
        )
        pruned = _prune_checkpoints(
            absolute_directory,
            keep_last=keep_last,
            root=root_directory,
            published_updates=pointer.published_updates,
            protected_checkpoint=destination,
        )
        if checkpoint_relative in pruned or not destination.is_file():
            raise ArtifactPruneError(
                "retention attempted to remove the published latest checkpoint"
            )
        return TrainingCheckpointArtifact(
            checkpoint=checkpoint_record,
            latest_pointer=pointer_record,
            metadata=metadata,
            pruned_relative_paths=pruned,
        )


__all__ = [
    "ARTIFACT_SCHEMA_VERSION",
    "LATEST_CHECKPOINT_SCHEMA_VERSION",
    "M7_FEATURE_SCHEMA_VERSION",
    "M7_REWARD_SCHEMA_VERSION",
    "RESUME_SEMANTICS",
    "RUN_IDENTITY_SCHEMA_VERSION",
    "TRAINING_CHECKPOINT_SCHEMA_VERSION",
    "TRAINING_CONTINUATION_SCHEMA_VERSION",
    "ArtifactError",
    "ArtifactPruneError",
    "ArtifactRecord",
    "ArtifactValidationError",
    "ArtifactWriteError",
    "LatestCheckpointPointer",
    "LoadedTrainingCheckpoint",
    "TrainingCheckpointArtifact",
    "TrainingCheckpointMetadata",
    "TrainingContinuationState",
    "TrainingRunIdentity",
    "atomic_write_bytes",
    "atomic_write_json",
    "canonical_json_bytes",
    "load_training_checkpoint",
    "read_latest_checkpoint_pointer",
    "read_strict_json",
    "run_identity_sha256",
    "save_training_checkpoint",
    "sha256_bytes",
    "sha256_file",
]
