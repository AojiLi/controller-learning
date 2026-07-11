"""Canonical manifests and the global report for the frozen M8 Test comparison.

This module only serializes already validated evidence.  It never opens Track assets, executes a
Controller, inspects the host, or writes an artifact.  The matching validators recompute the exact
canonical bytes from trusted in-memory inputs so omitted, added, reordered, or drifted evidence is
rejected rather than normalized away.
"""

from __future__ import annotations

import hashlib
import json
import math
import re
from collections.abc import Mapping
from dataclasses import asdict, dataclass
from dataclasses import field as dataclass_field
from pathlib import PurePosixPath
from types import MappingProxyType
from typing import Any, Final

from controller_learning.evaluation.controller_identity import (
    M8_CONTROLLER_FILE_MANIFEST,
    FrozenControllerIdentity,
)
from controller_learning.evaluation.final_benchmark import (
    M8_ACCEPTED_RESULT_RULE,
    M8_CONTROLLER_EXECUTION_MODEL,
    M8_CONTROLLER_ORDER,
    M8_ENVIRONMENT_LIFECYCLE,
    M8_FINAL_REPORT_SCHEMA_VERSION,
    M8_RANKING_RULE,
    M8_TEST_TRACK_COUNT,
    M8_TOTAL_EPISODES,
    M8FinalEvaluationConfig,
    controller_output_paths,
    formal_output_paths,
)
from controller_learning.evaluation.final_metrics import FINAL_METRICS_SCHEMA_VERSION
from controller_learning.evaluation.final_results import (
    FINAL_COMPARISON_SCHEMA_VERSION,
    FINAL_CONTROLLER_RESULTS_SCHEMA_VERSION,
    FINAL_CONTROLLER_SUMMARY_SCHEMA_VERSION,
    FinalControllerResult,
    controller_summary_payload,
    rank_final_controller_results,
)
from controller_learning.evaluation.final_runtime import (
    FINAL_ENVIRONMENT_LIFECYCLE_SCHEMA_VERSION,
    validate_final_memory_evidence,
    validate_final_runtime_evidence,
)
from controller_learning.evaluation.test_assets import TestPoolAccessEvidence
from controller_learning.evaluation.trajectory import TRAJECTORY_SCHEMA_VERSION

M8_CONTROLLER_RUN_MANIFEST_SCHEMA_VERSION: Final = (
    "controller-learning.m8-controller-run-manifest.v2"
)
M8_SOURCE_EVIDENCE_SCHEMA_VERSION: Final = "controller-learning.m8-source-evidence.v1"
M8_CONFIG_EVIDENCE_SCHEMA_VERSION: Final = "controller-learning.m8-config-evidence.v2"
M8_RUNTIME_EVIDENCE_SCHEMA_VERSION: Final = "controller-learning.m8-runtime-evidence.v1"
M8_MEMORY_EVIDENCE_SCHEMA_VERSION: Final = "controller-learning.m8-memory-evidence.v1"
M8_EXECUTION_EVIDENCE_SCHEMA_VERSION: Final = "controller-learning.m8-execution-evidence.v1"
M8_TRANSACTION_EVIDENCE_SCHEMA_VERSION: Final = "controller-learning.m8-transaction-evidence.v2"
M8_PRIVACY_EVIDENCE_SCHEMA_VERSION: Final = "controller-learning.m8-privacy-evidence.v1"
M8_TEST_ACCESS_AUDIT_SCHEMA_VERSION: Final = "controller-learning.m8-test-access-audit.v1"
M8_REPLACEMENT_LINEAGE_SCHEMA_VERSION: Final = "controller-learning.m8-replacement-lineage.v1"
M8_FINAL_CONFIG_PATH: Final = "configs/final_evaluation.toml"
M8_PIXI_LOCK_PATH: Final = "pixi.lock"
M8_REPORT_STATUS_BASIS: Final = "protocol_and_artifact_validation_only"
M8_REPORT_PHASE: Final = "EVALUATION_COMPLETE"

_SHA256_PATTERN = re.compile(r"^[0-9a-f]{64}$")
_REVISION_PATTERN = re.compile(r"^[0-9a-f]{40}$")
_MEDIA_TYPE_PATTERN = re.compile(r"^[a-z0-9][a-z0-9.+-]*/[a-z0-9][a-z0-9.+-]*$")
_WINDOWS_ABSOLUTE_PATH = re.compile(r"^[A-Za-z]:[\\/]")
_SECRET_PATTERNS: Final = (
    re.compile(r"ghp_[A-Za-z0-9]{20,}"),
    re.compile(r"github_pat_[A-Za-z0-9_]{20,}"),
    re.compile(r"AKIA[0-9A-Z]{16}"),
    re.compile(r"-----BEGIN (?:RSA |EC |OPENSSH )?PRIVATE KEY-----"),
    re.compile(r"(?i)(?:password|access[_-]?token|api[_-]?key)\s*[:=]\s*[^\s,}]+"),
)
_INPUT_REPORT_NAMES: Final = (
    "m5_track_admission_report",
    "m6_report",
    "m7_selection_report",
    "m7_export_report",
    "m7_controller_report",
    "m8_attempt_001_failure_report",
)
_ACCESS_CATEGORIES: Final = ("official_test_asset", "official_test_manifest")
_REPORT_MAX_BYTES: Final = 8 * 1024 * 1024


class FinalReportArtifactError(ValueError):
    """M8 report evidence or canonical bytes violate the frozen publication contract."""


def _safe_relative_path(value: object, *, field: str) -> str:
    if not isinstance(value, str) or not value or "\\" in value or "\x00" in value:
        raise FinalReportArtifactError(f"{field} must be a normalized relative POSIX path")
    path = PurePosixPath(value)
    if (
        path.is_absolute()
        or path.as_posix() != value
        or any(part in {"", ".", ".."} for part in path.parts)
    ):
        raise FinalReportArtifactError(f"{field} must be a normalized relative POSIX path")
    return value


def _sha256(value: object, *, field: str) -> str:
    if not isinstance(value, str) or _SHA256_PATTERN.fullmatch(value) is None:
        raise FinalReportArtifactError(f"{field} must be a lowercase SHA-256 digest")
    return value


def _finite(value: object, *, field: str, positive: bool = False) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise FinalReportArtifactError(f"{field} must be a finite number")
    result = float(value)
    if not math.isfinite(result) or result < 0.0 or (positive and result <= 0.0):
        qualifier = "positive" if positive else "non-negative"
        raise FinalReportArtifactError(f"{field} must be finite and {qualifier}")
    return result


def _nonnegative_integer(value: object, *, field: str, positive: bool = False) -> int:
    if type(value) is not int or value < (1 if positive else 0):
        qualifier = "positive" if positive else "non-negative"
        raise FinalReportArtifactError(f"{field} must be a {qualifier} integer")
    return value


def _public_text(value: object, *, field: str, maximum: int = 256) -> str:
    if (
        not isinstance(value, str)
        or not value
        or len(value) > maximum
        or any(ord(character) < 32 or ord(character) == 127 for character in value)
        or value.startswith("/")
        or _WINDOWS_ABSOLUTE_PATH.match(value) is not None
    ):
        raise FinalReportArtifactError(f"{field} must be sanitized public text")
    for pattern in _SECRET_PATTERNS:
        if pattern.search(value) is not None:
            raise FinalReportArtifactError(f"{field} contains secret-shaped text")
    return value


def _freeze_json(value: object, *, field: str) -> object:
    if value is None or type(value) in {bool, str}:
        return value
    if type(value) is int:
        return value
    if type(value) is float:
        if not math.isfinite(value):
            raise FinalReportArtifactError(f"{field} contains NaN or infinity")
        return value
    if isinstance(value, Mapping):
        if any(type(key) is not str for key in value):
            raise FinalReportArtifactError(f"{field} contains a non-string JSON key")
        return MappingProxyType(
            {key: _freeze_json(item, field=f"{field}.{key}") for key, item in value.items()}
        )
    if isinstance(value, (list, tuple)):
        return tuple(
            _freeze_json(item, field=f"{field}[{index}]") for index, item in enumerate(value)
        )
    raise FinalReportArtifactError(f"{field} is not JSON-compatible")


def _thaw_json(value: object) -> object:
    if isinstance(value, Mapping):
        return {key: _thaw_json(item) for key, item in value.items()}
    if isinstance(value, tuple):
        return [_thaw_json(item) for item in value]
    return value


def _canonical_json_bytes(value: Mapping[str, object]) -> bytes:
    content = json.dumps(
        _thaw_json(value),
        allow_nan=False,
        ensure_ascii=True,
        indent=2,
        sort_keys=True,
    )
    payload = f"{content}\n".encode("ascii")
    _reject_private_values(json.loads(payload))
    return payload


def _reject_private_values(value: object) -> None:
    if isinstance(value, Mapping):
        for item in value.values():
            _reject_private_values(item)
        return
    if isinstance(value, list):
        for item in value:
            _reject_private_values(item)
        return
    if not isinstance(value, str):
        return
    if value.startswith("/") or _WINDOWS_ABSOLUTE_PATH.match(value) is not None:
        raise FinalReportArtifactError("public report contains an absolute filesystem path")
    for pattern in _SECRET_PATTERNS:
        if pattern.search(value) is not None:
            raise FinalReportArtifactError("public report contains secret-shaped text")


def _strict_json_object(payload: bytes, *, artifact: str) -> Mapping[str, Any]:
    if type(payload) is not bytes:
        raise TypeError(f"{artifact} payload must be bytes")
    if not payload or len(payload) > _REPORT_MAX_BYTES:
        raise FinalReportArtifactError(f"{artifact} payload size is invalid")
    if not payload.endswith(b"\n") or b"\r" in payload:
        raise FinalReportArtifactError(f"{artifact} must use canonical LF JSON")

    def unique_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in pairs:
            if key in result:
                raise FinalReportArtifactError(f"{artifact} contains duplicate JSON key {key!r}")
            result[key] = value
        return result

    try:
        parsed = json.loads(
            payload,
            object_pairs_hook=unique_object,
            parse_constant=lambda value: (_ for _ in ()).throw(
                FinalReportArtifactError(f"{artifact} contains non-finite JSON number {value}")
            ),
        )
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise FinalReportArtifactError(f"{artifact} is not valid UTF-8 JSON") from error
    if not isinstance(parsed, Mapping):
        raise FinalReportArtifactError(f"{artifact} root must be an object")
    _reject_private_values(parsed)
    return parsed


@dataclass(frozen=True, slots=True)
class ArtifactRecord:
    """Path-safe content identity for one immutable public or frozen input artifact."""

    relative_path: str
    sha256: str
    size_bytes: int
    media_type: str
    schema_version: str | int | None = None

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "relative_path",
            _safe_relative_path(self.relative_path, field="artifact.relative_path"),
        )
        object.__setattr__(self, "sha256", _sha256(self.sha256, field="artifact.sha256"))
        _nonnegative_integer(self.size_bytes, field="artifact.size_bytes")
        if (
            not isinstance(self.media_type, str)
            or _MEDIA_TYPE_PATTERN.fullmatch(self.media_type) is None
        ):
            raise FinalReportArtifactError("artifact.media_type must be a lowercase media type")
        if self.schema_version is not None:
            if type(self.schema_version) is int:
                if self.schema_version < 1:
                    raise FinalReportArtifactError("numeric schema_version must be positive")
            elif isinstance(self.schema_version, str):
                _public_text(self.schema_version, field="artifact.schema_version")
            else:
                raise FinalReportArtifactError("schema_version must be a string, integer, or null")

    @classmethod
    def from_bytes(
        cls,
        relative_path: str,
        payload: bytes,
        media_type: str,
        schema_version: str | int | None = None,
    ) -> ArtifactRecord:
        """Build an exact record without opening or trusting a filesystem path."""

        if type(payload) is not bytes:
            raise TypeError("payload must be immutable bytes")
        return cls(
            relative_path=relative_path,
            sha256=hashlib.sha256(payload).hexdigest(),
            size_bytes=len(payload),
            media_type=media_type,
            schema_version=schema_version,
        )

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


@dataclass(frozen=True, slots=True)
class SourceEvidence:
    """One clean, full Git source revision."""

    revision: str
    worktree_clean: bool = True
    schema_version: str = M8_SOURCE_EVIDENCE_SCHEMA_VERSION

    def __post_init__(self) -> None:
        if not isinstance(self.revision, str) or _REVISION_PATTERN.fullmatch(self.revision) is None:
            raise FinalReportArtifactError("source revision must be 40 lowercase hexadecimal chars")
        if self.worktree_clean is not True:
            raise FinalReportArtifactError("formal source evidence must be clean")
        if self.schema_version != M8_SOURCE_EVIDENCE_SCHEMA_VERSION:
            raise FinalReportArtifactError("source evidence schema differs")

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


@dataclass(frozen=True, slots=True)
class FinalConfigEvidence:
    """Raw TOML identity plus its strictly parsed frozen configuration value."""

    path: str
    sha256: str
    value: Mapping[str, object]
    schema_version: str = M8_CONFIG_EVIDENCE_SCHEMA_VERSION

    def __post_init__(self) -> None:
        if _safe_relative_path(self.path, field="config.path") != M8_FINAL_CONFIG_PATH:
            raise FinalReportArtifactError(f"final config path must be {M8_FINAL_CONFIG_PATH!r}")
        object.__setattr__(self, "sha256", _sha256(self.sha256, field="config.sha256"))
        if not isinstance(self.value, Mapping):
            raise TypeError("config.value must be a mapping")
        frozen = _freeze_json(self.value, field="config.value")
        if not isinstance(frozen, Mapping):  # pragma: no cover - guarded above
            raise TypeError("config.value must be a mapping")
        object.__setattr__(self, "value", frozen)
        if self.schema_version != M8_CONFIG_EVIDENCE_SCHEMA_VERSION:
            raise FinalReportArtifactError("config evidence schema differs")

    @classmethod
    def from_bytes(
        cls,
        config: M8FinalEvaluationConfig,
        payload: bytes,
        *,
        path: str = M8_FINAL_CONFIG_PATH,
    ) -> FinalConfigEvidence:
        if not isinstance(config, M8FinalEvaluationConfig):
            raise TypeError("config must be an M8FinalEvaluationConfig")
        if type(payload) is not bytes:
            raise TypeError("payload must be immutable bytes")
        return cls(
            path=path,
            sha256=hashlib.sha256(payload).hexdigest(),
            value=config.to_dict(),
        )

    def to_dict(self) -> dict[str, object]:
        return {
            "path": self.path,
            "schema_version": self.schema_version,
            "sha256": self.sha256,
            "value": _thaw_json(self.value),
        }


@dataclass(frozen=True, slots=True)
class RuntimeEvidence:
    """Immutable wrapper around measured ``final_runtime`` public evidence."""

    value: Mapping[str, object]

    def __post_init__(self) -> None:
        if not isinstance(self.value, Mapping):
            raise TypeError("runtime evidence must be a mapping")
        try:
            validate_final_runtime_evidence(self.value)
        except (TypeError, ValueError) as error:
            raise FinalReportArtifactError("runtime evidence differs from final_runtime") from error
        frozen = _freeze_json(self.value, field="runtime")
        if not isinstance(frozen, Mapping):  # pragma: no cover - guarded above
            raise TypeError("runtime evidence must be a mapping")
        object.__setattr__(self, "value", frozen)

    @classmethod
    def from_mapping(cls, value: Mapping[str, object]) -> RuntimeEvidence:
        """Validate and freeze ``collect_final_runtime_evidence`` public output."""

        return cls(value=value)

    def to_dict(self) -> dict[str, object]:
        value = _thaw_json(self.value)
        if not isinstance(value, dict):  # pragma: no cover - guaranteed by constructor
            raise TypeError("runtime evidence must be a mapping")
        return value


@dataclass(frozen=True, slots=True)
class MemoryEvidence:
    """Immutable labelled phase-boundary samples from ``FinalMemoryRecorder``."""

    value: Mapping[str, object]

    def __post_init__(self) -> None:
        if not isinstance(self.value, Mapping):
            raise TypeError("memory evidence must be a mapping")
        try:
            validate_final_memory_evidence(self.value)
        except (TypeError, ValueError) as error:
            raise FinalReportArtifactError("memory evidence differs from final_runtime") from error
        frozen = _freeze_json(self.value, field="memory")
        if not isinstance(frozen, Mapping):  # pragma: no cover - guarded above
            raise TypeError("memory evidence must be a mapping")
        object.__setattr__(self, "value", frozen)

    @classmethod
    def from_mapping(cls, value: Mapping[str, object]) -> MemoryEvidence:
        """Validate and freeze ``FinalMemoryRecorder.evidence`` output."""

        return cls(value=value)

    def to_dict(self) -> dict[str, object]:
        value = _thaw_json(self.value)
        if not isinstance(value, dict):  # pragma: no cover - guaranteed by constructor
            raise TypeError("memory evidence must be a mapping")
        return value


@dataclass(frozen=True, slots=True)
class EnvironmentLifecycleEvidence:
    """Immutable measured lifecycle from ``MeasuredFinalEnvironmentFactory.evidence``."""

    value: Mapping[str, object]

    def __post_init__(self) -> None:
        if not isinstance(self.value, Mapping):
            raise TypeError("measured environment lifecycle must be a mapping")
        expected = {
            "schema_version",
            "environment_instance_count",
            "environment_create_wall_time_s",
            "first_reset_wall_time_including_lazy_compilation_s",
            "first_step_wall_time_including_lazy_compilation_s",
            "reset_count",
            "expected_reset_count",
            "step_count",
            "expected_step_count",
            "close_count",
            "method",
        }
        if set(self.value) != expected:
            raise FinalReportArtifactError("measured environment lifecycle keys differ")
        if self.value["schema_version"] != FINAL_ENVIRONMENT_LIFECYCLE_SCHEMA_VERSION:
            raise FinalReportArtifactError("measured environment lifecycle schema differs")
        exact_counts = {
            "environment_instance_count": 1,
            "reset_count": M8_TOTAL_EPISODES,
            "expected_reset_count": M8_TOTAL_EPISODES,
            "close_count": 1,
        }
        for name, expected_value in exact_counts.items():
            if type(self.value[name]) is not int or self.value[name] != expected_value:
                raise FinalReportArtifactError(
                    f"measured environment lifecycle {name} must be {expected_value}"
                )
        for name in ("step_count", "expected_step_count"):
            _nonnegative_integer(
                self.value[name], field=f"measured lifecycle {name}", positive=True
            )
        if self.value["step_count"] != self.value["expected_step_count"]:
            raise FinalReportArtifactError("measured lifecycle actual/expected steps differ")
        for name in (
            "environment_create_wall_time_s",
            "first_reset_wall_time_including_lazy_compilation_s",
            "first_step_wall_time_including_lazy_compilation_s",
        ):
            _finite(self.value[name], field=f"measured lifecycle {name}")
        _public_text(self.value["method"], field="measured lifecycle method", maximum=512)
        frozen = _freeze_json(self.value, field="measured_environment_lifecycle")
        if not isinstance(frozen, Mapping):  # pragma: no cover
            raise TypeError("measured environment lifecycle must be a mapping")
        object.__setattr__(self, "value", frozen)

    @classmethod
    def from_mapping(cls, value: Mapping[str, object]) -> EnvironmentLifecycleEvidence:
        return cls(value=value)

    def to_dict(self) -> dict[str, object]:
        value = _thaw_json(self.value)
        if not isinstance(value, dict):  # pragma: no cover
            raise TypeError("measured environment lifecycle must be a mapping")
        return value


@dataclass(frozen=True, slots=True)
class ExecutionEvidence:
    """Exact lifecycle, workload, throughput, replay, retry, and diagnostic evidence."""

    environment_steps_by_controller: Mapping[str, int]
    controller_wall_time_s: Mapping[str, float]
    total_environment_steps: int
    wall_time_s: float
    environment_steps_per_second: float
    initialization_over_soft_limit_rows: Mapping[str, tuple[int, ...]]
    measured_environment_lifecycle: EnvironmentLifecycleEvidence | Mapping[str, object]
    environment_lifecycle: str = M8_ENVIRONMENT_LIFECYCLE
    controller_execution_model: str = M8_CONTROLLER_EXECUTION_MODEL
    controller_order: tuple[str, ...] = M8_CONTROLLER_ORDER
    row_order: tuple[int, ...] = tuple(range(M8_TEST_TRACK_COUNT))
    environment_instance_count: int = 1
    fresh_controller_per_episode: bool = True
    fresh_controller_instance_count: int = M8_TOTAL_EPISODES
    episode_count: int = M8_TOTAL_EPISODES
    controller_init_soft_limit_s: float = 30.0
    replay_row_index: int = 0
    replay_captured_from_same_rollout: bool = True
    replay_environment_instance_count: int = 0
    retry_count: int = 0
    automatic_retry_after_test_bound: bool = False
    numerical_failure_count: int = 0
    schema_version: str = M8_EXECUTION_EVIDENCE_SCHEMA_VERSION

    def __post_init__(self) -> None:
        if self.environment_lifecycle != M8_ENVIRONMENT_LIFECYCLE:
            raise FinalReportArtifactError("execution environment lifecycle differs")
        if self.controller_execution_model != M8_CONTROLLER_EXECUTION_MODEL:
            raise FinalReportArtifactError("Controller execution model differs")
        if self.controller_order != M8_CONTROLLER_ORDER:
            raise FinalReportArtifactError("execution Controller order must be pid, mpc, ppo")
        if self.row_order != tuple(range(M8_TEST_TRACK_COUNT)):
            raise FinalReportArtifactError("execution rows must preserve 0..19 order")
        exact_values = {
            "environment_instance_count": (self.environment_instance_count, 1),
            "fresh_controller_instance_count": (
                self.fresh_controller_instance_count,
                M8_TOTAL_EPISODES,
            ),
            "episode_count": (self.episode_count, M8_TOTAL_EPISODES),
            "replay_row_index": (self.replay_row_index, 0),
            "replay_environment_instance_count": (self.replay_environment_instance_count, 0),
            "retry_count": (self.retry_count, 0),
            "numerical_failure_count": (self.numerical_failure_count, 0),
        }
        for field, (actual, expected) in exact_values.items():
            if type(actual) is not int or actual != expected:
                raise FinalReportArtifactError(f"execution {field} must be exactly {expected}")
        if self.fresh_controller_per_episode is not True:
            raise FinalReportArtifactError("every episode must use a fresh Controller instance")
        if self.replay_captured_from_same_rollout is not True:
            raise FinalReportArtifactError("row-zero replay must come from the canonical rollout")
        if self.automatic_retry_after_test_bound is not False:
            raise FinalReportArtifactError("automatic Test-bound retry must be disabled")
        if self.controller_init_soft_limit_s != 30.0:
            raise FinalReportArtifactError("Controller init soft limit must be exactly 30 seconds")

        steps = dict(self.environment_steps_by_controller)
        times = dict(self.controller_wall_time_s)
        slow_rows = dict(self.initialization_over_soft_limit_rows)
        if set(steps) != set(M8_CONTROLLER_ORDER):
            raise FinalReportArtifactError("environment step evidence must cover all Controllers")
        if set(times) != set(M8_CONTROLLER_ORDER):
            raise FinalReportArtifactError("Controller wall times must cover all Controllers")
        if set(slow_rows) != set(M8_CONTROLLER_ORDER):
            raise FinalReportArtifactError("init diagnostics must cover all Controllers")
        for name in M8_CONTROLLER_ORDER:
            _nonnegative_integer(steps[name], field=f"execution.steps.{name}", positive=True)
            times[name] = _finite(
                times[name], field=f"execution.controller_wall_time_s.{name}", positive=True
            )
            rows = slow_rows[name]
            if not isinstance(rows, tuple) or any(
                type(row) is not int or not 0 <= row < M8_TEST_TRACK_COUNT for row in rows
            ):
                raise FinalReportArtifactError("init soft-limit rows must be ordered row indices")
            if tuple(sorted(set(rows))) != rows:
                raise FinalReportArtifactError("init soft-limit rows must be unique and sorted")
        if self.total_environment_steps != sum(steps.values()):
            raise FinalReportArtifactError("total environment steps differ from Controller totals")
        wall = _finite(self.wall_time_s, field="execution.wall_time_s", positive=True)
        throughput = _finite(
            self.environment_steps_per_second,
            field="execution.environment_steps_per_second",
            positive=True,
        )
        if not math.isclose(
            throughput,
            self.total_environment_steps / wall,
            rel_tol=1e-12,
            abs_tol=0.0,
        ):
            raise FinalReportArtifactError("execution throughput is not steps / wall time")
        object.__setattr__(self, "environment_steps_by_controller", MappingProxyType(steps))
        object.__setattr__(self, "controller_wall_time_s", MappingProxyType(times))
        object.__setattr__(self, "initialization_over_soft_limit_rows", MappingProxyType(slow_rows))
        object.__setattr__(self, "wall_time_s", wall)
        object.__setattr__(self, "environment_steps_per_second", throughput)
        lifecycle = self.measured_environment_lifecycle
        if not isinstance(lifecycle, EnvironmentLifecycleEvidence):
            lifecycle = EnvironmentLifecycleEvidence.from_mapping(lifecycle)
        if lifecycle.value["step_count"] != self.total_environment_steps:
            raise FinalReportArtifactError(
                "measured environment step count differs from execution totals"
            )
        object.__setattr__(self, "measured_environment_lifecycle", lifecycle)
        if self.schema_version != M8_EXECUTION_EVIDENCE_SCHEMA_VERSION:
            raise FinalReportArtifactError("execution evidence schema differs")

    @classmethod
    def from_results(
        cls,
        results: Mapping[str, FinalControllerResult],
        *,
        wall_time_s: float,
        controller_wall_time_s: Mapping[str, float],
        initialization_over_soft_limit_rows: Mapping[str, tuple[int, ...]],
        measured_environment_lifecycle: EnvironmentLifecycleEvidence | Mapping[str, object],
    ) -> ExecutionEvidence:
        ordered = _validated_results(results)
        steps = {name: ordered[name].summary.environment_steps for name in M8_CONTROLLER_ORDER}
        wall = _finite(wall_time_s, field="execution.wall_time_s", positive=True)
        return cls(
            environment_steps_by_controller=steps,
            controller_wall_time_s=controller_wall_time_s,
            total_environment_steps=sum(steps.values()),
            wall_time_s=wall,
            environment_steps_per_second=sum(steps.values()) / wall,
            initialization_over_soft_limit_rows=initialization_over_soft_limit_rows,
            measured_environment_lifecycle=measured_environment_lifecycle,
        )

    @classmethod
    def from_workload(
        cls,
        workload: object,
        results: Mapping[str, FinalControllerResult],
        *,
        measured_environment_lifecycle: EnvironmentLifecycleEvidence | Mapping[str, object],
    ) -> ExecutionEvidence:
        """Build evidence from ``FinalWorkloadExecution`` without importing it at module load."""

        from controller_learning.evaluation.final_execution import FinalWorkloadExecution

        if not isinstance(workload, FinalWorkloadExecution):
            raise TypeError("workload must be a FinalWorkloadExecution")
        if workload.environment_instance_count != 1 or workload.fresh_runner_instance_count != 60:
            raise FinalReportArtifactError("workload lifecycle differs from the frozen protocol")
        return cls.from_results(
            results,
            wall_time_s=workload.wall_time_s,
            controller_wall_time_s={
                name: workload.controller_results[name].wall_time_s for name in M8_CONTROLLER_ORDER
            },
            initialization_over_soft_limit_rows={
                name: workload.controller_results[name].initialization_over_30s_rows
                for name in M8_CONTROLLER_ORDER
            },
            measured_environment_lifecycle=measured_environment_lifecycle,
        )

    @classmethod
    def from_mapping(cls, value: Mapping[str, object]) -> ExecutionEvidence:
        """Restore and revalidate the exact mapping emitted by :meth:`to_dict`."""

        if not isinstance(value, Mapping):
            raise TypeError("execution evidence must be a mapping")
        expected = {
            "automatic_retry_after_test_bound",
            "controller_execution_model",
            "controller_init_soft_limit_s",
            "controller_order",
            "controller_wall_time_s",
            "environment_instance_count",
            "environment_lifecycle",
            "environment_steps_by_controller",
            "environment_steps_per_second",
            "episode_count",
            "fresh_controller_instance_count",
            "fresh_controller_per_episode",
            "initialization_over_soft_limit_rows",
            "measured_environment_lifecycle",
            "numerical_failure_count",
            "replay_captured_from_same_rollout",
            "replay_environment_instance_count",
            "replay_row_index",
            "retry_count",
            "row_order",
            "schema_version",
            "total_environment_steps",
            "wall_time_s",
        }
        if set(value) != expected:
            raise FinalReportArtifactError("execution evidence mapping keys differ")
        slow_rows = value["initialization_over_soft_limit_rows"]
        if not isinstance(slow_rows, Mapping):
            raise FinalReportArtifactError("execution init diagnostics must be a mapping")
        try:
            return cls(
                environment_steps_by_controller=value["environment_steps_by_controller"],  # type: ignore[arg-type]
                controller_wall_time_s=value["controller_wall_time_s"],  # type: ignore[arg-type]
                total_environment_steps=value["total_environment_steps"],  # type: ignore[arg-type]
                wall_time_s=value["wall_time_s"],  # type: ignore[arg-type]
                environment_steps_per_second=value["environment_steps_per_second"],  # type: ignore[arg-type]
                initialization_over_soft_limit_rows={
                    name: tuple(slow_rows[name])  # type: ignore[arg-type]
                    for name in M8_CONTROLLER_ORDER
                },
                measured_environment_lifecycle=value["measured_environment_lifecycle"],  # type: ignore[arg-type]
                environment_lifecycle=value["environment_lifecycle"],  # type: ignore[arg-type]
                controller_execution_model=value["controller_execution_model"],  # type: ignore[arg-type]
                controller_order=tuple(value["controller_order"]),  # type: ignore[arg-type]
                row_order=tuple(value["row_order"]),  # type: ignore[arg-type]
                environment_instance_count=value["environment_instance_count"],  # type: ignore[arg-type]
                fresh_controller_per_episode=value["fresh_controller_per_episode"],  # type: ignore[arg-type]
                fresh_controller_instance_count=value["fresh_controller_instance_count"],  # type: ignore[arg-type]
                episode_count=value["episode_count"],  # type: ignore[arg-type]
                controller_init_soft_limit_s=value["controller_init_soft_limit_s"],  # type: ignore[arg-type]
                replay_row_index=value["replay_row_index"],  # type: ignore[arg-type]
                replay_captured_from_same_rollout=value["replay_captured_from_same_rollout"],  # type: ignore[arg-type]
                replay_environment_instance_count=value["replay_environment_instance_count"],  # type: ignore[arg-type]
                retry_count=value["retry_count"],  # type: ignore[arg-type]
                automatic_retry_after_test_bound=value["automatic_retry_after_test_bound"],  # type: ignore[arg-type]
                numerical_failure_count=value["numerical_failure_count"],  # type: ignore[arg-type]
                schema_version=value["schema_version"],  # type: ignore[arg-type]
            )
        except KeyError as error:
            raise FinalReportArtifactError(
                "execution evidence Controller coverage differs"
            ) from error

    def to_dict(self) -> dict[str, object]:
        return {
            "automatic_retry_after_test_bound": self.automatic_retry_after_test_bound,
            "controller_execution_model": self.controller_execution_model,
            "controller_init_soft_limit_s": self.controller_init_soft_limit_s,
            "controller_order": list(self.controller_order),
            "controller_wall_time_s": dict(self.controller_wall_time_s),
            "environment_instance_count": self.environment_instance_count,
            "environment_lifecycle": self.environment_lifecycle,
            "environment_steps_by_controller": dict(self.environment_steps_by_controller),
            "environment_steps_per_second": self.environment_steps_per_second,
            "episode_count": self.episode_count,
            "fresh_controller_instance_count": self.fresh_controller_instance_count,
            "fresh_controller_per_episode": self.fresh_controller_per_episode,
            "initialization_over_soft_limit_rows": {
                name: list(self.initialization_over_soft_limit_rows[name])
                for name in M8_CONTROLLER_ORDER
            },
            "measured_environment_lifecycle": self.measured_environment_lifecycle.to_dict(),
            "numerical_failure_count": self.numerical_failure_count,
            "replay_captured_from_same_rollout": self.replay_captured_from_same_rollout,
            "replay_environment_instance_count": self.replay_environment_instance_count,
            "replay_row_index": self.replay_row_index,
            "retry_count": self.retry_count,
            "row_order": list(self.row_order),
            "schema_version": self.schema_version,
            "total_environment_steps": self.total_environment_steps,
            "wall_time_s": self.wall_time_s,
        }


@dataclass(frozen=True, slots=True)
class TestAccessAuditEvidence:
    """Path-sanitized proof that the process opened only the two official Test files."""

    open_event_counts: Mapping[str, int]
    open_event_sequence: tuple[Mapping[str, object], ...]
    all_track_reads_forbidden: bool = True
    audit_hook_installed_before_preflight: bool = True
    denied_event_count: int = 0
    denied_mutation_event_count: int = 0
    denied_mutation_event_types: Mapping[str, int] = dataclass_field(
        default_factory=lambda: MappingProxyType({})
    )
    opened_path_categories: tuple[str, ...] = _ACCESS_CATEGORIES
    opened_splits: tuple[str, ...] = ("test",)
    pre_test_open_event_count: int = 0
    test_loaded: bool = True
    test_reads_enabled: bool = True
    track_cache_opened: bool = False
    train_opened: bool = False
    validation_opened: bool = False
    schema_version: str = M8_TEST_ACCESS_AUDIT_SCHEMA_VERSION

    def __post_init__(self) -> None:
        required_true = {
            "all_track_reads_forbidden": self.all_track_reads_forbidden,
            "audit_hook_installed_before_preflight": self.audit_hook_installed_before_preflight,
            "test_loaded": self.test_loaded,
            "test_reads_enabled": self.test_reads_enabled,
        }
        if any(value is not True for value in required_true.values()):
            raise FinalReportArtifactError("Test access audit did not complete its guarded load")
        required_false = {
            "track_cache_opened": self.track_cache_opened,
            "train_opened": self.train_opened,
            "validation_opened": self.validation_opened,
        }
        if any(value is not False for value in required_false.values()):
            raise FinalReportArtifactError("Test access audit contains split or cache leakage")
        for field in (
            "denied_event_count",
            "denied_mutation_event_count",
            "pre_test_open_event_count",
        ):
            if getattr(self, field) != 0:
                raise FinalReportArtifactError(f"Test access audit {field} must be zero")
        if dict(self.denied_mutation_event_types):
            raise FinalReportArtifactError("Test access audit contains denied mutation evidence")
        if self.opened_path_categories != _ACCESS_CATEGORIES or self.opened_splits != ("test",):
            raise FinalReportArtifactError("Test access audit must identify only the Test split")
        counts = dict(self.open_event_counts)
        if set(counts) != set(_ACCESS_CATEGORIES) or any(
            type(value) is not int or value < 1 for value in counts.values()
        ):
            raise FinalReportArtifactError("Test access audit must open both Test files")
        if not isinstance(self.open_event_sequence, tuple) or not self.open_event_sequence:
            raise FinalReportArtifactError("Test access sequence must be a non-empty tuple")
        sequence: list[Mapping[str, object]] = []
        observed_counts = {category: 0 for category in _ACCESS_CATEGORIES}
        for index, event in enumerate(self.open_event_sequence):
            if not isinstance(event, Mapping) or set(event) != {"category", "flags", "mode"}:
                raise FinalReportArtifactError("Test access events must contain exact keys")
            category = event["category"]
            flags = event["flags"]
            mode = event["mode"]
            if category not in _ACCESS_CATEGORIES:
                raise FinalReportArtifactError("Test access sequence contains a forbidden category")
            if flags is not None and type(flags) is not int:
                raise FinalReportArtifactError("Test access flags must be an integer or null")
            if mode is not None and not isinstance(mode, str):
                raise FinalReportArtifactError("Test access mode must be a string or null")
            if isinstance(mode, str) and any(token in mode for token in "wax+"):
                raise FinalReportArtifactError("Test access sequence contains a write mode")
            observed_counts[str(category)] += 1
            frozen = _freeze_json(event, field=f"test_access.event[{index}]")
            if not isinstance(frozen, Mapping):  # pragma: no cover
                raise TypeError("Test access event must be a mapping")
            sequence.append(frozen)
        if observed_counts != counts:
            raise FinalReportArtifactError("Test access counts differ from the event sequence")
        object.__setattr__(self, "open_event_counts", MappingProxyType(counts))
        object.__setattr__(self, "open_event_sequence", tuple(sequence))
        object.__setattr__(self, "denied_mutation_event_types", MappingProxyType({}))
        if self.schema_version != M8_TEST_ACCESS_AUDIT_SCHEMA_VERSION:
            raise FinalReportArtifactError("Test access audit schema differs")

    @classmethod
    def from_mapping(cls, value: Mapping[str, object]) -> TestAccessAuditEvidence:
        """Validate and freeze the exact mapping returned by ``M8TestAssetAccessGuard``."""

        if not isinstance(value, Mapping):
            raise TypeError("Test access evidence must be a mapping")
        expected = {
            "all_track_reads_forbidden",
            "audit_hook_installed_before_preflight",
            "denied_event_count",
            "denied_mutation_event_count",
            "denied_mutation_event_types",
            "open_event_counts",
            "open_event_sequence",
            "opened_path_categories",
            "opened_splits",
            "pre_test_open_event_count",
            "test_loaded",
            "test_reads_enabled",
            "track_cache_opened",
            "train_opened",
            "validation_opened",
        }
        if set(value) != expected:
            raise FinalReportArtifactError("Test access evidence keys differ")
        sequence_value = value["open_event_sequence"]
        if not isinstance(sequence_value, list):
            raise FinalReportArtifactError("open_event_sequence must be a list")
        return cls(
            open_event_counts=value["open_event_counts"],  # type: ignore[arg-type]
            open_event_sequence=tuple(sequence_value),  # type: ignore[arg-type]
            all_track_reads_forbidden=value["all_track_reads_forbidden"],  # type: ignore[arg-type]
            audit_hook_installed_before_preflight=value["audit_hook_installed_before_preflight"],  # type: ignore[arg-type]
            denied_event_count=value["denied_event_count"],  # type: ignore[arg-type]
            denied_mutation_event_count=value["denied_mutation_event_count"],  # type: ignore[arg-type]
            denied_mutation_event_types=value["denied_mutation_event_types"],  # type: ignore[arg-type]
            opened_path_categories=tuple(value["opened_path_categories"]),  # type: ignore[arg-type]
            opened_splits=tuple(value["opened_splits"]),  # type: ignore[arg-type]
            pre_test_open_event_count=value["pre_test_open_event_count"],  # type: ignore[arg-type]
            test_loaded=value["test_loaded"],  # type: ignore[arg-type]
            test_reads_enabled=value["test_reads_enabled"],  # type: ignore[arg-type]
            track_cache_opened=value["track_cache_opened"],  # type: ignore[arg-type]
            train_opened=value["train_opened"],  # type: ignore[arg-type]
            validation_opened=value["validation_opened"],  # type: ignore[arg-type]
        )

    def to_dict(self) -> dict[str, object]:
        return {
            "all_track_reads_forbidden": self.all_track_reads_forbidden,
            "audit_hook_installed_before_preflight": self.audit_hook_installed_before_preflight,
            "denied_event_count": self.denied_event_count,
            "denied_mutation_event_count": self.denied_mutation_event_count,
            "denied_mutation_event_types": dict(self.denied_mutation_event_types),
            "open_event_counts": dict(self.open_event_counts),
            "open_event_sequence": [_thaw_json(event) for event in self.open_event_sequence],
            "opened_path_categories": list(self.opened_path_categories),
            "opened_splits": list(self.opened_splits),
            "pre_test_open_event_count": self.pre_test_open_event_count,
            "schema_version": self.schema_version,
            "test_loaded": self.test_loaded,
            "test_reads_enabled": self.test_reads_enabled,
            "track_cache_opened": self.track_cache_opened,
            "train_opened": self.train_opened,
            "validation_opened": self.validation_opened,
        }


def validate_durable_execution_evidence_mapping(
    value: Mapping[str, object],
) -> None:
    """Validate every nested object in the post-close transaction seal.

    This is intentionally independent from staged publication bytes so the durable transaction
    cannot accept an empty or syntactically plausible seal before report construction begins.
    """

    if not isinstance(value, Mapping):
        raise TypeError("durable execution evidence must be a mapping")
    expected = {
        "asset_access",
        "execution",
        "memory",
        "runtime",
        "schema_version",
        "test_assets",
    }
    if any(type(key) is not str for key in value) or set(value) != expected:
        raise FinalReportArtifactError("durable execution evidence mapping keys differ")
    if value["schema_version"] != M8_EXECUTION_EVIDENCE_SCHEMA_VERSION:
        raise FinalReportArtifactError("durable execution evidence schema differs")
    for field in ("asset_access", "execution", "memory", "runtime", "test_assets"):
        if not isinstance(value[field], Mapping):
            raise TypeError(f"durable execution evidence {field} must be a mapping")
    thawed: dict[str, Mapping[str, object]] = {}
    for field in ("asset_access", "execution", "memory", "runtime", "test_assets"):
        restored = _thaw_json(value[field])
        if not isinstance(restored, dict):  # pragma: no cover - mapping checked above
            raise TypeError(f"durable execution evidence {field} must be a mapping")
        thawed[field] = restored
    asset_access = dict(thawed["asset_access"])
    if asset_access.pop("schema_version", None) != M8_TEST_ACCESS_AUDIT_SCHEMA_VERSION:
        raise FinalReportArtifactError("durable Test access audit schema differs")
    TestAccessAuditEvidence.from_mapping(asset_access)
    ExecutionEvidence.from_mapping(thawed["execution"])
    MemoryEvidence.from_mapping(thawed["memory"])
    RuntimeEvidence.from_mapping(thawed["runtime"])
    TestPoolAccessEvidence.from_mapping(thawed["test_assets"])


@dataclass(frozen=True, slots=True)
class DurableExecutionEvidenceSeal:
    """Canonical post-close transaction seal cross-checked against report evidence."""

    value: Mapping[str, object]

    def __post_init__(self) -> None:
        if not isinstance(self.value, Mapping):
            raise TypeError("durable execution evidence seal must be a mapping")
        validate_durable_execution_evidence_mapping(self.value)
        from controller_learning.evaluation.attempt_transaction import (
            canonical_execution_evidence_bytes,
        )

        try:
            canonical_execution_evidence_bytes(self.value)
        except (TypeError, ValueError) as error:
            raise FinalReportArtifactError(
                "durable execution evidence seal differs from transaction schema"
            ) from error
        frozen = _freeze_json(self.value, field="durable_execution_evidence")
        if not isinstance(frozen, Mapping):  # pragma: no cover
            raise TypeError("durable execution evidence seal must be a mapping")
        object.__setattr__(self, "value", frozen)

    @classmethod
    def from_mapping(cls, value: Mapping[str, object]) -> DurableExecutionEvidenceSeal:
        return cls(value=value)

    @classmethod
    def from_evidence(
        cls,
        *,
        test_access_audit: TestAccessAuditEvidence,
        execution: ExecutionEvidence,
        memory: MemoryEvidence,
        runtime: RuntimeEvidence,
        test_pool_access: TestPoolAccessEvidence,
    ) -> DurableExecutionEvidenceSeal:
        """Create the exact mapping that must be sealed before deterministic finalization."""

        if not isinstance(test_access_audit, TestAccessAuditEvidence):
            raise TypeError("test_access_audit must be TestAccessAuditEvidence")
        if not isinstance(execution, ExecutionEvidence):
            raise TypeError("execution must be ExecutionEvidence")
        if not isinstance(memory, MemoryEvidence) or not isinstance(runtime, RuntimeEvidence):
            raise TypeError("memory and runtime must use their typed evidence classes")
        if not isinstance(test_pool_access, TestPoolAccessEvidence):
            raise TypeError("test_pool_access must be TestPoolAccessEvidence")
        return cls(
            value={
                "asset_access": test_access_audit.to_dict(),
                "execution": execution.to_dict(),
                "memory": memory.to_dict(),
                "runtime": runtime.to_dict(),
                "schema_version": M8_EXECUTION_EVIDENCE_SCHEMA_VERSION,
                "test_assets": _pool_access_payload(test_pool_access),
            }
        )

    def cross_check(
        self,
        *,
        test_access_audit: TestAccessAuditEvidence,
        execution: ExecutionEvidence,
        memory: MemoryEvidence,
        runtime: RuntimeEvidence,
        test_pool_access: TestPoolAccessEvidence,
    ) -> None:
        expected = type(self).from_evidence(
            test_access_audit=test_access_audit,
            execution=execution,
            memory=memory,
            runtime=runtime,
            test_pool_access=test_pool_access,
        )
        if self.to_dict() != expected.to_dict():
            raise FinalReportArtifactError(
                "durable execution seal differs from report runtime, memory, access, or workload"
            )

    @property
    def sha256(self) -> str:
        from controller_learning.evaluation.attempt_transaction import (
            canonical_execution_evidence_bytes,
        )

        return hashlib.sha256(canonical_execution_evidence_bytes(self.value)).hexdigest()

    def to_dict(self) -> dict[str, object]:
        value = _thaw_json(self.value)
        if not isinstance(value, dict):  # pragma: no cover
            raise TypeError("durable execution evidence seal must be a mapping")
        return value

    def public_binding(self) -> dict[str, object]:
        return {
            "cross_checked": True,
            "schema_version": M8_EXECUTION_EVIDENCE_SCHEMA_VERSION,
            "sha256": self.sha256,
        }


@dataclass(frozen=True, slots=True)
class TransactionEvidence:
    """Durable attempt state when staged report bytes receive semantic validation."""

    durable_episode_record_count: int = M8_TOTAL_EPISODES
    durable_trajectory_blob_count: int = M8_TOTAL_EPISODES
    formal_output_count: int = 24
    attempt_count: int = 1
    retry_count: int = 0
    attempt_count_scope: str = "current_replacement_transaction_only"
    observed_phase_sequence: tuple[str, ...] = ("PREPARED", "TEST_BOUND", M8_REPORT_PHASE)
    phase_at_semantic_validation: str = M8_REPORT_PHASE
    required_publication_phase_sequence: tuple[str, ...] = ("ARTIFACTS_VALIDATED", "COMMITTED")
    accepted_result: str = M8_ACCEPTED_RESULT_RULE
    automatic_retry_after_test_bound: bool = False
    performance_outcome_can_trigger_retry: bool = False
    completed_workload_can_only_finalize_from_durable_journal_and_execution_evidence: bool = True
    schema_version: str = M8_TRANSACTION_EVIDENCE_SCHEMA_VERSION

    def __post_init__(self) -> None:
        expected_integers = {
            "durable_episode_record_count": (self.durable_episode_record_count, 60),
            "durable_trajectory_blob_count": (self.durable_trajectory_blob_count, 60),
            "formal_output_count": (self.formal_output_count, 24),
            "attempt_count": (self.attempt_count, 1),
            "retry_count": (self.retry_count, 0),
        }
        for field, (actual, expected) in expected_integers.items():
            if type(actual) is not int or actual != expected:
                raise FinalReportArtifactError(f"transaction {field} must be exactly {expected}")
        if self.observed_phase_sequence != ("PREPARED", "TEST_BOUND", M8_REPORT_PHASE):
            raise FinalReportArtifactError("transaction phase sequence differs before staging")
        if self.attempt_count_scope != "current_replacement_transaction_only":
            raise FinalReportArtifactError("transaction attempt count scope differs")
        if self.phase_at_semantic_validation != M8_REPORT_PHASE:
            raise FinalReportArtifactError(
                "staged report must be semantically validated after evaluation completion"
            )
        if self.required_publication_phase_sequence != ("ARTIFACTS_VALIDATED", "COMMITTED"):
            raise FinalReportArtifactError("publication phase sequence differs")
        if self.accepted_result != M8_ACCEPTED_RESULT_RULE:
            raise FinalReportArtifactError("transaction accepted-result policy differs")
        if self.automatic_retry_after_test_bound is not False:
            raise FinalReportArtifactError("transaction cannot permit automatic Test retry")
        if self.performance_outcome_can_trigger_retry is not False:
            raise FinalReportArtifactError("performance cannot trigger a Test retry")
        if (
            self.completed_workload_can_only_finalize_from_durable_journal_and_execution_evidence
            is not True
        ):
            raise FinalReportArtifactError(
                "completed workload must finalize from durable journal and execution evidence"
            )
        if self.schema_version != M8_TRANSACTION_EVIDENCE_SCHEMA_VERSION:
            raise FinalReportArtifactError("transaction evidence schema differs")

    def to_dict(self) -> dict[str, object]:
        value = asdict(self)
        value["observed_phase_sequence"] = list(self.observed_phase_sequence)
        value["required_publication_phase_sequence"] = list(
            self.required_publication_phase_sequence
        )
        return value


@dataclass(frozen=True, slots=True)
class PrivacyEvidence:
    """Explicit public-report privacy gates; validators also scan every string value."""

    gpu_uuid_redacted: bool = True
    absolute_project_paths_present: bool = False
    secrets_present: bool = False
    secret_finding_count: int = 0
    raw_test_geometry_embedded_in_report: bool = False
    schema_version: str = M8_PRIVACY_EVIDENCE_SCHEMA_VERSION

    def __post_init__(self) -> None:
        if self.gpu_uuid_redacted is not True:
            raise FinalReportArtifactError("privacy evidence must redact the GPU UUID")
        if self.absolute_project_paths_present is not False:
            raise FinalReportArtifactError("privacy evidence cannot contain project paths")
        if self.secrets_present is not False or self.secret_finding_count != 0:
            raise FinalReportArtifactError("privacy evidence cannot contain secrets")
        if self.raw_test_geometry_embedded_in_report is not False:
            raise FinalReportArtifactError("global JSON report cannot embed raw Test geometry")
        if self.schema_version != M8_PRIVACY_EVIDENCE_SCHEMA_VERSION:
            raise FinalReportArtifactError("privacy evidence schema differs")

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


def _validated_config(
    protocol_config: M8FinalEvaluationConfig,
    config_evidence: FinalConfigEvidence,
) -> None:
    if not isinstance(protocol_config, M8FinalEvaluationConfig):
        raise TypeError("protocol_config must be an M8FinalEvaluationConfig")
    if not isinstance(config_evidence, FinalConfigEvidence):
        raise TypeError("config_evidence must be FinalConfigEvidence")
    if _thaw_json(config_evidence.value) != protocol_config.to_dict():
        raise FinalReportArtifactError("final config parsed value differs from frozen protocol")


def _validated_dependency_lock(record: ArtifactRecord) -> None:
    if not isinstance(record, ArtifactRecord):
        raise TypeError("pixi_lock must be an ArtifactRecord")
    if (
        record.relative_path != M8_PIXI_LOCK_PATH
        or record.media_type not in {"application/yaml", "text/yaml"}
        or record.schema_version is not None
        or record.size_bytes < 1
    ):
        raise FinalReportArtifactError("Pixi lock artifact record differs")


def _validated_input_reports(
    config: M8FinalEvaluationConfig,
    reports: Mapping[str, ArtifactRecord],
) -> Mapping[str, ArtifactRecord]:
    if not isinstance(reports, Mapping) or set(reports) != set(_INPUT_REPORT_NAMES):
        raise FinalReportArtifactError("input reports must contain the six frozen reports")
    result: dict[str, ArtifactRecord] = {}
    for name in _INPUT_REPORT_NAMES:
        record = reports[name]
        if not isinstance(record, ArtifactRecord):
            raise TypeError("input report values must be ArtifactRecord values")
        if (
            record.relative_path != config.input_paths[name]
            or record.media_type != "application/json"
            or record.size_bytes < 1
        ):
            raise FinalReportArtifactError(f"input report {name!r} differs from frozen config")
        result[name] = record
    return MappingProxyType(result)


def _replacement_lineage_payload(
    config: M8FinalEvaluationConfig,
    failure_report_record: ArtifactRecord,
    failure_report_payload: Mapping[str, Any],
    transaction: TransactionEvidence,
) -> Mapping[str, object]:
    """Recompute the public replacement lineage from the frozen predecessor report."""

    from controller_learning.evaluation import replacement

    if not isinstance(failure_report_payload, Mapping):
        raise TypeError("replacement_failure_report must be a mapping")
    canonical = replacement.canonical_failure_report_bytes(failure_report_payload)
    if (
        failure_report_record.relative_path != config.replacement_failure_report_path
        or failure_report_record.sha256 != config.replacement_failure_report_sha256
        or failure_report_record.size_bytes != len(canonical)
        or hashlib.sha256(canonical).hexdigest() != failure_report_record.sha256
    ):
        raise FinalReportArtifactError(
            "replacement failure report payload differs from its frozen artifact record"
        )
    validated = replacement.validate_failure_report_bytes(
        canonical,
        expected_sha256=config.replacement_failure_report_sha256,
    )
    predecessor = validated["predecessor"]
    failure = validated["failure"]
    authorization = validated["authorization"]
    predecessor_transaction = validated["transaction"]
    if (
        predecessor["run_id"] != config.replacement_of_run_id
        or authorization["successor_run_id"] != config.run_id
        or authorization["max_replacement_attempts"] != config.replacement_attempt_limit
        or authorization["third_attempt_allowed"] != config.third_attempt_allowed
        or predecessor["transaction_phase"] != "TEST_BOUND"
        or predecessor["journal_record_count"] != 0
        or predecessor["execution_evidence"] is not None
        or predecessor["performance_observed"] is not False
        or failure["infrastructure_phase"] != "environment_create"
        or failure["workload"] is not None
        or predecessor_transaction["episode_blob_count"] != 0
        or predecessor_transaction["execution_seal_present"] is not False
        or predecessor_transaction["final_staged_present"] is not False
        or predecessor_transaction["publication_present"] is not False
    ):
        raise FinalReportArtifactError("replacement predecessor eligibility differs")
    if (
        transaction.attempt_count != 1
        or transaction.retry_count != 0
        or transaction.attempt_count_scope != "current_replacement_transaction_only"
    ):
        raise FinalReportArtifactError("replacement transaction counts differ")
    return MappingProxyType(
        {
            "authorization": {
                "replacement_attempt_limit": config.replacement_attempt_limit,
                "performance_outcome_can_trigger_replacement": authorization[
                    "performance_outcome_can_trigger_replacement"
                ],
                "third_attempt_allowed": config.third_attempt_allowed,
            },
            "failure_report": failure_report_record.to_dict(),
            "predecessor": {
                "durable_episode_record_count": predecessor["journal_record_count"],
                "environment_create_completed": False,
                "environment_reset_count": 0,
                "environment_step_count": 0,
                "execution_evidence_present": False,
                "expected_episode_count": predecessor["expected_episode_count"],
                "failure": dict(failure),
                "performance_observed": predecessor["performance_observed"],
                "plugin_controller_instance_count": 0,
                "run_id": predecessor["run_id"],
                "test_pool_load_completed": True,
                "transaction_phase": predecessor["transaction_phase"],
            },
            "schema_version": M8_REPLACEMENT_LINEAGE_SCHEMA_VERSION,
            "successor": {
                "automatic_retry_after_test_bound": config.automatic_retry_after_test_bound,
                "current_transaction_attempt_count": transaction.attempt_count,
                "current_transaction_attempt_count_scope": transaction.attempt_count_scope,
                "current_transaction_retry_count": transaction.retry_count,
                "replacement_attempt_ordinal": 1,
                "run_id": config.run_id,
            },
        }
    )


def _validated_identity(
    config: M8FinalEvaluationConfig,
    controller: str,
    identity: FrozenControllerIdentity,
) -> FrozenControllerIdentity:
    if not isinstance(identity, FrozenControllerIdentity):
        raise TypeError("Controller identity must be FrozenControllerIdentity")
    if (
        identity.controller != controller
        or identity.directory != config.controller_directories[controller]
    ):
        raise FinalReportArtifactError("Controller identity name or directory differs")
    if tuple(item.path for item in identity.files) != M8_CONTROLLER_FILE_MANIFEST[controller]:
        raise FinalReportArtifactError("Controller identity file order differs")
    for item in identity.files:
        _safe_relative_path(item.path, field="controller file path")
        _sha256(item.sha256, field="controller file sha256")
        _nonnegative_integer(item.size_bytes, field="controller file size", positive=True)
    canonical_files = json.dumps(
        [item.to_dict() for item in identity.files],
        allow_nan=False,
        ensure_ascii=True,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("ascii")
    if hashlib.sha256(canonical_files).hexdigest() != identity.aggregate_sha256:
        raise FinalReportArtifactError("Controller aggregate identity does not bind its files")
    if identity.aggregate_sha256 != config.controller_aggregate_sha256[controller]:
        raise FinalReportArtifactError("Controller aggregate identity differs from final config")
    if identity.config_sha256 != config.controller_config_sha256[controller]:
        raise FinalReportArtifactError("Controller config identity differs from final config")
    return identity


def _validated_identities(
    config: M8FinalEvaluationConfig,
    identities: Mapping[str, FrozenControllerIdentity],
) -> Mapping[str, FrozenControllerIdentity]:
    if not isinstance(identities, Mapping) or set(identities) != set(M8_CONTROLLER_ORDER):
        raise FinalReportArtifactError("Controller identities must cover pid, mpc, and ppo")
    return MappingProxyType(
        {name: _validated_identity(config, name, identities[name]) for name in M8_CONTROLLER_ORDER}
    )


def _validated_results(
    results: Mapping[str, FinalControllerResult],
) -> Mapping[str, FinalControllerResult]:
    if not isinstance(results, Mapping) or set(results) != set(M8_CONTROLLER_ORDER):
        raise FinalReportArtifactError("results must contain exactly pid, mpc, and ppo")
    ordered: dict[str, FinalControllerResult] = {}
    expected_tracks: tuple[int, ...] | None = None
    expected_seeds: tuple[tuple[int, int, int], ...] | None = None
    for name in M8_CONTROLLER_ORDER:
        value = results[name]
        if not isinstance(value, FinalControllerResult) or value.controller_name != name:
            raise FinalReportArtifactError("result mapping key and Controller name differ")
        tracks = tuple(row.track_id for row in value.episodes)
        seeds = tuple(
            (row.reset_seed, row.episode_seed, row.controller_seed) for row in value.episodes
        )
        if expected_tracks is None:
            expected_tracks = tracks
            expected_seeds = seeds
        elif tracks != expected_tracks or seeds != expected_seeds:
            raise FinalReportArtifactError("Controllers differ in Track order or public seeds")
        ordered[name] = value
    return MappingProxyType(ordered)


def _validated_pool_access(
    access: TestPoolAccessEvidence,
    audit: TestAccessAuditEvidence,
    config: M8FinalEvaluationConfig,
    track_ids: tuple[int, ...],
) -> None:
    if not isinstance(access, TestPoolAccessEvidence):
        raise TypeError("test_pool_access must be TestPoolAccessEvidence")
    if not isinstance(audit, TestAccessAuditEvidence):
        raise TypeError("test_access_audit must be TestAccessAuditEvidence")
    if access.track_ids != track_ids:
        raise FinalReportArtifactError("Test pool Track order differs from final results")
    if access.manifest_sha256 != config.test_manifest_sha256:
        raise FinalReportArtifactError("Test manifest hash differs from final config")
    if access.asset_file_sha256 != config.test_asset_sha256:
        raise FinalReportArtifactError("Test asset hash differs from final config")
    if access.loader_accessed_train or access.loader_accessed_validation:
        raise FinalReportArtifactError("Test loader evidence contains split leakage")


def _pool_access_payload(access: TestPoolAccessEvidence) -> dict[str, object]:
    return access.to_dict()


def _public_runtime_payload(
    runtime: RuntimeEvidence,
    pixi_lock: ArtifactRecord,
) -> dict[str, object]:
    """Add the measured Pixi workflow binding without inventing an unmeasured CLI version."""

    payload = runtime.to_dict()
    payload["pixi"] = {
        "environment_manager": "pixi",
        "lock_path": pixi_lock.relative_path,
        "lock_sha256": pixi_lock.sha256,
    }
    return payload


def _expected_artifact_metadata(path: str) -> tuple[str, str | None]:
    if path.endswith("m8_final_evaluation_report.json"):
        return "application/json", M8_FINAL_REPORT_SCHEMA_VERSION
    if path.endswith("/metrics.npz"):
        return "application/x-npz", FINAL_METRICS_SCHEMA_VERSION
    if path.endswith("/results.csv"):
        return "text/csv", FINAL_CONTROLLER_RESULTS_SCHEMA_VERSION
    if path.endswith("/summary.json"):
        return "application/json", FINAL_CONTROLLER_SUMMARY_SCHEMA_VERSION
    if path.endswith("/run_manifest.json"):
        return "application/json", M8_CONTROLLER_RUN_MANIFEST_SCHEMA_VERSION
    if path.endswith("/selected_replays/test_row_000_trajectory.json"):
        return "application/json", TRAJECTORY_SCHEMA_VERSION
    if path.endswith(".png"):
        return "image/png", None
    if path.endswith("m8_final_results.csv"):
        return "text/csv", FINAL_COMPARISON_SCHEMA_VERSION
    raise FinalReportArtifactError(f"unknown M8 output path {path!r}")


def _validated_record_for_path(record: ArtifactRecord, path: str) -> ArtifactRecord:
    if not isinstance(record, ArtifactRecord) or record.relative_path != path:
        raise FinalReportArtifactError(f"artifact record path differs for {path!r}")
    media_type, schema_version = _expected_artifact_metadata(path)
    if (
        record.media_type != media_type
        or record.schema_version != schema_version
        or record.size_bytes < 1
    ):
        raise FinalReportArtifactError(f"artifact metadata differs for {path!r}")
    return record


def _validated_controller_outputs(
    config: M8FinalEvaluationConfig,
    controller: str,
    artifacts: Mapping[str, ArtifactRecord],
) -> Mapping[str, ArtifactRecord]:
    expected_paths = dict(controller_output_paths(config, controller))
    expected_paths.pop("run_manifest")
    if not isinstance(artifacts, Mapping) or set(artifacts) != set(expected_paths):
        raise FinalReportArtifactError("Controller manifest must bind its other six outputs")
    return MappingProxyType(
        {
            name: _validated_record_for_path(artifacts[name], expected_paths[name])
            for name in sorted(expected_paths)
        }
    )


def _validated_global_outputs(
    config: M8FinalEvaluationConfig,
    artifacts: Mapping[str, ArtifactRecord],
) -> tuple[ArtifactRecord, ...]:
    expected = tuple(path for path in formal_output_paths(config) if path != config.report_path)
    if not isinstance(artifacts, Mapping) or set(artifacts) != set(expected):
        raise FinalReportArtifactError("global report must bind exactly its other 23 outputs")
    return tuple(_validated_record_for_path(artifacts[path], path) for path in expected)


def _validated_execution(
    execution: ExecutionEvidence,
    results: Mapping[str, FinalControllerResult],
) -> None:
    if not isinstance(execution, ExecutionEvidence):
        raise TypeError("execution must be ExecutionEvidence")
    if not results or any(name not in M8_CONTROLLER_ORDER for name in results):
        raise FinalReportArtifactError("execution comparison contains an unknown Controller")
    for name, result in results.items():
        if execution.environment_steps_by_controller[name] != result.summary.environment_steps:
            raise FinalReportArtifactError(
                f"execution step total for {name!r} differs from canonical results"
            )


def _timing_payload(value: object) -> dict[str, object]:
    return {
        "deadline_miss_count": value.deadline_miss_count,
        "deadline_miss_rate": value.deadline_miss_rate,
        "deadline_s": value.deadline_s,
        "max_s": value.max_s,
        "p50_s": value.p50_s,
        "p95_s": value.p95_s,
        "p99_s": value.p99_s,
        "sample_count": value.sample_count,
    }


def _episode_payload(row: object) -> dict[str, object]:
    metrics = row.metric_summary
    return {
        "benchmark_version": row.benchmark_version,
        "compute_timing": _timing_payload(row.compute_timing),
        "controller_import_time_s": row.controller_import_time_s,
        "controller_init_time_s": row.controller_init_time_s,
        "controller_name": row.controller_name,
        "controller_seed": row.controller_seed,
        "environment_steps": row.environment_steps,
        "episode_seed": row.episode_seed,
        "lap_time_s": row.lap_time_s,
        "metrics": {
            "acceleration_rate_rms_mps3": metrics.acceleration_rate_rms_mps3,
            "action_delta_count": metrics.action_delta_count,
            "lateral_error_abs_max_m": metrics.lateral_error_abs_max_m,
            "lateral_error_abs_p95_m": metrics.lateral_error_abs_p95_m,
            "lateral_error_rms_m": metrics.lateral_error_rms_m,
            "longitudinal_saturation_rate": metrics.longitudinal_saturation_rate,
            "mean_speed_mps": metrics.mean_speed_mps,
            "steering_rate_rms_rad_s": metrics.steering_rate_rms_rad_s,
            "steering_saturation_rate": metrics.steering_saturation_rate,
            "transition_count": metrics.transition_count,
        },
        "reset_seed": row.reset_seed,
        "row_index": row.row_index,
        "schema_version": row.schema_version,
        "success": row.success,
        "terminated": row.terminated,
        "termination_reason": row.termination_reason,
        "termination_reason_name": row.termination_reason_name,
        "total_reward": row.total_reward,
        "track_id": row.track_id,
        "truncated": row.truncated,
    }


def _controller_execution_subset(
    execution: ExecutionEvidence,
    controller: str,
) -> dict[str, object]:
    return {
        "controller_name": controller,
        "controller_wall_time_s": execution.controller_wall_time_s[controller],
        "environment_instance_count": execution.environment_instance_count,
        "environment_lifecycle": execution.environment_lifecycle,
        "environment_steps": execution.environment_steps_by_controller[controller],
        "fresh_controller_instance_count": M8_TEST_TRACK_COUNT,
        "initialization_over_soft_limit_rows": list(
            execution.initialization_over_soft_limit_rows[controller]
        ),
        "measured_environment_lifecycle": execution.measured_environment_lifecycle.to_dict(),
        "numerical_failure_count": execution.numerical_failure_count,
        "replay_captured_from_same_rollout": execution.replay_captured_from_same_rollout,
        "replay_row_index": execution.replay_row_index,
        "retry_count": execution.retry_count,
        "row_order": list(execution.row_order),
        "schema_version": execution.schema_version,
    }


def canonical_controller_run_manifest_json_bytes(
    controller_result: FinalControllerResult,
    *,
    source: SourceEvidence,
    protocol_config: M8FinalEvaluationConfig,
    config_evidence: FinalConfigEvidence,
    pixi_lock: ArtifactRecord,
    input_reports: Mapping[str, ArtifactRecord],
    controller_identity: FrozenControllerIdentity,
    test_pool_access: TestPoolAccessEvidence,
    test_access_audit: TestAccessAuditEvidence,
    runtime: RuntimeEvidence,
    memory: MemoryEvidence,
    execution: ExecutionEvidence,
    durable_execution_evidence: DurableExecutionEvidenceSeal,
    output_artifacts: Mapping[str, ArtifactRecord],
) -> bytes:
    """Build one Controller manifest bound to its six non-manifest outputs."""

    if not isinstance(controller_result, FinalControllerResult):
        raise TypeError("controller_result must be a FinalControllerResult")
    controller = controller_result.controller_name
    if controller not in M8_CONTROLLER_ORDER:
        raise FinalReportArtifactError("unknown final Controller")
    if not isinstance(source, SourceEvidence):
        raise TypeError("source must be SourceEvidence")
    _validated_config(protocol_config, config_evidence)
    _validated_dependency_lock(pixi_lock)
    inputs = _validated_input_reports(protocol_config, input_reports)
    identity = _validated_identity(protocol_config, controller, controller_identity)
    tracks = tuple(row.track_id for row in controller_result.episodes)
    _validated_pool_access(test_pool_access, test_access_audit, protocol_config, tracks)
    if not isinstance(runtime, RuntimeEvidence) or not isinstance(memory, MemoryEvidence):
        raise TypeError("runtime and memory must use their typed evidence classes")
    _validated_execution(execution, {controller: controller_result})
    if not isinstance(durable_execution_evidence, DurableExecutionEvidenceSeal):
        raise TypeError("durable_execution_evidence must be DurableExecutionEvidenceSeal")
    durable_execution_evidence.cross_check(
        test_access_audit=test_access_audit,
        execution=execution,
        memory=memory,
        runtime=runtime,
        test_pool_access=test_pool_access,
    )
    outputs = _validated_controller_outputs(
        protocol_config,
        controller,
        output_artifacts,
    )
    rows = controller_result.episodes
    payload: dict[str, object] = {
        "benchmark": {
            "backend": protocol_config.backend,
            "benchmark_version": protocol_config.benchmark_version,
            "level_id": protocol_config.level_id,
            "test_track_count": protocol_config.test_track_count,
        },
        "controller_identity": identity.to_dict(),
        "controller_name": controller,
        "dependency_lock": pixi_lock.to_dict(),
        "durable_execution_evidence": durable_execution_evidence.public_binding(),
        "execution": _controller_execution_subset(execution, controller),
        "final_config": config_evidence.to_dict(),
        "frozen_input_reports": {name: inputs[name].to_dict() for name in _INPUT_REPORT_NAMES},
        "memory": memory.to_dict(),
        "output_artifact_count": 6,
        "output_artifacts": {name: outputs[name].to_dict() for name in sorted(outputs)},
        "public_episode_seeds": [row.episode_seed for row in rows],
        "public_controller_seeds": [row.controller_seed for row in rows],
        "reset_seeds": [row.reset_seed for row in rows],
        "result_summary": dict(controller_summary_payload(controller_result)),
        "run_id": protocol_config.run_id,
        "runtime": _public_runtime_payload(runtime, pixi_lock),
        "schema_version": M8_CONTROLLER_RUN_MANIFEST_SCHEMA_VERSION,
        "source": source.to_dict(),
        "test_access_audit": test_access_audit.to_dict(),
        "test_pool_access": _pool_access_payload(test_pool_access),
        "test_track_ids": list(tracks),
    }
    return _canonical_json_bytes(payload)


def validate_controller_run_manifest_json_bytes(
    payload: bytes,
    controller_result: FinalControllerResult,
    **evidence: object,
) -> Mapping[str, Any]:
    """Reject any byte or evidence drift from the uniquely recomputed Controller manifest."""

    parsed = _strict_json_object(payload, artifact="Controller run manifest")
    expected = canonical_controller_run_manifest_json_bytes(
        controller_result,
        **evidence,  # type: ignore[arg-type]
    )
    if payload != expected:
        raise FinalReportArtifactError(
            "Controller run manifest differs from exact canonical recomputation"
        )
    return MappingProxyType(dict(parsed))


def canonical_m8_final_report_json_bytes(
    results: Mapping[str, FinalControllerResult],
    *,
    source: SourceEvidence,
    protocol_config: M8FinalEvaluationConfig,
    config_evidence: FinalConfigEvidence,
    pixi_lock: ArtifactRecord,
    input_reports: Mapping[str, ArtifactRecord],
    replacement_failure_report: Mapping[str, Any],
    controller_identities_before: Mapping[str, FrozenControllerIdentity],
    controller_identities_after: Mapping[str, FrozenControllerIdentity],
    test_pool_access: TestPoolAccessEvidence,
    test_access_audit: TestAccessAuditEvidence,
    runtime: RuntimeEvidence,
    memory: MemoryEvidence,
    execution: ExecutionEvidence,
    durable_execution_evidence: DurableExecutionEvidenceSeal,
    transaction: TransactionEvidence,
    privacy: PrivacyEvidence,
    output_artifacts: Mapping[str, ArtifactRecord],
) -> bytes:
    """Build the global protocol-valid report without a performance pass gate or score."""

    ordered = _validated_results(results)
    if not isinstance(source, SourceEvidence):
        raise TypeError("source must be SourceEvidence")
    _validated_config(protocol_config, config_evidence)
    _validated_dependency_lock(pixi_lock)
    inputs = _validated_input_reports(protocol_config, input_reports)
    identities_before = _validated_identities(protocol_config, controller_identities_before)
    identities_after = _validated_identities(protocol_config, controller_identities_after)
    for name in M8_CONTROLLER_ORDER:
        if identities_before[name].to_dict() != identities_after[name].to_dict():
            raise FinalReportArtifactError(f"Controller {name!r} changed during Test evaluation")
    tracks = tuple(row.track_id for row in ordered[M8_CONTROLLER_ORDER[0]].episodes)
    _validated_pool_access(test_pool_access, test_access_audit, protocol_config, tracks)
    if not isinstance(runtime, RuntimeEvidence) or not isinstance(memory, MemoryEvidence):
        raise TypeError("runtime and memory must use their typed evidence classes")
    _validated_execution(execution, ordered)
    if not isinstance(durable_execution_evidence, DurableExecutionEvidenceSeal):
        raise TypeError("durable_execution_evidence must be DurableExecutionEvidenceSeal")
    durable_execution_evidence.cross_check(
        test_access_audit=test_access_audit,
        execution=execution,
        memory=memory,
        runtime=runtime,
        test_pool_access=test_pool_access,
    )
    if not isinstance(transaction, TransactionEvidence):
        raise TypeError("transaction must be TransactionEvidence")
    if not isinstance(privacy, PrivacyEvidence):
        raise TypeError("privacy must be PrivacyEvidence")
    outputs = _validated_global_outputs(protocol_config, output_artifacts)
    replacement_lineage = _replacement_lineage_payload(
        protocol_config,
        inputs["m8_attempt_001_failure_report"],
        replacement_failure_report,
        transaction,
    )
    rank_order = rank_final_controller_results(ordered)
    ranks = {name: rank for rank, name in enumerate(rank_order, start=1)}
    controllers = []
    for name in M8_CONTROLLER_ORDER:
        result = ordered[name]
        paths = controller_output_paths(protocol_config, name)
        controllers.append(
            {
                "canonical_table_artifacts": {
                    "metrics": output_artifacts[paths["metrics"]].to_dict(),
                    "results": output_artifacts[paths["results"]].to_dict(),
                    "summary": output_artifacts[paths["summary"]].to_dict(),
                },
                "controller_name": name,
                "episode_rows": [_episode_payload(row) for row in result.episodes],
                "rank": ranks[name],
                "summary": dict(controller_summary_payload(result)),
            }
        )

    payload: dict[str, object] = {
        "benchmark": {
            "backend": protocol_config.backend,
            "benchmark_version": protocol_config.benchmark_version,
            "level_id": protocol_config.level_id,
            "test_track_count": protocol_config.test_track_count,
        },
        "controller_identities": {
            "after": {name: identities_after[name].to_dict() for name in M8_CONTROLLER_ORDER},
            "before": {name: identities_before[name].to_dict() for name in M8_CONTROLLER_ORDER},
            "unchanged": True,
        },
        "controller_order": list(M8_CONTROLLER_ORDER),
        "controllers": controllers,
        "dependency_lock": pixi_lock.to_dict(),
        "durable_execution_evidence": durable_execution_evidence.public_binding(),
        "execution": execution.to_dict(),
        "final_config": config_evidence.to_dict(),
        "frozen_input_reports": {name: inputs[name].to_dict() for name in _INPUT_REPORT_NAMES},
        "memory": memory.to_dict(),
        "output_artifact_count": 23,
        "output_artifacts": [record.to_dict() for record in outputs],
        "privacy": privacy.to_dict(),
        "protocol_result": {
            "combined_score_present": False,
            "performance_pass_gate": False,
            "ranking_rule": M8_RANKING_RULE,
            "status_basis": M8_REPORT_STATUS_BASIS,
            "success_rate_pass_gate": False,
        },
        "rank_order": list(rank_order),
        "replacement_lineage": _thaw_json(replacement_lineage),
        "run_id": protocol_config.run_id,
        "runtime": _public_runtime_payload(runtime, pixi_lock),
        "schema_version": M8_FINAL_REPORT_SCHEMA_VERSION,
        "source": source.to_dict(),
        "status": "passed",
        "test_access_audit": test_access_audit.to_dict(),
        "test_pool_access": _pool_access_payload(test_pool_access),
        "transaction": transaction.to_dict(),
    }
    return _canonical_json_bytes(payload)


def validate_m8_final_report_json_bytes(
    payload: bytes,
    results: Mapping[str, FinalControllerResult],
    **evidence: object,
) -> Mapping[str, Any]:
    """Reject noncanonical, incomplete, leaked, or drifted global M8 report bytes."""

    parsed = _strict_json_object(payload, artifact="M8 final report")
    expected = canonical_m8_final_report_json_bytes(
        results,
        **evidence,  # type: ignore[arg-type]
    )
    if payload != expected:
        raise FinalReportArtifactError("M8 final report differs from exact canonical recomputation")
    return MappingProxyType(dict(parsed))


def validate_m8_publication(
    outputs: Mapping[str, bytes],
    results: Mapping[str, FinalControllerResult],
    *,
    source: SourceEvidence,
    protocol_config: M8FinalEvaluationConfig,
    config_evidence: FinalConfigEvidence,
    pixi_lock: ArtifactRecord,
    input_reports: Mapping[str, ArtifactRecord],
    replacement_failure_report: Mapping[str, Any],
    controller_identities_before: Mapping[str, FrozenControllerIdentity],
    controller_identities_after: Mapping[str, FrozenControllerIdentity],
    test_pool_access: TestPoolAccessEvidence,
    test_access_audit: TestAccessAuditEvidence,
    runtime: RuntimeEvidence,
    memory: MemoryEvidence,
    execution: ExecutionEvidence,
    durable_execution_evidence: DurableExecutionEvidenceSeal,
    transaction: TransactionEvidence,
    privacy: PrivacyEvidence,
) -> str:
    """Semantically validate all 24 staged outputs and return one attestation digest.

    The returned lowercase SHA-256 binds the exact sorted output identities after every canonical
    table, metric archive, replay, plot, Controller manifest, and global report has been recomputed.
    It is suitable for ``M8AttemptTransaction.mark_artifacts_validated``.
    """

    import numpy as np

    from controller_learning.evaluation.final_metrics import canonical_final_metrics_bytes
    from controller_learning.evaluation.final_results import (
        validate_controller_results_csv_bytes,
        validate_controller_summary_json_bytes,
        validate_final_comparison_csv_bytes,
    )
    from controller_learning.evaluation.trajectory import load_trajectory_json_bytes
    from controller_learning.visualization.final_results import (
        render_controller_telemetry_png,
        render_final_comparison_png,
    )
    from controller_learning.visualization.replay import render_trajectory_overview_png

    expected_paths = formal_output_paths(protocol_config)
    if not isinstance(outputs, Mapping) or set(outputs) != set(expected_paths):
        raise FinalReportArtifactError("publication must contain exactly 24 formal outputs")
    if any(
        type(path) is not str or type(content) is not bytes for path, content in outputs.items()
    ):
        raise FinalReportArtifactError("publication paths must map to immutable bytes")
    ordered_results = _validated_results(results)
    records = {
        path: ArtifactRecord.from_bytes(
            path,
            outputs[path],
            *_expected_artifact_metadata(path),
        )
        for path in expected_paths
    }

    trajectories: dict[str, object] = {}
    for name in M8_CONTROLLER_ORDER:
        result = ordered_results[name]
        paths = controller_output_paths(protocol_config, name)
        validate_controller_results_csv_bytes(outputs[paths["results"]], result)
        validate_controller_summary_json_bytes(outputs[paths["summary"]], result)
        expected_metrics = canonical_final_metrics_bytes(result.metrics)
        if outputs[paths["metrics"]] != expected_metrics:
            raise FinalReportArtifactError(
                f"Controller {name!r} metrics differ from canonical transition samples"
            )

        trajectory = load_trajectory_json_bytes(outputs[paths["replay_trajectory"]])
        row = result.episodes[0]
        samples = result.metrics.episode(0)
        if (
            trajectory.step_count != row.environment_steps
            or int(trajectory.reset_info["track_id"]) != row.track_id
            or int(trajectory.reset_info["episode_seed"]) != row.episode_seed
            or int(trajectory.reset_info["controller_seed"]) != row.controller_seed
            or int(trajectory.final_info["termination_reason"]) != row.termination_reason
            or bool(trajectory.final_info["lap_completed"]) != row.success
            or not np.array_equal(trajectory.action, samples.requested_action)
        ):
            raise FinalReportArtifactError(
                f"Controller {name!r} row-zero replay differs from canonical result"
            )
        trajectories[name] = trajectory
        if outputs[paths["trajectory"]] != render_trajectory_overview_png(trajectory):
            raise FinalReportArtifactError(
                f"Controller {name!r} trajectory PNG is not an exact replay rendering"
            )
        expected_telemetry = render_controller_telemetry_png(
            controller_name=name,
            control_dt_s=protocol_config.control_dt_s,
            speed_mps=samples.speed_mps,
            lateral_error_m=samples.lateral_error_m,
            requested_action=samples.requested_action,
            steering_saturated=samples.steering_saturated,
            longitudinal_saturated=samples.longitudinal_saturated,
        )
        if outputs[paths["telemetry"]] != expected_telemetry:
            raise FinalReportArtifactError(
                f"Controller {name!r} telemetry PNG is not an exact metric rendering"
            )

        controller_outputs = {
            key: records[path] for key, path in paths.items() if key != "run_manifest"
        }
        validate_controller_run_manifest_json_bytes(
            outputs[paths["run_manifest"]],
            result,
            source=source,
            protocol_config=protocol_config,
            config_evidence=config_evidence,
            pixi_lock=pixi_lock,
            input_reports=input_reports,
            controller_identity=controller_identities_after[name],
            test_pool_access=test_pool_access,
            test_access_audit=test_access_audit,
            runtime=runtime,
            memory=memory,
            execution=execution,
            durable_execution_evidence=durable_execution_evidence,
            output_artifacts=controller_outputs,
        )

    validate_final_comparison_csv_bytes(
        outputs[protocol_config.comparison_csv_path],
        ordered_results,
    )
    first = trajectories[M8_CONTROLLER_ORDER[0]]
    for name in M8_CONTROLLER_ORDER[1:]:
        candidate = trajectories[name]
        if (
            not np.array_equal(candidate.centerline_m, first.centerline_m)
            or not np.array_equal(candidate.left_boundary_m, first.left_boundary_m)
            or not np.array_equal(candidate.right_boundary_m, first.right_boundary_m)
            or not np.array_equal(candidate.track_mask, first.track_mask)
        ):
            raise FinalReportArtifactError("row-zero replay Track geometry differs by Controller")
    expected_comparison = render_final_comparison_png(
        benchmark_version=protocol_config.benchmark_version,
        track_id=int(first.reset_info["track_id"]),
        centerline_m=first.centerline_m,
        left_boundary_m=first.left_boundary_m,
        right_boundary_m=first.right_boundary_m,
        track_mask=first.track_mask,
        trajectories_m={name: trajectories[name].position_m for name in M8_CONTROLLER_ORDER},
    )
    if outputs[protocol_config.comparison_png_path] != expected_comparison:
        raise FinalReportArtifactError(
            "comparison PNG is not an exact rendering of the three row-zero replays"
        )

    global_artifacts = {
        path: records[path] for path in expected_paths if path != protocol_config.report_path
    }
    validate_m8_final_report_json_bytes(
        outputs[protocol_config.report_path],
        ordered_results,
        source=source,
        protocol_config=protocol_config,
        config_evidence=config_evidence,
        pixi_lock=pixi_lock,
        input_reports=input_reports,
        replacement_failure_report=replacement_failure_report,
        controller_identities_before=controller_identities_before,
        controller_identities_after=controller_identities_after,
        test_pool_access=test_pool_access,
        test_access_audit=test_access_audit,
        runtime=runtime,
        memory=memory,
        execution=execution,
        durable_execution_evidence=durable_execution_evidence,
        transaction=transaction,
        privacy=privacy,
        output_artifacts=global_artifacts,
    )
    attestation = _canonical_json_bytes(
        {
            "output_artifacts": [records[path].to_dict() for path in expected_paths],
            "schema_version": "controller-learning.m8-semantic-publication-validation.v1",
            "validated_output_count": 24,
            "validator_id": ("controller_learning.evaluation.final_report.validate_m8_publication"),
        }
    )
    return hashlib.sha256(attestation).hexdigest()


__all__ = [
    "M8_CONFIG_EVIDENCE_SCHEMA_VERSION",
    "M8_CONTROLLER_RUN_MANIFEST_SCHEMA_VERSION",
    "M8_EXECUTION_EVIDENCE_SCHEMA_VERSION",
    "M8_FINAL_CONFIG_PATH",
    "M8_MEMORY_EVIDENCE_SCHEMA_VERSION",
    "M8_PIXI_LOCK_PATH",
    "M8_PRIVACY_EVIDENCE_SCHEMA_VERSION",
    "M8_REPLACEMENT_LINEAGE_SCHEMA_VERSION",
    "M8_RUNTIME_EVIDENCE_SCHEMA_VERSION",
    "M8_SOURCE_EVIDENCE_SCHEMA_VERSION",
    "M8_TEST_ACCESS_AUDIT_SCHEMA_VERSION",
    "M8_TRANSACTION_EVIDENCE_SCHEMA_VERSION",
    "ArtifactRecord",
    "DurableExecutionEvidenceSeal",
    "EnvironmentLifecycleEvidence",
    "ExecutionEvidence",
    "FinalConfigEvidence",
    "FinalReportArtifactError",
    "MemoryEvidence",
    "PrivacyEvidence",
    "RuntimeEvidence",
    "SourceEvidence",
    "TestAccessAuditEvidence",
    "TransactionEvidence",
    "canonical_controller_run_manifest_json_bytes",
    "canonical_m8_final_report_json_bytes",
    "validate_controller_run_manifest_json_bytes",
    "validate_durable_execution_evidence_mapping",
    "validate_m8_final_report_json_bytes",
    "validate_m8_publication",
]
