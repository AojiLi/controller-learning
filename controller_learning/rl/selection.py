"""Frozen M7 Validation checkpoint-selection configuration and report protocol."""

from __future__ import annotations

import hashlib
import json
import math
import re
import tomllib
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path, PurePosixPath
from typing import Any, Final, Literal

SELECTION_CONFIG_SCHEMA_VERSION: Final = 1
SELECTION_REPORT_SCHEMA_VERSION: Final = "controller-learning.m7-ppo-selection-report.v1"
FROZEN_CANDIDATE_UPDATES: Final = (10, 20, 30, 40, 50, 60, 70, 80)
FROZEN_VALIDATION_TRACK_COUNT: Final = 100
FROZEN_RANDOM_BASELINE_SEED: Final = 17
FROZEN_SUCCESS_RATE_MARGIN: Final = 0.10
FROZEN_RANKING: Final = "success_count_desc_mean_successful_lap_time_asc_update_asc"
FROZEN_WRAPPER_ORDER: Final = (
    "VecCarRacingEnv",
    "SelectionPublicObservationVecEnv",
    "JaxToTorchVecEnv",
)

_SHA256_PATTERN = re.compile(r"^[0-9a-f]{64}$")
_REASON_LABELS = {
    1: "success",
    2: "off_track",
    3: "invalid_action",
    4: "timeout",
}


class SelectionProtocolError(ValueError):
    """Raised when frozen selection configuration or report data is invalid."""


def torch_state_dict_sha256(state_dict: object, *, torch_module: Any) -> str:
    """Hash tensor names, dtypes, shapes, and exact CPU bytes in canonical key order."""

    if not isinstance(state_dict, Mapping) or not state_dict:
        raise TypeError("state_dict must be a non-empty mapping")
    digest = hashlib.sha256()
    for name in sorted(state_dict):
        value = state_dict[name]
        if type(name) is not str or not isinstance(value, torch_module.Tensor):
            raise TypeError("state_dict must map string names to Torch tensors")
        tensor = value.detach().to(device="cpu").contiguous()
        if not tensor.dtype.is_floating_point or not bool(
            torch_module.all(torch_module.isfinite(tensor))
        ):
            raise ValueError("candidate state tensors must be finite floating-point values")
        header = json.dumps(
            {"dtype": str(tensor.dtype), "name": name, "shape": list(tensor.shape)},
            separators=(",", ":"),
            sort_keys=True,
        ).encode("utf-8")
        digest.update(len(header).to_bytes(8, "big"))
        digest.update(header)
        payload = tensor.numpy().tobytes(order="C")
        digest.update(len(payload).to_bytes(8, "big"))
        digest.update(payload)
    return digest.hexdigest()


def _exact_keys(value: Mapping[str, Any], expected: set[str], *, field: str) -> None:
    if set(value) != expected or any(type(key) is not str for key in value):
        missing = sorted(expected - set(value))
        extra = sorted(set(value) - expected)
        raise SelectionProtocolError(f"{field} keys differ; missing={missing}, extra={extra}")


def _table(value: Mapping[str, Any], key: str) -> Mapping[str, Any]:
    candidate = value.get(key)
    if not isinstance(candidate, Mapping):
        raise SelectionProtocolError(f"{key} must be a TOML table")
    return candidate


def _plain_integer(value: object, *, field: str, minimum: int = 0) -> int:
    if type(value) is not int or value < minimum:
        raise SelectionProtocolError(f"{field} must be an integer >= {minimum}")
    return value


def _finite_number(value: object, *, field: str, minimum: float = 0.0) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise SelectionProtocolError(f"{field} must be a finite number >= {minimum}")
    result = float(value)
    if not math.isfinite(result) or result < minimum:
        raise SelectionProtocolError(f"{field} must be a finite number >= {minimum}")
    return result


def _safe_relative_path(value: object, *, field: str, suffix: str | None = None) -> str:
    if not isinstance(value, str) or not value or "\\" in value:
        raise SelectionProtocolError(f"{field} must be a normalized relative POSIX path")
    path = PurePosixPath(value)
    if (
        path.is_absolute()
        or path.as_posix() != value
        or any(part in {"", ".", ".."} for part in path.parts)
    ):
        raise SelectionProtocolError(f"{field} must be a normalized relative POSIX path")
    if suffix is not None and path.suffix != suffix:
        raise SelectionProtocolError(f"{field} must use the {suffix} suffix")
    return value


@dataclass(frozen=True, slots=True)
class PpoSelectionConfig:
    """Exact frozen inputs for the one M7 Validation selection pass."""

    benchmark_version: str
    level_id: int
    backend: str
    num_envs: int
    validation_track_count: int
    validation_reset_seed: int
    max_vector_steps: int
    run_id: str
    run_directory: str
    training_config: str
    checkpoint_directory: str
    candidate_updates: tuple[int, ...]
    random_baseline_seed: int
    random_distribution: str
    ranking: str
    minimum_success_rate_margin: float
    require_strict_success_count_improvement: bool
    report_path: str
    training_curve_path: str
    training_curve_width_px: int
    training_curve_height_px: int
    training_curve_dpi: int
    schema_version: int = SELECTION_CONFIG_SCHEMA_VERSION

    def __post_init__(self) -> None:
        if (
            type(self.schema_version) is not int
            or self.schema_version != SELECTION_CONFIG_SCHEMA_VERSION
        ):
            raise SelectionProtocolError("selection schema_version must be 1")
        exact_values = {
            "benchmark_version": (self.benchmark_version, "0.1"),
            "backend": (self.backend, "mjx_warp"),
            "run_id": (self.run_id, "m7-formal-v0-1-001"),
            "random_distribution": (self.random_distribution, "uniform_action_bounds"),
            "ranking": (self.ranking, FROZEN_RANKING),
        }
        for field, (actual, expected) in exact_values.items():
            if actual != expected:
                raise SelectionProtocolError(f"{field} must be exactly {expected!r}")
        if self.level_id != 1:
            raise SelectionProtocolError("level_id must be exactly 1")
        if self.num_envs != FROZEN_VALIDATION_TRACK_COUNT:
            raise SelectionProtocolError("num_envs must be exactly 100")
        if self.validation_track_count != FROZEN_VALIDATION_TRACK_COUNT:
            raise SelectionProtocolError("validation_track_count must be exactly 100")
        if (
            type(self.validation_reset_seed) is not int
            or not 0 <= self.validation_reset_seed < 2**32
        ):
            raise SelectionProtocolError("validation_reset_seed must fit in uint32")
        if self.max_vector_steps != 4000:
            raise SelectionProtocolError("max_vector_steps must be exactly 4000")
        if tuple(self.candidate_updates) != FROZEN_CANDIDATE_UPDATES:
            raise SelectionProtocolError(
                f"candidate_updates must be exactly {FROZEN_CANDIDATE_UPDATES}"
            )
        if self.random_baseline_seed != FROZEN_RANDOM_BASELINE_SEED:
            raise SelectionProtocolError("random baseline seed must be exactly 17")
        if not math.isclose(
            self.minimum_success_rate_margin,
            FROZEN_SUCCESS_RATE_MARGIN,
            rel_tol=0.0,
            abs_tol=0.0,
        ):
            raise SelectionProtocolError("minimum_success_rate_margin must be exactly 0.10")
        if self.require_strict_success_count_improvement is not True:
            raise SelectionProtocolError("strict success-count improvement must be enabled")
        expected_paths = {
            "run_directory": "runs/ppo/m7-formal-v0-1-001",
            "training_config": "configs/ppo.toml",
            "checkpoint_directory": "checkpoints",
            "report_path": "benchmarks/v0.1/m7_ppo_selection_report.json",
            "training_curve_path": "benchmarks/v0.1/m7_training_curve.png",
        }
        for field, expected in expected_paths.items():
            actual = _safe_relative_path(
                getattr(self, field),
                field=field,
                suffix=Path(expected).suffix or None,
            )
            if actual != expected:
                raise SelectionProtocolError(f"{field} must be exactly {expected!r}")
        if (
            self.training_curve_width_px,
            self.training_curve_height_px,
            self.training_curve_dpi,
        ) != (1200, 800, 100):
            raise SelectionProtocolError(
                "training curve geometry must be exactly 1200x800 at 100 DPI"
            )


def load_ppo_selection_config(path: str | Path) -> PpoSelectionConfig:
    """Load the exact frozen selection TOML without accepting unknown keys."""

    source = Path(path)
    if source.suffix != ".toml" or source.is_symlink() or not source.is_file():
        raise SelectionProtocolError("selection config must be a regular non-symlink TOML file")
    try:
        with source.open("rb") as file:
            data = tomllib.load(file)
    except (OSError, tomllib.TOMLDecodeError) as error:
        raise SelectionProtocolError("selection config is not valid TOML") from error
    _exact_keys(
        data,
        {
            "schema_version",
            "protocol",
            "training_run",
            "random_baseline",
            "selection",
            "artifacts",
        },
        field="selection config",
    )
    protocol = _table(data, "protocol")
    training = _table(data, "training_run")
    baseline = _table(data, "random_baseline")
    selection = _table(data, "selection")
    artifacts = _table(data, "artifacts")
    _exact_keys(
        protocol,
        {
            "benchmark_version",
            "level_id",
            "backend",
            "num_envs",
            "validation_track_count",
            "validation_reset_seed",
            "max_vector_steps",
        },
        field="protocol",
    )
    _exact_keys(
        training,
        {
            "run_id",
            "run_directory",
            "training_config",
            "checkpoint_directory",
            "candidate_updates",
        },
        field="training_run",
    )
    _exact_keys(baseline, {"seed", "distribution"}, field="random_baseline")
    _exact_keys(
        selection,
        {
            "ranking",
            "minimum_success_rate_margin",
            "require_strict_success_count_improvement",
        },
        field="selection",
    )
    _exact_keys(
        artifacts,
        {
            "report_path",
            "training_curve_path",
            "training_curve_width_px",
            "training_curve_height_px",
            "training_curve_dpi",
        },
        field="artifacts",
    )
    updates = training["candidate_updates"]
    if not isinstance(updates, list) or any(type(update) is not int for update in updates):
        raise SelectionProtocolError("candidate_updates must be an integer array")
    strict_improvement = selection["require_strict_success_count_improvement"]
    if type(strict_improvement) is not bool:
        raise SelectionProtocolError("require_strict_success_count_improvement must be boolean")
    return PpoSelectionConfig(
        schema_version=_plain_integer(data["schema_version"], field="schema_version"),
        benchmark_version=protocol["benchmark_version"],
        level_id=_plain_integer(protocol["level_id"], field="protocol.level_id"),
        backend=protocol["backend"],
        num_envs=_plain_integer(protocol["num_envs"], field="protocol.num_envs", minimum=1),
        validation_track_count=_plain_integer(
            protocol["validation_track_count"],
            field="protocol.validation_track_count",
            minimum=1,
        ),
        validation_reset_seed=_plain_integer(
            protocol["validation_reset_seed"],
            field="protocol.validation_reset_seed",
        ),
        max_vector_steps=_plain_integer(
            protocol["max_vector_steps"],
            field="protocol.max_vector_steps",
            minimum=1,
        ),
        run_id=training["run_id"],
        run_directory=training["run_directory"],
        training_config=training["training_config"],
        checkpoint_directory=training["checkpoint_directory"],
        candidate_updates=tuple(updates),
        random_baseline_seed=_plain_integer(baseline["seed"], field="random_baseline.seed"),
        random_distribution=baseline["distribution"],
        ranking=selection["ranking"],
        minimum_success_rate_margin=_finite_number(
            selection["minimum_success_rate_margin"],
            field="selection.minimum_success_rate_margin",
        ),
        require_strict_success_count_improvement=strict_improvement,
        report_path=artifacts["report_path"],
        training_curve_path=artifacts["training_curve_path"],
        training_curve_width_px=_plain_integer(
            artifacts["training_curve_width_px"],
            field="artifacts.training_curve_width_px",
            minimum=1,
        ),
        training_curve_height_px=_plain_integer(
            artifacts["training_curve_height_px"],
            field="artifacts.training_curve_height_px",
            minimum=1,
        ),
        training_curve_dpi=_plain_integer(
            artifacts["training_curve_dpi"],
            field="artifacts.training_curve_dpi",
            minimum=1,
        ),
    )


@dataclass(frozen=True, slots=True)
class SelectionTrackResult:
    """First terminal outcome for one fixed Validation Track."""

    track_index: int
    track_id: int
    termination_reason: int
    success: bool
    lap_time_s: float
    max_progress: float
    steps: int

    def __post_init__(self) -> None:
        _plain_integer(self.track_index, field="track_index")
        if type(self.track_id) is not int or not 0 <= self.track_id < 2**32:
            raise SelectionProtocolError("track_id must fit in uint32")
        if self.termination_reason not in _REASON_LABELS:
            raise SelectionProtocolError(
                "termination_reason must be one of the four public reasons"
            )
        if type(self.success) is not bool or self.success != (self.termination_reason == 1):
            raise SelectionProtocolError("success must match termination_reason SUCCESS")
        lap_time = _finite_number(self.lap_time_s, field="lap_time_s")
        if self.success != (lap_time > 0.0):
            raise SelectionProtocolError("lap_time_s must be positive exactly for a success")
        progress = _finite_number(self.max_progress, field="max_progress")
        if progress > 1.0:
            raise SelectionProtocolError("max_progress must be in [0, 1]")
        steps = _plain_integer(self.steps, field="steps", minimum=1)
        if self.success and not math.isclose(
            lap_time,
            steps * 0.05,
            rel_tol=0.0,
            abs_tol=2.0e-5,
        ):
            raise SelectionProtocolError("successful lap_time_s must match steps * control_dt_s")
        object.__setattr__(self, "lap_time_s", lap_time)
        object.__setattr__(self, "max_progress", progress)

    @property
    def termination_label(self) -> str:
        return _REASON_LABELS[self.termination_reason]

    def to_dict(self) -> dict[str, Any]:
        return {
            "lap_time_s": self.lap_time_s,
            "max_progress": self.max_progress,
            "steps": self.steps,
            "success": self.success,
            "termination_label": self.termination_label,
            "termination_reason": self.termination_reason,
            "track_id": self.track_id,
            "track_index": self.track_index,
        }

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> SelectionTrackResult:
        _exact_keys(
            value,
            {
                "lap_time_s",
                "max_progress",
                "steps",
                "success",
                "termination_label",
                "termination_reason",
                "track_id",
                "track_index",
            },
            field="selection row",
        )
        row = cls(
            track_index=value["track_index"],
            track_id=value["track_id"],
            termination_reason=value["termination_reason"],
            success=value["success"],
            lap_time_s=value["lap_time_s"],
            max_progress=value["max_progress"],
            steps=value["steps"],
        )
        if value["termination_label"] != row.termination_label:
            raise SelectionProtocolError("termination_label differs from termination_reason")
        return row


PolicyKind = Literal["candidate", "random_baseline"]


@dataclass(frozen=True, slots=True)
class PolicySelectionResult:
    """Raw fixed-order rows plus recomputable aggregate values for one policy."""

    policy_kind: PolicyKind
    policy_id: str
    update_index: int | None
    parameter_sha256_before: str | None
    parameter_sha256_after: str | None
    rows: tuple[SelectionTrackResult, ...]

    def __post_init__(self) -> None:
        if self.policy_kind not in {"candidate", "random_baseline"}:
            raise SelectionProtocolError("policy_kind is invalid")
        if not isinstance(self.policy_id, str) or not self.policy_id:
            raise SelectionProtocolError("policy_id must be non-empty")
        if self.policy_kind == "candidate":
            if self.update_index not in FROZEN_CANDIDATE_UPDATES:
                raise SelectionProtocolError("candidate update_index is not frozen")
            for field in ("parameter_sha256_before", "parameter_sha256_after"):
                digest = getattr(self, field)
                if not isinstance(digest, str) or _SHA256_PATTERN.fullmatch(digest) is None:
                    raise SelectionProtocolError(f"{field} must be a SHA-256 digest")
            if self.parameter_sha256_before != self.parameter_sha256_after:
                raise SelectionProtocolError("candidate parameters changed during selection")
        elif any(
            value is not None
            for value in (
                self.update_index,
                self.parameter_sha256_before,
                self.parameter_sha256_after,
            )
        ):
            raise SelectionProtocolError("random baseline cannot claim checkpoint parameters")
        if len(self.rows) != FROZEN_VALIDATION_TRACK_COUNT:
            raise SelectionProtocolError("every selection result must contain exactly 100 rows")
        if tuple(row.track_index for row in self.rows) != tuple(range(len(self.rows))):
            raise SelectionProtocolError("selection rows must preserve fixed Track index order")
        if len({row.track_id for row in self.rows}) != len(self.rows):
            raise SelectionProtocolError("selection Track IDs must be unique")

    @property
    def success_count(self) -> int:
        return sum(row.success for row in self.rows)

    @property
    def success_rate(self) -> float:
        return self.success_count / len(self.rows)

    @property
    def mean_successful_lap_time_s(self) -> float | None:
        successful = tuple(row.lap_time_s for row in self.rows if row.success)
        return sum(successful) / len(successful) if successful else None

    def to_dict(self) -> dict[str, Any]:
        return {
            "mean_successful_lap_time_s": self.mean_successful_lap_time_s,
            "parameter_sha256_after": self.parameter_sha256_after,
            "parameter_sha256_before": self.parameter_sha256_before,
            "parameter_unchanged": (
                self.parameter_sha256_before == self.parameter_sha256_after
                if self.policy_kind == "candidate"
                else None
            ),
            "policy_id": self.policy_id,
            "policy_kind": self.policy_kind,
            "rows": [row.to_dict() for row in self.rows],
            "success_count": self.success_count,
            "success_rate": self.success_rate,
            "track_count": len(self.rows),
            "update_index": self.update_index,
        }


def rank_candidate_results(
    candidates: Sequence[PolicySelectionResult],
) -> tuple[PolicySelectionResult, ...]:
    """Apply the frozen success/time/update ordering, treating no-success time as worst."""

    values = tuple(candidates)
    if tuple(
        sorted(result.update_index for result in values if result.update_index is not None)
    ) != (FROZEN_CANDIDATE_UPDATES):
        raise SelectionProtocolError("candidate results must cover the exact eight frozen updates")
    if any(result.policy_kind != "candidate" for result in values):
        raise SelectionProtocolError("ranking accepts only candidate results")
    return tuple(
        sorted(
            values,
            key=lambda result: (
                -result.success_count,
                (
                    result.mean_successful_lap_time_s
                    if result.mean_successful_lap_time_s is not None
                    else math.inf
                ),
                result.update_index,
            ),
        )
    )


def selection_gate_values(
    selected: PolicySelectionResult,
    random_baseline: PolicySelectionResult,
    *,
    config: PpoSelectionConfig,
) -> dict[str, Any]:
    """Return the two predeclared learning gates and their recomputable observations."""

    if selected.policy_kind != "candidate" or random_baseline.policy_kind != "random_baseline":
        raise SelectionProtocolError("gate inputs must be selected candidate and random baseline")
    # Both policies run the same 100 Tracks; count-space subtraction avoids a one-ULP false
    # failure for mathematically exact margins such as 0.15 - 0.05 == 0.10.
    margin = (selected.success_count - random_baseline.success_count) / len(selected.rows)
    rate_passed = margin >= config.minimum_success_rate_margin
    count_passed = selected.success_count > random_baseline.success_count
    return {
        "checks": [
            {
                "expected": config.minimum_success_rate_margin,
                "id": "success_rate_margin",
                "observed": margin,
                "passed": rate_passed,
            },
            {
                "expected": "selected_success_count > random_success_count",
                "id": "strict_success_count_improvement",
                "observed": {
                    "random_success_count": random_baseline.success_count,
                    "selected_success_count": selected.success_count,
                },
                "passed": count_passed,
            },
        ],
        "minimum_success_rate_margin": config.minimum_success_rate_margin,
        "passed": rate_passed and count_passed,
        "random_success_count": random_baseline.success_count,
        "random_success_rate": random_baseline.success_rate,
        "selected_success_count": selected.success_count,
        "selected_success_rate": selected.success_rate,
        "success_rate_margin": margin,
    }


def _evaluation_from_report(value: object) -> PolicySelectionResult:
    if not isinstance(value, Mapping):
        raise SelectionProtocolError("evaluation must be an object")
    _exact_keys(
        value,
        {
            "mean_successful_lap_time_s",
            "parameter_sha256_after",
            "parameter_sha256_before",
            "parameter_unchanged",
            "policy_id",
            "policy_kind",
            "rows",
            "success_count",
            "success_rate",
            "track_count",
            "update_index",
        },
        field="evaluation",
    )
    rows = value["rows"]
    if not isinstance(rows, list):
        raise SelectionProtocolError("evaluation rows must be an array")
    parsed_rows: list[SelectionTrackResult] = []
    for row in rows:
        if not isinstance(row, Mapping):
            raise SelectionProtocolError("evaluation row must be an object")
        parsed_rows.append(SelectionTrackResult.from_mapping(row))
    result = PolicySelectionResult(
        policy_kind=value["policy_kind"],
        policy_id=value["policy_id"],
        update_index=value["update_index"],
        parameter_sha256_before=value["parameter_sha256_before"],
        parameter_sha256_after=value["parameter_sha256_after"],
        rows=tuple(parsed_rows),
    )
    expected = result.to_dict()
    for field in (
        "mean_successful_lap_time_s",
        "parameter_unchanged",
        "success_count",
        "success_rate",
        "track_count",
    ):
        if value[field] != expected[field]:
            raise SelectionProtocolError(f"evaluation aggregate {field} differs from raw rows")
    return result


def _artifact_record(value: object, *, field: str, expected_path: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise SelectionProtocolError(f"{field} must be an artifact record")
    _exact_keys(
        value,
        {"relative_path", "schema_version", "sha256", "size_bytes"},
        field=field,
    )
    if (
        value["relative_path"] != expected_path
        or type(value["schema_version"]) is not int
        or value["schema_version"] != 1
    ):
        raise SelectionProtocolError(f"{field} identity differs")
    if not isinstance(value["sha256"], str) or _SHA256_PATTERN.fullmatch(value["sha256"]) is None:
        raise SelectionProtocolError(f"{field} SHA-256 is invalid")
    if type(value["size_bytes"]) is not int or value["size_bytes"] < 1:
        raise SelectionProtocolError(f"{field} size is invalid")
    return value


def _source_snapshot(value: object, *, field: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise SelectionProtocolError(f"{field} must be an object")
    _exact_keys(value, {"revision", "worktree_clean"}, field=field)
    revision = value["revision"]
    if not isinstance(revision, str) or re.fullmatch(r"[0-9a-f]{40}", revision) is None:
        raise SelectionProtocolError(f"{field} revision is invalid")
    if value["worktree_clean"] is not True:
        raise SelectionProtocolError(f"{field} must prove a clean worktree")
    return value


def _validate_access_evidence(value: object, *, loaded: bool, field: str) -> None:
    if not isinstance(value, Mapping):
        raise SelectionProtocolError(f"{field} must be an object")
    _exact_keys(
        value,
        {
            "audit_hook_installed_before_preflight",
            "denied_event_count",
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
        field=field,
    )
    if (
        value["audit_hook_installed_before_preflight"] is not True
        or value["denied_event_count"] != 0
        or value["pre_validation_open_event_count"] != 0
        or value["test_opened"] is not False
        or value["track_cache_opened"] is not False
        or value["train_opened"] is not False
        or value["validation_loaded"] is not loaded
        or value["validation_reads_enabled"] is not loaded
    ):
        raise SelectionProtocolError(f"{field} contains a forbidden access claim")
    counts = value["open_event_counts"]
    sequence = value["open_event_sequence"]
    if not isinstance(counts, Mapping) or not isinstance(sequence, list):
        raise SelectionProtocolError(f"{field} open evidence is invalid")
    expected_categories = ["official_validation_asset", "official_validation_manifest"]
    if loaded:
        if (
            value["opened_path_categories"] != expected_categories
            or value["opened_splits"] != ["validation"]
            or set(counts) != set(expected_categories)
            or any(type(count) is not int or count < 1 for count in counts.values())
            or sum(counts.values()) != len(sequence)
        ):
            raise SelectionProtocolError(f"{field} does not prove Validation-only opens")
        observed_counts = {category: 0 for category in expected_categories}
        for event in sequence:
            if not isinstance(event, Mapping):
                raise SelectionProtocolError(f"{field} open event must be an object")
            _exact_keys(event, {"category", "flags", "mode"}, field=f"{field} open event")
            category = event["category"]
            if category not in observed_counts:
                raise SelectionProtocolError(f"{field} open category is forbidden")
            mode = event["mode"]
            flags = event["flags"]
            if (isinstance(mode, str) and any(token in mode for token in "wax+")) or (
                type(flags) is int and bool(flags & 0o3103)
            ):
                raise SelectionProtocolError(f"{field} contains a writable Validation open")
            observed_counts[category] += 1
        if observed_counts != dict(counts):
            raise SelectionProtocolError(f"{field} event counts do not recompute")
    elif any(
        (
            dict(counts),
            sequence,
            value["opened_path_categories"],
            value["opened_splits"],
        )
    ):
        raise SelectionProtocolError(f"{field} opened Validation before preflight completed")


def _validate_runtime(value: object, *, expected_policy_ids: list[str]) -> None:
    if not isinstance(value, Mapping):
        raise SelectionProtocolError("runtime must be an object")
    _exact_keys(
        value,
        {
            "cuda_device_order",
            "cuda_visible_devices_configured",
            "evaluation_timings",
            "kernel",
            "machine",
            "packages",
            "platform",
            "python",
            "selected_gpu",
            "torch_cuda_runtime",
            "torch_device",
            "xla_python_client_preallocate",
        },
        field="runtime",
    )
    if (
        value["cuda_device_order"] != "PCI_BUS_ID"
        or type(value["cuda_visible_devices_configured"]) is not bool
        or value["torch_device"] != "cuda:0"
        or value["xla_python_client_preallocate"] != "false"
        or value["platform"] != "Linux"
        or value["machine"] != "x86_64"
        or not isinstance(value["python"], str)
        or re.fullmatch(r"3\.11(?:\.[0-9]+)?", value["python"]) is None
        or not isinstance(value["torch_cuda_runtime"], str)
        or not value["torch_cuda_runtime"]
    ):
        raise SelectionProtocolError("runtime allocator/device evidence differs")
    for field in ("kernel",):
        if not isinstance(value[field], str) or not value[field]:
            raise SelectionProtocolError(f"runtime.{field} must be non-empty")
    packages = value["packages"]
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
        or any(not isinstance(package, str) or not package for package in packages.values())
    ):
        raise SelectionProtocolError("runtime package inventory differs")
    gpu = value["selected_gpu"]
    if not isinstance(gpu, Mapping):
        raise SelectionProtocolError("runtime selected_gpu must be an object")
    _exact_keys(
        gpu,
        {"driver_version", "index", "memory_total_mib", "name", "uuid"},
        field="selected_gpu",
    )
    if (
        gpu["index"] != 0
        or not isinstance(gpu["name"], str)
        or not gpu["name"]
        or not isinstance(gpu["driver_version"], str)
        or not gpu["driver_version"]
        or not isinstance(gpu["uuid"], str)
        or not gpu["uuid"]
        or _finite_number(gpu["memory_total_mib"], field="memory_total_mib") <= 0.0
    ):
        raise SelectionProtocolError("runtime selected_gpu evidence is invalid")
    timings = value["evaluation_timings"]
    if not isinstance(timings, list) or len(timings) != len(expected_policy_ids):
        raise SelectionProtocolError("runtime evaluation timings are incomplete")
    observed_ids: list[str] = []
    for timing in timings:
        if not isinstance(timing, Mapping):
            raise SelectionProtocolError("evaluation timing must be an object")
        _exact_keys(timing, {"elapsed_seconds", "policy_id"}, field="evaluation timing")
        _finite_number(timing["elapsed_seconds"], field="evaluation elapsed_seconds")
        observed_ids.append(timing["policy_id"])
    if observed_ids != expected_policy_ids:
        raise SelectionProtocolError("evaluation timing policy order differs")


def _validate_memory(value: object) -> None:
    if not isinstance(value, Mapping):
        raise SelectionProtocolError("memory must be an object")
    _exact_keys(
        value,
        {
            "peak_jax_bytes_in_use",
            "peak_sampled_process_vram_mib",
            "peak_torch_allocated_bytes",
            "sample_count",
            "samples",
        },
        field="memory",
    )
    samples = value["samples"]
    expected_phases = ["after_stack_build", "after_evaluations", "after_environment_close"]
    if not isinstance(samples, list) or value["sample_count"] != 3 or len(samples) != 3:
        raise SelectionProtocolError("memory must contain exactly three samples")
    byte_fields = {
        "jax_bytes_in_use",
        "jax_peak_bytes_in_use",
        "torch_allocated_bytes",
        "torch_max_allocated_bytes",
        "torch_reserved_bytes",
    }
    phases: list[str] = []
    for sample in samples:
        if not isinstance(sample, Mapping):
            raise SelectionProtocolError("memory sample must be an object")
        _exact_keys(
            sample,
            byte_fields | {"phase", "process_vram_error", "process_vram_mib", "synchronized"},
            field="memory sample",
        )
        if sample["synchronized"] is not True or any(
            type(sample[field]) is not int or sample[field] < 0 for field in byte_fields
        ):
            raise SelectionProtocolError("memory sample values are invalid")
        if (
            _finite_number(sample["process_vram_mib"], field="process_vram_mib") <= 0.0
            or sample["process_vram_error"] is not None
        ):
            raise SelectionProtocolError("process VRAM sampling must succeed with a positive value")
        phases.append(sample["phase"])
    if phases != expected_phases:
        raise SelectionProtocolError("memory sample phases differ")
    if value["peak_jax_bytes_in_use"] != max(sample["jax_peak_bytes_in_use"] for sample in samples):
        raise SelectionProtocolError("JAX memory peak does not recompute")
    if value["peak_torch_allocated_bytes"] != max(
        sample["torch_max_allocated_bytes"] for sample in samples
    ):
        raise SelectionProtocolError("Torch memory peak does not recompute")
    if value["peak_sampled_process_vram_mib"] != max(
        sample["process_vram_mib"] for sample in samples
    ):
        raise SelectionProtocolError("process VRAM peak does not recompute")


def selection_report_findings(
    report: object,
    *,
    config: PpoSelectionConfig,
) -> tuple[str, ...]:
    """Return structural/recomputation findings for one strict M7 selection report."""

    findings: list[str] = []
    try:
        if not isinstance(config, PpoSelectionConfig):
            raise SelectionProtocolError("config must be PpoSelectionConfig")
        if not isinstance(report, Mapping):
            raise SelectionProtocolError("report must be an object")
        _exact_keys(
            report,
            {
                "artifacts",
                "asset_access",
                "configuration",
                "evaluations",
                "gates",
                "memory",
                "post_selection",
                "protocol",
                "runtime",
                "schema_version",
                "selection",
                "source",
                "status",
                "training_run",
                "validation_assets",
            },
            field="report",
        )
        if report["schema_version"] != SELECTION_REPORT_SCHEMA_VERSION:
            raise SelectionProtocolError("report schema_version differs")
        expected_configuration = asdict(config)
        expected_configuration["candidate_updates"] = list(config.candidate_updates)
        if report["configuration"] != expected_configuration:
            raise SelectionProtocolError("report configuration differs from frozen TOML")

        evaluations = report["evaluations"]
        if not isinstance(evaluations, Mapping):
            raise SelectionProtocolError("evaluations must be an object")
        _exact_keys(evaluations, {"candidates", "random_baseline"}, field="evaluations")
        candidate_values = evaluations["candidates"]
        if not isinstance(candidate_values, list):
            raise SelectionProtocolError("candidate evaluations must be an array")
        candidates = tuple(_evaluation_from_report(value) for value in candidate_values)
        if tuple(result.update_index for result in candidates) != FROZEN_CANDIDATE_UPDATES:
            raise SelectionProtocolError("candidate evaluations must preserve frozen update order")
        expected_candidate_ids = [
            f"checkpoint_update_{update:08d}" for update in FROZEN_CANDIDATE_UPDATES
        ]
        if [result.policy_id for result in candidates] != expected_candidate_ids:
            raise SelectionProtocolError("candidate policy IDs differ from frozen updates")
        baseline = _evaluation_from_report(evaluations["random_baseline"])
        if baseline.policy_kind != "random_baseline" or baseline.policy_id != "random_seed_17":
            raise SelectionProtocolError("random baseline identity differs")
        common_track_ids = tuple(row.track_id for row in candidates[0].rows)
        if any(
            tuple(row.track_id for row in result.rows) != common_track_ids for result in candidates
        ):
            raise SelectionProtocolError("candidate Track order differs")
        if tuple(row.track_id for row in baseline.rows) != common_track_ids:
            raise SelectionProtocolError("random baseline Track order differs from candidates")
        ranking = rank_candidate_results(candidates)
        selection = report["selection"]
        if not isinstance(selection, Mapping):
            raise SelectionProtocolError("selection must be an object")
        _exact_keys(
            selection,
            {
                "candidate_updates_in_rank_order",
                "mean_successful_lap_time_no_success_policy",
                "ranking",
                "selected_mean_successful_lap_time_s",
                "selected_success_count",
                "selected_success_rate",
                "selected_update",
            },
            field="selection",
        )
        selected = ranking[0]
        expected_selection = {
            "candidate_updates_in_rank_order": [result.update_index for result in ranking],
            "mean_successful_lap_time_no_success_policy": "positive_infinity_worst",
            "ranking": config.ranking,
            "selected_mean_successful_lap_time_s": selected.mean_successful_lap_time_s,
            "selected_success_count": selected.success_count,
            "selected_success_rate": selected.success_rate,
            "selected_update": selected.update_index,
        }
        if dict(selection) != expected_selection:
            raise SelectionProtocolError("selection result differs from recomputed ranking")
        expected_gates = selection_gate_values(selected, baseline, config=config)
        if report["gates"] != expected_gates:
            raise SelectionProtocolError("gate results differ from raw evaluation rows")
        expected_status = "passed" if expected_gates["passed"] else "gate_failed"
        if report["status"] != expected_status:
            raise SelectionProtocolError("report status differs from gate result")

        protocol = report["protocol"]
        if not isinstance(protocol, Mapping):
            raise SelectionProtocolError("protocol must be an object")
        expected_protocol = {
            "autoreset_mode": "NEXT_STEP",
            "backend": config.backend,
            "benchmark_version": config.benchmark_version,
            "candidate_count": len(FROZEN_CANDIDATE_UPDATES),
            "candidate_updates": list(FROZEN_CANDIDATE_UPDATES),
            "control_dt_s": 0.05,
            "deterministic_candidate_actions": True,
            "first_terminal_event_only": True,
            "level_id": config.level_id,
            "max_vector_steps": config.max_vector_steps,
            "no_gradient_updates": True,
            "num_envs": config.num_envs,
            "one_long_lived_environment": True,
            "random_baseline_seed": FROZEN_RANDOM_BASELINE_SEED,
            "reset_options_track_indices": "numpy.arange(100,dtype=int32)",
            "reward_wrapper_used": False,
            "same_reset_seed_and_track_order_for_every_policy": True,
            "test_accessed": False,
            "train_assets_accessed": False,
            "validation_reset_seed": config.validation_reset_seed,
            "validation_track_count": FROZEN_VALIDATION_TRACK_COUNT,
            "wrapper_order": list(FROZEN_WRAPPER_ORDER),
        }
        if dict(protocol) != expected_protocol:
            raise SelectionProtocolError("report protocol differs from the frozen selector")

        validation_assets = report["validation_assets"]
        if not isinstance(validation_assets, Mapping):
            raise SelectionProtocolError("validation_assets must be an object")
        _exact_keys(
            validation_assets,
            {
                "asset_file",
                "asset_file_sha256",
                "benchmark_version",
                "capacity",
                "first_track_id",
                "generator_version",
                "geometry_hashes_sha256",
                "last_track_id",
                "level_id",
                "loaded_splits",
                "loader_accessed_test",
                "loader_accessed_train",
                "manifest_asset_sha256",
                "manifest_file",
                "manifest_sha256",
                "schema_version",
                "split",
                "track_count",
                "track_ids_sha256",
            },
            field="validation_assets",
        )
        if (
            validation_assets["schema_version"]
            != "controller-learning.m7-validation-pool-access.v1"
            or validation_assets["loaded_splits"] != ["validation"]
            or validation_assets["benchmark_version"] != config.benchmark_version
            or validation_assets["level_id"] != config.level_id
            or validation_assets["split"] != "validation"
            or validation_assets["manifest_file"] != "validation.json"
            or validation_assets["asset_file"] != "validation.npz"
            or validation_assets["track_count"] != FROZEN_VALIDATION_TRACK_COUNT
            or validation_assets["loader_accessed_train"] is not False
            or validation_assets["loader_accessed_test"] is not False
            or validation_assets["manifest_asset_sha256"] != validation_assets["asset_file_sha256"]
        ):
            raise SelectionProtocolError("Validation asset evidence differs")
        for digest_field in (
            "asset_file_sha256",
            "geometry_hashes_sha256",
            "manifest_asset_sha256",
            "manifest_sha256",
            "track_ids_sha256",
        ):
            digest = validation_assets[digest_field]
            if not isinstance(digest, str) or _SHA256_PATTERN.fullmatch(digest) is None:
                raise SelectionProtocolError(f"validation_assets.{digest_field} is invalid")
        capacity = validation_assets["capacity"]
        if not isinstance(capacity, Mapping):
            raise SelectionProtocolError("Validation capacity must be an object")
        _exact_keys(capacity, {"max_checkpoints", "max_track_points"}, field="capacity")
        if any(type(value) is not int or value < 1 for value in capacity.values()):
            raise SelectionProtocolError("Validation capacity values are invalid")
        if (
            validation_assets["first_track_id"] != common_track_ids[0]
            or validation_assets["last_track_id"] != common_track_ids[-1]
        ):
            raise SelectionProtocolError("Validation boundary Track IDs differ from raw rows")
        track_digest = hashlib.sha256()
        for track_id in common_track_ids:
            track_digest.update(str(track_id).encode("ascii"))
            track_digest.update(b"\n")
        if track_digest.hexdigest() != validation_assets["track_ids_sha256"]:
            raise SelectionProtocolError("Validation Track-ID digest differs from raw rows")

        _validate_access_evidence(report["asset_access"], loaded=True, field="asset_access")

        artifacts = report["artifacts"]
        if not isinstance(artifacts, Mapping):
            raise SelectionProtocolError("artifacts must be an object")
        _exact_keys(
            artifacts,
            {
                "latest_checkpoint_pointer",
                "metrics_csv",
                "pixi_lock",
                "selection_config",
                "training_config",
                "training_curve",
                "training_manifest",
                "training_run_config",
                "validation_asset",
                "validation_manifest",
            },
            field="artifacts",
        )
        artifact_paths = {
            "latest_checkpoint_pointer": "checkpoints/latest.json",
            "metrics_csv": "metrics.csv",
            "pixi_lock": "pixi.lock",
            "selection_config": "configs/ppo_selection.toml",
            "training_config": config.training_config,
            "training_curve": config.training_curve_path,
            "training_manifest": f"{config.run_directory}/manifest.json",
            "training_run_config": f"{config.run_directory}/config.toml",
            "validation_asset": "controller_learning/assets/tracks/v0.1/validation.npz",
            "validation_manifest": "controller_learning/assets/tracks/v0.1/validation.json",
        }
        artifact_records = {
            name: _artifact_record(artifacts[name], field=name, expected_path=path)
            for name, path in artifact_paths.items()
        }
        if (
            artifact_records["validation_asset"]["sha256"] != validation_assets["asset_file_sha256"]
            or artifact_records["validation_manifest"]["sha256"]
            != validation_assets["manifest_sha256"]
        ):
            raise SelectionProtocolError("Validation artifact records differ from loader evidence")

        source = report["source"]
        if not isinstance(source, Mapping):
            raise SelectionProtocolError("source must be an object")
        _exact_keys(
            source,
            {"input_stability", "post_input_check", "post_output_worktree", "preflight"},
            field="source",
        )
        preflight_source = _source_snapshot(source["preflight"], field="source.preflight")
        post_input_source = _source_snapshot(
            source["post_input_check"], field="source.post_input_check"
        )
        if dict(preflight_source) != dict(post_input_source):
            raise SelectionProtocolError("source changed during evaluation")
        stability = source["input_stability"]
        if not isinstance(stability, Mapping):
            raise SelectionProtocolError("source.input_stability must be an object")
        _exact_keys(
            stability,
            {"all_inputs_unchanged", "expected_post_sha256", "post_evaluation_sha256"},
            field="source.input_stability",
        )
        expected_hashes = stability["expected_post_sha256"]
        observed_hashes = stability["post_evaluation_sha256"]
        expected_hash_keys = {
            "latest_checkpoint_pointer",
            "pixi_lock",
            "selection_config",
            "training_config",
            "training_manifest",
            "training_metrics",
            "training_run_config",
            "validation_asset",
            "validation_manifest",
            *(f"checkpoint_update_{update:08d}" for update in FROZEN_CANDIDATE_UPDATES),
        }
        if (
            stability["all_inputs_unchanged"] is not True
            or not isinstance(expected_hashes, Mapping)
            or not isinstance(observed_hashes, Mapping)
            or set(expected_hashes) != expected_hash_keys
            or dict(expected_hashes) != dict(observed_hashes)
            or any(
                not isinstance(digest, str) or _SHA256_PATTERN.fullmatch(digest) is None
                for digest in expected_hashes.values()
            )
        ):
            raise SelectionProtocolError("formal input stability evidence differs")
        output_source = source["post_output_worktree"]
        if not isinstance(output_source, Mapping):
            raise SelectionProtocolError("source.post_output_worktree must be an object")
        _exact_keys(
            output_source,
            {
                "allowed_generated_output_paths",
                "observed_changed_paths",
                "only_allowed_generated_outputs",
                "revision",
                "unexpected_changed_paths",
            },
            field="source.post_output_worktree",
        )
        allowed_outputs = sorted([config.report_path, config.training_curve_path])
        observed_outputs = output_source["observed_changed_paths"]
        if (
            output_source["allowed_generated_output_paths"] != allowed_outputs
            or not isinstance(observed_outputs, list)
            or not set(observed_outputs) <= set(allowed_outputs)
            or output_source["only_allowed_generated_outputs"] is not True
            or output_source["revision"] != preflight_source["revision"]
            or output_source["unexpected_changed_paths"] != []
        ):
            raise SelectionProtocolError("generated-output worktree evidence differs")
        artifact_hash_links = {
            "latest_checkpoint_pointer": "latest_checkpoint_pointer",
            "pixi_lock": "pixi_lock",
            "selection_config": "selection_config",
            "training_config": "training_config",
            "training_manifest": "training_manifest",
            "metrics_csv": "training_metrics",
            "training_run_config": "training_run_config",
            "validation_asset": "validation_asset",
            "validation_manifest": "validation_manifest",
        }
        if any(
            artifact_records[artifact_name]["sha256"] != expected_hashes[input_name]
            for artifact_name, input_name in artifact_hash_links.items()
        ):
            raise SelectionProtocolError("artifact records differ from stable input hashes")

        training_run = report["training_run"]
        if not isinstance(training_run, Mapping):
            raise SelectionProtocolError("training_run must be an object")
        _exact_keys(
            training_run,
            {
                "candidate_checkpoints",
                "identity",
                "manifest_sha256",
                "pre_validation_access",
                "run_directory",
            },
            field="training_run",
        )
        if (
            training_run["run_directory"] != config.run_directory
            or training_run["manifest_sha256"] != artifact_records["training_manifest"]["sha256"]
        ):
            raise SelectionProtocolError("training-run artifact identity differs")
        _validate_access_evidence(
            training_run["pre_validation_access"],
            loaded=False,
            field="training_run.pre_validation_access",
        )
        identity = training_run["identity"]
        if not isinstance(identity, Mapping):
            raise SelectionProtocolError("training_run.identity must be an object")
        _exact_keys(
            identity,
            {
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
            },
            field="training_run.identity",
        )
        if (
            identity["run_id"] != config.run_id
            or identity["benchmark_version"] != config.benchmark_version
            or identity["source_revision"] != preflight_source["revision"]
            or identity["configuration_sha256"] != artifact_records["training_config"]["sha256"]
            or identity["lock_sha256"] != artifact_records["pixi_lock"]["sha256"]
            or type(identity["feature_schema_version"]) is not int
            or identity["feature_schema_version"] != 1
            or identity["reward_schema_version"] != "controller-learning.m7-public-reward.v1"
            or type(identity["schema_version"]) is not int
            or identity["schema_version"] != 1
        ):
            raise SelectionProtocolError("training-run identity differs from source/config/lock")
        for digest_field in (
            "configuration_sha256",
            "lock_sha256",
            "train_cache_sha256",
            "train_manifest_sha256",
        ):
            digest = identity[digest_field]
            if not isinstance(digest, str) or _SHA256_PATTERN.fullmatch(digest) is None:
                raise SelectionProtocolError(f"training identity {digest_field} is invalid")
        for seed_field in ("environment_seed", "minibatch_seed", "policy_seed"):
            seed = identity[seed_field]
            if type(seed) is not int or not 0 <= seed < 2**32:
                raise SelectionProtocolError(f"training identity {seed_field} is invalid")
        if (
            len(
                {identity[field] for field in ("environment_seed", "minibatch_seed", "policy_seed")}
            )
            != 3
        ):
            raise SelectionProtocolError("training identity seeds must be distinct")

        checkpoints = training_run["candidate_checkpoints"]
        if not isinstance(checkpoints, list) or len(checkpoints) != len(FROZEN_CANDIDATE_UPDATES):
            raise SelectionProtocolError("candidate checkpoint evidence is incomplete")
        prior_vector_steps = 0
        prior_valid_transitions = 0
        for checkpoint, candidate, update in zip(
            checkpoints, candidates, FROZEN_CANDIDATE_UPDATES, strict=True
        ):
            if not isinstance(checkpoint, Mapping):
                raise SelectionProtocolError("candidate checkpoint evidence must be an object")
            _exact_keys(
                checkpoint,
                {
                    "checkpoint",
                    "inference_policy",
                    "parameter_sha256",
                    "update_index",
                    "valid_transitions",
                    "vector_steps",
                },
                field="candidate checkpoint evidence",
            )
            checkpoint_record = _artifact_record(
                checkpoint["checkpoint"],
                field=f"checkpoint update {update}",
                expected_path=f"{config.checkpoint_directory}/update_{update:08d}.pt",
            )
            inference_policy = checkpoint["inference_policy"]
            if not isinstance(inference_policy, Mapping):
                raise SelectionProtocolError("candidate inference policy must be an object")
            _exact_keys(
                inference_policy,
                {"schema_version", "sha256", "size_bytes"},
                field="candidate inference policy",
            )
            if (
                type(inference_policy["schema_version"]) is not int
                or inference_policy["schema_version"] != 1
                or not isinstance(inference_policy["sha256"], str)
                or _SHA256_PATTERN.fullmatch(inference_policy["sha256"]) is None
                or type(inference_policy["size_bytes"]) is not int
                or inference_policy["size_bytes"] < 1
            ):
                raise SelectionProtocolError("candidate inference policy identity is invalid")
            if (
                checkpoint["update_index"] != update
                or checkpoint["parameter_sha256"] != candidate.parameter_sha256_before
                or checkpoint_record["sha256"] != expected_hashes[f"checkpoint_update_{update:08d}"]
                or type(checkpoint["vector_steps"]) is not int
                or checkpoint["vector_steps"] <= prior_vector_steps
                or type(checkpoint["valid_transitions"]) is not int
                or checkpoint["valid_transitions"] < prior_valid_transitions
            ):
                raise SelectionProtocolError("candidate checkpoint binding differs")
            prior_vector_steps = checkpoint["vector_steps"]
            prior_valid_transitions = checkpoint["valid_transitions"]

        _validate_runtime(
            report["runtime"], expected_policy_ids=[*expected_candidate_ids, "random_seed_17"]
        )
        _validate_memory(report["memory"])
        post_selection = report["post_selection"]
        if not isinstance(post_selection, Mapping):
            raise SelectionProtocolError("post_selection must be an object")
        _exact_keys(
            post_selection,
            {"controller_evaluation_status", "export_status"},
            field="post_selection",
        )
        if dict(post_selection) != {
            "controller_evaluation_status": "not_run",
            "export_status": "not_run",
        }:
            raise SelectionProtocolError("post-selection work must remain not_run")
    except (SelectionProtocolError, KeyError, TypeError, ValueError) as error:
        findings.append(str(error))
    return tuple(findings)


def validate_selection_report(report: object, *, config: PpoSelectionConfig) -> None:
    """Raise when a report cannot be recomputed under the frozen protocol."""

    findings = selection_report_findings(report, config=config)
    if findings:
        raise SelectionProtocolError("; ".join(findings))


__all__ = [
    "FROZEN_CANDIDATE_UPDATES",
    "FROZEN_RANDOM_BASELINE_SEED",
    "FROZEN_RANKING",
    "FROZEN_SUCCESS_RATE_MARGIN",
    "FROZEN_VALIDATION_TRACK_COUNT",
    "FROZEN_WRAPPER_ORDER",
    "SELECTION_CONFIG_SCHEMA_VERSION",
    "SELECTION_REPORT_SCHEMA_VERSION",
    "PolicySelectionResult",
    "PpoSelectionConfig",
    "SelectionProtocolError",
    "SelectionTrackResult",
    "load_ppo_selection_config",
    "rank_candidate_results",
    "selection_gate_values",
    "selection_report_findings",
    "torch_state_dict_sha256",
    "validate_selection_report",
]
