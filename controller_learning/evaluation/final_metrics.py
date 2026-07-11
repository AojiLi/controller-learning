"""Public-trajectory metrics and their canonical final-evaluation artifact.

The metric path intentionally reconstructs every sample from values visible at the public
Controller boundary.  It does not inspect simulator state.  The resulting NPZ is uncompressed,
byte deterministic, non-pickled, and strict enough to bind one Controller to the fixed 20-row
v0.1 evaluation order.
"""

from __future__ import annotations

import hashlib
import io
import math
import os
import re
import stat
import tempfile
import zipfile
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Final

import numpy as np
from numpy.typing import NDArray

from controller_learning.control.geometry import CenterlineReference
from controller_learning.evaluation.trajectory import RecordedControllerEpisode

FINAL_METRICS_SCHEMA_VERSION: Final = 1
FINAL_METRICS_BENCHMARK_VERSION: Final = "0.1"
FINAL_METRICS_EPISODE_COUNT: Final = 20
FINAL_METRICS_CONTROL_DT_S: Final = 0.05
FINAL_METRICS_PROJECTION_BACKWARD_SEGMENTS: Final = 4
FINAL_METRICS_PROJECTION_FORWARD_SEGMENTS: Final = 12
MAX_FINAL_METRICS_NPZ_BYTES: Final = 64 * 1024 * 1024

_ZIP_TIMESTAMP: Final = (1980, 1, 1, 0, 0, 0)
_ZIP_EXTERNAL_ATTR: Final = 0o100600 << 16
_SHA256_PATTERN: Final = re.compile(r"[0-9a-f]{64}")
_CONTROLLER_NAME_PATTERN: Final = re.compile(r"[a-z][a-z0-9_-]{0,31}")
_UINT32_MAX: Final = int(np.iinfo(np.uint32).max)
_METADATA_NAMES: Final = (
    "schema_version",
    "benchmark_version",
    "controller_name",
    "track_id",
    "reset_seed",
)
_SAMPLE_NAMES: Final = (
    "episode_offsets",
    "compute_time_s",
    "speed_mps",
    "lateral_error_m",
    "requested_action",
    "steering_saturated",
    "longitudinal_saturated",
)
_ARRAY_NAMES: Final = _METADATA_NAMES + _SAMPLE_NAMES
_ARRAY_DTYPES: Final[Mapping[str, np.dtype[Any]]] = {
    "schema_version": np.dtype("<u4"),
    "benchmark_version": np.dtype("|S16"),
    "controller_name": np.dtype("|S32"),
    "track_id": np.dtype("<u4"),
    "reset_seed": np.dtype("<u4"),
    "episode_offsets": np.dtype("<i8"),
    "compute_time_s": np.dtype("<f8"),
    "speed_mps": np.dtype("<f8"),
    "lateral_error_m": np.dtype("<f8"),
    "requested_action": np.dtype("<f4"),
    "steering_saturated": np.dtype("|b1"),
    "longitudinal_saturated": np.dtype("|b1"),
}


class FinalMetricsArtifactError(ValueError):
    """A final metrics artifact violates its public schema or persistence contract."""


def _finite_positive(value: object, *, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float, np.integer, np.floating)):
        raise TypeError(f"{name} must be a real number")
    result = float(value)
    if not math.isfinite(result) or result <= 0.0:
        raise ValueError(f"{name} must be finite and positive")
    return result


def _uint32(value: object, *, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, (int, np.integer)):
        raise TypeError(f"{name} must be an integer")
    result = int(value)
    if not 0 <= result <= _UINT32_MAX:
        raise ValueError(f"{name} must fit in uint32")
    return result


def _readonly_array(
    value: object,
    *,
    dtype: np.dtype[Any],
    shape: tuple[int, ...],
    name: str,
    finite: bool = True,
) -> NDArray[Any]:
    try:
        with np.errstate(over="ignore", invalid="ignore"):
            source = np.asarray(value, dtype=dtype)
    except (OverflowError, TypeError, ValueError) as error:
        raise ValueError(f"{name} must be convertible to {dtype.str}") from error
    if source.shape != shape:
        raise ValueError(f"{name} must have shape {shape}, got {source.shape}")
    if finite and not np.isfinite(source).all():
        raise ValueError(f"{name} must contain only finite values")
    result = np.array(source, dtype=dtype, order="C", copy=True)
    result.setflags(write=False)
    return result


def _readonly_bool(value: object, *, shape: tuple[int, ...], name: str) -> NDArray[np.bool_]:
    source = np.asarray(value)
    if source.dtype != np.dtype(np.bool_):
        raise TypeError(f"{name} must use the boolean dtype")
    return _readonly_array(
        source,
        dtype=_ARRAY_DTYPES["steering_saturated"],
        shape=shape,
        name=name,
        finite=False,
    )


@dataclass(frozen=True, slots=True)
class MetricActionLimits:
    """Physical bounds used to classify raw requested action saturation."""

    max_steering_angle_rad: float
    max_acceleration_mps2: float
    max_deceleration_mps2: float

    def __post_init__(self) -> None:
        for name in (
            "max_steering_angle_rad",
            "max_acceleration_mps2",
            "max_deceleration_mps2",
        ):
            object.__setattr__(self, name, _finite_positive(getattr(self, name), name=name))


@dataclass(frozen=True, slots=True)
class EpisodeMetricSamples:
    """Immutable transition samples for one public Controller episode."""

    track_id: int
    reset_seed: int
    compute_time_s: NDArray[np.float64]
    speed_mps: NDArray[np.float64]
    lateral_error_m: NDArray[np.float64]
    requested_action: NDArray[np.float32]
    steering_saturated: NDArray[np.bool_]
    longitudinal_saturated: NDArray[np.bool_]

    def __post_init__(self) -> None:
        track_id = _uint32(self.track_id, name="track_id")
        reset_seed = _uint32(self.reset_seed, name="reset_seed")
        compute_source = np.asarray(self.compute_time_s)
        if compute_source.ndim != 1:
            raise ValueError("compute_time_s must be one-dimensional")
        step_count = int(compute_source.size)
        if step_count < 1:
            raise ValueError("an episode metric sample must contain at least one transition")
        arrays = {
            "compute_time_s": _readonly_array(
                self.compute_time_s,
                dtype=_ARRAY_DTYPES["compute_time_s"],
                shape=(step_count,),
                name="compute_time_s",
            ),
            "speed_mps": _readonly_array(
                self.speed_mps,
                dtype=_ARRAY_DTYPES["speed_mps"],
                shape=(step_count,),
                name="speed_mps",
            ),
            "lateral_error_m": _readonly_array(
                self.lateral_error_m,
                dtype=_ARRAY_DTYPES["lateral_error_m"],
                shape=(step_count,),
                name="lateral_error_m",
            ),
            "requested_action": _readonly_array(
                self.requested_action,
                dtype=_ARRAY_DTYPES["requested_action"],
                shape=(step_count, 2),
                name="requested_action",
            ),
            "steering_saturated": _readonly_bool(
                self.steering_saturated,
                shape=(step_count,),
                name="steering_saturated",
            ),
            "longitudinal_saturated": _readonly_bool(
                self.longitudinal_saturated,
                shape=(step_count,),
                name="longitudinal_saturated",
            ),
        }
        if np.any(arrays["compute_time_s"] < 0.0):
            raise ValueError("compute_time_s must be non-negative")
        if np.any(arrays["speed_mps"] < 0.0):
            raise ValueError("speed_mps must be non-negative")
        object.__setattr__(self, "track_id", track_id)
        object.__setattr__(self, "reset_seed", reset_seed)
        for name, array in arrays.items():
            object.__setattr__(self, name, array)

    @property
    def transition_count(self) -> int:
        """Number of post-step transition samples."""

        return int(self.compute_time_s.size)

    @property
    def steering_rate_rad_s(self) -> NDArray[np.float64]:
        """Requested steering deltas divided by 0.05 s, excluding the first action."""

        result = np.diff(self.requested_action[:, 0].astype(np.float64)) / (
            FINAL_METRICS_CONTROL_DT_S
        )
        result.setflags(write=False)
        return result

    @property
    def acceleration_rate_mps3(self) -> NDArray[np.float64]:
        """Requested acceleration deltas divided by 0.05 s, excluding the first action."""

        result = np.diff(self.requested_action[:, 1].astype(np.float64)) / (
            FINAL_METRICS_CONTROL_DT_S
        )
        result.setflags(write=False)
        return result


@dataclass(frozen=True, slots=True)
class EpisodeMetricSummary:
    """Exact scalar aggregates for one episode's public metric samples."""

    track_id: int
    reset_seed: int
    transition_count: int
    action_delta_count: int
    mean_speed_mps: float
    lateral_error_rms_m: float
    lateral_error_abs_p95_m: float
    lateral_error_abs_max_m: float
    steering_saturation_rate: float
    longitudinal_saturation_rate: float
    steering_rate_rms_rad_s: float
    acceleration_rate_rms_mps3: float


@dataclass(frozen=True, slots=True)
class AggregateMetricSummary:
    """Transition-sample aggregates across complete episodes without boundary deltas."""

    episode_count: int
    transition_count: int
    action_delta_count: int
    mean_speed_mps: float
    lateral_error_rms_m: float
    lateral_error_abs_p95_m: float
    lateral_error_abs_max_m: float
    steering_saturation_rate: float
    longitudinal_saturation_rate: float
    steering_rate_rms_rad_s: float
    acceleration_rate_rms_mps3: float


def compute_episode_metric_samples(
    episode: RecordedControllerEpisode,
    *,
    reset_seed: int,
    action_limits: MetricActionLimits,
) -> EpisodeMetricSamples:
    """Reconstruct one metric sample solely from the recorded public episode values."""

    if not isinstance(episode, RecordedControllerEpisode):
        raise TypeError("episode must be a RecordedControllerEpisode")
    if not isinstance(action_limits, MetricActionLimits):
        raise TypeError("action_limits must be MetricActionLimits")
    reset_seed = _uint32(reset_seed, name="reset_seed")
    trajectory = episode.trajectory
    step_count = trajectory.step_count

    compute_time_s = np.asarray(episode.result.compute_times_s, dtype=np.float64)
    if compute_time_s.shape != (step_count,):
        raise ValueError("Runner compute times must contain exactly one value per transition")
    velocity = np.asarray(trajectory.velocity_body_mps[1:], dtype=np.float64)
    speed_mps = np.linalg.norm(velocity, axis=1)

    reference = CenterlineReference.from_observation(trajectory.observation(0))
    lateral_error = np.empty(step_count, dtype=np.float64)
    hint_segment: int | None = None
    for index, position in enumerate(np.asarray(trajectory.position_m[1:], dtype=np.float64)):
        projection = reference.project(
            position,
            hint_segment=hint_segment,
            backward_segments=FINAL_METRICS_PROJECTION_BACKWARD_SEGMENTS,
            forward_segments=FINAL_METRICS_PROJECTION_FORWARD_SEGMENTS,
        )
        lateral_error[index] = projection.lateral_error_m
        hint_segment = projection.segment_index

    requested_action = np.asarray(trajectory.action, dtype=np.float32)
    steering = requested_action[:, 0].astype(np.float64)
    longitudinal = requested_action[:, 1].astype(np.float64)
    steering_saturated = (steering < -action_limits.max_steering_angle_rad) | (
        steering > action_limits.max_steering_angle_rad
    )
    longitudinal_saturated = (longitudinal < -action_limits.max_deceleration_mps2) | (
        longitudinal > action_limits.max_acceleration_mps2
    )
    return EpisodeMetricSamples(
        track_id=int(trajectory.reset_info["track_id"]),
        reset_seed=reset_seed,
        compute_time_s=compute_time_s,
        speed_mps=speed_mps,
        lateral_error_m=lateral_error,
        requested_action=requested_action,
        steering_saturated=steering_saturated,
        longitudinal_saturated=longitudinal_saturated,
    )


def _rms(values: NDArray[np.float64]) -> float:
    if values.size == 0:
        return 0.0
    return float(np.sqrt(np.mean(np.square(values), dtype=np.float64)))


def _summary_values(samples: Sequence[EpisodeMetricSamples]) -> dict[str, int | float]:
    if not samples:
        raise ValueError("metric samples cannot be empty")
    speed = np.concatenate(tuple(sample.speed_mps for sample in samples))
    lateral = np.concatenate(tuple(sample.lateral_error_m for sample in samples))
    steering_saturated = np.concatenate(tuple(sample.steering_saturated for sample in samples))
    longitudinal_saturated = np.concatenate(
        tuple(sample.longitudinal_saturated for sample in samples)
    )
    steering_rates = np.concatenate(tuple(sample.steering_rate_rad_s for sample in samples))
    acceleration_rates = np.concatenate(tuple(sample.acceleration_rate_mps3 for sample in samples))
    absolute_lateral = np.abs(lateral)
    return {
        "transition_count": int(speed.size),
        "action_delta_count": int(steering_rates.size),
        "mean_speed_mps": float(np.mean(speed, dtype=np.float64)),
        "lateral_error_rms_m": _rms(lateral),
        "lateral_error_abs_p95_m": float(np.percentile(absolute_lateral, 95.0, method="linear")),
        "lateral_error_abs_max_m": float(np.max(absolute_lateral)),
        "steering_saturation_rate": float(np.mean(steering_saturated, dtype=np.float64)),
        "longitudinal_saturation_rate": float(np.mean(longitudinal_saturated, dtype=np.float64)),
        "steering_rate_rms_rad_s": _rms(steering_rates),
        "acceleration_rate_rms_mps3": _rms(acceleration_rates),
    }


def summarize_episode_metrics(samples: EpisodeMetricSamples) -> EpisodeMetricSummary:
    """Compute exact aggregates for one episode."""

    if not isinstance(samples, EpisodeMetricSamples):
        raise TypeError("samples must be EpisodeMetricSamples")
    return EpisodeMetricSummary(
        track_id=samples.track_id,
        reset_seed=samples.reset_seed,
        **_summary_values((samples,)),
    )


def summarize_metric_episodes(
    samples: Sequence[EpisodeMetricSamples],
) -> AggregateMetricSummary:
    """Aggregate concatenated transitions while excluding cross-episode action deltas."""

    if isinstance(samples, (str, bytes)) or not isinstance(samples, Sequence):
        raise TypeError("samples must be a sequence of EpisodeMetricSamples")
    episodes = tuple(samples)
    if not episodes or not all(isinstance(item, EpisodeMetricSamples) for item in episodes):
        raise ValueError("samples must contain at least one EpisodeMetricSamples value")
    return AggregateMetricSummary(episode_count=len(episodes), **_summary_values(episodes))


@dataclass(frozen=True, slots=True)
class FinalMetricsData:
    """Strict in-memory representation of one Controller's fixed 20-row metrics."""

    controller_name: str
    track_id: NDArray[np.uint32]
    reset_seed: NDArray[np.uint32]
    episode_offsets: NDArray[np.int64]
    compute_time_s: NDArray[np.float64]
    speed_mps: NDArray[np.float64]
    lateral_error_m: NDArray[np.float64]
    requested_action: NDArray[np.float32]
    steering_saturated: NDArray[np.bool_]
    longitudinal_saturated: NDArray[np.bool_]
    benchmark_version: str = FINAL_METRICS_BENCHMARK_VERSION
    schema_version: int = FINAL_METRICS_SCHEMA_VERSION

    def __post_init__(self) -> None:
        if self.schema_version != FINAL_METRICS_SCHEMA_VERSION:
            raise ValueError(
                f"schema_version must be {FINAL_METRICS_SCHEMA_VERSION}, got {self.schema_version}"
            )
        if self.benchmark_version != FINAL_METRICS_BENCHMARK_VERSION:
            raise ValueError(f"benchmark_version must be {FINAL_METRICS_BENCHMARK_VERSION!r}")
        if (
            not isinstance(self.controller_name, str)
            or _CONTROLLER_NAME_PATTERN.fullmatch(self.controller_name) is None
        ):
            raise ValueError("controller_name must be a lowercase canonical identifier")

        track_id = _readonly_array(
            self.track_id,
            dtype=_ARRAY_DTYPES["track_id"],
            shape=(FINAL_METRICS_EPISODE_COUNT,),
            name="track_id",
        )
        if np.unique(track_id).size != FINAL_METRICS_EPISODE_COUNT:
            raise ValueError("track_id must contain 20 unique ordered row identities")
        reset_seed = _readonly_array(
            self.reset_seed,
            dtype=_ARRAY_DTYPES["reset_seed"],
            shape=(FINAL_METRICS_EPISODE_COUNT,),
            name="reset_seed",
        )
        expected_seeds = np.arange(FINAL_METRICS_EPISODE_COUNT, dtype=np.uint32)
        if not np.array_equal(reset_seed, expected_seeds):
            raise ValueError("reset_seed must equal the fixed row-index sequence 0..19")

        offsets = _readonly_array(
            self.episode_offsets,
            dtype=_ARRAY_DTYPES["episode_offsets"],
            shape=(FINAL_METRICS_EPISODE_COUNT + 1,),
            name="episode_offsets",
        )
        if int(offsets[0]) != 0 or np.any(np.diff(offsets) <= 0):
            raise ValueError("episode_offsets must start at zero and be strictly increasing")
        sample_count = int(offsets[-1])
        arrays = {
            "compute_time_s": _readonly_array(
                self.compute_time_s,
                dtype=_ARRAY_DTYPES["compute_time_s"],
                shape=(sample_count,),
                name="compute_time_s",
            ),
            "speed_mps": _readonly_array(
                self.speed_mps,
                dtype=_ARRAY_DTYPES["speed_mps"],
                shape=(sample_count,),
                name="speed_mps",
            ),
            "lateral_error_m": _readonly_array(
                self.lateral_error_m,
                dtype=_ARRAY_DTYPES["lateral_error_m"],
                shape=(sample_count,),
                name="lateral_error_m",
            ),
            "requested_action": _readonly_array(
                self.requested_action,
                dtype=_ARRAY_DTYPES["requested_action"],
                shape=(sample_count, 2),
                name="requested_action",
            ),
            "steering_saturated": _readonly_bool(
                self.steering_saturated,
                shape=(sample_count,),
                name="steering_saturated",
            ),
            "longitudinal_saturated": _readonly_bool(
                self.longitudinal_saturated,
                shape=(sample_count,),
                name="longitudinal_saturated",
            ),
        }
        if np.any(arrays["compute_time_s"] < 0.0):
            raise ValueError("compute_time_s must be non-negative")
        if np.any(arrays["speed_mps"] < 0.0):
            raise ValueError("speed_mps must be non-negative")

        object.__setattr__(self, "track_id", track_id)
        object.__setattr__(self, "reset_seed", reset_seed)
        object.__setattr__(self, "episode_offsets", offsets)
        for name, array in arrays.items():
            object.__setattr__(self, name, array)

    @property
    def transition_count(self) -> int:
        """Number of stored transitions across all 20 episodes."""

        return int(self.episode_offsets[-1])

    def episode(self, row_index: int) -> EpisodeMetricSamples:
        """Return one immutable episode slice in fixed evaluation order."""

        if isinstance(row_index, bool) or not isinstance(row_index, int):
            raise TypeError("row_index must be an integer")
        if not 0 <= row_index < FINAL_METRICS_EPISODE_COUNT:
            raise IndexError("row_index must lie in [0, 20)")
        start = int(self.episode_offsets[row_index])
        stop = int(self.episode_offsets[row_index + 1])
        return EpisodeMetricSamples(
            track_id=int(self.track_id[row_index]),
            reset_seed=int(self.reset_seed[row_index]),
            compute_time_s=self.compute_time_s[start:stop],
            speed_mps=self.speed_mps[start:stop],
            lateral_error_m=self.lateral_error_m[start:stop],
            requested_action=self.requested_action[start:stop],
            steering_saturated=self.steering_saturated[start:stop],
            longitudinal_saturated=self.longitudinal_saturated[start:stop],
        )

    def episodes(self) -> tuple[EpisodeMetricSamples, ...]:
        """Return all 20 immutable episode slices."""

        return tuple(self.episode(index) for index in range(FINAL_METRICS_EPISODE_COUNT))


def build_final_metrics_data(
    controller_name: str,
    samples: Sequence[EpisodeMetricSamples],
) -> FinalMetricsData:
    """Pack exactly 20 ordered episode samples into the canonical in-memory schema."""

    if isinstance(samples, (str, bytes)) or not isinstance(samples, Sequence):
        raise TypeError("samples must be a sequence of EpisodeMetricSamples")
    episodes = tuple(samples)
    if len(episodes) != FINAL_METRICS_EPISODE_COUNT or not all(
        isinstance(item, EpisodeMetricSamples) for item in episodes
    ):
        raise ValueError("samples must contain exactly 20 EpisodeMetricSamples values")
    offsets = np.concatenate(
        (
            np.zeros(1, dtype=np.int64),
            np.cumsum(
                np.asarray([item.transition_count for item in episodes], dtype=np.int64),
                dtype=np.int64,
            ),
        )
    )
    return FinalMetricsData(
        controller_name=controller_name,
        track_id=np.asarray([item.track_id for item in episodes], dtype=np.uint32),
        reset_seed=np.asarray([item.reset_seed for item in episodes], dtype=np.uint32),
        episode_offsets=offsets,
        compute_time_s=np.concatenate(tuple(item.compute_time_s for item in episodes)),
        speed_mps=np.concatenate(tuple(item.speed_mps for item in episodes)),
        lateral_error_m=np.concatenate(tuple(item.lateral_error_m for item in episodes)),
        requested_action=np.concatenate(tuple(item.requested_action for item in episodes)),
        steering_saturated=np.concatenate(tuple(item.steering_saturated for item in episodes)),
        longitudinal_saturated=np.concatenate(
            tuple(item.longitudinal_saturated for item in episodes)
        ),
    )


def summarize_final_metrics(data: FinalMetricsData) -> AggregateMetricSummary:
    """Recompute the transition-weighted aggregate from the 20 stored episode slices."""

    if not isinstance(data, FinalMetricsData):
        raise TypeError("data must be FinalMetricsData")
    return summarize_metric_episodes(data.episodes())


@dataclass(frozen=True, slots=True)
class FinalMetricsArtifact:
    """Filesystem identity of one canonical final metrics NPZ."""

    path: Path
    sha256: str
    size_bytes: int
    schema_version: int = FINAL_METRICS_SCHEMA_VERSION

    def __post_init__(self) -> None:
        if not isinstance(self.path, Path):
            raise TypeError("path must be pathlib.Path")
        if self.schema_version != FINAL_METRICS_SCHEMA_VERSION:
            raise ValueError("artifact schema_version is invalid")
        if not isinstance(self.sha256, str) or _SHA256_PATTERN.fullmatch(self.sha256) is None:
            raise ValueError("sha256 must be a lowercase SHA-256 digest")
        if isinstance(self.size_bytes, bool) or not isinstance(self.size_bytes, int):
            raise TypeError("size_bytes must be an integer")
        if not 0 < self.size_bytes <= MAX_FINAL_METRICS_NPZ_BYTES:
            raise ValueError("size_bytes is outside the final metrics artifact limit")


@dataclass(frozen=True, slots=True)
class LoadedFinalMetricsArtifact:
    """Strictly loaded immutable data and the identity of its source bytes."""

    data: FinalMetricsData
    artifact: FinalMetricsArtifact

    def __post_init__(self) -> None:
        if not isinstance(self.data, FinalMetricsData):
            raise TypeError("data must be FinalMetricsData")
        if not isinstance(self.artifact, FinalMetricsArtifact):
            raise TypeError("artifact must be FinalMetricsArtifact")


def _ascii_scalar(value: str, *, dtype: np.dtype[Any], name: str) -> NDArray[Any]:
    try:
        encoded = value.encode("ascii")
    except UnicodeEncodeError as error:
        raise ValueError(f"{name} must contain only ASCII characters") from error
    if len(encoded) > dtype.itemsize:
        raise ValueError(f"{name} exceeds its fixed artifact width")
    return np.asarray(encoded, dtype=dtype)


def _artifact_arrays(data: FinalMetricsData) -> dict[str, NDArray[Any]]:
    if not isinstance(data, FinalMetricsData):
        raise TypeError("data must be FinalMetricsData")
    return {
        "schema_version": np.asarray(data.schema_version, dtype=_ARRAY_DTYPES["schema_version"]),
        "benchmark_version": _ascii_scalar(
            data.benchmark_version,
            dtype=_ARRAY_DTYPES["benchmark_version"],
            name="benchmark_version",
        ),
        "controller_name": _ascii_scalar(
            data.controller_name,
            dtype=_ARRAY_DTYPES["controller_name"],
            name="controller_name",
        ),
        "track_id": data.track_id,
        "reset_seed": data.reset_seed,
        "episode_offsets": data.episode_offsets,
        "compute_time_s": data.compute_time_s,
        "speed_mps": data.speed_mps,
        "lateral_error_m": data.lateral_error_m,
        "requested_action": data.requested_action,
        "steering_saturated": data.steering_saturated,
        "longitudinal_saturated": data.longitudinal_saturated,
    }


def _npy_bytes(array: NDArray[Any]) -> bytes:
    output = io.BytesIO()
    contiguous = array if array.ndim == 0 else np.ascontiguousarray(array)
    np.lib.format.write_array(output, contiguous, version=(1, 0), allow_pickle=False)
    return output.getvalue()


def canonical_final_metrics_bytes(data: FinalMetricsData) -> bytes:
    """Return the unique uncompressed NPZ representation for final metric data."""

    arrays = _artifact_arrays(data)
    output = io.BytesIO()
    with zipfile.ZipFile(output, mode="w", compression=zipfile.ZIP_STORED) as archive:
        for name in _ARRAY_NAMES:
            information = zipfile.ZipInfo(f"{name}.npy", date_time=_ZIP_TIMESTAMP)
            information.compress_type = zipfile.ZIP_STORED
            information.create_system = 3
            information.external_attr = _ZIP_EXTERNAL_ATTR
            archive.writestr(information, _npy_bytes(arrays[name]))
    result = output.getvalue()
    if not result or len(result) > MAX_FINAL_METRICS_NPZ_BYTES:
        raise FinalMetricsArtifactError("canonical final metrics NPZ exceeds its public size limit")
    return result


def _secure_output_parent(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    metadata = path.lstat()
    if stat.S_ISLNK(metadata.st_mode) or not stat.S_ISDIR(metadata.st_mode):
        raise FinalMetricsArtifactError("metrics output parent must be a non-symlink directory")
    resolved = path.resolve(strict=True)
    if resolved != path.absolute():
        raise FinalMetricsArtifactError("metrics output parent path cannot traverse symbolic links")
    return resolved


def _fsync_directory(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def write_final_metrics_npz(
    data: FinalMetricsData,
    output_path: str | Path,
) -> FinalMetricsArtifact:
    """Atomically persist and read back one canonical deterministic metrics NPZ."""

    destination = Path(output_path)
    if destination.suffix != ".npz":
        raise ValueError("final metrics output_path must use the .npz suffix")
    parent = _secure_output_parent(destination.parent)
    destination = parent / destination.name
    if destination.exists() or destination.is_symlink():
        metadata = destination.lstat()
        if stat.S_ISLNK(metadata.st_mode) or not stat.S_ISREG(metadata.st_mode):
            raise FinalMetricsArtifactError(
                "final metrics destination must be a non-symlink regular file or absent"
            )
    content = canonical_final_metrics_bytes(data)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{destination.name}.", suffix=".tmp", dir=parent
    )
    temporary = Path(temporary_name)
    try:
        os.fchmod(descriptor, 0o600)
        with os.fdopen(descriptor, "wb") as stream:
            descriptor = -1
            stream.write(content)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, destination)
        _fsync_directory(parent)
        loaded = load_final_metrics_npz(destination)
        if canonical_final_metrics_bytes(loaded.data) != content:
            raise FinalMetricsArtifactError("final metrics artifact failed exact byte readback")
        return loaded.artifact
    except BaseException as error:
        if descriptor >= 0:
            os.close(descriptor)
        temporary.unlink(missing_ok=True)
        if isinstance(error, (FinalMetricsArtifactError, ValueError, TypeError)):
            raise
        raise FinalMetricsArtifactError("failed to persist canonical final metrics") from error
    finally:
        temporary.unlink(missing_ok=True)


def _validate_expected_identity(
    expected_sha256: str | None,
    expected_size_bytes: int | None,
) -> None:
    if expected_sha256 is not None and (
        not isinstance(expected_sha256, str) or _SHA256_PATTERN.fullmatch(expected_sha256) is None
    ):
        raise ValueError("expected_sha256 must be a lowercase SHA-256 digest")
    if expected_size_bytes is not None and (
        isinstance(expected_size_bytes, bool)
        or not isinstance(expected_size_bytes, int)
        or expected_size_bytes < 1
    ):
        raise ValueError("expected_size_bytes must be a positive integer")


def _strict_file_bytes(path: Path) -> bytes:
    try:
        before = path.lstat()
    except FileNotFoundError:
        raise
    if stat.S_ISLNK(before.st_mode) or not stat.S_ISREG(before.st_mode):
        raise FinalMetricsArtifactError(
            "final metrics input_path must be a non-symlink regular file"
        )
    if not 0 < before.st_size <= MAX_FINAL_METRICS_NPZ_BYTES:
        raise FinalMetricsArtifactError("final metrics file size is outside the public limit")
    try:
        descriptor = os.open(path, os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0))
    except OSError as error:
        raise FinalMetricsArtifactError("cannot securely open final metrics input_path") from error
    try:
        opened = os.fstat(descriptor)
        if (
            not stat.S_ISREG(opened.st_mode)
            or opened.st_dev != before.st_dev
            or opened.st_ino != before.st_ino
            or opened.st_size != before.st_size
        ):
            raise FinalMetricsArtifactError("final metrics input_path changed while opening")
        content = bytearray()
        while len(content) <= MAX_FINAL_METRICS_NPZ_BYTES:
            chunk = os.read(
                descriptor,
                min(1024 * 1024, MAX_FINAL_METRICS_NPZ_BYTES + 1 - len(content)),
            )
            if not chunk:
                break
            content.extend(chunk)
        after = os.fstat(descriptor)
    finally:
        os.close(descriptor)
    if len(content) != before.st_size or after.st_size != before.st_size:
        raise FinalMetricsArtifactError("final metrics input_path changed while reading")
    return bytes(content)


def _decode_ascii_scalar(array: NDArray[Any], *, name: str) -> str:
    try:
        value = array.item().decode("ascii")
    except (AttributeError, UnicodeDecodeError, ValueError) as error:
        raise FinalMetricsArtifactError(f"{name} metadata is not canonical ASCII") from error
    return value


def _data_from_arrays(arrays: Mapping[str, NDArray[Any]]) -> FinalMetricsData:
    if tuple(arrays) != _ARRAY_NAMES:
        raise FinalMetricsArtifactError("final metrics arrays differ from the strict schema")
    for name, array in arrays.items():
        if array.dtype != _ARRAY_DTYPES[name]:
            raise FinalMetricsArtifactError(
                f"{name} must use canonical dtype {_ARRAY_DTYPES[name].str}"
            )
    if arrays["schema_version"].shape != ():
        raise FinalMetricsArtifactError("schema_version must be a scalar")
    for name in ("benchmark_version", "controller_name"):
        if arrays[name].shape != ():
            raise FinalMetricsArtifactError(f"{name} must be a scalar")
    try:
        return FinalMetricsData(
            schema_version=int(arrays["schema_version"]),
            benchmark_version=_decode_ascii_scalar(
                arrays["benchmark_version"], name="benchmark_version"
            ),
            controller_name=_decode_ascii_scalar(arrays["controller_name"], name="controller_name"),
            track_id=arrays["track_id"],
            reset_seed=arrays["reset_seed"],
            episode_offsets=arrays["episode_offsets"],
            compute_time_s=arrays["compute_time_s"],
            speed_mps=arrays["speed_mps"],
            lateral_error_m=arrays["lateral_error_m"],
            requested_action=arrays["requested_action"],
            steering_saturated=arrays["steering_saturated"],
            longitudinal_saturated=arrays["longitudinal_saturated"],
        )
    except (TypeError, ValueError) as error:
        raise FinalMetricsArtifactError(
            "final metrics arrays violate shape, identity, or value rules"
        ) from error


def load_final_metrics_npz(
    input_path: str | Path,
    *,
    expected_sha256: str | None = None,
    expected_size_bytes: int | None = None,
) -> LoadedFinalMetricsArtifact:
    """Load only an exact canonical, hash-bindable, non-pickled metrics NPZ."""

    path = Path(input_path)
    if path.suffix != ".npz":
        raise ValueError("final metrics input_path must use the .npz suffix")
    _validate_expected_identity(expected_sha256, expected_size_bytes)
    content = _strict_file_bytes(path)
    sha256 = hashlib.sha256(content).hexdigest()
    if expected_sha256 is not None and sha256 != expected_sha256:
        raise FinalMetricsArtifactError("final metrics SHA-256 differs from expected_sha256")
    if expected_size_bytes is not None and len(content) != expected_size_bytes:
        raise FinalMetricsArtifactError("final metrics size differs from expected_size_bytes")

    expected_entries = tuple(f"{name}.npy" for name in _ARRAY_NAMES)
    try:
        with zipfile.ZipFile(io.BytesIO(content), mode="r") as archive:
            information = archive.infolist()
            if tuple(item.filename for item in information) != expected_entries:
                raise FinalMetricsArtifactError(
                    "final metrics ZIP entries differ from the strict schema"
                )
            if any(
                item.compress_type != zipfile.ZIP_STORED
                or item.date_time != _ZIP_TIMESTAMP
                or item.create_system != 3
                or item.external_attr != _ZIP_EXTERNAL_ATTR
                or item.file_size > MAX_FINAL_METRICS_NPZ_BYTES
                for item in information
            ):
                raise FinalMetricsArtifactError("final metrics ZIP metadata is not canonical")
            if sum(item.file_size for item in information) > MAX_FINAL_METRICS_NPZ_BYTES:
                raise FinalMetricsArtifactError("final metrics ZIP payload exceeds its size limit")
        with np.load(io.BytesIO(content), allow_pickle=False) as archive:
            if tuple(archive.files) != _ARRAY_NAMES:
                raise FinalMetricsArtifactError(
                    "final metrics NPZ keys differ from the strict schema"
                )
            arrays = {name: np.array(archive[name], copy=True) for name in _ARRAY_NAMES}
    except FinalMetricsArtifactError:
        raise
    except (OSError, ValueError, zipfile.BadZipFile) as error:
        raise FinalMetricsArtifactError(
            "final metrics artifact is not a valid non-pickled NPZ"
        ) from error

    data = _data_from_arrays(arrays)
    if canonical_final_metrics_bytes(data) != content:
        raise FinalMetricsArtifactError("final metrics bytes are not the canonical representation")
    artifact = FinalMetricsArtifact(path=path, sha256=sha256, size_bytes=len(content))
    return LoadedFinalMetricsArtifact(data=data, artifact=artifact)


__all__ = [
    "FINAL_METRICS_BENCHMARK_VERSION",
    "FINAL_METRICS_CONTROL_DT_S",
    "FINAL_METRICS_EPISODE_COUNT",
    "FINAL_METRICS_PROJECTION_BACKWARD_SEGMENTS",
    "FINAL_METRICS_PROJECTION_FORWARD_SEGMENTS",
    "FINAL_METRICS_SCHEMA_VERSION",
    "MAX_FINAL_METRICS_NPZ_BYTES",
    "AggregateMetricSummary",
    "EpisodeMetricSamples",
    "EpisodeMetricSummary",
    "FinalMetricsArtifact",
    "FinalMetricsArtifactError",
    "FinalMetricsData",
    "LoadedFinalMetricsArtifact",
    "MetricActionLimits",
    "build_final_metrics_data",
    "canonical_final_metrics_bytes",
    "compute_episode_metric_samples",
    "load_final_metrics_npz",
    "summarize_episode_metrics",
    "summarize_final_metrics",
    "summarize_metric_episodes",
    "write_final_metrics_npz",
]
