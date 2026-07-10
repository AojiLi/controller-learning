"""Strict, deterministic storage for versioned benchmark Track assets."""

from __future__ import annotations

import hashlib
import io
import json
import os
import re
import tempfile
import zipfile
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Any, Literal

import numpy as np

from controller_learning.tracks.hashing import track_batch_geometry_sha256
from controller_learning.tracks.level0 import LEVEL0_TRACK_SEED
from controller_learning.tracks.types import TrackBatch, TrackCapacity

TRACK_ASSET_SCHEMA_VERSION = 1
TrackSplit = Literal["level0", "train", "validation", "test"]

_SPLITS = {"level0", "train", "validation", "test"}
_SHA256_PATTERN = re.compile(r"[0-9a-f]{64}", flags=re.ASCII)
_TRACK_BATCH_DTYPES: dict[str, np.dtype] = {
    "seed": np.dtype(np.uint32),
    "centerline_m": np.dtype(np.float32),
    "left_boundary_m": np.dtype(np.float32),
    "right_boundary_m": np.dtype(np.float32),
    "tangent": np.dtype(np.float32),
    "curvature_1pm": np.dtype(np.float32),
    "cumulative_s_m": np.dtype(np.float32),
    "track_mask": np.dtype(np.bool_),
    "checkpoint_center_m": np.dtype(np.float32),
    "checkpoint_tangent": np.dtype(np.float32),
    "checkpoint_s_m": np.dtype(np.float32),
    "checkpoint_mask": np.dtype(np.bool_),
    "start_pose": np.dtype(np.float32),
    "point_count": np.dtype(np.int32),
    "checkpoint_count": np.dtype(np.int32),
    "length_m": np.dtype(np.float32),
    "width_m": np.dtype(np.float32),
}
_TRACK_BATCH_FIELDS = tuple(_TRACK_BATCH_DTYPES)
_FLOAT_FIELDS = tuple(name for name, dtype in _TRACK_BATCH_DTYPES.items() if dtype.kind == "f")


class TrackAssetError(ValueError):
    """Raised when a Track asset or manifest violates the versioned schema."""


def _require_sha256(value: str, field: str) -> None:
    if not isinstance(value, str) or _SHA256_PATTERN.fullmatch(value) is None:
        raise TrackAssetError(f"{field} must be a lowercase SHA-256 digest")


def _require_nonempty_string(value: str, field: str) -> None:
    if not isinstance(value, str) or not value:
        raise TrackAssetError(f"{field} must be a non-empty string")


def _require_exact_keys(value: dict[str, Any], expected: set[str], context: str) -> None:
    actual = set(value)
    missing = expected - actual
    extra = actual - expected
    details: list[str] = []
    if missing:
        details.append(f"missing keys: {', '.join(sorted(missing))}")
    if extra:
        details.append(f"unexpected keys: {', '.join(sorted(extra))}")
    if details:
        raise TrackAssetError(f"{context} has {'; '.join(details)}")


@dataclass(frozen=True, slots=True)
class TrackAssetRecord:
    """Identity, geometry digest, and admission results for one accepted Track."""

    seed: int
    geometry_sha256: str
    geometry_validation: str
    driveability_validation: str

    def __post_init__(self) -> None:
        if type(self.seed) is not int or not 0 <= self.seed <= np.iinfo(np.uint32).max:
            raise TrackAssetError("record.seed must fit in uint32")
        _require_sha256(self.geometry_sha256, "record.geometry_sha256")
        if self.geometry_validation != "passed":
            raise TrackAssetError("accepted Track geometry_validation must be 'passed'")
        if self.driveability_validation != "passed":
            raise TrackAssetError("accepted Track driveability_validation must be 'passed'")


@dataclass(frozen=True, slots=True)
class TrackAssetManifest:
    """One strict Level/split manifest and the exact NPZ artifact it identifies."""

    schema_version: int
    benchmark_version: str
    level_id: int
    split: TrackSplit
    generator_version: str
    geometry_validation_version: str
    driveability_protocol_version: str
    track_width_m: float
    track_count: int
    capacity: TrackCapacity
    asset_file: str
    asset_sha256: str
    tracks: tuple[TrackAssetRecord, ...]

    def __post_init__(self) -> None:
        if (
            type(self.schema_version) is not int
            or self.schema_version != TRACK_ASSET_SCHEMA_VERSION
        ):
            raise TrackAssetError(
                f"schema_version must be {TRACK_ASSET_SCHEMA_VERSION}, got {self.schema_version!r}"
            )
        _require_nonempty_string(self.benchmark_version, "benchmark_version")
        if type(self.level_id) is not int or self.level_id < 0:
            raise TrackAssetError("level_id must be a non-negative integer")
        if self.split not in _SPLITS:
            raise TrackAssetError(f"split must be one of {sorted(_SPLITS)}")
        if self.split == "level0":
            if self.level_id != 0 or self.track_count != 1:
                raise TrackAssetError("the level0 split must contain exactly one Level 0 Track")
        elif self.level_id != 1:
            raise TrackAssetError("train, validation, and test splits must belong to Level 1")
        for value, field in (
            (self.generator_version, "generator_version"),
            (self.geometry_validation_version, "geometry_validation_version"),
            (self.driveability_protocol_version, "driveability_protocol_version"),
        ):
            _require_nonempty_string(value, field)
        if (
            isinstance(self.track_width_m, bool)
            or not isinstance(self.track_width_m, (int, float))
            or not np.isfinite(self.track_width_m)
            or self.track_width_m <= 0.0
        ):
            raise TrackAssetError("track_width_m must be finite and positive")
        if type(self.track_count) is not int or self.track_count < 1:
            raise TrackAssetError("track_count must be a positive integer")
        if not isinstance(self.capacity, TrackCapacity):
            raise TrackAssetError("capacity must be a TrackCapacity")
        if (
            type(self.capacity.max_track_points) is not int
            or type(self.capacity.max_checkpoints) is not int
        ):
            raise TrackAssetError("capacity values must be integers")
        if not isinstance(self.asset_file, str):
            raise TrackAssetError("asset_file must be a string")
        asset_path = PurePosixPath(self.asset_file)
        if (
            not self.asset_file
            or "\\" in self.asset_file
            or asset_path.is_absolute()
            or ".." in asset_path.parts
            or asset_path.suffix != ".npz"
        ):
            raise TrackAssetError("asset_file must be a safe relative POSIX .npz path")
        _require_sha256(self.asset_sha256, "asset_sha256")

        if not isinstance(self.tracks, (tuple, list)):
            raise TrackAssetError("tracks must be a sequence of TrackAssetRecords")
        tracks = tuple(self.tracks)
        if len(tracks) != self.track_count or not all(
            isinstance(record, TrackAssetRecord) for record in tracks
        ):
            raise TrackAssetError("tracks must contain exactly track_count TrackAssetRecords")
        seeds = tuple(record.seed for record in tracks)
        geometry_hashes = tuple(record.geometry_sha256 for record in tracks)
        if len(set(seeds)) != len(seeds):
            raise TrackAssetError("Track seeds must be unique within a split")
        if len(set(geometry_hashes)) != len(geometry_hashes):
            raise TrackAssetError("geometry hashes must be unique within a split")
        if self.split == "level0":
            if seeds != (LEVEL0_TRACK_SEED,):
                raise TrackAssetError("the Level 0 Track must use the reserved Level 0 seed")
        elif LEVEL0_TRACK_SEED in seeds:
            raise TrackAssetError("the reserved Level 0 seed cannot appear in a Level 1 split")
        object.__setattr__(self, "tracks", tracks)


def sha256_file(path: str | Path) -> str:
    """Return the SHA-256 digest of one file without loading it all at once."""

    digest = hashlib.sha256()
    with Path(path).open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _expected_shapes(track_count: int, point_count: int, checkpoint_count: int) -> dict[str, tuple]:
    vector = (track_count,)
    points = (track_count, point_count)
    checkpoints = (track_count, checkpoint_count)
    return {
        "seed": vector,
        "centerline_m": (*points, 2),
        "left_boundary_m": (*points, 2),
        "right_boundary_m": (*points, 2),
        "tangent": (*points, 2),
        "curvature_1pm": points,
        "cumulative_s_m": points,
        "track_mask": points,
        "checkpoint_center_m": (*checkpoints, 2),
        "checkpoint_tangent": (*checkpoints, 2),
        "checkpoint_s_m": checkpoints,
        "checkpoint_mask": checkpoints,
        "start_pose": (track_count, 3),
        "point_count": vector,
        "checkpoint_count": vector,
        "length_m": vector,
        "width_m": vector,
    }


def _batch_arrays(batch: TrackBatch) -> dict[str, np.ndarray]:
    if not isinstance(batch, TrackBatch):
        raise TrackAssetError("batch must be a TrackBatch")
    arrays = {name: getattr(batch, name) for name in _TRACK_BATCH_FIELDS}
    if not all(isinstance(value, np.ndarray) for value in arrays.values()):
        raise TrackAssetError("Track asset serialization requires host NumPy arrays")
    return arrays


def validate_track_batch(batch: TrackBatch) -> TrackCapacity:
    """Validate exact host dtypes, shapes, padding, masks, and structural integrity."""

    arrays = _batch_arrays(batch)
    seed = arrays["seed"]
    centerline = arrays["centerline_m"]
    checkpoints = arrays["checkpoint_center_m"]
    if seed.ndim != 1 or seed.shape[0] < 1:
        raise TrackAssetError("seed must have shape (track_count,) with track_count positive")
    if centerline.ndim != 3 or centerline.shape[2] != 2 or centerline.shape[1] < 4:
        raise TrackAssetError("centerline_m must have shape (track_count, points>=4, 2)")
    if checkpoints.ndim != 3 or checkpoints.shape[2] != 2 or checkpoints.shape[1] < 1:
        raise TrackAssetError(
            "checkpoint_center_m must have shape (track_count, checkpoints>=1, 2)"
        )
    track_count = seed.shape[0]
    point_capacity = centerline.shape[1]
    checkpoint_capacity = checkpoints.shape[1]
    shapes = _expected_shapes(track_count, point_capacity, checkpoint_capacity)
    for name, array in arrays.items():
        expected_dtype = _TRACK_BATCH_DTYPES[name]
        if (
            array.dtype.kind != expected_dtype.kind
            or array.dtype.itemsize != expected_dtype.itemsize
        ):
            raise TrackAssetError(
                f"{name} must use dtype {expected_dtype.name}, got {array.dtype.name}"
            )
        if array.shape != shapes[name]:
            raise TrackAssetError(f"{name} must have shape {shapes[name]}, got {array.shape}")
    for name in _FLOAT_FIELDS:
        if not np.isfinite(arrays[name]).all():
            raise TrackAssetError(f"{name} must contain only finite values")

    point_count = arrays["point_count"]
    checkpoint_count = arrays["checkpoint_count"]
    if np.any((point_count < 4) | (point_count > point_capacity)):
        raise TrackAssetError("point_count must fit the point capacity and include closure")
    if np.any((checkpoint_count < 1) | (checkpoint_count > checkpoint_capacity)):
        raise TrackAssetError("checkpoint_count must fit the checkpoint capacity")
    point_valid = np.arange(point_capacity)[None, :] < point_count[:, None]
    checkpoint_valid = np.arange(checkpoint_capacity)[None, :] < checkpoint_count[:, None]
    if not np.array_equal(arrays["track_mask"], point_valid):
        raise TrackAssetError("track_mask must be one contiguous valid prefix per Track")
    if not np.array_equal(arrays["checkpoint_mask"], checkpoint_valid):
        raise TrackAssetError("checkpoint_mask must be one contiguous valid prefix per Track")

    for name in (
        "centerline_m",
        "left_boundary_m",
        "right_boundary_m",
        "tangent",
        "curvature_1pm",
        "cumulative_s_m",
    ):
        if np.any(arrays[name][~point_valid] != 0):
            raise TrackAssetError(f"{name} padding must be zero")
    for name in (
        "checkpoint_center_m",
        "checkpoint_tangent",
        "checkpoint_s_m",
    ):
        if np.any(arrays[name][~checkpoint_valid] != 0):
            raise TrackAssetError(f"{name} padding must be zero")

    indices = np.arange(track_count)
    closing_points = point_count - 1
    for name in ("centerline_m", "left_boundary_m", "right_boundary_m", "tangent"):
        if not np.array_equal(arrays[name][:, 0], arrays[name][indices, closing_points]):
            raise TrackAssetError(f"{name} must include an exact closure point")
    cumulative = arrays["cumulative_s_m"]
    if np.any(cumulative[:, 0] != 0.0):
        raise TrackAssetError("cumulative_s_m must start at zero")
    segment_valid = np.arange(point_capacity - 1)[None, :] < (point_count - 1)[:, None]
    if np.any(np.diff(cumulative, axis=1)[segment_valid] <= 0.0):
        raise TrackAssetError("valid cumulative_s_m values must increase strictly")
    if not np.allclose(
        cumulative[indices, closing_points],
        arrays["length_m"],
        rtol=0.0,
        atol=1.0e-4,
    ):
        raise TrackAssetError("the closing cumulative distance must equal length_m")
    if np.any(arrays["length_m"] <= 0.0) or np.any(arrays["width_m"] <= 0.0):
        raise TrackAssetError("length_m and width_m must be positive")

    checkpoint_s = arrays["checkpoint_s_m"]
    checkpoint_closing = checkpoint_count - 1
    if np.any(checkpoint_s[:, 0] <= 0.0):
        raise TrackAssetError("the first checkpoint distance must be positive")
    checkpoint_segment_valid = (
        np.arange(checkpoint_capacity - 1)[None, :] < (checkpoint_count - 1)[:, None]
    )
    if np.any(np.diff(checkpoint_s, axis=1)[checkpoint_segment_valid] <= 0.0):
        raise TrackAssetError("valid checkpoint_s_m values must increase strictly")
    if not np.allclose(
        checkpoint_s[indices, checkpoint_closing],
        arrays["length_m"],
        rtol=0.0,
        atol=1.0e-4,
    ):
        raise TrackAssetError("the final checkpoint distance must equal length_m")
    if not np.allclose(
        arrays["checkpoint_center_m"][indices, checkpoint_closing],
        centerline[:, 0],
        rtol=0.0,
        atol=2.0e-2,
    ):
        raise TrackAssetError("the final checkpoint center must close at the start")
    return TrackCapacity(point_capacity, checkpoint_capacity)


def _canonical_batch_arrays(batch: TrackBatch) -> dict[str, np.ndarray]:
    validate_track_batch(batch)
    return {
        name: np.ascontiguousarray(
            getattr(batch, name),
            dtype=dtype.newbyteorder("<"),
        )
        for name, dtype in _TRACK_BATCH_DTYPES.items()
    }


def save_track_batch_npz(batch: TrackBatch, path: str | Path) -> str:
    """Write a canonical uncompressed NPZ and return its SHA-256 digest.

    The archive uses a fixed member order, timestamp, permissions, NPY version, and little-endian
    dtype. It therefore has byte-identical output for byte-identical TrackBatch values.
    """

    output = Path(path)
    if output.suffix != ".npz":
        raise TrackAssetError("Track asset path must use the .npz suffix")
    arrays = _canonical_batch_arrays(batch)
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            prefix=f".{output.name}.", suffix=".tmp", dir=output.parent, delete=False
        ) as file:
            temporary = Path(file.name)
        with zipfile.ZipFile(temporary, mode="w", compression=zipfile.ZIP_STORED) as archive:
            for name in _TRACK_BATCH_FIELDS:
                payload = io.BytesIO()
                np.lib.format.write_array(
                    payload,
                    arrays[name],
                    version=(2, 0),
                    allow_pickle=False,
                )
                member = zipfile.ZipInfo(f"{name}.npy", date_time=(1980, 1, 1, 0, 0, 0))
                member.compress_type = zipfile.ZIP_STORED
                member.create_system = 3
                member.external_attr = 0o100644 << 16
                archive.writestr(member, payload.getvalue())
        os.replace(temporary, output)
        temporary = None
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)
    return sha256_file(output)


def load_track_batch_npz(
    path: str | Path,
    *,
    expected_sha256: str | None = None,
    expected_track_count: int | None = None,
    expected_capacity: TrackCapacity | None = None,
) -> TrackBatch:
    """Load a canonical NPZ with pickle disabled and verify its complete schema."""

    source = Path(path)
    if source.suffix != ".npz":
        raise TrackAssetError("Track asset path must use the .npz suffix")
    if expected_sha256 is not None:
        _require_sha256(expected_sha256, "expected_sha256")
        try:
            actual_sha256 = sha256_file(source)
        except FileNotFoundError as error:
            raise TrackAssetError(f"Track asset does not exist: {source}") from error
        if actual_sha256 != expected_sha256:
            raise TrackAssetError("Track asset SHA-256 does not match the manifest")
    expected_members = tuple(f"{name}.npy" for name in _TRACK_BATCH_FIELDS)
    try:
        with zipfile.ZipFile(source, mode="r") as archive:
            members = tuple(member.filename for member in archive.infolist())
            if members != expected_members:
                raise TrackAssetError("Track NPZ members do not exactly match the schema")
            if any(member.compress_type != zipfile.ZIP_STORED for member in archive.infolist()):
                raise TrackAssetError("Track NPZ members must use canonical uncompressed storage")
        with np.load(source, allow_pickle=False) as archive:
            arrays = {name: np.array(archive[name], copy=True) for name in _TRACK_BATCH_FIELDS}
        for name, array in arrays.items():
            canonical_dtype = _TRACK_BATCH_DTYPES[name].newbyteorder("<")
            if array.dtype.str != canonical_dtype.str:
                raise TrackAssetError(
                    f"{name} must use canonical little-endian dtype {canonical_dtype.str}"
                )
    except TrackAssetError:
        raise
    except FileNotFoundError as error:
        raise TrackAssetError(f"Track asset does not exist: {source}") from error
    except (OSError, ValueError, zipfile.BadZipFile) as error:
        raise TrackAssetError(f"invalid Track NPZ: {source}") from error
    batch = TrackBatch(**arrays)
    capacity = validate_track_batch(batch)
    if expected_track_count is not None:
        if type(expected_track_count) is not int or expected_track_count < 1:
            raise TrackAssetError("expected_track_count must be a positive integer")
        if batch.seed.shape[0] != expected_track_count:
            raise TrackAssetError("Track asset count does not match the manifest")
    if expected_capacity is not None and capacity != expected_capacity:
        raise TrackAssetError("Track asset capacity does not match the manifest")
    for array in batch:
        array.setflags(write=False)
    return batch


def _record_dict(record: TrackAssetRecord) -> dict[str, Any]:
    return {
        "driveability_validation": record.driveability_validation,
        "geometry_sha256": record.geometry_sha256,
        "geometry_validation": record.geometry_validation,
        "seed": record.seed,
    }


def manifest_dict(manifest: TrackAssetManifest) -> dict[str, Any]:
    """Return the canonical JSON-compatible mapping for a Track asset manifest."""

    return {
        "asset_file": manifest.asset_file,
        "asset_sha256": manifest.asset_sha256,
        "benchmark_version": manifest.benchmark_version,
        "capacity": {
            "max_checkpoints": manifest.capacity.max_checkpoints,
            "max_track_points": manifest.capacity.max_track_points,
        },
        "driveability_protocol_version": manifest.driveability_protocol_version,
        "generator_version": manifest.generator_version,
        "geometry_validation_version": manifest.geometry_validation_version,
        "level_id": manifest.level_id,
        "schema_version": manifest.schema_version,
        "split": manifest.split,
        "track_count": manifest.track_count,
        "track_width_m": float(manifest.track_width_m),
        "tracks": [_record_dict(record) for record in manifest.tracks],
    }


def write_track_asset_manifest(manifest: TrackAssetManifest, path: str | Path) -> str:
    """Write one canonical manifest JSON file and return its SHA-256 digest."""

    if not isinstance(manifest, TrackAssetManifest):
        raise TrackAssetError("manifest must be a TrackAssetManifest")
    output = Path(path)
    if output.suffix != ".json":
        raise TrackAssetError("Track manifest path must use the .json suffix")
    serialized = (
        json.dumps(manifest_dict(manifest), indent=2, sort_keys=True, allow_nan=False) + "\n"
    ).encode("utf-8")
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            prefix=f".{output.name}.", suffix=".tmp", dir=output.parent, delete=False
        ) as file:
            temporary = Path(file.name)
            file.write(serialized)
        os.replace(temporary, output)
        temporary = None
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)
    return sha256_file(output)


def _reject_duplicate_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    value: dict[str, Any] = {}
    for key, item in pairs:
        if key in value:
            raise TrackAssetError(f"duplicate JSON key: {key}")
        value[key] = item
    return value


def _reject_json_constant(value: str) -> None:
    raise TrackAssetError(f"non-finite JSON value is not allowed: {value}")


def _parse_record(value: Any, index: int) -> TrackAssetRecord:
    if not isinstance(value, dict):
        raise TrackAssetError(f"tracks[{index}] must be a JSON object")
    _require_exact_keys(
        value,
        {"seed", "geometry_sha256", "geometry_validation", "driveability_validation"},
        f"tracks[{index}]",
    )
    return TrackAssetRecord(
        seed=value["seed"],
        geometry_sha256=value["geometry_sha256"],
        geometry_validation=value["geometry_validation"],
        driveability_validation=value["driveability_validation"],
    )


def load_track_asset_manifest(path: str | Path) -> TrackAssetManifest:
    """Load a strict manifest, rejecting missing, extra, duplicate, or mistyped fields."""

    source = Path(path)
    if source.suffix != ".json":
        raise TrackAssetError("Track manifest path must use the .json suffix")
    try:
        data = json.loads(
            source.read_text(encoding="utf-8"),
            object_pairs_hook=_reject_duplicate_keys,
            parse_constant=_reject_json_constant,
        )
    except FileNotFoundError as error:
        raise TrackAssetError(f"Track manifest does not exist: {source}") from error
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise TrackAssetError(f"invalid Track manifest JSON: {source}") from error
    if not isinstance(data, dict):
        raise TrackAssetError("Track manifest root must be a JSON object")
    _require_exact_keys(
        data,
        {
            "asset_file",
            "asset_sha256",
            "benchmark_version",
            "capacity",
            "driveability_protocol_version",
            "generator_version",
            "geometry_validation_version",
            "level_id",
            "schema_version",
            "split",
            "track_count",
            "track_width_m",
            "tracks",
        },
        "Track manifest",
    )
    capacity = data["capacity"]
    if not isinstance(capacity, dict):
        raise TrackAssetError("capacity must be a JSON object")
    _require_exact_keys(capacity, {"max_track_points", "max_checkpoints"}, "capacity")
    if (
        type(capacity["max_track_points"]) is not int
        or type(capacity["max_checkpoints"]) is not int
    ):
        raise TrackAssetError("Track capacity values must be integers")
    if not isinstance(data["tracks"], list):
        raise TrackAssetError("tracks must be a JSON array")
    try:
        track_capacity = TrackCapacity(
            max_track_points=capacity["max_track_points"],
            max_checkpoints=capacity["max_checkpoints"],
        )
    except (TypeError, ValueError) as error:
        raise TrackAssetError("invalid Track capacity") from error
    return TrackAssetManifest(
        schema_version=data["schema_version"],
        benchmark_version=data["benchmark_version"],
        level_id=data["level_id"],
        split=data["split"],
        generator_version=data["generator_version"],
        geometry_validation_version=data["geometry_validation_version"],
        driveability_protocol_version=data["driveability_protocol_version"],
        track_width_m=data["track_width_m"],
        track_count=data["track_count"],
        capacity=track_capacity,
        asset_file=data["asset_file"],
        asset_sha256=data["asset_sha256"],
        tracks=tuple(_parse_record(value, index) for index, value in enumerate(data["tracks"])),
    )


def load_manifest_track_batch(
    manifest_path: str | Path,
) -> tuple[TrackAssetManifest, TrackBatch]:
    """Load a manifest and verify every identity and geometry digest in its NPZ."""

    path = Path(manifest_path)
    manifest = load_track_asset_manifest(path)
    batch = load_track_batch_npz(
        path.parent / PurePosixPath(manifest.asset_file),
        expected_sha256=manifest.asset_sha256,
        expected_track_count=manifest.track_count,
        expected_capacity=manifest.capacity,
    )
    expected_seeds = np.asarray([record.seed for record in manifest.tracks], dtype=np.uint32)
    if not np.array_equal(batch.seed, expected_seeds):
        raise TrackAssetError("Track asset seed order does not match the manifest")
    actual_hashes = track_batch_geometry_sha256(batch)
    expected_hashes = tuple(record.geometry_sha256 for record in manifest.tracks)
    if actual_hashes != expected_hashes:
        raise TrackAssetError("Track asset geometry hashes do not match the manifest")
    if not np.all(batch.width_m == np.float32(manifest.track_width_m)):
        raise TrackAssetError("Track asset width does not match the manifest")
    return manifest, batch


__all__ = [
    "TRACK_ASSET_SCHEMA_VERSION",
    "TrackAssetError",
    "TrackAssetManifest",
    "TrackAssetRecord",
    "TrackSplit",
    "load_manifest_track_batch",
    "load_track_asset_manifest",
    "load_track_batch_npz",
    "manifest_dict",
    "save_track_batch_npz",
    "sha256_file",
    "validate_track_batch",
    "write_track_asset_manifest",
]
