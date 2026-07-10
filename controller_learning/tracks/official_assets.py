"""Official v0.1 Track split verification and reproducible training-pool caches."""

from __future__ import annotations

import os
import tempfile
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from itertools import pairwise
from pathlib import Path
from types import MappingProxyType

import numpy as np

from controller_learning.config.models import ProjectConfig
from controller_learning.tracks.assets import (
    TrackAssetError,
    TrackAssetManifest,
    TrackSplit,
    load_manifest_track_batch,
    load_track_asset_manifest,
    load_track_batch_npz,
    save_track_batch_npz,
    sha256_file,
    validate_track_batch,
)
from controller_learning.tracks.generator import (
    TrackGenerationError,
    generate_track_candidate,
    pack_track,
)
from controller_learning.tracks.hashing import (
    track_batch_geometry_sha256,
    track_geometry_sha256,
)
from controller_learning.tracks.level0 import LEVEL0_TRACK_SEED
from controller_learning.tracks.specs import (
    generation_spec_from_project,
    track_capacity_from_project,
    validation_spec_from_project,
)
from controller_learning.tracks.types import Track, TrackBatch
from controller_learning.tracks.validator import validate_track_candidate

OFFICIAL_BENCHMARK_VERSION = "0.1"
OFFICIAL_GEOMETRY_VALIDATION_VERSION = "m3-geometry-v1"
OFFICIAL_DRIVEABILITY_PROTOCOL_VERSION = "m5-driveability-v1"
DEFAULT_TRAIN_CACHE = Path(".track-cache/v0.1/train_pool.npz")

ProgressCallback = Callable[[int, int, int], None]


@dataclass(frozen=True, slots=True)
class OfficialTrackSplitSpec:
    """Locked identity namespace and file contract for one official v0.1 split."""

    split: TrackSplit
    level_id: int
    track_count: int
    seed_start: int
    seed_stop: int
    manifest_file: str
    asset_file: str
    package_asset: bool

    def __post_init__(self) -> None:
        if type(self.level_id) is not int or self.level_id not in (0, 1):
            raise ValueError("level_id must be 0 or 1")
        if type(self.track_count) is not int or self.track_count < 1:
            raise ValueError("track_count must be positive")
        if (
            type(self.seed_start) is not int
            or type(self.seed_stop) is not int
            or not 0 <= self.seed_start < self.seed_stop <= 2**32
        ):
            raise ValueError("the seed namespace must be a non-empty uint32 half-open range")
        if not self.manifest_file.endswith(".json") or Path(self.manifest_file).name != (
            self.manifest_file
        ):
            raise ValueError("manifest_file must be a plain JSON filename")
        if not self.asset_file.endswith(".npz") or Path(self.asset_file).name != self.asset_file:
            raise ValueError("asset_file must be a plain NPZ filename")

    def contains_seed(self, seed: int) -> bool:
        """Return whether ``seed`` belongs to this split's locked namespace."""

        return self.seed_start <= seed < self.seed_stop


OFFICIAL_TRACK_SPLITS: tuple[OfficialTrackSplitSpec, ...] = (
    OfficialTrackSplitSpec(
        split="level0",
        level_id=0,
        track_count=1,
        seed_start=LEVEL0_TRACK_SEED,
        seed_stop=LEVEL0_TRACK_SEED + 1,
        manifest_file="level0.json",
        asset_file="level0.npz",
        package_asset=True,
    ),
    OfficialTrackSplitSpec(
        split="train",
        level_id=1,
        track_count=10_000,
        seed_start=0,
        seed_stop=1_000_000,
        manifest_file="train.json",
        asset_file="train_pool.npz",
        package_asset=False,
    ),
    OfficialTrackSplitSpec(
        split="validation",
        level_id=1,
        track_count=100,
        seed_start=1_000_000,
        seed_stop=2_000_000,
        manifest_file="validation.json",
        asset_file="validation.npz",
        package_asset=True,
    ),
    OfficialTrackSplitSpec(
        split="test",
        level_id=1,
        track_count=20,
        seed_start=2_000_000,
        seed_stop=3_000_000,
        manifest_file="test.json",
        asset_file="test.npz",
        package_asset=True,
    ),
)
_SPLITS_BY_NAME = MappingProxyType({spec.split: spec for spec in OFFICIAL_TRACK_SPLITS})


@dataclass(frozen=True, slots=True)
class OfficialAssetVerification:
    """Verified manifests, fixed package batches, and optional local training cache."""

    asset_directory: Path
    manifests: Mapping[TrackSplit, TrackAssetManifest]
    fixed_batches: Mapping[TrackSplit, TrackBatch]
    train_cache_path: Path | None
    train_cache_verified: bool

    def __post_init__(self) -> None:
        object.__setattr__(self, "asset_directory", Path(self.asset_directory))
        object.__setattr__(self, "manifests", MappingProxyType(dict(self.manifests)))
        object.__setattr__(self, "fixed_batches", MappingProxyType(dict(self.fixed_batches)))
        if self.train_cache_path is not None:
            object.__setattr__(self, "train_cache_path", Path(self.train_cache_path))


@dataclass(frozen=True, slots=True)
class TrainCacheMaterialization:
    """Summary of one verified local training-pool cache operation."""

    path: Path
    sha256: str
    track_count: int
    reused: bool


def official_track_asset_directory(
    version: str = OFFICIAL_BENCHMARK_VERSION,
    *,
    package_root: str | Path | None = None,
) -> Path:
    """Locate the versioned Track directory inside the installed/source package."""

    if version != OFFICIAL_BENCHMARK_VERSION:
        raise TrackAssetError(
            f"unsupported benchmark asset version {version!r}; expected "
            f"{OFFICIAL_BENCHMARK_VERSION!r}"
        )
    root = Path(__file__).resolve().parents[1] if package_root is None else Path(package_root)
    return root / "assets" / "tracks" / f"v{version}"


def official_track_split_spec(split: TrackSplit) -> OfficialTrackSplitSpec:
    """Return one locked official split specification."""

    try:
        return _SPLITS_BY_NAME[split]
    except KeyError as error:
        raise TrackAssetError(f"unknown official Track split: {split!r}") from error


def _level_width(config: ProjectConfig, level_id: int) -> float:
    return next(level.track_width_m for level in config.levels if level.level_id == level_id)


def _validate_project_contract(config: ProjectConfig) -> None:
    if not isinstance(config, ProjectConfig):
        raise TrackAssetError("config must be a ProjectConfig")
    benchmark = config.benchmark
    if benchmark.version != OFFICIAL_BENCHMARK_VERSION:
        raise TrackAssetError(f"project benchmark version must be {OFFICIAL_BENCHMARK_VERSION!r}")
    configured_counts = {
        "train": benchmark.train_track_count,
        "validation": benchmark.validation_track_count,
        "test": benchmark.test_track_count,
    }
    for split, count in configured_counts.items():
        if count != official_track_split_spec(split).track_count:
            raise TrackAssetError(f"project {split} Track count does not match the locked split")


def validate_official_manifest(
    config: ProjectConfig,
    manifest: TrackAssetManifest,
) -> None:
    """Validate one manifest against the project and its locked split namespace."""

    _validate_project_contract(config)
    if not isinstance(manifest, TrackAssetManifest):
        raise TrackAssetError("manifest must be a TrackAssetManifest")
    spec = official_track_split_spec(manifest.split)
    if manifest.benchmark_version != OFFICIAL_BENCHMARK_VERSION:
        raise TrackAssetError(f"{manifest.split} manifest has the wrong benchmark version")
    if manifest.level_id != spec.level_id:
        raise TrackAssetError(f"{manifest.split} manifest has the wrong Level")
    if manifest.track_count != spec.track_count:
        raise TrackAssetError(f"{manifest.split} manifest has the wrong Track count")
    if manifest.asset_file != spec.asset_file:
        raise TrackAssetError(f"{manifest.split} manifest has the wrong asset filename")
    if manifest.generator_version != config.track.generator.generator_version:
        raise TrackAssetError(f"{manifest.split} manifest has the wrong generator version")
    if manifest.geometry_validation_version != OFFICIAL_GEOMETRY_VALIDATION_VERSION:
        raise TrackAssetError(
            f"{manifest.split} manifest has the wrong geometry-validation version"
        )
    if manifest.driveability_protocol_version != OFFICIAL_DRIVEABILITY_PROTOCOL_VERSION:
        raise TrackAssetError(f"{manifest.split} manifest has the wrong driveability version")
    if manifest.capacity != track_capacity_from_project(config):
        raise TrackAssetError(f"{manifest.split} manifest has the wrong Track capacity")
    if manifest.track_width_m != _level_width(config, spec.level_id):
        raise TrackAssetError(f"{manifest.split} manifest has the wrong Track width")
    outside = [record.seed for record in manifest.tracks if not spec.contains_seed(record.seed)]
    if outside:
        raise TrackAssetError(
            f"{manifest.split} manifest contains seeds outside its locked namespace"
        )
    seeds = tuple(record.seed for record in manifest.tracks)
    if manifest.split == "level0":
        if seeds != (LEVEL0_TRACK_SEED,):
            raise TrackAssetError("the official Level 0 manifest must use its reserved singleton")
    elif any(left >= right for left, right in pairwise(seeds)):
        raise TrackAssetError(
            f"{manifest.split} manifest records must be strictly increasing by seed"
        )


def verify_manifest_disjointness(
    manifests: Mapping[TrackSplit, TrackAssetManifest],
) -> None:
    """Reject Track seeds or packed geometry shared by any two official splits."""

    seed_owner: dict[int, TrackSplit] = {}
    hash_owner: dict[str, TrackSplit] = {}
    for spec in OFFICIAL_TRACK_SPLITS:
        manifest = manifests[spec.split]
        for record in manifest.tracks:
            previous_seed = seed_owner.setdefault(record.seed, spec.split)
            if previous_seed != spec.split:
                raise TrackAssetError(
                    f"Track seed {record.seed} appears in both {previous_seed} and {spec.split}"
                )
            previous_hash = hash_owner.setdefault(record.geometry_sha256, spec.split)
            if previous_hash != spec.split:
                raise TrackAssetError(
                    f"packed Track geometry appears in both {previous_hash} and {spec.split}"
                )


def validate_official_manifest_set(
    config: ProjectConfig,
    manifests: Mapping[TrackSplit, TrackAssetManifest],
) -> None:
    """Validate the complete official manifest set and cross-split disjointness."""

    expected = set(_SPLITS_BY_NAME)
    actual = set(manifests)
    if actual != expected:
        missing = sorted(expected - actual)
        extra = sorted(actual - expected)
        details: list[str] = []
        if missing:
            details.append(f"missing: {', '.join(missing)}")
        if extra:
            details.append(f"unexpected: {', '.join(extra)}")
        raise TrackAssetError(f"official manifest set is incomplete ({'; '.join(details)})")
    for spec in OFFICIAL_TRACK_SPLITS:
        manifest = manifests[spec.split]
        if manifest.split != spec.split:
            raise TrackAssetError(f"manifest mapping key {spec.split!r} does not match its value")
        validate_official_manifest(config, manifest)
    verify_manifest_disjointness(manifests)


def _verify_batch_against_manifest(
    manifest: TrackAssetManifest,
    batch: TrackBatch,
) -> None:
    capacity = validate_track_batch(batch)
    if capacity != manifest.capacity:
        raise TrackAssetError("Track asset capacity does not match the manifest")
    if batch.seed.shape[0] != manifest.track_count:
        raise TrackAssetError("Track asset count does not match the manifest")
    expected_seeds = np.asarray([record.seed for record in manifest.tracks], dtype=np.uint32)
    if not np.array_equal(batch.seed, expected_seeds):
        raise TrackAssetError("Track asset seed order does not match the manifest")
    expected_hashes = tuple(record.geometry_sha256 for record in manifest.tracks)
    if track_batch_geometry_sha256(batch) != expected_hashes:
        raise TrackAssetError("Track asset geometry hashes do not match the manifest")
    if not np.all(batch.width_m == np.float32(manifest.track_width_m)):
        raise TrackAssetError("Track asset width does not match the manifest")


def load_verified_manifest_batch(
    manifest: TrackAssetManifest,
    asset_path: str | Path,
) -> TrackBatch:
    """Load an asset from an explicit path and verify every manifest field."""

    batch = load_track_batch_npz(
        asset_path,
        expected_sha256=manifest.asset_sha256,
        expected_track_count=manifest.track_count,
        expected_capacity=manifest.capacity,
    )
    _verify_batch_against_manifest(manifest, batch)
    return batch


def verify_official_track_assets(
    config: ProjectConfig,
    *,
    asset_directory: str | Path | None = None,
    train_cache_path: str | Path | None = None,
    require_train_cache: bool = False,
) -> OfficialAssetVerification:
    """Verify every official manifest, fixed package asset, and optional training cache."""

    directory = (
        official_track_asset_directory(config.benchmark.version)
        if asset_directory is None
        else Path(asset_directory)
    )
    manifests: dict[TrackSplit, TrackAssetManifest] = {}
    for spec in OFFICIAL_TRACK_SPLITS:
        manifests[spec.split] = load_track_asset_manifest(directory / spec.manifest_file)
    validate_official_manifest_set(config, manifests)

    fixed_batches: dict[TrackSplit, TrackBatch] = {}
    for spec in OFFICIAL_TRACK_SPLITS:
        if not spec.package_asset:
            continue
        loaded_manifest, batch = load_manifest_track_batch(directory / spec.manifest_file)
        if loaded_manifest != manifests[spec.split]:
            raise TrackAssetError(f"{spec.split} manifest changed during verification")
        fixed_batches[spec.split] = batch

    cache_path = None if train_cache_path is None else Path(train_cache_path)
    train_cache_verified = False
    if cache_path is not None:
        if cache_path.exists():
            load_verified_manifest_batch(manifests["train"], cache_path)
            train_cache_verified = True
        elif require_train_cache:
            raise TrackAssetError(f"training Track cache does not exist: {cache_path}")
    elif require_train_cache:
        raise TrackAssetError("require_train_cache needs an explicit training cache path")

    return OfficialAssetVerification(
        asset_directory=directory,
        manifests=manifests,
        fixed_batches=fixed_batches,
        train_cache_path=cache_path,
        train_cache_verified=train_cache_verified,
    )


def _track_row(track: Track, field: str) -> np.ndarray:
    if field == "seed":
        return np.asarray(track.seed, dtype=np.uint32)
    if field in ("point_count", "checkpoint_count"):
        return np.asarray(getattr(track, field), dtype=np.int32)
    if field in ("length_m", "width_m"):
        return np.asarray(getattr(track, field), dtype=np.float32)
    return np.asarray(getattr(track, field))


def _allocate_track_batch(track_count: int, first_track: Track) -> dict[str, np.ndarray]:
    arrays: dict[str, np.ndarray] = {}
    for field in TrackBatch._fields:
        row = _track_row(first_track, field)
        arrays[field] = np.empty((track_count, *row.shape), dtype=row.dtype)
    return arrays


def regenerate_manifest_track_batch(
    config: ProjectConfig,
    manifest: TrackAssetManifest,
    *,
    progress: ProgressCallback | None = None,
) -> TrackBatch:
    """Regenerate and geometrically revalidate a Level 1 manifest in record order.

    Driveability admission is intentionally not repeated here: it belongs to the formal GPU
    admission protocol recorded by the manifest. This function proves that the CPU generator,
    geometry validator, packing, and canonical hashes reproduce the admitted geometry exactly.
    """

    _validate_project_contract(config)
    if manifest.level_id != 1 or manifest.split not in ("train", "validation", "test"):
        raise TrackAssetError("only Level 1 manifests can be regenerated procedurally")
    if manifest.benchmark_version != config.benchmark.version:
        raise TrackAssetError("manifest benchmark version does not match the project")
    if manifest.generator_version != config.track.generator.generator_version:
        raise TrackAssetError("manifest generator version does not match the project")
    if manifest.capacity != track_capacity_from_project(config):
        raise TrackAssetError("manifest Track capacity does not match the project")
    if manifest.track_width_m != _level_width(config, 1):
        raise TrackAssetError("manifest Track width does not match the project")

    generation_spec = generation_spec_from_project(config)
    validation_spec = validation_spec_from_project(config)
    capacity = track_capacity_from_project(config)
    arrays: dict[str, np.ndarray] | None = None
    total = manifest.track_count
    for index, record in enumerate(manifest.tracks):
        try:
            candidate = generate_track_candidate(record.seed, generation_spec)
        except TrackGenerationError as error:
            raise TrackAssetError(
                f"{manifest.split} seed {record.seed} failed deterministic generation: "
                f"{error.reason}"
            ) from error
        validation = validate_track_candidate(candidate, validation_spec)
        if not validation.valid:
            reasons = ", ".join(validation.reasons)
            raise TrackAssetError(
                f"{manifest.split} seed {record.seed} failed geometry validation: {reasons}"
            )
        try:
            track = pack_track(candidate, capacity)
        except TrackGenerationError as error:
            raise TrackAssetError(
                f"{manifest.split} seed {record.seed} failed fixed-capacity packing: {error.reason}"
            ) from error
        actual_hash = track_geometry_sha256(track)
        if actual_hash != record.geometry_sha256:
            raise TrackAssetError(
                f"{manifest.split} seed {record.seed} geometry hash does not match the manifest"
            )
        if arrays is None:
            arrays = _allocate_track_batch(total, track)
        for field in TrackBatch._fields:
            arrays[field][index] = _track_row(track, field)
        if progress is not None:
            progress(index + 1, total, record.seed)

    if arrays is None:  # TrackAssetManifest already forbids this; keep the allocation total.
        raise TrackAssetError("cannot regenerate an empty Track manifest")
    batch = TrackBatch(**arrays)
    _verify_batch_against_manifest(manifest, batch)
    for array in batch:
        array.setflags(write=False)
    return batch


def write_verified_manifest_cache(
    batch: TrackBatch,
    manifest: TrackAssetManifest,
    output: str | Path,
) -> str:
    """Atomically replace a cache only after full NPZ and manifest verification."""

    destination = Path(output)
    if destination.suffix != ".npz":
        raise TrackAssetError("training Track cache path must use the .npz suffix")
    _verify_batch_against_manifest(manifest, batch)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            prefix=f".{destination.stem}.",
            suffix=".npz",
            dir=destination.parent,
            delete=False,
        ) as file:
            temporary = Path(file.name)
        actual_sha256 = save_track_batch_npz(batch, temporary)
        if actual_sha256 != manifest.asset_sha256:
            raise TrackAssetError("regenerated training cache SHA-256 does not match the manifest")
        load_verified_manifest_batch(manifest, temporary)
        os.replace(temporary, destination)
        temporary = None
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)
    # The fully verified temporary file is renamed atomically. Re-hash the destination to catch
    # an I/O fault without allocating a second 260 MiB TrackBatch after the rename.
    if sha256_file(destination) != manifest.asset_sha256:
        raise TrackAssetError("materialized training cache SHA-256 changed after atomic replace")
    return manifest.asset_sha256


def materialize_official_train_cache(
    config: ProjectConfig,
    *,
    asset_directory: str | Path | None = None,
    output: str | Path = DEFAULT_TRAIN_CACHE,
    force: bool = False,
    progress: ProgressCallback | None = None,
) -> TrainCacheMaterialization:
    """Reproduce the official training pool or verify and reuse an existing cache."""

    verification = verify_official_track_assets(config, asset_directory=asset_directory)
    manifest = verification.manifests["train"]
    destination = Path(output)
    if destination.exists() and not force:
        load_verified_manifest_batch(manifest, destination)
        return TrainCacheMaterialization(
            path=destination,
            sha256=manifest.asset_sha256,
            track_count=manifest.track_count,
            reused=True,
        )
    batch = regenerate_manifest_track_batch(config, manifest, progress=progress)
    digest = write_verified_manifest_cache(batch, manifest, destination)
    return TrainCacheMaterialization(
        path=destination,
        sha256=digest,
        track_count=manifest.track_count,
        reused=False,
    )


__all__ = [
    "DEFAULT_TRAIN_CACHE",
    "OFFICIAL_BENCHMARK_VERSION",
    "OFFICIAL_DRIVEABILITY_PROTOCOL_VERSION",
    "OFFICIAL_GEOMETRY_VALIDATION_VERSION",
    "OFFICIAL_TRACK_SPLITS",
    "OfficialAssetVerification",
    "OfficialTrackSplitSpec",
    "ProgressCallback",
    "TrainCacheMaterialization",
    "load_verified_manifest_batch",
    "materialize_official_train_cache",
    "official_track_asset_directory",
    "official_track_split_spec",
    "regenerate_manifest_track_batch",
    "validate_official_manifest",
    "validate_official_manifest_set",
    "verify_manifest_disjointness",
    "verify_official_track_assets",
    "write_verified_manifest_cache",
]
