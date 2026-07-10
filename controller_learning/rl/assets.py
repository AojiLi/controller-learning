"""Train-only official TrackPool loading for PPO optimization."""

from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import numpy as np

from controller_learning.config import ProjectConfig
from controller_learning.tracks.assets import (
    TrackAssetError,
    TrackAssetManifest,
    load_track_asset_manifest,
    sha256_file,
)
from controller_learning.tracks.hashing import track_batch_geometry_sha256
from controller_learning.tracks.official_assets import (
    DEFAULT_TRAIN_CACHE,
    load_verified_manifest_batch,
    official_track_asset_directory,
    official_track_split_spec,
    validate_official_manifest,
)
from controller_learning.tracks.pool import TrackPool
from controller_learning.tracks.specs import track_capacity_from_project
from controller_learning.tracks.types import TrackCapacity

TRAIN_POOL_ACCESS_SCHEMA_VERSION = "controller-learning.m7-train-pool-access.v1"
_SHA256_PATTERN = re.compile(r"^[0-9a-f]{64}$")


def _sha256_lines(values: tuple[str, ...]) -> str:
    digest = hashlib.sha256()
    for value in values:
        digest.update(value.encode("ascii"))
        digest.update(b"\n")
    return digest.hexdigest()


@dataclass(frozen=True, slots=True)
class TrainPoolAccessEvidence:
    """Path-sanitized proof of one Train-only loader invocation."""

    schema_version: str
    loaded_splits: tuple[str, ...]
    benchmark_version: str
    generator_version: str
    level_id: int
    split: Literal["train"]
    manifest_file: str
    manifest_sha256: str
    cache_file: str
    manifest_asset_sha256: str
    cache_file_sha256: str
    track_count: int
    capacity: TrackCapacity
    first_track_id: int
    last_track_id: int
    track_ids_sha256: str
    geometry_hashes_sha256: str
    loader_accessed_validation: bool
    loader_accessed_test: bool

    def __post_init__(self) -> None:
        if self.schema_version != TRAIN_POOL_ACCESS_SCHEMA_VERSION:
            raise ValueError("unexpected Train-pool access evidence schema")
        if self.loaded_splits != ("train",) or self.split != "train":
            raise ValueError("Train-pool access evidence must contain only the Train split")
        if not isinstance(self.benchmark_version, str) or not self.benchmark_version:
            raise ValueError("benchmark_version must be a non-empty string")
        if not isinstance(self.generator_version, str) or not self.generator_version:
            raise ValueError("generator_version must be a non-empty string")
        if type(self.level_id) is not int or self.level_id != 1:
            raise ValueError("Train-pool access evidence must identify Level 1")
        for field in ("manifest_file", "cache_file"):
            value = getattr(self, field)
            if not isinstance(value, str) or not value or Path(value).name != value:
                raise ValueError(f"{field} must be a path-sanitized filename")
        for field in (
            "manifest_sha256",
            "manifest_asset_sha256",
            "cache_file_sha256",
            "track_ids_sha256",
            "geometry_hashes_sha256",
        ):
            value = getattr(self, field)
            if not isinstance(value, str) or _SHA256_PATTERN.fullmatch(value) is None:
                raise ValueError(f"{field} must be a lowercase SHA-256 digest")
        if self.manifest_asset_sha256 != self.cache_file_sha256:
            raise ValueError("cache_file_sha256 must match the manifest asset digest")
        if type(self.track_count) is not int or self.track_count < 1:
            raise ValueError("track_count must be a positive integer")
        if not isinstance(self.capacity, TrackCapacity):
            raise TypeError("capacity must be a TrackCapacity")
        for field in ("first_track_id", "last_track_id"):
            value = getattr(self, field)
            if type(value) is not int or not 0 <= value < 2**32:
                raise ValueError(f"{field} must fit in uint32")
        if self.first_track_id > self.last_track_id:
            raise ValueError("Train Track IDs must retain manifest order")
        if self.loader_accessed_validation is not False or self.loader_accessed_test is not False:
            raise ValueError("this Train-only loader invocation cannot claim another split")


@dataclass(frozen=True, slots=True)
class VerifiedTrainPool:
    """One immutable official Train TrackPool and its access evidence."""

    pool: TrackPool
    evidence: TrainPoolAccessEvidence

    def __post_init__(self) -> None:
        if not isinstance(self.pool, TrackPool) or self.pool.split != "train":
            raise TypeError("pool must be an immutable Train TrackPool")
        if not isinstance(self.evidence, TrainPoolAccessEvidence):
            raise TypeError("evidence must be TrainPoolAccessEvidence")
        if self.pool.size != self.evidence.track_count:
            raise ValueError("Train TrackPool size must match its access evidence")
        if self.pool.capacity != self.evidence.capacity:
            raise ValueError("Train TrackPool capacity must match its access evidence")
        if self.pool.benchmark_version != self.evidence.benchmark_version:
            raise ValueError("Train TrackPool benchmark version must match its access evidence")
        if self.pool.generator_version != self.evidence.generator_version:
            raise ValueError("Train TrackPool generator version must match its access evidence")

        track_ids = tuple(str(int(value)) for value in self.pool.batch.seed)
        if int(self.pool.batch.seed[0]) != self.evidence.first_track_id:
            raise ValueError("Train TrackPool first Track ID must match its access evidence")
        if int(self.pool.batch.seed[-1]) != self.evidence.last_track_id:
            raise ValueError("Train TrackPool last Track ID must match its access evidence")
        if _sha256_lines(track_ids) != self.evidence.track_ids_sha256:
            raise ValueError("Train TrackPool IDs must match their access evidence")

        geometry_hashes = track_batch_geometry_sha256(self.pool.batch)
        if _sha256_lines(geometry_hashes) != self.evidence.geometry_hashes_sha256:
            raise ValueError("Train TrackPool geometry must match its access evidence")


def _load_stable_manifest(path: Path) -> tuple[TrackAssetManifest, str]:
    """Load one manifest only when its bytes remain stable across parsing."""

    try:
        digest_before = sha256_file(path)
    except FileNotFoundError as error:
        raise TrackAssetError(f"Train manifest does not exist: {path}") from error
    manifest = load_track_asset_manifest(path)
    try:
        digest_after = sha256_file(path)
    except FileNotFoundError as error:
        raise TrackAssetError("Train manifest changed while it was being loaded") from error
    if digest_before != digest_after:
        raise TrackAssetError("Train manifest changed while it was being loaded")
    return manifest, digest_before


def load_verified_train_pool(
    project_config: ProjectConfig,
    *,
    train_cache_path: str | Path = DEFAULT_TRAIN_CACHE,
    asset_directory: str | Path | None = None,
) -> VerifiedTrainPool:
    """Load only the official Train manifest and verified local Train NPZ cache.

    This function deliberately does not call the all-split asset verifier. Validation selection
    has a separate M7 phase, and Test geometry remains inaccessible until M8.
    """

    if not isinstance(project_config, ProjectConfig):
        raise TypeError("project_config must be a ProjectConfig")
    directory = (
        official_track_asset_directory(project_config.benchmark.version)
        if asset_directory is None
        else Path(asset_directory)
    )
    cache_path = Path(train_cache_path)
    spec = official_track_split_spec("train")
    if cache_path.name != spec.asset_file:
        raise TrackAssetError(
            "training Track cache filename must match the official Train asset filename"
        )
    manifest_path = directory / spec.manifest_file
    manifest, manifest_digest = _load_stable_manifest(manifest_path)
    validate_official_manifest(project_config, manifest)

    if manifest.split != "train" or manifest.level_id != 1:
        raise TrackAssetError("PPO optimization requires the official Level 1 Train manifest")
    if manifest.track_count != spec.track_count:
        raise TrackAssetError(f"Train manifest must contain exactly {spec.track_count} Tracks")
    if manifest.track_count != project_config.benchmark.train_track_count:
        raise TrackAssetError("Train manifest count does not match project_config")
    if manifest.capacity != track_capacity_from_project(project_config):
        raise TrackAssetError("Train manifest capacity does not match project_config")
    if manifest.generator_version != project_config.track.generator.generator_version:
        raise TrackAssetError("Train manifest generator version does not match project_config")
    if manifest.asset_file != spec.asset_file:
        raise TrackAssetError("Train manifest asset filename does not match the official split")

    try:
        cache_digest = sha256_file(cache_path)
    except FileNotFoundError as error:
        raise TrackAssetError(f"training Track cache does not exist: {cache_path}") from error
    if cache_digest != manifest.asset_sha256:
        raise TrackAssetError("training Track cache SHA-256 does not match the Train manifest")
    batch = load_verified_manifest_batch(manifest, cache_path)
    try:
        cache_digest_after = sha256_file(cache_path)
    except FileNotFoundError as error:
        raise TrackAssetError("training Track cache changed while it was being loaded") from error
    if cache_digest != cache_digest_after:
        raise TrackAssetError("training Track cache changed while it was being loaded")

    expected_track_ids = np.asarray(
        tuple(record.seed for record in manifest.tracks),
        dtype=np.uint32,
    )
    if batch.seed.shape != (manifest.track_count,):
        raise TrackAssetError("Train cache Track count does not match the Train manifest")
    if not np.array_equal(batch.seed, expected_track_ids):
        raise TrackAssetError("Train cache Track order does not match the Train manifest")

    pool = TrackPool(
        benchmark_version=manifest.benchmark_version,
        generator_version=manifest.generator_version,
        split="train",
        batch=batch,
    )
    evidence = TrainPoolAccessEvidence(
        schema_version=TRAIN_POOL_ACCESS_SCHEMA_VERSION,
        loaded_splits=("train",),
        benchmark_version=manifest.benchmark_version,
        generator_version=manifest.generator_version,
        level_id=manifest.level_id,
        split="train",
        manifest_file=manifest_path.name,
        manifest_sha256=manifest_digest,
        cache_file=cache_path.name,
        manifest_asset_sha256=manifest.asset_sha256,
        cache_file_sha256=cache_digest,
        track_count=manifest.track_count,
        capacity=manifest.capacity,
        first_track_id=int(expected_track_ids[0]),
        last_track_id=int(expected_track_ids[-1]),
        track_ids_sha256=_sha256_lines(tuple(str(int(value)) for value in expected_track_ids)),
        geometry_hashes_sha256=_sha256_lines(
            tuple(record.geometry_sha256 for record in manifest.tracks)
        ),
        loader_accessed_validation=False,
        loader_accessed_test=False,
    )
    return VerifiedTrainPool(pool=pool, evidence=evidence)


__all__ = [
    "TRAIN_POOL_ACCESS_SCHEMA_VERSION",
    "TrainPoolAccessEvidence",
    "VerifiedTrainPool",
    "load_verified_train_pool",
]
