"""Validation-only official TrackPool loading for M7 checkpoint selection."""

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
    OFFICIAL_BENCHMARK_VERSION,
    load_verified_manifest_batch,
    official_track_asset_directory,
    official_track_split_spec,
    validate_official_manifest,
)
from controller_learning.tracks.pool import TrackPool
from controller_learning.tracks.specs import track_capacity_from_project
from controller_learning.tracks.types import TrackCapacity

VALIDATION_POOL_ACCESS_SCHEMA_VERSION = "controller-learning.m7-validation-pool-access.v1"
FORMAL_VALIDATION_TRACK_COUNT = 100
_SHA256_PATTERN = re.compile(r"^[0-9a-f]{64}$")


def _sha256_lines(values: tuple[str, ...]) -> str:
    digest = hashlib.sha256()
    for value in values:
        digest.update(value.encode("ascii"))
        digest.update(b"\n")
    return digest.hexdigest()


@dataclass(frozen=True, slots=True)
class ValidationPoolAccessEvidence:
    """Path-sanitized proof of one Validation-only loader invocation."""

    schema_version: str
    loaded_splits: tuple[str, ...]
    benchmark_version: str
    generator_version: str
    level_id: int
    split: Literal["validation"]
    manifest_file: str
    manifest_sha256: str
    asset_file: str
    manifest_asset_sha256: str
    asset_file_sha256: str
    track_count: int
    capacity: TrackCapacity
    first_track_id: int
    last_track_id: int
    track_ids_sha256: str
    geometry_hashes_sha256: str
    loader_accessed_train: bool
    loader_accessed_test: bool

    def __post_init__(self) -> None:
        if self.schema_version != VALIDATION_POOL_ACCESS_SCHEMA_VERSION:
            raise ValueError("unexpected Validation-pool access evidence schema")
        if self.loaded_splits != ("validation",) or self.split != "validation":
            raise ValueError("Validation-pool evidence must contain only Validation")
        if self.benchmark_version != OFFICIAL_BENCHMARK_VERSION:
            raise ValueError("Validation-pool evidence must identify benchmark 0.1")
        if not isinstance(self.generator_version, str) or not self.generator_version:
            raise ValueError("generator_version must be a non-empty string")
        if type(self.level_id) is not int or self.level_id != 1:
            raise ValueError("Validation-pool evidence must identify Level 1")
        for field in ("manifest_file", "asset_file"):
            value = getattr(self, field)
            if not isinstance(value, str) or not value or Path(value).name != value:
                raise ValueError(f"{field} must be a path-sanitized filename")
        for field in (
            "manifest_sha256",
            "manifest_asset_sha256",
            "asset_file_sha256",
            "track_ids_sha256",
            "geometry_hashes_sha256",
        ):
            value = getattr(self, field)
            if not isinstance(value, str) or _SHA256_PATTERN.fullmatch(value) is None:
                raise ValueError(f"{field} must be a lowercase SHA-256 digest")
        if self.manifest_asset_sha256 != self.asset_file_sha256:
            raise ValueError("asset_file_sha256 must match the manifest asset digest")
        if type(self.track_count) is not int or self.track_count != FORMAL_VALIDATION_TRACK_COUNT:
            raise ValueError(
                f"Validation-pool evidence must contain {FORMAL_VALIDATION_TRACK_COUNT} Tracks"
            )
        if not isinstance(self.capacity, TrackCapacity):
            raise TypeError("capacity must be a TrackCapacity")
        for field in ("first_track_id", "last_track_id"):
            value = getattr(self, field)
            if type(value) is not int or not 0 <= value < 2**32:
                raise ValueError(f"{field} must fit in uint32")
        if self.first_track_id >= self.last_track_id:
            raise ValueError("Validation Track IDs must retain strict manifest order")
        if self.loader_accessed_train is not False or self.loader_accessed_test is not False:
            raise ValueError("this Validation-only loader cannot claim another split")


@dataclass(frozen=True, slots=True)
class VerifiedValidationPool:
    """One immutable official Validation TrackPool and its access evidence."""

    pool: TrackPool
    evidence: ValidationPoolAccessEvidence

    def __post_init__(self) -> None:
        if not isinstance(self.pool, TrackPool) or self.pool.split != "validation":
            raise TypeError("pool must be an immutable Validation TrackPool")
        if not isinstance(self.evidence, ValidationPoolAccessEvidence):
            raise TypeError("evidence must be ValidationPoolAccessEvidence")
        if self.pool.size != self.evidence.track_count:
            raise ValueError("Validation TrackPool size differs from its evidence")
        if self.pool.capacity != self.evidence.capacity:
            raise ValueError("Validation TrackPool capacity differs from its evidence")
        if self.pool.benchmark_version != self.evidence.benchmark_version:
            raise ValueError("Validation TrackPool benchmark differs from its evidence")
        if self.pool.generator_version != self.evidence.generator_version:
            raise ValueError("Validation TrackPool generator differs from its evidence")

        track_ids = tuple(str(int(value)) for value in self.pool.batch.seed)
        if int(self.pool.batch.seed[0]) != self.evidence.first_track_id:
            raise ValueError("Validation first Track ID differs from its evidence")
        if int(self.pool.batch.seed[-1]) != self.evidence.last_track_id:
            raise ValueError("Validation last Track ID differs from its evidence")
        if _sha256_lines(track_ids) != self.evidence.track_ids_sha256:
            raise ValueError("Validation Track IDs differ from their evidence")
        geometry_hashes = track_batch_geometry_sha256(self.pool.batch)
        if _sha256_lines(geometry_hashes) != self.evidence.geometry_hashes_sha256:
            raise ValueError("Validation geometry differs from its evidence")


def _load_stable_manifest(path: Path) -> tuple[TrackAssetManifest, str]:
    try:
        digest_before = sha256_file(path)
    except FileNotFoundError as error:
        raise TrackAssetError(f"Validation manifest does not exist: {path}") from error
    manifest = load_track_asset_manifest(path)
    try:
        digest_after = sha256_file(path)
    except FileNotFoundError as error:
        raise TrackAssetError("Validation manifest changed while loading") from error
    if digest_before != digest_after:
        raise TrackAssetError("Validation manifest changed while loading")
    return manifest, digest_before


def load_verified_validation_pool(
    project_config: ProjectConfig,
    *,
    asset_directory: str | Path | None = None,
) -> VerifiedValidationPool:
    """Load only the official benchmark-0.1 Level-1 Validation manifest and NPZ.

    The path is derived from the locked Validation split contract.  This function never calls the
    all-split verifier and has no parameter through which a Train or Test asset can be selected.
    """

    if not isinstance(project_config, ProjectConfig):
        raise TypeError("project_config must be a ProjectConfig")
    if project_config.benchmark.version != OFFICIAL_BENCHMARK_VERSION:
        raise TrackAssetError("Validation selection requires benchmark version 0.1")
    directory = (
        official_track_asset_directory(project_config.benchmark.version)
        if asset_directory is None
        else Path(asset_directory)
    )
    spec = official_track_split_spec("validation")
    if (
        spec.split != "validation"
        or spec.level_id != 1
        or spec.track_count != FORMAL_VALIDATION_TRACK_COUNT
        or spec.manifest_file != "validation.json"
        or spec.asset_file != "validation.npz"
        or spec.package_asset is not True
    ):
        raise TrackAssetError("official Validation split contract is not the locked v0.1 contract")

    manifest_path = directory / spec.manifest_file
    asset_path = directory / spec.asset_file
    manifest, manifest_digest = _load_stable_manifest(manifest_path)
    validate_official_manifest(project_config, manifest)
    if (
        manifest.benchmark_version != OFFICIAL_BENCHMARK_VERSION
        or manifest.split != "validation"
        or manifest.level_id != 1
        or manifest.track_count != FORMAL_VALIDATION_TRACK_COUNT
        or manifest.asset_file != spec.asset_file
    ):
        raise TrackAssetError("checkpoint selection requires the official v0.1 Validation manifest")
    if manifest.track_count != project_config.benchmark.validation_track_count:
        raise TrackAssetError("Validation manifest count differs from project_config")
    if manifest.capacity != track_capacity_from_project(project_config):
        raise TrackAssetError("Validation manifest capacity differs from project_config")
    if manifest.generator_version != project_config.track.generator.generator_version:
        raise TrackAssetError("Validation manifest generator differs from project_config")

    try:
        asset_digest = sha256_file(asset_path)
    except FileNotFoundError as error:
        raise TrackAssetError(f"Validation NPZ does not exist: {asset_path}") from error
    if asset_digest != manifest.asset_sha256:
        raise TrackAssetError("Validation NPZ SHA-256 differs from the manifest")
    batch = load_verified_manifest_batch(manifest, asset_path)
    try:
        asset_digest_after = sha256_file(asset_path)
    except FileNotFoundError as error:
        raise TrackAssetError("Validation NPZ changed while loading") from error
    if asset_digest != asset_digest_after:
        raise TrackAssetError("Validation NPZ changed while loading")

    expected_track_ids = np.asarray(
        tuple(record.seed for record in manifest.tracks),
        dtype=np.uint32,
    )
    if batch.seed.shape != (FORMAL_VALIDATION_TRACK_COUNT,):
        raise TrackAssetError("Validation NPZ does not contain exactly 100 Tracks")
    if not np.array_equal(batch.seed, expected_track_ids):
        raise TrackAssetError("Validation NPZ Track order differs from the manifest")

    pool = TrackPool(
        benchmark_version=manifest.benchmark_version,
        generator_version=manifest.generator_version,
        split="validation",
        batch=batch,
    )
    evidence = ValidationPoolAccessEvidence(
        schema_version=VALIDATION_POOL_ACCESS_SCHEMA_VERSION,
        loaded_splits=("validation",),
        benchmark_version=manifest.benchmark_version,
        generator_version=manifest.generator_version,
        level_id=manifest.level_id,
        split="validation",
        manifest_file=manifest_path.name,
        manifest_sha256=manifest_digest,
        asset_file=asset_path.name,
        manifest_asset_sha256=manifest.asset_sha256,
        asset_file_sha256=asset_digest,
        track_count=manifest.track_count,
        capacity=manifest.capacity,
        first_track_id=int(expected_track_ids[0]),
        last_track_id=int(expected_track_ids[-1]),
        track_ids_sha256=_sha256_lines(tuple(str(int(value)) for value in expected_track_ids)),
        geometry_hashes_sha256=_sha256_lines(
            tuple(record.geometry_sha256 for record in manifest.tracks)
        ),
        loader_accessed_train=False,
        loader_accessed_test=False,
    )
    return VerifiedValidationPool(pool=pool, evidence=evidence)


__all__ = [
    "FORMAL_VALIDATION_TRACK_COUNT",
    "VALIDATION_POOL_ACCESS_SCHEMA_VERSION",
    "ValidationPoolAccessEvidence",
    "VerifiedValidationPool",
    "load_verified_validation_pool",
]
