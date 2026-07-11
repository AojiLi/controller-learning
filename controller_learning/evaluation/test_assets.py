"""Test-only official TrackPool loading for the frozen M8 evaluation."""

from __future__ import annotations

import hashlib
import re
from collections.abc import Mapping
from dataclasses import dataclass
from itertools import pairwise
from pathlib import Path
from typing import Any, Literal

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
    OfficialTrackSplitSpec,
    load_verified_manifest_batch,
    official_track_asset_directory,
    official_track_split_spec,
    validate_official_manifest,
)
from controller_learning.tracks.pool import TrackPool
from controller_learning.tracks.specs import track_capacity_from_project
from controller_learning.tracks.types import TrackCapacity

M8_TEST_POOL_ACCESS_SCHEMA_VERSION = "controller-learning.m8-test-pool-access.v1"
FORMAL_TEST_TRACK_COUNT = 20
FORMAL_TEST_MANIFEST_SHA256 = "2230e29f3e13029d4ca09de32a703e9a80c070e654386563b9ef4f7a2d197f8b"
FORMAL_TEST_ASSET_SHA256 = "0d654395630ec0b64952b076a2595de96f3926ea208fac3796a50be37df29c71"
_TEST_SEED_START = 2_000_000
_TEST_SEED_STOP = 3_000_000
_SHA256_PATTERN = re.compile(r"^[0-9a-f]{64}$")


def _sha256_lines(values: tuple[str, ...]) -> str:
    digest = hashlib.sha256()
    for value in values:
        digest.update(value.encode("ascii"))
        digest.update(b"\n")
    return digest.hexdigest()


@dataclass(frozen=True, slots=True)
class TestPoolAccessEvidence:
    """Path-sanitized proof of one Test-only TrackPool load."""

    schema_version: str
    loaded_splits: tuple[str, ...]
    benchmark_version: str
    generator_version: str
    level_id: int
    split: Literal["test"]
    manifest_file: str
    manifest_sha256: str
    asset_file: str
    manifest_asset_sha256: str
    asset_file_sha256: str
    track_count: int
    capacity: TrackCapacity
    track_ids: tuple[int, ...]
    track_ids_sha256: str
    geometry_hashes_sha256: str
    loader_accessed_train: bool
    loader_accessed_validation: bool

    def __post_init__(self) -> None:
        if self.schema_version != M8_TEST_POOL_ACCESS_SCHEMA_VERSION:
            raise ValueError("unexpected M8 Test-pool access evidence schema")
        if self.loaded_splits != ("test",) or self.split != "test":
            raise ValueError("Test-pool access evidence must contain only the Test split")
        if self.benchmark_version != OFFICIAL_BENCHMARK_VERSION:
            raise ValueError("Test-pool access evidence must identify benchmark 0.1")
        if not isinstance(self.generator_version, str) or not self.generator_version:
            raise ValueError("generator_version must be a non-empty string")
        if type(self.level_id) is not int or self.level_id != 1:
            raise ValueError("Test-pool access evidence must identify Level 1")
        if self.manifest_file != "test.json":
            raise ValueError("manifest_file must be the path-sanitized Test filename")
        if self.asset_file != "test.npz":
            raise ValueError("asset_file must be the path-sanitized Test filename")
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
        if type(self.track_count) is not int or self.track_count != FORMAL_TEST_TRACK_COUNT:
            raise ValueError(
                f"Test-pool access evidence must contain {FORMAL_TEST_TRACK_COUNT} Tracks"
            )
        if not isinstance(self.capacity, TrackCapacity):
            raise TypeError("capacity must be a TrackCapacity")
        if type(self.track_ids) is not tuple or len(self.track_ids) != self.track_count:
            raise ValueError("track_ids must contain all Test Track IDs in manifest order")
        if any(
            type(track_id) is not int or not _TEST_SEED_START <= track_id < _TEST_SEED_STOP
            for track_id in self.track_ids
        ):
            raise ValueError("Test Track IDs must fit the locked Test seed namespace")
        if any(left >= right for left, right in pairwise(self.track_ids)):
            raise ValueError("Test Track IDs must retain strict manifest order")
        expected_track_ids_digest = _sha256_lines(tuple(map(str, self.track_ids)))
        if self.track_ids_sha256 != expected_track_ids_digest:
            raise ValueError("track_ids_sha256 must bind the ordered Test Track IDs")
        if self.loader_accessed_train is not False:
            raise ValueError("the Test-only loader cannot claim Train access")
        if self.loader_accessed_validation is not False:
            raise ValueError("the Test-only loader cannot claim Validation access")

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> TestPoolAccessEvidence:
        """Restore and validate the exact public Test-pool evidence mapping."""

        if not isinstance(value, Mapping):
            raise TypeError("Test-pool access evidence must be a mapping")
        expected = {
            "asset_file",
            "asset_file_sha256",
            "benchmark_version",
            "capacity",
            "generator_version",
            "geometry_hashes_sha256",
            "level_id",
            "loaded_splits",
            "loader_accessed_train",
            "loader_accessed_validation",
            "manifest_asset_sha256",
            "manifest_file",
            "manifest_sha256",
            "schema_version",
            "split",
            "track_count",
            "track_ids",
            "track_ids_sha256",
        }
        if any(type(key) is not str for key in value) or set(value) != expected:
            raise ValueError("Test-pool access evidence mapping keys differ")
        capacity = value["capacity"]
        if not isinstance(capacity, Mapping) or set(capacity) != {
            "max_checkpoints",
            "max_track_points",
        }:
            raise ValueError("Test-pool capacity mapping differs")
        loaded_splits = value["loaded_splits"]
        track_ids = value["track_ids"]
        if not isinstance(loaded_splits, list) or not isinstance(track_ids, list):
            raise ValueError("Test-pool ordered values must use JSON arrays")
        return cls(
            schema_version=value["schema_version"],
            loaded_splits=tuple(loaded_splits),
            benchmark_version=value["benchmark_version"],
            generator_version=value["generator_version"],
            level_id=value["level_id"],
            split=value["split"],
            manifest_file=value["manifest_file"],
            manifest_sha256=value["manifest_sha256"],
            asset_file=value["asset_file"],
            manifest_asset_sha256=value["manifest_asset_sha256"],
            asset_file_sha256=value["asset_file_sha256"],
            track_count=value["track_count"],
            capacity=TrackCapacity(
                max_track_points=capacity["max_track_points"],
                max_checkpoints=capacity["max_checkpoints"],
            ),
            track_ids=tuple(track_ids),
            track_ids_sha256=value["track_ids_sha256"],
            geometry_hashes_sha256=value["geometry_hashes_sha256"],
            loader_accessed_train=value["loader_accessed_train"],
            loader_accessed_validation=value["loader_accessed_validation"],
        )

    def to_dict(self) -> dict[str, Any]:
        """Return the exact JSON-compatible public evidence mapping."""

        return {
            "asset_file": self.asset_file,
            "asset_file_sha256": self.asset_file_sha256,
            "benchmark_version": self.benchmark_version,
            "capacity": {
                "max_checkpoints": self.capacity.max_checkpoints,
                "max_track_points": self.capacity.max_track_points,
            },
            "generator_version": self.generator_version,
            "geometry_hashes_sha256": self.geometry_hashes_sha256,
            "level_id": self.level_id,
            "loaded_splits": list(self.loaded_splits),
            "loader_accessed_train": self.loader_accessed_train,
            "loader_accessed_validation": self.loader_accessed_validation,
            "manifest_asset_sha256": self.manifest_asset_sha256,
            "manifest_file": self.manifest_file,
            "manifest_sha256": self.manifest_sha256,
            "schema_version": self.schema_version,
            "split": self.split,
            "track_count": self.track_count,
            "track_ids": list(self.track_ids),
            "track_ids_sha256": self.track_ids_sha256,
        }


@dataclass(frozen=True, slots=True)
class VerifiedTestPool:
    """One immutable official Test TrackPool and its access evidence."""

    pool: TrackPool
    evidence: TestPoolAccessEvidence

    def __post_init__(self) -> None:
        if not isinstance(self.pool, TrackPool) or self.pool.split != "test":
            raise TypeError("pool must be an immutable Test TrackPool")
        if not isinstance(self.evidence, TestPoolAccessEvidence):
            raise TypeError("evidence must be TestPoolAccessEvidence")
        if self.pool.size != self.evidence.track_count:
            raise ValueError("Test TrackPool size differs from its access evidence")
        if self.pool.capacity != self.evidence.capacity:
            raise ValueError("Test TrackPool capacity differs from its access evidence")
        if self.pool.benchmark_version != self.evidence.benchmark_version:
            raise ValueError("Test TrackPool benchmark differs from its access evidence")
        if self.pool.generator_version != self.evidence.generator_version:
            raise ValueError("Test TrackPool generator differs from its access evidence")
        if any(array.flags.writeable for array in self.pool.batch):
            raise ValueError("Test TrackPool arrays must be immutable")

        track_ids = tuple(int(value) for value in self.pool.batch.seed)
        if track_ids != self.evidence.track_ids:
            raise ValueError("Test Track IDs differ from their access evidence")
        if _sha256_lines(tuple(map(str, track_ids))) != self.evidence.track_ids_sha256:
            raise ValueError("Test Track ID order differs from its access evidence")
        geometry_hashes = track_batch_geometry_sha256(self.pool.batch)
        if _sha256_lines(geometry_hashes) != self.evidence.geometry_hashes_sha256:
            raise ValueError("Test TrackPool geometry differs from its access evidence")


def _load_stable_manifest(path: Path) -> tuple[TrackAssetManifest, str]:
    """Parse a Test manifest only when its bytes remain stable."""

    try:
        digest_before = sha256_file(path)
    except FileNotFoundError as error:
        raise TrackAssetError(f"Test manifest does not exist: {path}") from error
    manifest = load_track_asset_manifest(path)
    try:
        digest_after = sha256_file(path)
    except FileNotFoundError as error:
        raise TrackAssetError("Test manifest changed while it was being loaded") from error
    if digest_before != digest_after:
        raise TrackAssetError("Test manifest changed while it was being loaded")
    return manifest, digest_before


def _locked_test_spec() -> OfficialTrackSplitSpec:
    spec = official_track_split_spec("test")
    if (
        spec.split != "test"
        or spec.level_id != 1
        or spec.track_count != FORMAL_TEST_TRACK_COUNT
        or spec.seed_start != _TEST_SEED_START
        or spec.seed_stop != _TEST_SEED_STOP
        or spec.manifest_file != "test.json"
        or spec.asset_file != "test.npz"
        or spec.package_asset is not True
    ):
        raise TrackAssetError("official Test split contract is not the locked v0.1 contract")
    return spec


def load_verified_test_pool(
    project_config: ProjectConfig,
    *,
    asset_directory: str | Path | None = None,
) -> VerifiedTestPool:
    """Load only the official benchmark-0.1 Level-1 Test manifest and package NPZ.

    The split, filenames, namespace, count, and package-asset status are fixed internally. There is
    no split selector and this function does not invoke the general all-split asset verifier.
    """

    if not isinstance(project_config, ProjectConfig):
        raise TypeError("project_config must be a ProjectConfig")
    if project_config.benchmark.version != OFFICIAL_BENCHMARK_VERSION:
        raise TrackAssetError("final evaluation requires benchmark version 0.1")
    if project_config.benchmark.test_track_count != FORMAL_TEST_TRACK_COUNT:
        raise TrackAssetError("project_config must declare exactly 20 Test Tracks")
    spec = _locked_test_spec()
    directory = (
        official_track_asset_directory(project_config.benchmark.version)
        if asset_directory is None
        else Path(asset_directory)
    )
    manifest_path = directory / spec.manifest_file
    asset_path = directory / spec.asset_file
    require_published_identity = asset_directory is None

    manifest, manifest_digest = _load_stable_manifest(manifest_path)
    if require_published_identity and manifest_digest != FORMAL_TEST_MANIFEST_SHA256:
        raise TrackAssetError("official Test manifest differs from the published M5 identity")
    if (
        manifest.benchmark_version != OFFICIAL_BENCHMARK_VERSION
        or manifest.split != "test"
        or manifest.level_id != 1
        or manifest.track_count != FORMAL_TEST_TRACK_COUNT
        or manifest.asset_file != "test.npz"
    ):
        raise TrackAssetError("final evaluation requires the official v0.1 Test manifest")
    validate_official_manifest(project_config, manifest)
    if manifest.track_count != project_config.benchmark.test_track_count:
        raise TrackAssetError("Test manifest count differs from project_config")
    if manifest.capacity != track_capacity_from_project(project_config):
        raise TrackAssetError("Test manifest capacity differs from project_config")
    if manifest.generator_version != project_config.track.generator.generator_version:
        raise TrackAssetError("Test manifest generator differs from project_config")

    expected_track_ids = tuple(record.seed for record in manifest.tracks)
    if any(
        type(track_id) is not int or not _TEST_SEED_START <= track_id < _TEST_SEED_STOP
        for track_id in expected_track_ids
    ):
        raise TrackAssetError("Test manifest contains a Track outside its locked namespace")
    if any(left >= right for left, right in pairwise(expected_track_ids)):
        raise TrackAssetError("Test manifest Track IDs are not in strict manifest order")

    try:
        asset_digest = sha256_file(asset_path)
    except FileNotFoundError as error:
        raise TrackAssetError(f"Test NPZ does not exist: {asset_path}") from error
    if asset_digest != manifest.asset_sha256:
        raise TrackAssetError("Test NPZ SHA-256 differs from the manifest")
    if require_published_identity and asset_digest != FORMAL_TEST_ASSET_SHA256:
        raise TrackAssetError("official Test NPZ differs from the published M5 identity")
    batch = load_verified_manifest_batch(manifest, asset_path)
    try:
        asset_digest_after = sha256_file(asset_path)
    except FileNotFoundError as error:
        raise TrackAssetError("Test NPZ changed while it was being loaded") from error
    if asset_digest != asset_digest_after:
        raise TrackAssetError("Test NPZ changed while it was being loaded")

    expected_track_id_array = np.asarray(expected_track_ids, dtype=np.uint32)
    if batch.seed.shape != (FORMAL_TEST_TRACK_COUNT,):
        raise TrackAssetError("Test NPZ does not contain exactly 20 Tracks")
    if not np.array_equal(batch.seed, expected_track_id_array):
        raise TrackAssetError("Test NPZ Track order differs from the manifest")
    expected_geometry_hashes = tuple(record.geometry_sha256 for record in manifest.tracks)
    if track_batch_geometry_sha256(batch) != expected_geometry_hashes:
        raise TrackAssetError("Test NPZ geometry hashes differ from the manifest")

    pool = TrackPool(
        benchmark_version=manifest.benchmark_version,
        generator_version=manifest.generator_version,
        split="test",
        batch=batch,
    )
    evidence = TestPoolAccessEvidence(
        schema_version=M8_TEST_POOL_ACCESS_SCHEMA_VERSION,
        loaded_splits=("test",),
        benchmark_version=manifest.benchmark_version,
        generator_version=manifest.generator_version,
        level_id=manifest.level_id,
        split="test",
        manifest_file=manifest_path.name,
        manifest_sha256=manifest_digest,
        asset_file=asset_path.name,
        manifest_asset_sha256=manifest.asset_sha256,
        asset_file_sha256=asset_digest,
        track_count=manifest.track_count,
        capacity=manifest.capacity,
        track_ids=tuple(int(value) for value in expected_track_ids),
        track_ids_sha256=_sha256_lines(tuple(str(value) for value in expected_track_ids)),
        geometry_hashes_sha256=_sha256_lines(expected_geometry_hashes),
        loader_accessed_train=False,
        loader_accessed_validation=False,
    )
    return VerifiedTestPool(pool=pool, evidence=evidence)


__all__ = [
    "FORMAL_TEST_ASSET_SHA256",
    "FORMAL_TEST_MANIFEST_SHA256",
    "FORMAL_TEST_TRACK_COUNT",
    "M8_TEST_POOL_ACCESS_SCHEMA_VERSION",
    "TestPoolAccessEvidence",
    "VerifiedTestPool",
    "load_verified_test_pool",
]
