"""Tests for official split verification and reproducible training-pool caches."""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

from controller_learning.config import load_project_config
from controller_learning.tracks.assets import (
    TRACK_ASSET_SCHEMA_VERSION,
    TrackAssetError,
    TrackAssetManifest,
    TrackAssetRecord,
    load_track_batch_npz,
    save_track_batch_npz,
)
from controller_learning.tracks.generator import generate_track_candidate, pack_track
from controller_learning.tracks.hashing import track_geometry_sha256
from controller_learning.tracks.level0 import build_level0_track
from controller_learning.tracks.official_assets import (
    DEFAULT_TRAIN_CACHE,
    OFFICIAL_TRACK_SPLITS,
    load_verified_manifest_batch,
    official_track_asset_directory,
    regenerate_manifest_track_batch,
    validate_official_manifest,
    verify_manifest_disjointness,
    write_verified_manifest_cache,
)
from controller_learning.tracks.specs import (
    generation_spec_from_project,
    track_capacity_from_project,
    validation_spec_from_project,
)
from controller_learning.tracks.types import Track, TrackBatch, stack_tracks
from controller_learning.tracks.validator import validate_track_candidate

PROJECT_ROOT = Path(__file__).parents[3]


@pytest.fixture(scope="module")
def project_config():
    return load_project_config(PROJECT_ROOT)


@pytest.fixture(scope="module")
def generated_track(project_config) -> Track:
    candidate = generate_track_candidate(42, generation_spec_from_project(project_config))
    result = validate_track_candidate(candidate, validation_spec_from_project(project_config))
    assert result.valid, result.reasons
    return pack_track(candidate, track_capacity_from_project(project_config))


def _record(seed: int, digest_number: int) -> TrackAssetRecord:
    return TrackAssetRecord(
        seed=seed,
        geometry_sha256=f"{digest_number:064x}",
        geometry_validation="passed",
        driveability_validation="passed",
    )


def _manifest(
    project_config,
    *,
    split: str,
    level_id: int,
    records: tuple[TrackAssetRecord, ...],
    asset_file: str,
    asset_sha256: str = "a" * 64,
) -> TrackAssetManifest:
    return TrackAssetManifest(
        schema_version=TRACK_ASSET_SCHEMA_VERSION,
        benchmark_version="0.1",
        level_id=level_id,
        split=split,
        generator_version=project_config.track.generator.generator_version,
        geometry_validation_version="m3-geometry-v1",
        driveability_protocol_version="m5-driveability-v1",
        track_width_m=7.0,
        track_count=len(records),
        capacity=track_capacity_from_project(project_config),
        asset_file=asset_file,
        asset_sha256=asset_sha256,
        tracks=records,
    )


def _generated_manifest(project_config, track: Track, *, asset_sha256: str = "a" * 64):
    return _manifest(
        project_config,
        split="train",
        level_id=1,
        records=(
            TrackAssetRecord(
                seed=track.seed,
                geometry_sha256=track_geometry_sha256(track),
                geometry_validation="passed",
                driveability_validation="passed",
            ),
        ),
        asset_file="train_pool.npz",
        asset_sha256=asset_sha256,
    )


def test_official_split_specs_lock_namespaces_counts_and_files() -> None:
    specs = {spec.split: spec for spec in OFFICIAL_TRACK_SPLITS}

    assert Path(".track-cache/v0.1/train_pool.npz") == DEFAULT_TRAIN_CACHE
    assert (specs["train"].track_count, specs["train"].seed_start, specs["train"].seed_stop) == (
        10_000,
        0,
        1_000_000,
    )
    assert (
        specs["validation"].track_count,
        specs["validation"].seed_start,
        specs["validation"].seed_stop,
    ) == (100, 1_000_000, 2_000_000)
    assert (specs["test"].track_count, specs["test"].seed_start, specs["test"].seed_stop) == (
        20,
        2_000_000,
        3_000_000,
    )
    assert specs["level0"].package_asset
    assert not specs["train"].package_asset
    assert specs["train"].asset_file == "train_pool.npz"
    assert official_track_asset_directory(package_root=Path("package")) == Path(
        "package/assets/tracks/v0.1"
    )


def test_official_manifest_checks_counts_files_and_seed_namespace(project_config) -> None:
    level0 = build_level0_track(track_capacity_from_project(project_config))
    level0_manifest = _manifest(
        project_config,
        split="level0",
        level_id=0,
        records=(
            TrackAssetRecord(
                seed=level0.seed,
                geometry_sha256=track_geometry_sha256(level0),
                geometry_validation="passed",
                driveability_validation="passed",
            ),
        ),
        asset_file="level0.npz",
    )
    validate_official_manifest(project_config, level0_manifest)

    with pytest.raises(TrackAssetError, match="reserved Level 0 seed"):
        replace(
            level0_manifest,
            tracks=(replace(level0_manifest.tracks[0], seed=0),),
        )

    with pytest.raises(TrackAssetError, match="wrong asset filename"):
        validate_official_manifest(
            project_config,
            replace(level0_manifest, asset_file="wrong.npz"),
        )

    validation_records = tuple(_record(1_000_000 + index, index + 100) for index in range(100))
    validation_manifest = _manifest(
        project_config,
        split="validation",
        level_id=1,
        records=validation_records,
        asset_file="validation.npz",
    )
    validate_official_manifest(project_config, validation_manifest)
    reordered = (
        validation_records[1],
        validation_records[0],
        *validation_records[2:],
    )
    with pytest.raises(TrackAssetError, match="strictly increasing by seed"):
        validate_official_manifest(
            project_config,
            replace(validation_manifest, tracks=reordered),
        )
    outside_namespace = (
        _record(999_999, 99),
        *validation_records[1:],
    )
    with pytest.raises(TrackAssetError, match="outside its locked namespace"):
        validate_official_manifest(
            project_config,
            replace(validation_manifest, tracks=outside_namespace),
        )


def test_cross_split_seed_and_geometry_disjointness_is_explicit(project_config) -> None:
    level0_track = build_level0_track(track_capacity_from_project(project_config))
    manifests = {
        "level0": _manifest(
            project_config,
            split="level0",
            level_id=0,
            records=(
                TrackAssetRecord(
                    seed=level0_track.seed,
                    geometry_sha256=track_geometry_sha256(level0_track),
                    geometry_validation="passed",
                    driveability_validation="passed",
                ),
            ),
            asset_file="level0.npz",
        ),
        "train": _manifest(
            project_config,
            split="train",
            level_id=1,
            records=(_record(1, 1),),
            asset_file="train_pool.npz",
        ),
        "validation": _manifest(
            project_config,
            split="validation",
            level_id=1,
            records=(_record(2, 2),),
            asset_file="validation.npz",
        ),
        "test": _manifest(
            project_config,
            split="test",
            level_id=1,
            records=(_record(3, 3),),
            asset_file="test.npz",
        ),
    }
    verify_manifest_disjointness(manifests)

    duplicate_seed = dict(manifests)
    duplicate_seed["test"] = replace(
        duplicate_seed["test"],
        tracks=(_record(1, 3),),
    )
    with pytest.raises(TrackAssetError, match="seed 1 appears in both train and test"):
        verify_manifest_disjointness(duplicate_seed)

    duplicate_geometry = dict(manifests)
    duplicate_geometry["test"] = replace(
        duplicate_geometry["test"],
        tracks=(_record(3, 1),),
    )
    with pytest.raises(TrackAssetError, match="geometry appears in both train and test"):
        verify_manifest_disjointness(duplicate_geometry)


def test_manifest_regeneration_revalidates_and_reproduces_exact_batch(
    project_config,
    generated_track: Track,
) -> None:
    manifest = _generated_manifest(project_config, generated_track)
    progress: list[tuple[int, int, int]] = []

    batch = regenerate_manifest_track_batch(
        project_config,
        manifest,
        progress=lambda completed, total, seed: progress.append((completed, total, seed)),
    )

    expected = stack_tracks([generated_track])
    for expected_array, actual_array in zip(expected, batch, strict=True):
        np.testing.assert_array_equal(actual_array, expected_array)
        assert not actual_array.flags.writeable
    assert progress == [(1, 1, 42)]

    wrong_record = replace(manifest.tracks[0], geometry_sha256="0" * 64)
    with pytest.raises(TrackAssetError, match="geometry hash does not match"):
        regenerate_manifest_track_batch(
            project_config,
            replace(manifest, tracks=(wrong_record,)),
        )


def test_verified_cache_write_is_atomic_and_manifest_bound(
    tmp_path: Path,
    project_config,
    generated_track: Track,
) -> None:
    batch: TrackBatch = stack_tracks([generated_track])
    canonical = tmp_path / "canonical.npz"
    digest = save_track_batch_npz(batch, canonical)
    manifest = _generated_manifest(project_config, generated_track, asset_sha256=digest)
    destination = tmp_path / "train_pool.npz"

    assert write_verified_manifest_cache(batch, manifest, destination) == digest
    assert destination.read_bytes() == canonical.read_bytes()
    verified = load_verified_manifest_batch(manifest, destination)
    loaded = load_track_batch_npz(destination, expected_sha256=digest)
    np.testing.assert_array_equal(verified.seed, loaded.seed)

    destination.write_bytes(b"existing-cache-must-survive")
    wrong_digest_manifest = replace(manifest, asset_sha256="0" * 64)
    with pytest.raises(TrackAssetError, match="cache SHA-256"):
        write_verified_manifest_cache(batch, wrong_digest_manifest, destination)
    assert destination.read_bytes() == b"existing-cache-must-survive"
