"""Tests for the split-specific M7 Train TrackPool loader."""

from __future__ import annotations

from dataclasses import FrozenInstanceError, dataclass, replace
from pathlib import Path

import numpy as np
import pytest

import controller_learning.rl.assets as rl_assets
import controller_learning.tracks.official_assets as official_assets
from controller_learning.config import ProjectConfig, load_project_config
from controller_learning.tracks.assets import (
    TRACK_ASSET_SCHEMA_VERSION,
    TrackAssetError,
    TrackAssetManifest,
    TrackAssetRecord,
    save_track_batch_npz,
    write_track_asset_manifest,
)
from controller_learning.tracks.generator import generate_track_candidate, pack_track
from controller_learning.tracks.hashing import track_geometry_sha256
from controller_learning.tracks.official_assets import (
    OFFICIAL_DRIVEABILITY_PROTOCOL_VERSION,
    OFFICIAL_GEOMETRY_VALIDATION_VERSION,
    OfficialTrackSplitSpec,
)
from controller_learning.tracks.specs import (
    generation_spec_from_project,
    track_capacity_from_project,
)
from controller_learning.tracks.types import Track, TrackCapacity, stack_tracks

PROJECT_ROOT = Path(__file__).parents[3]


@dataclass(frozen=True)
class TinyTrainAsset:
    config: ProjectConfig
    directory: Path
    cache: Path
    manifest: TrackAssetManifest
    spec: OfficialTrackSplitSpec
    tracks: tuple[Track, Track]


@pytest.fixture
def tiny_train_asset(tmp_path: Path) -> TinyTrainAsset:
    project = load_project_config(PROJECT_ROOT)
    capacity = track_capacity_from_project(project)
    generation = generation_spec_from_project(project)
    tracks = tuple(
        pack_track(generate_track_candidate(seed, generation), capacity) for seed in (42, 43)
    )
    batch = stack_tracks(tracks)
    directory = tmp_path / "assets"
    cache = tmp_path / "cache" / "train_pool.npz"
    digest = save_track_batch_npz(batch, cache)
    records = tuple(
        TrackAssetRecord(
            seed=track.seed,
            geometry_sha256=track_geometry_sha256(track),
            geometry_validation="passed",
            driveability_validation="passed",
        )
        for track in tracks
    )
    manifest = TrackAssetManifest(
        schema_version=TRACK_ASSET_SCHEMA_VERSION,
        benchmark_version=project.benchmark.version,
        level_id=1,
        split="train",
        generator_version=project.track.generator.generator_version,
        geometry_validation_version=OFFICIAL_GEOMETRY_VALIDATION_VERSION,
        driveability_protocol_version=OFFICIAL_DRIVEABILITY_PROTOCOL_VERSION,
        track_width_m=7.0,
        track_count=2,
        capacity=capacity,
        asset_file="train_pool.npz",
        asset_sha256=digest,
        tracks=records,
    )
    write_track_asset_manifest(manifest, directory / "train.json")
    spec = OfficialTrackSplitSpec(
        split="train",
        level_id=1,
        track_count=2,
        seed_start=0,
        seed_stop=1_000_000,
        manifest_file="train.json",
        asset_file="train_pool.npz",
        package_asset=False,
    )
    config = replace(
        project,
        benchmark=replace(project.benchmark, train_track_count=2),
    )
    return TinyTrainAsset(config, directory, cache, manifest, spec, tracks)  # type: ignore[arg-type]


def _install_tiny_official_contract(
    monkeypatch: pytest.MonkeyPatch,
    asset: TinyTrainAsset,
) -> list[TrackAssetManifest]:
    validated: list[TrackAssetManifest] = []

    def train_spec(split: str) -> OfficialTrackSplitSpec:
        if split != "train":
            raise AssertionError(f"forbidden split access: {split}")
        return asset.spec

    def validate(_config: ProjectConfig, manifest: TrackAssetManifest) -> None:
        validated.append(manifest)

    monkeypatch.setattr(rl_assets, "official_track_split_spec", train_spec)
    monkeypatch.setattr(rl_assets, "validate_official_manifest", validate)
    return validated


def test_train_loader_reads_only_train_and_returns_immutable_evidence(
    monkeypatch: pytest.MonkeyPatch,
    tiny_train_asset: TinyTrainAsset,
) -> None:
    asset = tiny_train_asset
    (asset.directory / "validation.json").write_text("forbidden", encoding="utf-8")
    (asset.directory / "test.json").write_text("forbidden", encoding="utf-8")
    validated = _install_tiny_official_contract(monkeypatch, asset)
    manifest_reads: list[str] = []
    actual_manifest_loader = rl_assets.load_track_asset_manifest

    def load_manifest(path: str | Path) -> TrackAssetManifest:
        name = Path(path).name
        manifest_reads.append(name)
        if name != "train.json":
            raise AssertionError(f"forbidden manifest access: {name}")
        return actual_manifest_loader(path)

    def forbidden_all_split_verifier(*_args, **_kwargs):
        raise AssertionError("the all-split verifier is forbidden during PPO optimization")

    monkeypatch.setattr(rl_assets, "load_track_asset_manifest", load_manifest)
    monkeypatch.setattr(
        official_assets,
        "verify_official_track_assets",
        forbidden_all_split_verifier,
    )

    result = rl_assets.load_verified_train_pool(
        asset.config,
        train_cache_path=asset.cache,
        asset_directory=asset.directory,
    )

    assert manifest_reads == ["train.json"]
    assert validated == [asset.manifest]
    assert result.pool.split == "train"
    assert result.pool.size == 2
    assert all(not array.flags.writeable for array in result.pool.batch)
    evidence = result.evidence
    assert evidence.loaded_splits == ("train",)
    assert evidence.generator_version == asset.manifest.generator_version
    assert evidence.manifest_file == "train.json"
    assert evidence.manifest_sha256 == rl_assets.sha256_file(asset.directory / "train.json")
    assert evidence.cache_file == "train_pool.npz"
    assert evidence.manifest_asset_sha256 == evidence.cache_file_sha256
    assert evidence.cache_file_sha256 == rl_assets.sha256_file(asset.cache)
    assert evidence.track_count == 2
    assert (evidence.first_track_id, evidence.last_track_id) == (42, 43)
    assert evidence.loader_accessed_validation is False
    assert evidence.loader_accessed_test is False
    with pytest.raises(FrozenInstanceError):
        evidence.loader_accessed_test = True  # type: ignore[misc]


def test_train_loader_rejects_non_train_cache_filename_before_reading_it(
    monkeypatch: pytest.MonkeyPatch,
    tiny_train_asset: TinyTrainAsset,
    tmp_path: Path,
) -> None:
    asset = tiny_train_asset
    validated = _install_tiny_official_contract(monkeypatch, asset)
    decoy = tmp_path / "test.npz"
    decoy.write_bytes(b"must not be read")
    actual_sha256_file = rl_assets.sha256_file
    actual_batch_loader = rl_assets.load_verified_manifest_batch

    def guarded_sha256_file(path: str | Path) -> str:
        if Path(path) == decoy:
            raise AssertionError("the decoy Test cache was hashed")
        return actual_sha256_file(path)

    def guarded_batch_loader(manifest: TrackAssetManifest, path: str | Path):
        if Path(path) == decoy:
            raise AssertionError("the decoy Test cache was opened")
        return actual_batch_loader(manifest, path)

    monkeypatch.setattr(rl_assets, "sha256_file", guarded_sha256_file)
    monkeypatch.setattr(rl_assets, "load_verified_manifest_batch", guarded_batch_loader)

    with pytest.raises(TrackAssetError, match="official Train asset filename"):
        rl_assets.load_verified_train_pool(
            asset.config,
            train_cache_path=decoy,
            asset_directory=asset.directory,
        )
    assert validated == []


def test_train_loader_rejects_manifest_or_cache_changed_during_loading(
    monkeypatch: pytest.MonkeyPatch,
    tiny_train_asset: TinyTrainAsset,
) -> None:
    asset = tiny_train_asset
    _install_tiny_official_contract(monkeypatch, asset)
    actual_manifest_loader = rl_assets.load_track_asset_manifest

    def mutate_manifest_after_parse(path: str | Path) -> TrackAssetManifest:
        manifest = actual_manifest_loader(path)
        source = Path(path)
        source.write_bytes(source.read_bytes() + b"\n")
        return manifest

    monkeypatch.setattr(rl_assets, "load_track_asset_manifest", mutate_manifest_after_parse)
    with pytest.raises(TrackAssetError, match="manifest changed"):
        rl_assets.load_verified_train_pool(
            asset.config,
            train_cache_path=asset.cache,
            asset_directory=asset.directory,
        )

    write_track_asset_manifest(asset.manifest, asset.directory / "train.json")
    monkeypatch.setattr(rl_assets, "load_track_asset_manifest", actual_manifest_loader)
    actual_batch_loader = rl_assets.load_verified_manifest_batch

    def mutate_cache_after_load(manifest: TrackAssetManifest, path: str | Path):
        batch = actual_batch_loader(manifest, path)
        source = Path(path)
        source.write_bytes(source.read_bytes() + b"changed")
        return batch

    monkeypatch.setattr(rl_assets, "load_verified_manifest_batch", mutate_cache_after_load)
    with pytest.raises(TrackAssetError, match="cache changed"):
        rl_assets.load_verified_train_pool(
            asset.config,
            train_cache_path=asset.cache,
            asset_directory=asset.directory,
        )


def test_verified_train_pool_rejects_identity_and_geometry_contradictions(
    monkeypatch: pytest.MonkeyPatch,
    tiny_train_asset: TinyTrainAsset,
) -> None:
    asset = tiny_train_asset
    _install_tiny_official_contract(monkeypatch, asset)
    result = rl_assets.load_verified_train_pool(
        asset.config,
        train_cache_path=asset.cache,
        asset_directory=asset.directory,
    )

    wrong_benchmark = replace(result.pool, benchmark_version="contradictory")
    with pytest.raises(ValueError, match="benchmark version"):
        rl_assets.VerifiedTrainPool(wrong_benchmark, result.evidence)

    wrong_generator = replace(result.pool, generator_version="contradictory")
    with pytest.raises(ValueError, match="generator version"):
        rl_assets.VerifiedTrainPool(wrong_generator, result.evidence)

    wrong_ids = replace(result.evidence, track_ids_sha256="0" * 64)
    with pytest.raises(ValueError, match="IDs"):
        rl_assets.VerifiedTrainPool(result.pool, wrong_ids)

    wrong_geometry = replace(result.evidence, geometry_hashes_sha256="1" * 64)
    with pytest.raises(ValueError, match="geometry"):
        rl_assets.VerifiedTrainPool(result.pool, wrong_geometry)

    wrong_width = np.full_like(result.pool.batch.width_m, 8.0)
    changed_batch = result.pool.batch._replace(width_m=wrong_width)
    changed_pool = replace(result.pool, batch=changed_batch)
    with pytest.raises(ValueError, match="geometry"):
        rl_assets.VerifiedTrainPool(changed_pool, result.evidence)


def test_train_loader_rejects_cache_hash_mismatch(
    monkeypatch: pytest.MonkeyPatch,
    tiny_train_asset: TinyTrainAsset,
) -> None:
    asset = tiny_train_asset
    _install_tiny_official_contract(monkeypatch, asset)
    asset.cache.write_bytes(asset.cache.read_bytes() + b"tampered")

    with pytest.raises(TrackAssetError, match="cache SHA-256"):
        rl_assets.load_verified_train_pool(
            asset.config,
            train_cache_path=asset.cache,
            asset_directory=asset.directory,
        )


def test_train_loader_rejects_count_and_capacity_drift(
    monkeypatch: pytest.MonkeyPatch,
    tiny_train_asset: TinyTrainAsset,
) -> None:
    asset = tiny_train_asset
    validated = _install_tiny_official_contract(monkeypatch, asset)

    monkeypatch.setattr(
        rl_assets,
        "official_track_split_spec",
        lambda split: replace(asset.spec, track_count=3),
    )
    with pytest.raises(TrackAssetError, match="exactly 3 Tracks"):
        rl_assets.load_verified_train_pool(
            asset.config,
            train_cache_path=asset.cache,
            asset_directory=asset.directory,
        )

    monkeypatch.setattr(rl_assets, "official_track_split_spec", lambda split: asset.spec)
    wrong_capacity = replace(
        asset.manifest,
        capacity=TrackCapacity(
            max_track_points=asset.manifest.capacity.max_track_points + 1,
            max_checkpoints=asset.manifest.capacity.max_checkpoints,
        ),
    )
    write_track_asset_manifest(wrong_capacity, asset.directory / "train.json")
    with pytest.raises(TrackAssetError, match="capacity does not match"):
        rl_assets.load_verified_train_pool(
            asset.config,
            train_cache_path=asset.cache,
            asset_directory=asset.directory,
        )
    assert validated


def test_train_loader_rejects_manifest_order_drift(
    monkeypatch: pytest.MonkeyPatch,
    tiny_train_asset: TinyTrainAsset,
) -> None:
    asset = tiny_train_asset
    _install_tiny_official_contract(monkeypatch, asset)
    reordered = replace(asset.manifest, tracks=tuple(reversed(asset.manifest.tracks)))
    write_track_asset_manifest(reordered, asset.directory / "train.json")

    with pytest.raises(TrackAssetError, match=r"seed order|Track order"):
        rl_assets.load_verified_train_pool(
            asset.config,
            train_cache_path=asset.cache,
            asset_directory=asset.directory,
        )


def test_train_loader_rejects_geometry_drift_even_with_matching_cache_digest(
    monkeypatch: pytest.MonkeyPatch,
    tiny_train_asset: TinyTrainAsset,
) -> None:
    asset = tiny_train_asset
    _install_tiny_official_contract(monkeypatch, asset)
    generation = generation_spec_from_project(asset.config)
    replacement = pack_track(
        generate_track_candidate(44, generation),
        asset.manifest.capacity,
    )
    replacement = replace(replacement, seed=asset.tracks[0].seed)
    changed_batch = stack_tracks((replacement, asset.tracks[1]))
    changed_digest = save_track_batch_npz(changed_batch, asset.cache)
    changed_manifest = replace(asset.manifest, asset_sha256=changed_digest)
    write_track_asset_manifest(changed_manifest, asset.directory / "train.json")

    with pytest.raises(TrackAssetError, match="geometry hashes"):
        rl_assets.load_verified_train_pool(
            asset.config,
            train_cache_path=asset.cache,
            asset_directory=asset.directory,
        )


def test_official_train_loader_fails_before_any_non_train_manifest_access(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    project = load_project_config(PROJECT_ROOT)
    actual_manifest_loader = rl_assets.load_track_asset_manifest
    seen: list[str] = []

    def train_only(path: str | Path) -> TrackAssetManifest:
        name = Path(path).name
        seen.append(name)
        if name != "train.json":
            raise AssertionError(f"forbidden manifest access: {name}")
        return actual_manifest_loader(path)

    monkeypatch.setattr(rl_assets, "load_track_asset_manifest", train_only)

    with pytest.raises(TrackAssetError, match="cache does not exist"):
        rl_assets.load_verified_train_pool(
            project,
            train_cache_path=tmp_path / "train_pool.npz",
        )
    assert seen == ["train.json"]
