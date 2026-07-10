"""Tests for the strict M7 Validation-only TrackPool loader."""

from __future__ import annotations

import shutil
from dataclasses import FrozenInstanceError, replace
from pathlib import Path

import pytest

import controller_learning.rl.validation_assets as validation_assets
import controller_learning.tracks.official_assets as official_assets
from controller_learning.config import load_project_config
from controller_learning.tracks.assets import TrackAssetError, TrackAssetManifest

PROJECT_ROOT = Path(__file__).parents[3]


def test_official_validation_loader_reads_only_validation_and_binds_evidence(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    project = load_project_config(PROJECT_ROOT)
    manifest_reads: list[str] = []
    digest_reads: list[str] = []
    batch_reads: list[str] = []
    actual_manifest_loader = validation_assets.load_track_asset_manifest
    actual_sha256_file = validation_assets.sha256_file
    actual_batch_loader = validation_assets.load_verified_manifest_batch

    def load_manifest(path: str | Path) -> TrackAssetManifest:
        name = Path(path).name
        manifest_reads.append(name)
        if name != "validation.json":
            raise AssertionError(f"forbidden manifest access: {name}")
        return actual_manifest_loader(path)

    def sha256_file(path: str | Path) -> str:
        name = Path(path).name
        digest_reads.append(name)
        if name not in {"validation.json", "validation.npz"}:
            raise AssertionError(f"forbidden asset digest access: {name}")
        return actual_sha256_file(path)

    def load_batch(manifest: TrackAssetManifest, path: str | Path):
        name = Path(path).name
        batch_reads.append(name)
        if manifest.split != "validation" or name != "validation.npz":
            raise AssertionError("non-Validation batch access")
        return actual_batch_loader(manifest, path)

    def forbidden_all_split_verifier(*_args, **_kwargs):
        raise AssertionError("the all-split verifier is forbidden during checkpoint selection")

    monkeypatch.setattr(validation_assets, "load_track_asset_manifest", load_manifest)
    monkeypatch.setattr(validation_assets, "sha256_file", sha256_file)
    monkeypatch.setattr(validation_assets, "load_verified_manifest_batch", load_batch)
    monkeypatch.setattr(
        official_assets,
        "verify_official_track_assets",
        forbidden_all_split_verifier,
    )

    result = validation_assets.load_verified_validation_pool(project)

    assert manifest_reads == ["validation.json"]
    assert digest_reads == [
        "validation.json",
        "validation.json",
        "validation.npz",
        "validation.npz",
    ]
    assert batch_reads == ["validation.npz"]
    assert result.pool.split == "validation"
    assert result.pool.size == validation_assets.FORMAL_VALIDATION_TRACK_COUNT == 100
    assert all(not array.flags.writeable for array in result.pool.batch)
    evidence = result.evidence
    assert evidence.loaded_splits == ("validation",)
    assert evidence.benchmark_version == "0.1"
    assert evidence.level_id == 1
    assert evidence.manifest_file == "validation.json"
    assert evidence.asset_file == "validation.npz"
    assert evidence.manifest_asset_sha256 == evidence.asset_file_sha256
    assert evidence.first_track_id == int(result.pool.batch.seed[0])
    assert evidence.last_track_id == int(result.pool.batch.seed[-1])
    assert evidence.loader_accessed_train is False
    assert evidence.loader_accessed_test is False
    with pytest.raises(FrozenInstanceError):
        evidence.loader_accessed_test = True  # type: ignore[misc]


def test_validation_loader_rejects_manifest_or_npz_changed_during_loading(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    project = load_project_config(PROJECT_ROOT)
    source = official_assets.official_track_asset_directory(project.benchmark.version)
    directory = tmp_path / "validation-only"
    directory.mkdir()
    for name in ("validation.json", "validation.npz"):
        shutil.copy2(source / name, directory / name)

    actual_manifest_loader = validation_assets.load_track_asset_manifest

    def mutate_manifest_after_parse(path: str | Path) -> TrackAssetManifest:
        manifest = actual_manifest_loader(path)
        candidate = Path(path)
        candidate.write_bytes(candidate.read_bytes() + b"\n")
        return manifest

    monkeypatch.setattr(
        validation_assets,
        "load_track_asset_manifest",
        mutate_manifest_after_parse,
    )
    with pytest.raises(TrackAssetError, match="manifest changed"):
        validation_assets.load_verified_validation_pool(
            project,
            asset_directory=directory,
        )

    shutil.copy2(source / "validation.json", directory / "validation.json")
    monkeypatch.setattr(
        validation_assets,
        "load_track_asset_manifest",
        actual_manifest_loader,
    )
    actual_batch_loader = validation_assets.load_verified_manifest_batch

    def mutate_npz_after_load(manifest: TrackAssetManifest, path: str | Path):
        batch = actual_batch_loader(manifest, path)
        candidate = Path(path)
        candidate.write_bytes(candidate.read_bytes() + b"changed")
        return batch

    monkeypatch.setattr(
        validation_assets,
        "load_verified_manifest_batch",
        mutate_npz_after_load,
    )
    with pytest.raises(TrackAssetError, match="NPZ changed"):
        validation_assets.load_verified_validation_pool(
            project,
            asset_directory=directory,
        )


def test_validation_evidence_rejects_split_and_pool_identity_contradictions() -> None:
    project = load_project_config(PROJECT_ROOT)
    result = validation_assets.load_verified_validation_pool(project)

    with pytest.raises(ValueError, match="cannot claim another split"):
        replace(result.evidence, loader_accessed_test=True)
    with pytest.raises(ValueError, match="100 Tracks"):
        replace(result.evidence, track_count=99)
    with pytest.raises(ValueError, match="Track IDs differ"):
        validation_assets.VerifiedValidationPool(
            pool=result.pool,
            evidence=replace(result.evidence, track_ids_sha256="0" * 64),
        )


def test_validation_loader_source_has_no_all_split_or_selectable_asset_path() -> None:
    source = (PROJECT_ROOT / "controller_learning" / "rl" / "validation_assets.py").read_text(
        encoding="utf-8"
    )
    assert "verify_official_track_assets" not in source
    assert "load_verified_test_pool" not in source
    assert "load_verified_train_pool" not in source
    assert "validation_asset_path" not in source
    assert 'official_track_split_spec("validation")' in source
