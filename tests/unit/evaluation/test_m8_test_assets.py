"""CPU-only tests for the strict M8 Test-only TrackPool loader."""

from __future__ import annotations

import inspect
import shutil
from dataclasses import FrozenInstanceError, dataclass, replace
from pathlib import Path

import pytest

import controller_learning.evaluation.test_assets as test_assets
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
from controller_learning.tracks.pool import TrackPool
from controller_learning.tracks.specs import (
    generation_spec_from_project,
    track_capacity_from_project,
    validation_spec_from_project,
)
from controller_learning.tracks.types import Track, TrackCapacity, stack_tracks
from controller_learning.tracks.validator import validate_track_candidate

PROJECT_ROOT = Path(__file__).parents[3]


@dataclass(frozen=True, slots=True)
class SyntheticTestAsset:
    config: ProjectConfig
    directory: Path
    manifest: TrackAssetManifest
    tracks: tuple[Track, ...]


@pytest.fixture(scope="module")
def synthetic_test_asset_source(
    tmp_path_factory: pytest.TempPathFactory,
) -> SyntheticTestAsset:
    """Create 20 valid generated rows without reading the repository's Test assets."""

    project = load_project_config(PROJECT_ROOT)
    generation = generation_spec_from_project(project)
    validation = validation_spec_from_project(project)
    capacity = track_capacity_from_project(project)
    tracks: list[Track] = []
    for seed in range(2_000_000, 3_000_000):
        candidate = generate_track_candidate(seed, generation)
        if not validate_track_candidate(candidate, validation).valid:
            continue
        tracks.append(pack_track(candidate, capacity))
        if len(tracks) == test_assets.FORMAL_TEST_TRACK_COUNT:
            break
    assert len(tracks) == test_assets.FORMAL_TEST_TRACK_COUNT
    immutable_tracks = tuple(tracks)
    directory = tmp_path_factory.mktemp("synthetic-test-assets")
    asset_digest = save_track_batch_npz(stack_tracks(immutable_tracks), directory / "test.npz")
    manifest = TrackAssetManifest(
        schema_version=TRACK_ASSET_SCHEMA_VERSION,
        benchmark_version=project.benchmark.version,
        level_id=1,
        split="test",
        generator_version=project.track.generator.generator_version,
        geometry_validation_version=OFFICIAL_GEOMETRY_VALIDATION_VERSION,
        driveability_protocol_version=OFFICIAL_DRIVEABILITY_PROTOCOL_VERSION,
        track_width_m=7.0,
        track_count=test_assets.FORMAL_TEST_TRACK_COUNT,
        capacity=capacity,
        asset_file="test.npz",
        asset_sha256=asset_digest,
        tracks=tuple(
            TrackAssetRecord(
                seed=track.seed,
                geometry_sha256=track_geometry_sha256(track),
                geometry_validation="passed",
                driveability_validation="passed",
            )
            for track in immutable_tracks
        ),
    )
    write_track_asset_manifest(manifest, directory / "test.json")
    official_assets.validate_official_manifest(project, manifest)
    return SyntheticTestAsset(project, directory, manifest, immutable_tracks)


@pytest.fixture
def synthetic_test_asset(
    tmp_path: Path,
    synthetic_test_asset_source: SyntheticTestAsset,
) -> SyntheticTestAsset:
    directory = tmp_path / "assets"
    shutil.copytree(synthetic_test_asset_source.directory, directory)
    return replace(synthetic_test_asset_source, directory=directory)


def test_loader_reads_only_test_and_binds_sanitized_immutable_evidence(
    monkeypatch: pytest.MonkeyPatch,
    synthetic_test_asset: SyntheticTestAsset,
) -> None:
    asset = synthetic_test_asset
    (asset.directory / "train.json").write_text("forbidden", encoding="utf-8")
    (asset.directory / "validation.json").write_text("forbidden", encoding="utf-8")
    manifest_reads: list[str] = []
    digest_reads: list[str] = []
    batch_reads: list[str] = []
    actual_manifest_loader = test_assets.load_track_asset_manifest
    actual_sha256_file = test_assets.sha256_file
    actual_batch_loader = test_assets.load_verified_manifest_batch

    def load_manifest(path: str | Path) -> TrackAssetManifest:
        name = Path(path).name
        manifest_reads.append(name)
        if name != "test.json":
            raise AssertionError(f"forbidden manifest access: {name}")
        return actual_manifest_loader(path)

    def sha256_file(path: str | Path) -> str:
        name = Path(path).name
        digest_reads.append(name)
        if name not in {"test.json", "test.npz"}:
            raise AssertionError(f"forbidden asset digest access: {name}")
        return actual_sha256_file(path)

    def load_batch(manifest: TrackAssetManifest, path: str | Path):
        name = Path(path).name
        batch_reads.append(name)
        if manifest.split != "test" or name != "test.npz":
            raise AssertionError("non-Test batch access")
        return actual_batch_loader(manifest, path)

    def forbidden_all_split_verifier(*_args, **_kwargs):
        raise AssertionError("the all-split verifier is forbidden during final evaluation")

    monkeypatch.setattr(test_assets, "load_track_asset_manifest", load_manifest)
    monkeypatch.setattr(test_assets, "sha256_file", sha256_file)
    monkeypatch.setattr(test_assets, "load_verified_manifest_batch", load_batch)
    monkeypatch.setattr(
        official_assets,
        "verify_official_track_assets",
        forbidden_all_split_verifier,
    )

    result = test_assets.load_verified_test_pool(
        asset.config,
        asset_directory=asset.directory,
    )

    assert manifest_reads == ["test.json"]
    assert digest_reads == ["test.json", "test.json", "test.npz", "test.npz"]
    assert batch_reads == ["test.npz"]
    assert result.pool.split == "test"
    assert result.pool.size == test_assets.FORMAL_TEST_TRACK_COUNT == 20
    assert all(not array.flags.writeable for array in result.pool.batch)
    evidence = result.evidence
    assert evidence.loaded_splits == ("test",)
    assert evidence.benchmark_version == "0.1"
    assert evidence.level_id == 1
    assert evidence.manifest_file == "test.json"
    assert evidence.asset_file == "test.npz"
    assert evidence.track_ids == tuple(int(value) for value in result.pool.batch.seed)
    assert evidence.manifest_asset_sha256 == evidence.asset_file_sha256
    assert evidence.loader_accessed_train is False
    assert evidence.loader_accessed_validation is False
    assert str(asset.directory) not in repr(evidence)
    with pytest.raises(FrozenInstanceError):
        evidence.loader_accessed_train = True  # type: ignore[misc]


def test_loader_rejects_manifest_or_npz_changed_during_loading(
    monkeypatch: pytest.MonkeyPatch,
    synthetic_test_asset: SyntheticTestAsset,
) -> None:
    asset = synthetic_test_asset
    actual_manifest_loader = test_assets.load_track_asset_manifest

    def mutate_manifest_after_parse(path: str | Path) -> TrackAssetManifest:
        manifest = actual_manifest_loader(path)
        candidate = Path(path)
        candidate.write_bytes(candidate.read_bytes() + b"\n")
        return manifest

    monkeypatch.setattr(test_assets, "load_track_asset_manifest", mutate_manifest_after_parse)
    with pytest.raises(TrackAssetError, match="manifest changed"):
        test_assets.load_verified_test_pool(asset.config, asset_directory=asset.directory)

    write_track_asset_manifest(asset.manifest, asset.directory / "test.json")
    monkeypatch.setattr(test_assets, "load_track_asset_manifest", actual_manifest_loader)
    actual_batch_loader = test_assets.load_verified_manifest_batch

    def mutate_npz_after_load(manifest: TrackAssetManifest, path: str | Path):
        batch = actual_batch_loader(manifest, path)
        candidate = Path(path)
        candidate.write_bytes(candidate.read_bytes() + b"changed")
        return batch

    monkeypatch.setattr(test_assets, "load_verified_manifest_batch", mutate_npz_after_load)
    with pytest.raises(TrackAssetError, match="NPZ changed"):
        test_assets.load_verified_test_pool(asset.config, asset_directory=asset.directory)


def test_loader_rejects_asset_tamper_and_manifest_order_drift(
    synthetic_test_asset: SyntheticTestAsset,
) -> None:
    asset = synthetic_test_asset
    asset_path = asset.directory / "test.npz"
    asset_path.write_bytes(asset_path.read_bytes() + b"tampered")
    with pytest.raises(TrackAssetError, match="NPZ SHA-256"):
        test_assets.load_verified_test_pool(asset.config, asset_directory=asset.directory)

    save_track_batch_npz(stack_tracks(asset.tracks), asset_path)
    reordered = replace(asset.manifest, tracks=tuple(reversed(asset.manifest.tracks)))
    write_track_asset_manifest(reordered, asset.directory / "test.json")
    with pytest.raises(TrackAssetError, match=r"strictly increasing|strict manifest order"):
        test_assets.load_verified_test_pool(asset.config, asset_directory=asset.directory)


@pytest.mark.parametrize(
    ("changes"),
    [
        {"level_id": 0},
        {"track_count": 19},
        {"seed_start": 2_000_001},
        {"seed_stop": 2_999_999},
        {"manifest_file": "other.json"},
        {"asset_file": "other.npz"},
        {"package_asset": False},
    ],
)
def test_loader_rejects_official_test_spec_drift_before_asset_access(
    monkeypatch: pytest.MonkeyPatch,
    synthetic_test_asset: SyntheticTestAsset,
    changes: dict[str, object],
) -> None:
    official_spec = official_assets.official_track_split_spec("test")
    drifted = replace(official_spec, **changes)
    monkeypatch.setattr(test_assets, "official_track_split_spec", lambda split: drifted)
    monkeypatch.setattr(
        test_assets,
        "load_track_asset_manifest",
        lambda path: pytest.fail(f"asset opened before spec rejection: {path}"),
    )

    with pytest.raises(TrackAssetError, match=r"locked v0\.1 contract"):
        test_assets.load_verified_test_pool(
            synthetic_test_asset.config,
            asset_directory=synthetic_test_asset.directory,
        )


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("generator_version", "wrong-generator", "generator"),
        (
            "capacity",
            TrackCapacity(max_track_points=641, max_checkpoints=48),
            "capacity",
        ),
    ],
)
def test_loader_rejects_generator_or_capacity_drift(
    synthetic_test_asset: SyntheticTestAsset,
    field: str,
    value: object,
    message: str,
) -> None:
    asset = synthetic_test_asset
    drifted = replace(asset.manifest, **{field: value})
    write_track_asset_manifest(drifted, asset.directory / "test.json")

    with pytest.raises(TrackAssetError, match=message):
        test_assets.load_verified_test_pool(asset.config, asset_directory=asset.directory)


def test_loader_rejects_seed_outside_locked_test_namespace(
    synthetic_test_asset: SyntheticTestAsset,
) -> None:
    asset = synthetic_test_asset
    first = replace(asset.manifest.tracks[0], seed=1_999_999)
    drifted = replace(asset.manifest, tracks=(first, *asset.manifest.tracks[1:]))
    write_track_asset_manifest(drifted, asset.directory / "test.json")

    with pytest.raises(TrackAssetError, match="namespace"):
        test_assets.load_verified_test_pool(asset.config, asset_directory=asset.directory)


def test_loader_and_evidence_reject_type_or_identity_contradictions(
    synthetic_test_asset: SyntheticTestAsset,
) -> None:
    asset = synthetic_test_asset
    with pytest.raises(TypeError, match="ProjectConfig"):
        test_assets.load_verified_test_pool(object())  # type: ignore[arg-type]

    result = test_assets.load_verified_test_pool(asset.config, asset_directory=asset.directory)
    assert test_assets.TestPoolAccessEvidence.from_mapping(result.evidence.to_dict()) == (
        result.evidence
    )
    with pytest.raises(ValueError, match="Train access"):
        replace(result.evidence, loader_accessed_train=True)
    with pytest.raises(ValueError, match="Validation access"):
        replace(result.evidence, loader_accessed_validation=True)
    with pytest.raises(ValueError, match="path-sanitized"):
        replace(result.evidence, manifest_file="private/test.json")
    with pytest.raises(ValueError, match="bind the ordered"):
        replace(result.evidence, track_ids_sha256="0" * 64)
    with pytest.raises(ValueError, match="strict manifest order"):
        replace(result.evidence, track_ids=tuple(reversed(result.evidence.track_ids)))
    invalid_mapping = result.evidence.to_dict()
    invalid_mapping["capacity"]["unexpected"] = 1
    with pytest.raises(ValueError, match="capacity mapping"):
        test_assets.TestPoolAccessEvidence.from_mapping(invalid_mapping)

    validation_pool = TrackPool(
        benchmark_version=result.pool.benchmark_version,
        generator_version=result.pool.generator_version,
        split="validation",
        batch=result.pool.batch,
    )
    with pytest.raises(TypeError, match="Test TrackPool"):
        test_assets.VerifiedTestPool(validation_pool, result.evidence)
    with pytest.raises(ValueError, match="geometry"):
        test_assets.VerifiedTestPool(
            result.pool,
            replace(result.evidence, geometry_hashes_sha256="1" * 64),
        )


def test_loader_has_no_split_selector_or_general_asset_loading_path() -> None:
    signature = inspect.signature(test_assets.load_verified_test_pool)
    assert tuple(signature.parameters) == ("project_config", "asset_directory")
    assert signature.parameters["asset_directory"].kind is inspect.Parameter.KEYWORD_ONLY

    source = (PROJECT_ROOT / "controller_learning/evaluation/test_assets.py").read_text(
        encoding="utf-8"
    )
    for forbidden in (
        "verify_official_track_assets",
        "load_manifest_track_batch",
        "load_track_batch_npz",
        "load_verified_train_pool",
        "load_verified_validation_pool",
    ):
        assert forbidden not in source
    assert 'official_track_split_spec("test")' in source


def test_locked_contract_type_is_the_official_test_spec() -> None:
    spec = test_assets._locked_test_spec()
    assert isinstance(spec, OfficialTrackSplitSpec)
    assert spec.split == "test"
