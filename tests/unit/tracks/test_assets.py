"""Tests for strict deterministic Track assets and split manifests."""

from __future__ import annotations

import json
import zipfile
from dataclasses import replace

import numpy as np
import pytest

from controller_learning.tracks.assets import (
    TRACK_ASSET_SCHEMA_VERSION,
    TrackAssetError,
    TrackAssetManifest,
    TrackAssetRecord,
    load_manifest_track_batch,
    load_track_asset_manifest,
    load_track_batch_npz,
    save_track_batch_npz,
    write_track_asset_manifest,
)
from controller_learning.tracks.hashing import track_batch_geometry_sha256
from controller_learning.tracks.level0 import build_level0_track
from controller_learning.tracks.types import TrackBatch, stack_tracks


def _batch() -> TrackBatch:
    track = replace(build_level0_track(), seed=41)
    return stack_tracks([track])


def _manifest(batch: TrackBatch, *, asset_sha256: str, geometry_sha256: str) -> TrackAssetManifest:
    return TrackAssetManifest(
        schema_version=TRACK_ASSET_SCHEMA_VERSION,
        benchmark_version="0.1",
        level_id=1,
        split="validation",
        generator_version="v0.1",
        geometry_validation_version="v0.1",
        driveability_protocol_version="v0.1",
        track_width_m=7.0,
        track_count=1,
        capacity=build_level0_track().capacity,
        asset_file="validation.npz",
        asset_sha256=asset_sha256,
        tracks=(
            TrackAssetRecord(
                seed=int(batch.seed[0]),
                geometry_sha256=geometry_sha256,
                geometry_validation="passed",
                driveability_validation="passed",
            ),
        ),
    )


def test_npz_serialization_is_byte_deterministic_and_round_trips(tmp_path) -> None:
    batch = _batch()
    first_path = tmp_path / "first.npz"
    second_path = tmp_path / "second.npz"

    first_digest = save_track_batch_npz(batch, first_path)
    second_digest = save_track_batch_npz(batch, second_path)
    loaded = load_track_batch_npz(
        first_path,
        expected_sha256=first_digest,
        expected_track_count=1,
        expected_capacity=build_level0_track().capacity,
    )

    assert first_path.read_bytes() == second_path.read_bytes()
    assert first_digest == second_digest
    for expected, actual in zip(batch, loaded, strict=True):
        assert np.array_equal(actual, expected)
        assert actual.dtype == expected.dtype
        assert not actual.flags.writeable


def test_npz_rejects_wrong_dtype_members_and_digest_mismatch(tmp_path) -> None:
    batch = _batch()
    wrong_dtype = batch._replace(seed=batch.seed.astype(np.int64))

    with pytest.raises(TrackAssetError, match="seed must use dtype uint32"):
        save_track_batch_npz(wrong_dtype, tmp_path / "wrong.npz")

    path = tmp_path / "valid.npz"
    save_track_batch_npz(batch, path)
    with pytest.raises(TrackAssetError, match="SHA-256"):
        load_track_batch_npz(path, expected_sha256="0" * 64)


def test_npz_rejects_wrong_shapes_and_nonzero_padding(tmp_path) -> None:
    batch = _batch()
    wrong_shape = batch._replace(centerline_m=batch.centerline_m[..., :1])
    padded_curvature = batch.curvature_1pm.copy()
    padded_curvature[0, int(batch.point_count[0])] = 1.0
    wrong_padding = batch._replace(curvature_1pm=padded_curvature)

    with pytest.raises(TrackAssetError, match="centerline_m must have shape"):
        save_track_batch_npz(wrong_shape, tmp_path / "wrong-shape.npz")
    with pytest.raises(TrackAssetError, match="curvature_1pm padding must be zero"):
        save_track_batch_npz(wrong_padding, tmp_path / "wrong-padding.npz")


def test_npz_rejects_unexpected_members(tmp_path) -> None:
    path = tmp_path / "tracks.npz"
    save_track_batch_npz(_batch(), path)
    with zipfile.ZipFile(path, mode="a", compression=zipfile.ZIP_STORED) as archive:
        archive.writestr("unexpected.npy", b"not an array")

    with pytest.raises(TrackAssetError, match="members do not exactly match"):
        load_track_batch_npz(path)


def test_manifest_and_asset_round_trip_with_full_integrity_check(tmp_path) -> None:
    batch = _batch()
    asset_path = tmp_path / "validation.npz"
    asset_digest = save_track_batch_npz(batch, asset_path)
    geometry_digest = track_batch_geometry_sha256(batch)[0]
    manifest = _manifest(
        batch,
        asset_sha256=asset_digest,
        geometry_sha256=geometry_digest,
    )
    first_path = tmp_path / "manifest.json"
    second_path = tmp_path / "manifest-copy.json"

    first_digest = write_track_asset_manifest(manifest, first_path)
    second_digest = write_track_asset_manifest(manifest, second_path)
    loaded_manifest, loaded_batch = load_manifest_track_batch(first_path)

    assert first_path.read_bytes() == second_path.read_bytes()
    assert first_digest == second_digest
    assert loaded_manifest == manifest
    assert np.array_equal(loaded_batch.seed, batch.seed)


def test_manifest_rejects_unknown_and_duplicate_json_keys(tmp_path) -> None:
    batch = _batch()
    asset_digest = save_track_batch_npz(batch, tmp_path / "validation.npz")
    manifest = _manifest(
        batch,
        asset_sha256=asset_digest,
        geometry_sha256=track_batch_geometry_sha256(batch)[0],
    )
    path = tmp_path / "manifest.json"
    write_track_asset_manifest(manifest, path)

    data = json.loads(path.read_text(encoding="utf-8"))
    data["unknown"] = True
    path.write_text(json.dumps(data), encoding="utf-8")
    with pytest.raises(TrackAssetError, match="unexpected keys: unknown"):
        load_track_asset_manifest(path)

    write_track_asset_manifest(manifest, path)
    duplicate = path.read_text(encoding="utf-8").replace(
        '"asset_file":',
        '"asset_file": "other.npz", "asset_file":',
        1,
    )
    path.write_text(duplicate, encoding="utf-8")
    with pytest.raises(TrackAssetError, match="duplicate JSON key: asset_file"):
        load_track_asset_manifest(path)


def test_manifest_rejects_mistyped_capacity(tmp_path) -> None:
    path = tmp_path / "manifest.json"
    path.write_text(
        json.dumps(
            {
                "asset_file": "validation.npz",
                "asset_sha256": "0" * 64,
                "benchmark_version": "0.1",
                "capacity": {"max_checkpoints": 48, "max_track_points": 640.0},
                "driveability_protocol_version": "v0.1",
                "generator_version": "v0.1",
                "geometry_validation_version": "v0.1",
                "level_id": 1,
                "schema_version": 1,
                "split": "validation",
                "track_count": 1,
                "track_width_m": 7.0,
                "tracks": [],
            }
        ),
        encoding="utf-8",
    )

    with pytest.raises(TrackAssetError, match="capacity values must be integers"):
        load_track_asset_manifest(path)


def test_manifest_asset_loader_detects_geometry_record_mismatch(tmp_path) -> None:
    batch = _batch()
    asset_digest = save_track_batch_npz(batch, tmp_path / "validation.npz")
    manifest = _manifest(
        batch,
        asset_sha256=asset_digest,
        geometry_sha256="0" * 64,
    )
    path = tmp_path / "manifest.json"
    write_track_asset_manifest(manifest, path)

    with pytest.raises(TrackAssetError, match="geometry hashes"):
        load_manifest_track_batch(path)


def test_manifest_enforces_split_level_and_reserved_seed_contracts() -> None:
    batch = _batch()
    template = _manifest(
        batch,
        asset_sha256="0" * 64,
        geometry_sha256=track_batch_geometry_sha256(batch)[0],
    )

    with pytest.raises(TrackAssetError, match="belong to Level 1"):
        replace(template, level_id=0)
    with pytest.raises(TrackAssetError, match="reserved Level 0 seed"):
        replace(
            template,
            tracks=(replace(template.tracks[0], seed=np.iinfo(np.uint32).max),),
        )
