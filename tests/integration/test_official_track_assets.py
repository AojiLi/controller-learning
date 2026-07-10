"""Integration contract for the committed official v0.1 Track assets."""

from __future__ import annotations

from pathlib import Path

from controller_learning.config import load_project_config
from controller_learning.tracks.official_assets import verify_official_track_assets

PROJECT_ROOT = Path(__file__).parents[2]


def test_official_manifests_and_fixed_assets_verify_without_local_train_cache(
    tmp_path: Path,
) -> None:
    project = load_project_config(PROJECT_ROOT)
    absent_cache = tmp_path / "absent-train-pool.npz"
    verification = verify_official_track_assets(
        project,
        train_cache_path=absent_cache,
        require_train_cache=False,
    )

    assert tuple(verification.manifests) == ("level0", "train", "validation", "test")
    assert {split: manifest.track_count for split, manifest in verification.manifests.items()} == {
        "level0": 1,
        "train": 10_000,
        "validation": 100,
        "test": 20,
    }
    assert tuple(verification.fixed_batches) == ("level0", "validation", "test")
    assert verification.train_cache_verified is False
    assert verification.train_cache_path == absent_cache
