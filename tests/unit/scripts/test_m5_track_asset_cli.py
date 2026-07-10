"""Focused command-line tests for official Track verification and materialization."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

from scripts import materialize_track_pool, verify_track_assets


def test_verify_cli_defaults_to_optional_local_cache() -> None:
    options = verify_track_assets._parse_args([])

    assert options.project_root == verify_track_assets.PROJECT_ROOT
    assert options.asset_directory is None
    assert options.train_cache == Path(".track-cache/v0.1/train_pool.npz")
    assert not options.require_train_cache


def test_verify_cli_delegates_explicit_paths_and_reports_splits(
    monkeypatch, tmp_path: Path
) -> None:
    config = object()
    captured = {}
    manifests = {
        split: SimpleNamespace(benchmark_version="0.1", track_count=count)
        for split, count in (("level0", 1), ("train", 10_000), ("validation", 100), ("test", 20))
    }
    verification = SimpleNamespace(
        asset_directory=tmp_path / "assets",
        manifests=manifests,
        fixed_batches={"level0": object(), "validation": object(), "test": object()},
        train_cache_path=tmp_path / "cache.npz",
        train_cache_verified=True,
    )
    monkeypatch.setattr(verify_track_assets, "load_project_config", lambda root: config)

    def verify(received, **kwargs):
        captured["config"] = received
        captured.update(kwargs)
        return verification

    monkeypatch.setattr(verify_track_assets, "verify_official_track_assets", verify)
    options = verify_track_assets.VerifyTrackAssetsOptions(
        project_root=tmp_path,
        asset_directory=tmp_path / "assets",
        train_cache=tmp_path / "cache.npz",
        require_train_cache=True,
    )

    report = verify_track_assets._run(options)

    assert captured == {
        "config": config,
        "asset_directory": tmp_path / "assets",
        "train_cache_path": tmp_path / "cache.npz",
        "require_train_cache": True,
    }
    assert report["split_track_counts"] == {
        "level0": 1,
        "test": 20,
        "train": 10_000,
        "validation": 100,
    }
    assert report["train_cache"]["verified"] is True


def test_materialize_cli_defaults_and_delegates_force_policy(monkeypatch, tmp_path: Path) -> None:
    defaults = materialize_track_pool._parse_args([])
    assert defaults.output == Path(".track-cache/v0.1/train_pool.npz")
    assert not defaults.force

    config = object()
    callback = object()
    captured = {}
    monkeypatch.setattr(materialize_track_pool, "load_project_config", lambda root: config)

    def materialize(received, **kwargs):
        captured["config"] = received
        captured.update(kwargs)
        return SimpleNamespace(
            path=tmp_path / "train.npz",
            sha256="a" * 64,
            track_count=10_000,
            reused=False,
        )

    monkeypatch.setattr(
        materialize_track_pool,
        "materialize_official_train_cache",
        materialize,
    )
    options = materialize_track_pool.MaterializeTrackPoolOptions(
        project_root=tmp_path,
        asset_directory=tmp_path / "assets",
        output=tmp_path / "train.npz",
        force=True,
    )

    report = materialize_track_pool._run(options, progress=callback)  # type: ignore[arg-type]

    assert captured == {
        "config": config,
        "asset_directory": tmp_path / "assets",
        "output": tmp_path / "train.npz",
        "force": True,
        "progress": callback,
    }
    assert report == {
        "output": str(tmp_path / "train.npz"),
        "reused": False,
        "sha256": "a" * 64,
        "track_count": 10_000,
    }


def test_verify_cli_resolves_relative_paths_against_project_root(
    monkeypatch,
    tmp_path: Path,
) -> None:
    project_root = tmp_path / "repository"
    working_directory = tmp_path / "elsewhere"
    project_root.mkdir()
    working_directory.mkdir()
    monkeypatch.chdir(working_directory)
    captured = {}
    manifests = {
        split: SimpleNamespace(benchmark_version="0.1", track_count=count)
        for split, count in (("level0", 1), ("train", 10_000), ("validation", 100), ("test", 20))
    }
    monkeypatch.setattr(
        verify_track_assets,
        "load_project_config",
        lambda root: captured.setdefault("project_root", root),
    )

    def verify(config, **kwargs):
        captured.update(kwargs)
        return SimpleNamespace(
            asset_directory=kwargs["asset_directory"],
            manifests=manifests,
            fixed_batches={"level0": object(), "validation": object(), "test": object()},
            train_cache_path=kwargs["train_cache_path"],
            train_cache_verified=False,
        )

    monkeypatch.setattr(verify_track_assets, "verify_official_track_assets", verify)

    verify_track_assets._run(
        verify_track_assets.VerifyTrackAssetsOptions(
            project_root=Path("../repository"),
            asset_directory=Path("controller_learning/assets/tracks/v0.1"),
            train_cache=Path(".track-cache/v0.1/train_pool.npz"),
        )
    )

    assert captured["project_root"] == project_root
    assert captured["asset_directory"] == (project_root / "controller_learning/assets/tracks/v0.1")
    assert captured["train_cache_path"] == project_root / ".track-cache/v0.1/train_pool.npz"


def test_materialize_cli_resolves_relative_paths_against_project_root(
    monkeypatch,
    tmp_path: Path,
) -> None:
    project_root = tmp_path / "repository"
    working_directory = tmp_path / "elsewhere"
    project_root.mkdir()
    working_directory.mkdir()
    monkeypatch.chdir(working_directory)
    captured = {}
    monkeypatch.setattr(
        materialize_track_pool,
        "load_project_config",
        lambda root: captured.setdefault("project_root", root),
    )

    def materialize(config, **kwargs):
        captured.update(kwargs)
        return SimpleNamespace(
            path=kwargs["output"],
            sha256="a" * 64,
            track_count=10_000,
            reused=True,
        )

    monkeypatch.setattr(
        materialize_track_pool,
        "materialize_official_train_cache",
        materialize,
    )

    materialize_track_pool._run(
        materialize_track_pool.MaterializeTrackPoolOptions(
            project_root=Path("../repository"),
            asset_directory=Path("controller_learning/assets/tracks/v0.1"),
            output=Path(".track-cache/v0.1/train_pool.npz"),
        )
    )

    assert captured["project_root"] == project_root
    assert captured["asset_directory"] == (project_root / "controller_learning/assets/tracks/v0.1")
    assert captured["output"] == project_root / ".track-cache/v0.1/train_pool.npz"
