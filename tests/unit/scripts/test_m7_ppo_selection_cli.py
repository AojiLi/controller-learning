"""CPU protocol tests for the dedicated M7 Validation-selection process."""

from __future__ import annotations

import struct
import subprocess
import sys
from pathlib import Path

import pytest

from scripts import benchmark_m7_ppo as benchmark

PROJECT_ROOT = Path(__file__).parents[3]


def test_cli_import_sets_allocator_policy_and_does_not_import_gpu_or_training_modules() -> None:
    completed = subprocess.run(
        (
            sys.executable,
            "-c",
            (
                "import os,sys; import scripts.benchmark_m7_ppo; "
                "assert os.environ['CUDA_DEVICE_ORDER']=='PCI_BUS_ID'; "
                "assert os.environ['XLA_PYTHON_CLIENT_PREALLOCATE']=='false'; "
                "assert 'torch' not in sys.modules; "
                "assert 'jax' not in sys.modules; "
                "assert 'controller_learning.rl.trainer' not in sys.modules; "
                "assert 'controller_learning.rl.policy' not in sys.modules"
            ),
        ),
        cwd=PROJECT_ROOT,
        check=False,
        capture_output=True,
        text=True,
    )
    assert completed.returncode == 0, completed.stderr


def test_validation_guard_allows_only_two_read_only_files(tmp_path: Path) -> None:
    official = tmp_path / "official" / "v0.1"
    cache_root = tmp_path / "cache"
    official.mkdir(parents=True)
    cache_root.mkdir()
    validation_manifest = official / "validation.json"
    validation_asset = official / "validation.npz"
    train_asset = official / "train.npz"
    test_asset = official / "test.npz"
    cache_asset = cache_root / "train_pool.npz"
    for path in (
        validation_manifest,
        validation_asset,
        train_asset,
        test_asset,
        cache_asset,
    ):
        path.write_bytes(b"asset")

    guard = benchmark.OfficialValidationAssetAccessGuard(
        official_track_root=official.parent,
        validation_manifest=validation_manifest,
        validation_asset=validation_asset,
        track_cache_root=cache_root,
    )
    guard.install()
    assert guard.evidence(validation_loaded=False)["pre_validation_open_event_count"] == 0
    with pytest.raises(benchmark.ForbiddenSelectionAssetAccessError, match="remain disabled"):
        validation_manifest.read_bytes()
    guard.enable_validation_reads()
    assert validation_manifest.read_bytes() == b"asset"
    assert validation_asset.read_bytes() == b"asset"

    for forbidden in (train_asset, test_asset, cache_asset):
        with pytest.raises(benchmark.ForbiddenSelectionAssetAccessError):
            forbidden.read_bytes()
    with pytest.raises(benchmark.ForbiddenSelectionAssetAccessError, match="read-only"):
        validation_asset.write_bytes(b"mutated")

    evidence = guard.evidence(validation_loaded=True)
    assert evidence["opened_splits"] == ["validation"]
    assert evidence["opened_path_categories"] == [
        "official_validation_asset",
        "official_validation_manifest",
    ]
    assert evidence["train_opened"] is False
    assert evidence["test_opened"] is False
    assert evidence["track_cache_opened"] is False
    assert evidence["denied_event_count"] == 5


def test_guard_blocks_test_before_read_in_a_dedicated_process(tmp_path: Path) -> None:
    official = tmp_path / "official" / "v0.1"
    cache_root = tmp_path / "cache"
    official.mkdir(parents=True)
    cache_root.mkdir()
    validation_manifest = official / "validation.json"
    validation_asset = official / "validation.npz"
    test_asset = official / "test.npz"
    for path in (validation_manifest, validation_asset, test_asset):
        path.write_bytes(b"asset")
    program = f"""
from pathlib import Path
from scripts.benchmark_m7_ppo import (
    ForbiddenSelectionAssetAccessError,
    OfficialValidationAssetAccessGuard,
)
guard = OfficialValidationAssetAccessGuard(
    official_track_root=Path({str(official.parent)!r}),
    validation_manifest=Path({str(validation_manifest)!r}),
    validation_asset=Path({str(validation_asset)!r}),
    track_cache_root=Path({str(cache_root)!r}),
)
guard.install()
try:
    Path({str(test_asset)!r}).read_bytes()
except ForbiddenSelectionAssetAccessError:
    print('blocked-before-read')
else:
    raise AssertionError('Test read was not blocked')
"""
    completed = subprocess.run(
        (sys.executable, "-c", program),
        cwd=PROJECT_ROOT,
        check=False,
        capture_output=True,
        text=True,
    )
    assert completed.returncode == 0, completed.stderr
    assert completed.stdout.strip() == "blocked-before-read"


def test_curve_metrics_and_png_are_strict_deterministic_and_fixed_size(tmp_path: Path) -> None:
    metrics = tmp_path / "metrics.csv"
    metrics.write_text(
        "update_index,cumulative_success_rate,cumulative_mean_valid_reward\n"
        "1,0.1,-2.0\n"
        "2,0.2,-1.0\n",
        encoding="utf-8",
    )
    series = benchmark._read_curve_metrics(metrics, expected_updates=2)
    first = benchmark.deterministic_training_curve_png(
        *series,
        width_px=1200,
        height_px=800,
        dpi=100,
    )
    second = benchmark.deterministic_training_curve_png(
        *series,
        width_px=1200,
        height_px=800,
        dpi=100,
    )
    assert first == second
    assert first.startswith(b"\x89PNG\r\n\x1a\n")
    assert struct.unpack(">II", first[16:24]) == (1200, 800)

    metrics.write_text(
        "update_index,cumulative_success_rate,cumulative_mean_valid_reward\n"
        "1,1.1,-2.0\n"
        "2,0.2,-1.0\n",
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match=r"\[0, 1\]"):
        benchmark._read_curve_metrics(metrics, expected_updates=2)


def test_formal_source_has_no_training_path_and_pixi_exposes_one_selection_task() -> None:
    source = (PROJECT_ROOT / "scripts/benchmark_m7_ppo.py").read_text(encoding="utf-8")
    for forbidden in (
        "controller_learning.rl.trainer",
        "PpoUpdater",
        "load_verified_train_pool",
        "PublicRewardShapingVecEnv",
        "optimizer.step",
        ".backward(",
    ):
        assert forbidden not in source
    pyproject = (PROJECT_ROOT / "pyproject.toml").read_text(encoding="utf-8")
    assert pyproject.count("benchmark-m7-ppo =") == 1
    assert "python scripts/benchmark_m7_ppo.py" in pyproject
