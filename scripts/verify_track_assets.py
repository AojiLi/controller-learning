"""Verify the official Track manifests, fixed assets, and optional training cache."""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from controller_learning.config import load_project_config
from controller_learning.tracks.official_assets import (
    DEFAULT_TRAIN_CACHE,
    OfficialAssetVerification,
    verify_official_track_assets,
)

PROJECT_ROOT = Path(__file__).resolve().parents[1]


@dataclass(frozen=True, slots=True)
class VerifyTrackAssetsOptions:
    """Validated paths and cache policy for the verification command."""

    project_root: Path = PROJECT_ROOT
    asset_directory: Path | None = None
    train_cache: Path = DEFAULT_TRAIN_CACHE
    require_train_cache: bool = False


def _parse_args(argv: list[str] | None = None) -> VerifyTrackAssetsOptions:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--project-root",
        type=Path,
        default=PROJECT_ROOT,
        help="Repository root containing configs/ (default: script parent)",
    )
    parser.add_argument(
        "--asset-directory",
        type=Path,
        default=None,
        help="Override the installed package's versioned Track asset directory",
    )
    parser.add_argument(
        "--train-cache",
        type=Path,
        default=DEFAULT_TRAIN_CACHE,
        help=f"Local training-pool cache (default: {DEFAULT_TRAIN_CACHE})",
    )
    parser.add_argument(
        "--require-train-cache",
        action="store_true",
        help="Fail when the local training-pool cache is absent",
    )
    args = parser.parse_args(argv)
    return VerifyTrackAssetsOptions(
        project_root=args.project_root,
        asset_directory=args.asset_directory,
        train_cache=args.train_cache,
        require_train_cache=args.require_train_cache,
    )


def _report(verification: OfficialAssetVerification) -> dict[str, Any]:
    return {
        "asset_directory": str(verification.asset_directory),
        "benchmark_version": verification.manifests["train"].benchmark_version,
        "fixed_assets_verified": sorted(verification.fixed_batches),
        "split_track_counts": {
            split: manifest.track_count
            for split, manifest in sorted(verification.manifests.items())
        },
        "train_cache": {
            "path": (
                None
                if verification.train_cache_path is None
                else str(verification.train_cache_path)
            ),
            "verified": verification.train_cache_verified,
        },
    }


def _run(options: VerifyTrackAssetsOptions) -> dict[str, Any]:
    config = load_project_config(options.project_root)
    verification = verify_official_track_assets(
        config,
        asset_directory=options.asset_directory,
        train_cache_path=options.train_cache,
        require_train_cache=options.require_train_cache,
    )
    return _report(verification)


def main() -> None:
    """Verify assets and print one machine-readable summary."""

    print(json.dumps(_run(_parse_args()), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
