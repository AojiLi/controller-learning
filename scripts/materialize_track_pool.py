"""Reproduce and verify the official local Level 1 training Track pool."""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from controller_learning.config import load_project_config
from controller_learning.tracks.official_assets import (
    DEFAULT_TRAIN_CACHE,
    ProgressCallback,
    materialize_official_train_cache,
)

PROJECT_ROOT = Path(__file__).resolve().parents[1]


@dataclass(frozen=True, slots=True)
class MaterializeTrackPoolOptions:
    """Validated paths and replacement policy for training-pool materialization."""

    project_root: Path = PROJECT_ROOT
    asset_directory: Path | None = None
    output: Path = DEFAULT_TRAIN_CACHE
    force: bool = False


def _parse_args(argv: list[str] | None = None) -> MaterializeTrackPoolOptions:
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
        "--output",
        type=Path,
        default=DEFAULT_TRAIN_CACHE,
        help=f"Local uncompressed NPZ cache path (default: {DEFAULT_TRAIN_CACHE})",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Regenerate and atomically replace an existing verified or invalid cache",
    )
    args = parser.parse_args(argv)
    return MaterializeTrackPoolOptions(
        project_root=args.project_root,
        asset_directory=args.asset_directory,
        output=args.output,
        force=args.force,
    )


def _run(
    options: MaterializeTrackPoolOptions,
    *,
    progress: ProgressCallback | None = None,
) -> dict[str, Any]:
    config = load_project_config(options.project_root)
    result = materialize_official_train_cache(
        config,
        asset_directory=options.asset_directory,
        output=options.output,
        force=options.force,
        progress=progress,
    )
    return {
        "output": str(result.path),
        "reused": result.reused,
        "sha256": result.sha256,
        "track_count": result.track_count,
    }


def _progress(completed: int, total: int, seed: int) -> None:
    if completed == 1 or completed == total or completed % 100 == 0:
        print(
            f"regenerated {completed}/{total} Tracks (latest seed: {seed})",
            file=sys.stderr,
            flush=True,
        )


def main() -> None:
    """Materialize the cache and print one machine-readable summary."""

    print(json.dumps(_run(_parse_args(), progress=_progress), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
