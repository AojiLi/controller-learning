"""Measure deterministic M3 track-capacity candidates across a seed range."""

from __future__ import annotations

import argparse
from pathlib import Path

from controller_learning.tracks.capacity_benchmark import (
    DEFAULT_ARC_SPACINGS_M,
    write_track_capacity_report,
)


def main() -> None:
    """Run the offline capacity sweep and write its strict JSON report."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--count",
        type=int,
        default=10_000,
        help="Number of contiguous seeds to sweep (default: 10000)",
    )
    parser.add_argument(
        "--start-seed",
        type=int,
        default=0,
        help="First seed in the contiguous range (default: 0)",
    )
    parser.add_argument(
        "--spacing",
        dest="spacings",
        type=float,
        nargs="+",
        default=None,
        help="One or more arc spacings in metres (default: 0.75 1.0 1.25)",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("benchmarks/v0.1/track_capacity_report.json"),
        help="Strict JSON report path",
    )
    args = parser.parse_args()
    spacings = DEFAULT_ARC_SPACINGS_M if args.spacings is None else tuple(args.spacings)
    report = write_track_capacity_report(
        args.output,
        seed_start=args.start_seed,
        seed_count=args.count,
        arc_spacings_m=spacings,
    )
    print(f"Wrote {args.output}")
    for result in report["spacing_results"]:
        selection = result["selected_capacity_candidate"]
        print(
            f"spacing={result['arc_spacing_m']}: "
            f"points={selection['max_track_points']}, "
            f"checkpoints={selection['max_checkpoints']}, "
            f"generated={result['generation']['succeeded_count']}/"
            f"{result['generation']['attempted_count']}"
        )


if __name__ == "__main__":
    main()
