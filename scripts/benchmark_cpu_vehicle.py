"""Run the formal M1 CPU vehicle benchmark and write versioned evidence."""

from __future__ import annotations

import argparse
from pathlib import Path

from controller_learning.physics.m1_benchmark import write_m1_report


def main() -> None:
    """Execute the fixed M1 protocol and fail if no timestep passes."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("benchmarks/v0.1/m1_cpu_report.json"),
        help="Strict JSON report path",
    )
    args = parser.parse_args()
    project_root = Path(__file__).resolve().parents[1]
    report = write_m1_report(project_root, args.output)
    selection = report["selection"]
    print(f"M1 status: {report['status']}")
    print(f"Selected physics_dt_s: {selection['selected_physics_dt_s']}")
    if not selection["m1_passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
