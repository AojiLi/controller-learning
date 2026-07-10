"""Run the formal isolated M2 GPU benchmark and write versioned evidence."""

from __future__ import annotations

import os

# The launcher and every child enforce this before JAX can be imported.
os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")

import argparse
from pathlib import Path

from controller_learning.physics.m2_benchmark import write_m2_report


def main() -> None:
    """Run all formal scales and exit nonzero unless every M2 gate passes."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("benchmarks/v0.1/gpu_report.json"),
        help="Strict JSON report path",
    )
    parser.add_argument(
        "--timeout-s",
        type=float,
        default=7_200.0,
        help="Maximum wall time for each fresh scale worker",
    )
    args = parser.parse_args()
    project_root = Path(__file__).resolve().parents[1]
    report = write_m2_report(
        project_root,
        args.output,
        timeout_s=args.timeout_s,
    )
    print(f"M2 status: {report['status']}")
    for result in report["scale_results"]:
        throughput = result.get("timing", {}).get("transitions_per_second")
        print(f"worlds={result['num_worlds']} status={result['status']} transitions/s={throughput}")
    if report["status"] != "pass":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
