"""Print a JSON report for the M0 NVIDIA dependency smoke check."""

from __future__ import annotations

import json

from controller_learning.diagnostics import inspect_gpu_environment


def main() -> None:
    """Run the GPU dependency check and print its report."""

    print(json.dumps(inspect_gpu_environment(), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
