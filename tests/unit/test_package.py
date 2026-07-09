"""Tests for package metadata and import safety."""

import subprocess
import sys
from importlib.metadata import version

import controller_learning


def test_package_version_matches_distribution() -> None:
    assert controller_learning.__version__ == version("controller-learning")


def test_top_level_import_does_not_register_environment_early() -> None:
    assert "CarRacingEnv" not in controller_learning.__all__


def test_top_level_import_does_not_import_simulation_stacks() -> None:
    command = (
        "import sys; import controller_learning; "
        "assert all(name not in sys.modules for name in ('jax', 'mujoco', 'torch', 'warp'))"
    )
    subprocess.run([sys.executable, "-c", command], check=True)
