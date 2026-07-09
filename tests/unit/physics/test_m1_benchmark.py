"""Tests for the frozen M1 benchmark protocol helpers."""

import subprocess
from pathlib import Path

from controller_learning.physics.m1_benchmark import _git, scenario_action, select_largest_passing


def test_scenario_action_boundaries_are_deterministic() -> None:
    assert scenario_action("straight", 0.99, 0.0) == (0.0, 0.0)
    assert scenario_action("straight", 1.0, 0.0) == (0.0, 2.0)
    assert scenario_action("straight", 4.0, 0.0) == (0.0, 0.0)
    assert scenario_action("steer_left", 4.0, 5.0) == (0.2, 0.0)
    assert scenario_action("steer_right", 4.0, 5.0) == (-0.2, 0.0)
    assert scenario_action("brake", 5.0, 6.0) == (0.0, -4.0)
    assert scenario_action("action_limits", 0.0, 0.0) == (2.0, 10.0)
    assert scenario_action("action_limits", 0.5, 0.0) == (-2.0, -20.0)
    assert scenario_action("action_limits", 1.0, 0.0) == (0.0, 0.0)


def test_largest_passing_timestep_is_selected() -> None:
    assert select_largest_passing({0.01: False, 0.005: True, 0.002: True}) == 0.005
    assert select_largest_passing({0.01: True, 0.005: True, 0.002: True}) == 0.01
    assert select_largest_passing({0.01: False, 0.005: False, 0.002: False}) is None


def test_git_helper_uses_the_explicit_project_root(tmp_path: Path, monkeypatch) -> None:
    repository = tmp_path / "repository"
    repository.mkdir()
    subprocess.run(("git", "init", "--quiet"), cwd=repository, check=True)
    outside = tmp_path / "outside"
    outside.mkdir()
    monkeypatch.chdir(outside)

    assert Path(_git(repository, "rev-parse", "--show-toplevel")) == repository
