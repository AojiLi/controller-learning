"""Tests for project-configuration adapters used by the Race Core."""

from __future__ import annotations

from pathlib import Path

from controller_learning.config import load_project_config
from controller_learning.envs.configuration import race_core_config_from_project
from controller_learning.envs.race_core import RaceCoreConfig

PROJECT_ROOT = Path(__file__).parents[3]


def test_race_core_config_is_the_exact_project_protocol() -> None:
    project = load_project_config(PROJECT_ROOT)

    assert race_core_config_from_project(project) == RaceCoreConfig(
        control_dt_s=project.vehicle.simulation.control_dt_s,
        vehicle_width_m=project.vehicle.vehicle.vehicle_width_m,
        safety_margin_m=project.track.race.safety_margin_m,
        projection_backward_segments=4,
        projection_forward_segments=12,
        min_timeout_s=project.benchmark.episode.minimum_timeout_s,
        timeout_reference_speed_mps=(project.benchmark.episode.timeout_reference_speed_mps),
    )
