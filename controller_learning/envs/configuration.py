"""Adapters from repository configuration to executable Challenge rules."""

from __future__ import annotations

from controller_learning.config.models import ProjectConfig
from controller_learning.envs.race_core import RaceCoreConfig


def race_core_config_from_project(config: ProjectConfig) -> RaceCoreConfig:
    """Build the Race Core rules represented by the project configuration."""

    race = config.track.race
    episode = config.benchmark.episode
    return RaceCoreConfig(
        control_dt_s=config.vehicle.simulation.control_dt_s,
        vehicle_width_m=config.vehicle.vehicle.vehicle_width_m,
        safety_margin_m=race.safety_margin_m,
        projection_backward_segments=race.projection_backward_segments,
        projection_forward_segments=race.projection_forward_segments,
        min_timeout_s=episode.minimum_timeout_s,
        timeout_reference_speed_mps=episode.timeout_reference_speed_mps,
    )


__all__ = ["race_core_config_from_project"]
