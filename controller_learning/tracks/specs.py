"""Adapters from the repository configuration to executable Track specifications."""

from __future__ import annotations

from controller_learning.config.models import ProjectConfig
from controller_learning.tracks.generator import TrackGenerationSpec
from controller_learning.tracks.types import TrackCapacity
from controller_learning.tracks.validator import TrackValidationSpec


def _level1_track_width_m(config: ProjectConfig) -> float:
    """Return the fixed Level 1 width from a cross-validated project config."""

    return next(level.track_width_m for level in config.levels if level.level_id == 1)


def generation_spec_from_project(config: ProjectConfig) -> TrackGenerationSpec:
    """Build the generator specification represented by ``configs/track.toml``."""

    representation = config.track.representation
    generator = config.track.generator
    validation = config.track.validation
    return TrackGenerationSpec(
        min_control_points=generator.min_control_points,
        max_control_points=generator.max_control_points,
        min_radius_m=generator.min_radius_m,
        max_radius_m=generator.max_radius_m,
        angular_gap_jitter=generator.angular_gap_jitter,
        radial_perturbation=generator.radial_perturbation,
        width_m=_level1_track_width_m(config),
        arc_spacing_m=representation.arc_spacing_m,
        checkpoint_spacing_m=representation.checkpoint_spacing_m,
        min_length_m=validation.min_length_m,
        max_length_m=validation.max_length_m,
        start_window_m=validation.start_window_m,
        start_max_curvature_1pm=validation.start_max_curvature_1pm,
        generator_version=generator.generator_version,
        dense_samples_per_control_point=generator.dense_samples_per_control_point,
        arc_length_convergence_m=generator.arc_length_convergence_m,
        tail_merge_fraction=generator.tail_merge_fraction,
    )


def validation_spec_from_project(config: ProjectConfig) -> TrackValidationSpec:
    """Build the geometric validation specification from project configuration."""

    validation = config.track.validation
    return TrackValidationSpec(
        min_length_m=validation.min_length_m,
        max_length_m=validation.max_length_m,
        expected_width_m=_level1_track_width_m(config),
        max_abs_curvature_1pm=validation.max_abs_curvature_1pm,
        start_window_m=validation.start_window_m,
        start_max_abs_curvature_1pm=validation.start_max_curvature_1pm,
        min_nonlocal_centerline_clearance_m=(validation.min_nonlocal_centerline_clearance_m),
        local_arc_exclusion_m=validation.local_arc_exclusion_m,
    )


def track_capacity_from_project(config: ProjectConfig) -> TrackCapacity:
    """Build the fixed Track array capacity from project configuration."""

    representation = config.track.representation
    return TrackCapacity(
        max_track_points=representation.max_track_points,
        max_checkpoints=representation.max_checkpoints,
    )


__all__ = [
    "generation_spec_from_project",
    "track_capacity_from_project",
    "validation_spec_from_project",
]
