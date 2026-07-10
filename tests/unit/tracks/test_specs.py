"""Tests for project-configuration adapters used by the Track subsystem."""

from __future__ import annotations

from pathlib import Path

import pytest

from controller_learning.config import load_project_config
from controller_learning.tracks.generator import generate_track_candidate, pack_track
from controller_learning.tracks.specs import (
    generation_spec_from_project,
    track_capacity_from_project,
    validation_spec_from_project,
)
from controller_learning.tracks.validator import validate_track_candidate

PROJECT_ROOT = Path(__file__).parents[3]


@pytest.fixture(scope="module")
def project_config():
    return load_project_config(PROJECT_ROOT)


def test_generation_spec_maps_every_configured_field(project_config) -> None:
    spec = generation_spec_from_project(project_config)
    representation = project_config.track.representation
    generator = project_config.track.generator
    validation = project_config.track.validation
    level1 = next(level for level in project_config.levels if level.level_id == 1)

    assert spec.min_control_points == generator.min_control_points
    assert spec.max_control_points == generator.max_control_points
    assert spec.min_radius_m == generator.min_radius_m
    assert spec.max_radius_m == generator.max_radius_m
    assert spec.angular_gap_jitter == generator.angular_gap_jitter
    assert spec.radial_perturbation == generator.radial_perturbation
    assert spec.width_m == level1.track_width_m
    assert spec.arc_spacing_m == representation.arc_spacing_m
    assert spec.checkpoint_spacing_m == representation.checkpoint_spacing_m
    assert spec.min_length_m == validation.min_length_m
    assert spec.max_length_m == validation.max_length_m
    assert spec.start_window_m == validation.start_window_m
    assert spec.start_max_curvature_1pm == validation.start_max_curvature_1pm
    assert spec.generator_version == generator.generator_version
    assert spec.dense_samples_per_control_point == generator.dense_samples_per_control_point
    assert spec.arc_length_convergence_m == generator.arc_length_convergence_m
    assert spec.tail_merge_fraction == generator.tail_merge_fraction


def test_validation_spec_maps_every_configured_field(project_config) -> None:
    spec = validation_spec_from_project(project_config)
    validation = project_config.track.validation
    level1 = next(level for level in project_config.levels if level.level_id == 1)

    assert spec.min_length_m == validation.min_length_m
    assert spec.max_length_m == validation.max_length_m
    assert spec.expected_width_m == level1.track_width_m
    assert spec.max_abs_curvature_1pm == validation.max_abs_curvature_1pm
    assert spec.start_window_m == validation.start_window_m
    assert spec.start_max_abs_curvature_1pm == validation.start_max_curvature_1pm
    assert (
        spec.min_nonlocal_centerline_clearance_m == validation.min_nonlocal_centerline_clearance_m
    )
    assert spec.local_arc_exclusion_m == validation.local_arc_exclusion_m


def test_configured_specs_generate_validate_and_pack(project_config) -> None:
    generation_spec = generation_spec_from_project(project_config)
    validation_spec = validation_spec_from_project(project_config)
    capacity = track_capacity_from_project(project_config)

    candidate = generate_track_candidate(42, generation_spec)
    result = validate_track_candidate(candidate, validation_spec)
    track = pack_track(candidate, capacity)

    assert result.valid, result.reasons
    assert track.capacity == capacity
    assert track.generator_version == generation_spec.generator_version
    assert track.width_m == pytest.approx(generation_spec.width_m)


def test_capacity_maps_both_representation_limits(project_config) -> None:
    capacity = track_capacity_from_project(project_config)
    representation = project_config.track.representation

    assert capacity.max_track_points == representation.max_track_points
    assert capacity.max_checkpoints == representation.max_checkpoints
