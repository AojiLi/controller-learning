"""Tests for strict project configuration."""

from __future__ import annotations

from dataclasses import FrozenInstanceError, replace
from pathlib import Path

import pytest

from controller_learning.config import (
    ConfigError,
    load_project_config,
    load_track_config,
    load_vehicle_config,
)

PROJECT_ROOT = Path(__file__).parents[2]


def test_repository_configuration_is_cross_validated() -> None:
    config = load_project_config(PROJECT_ROOT)

    assert config.vehicle.simulation.physics_steps_per_control == 10
    assert config.benchmark.official_level == 1
    assert config.benchmark.test_track_count == 20
    assert [level.level_id for level in config.levels] == [0, 1]
    assert config.levels[0].track_source == "fixed"
    assert config.levels[1].track_source == "procedural_pool"
    assert config.track.representation.arc_spacing_m == 1.0
    assert config.track.representation.max_track_points == 640
    assert config.track.representation.max_checkpoints == 48
    assert config.track.required_track_points == 601
    assert config.track.required_checkpoints == 40
    assert config.track.generator.generator_version == "spike-v1"
    assert config.track.validation.max_abs_curvature_1pm == pytest.approx(1.0 / 15.0)
    assert config.track.race.projection_backward_segments == 4
    assert config.track.race.projection_forward_segments == 12


def test_configuration_is_immutable() -> None:
    config = load_project_config(PROJECT_ROOT)

    with pytest.raises((FrozenInstanceError, AttributeError)):
        config.vehicle.vehicle.mass_kg = 1.0  # type: ignore[misc]


def test_unknown_configuration_key_is_rejected(tmp_path: Path) -> None:
    source = PROJECT_ROOT / "configs" / "vehicle.toml"
    candidate = tmp_path / "vehicle.toml"
    candidate.write_text(source.read_text() + "\nunknown = 1\n")

    with pytest.raises(ConfigError, match="unexpected keys: unknown"):
        load_vehicle_config(candidate)


def test_unknown_track_configuration_key_is_rejected(tmp_path: Path) -> None:
    source = PROJECT_ROOT / "configs" / "track.toml"
    candidate = tmp_path / "track.toml"
    text = source.read_text().replace(
        "projection_forward_segments = 12",
        "projection_forward_segments = 12\nunknown = 1",
    )
    candidate.write_text(text)

    with pytest.raises(ConfigError, match="unexpected keys: unknown"):
        load_track_config(candidate)


def test_non_integral_control_ratio_is_rejected(tmp_path: Path) -> None:
    source = PROJECT_ROOT / "configs" / "vehicle.toml"
    candidate = tmp_path / "vehicle.toml"
    text = source.read_text().replace("physics_dt_s = 0.005", "physics_dt_s = 0.007")
    candidate.write_text(text)

    with pytest.raises(ConfigError, match="must be an integer"):
        load_vehicle_config(candidate)


def test_wrong_scalar_type_is_rejected(tmp_path: Path) -> None:
    source = PROJECT_ROOT / "configs" / "vehicle.toml"
    candidate = tmp_path / "vehicle.toml"
    text = source.read_text().replace("mass_kg = 1200.0", 'mass_kg = "heavy"')
    candidate.write_text(text)

    with pytest.raises(ConfigError, match=r"vehicle\.mass_kg must be a number"):
        load_vehicle_config(candidate)


def test_missing_required_key_is_rejected(tmp_path: Path) -> None:
    source = PROJECT_ROOT / "configs" / "vehicle.toml"
    candidate = tmp_path / "vehicle.toml"
    text = source.read_text().replace("max_speed_mps = 15.0\n", "")
    candidate.write_text(text)

    with pytest.raises(ConfigError, match="missing keys: max_speed_mps"):
        load_vehicle_config(candidate)


def test_missing_required_track_key_is_rejected(tmp_path: Path) -> None:
    source = PROJECT_ROOT / "configs" / "track.toml"
    candidate = tmp_path / "track.toml"
    text = source.read_text().replace("projection_forward_segments = 12\n", "")
    candidate.write_text(text)

    with pytest.raises(ConfigError, match="missing keys: projection_forward_segments"):
        load_track_config(candidate)


def test_track_point_capacity_must_cover_maximum_valid_length(tmp_path: Path) -> None:
    source = PROJECT_ROOT / "configs" / "track.toml"
    candidate = tmp_path / "track.toml"
    text = source.read_text().replace("max_track_points = 640", "max_track_points = 600")
    candidate.write_text(text)

    with pytest.raises(ConfigError, match="requires at least 601"):
        load_track_config(candidate)


def test_track_checkpoint_capacity_must_cover_maximum_valid_length(tmp_path: Path) -> None:
    source = PROJECT_ROOT / "configs" / "track.toml"
    candidate = tmp_path / "track.toml"
    text = source.read_text().replace("max_checkpoints = 48", "max_checkpoints = 39")
    candidate.write_text(text)

    with pytest.raises(ConfigError, match="requires at least 40"):
        load_track_config(candidate)


def test_generator_radius_must_match_validation_limit(tmp_path: Path) -> None:
    source = PROJECT_ROOT / "configs" / "track.toml"
    candidate = tmp_path / "track.toml"
    text = source.read_text().replace("min_radius_m = 52.0", "min_radius_m = 10.0")
    candidate.write_text(text)

    with pytest.raises(ConfigError, match="minimum radius cannot be smaller"):
        load_track_config(candidate)


def test_effective_track_half_width_must_remain_positive() -> None:
    config = load_project_config(PROJECT_ROOT)
    unsafe_race = replace(config.track.race, safety_margin_m=2.7)
    unsafe_track = replace(config.track, race=unsafe_race)

    with pytest.raises(ConfigError, match="effective half-width must remain positive"):
        replace(config, track=unsafe_track)


def test_invalid_physical_range_is_rejected(tmp_path: Path) -> None:
    source = PROJECT_ROOT / "configs" / "vehicle.toml"
    candidate = tmp_path / "vehicle.toml"
    text = source.read_text().replace("mass_kg = 1200.0", "mass_kg = -1.0")
    candidate.write_text(text)

    with pytest.raises(
        ConfigError,
        match=r"vehicle\.mass_kg must be a finite positive number",
    ):
        load_vehicle_config(candidate)


def test_wrong_configuration_suffix_is_rejected(tmp_path: Path) -> None:
    candidate = tmp_path / "vehicle.json"
    candidate.write_text("{}")

    with pytest.raises(ConfigError, match=r"must use the \.toml suffix"):
        load_vehicle_config(candidate)


def test_missing_configuration_reports_domain_error(tmp_path: Path) -> None:
    with pytest.raises(ConfigError, match="does not exist"):
        load_vehicle_config(tmp_path / "missing.toml")


def test_invalid_toml_reports_domain_error(tmp_path: Path) -> None:
    candidate = tmp_path / "vehicle.toml"
    candidate.write_text("[vehicle\n")

    with pytest.raises(ConfigError, match="invalid TOML"):
        load_vehicle_config(candidate)
