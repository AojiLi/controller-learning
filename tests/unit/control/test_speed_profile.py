"""Tests for the shared curvature and braking speed profile."""

from __future__ import annotations

import inspect

import numpy as np
import pytest

from controller_learning.control import speed_profile
from controller_learning.control.speed_profile import curvature_speed_profile


def _profile(curvature, distances):
    return curvature_speed_profile(
        curvature,
        distances,
        minimum_speed_mps=2.5,
        maximum_speed_mps=5.0,
        maximum_lateral_acceleration_mps2=1.5,
        braking_deceleration_mps2=3.0,
    )


def test_straight_profile_uses_maximum_speed_and_is_readonly() -> None:
    profile = _profile(np.zeros(4), np.arange(4.0))

    np.testing.assert_array_equal(profile, np.full(4, 5.0))
    assert not profile.flags.writeable


def test_curve_limit_and_backward_braking_envelope_are_applied() -> None:
    profile = _profile((0.0, 0.0, 0.2), (0.0, 1.0, 2.0))
    curve_limit = max(2.5, np.sqrt(1.5 / 0.2))

    assert profile[2] == pytest.approx(curve_limit)
    assert profile[1] == pytest.approx(np.sqrt(curve_limit**2 + 6.0))
    assert profile[0] == pytest.approx(np.sqrt(curve_limit**2 + 12.0))


@pytest.mark.parametrize(
    ("curvature", "distances", "kwargs", "error"),
    [
        ((), (), {}, "non-empty"),
        ((0.0,), (0.0, 1.0), {}, "matching"),
        ((np.nan,), (0.0,), {}, "finite"),
        ((0.0, 0.0), (1.0, 0.0), {}, "non-decreasing"),
        ((0.0,), (0.0,), {"minimum_speed_mps": 6.0}, "cannot exceed"),
        ((0.0,), (0.0,), {"braking_deceleration_mps2": 0.0}, "positive"),
    ],
)
def test_invalid_profiles_are_rejected(curvature, distances, kwargs, error: str) -> None:
    parameters = {
        "minimum_speed_mps": 2.5,
        "maximum_speed_mps": 5.0,
        "maximum_lateral_acceleration_mps2": 1.5,
        "braking_deceleration_mps2": 3.0,
    }
    parameters.update(kwargs)
    with pytest.raises((TypeError, ValueError), match=error):
        curvature_speed_profile(curvature, distances, **parameters)


def test_speed_profile_source_has_no_challenge_or_backend_imports() -> None:
    source = inspect.getsource(speed_profile)
    for forbidden in (
        "controller_learning.envs",
        "race_core",
        "controller_learning.tracks",
        "TrackBatch",
        "controller_learning.physics",
        "import jax",
        "import mujoco",
        "import warp",
    ):
        assert forbidden not in source
