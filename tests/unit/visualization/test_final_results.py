"""Tests for deterministic M8 final-result PNG renderers."""

from __future__ import annotations

import inspect
import io
import struct
from collections.abc import Iterator

import matplotlib.image as mpimg
import numpy as np
import pytest

from controller_learning.visualization import (
    final_results,
    render_controller_telemetry_png,
    render_final_comparison_png,
)


def _telemetry() -> dict[str, object]:
    return {
        "controller_name": "pid",
        "control_dt_s": 0.05,
        "speed_mps": np.asarray((0.4, 1.2, 2.1, 2.7), dtype=np.float32),
        "lateral_error_m": np.asarray((0.0, -0.1, 0.25, 0.4), dtype=np.float32),
        "requested_action": np.asarray(
            ((0.0, 2.0), (0.2, 4.1), (0.7, 1.0), (-0.1, -8.2)),
            dtype=np.float32,
        ),
        "steering_saturated": np.asarray((False, False, True, False)),
        "longitudinal_saturated": np.asarray((False, True, False, True)),
    }


def _comparison() -> dict[str, object]:
    centerline = np.zeros((8, 2), dtype=np.float32)
    centerline[:5] = ((0, 0), (5, 0), (5, 5), (0, 5), (0, 0))
    left = np.zeros_like(centerline)
    left[:5] = ((0, 1), (4, 1), (4, 4), (1, 4), (0, 1))
    right = np.zeros_like(centerline)
    right[:5] = ((0, -1), (6, -1), (6, 6), (-1, 6), (0, -1))
    return {
        "benchmark_version": "0.1",
        "track_id": 2_000_001,
        "centerline_m": centerline,
        "left_boundary_m": left,
        "right_boundary_m": right,
        "track_mask": np.asarray((True, True, True, True, True, False, False, False)),
        "trajectories_m": {
            "pid": np.asarray(((0, 0), (4.8, 0.2), (5.0, 4.7)), dtype=np.float32),
            "mpc": np.asarray(((0, 0), (5.0, 0.0), (5.0, 5.0)), dtype=np.float32),
            "ppo": np.asarray(((0, 0), (4.7, -0.1), (5.2, 4.8)), dtype=np.float32),
        },
    }


def _png_text_chunks(content: bytes) -> Iterator[bytes]:
    offset = 8
    while offset < len(content):
        length = struct.unpack(">I", content[offset : offset + 4])[0]
        kind = content[offset + 4 : offset + 8]
        payload = content[offset + 8 : offset + 8 + length]
        if kind in (b"tEXt", b"zTXt", b"iTXt"):
            yield payload
        offset += 12 + length


def test_telemetry_png_is_byte_deterministic_and_has_fixed_metadata() -> None:
    first = render_controller_telemetry_png(**_telemetry())
    second = render_controller_telemetry_png(**_telemetry())

    assert first == second
    assert first.startswith(b"\x89PNG\r\n\x1a\n")
    assert len(first) > 10_000
    image = mpimg.imread(io.BytesIO(first))
    assert image.shape == (800, 1000, 4)
    metadata = b"\n".join(_png_text_chunks(first))
    assert b"controller-learning" in metadata
    assert b"Creation Time" not in metadata
    assert b"Date" not in metadata


def test_telemetry_saturation_markers_affect_the_rendered_artifact() -> None:
    marked = _telemetry()
    unmarked = _telemetry()
    unmarked["steering_saturated"] = np.zeros(4, dtype=np.bool_)
    unmarked["longitudinal_saturated"] = np.zeros(4, dtype=np.bool_)

    assert render_controller_telemetry_png(**marked) != render_controller_telemetry_png(**unmarked)


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("controller_name", "sac", "controller_name"),
        ("control_dt_s", 0.0, "positive"),
        ("speed_mps", np.asarray((1.0, -0.1)), "negative"),
        ("lateral_error_m", np.asarray((0.0, np.nan, 0.1, 0.2)), "finite"),
        ("requested_action", np.zeros((4, 3)), "shape"),
        ("steering_saturated", np.zeros(4, dtype=np.int8), "boolean dtype"),
        ("longitudinal_saturated", np.zeros(3, dtype=np.bool_), "shape"),
    ],
)
def test_telemetry_rejects_invalid_metric_samples(
    field: str,
    value: object,
    message: str,
) -> None:
    arguments = _telemetry()
    arguments[field] = value

    with pytest.raises((TypeError, ValueError), match=message):
        render_controller_telemetry_png(**arguments)


def test_final_comparison_png_is_deterministic_and_contains_three_distinct_paths() -> None:
    first = render_final_comparison_png(**_comparison())
    second = render_final_comparison_png(**_comparison())

    assert first == second
    assert first.startswith(b"\x89PNG\r\n\x1a\n")
    assert len(first) > 10_000
    image = mpimg.imread(io.BytesIO(first))
    assert image.shape == (800, 1000, 4)

    changed = _comparison()
    trajectories = dict(changed["trajectories_m"])  # type: ignore[arg-type]
    trajectories["ppo"] = np.asarray(((0, 0), (2.5, 2.5), (5.0, 5.0)), dtype=np.float32)
    changed["trajectories_m"] = trajectories
    assert render_final_comparison_png(**changed) != first


@pytest.mark.parametrize(
    ("mutate", "message"),
    [
        (lambda values: values.update(track_id=-1), "uint32"),
        (lambda values: values.update(benchmark_version=" 0.1"), "token"),
        (lambda values: values.update(centerline_m=np.zeros((2, 2))), "three points"),
        (lambda values: values.update(left_boundary_m=np.zeros((7, 2))), "shape"),
        (
            lambda values: values.update(
                track_mask=np.asarray((True, True, False, False, False, False, False, False))
            ),
            "three points",
        ),
        (
            lambda values: values.update(
                trajectories_m={
                    "pid": np.zeros((2, 2)),
                    "mpc": np.zeros((2, 2)),
                }
            ),
            "exactly pid, mpc, and ppo",
        ),
        (
            lambda values: values.update(
                trajectories_m={
                    "pid": np.zeros((2, 2)),
                    "mpc": np.zeros((2, 2)),
                    "ppo": np.asarray(((0.0, 0.0), (np.inf, 1.0))),
                }
            ),
            "finite",
        ),
    ],
)
def test_final_comparison_rejects_invalid_geometry_and_paths(
    mutate: object,
    message: str,
) -> None:
    arguments = _comparison()
    mutate(arguments)  # type: ignore[operator]

    with pytest.raises((TypeError, ValueError), match=message):
        render_final_comparison_png(**arguments)


def test_final_result_renderers_have_no_simulator_or_official_asset_access() -> None:
    source = inspect.getsource(final_results)

    assert "controller_learning.physics" not in source
    assert "controller_learning.tracks" not in source
    assert "controller_learning.evaluation" not in source
    assert "test.npz" not in source
    assert "test.json" not in source
