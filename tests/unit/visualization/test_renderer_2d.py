"""Tests for the public-observation-only M4 renderer."""

from __future__ import annotations

import inspect
from pathlib import Path

import matplotlib
import numpy as np
import pytest

from controller_learning.control.debug_draw import _DebugDrawBuffer
from controller_learning.visualization import Renderer2D, renderer_2d

matplotlib.use("Agg", force=True)


def _observation() -> dict[str, np.ndarray]:
    centerline = np.zeros((8, 2), dtype=np.float32)
    centerline[:5] = ((0, 0), (5, 0), (5, 5), (0, 5), (0, 0))
    left = np.zeros_like(centerline)
    left[:5] = ((0, 1), (4, 1), (4, 4), (1, 4), (0, 1))
    right = np.zeros_like(centerline)
    right[:5] = ((0, -1), (6, -1), (6, 6), (-1, 6), (0, -1))
    return {
        "position": np.asarray((1.0, 0.0), dtype=np.float32),
        "yaw": np.asarray(0.25, dtype=np.float32),
        "velocity_body": np.asarray((2.0, 0.0), dtype=np.float32),
        "yaw_rate": np.asarray(0.1, dtype=np.float32),
        "steering_angle": np.asarray(0.05, dtype=np.float32),
        "track_progress": np.asarray(0.25, dtype=np.float32),
        "centerline": centerline,
        "left_boundary": left,
        "right_boundary": right,
        "track_mask": np.asarray((1, 1, 1, 1, 1, 0, 0, 0), dtype=np.int8),
        "track_length": np.asarray(20.0, dtype=np.float32),
    }


def test_rgb_array_draws_only_public_observation_and_returns_pixels() -> None:
    renderer = Renderer2D("rgb_array")

    image = renderer.render(_observation())

    assert image is not None
    assert image.dtype == np.uint8
    assert image.ndim == 3
    assert image.shape[2] == 3
    assert image.size > 0
    renderer.close()


def test_all_debug_command_types_are_rendered() -> None:
    buffer = _DebugDrawBuffer()
    buffer.writer.line((1, 1), (3, 3), color=(1, 0, 0), width=2)
    buffer.writer.points(((2, 1), (3, 1)), color=(0, 1, 0), size=4)
    buffer.writer.text((2, 2), "target", color=(0, 0, 1))
    renderer = Renderer2D("rgb_array")

    image = renderer.render(_observation(), buffer.snapshot())

    assert image is not None
    assert len(renderer._axes.lines) == 5  # type: ignore[union-attr]
    assert len(renderer._axes.collections) == 2  # type: ignore[union-attr]
    assert [text.get_text() for text in renderer._axes.texts] == ["target"]  # type: ignore[union-attr]
    renderer.close()


def test_human_mode_reuses_figure_and_close_is_idempotent() -> None:
    renderer = Renderer2D("human")

    assert renderer.render(_observation()) is None
    first_figure = renderer._figure
    assert renderer.render(_observation()) is None
    assert renderer._figure is first_figure

    renderer.close()
    renderer.close()
    assert renderer._figure is None


@pytest.mark.parametrize("mode", ["", "ansi", None])
def test_invalid_render_mode_is_rejected(mode: object) -> None:
    with pytest.raises(ValueError, match="render_mode"):
        Renderer2D(mode)  # type: ignore[arg-type]


def test_invalid_observation_schema_and_geometry_are_rejected() -> None:
    renderer = Renderer2D("rgb_array")
    missing = _observation()
    del missing["yaw"]
    with pytest.raises(ValueError, match="public schema"):
        renderer.render(missing)

    wrong_shape = _observation()
    wrong_shape["position"] = np.zeros(3, dtype=np.float32)
    with pytest.raises(ValueError, match=r"position.*shape"):
        renderer.render(wrong_shape)

    invalid = _observation()
    invalid["centerline"][2, 0] = np.nan
    with pytest.raises(ValueError, match=r"centerline.*finite"):
        renderer.render(invalid)
    renderer.close()


def test_zero_padding_is_not_drawn() -> None:
    renderer = Renderer2D("rgb_array")
    observation = _observation()
    observation["centerline"][5:] = 10_000.0

    renderer.render(observation)

    rendered_centerline = renderer._axes.lines[0].get_xydata()  # type: ignore[union-attr]
    assert rendered_centerline.shape == (5, 2)
    assert np.max(np.abs(rendered_centerline)) < 10.0
    renderer.close()


def test_renderer_source_has_no_physics_backend_imports() -> None:
    source = inspect.getsource(renderer_2d)
    imported_modules = {
        node.split()[1].split(".")[0] for node in source.splitlines() if node.startswith("import ")
    }

    assert imported_modules.isdisjoint({"mujoco", "warp"})
    assert "controller_learning.physics" not in source
    assert Path(renderer_2d.__file__).name == "renderer_2d.py"
