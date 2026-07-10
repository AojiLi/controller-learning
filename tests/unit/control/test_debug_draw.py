"""Tests for the write-only Controller debug drawing boundary."""

from __future__ import annotations

from dataclasses import FrozenInstanceError

import numpy as np
import pytest

from controller_learning.control.debug_draw import (
    LineCommand,
    PointsCommand,
    TextCommand,
    _DebugDrawBuffer,
)


def test_controller_writer_exposes_no_read_or_simulator_surface() -> None:
    writer = _DebugDrawBuffer().writer
    public_names = {name for name in dir(writer) if not name.startswith("_")}

    assert public_names == {"line", "points", "text"}
    assert not hasattr(writer, "snapshot")
    assert not hasattr(writer, "drain")
    assert not hasattr(writer, "commands")
    assert not hasattr(writer, "simulator")
    assert not hasattr(writer, "environment")


def test_commands_are_normalized_to_immutable_float32_values() -> None:
    buffer = _DebugDrawBuffer()
    writer = buffer.writer

    writer.line((0, 1), np.array([2.0, 3.0], dtype=np.float64), color=(0.1, 0.2, 0.3))
    writer.points([[4, 5], [6, 7]], color=(0.4, 0.5, 0.6, 0.7), size=2)
    writer.text((8, 9), "look ahead", color=(1, 0, 0))

    snapshot = buffer.snapshot()
    assert isinstance(snapshot, tuple)
    assert isinstance(snapshot[0], LineCommand)
    assert isinstance(snapshot[1], PointsCommand)
    assert isinstance(snapshot[2], TextCommand)
    for command in snapshot:
        if isinstance(command, LineCommand):
            geometry = command.start
        elif isinstance(command, PointsCommand):
            geometry = command.positions[0]
        else:
            geometry = command.position
        assert all(isinstance(value, np.float32) for value in geometry)
        assert all(isinstance(value, np.float32) for value in command.color)
    assert snapshot[0].color[3] == np.float32(1.0)
    assert all(isinstance(value, np.float32) for point in snapshot[1].positions for value in point)
    with pytest.raises(FrozenInstanceError):
        snapshot[0].width = np.float32(4.0)  # type: ignore[misc]


def test_snapshot_does_not_clear_and_drain_does() -> None:
    buffer = _DebugDrawBuffer()
    buffer.writer.text((0, 0), "first")

    assert len(buffer.snapshot()) == 1
    assert len(buffer.snapshot()) == 1
    assert len(buffer.drain()) == 1
    assert buffer.snapshot() == ()


@pytest.mark.parametrize(
    ("method", "args", "match"),
    [
        ("line", ((0,), (1, 2)), "shape"),
        ("line", ((0, np.nan), (1, 2)), "finite"),
        ("line", ((0, 1), (np.inf, 2)), "finite"),
        ("points", ([],), "non-empty shape"),
        ("points", ([(0, 1, 2)],), "shape"),
        ("points", ([(0, np.nan)],), "finite"),
        ("text", ((np.inf, 0), "label"), "finite"),
    ],
)
def test_invalid_geometry_is_rejected(method: str, args: tuple[object, ...], match: str) -> None:
    writer = _DebugDrawBuffer().writer

    with pytest.raises(ValueError, match=match):
        getattr(writer, method)(*args)


@pytest.mark.parametrize("color", [(0, 0), (0, 0, 0, 0, 0), (0, np.nan, 0), (-0.1, 0, 0)])
def test_invalid_colors_are_rejected(color: tuple[float, ...]) -> None:
    writer = _DebugDrawBuffer().writer

    with pytest.raises(ValueError, match="color"):
        writer.line((0, 0), (1, 1), color=color)


@pytest.mark.parametrize(("method", "value"), [("line", 0.0), ("points", -1.0)])
def test_non_positive_width_and_size_are_rejected(method: str, value: float) -> None:
    writer = _DebugDrawBuffer().writer

    with pytest.raises(ValueError, match="positive"):
        if method == "line":
            writer.line((0, 0), (1, 1), width=value)
        else:
            writer.points([(0, 0)], size=value)


def test_text_must_be_a_non_empty_string() -> None:
    writer = _DebugDrawBuffer().writer

    with pytest.raises(ValueError, match="empty"):
        writer.text((0, 0), "")
    with pytest.raises(TypeError, match="string"):
        writer.text((0, 0), 12)  # type: ignore[arg-type]
