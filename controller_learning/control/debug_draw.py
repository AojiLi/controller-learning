"""Write-only debug drawing commands for trusted Controller callbacks."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from threading import Lock
from typing import TypeAlias

import numpy as np
from numpy.typing import ArrayLike

Point2D: TypeAlias = tuple[np.float32, np.float32]
Color: TypeAlias = tuple[np.float32, np.float32, np.float32, np.float32]

_DEFAULT_COLOR = (1.0, 1.0, 1.0, 1.0)


@dataclass(frozen=True, slots=True)
class LineCommand:
    """One immutable two-dimensional line command."""

    start: Point2D
    end: Point2D
    color: Color
    width: np.float32


@dataclass(frozen=True, slots=True)
class PointsCommand:
    """One immutable collection of two-dimensional points."""

    positions: tuple[Point2D, ...]
    color: Color
    size: np.float32


@dataclass(frozen=True, slots=True)
class TextCommand:
    """One immutable two-dimensional text annotation."""

    position: Point2D
    text: str
    color: Color


DebugDrawCommand: TypeAlias = LineCommand | PointsCommand | TextCommand


def _as_float32_array(value: ArrayLike, *, name: str, shape: tuple[int, ...]) -> np.ndarray:
    try:
        with np.errstate(over="ignore", invalid="ignore"):
            array = np.asarray(value, dtype=np.float32)
    except (OverflowError, TypeError, ValueError) as error:
        raise ValueError(f"{name} must be convertible to finite float32 values") from error
    if array.shape != shape:
        raise ValueError(f"{name} must have shape {shape}, got {array.shape}")
    if not np.isfinite(array).all():
        raise ValueError(f"{name} must contain only finite float32 values")
    return array


def _point(value: ArrayLike, *, name: str) -> Point2D:
    array = _as_float32_array(value, name=name, shape=(2,))
    return (np.float32(array[0]), np.float32(array[1]))


def _color(value: ArrayLike) -> Color:
    try:
        with np.errstate(over="ignore", invalid="ignore"):
            array = np.asarray(value, dtype=np.float32)
    except (OverflowError, TypeError, ValueError) as error:
        raise ValueError("color must be convertible to finite float32 values") from error
    if array.shape == (3,):
        array = np.concatenate((array, np.ones(1, dtype=np.float32)))
    if array.shape != (4,):
        raise ValueError(f"color must have shape (3,) or (4,), got {array.shape}")
    if not np.isfinite(array).all():
        raise ValueError("color must contain only finite float32 values")
    if np.any((array < 0.0) | (array > 1.0)):
        raise ValueError("color components must lie in [0, 1]")
    return tuple(np.float32(component) for component in array)  # type: ignore[return-value]


def _positive_float32(value: float, *, name: str) -> np.float32:
    array = _as_float32_array(value, name=name, shape=())
    scalar = np.float32(array)
    if scalar <= 0.0:
        raise ValueError(f"{name} must be positive")
    return scalar


class DebugDraw:
    """Write-only drawing surface passed to ``Controller.render_callback``.

    This object intentionally has no command, renderer, simulation, or state getters. The runner
    owns the separate internal command buffer and passes only this writer to a Controller.
    """

    __slots__ = ("__emit",)

    def __init__(self, emit: Callable[[DebugDrawCommand], None]) -> None:
        self.__emit = emit

    def line(
        self,
        start: ArrayLike,
        end: ArrayLike,
        *,
        color: ArrayLike = _DEFAULT_COLOR,
        width: float = 1.0,
    ) -> None:
        """Append a line in world ``(x, y)`` coordinates."""
        self.__emit(
            LineCommand(
                start=_point(start, name="start"),
                end=_point(end, name="end"),
                color=_color(color),
                width=_positive_float32(width, name="width"),
            )
        )

    def points(
        self,
        positions: ArrayLike,
        *,
        color: ArrayLike = _DEFAULT_COLOR,
        size: float = 3.0,
    ) -> None:
        """Append one or more points in world ``(x, y)`` coordinates."""
        try:
            with np.errstate(over="ignore", invalid="ignore"):
                array = np.asarray(positions, dtype=np.float32)
        except (OverflowError, TypeError, ValueError) as error:
            raise ValueError("positions must be convertible to finite float32 values") from error
        if array.ndim != 2 or array.shape[1:] != (2,) or array.shape[0] == 0:
            raise ValueError(f"positions must have non-empty shape (N, 2), got {array.shape}")
        if not np.isfinite(array).all():
            raise ValueError("positions must contain only finite float32 values")
        immutable_positions = tuple(
            (np.float32(position[0]), np.float32(position[1])) for position in array
        )
        self.__emit(
            PointsCommand(
                positions=immutable_positions,
                color=_color(color),
                size=_positive_float32(size, name="size"),
            )
        )

    def text(
        self,
        position: ArrayLike,
        text: str,
        *,
        color: ArrayLike = _DEFAULT_COLOR,
    ) -> None:
        """Append a non-empty text annotation in world ``(x, y)`` coordinates."""
        if not isinstance(text, str):
            raise TypeError("text must be a string")
        if not text:
            raise ValueError("text cannot be empty")
        self.__emit(
            TextCommand(position=_point(position, name="position"), text=text, color=_color(color))
        )


class _DebugDrawBuffer:
    """Runner-owned command storage; Controllers receive only :attr:`writer`."""

    __slots__ = ("_commands", "_lock", "writer")

    def __init__(self) -> None:
        self._commands: list[DebugDrawCommand] = []
        self._lock = Lock()
        self.writer = DebugDraw(self._append)

    def _append(self, command: DebugDrawCommand) -> None:
        with self._lock:
            self._commands.append(command)

    def snapshot(self) -> tuple[DebugDrawCommand, ...]:
        """Return an immutable renderer snapshot without clearing commands."""
        with self._lock:
            return tuple(self._commands)

    def drain(self) -> tuple[DebugDrawCommand, ...]:
        """Return an immutable renderer snapshot and clear the buffer."""
        with self._lock:
            commands = tuple(self._commands)
            self._commands.clear()
            return commands


__all__ = ["DebugDraw"]
