"""Minimal two-dimensional renderer for the public M4 observation contract."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any, ClassVar, Literal, TypeAlias

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from numpy.typing import NDArray

from controller_learning.control.debug_draw import (
    DebugDrawCommand,
    LineCommand,
    PointsCommand,
    TextCommand,
)
from controller_learning.envs.observation import OBSERVATION_KEYS

RenderMode: TypeAlias = Literal["human", "rgb_array"]
RenderResult: TypeAlias = NDArray[np.uint8] | None


def _float32(value: Any, *, name: str, shape: tuple[int, ...]) -> NDArray[np.float32]:
    try:
        with np.errstate(over="ignore", invalid="ignore"):
            array = np.asarray(value, dtype=np.float32)
    except (OverflowError, TypeError, ValueError) as error:
        raise ValueError(f"observation field {name!r} must be numeric") from error
    if array.shape != shape:
        raise ValueError(f"observation field {name!r} must have shape {shape}, got {array.shape}")
    if not np.isfinite(array).all():
        raise ValueError(f"observation field {name!r} must contain only finite values")
    return array


def _validated_observation(observation: Mapping[str, Any]) -> dict[str, NDArray[Any]]:
    if not isinstance(observation, Mapping):
        raise TypeError("observation must be a mapping")
    expected = set(OBSERVATION_KEYS)
    actual = set(observation)
    if actual != expected:
        missing = sorted(expected - actual, key=repr)
        extra = sorted(actual - expected, key=repr)
        raise ValueError(
            f"observation keys do not match public schema; missing={missing}, extra={extra}"
        )

    try:
        centerline = np.asarray(observation["centerline"])
    except (TypeError, ValueError) as error:
        raise ValueError("observation field 'centerline' must be a numeric array") from error
    if centerline.ndim != 2 or centerline.shape[1:] != (2,):
        raise ValueError(
            f"observation field 'centerline' must have shape (N, 2), got {centerline.shape}"
        )
    point_count = centerline.shape[0]
    if point_count < 2:
        raise ValueError("observation track capacity must contain at least two points")

    validated: dict[str, NDArray[Any]] = {
        "position": _float32(observation["position"], name="position", shape=(2,)),
        "yaw": _float32(observation["yaw"], name="yaw", shape=()),
        "velocity_body": _float32(observation["velocity_body"], name="velocity_body", shape=(2,)),
        "yaw_rate": _float32(observation["yaw_rate"], name="yaw_rate", shape=()),
        "steering_angle": _float32(observation["steering_angle"], name="steering_angle", shape=()),
        "track_progress": _float32(observation["track_progress"], name="track_progress", shape=()),
        "centerline": _float32(
            observation["centerline"], name="centerline", shape=(point_count, 2)
        ),
        "left_boundary": _float32(
            observation["left_boundary"], name="left_boundary", shape=(point_count, 2)
        ),
        "right_boundary": _float32(
            observation["right_boundary"], name="right_boundary", shape=(point_count, 2)
        ),
        "track_length": _float32(observation["track_length"], name="track_length", shape=()),
    }

    mask = np.asarray(observation["track_mask"])
    if mask.shape != (point_count,):
        raise ValueError(
            f"observation field 'track_mask' must have shape ({point_count},), got {mask.shape}"
        )
    if not np.isin(mask, (0, 1)).all():
        raise ValueError("observation field 'track_mask' must contain only 0 or 1")
    mask = mask.astype(np.bool_, copy=False)
    if np.count_nonzero(mask) < 2:
        raise ValueError("observation field 'track_mask' must select at least two points")
    validated["track_mask"] = mask

    progress = float(validated["track_progress"])
    if not 0.0 <= progress <= 1.0:
        raise ValueError("observation field 'track_progress' must lie in [0, 1]")
    if float(validated["track_length"]) <= 0.0:
        raise ValueError("observation field 'track_length' must be positive")
    return validated


class Renderer2D:
    """Render one public single-world observation and optional debug commands."""

    metadata: ClassVar[dict[str, object]] = {
        "render_modes": ["human", "rgb_array"],
        "render_fps": 20,
    }

    def __init__(self, render_mode: RenderMode = "human") -> None:
        if render_mode not in ("human", "rgb_array"):
            raise ValueError("render_mode must be 'human' or 'rgb_array'")
        self.render_mode = render_mode
        self._figure: Figure | None = None
        self._axes: Axes | None = None

    def _canvas(self) -> tuple[Figure, Axes]:
        if self._figure is None or self._axes is None:
            self._figure, self._axes = plt.subplots(figsize=(8.0, 6.0), dpi=100)
        return self._figure, self._axes

    def render(
        self,
        observation: Mapping[str, Any],
        commands: tuple[DebugDrawCommand, ...] = (),
    ) -> RenderResult:
        """Draw the current car, valid track geometry, and write-only debug snapshot."""

        if not isinstance(commands, tuple):
            raise TypeError("commands must be an immutable tuple")
        values = _validated_observation(observation)
        figure, axes = self._canvas()
        axes.clear()

        mask = values["track_mask"]
        centerline = values["centerline"][mask]
        left_boundary = values["left_boundary"][mask]
        right_boundary = values["right_boundary"][mask]
        axes.plot(centerline[:, 0], centerline[:, 1], color="0.55", linestyle="--", linewidth=1.0)
        axes.plot(left_boundary[:, 0], left_boundary[:, 1], color="tab:blue", linewidth=1.5)
        axes.plot(right_boundary[:, 0], right_boundary[:, 1], color="tab:blue", linewidth=1.5)

        position = values["position"]
        yaw = float(values["yaw"])
        heading = np.asarray((np.cos(yaw), np.sin(yaw)), dtype=np.float32)
        axes.scatter(position[0], position[1], color="tab:red", s=36.0, zorder=4)
        axes.plot(
            (position[0], position[0] + 2.0 * heading[0]),
            (position[1], position[1] + 2.0 * heading[1]),
            color="tab:red",
            linewidth=2.0,
            zorder=4,
        )

        for command in commands:
            if isinstance(command, LineCommand):
                axes.plot(
                    (command.start[0], command.end[0]),
                    (command.start[1], command.end[1]),
                    color=command.color,
                    linewidth=float(command.width),
                )
            elif isinstance(command, PointsCommand):
                positions = np.asarray(command.positions, dtype=np.float32)
                axes.scatter(
                    positions[:, 0],
                    positions[:, 1],
                    color=command.color,
                    s=float(command.size) ** 2,
                )
            elif isinstance(command, TextCommand):
                axes.text(
                    command.position[0],
                    command.position[1],
                    command.text,
                    color=command.color,
                )
            else:
                raise TypeError(f"unsupported DebugDraw command: {type(command).__name__}")

        axes.set_aspect("equal", adjustable="datalim")
        axes.set_xlabel("x [m]")
        axes.set_ylabel("y [m]")
        axes.set_title(f"Progress: {float(values['track_progress']) * 100.0:.1f}%")
        axes.grid(alpha=0.2)
        axes.margins(0.05)
        figure.tight_layout()
        figure.canvas.draw()

        if self.render_mode == "human":
            plt.show(block=False)
            figure.canvas.flush_events()
            return None

        rgba = np.asarray(figure.canvas.buffer_rgba(), dtype=np.uint8)
        return np.array(rgba[..., :3], dtype=np.uint8, copy=True)

    def close(self) -> None:
        """Close the reusable figure; repeated calls are safe."""

        if self._figure is not None:
            plt.close(self._figure)
        self._figure = None
        self._axes = None


__all__ = ["Renderer2D"]
