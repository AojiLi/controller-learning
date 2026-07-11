"""Deterministic M8 plots built only from canonical public result arrays."""

from __future__ import annotations

import io
import math
from collections.abc import Mapping
from numbers import Real
from typing import Final

import matplotlib
import numpy as np
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.figure import Figure
from numpy.typing import NDArray

FINAL_CONTROLLER_ORDER: Final = ("pid", "mpc", "ppo")

_CONTROLLER_LABELS: Final = {
    "pid": "PID",
    "mpc": "MPC",
    "ppo": "PPO",
}
_CONTROLLER_COLORS: Final = {
    "pid": "#0072B2",
    "mpc": "#D55E00",
    "ppo": "#009E73",
}
_PNG_SIGNATURE: Final = b"\x89PNG\r\n\x1a\n"
_UINT32_MAX: Final = int(np.iinfo(np.uint32).max)
_RC_PARAMS: Final = {
    "font.family": "DejaVu Sans",
    "font.size": 9.0,
    "axes.labelsize": 9.0,
    "axes.titlesize": 11.0,
    "legend.fontsize": 8.0,
    "xtick.labelsize": 8.0,
    "ytick.labelsize": 8.0,
    "axes.linewidth": 0.8,
    "lines.solid_capstyle": "round",
    "savefig.transparent": False,
}


def _controller_name(value: object) -> str:
    if not isinstance(value, str):
        raise TypeError("controller_name must be a string")
    if value not in FINAL_CONTROLLER_ORDER:
        raise ValueError("controller_name must be one of: pid, mpc, ppo")
    return value


def _positive_float(value: object, *, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, Real):
        raise TypeError(f"{name} must be a real number")
    result = float(value)
    if not math.isfinite(result) or result <= 0.0:
        raise ValueError(f"{name} must be finite and positive")
    return result


def _numeric_array(
    value: object,
    *,
    name: str,
    shape: tuple[int | None, ...],
) -> NDArray[np.float64]:
    try:
        source = np.asarray(value)
    except (TypeError, ValueError) as error:
        raise ValueError(f"{name} must be a numeric array") from error
    if source.dtype.kind not in "iuf":
        raise TypeError(f"{name} must use a real numeric dtype")
    if source.ndim != len(shape) or any(
        expected is not None and actual != expected
        for expected, actual in zip(shape, source.shape, strict=True)
    ):
        rendered_shape = tuple("N" if item is None else item for item in shape)
        raise ValueError(f"{name} must have shape {rendered_shape}, got {source.shape}")
    result = np.asarray(source, dtype=np.float64)
    if not np.isfinite(result).all():
        raise ValueError(f"{name} must contain only finite values")
    return result


def _boolean_vector(value: object, *, name: str, length: int) -> NDArray[np.bool_]:
    source = np.asarray(value)
    if source.shape != (length,):
        raise ValueError(f"{name} must have shape ({length},), got {source.shape}")
    if source.dtype != np.dtype(np.bool_):
        raise TypeError(f"{name} must use boolean dtype")
    return np.asarray(source, dtype=np.bool_)


def _png_bytes(figure: Figure) -> bytes:
    output = io.BytesIO()
    try:
        figure.savefig(
            output,
            format="png",
            dpi=100,
            facecolor="white",
            metadata={"Software": "controller-learning"},
        )
        content = output.getvalue()
    finally:
        figure.clear()
    if not content.startswith(_PNG_SIGNATURE):
        raise RuntimeError("Matplotlib did not produce a PNG artifact")
    return content


def render_controller_telemetry_png(
    *,
    controller_name: str,
    control_dt_s: float,
    speed_mps: object,
    lateral_error_m: object,
    requested_action: object,
    steering_saturated: object,
    longitudinal_saturated: object,
) -> bytes:
    """Render one Controller's canonical row-0 metric samples as PNG bytes.

    Every metric sample must describe the same post-step transition. Requested
    action columns are steering angle in radians and longitudinal acceleration
    in metres per second squared, respectively.
    """

    name = _controller_name(controller_name)
    dt = _positive_float(control_dt_s, name="control_dt_s")
    speed = _numeric_array(speed_mps, name="speed_mps", shape=(None,))
    sample_count = int(speed.shape[0])
    if sample_count < 1:
        raise ValueError("speed_mps must contain at least one sample")
    if np.any(speed < 0.0):
        raise ValueError("speed_mps cannot contain negative values")
    lateral_error = _numeric_array(
        lateral_error_m,
        name="lateral_error_m",
        shape=(sample_count,),
    )
    action = _numeric_array(
        requested_action,
        name="requested_action",
        shape=(sample_count, 2),
    )
    steering_flags = _boolean_vector(
        steering_saturated,
        name="steering_saturated",
        length=sample_count,
    )
    longitudinal_flags = _boolean_vector(
        longitudinal_saturated,
        name="longitudinal_saturated",
        length=sample_count,
    )
    time_s = dt * np.arange(1, sample_count + 1, dtype=np.float64)

    with matplotlib.rc_context(_RC_PARAMS):
        figure = Figure(figsize=(10.0, 8.0), dpi=100, facecolor="white")
        FigureCanvasAgg(figure)
        axes = figure.subplots(4, 1, sharex=True)

        axes[0].plot(time_s, speed, color="#0072B2", linewidth=1.5)
        axes[0].set_ylabel("Speed [m/s]")

        axes[1].plot(time_s, lateral_error, color="#D55E00", linewidth=1.5)
        axes[1].axhline(0.0, color="0.55", linewidth=0.75, linestyle="--")
        axes[1].set_ylabel("Lateral error [m]")

        axes[2].plot(time_s, action[:, 0], color="#009E73", linewidth=1.5)
        axes[2].axhline(0.0, color="0.55", linewidth=0.75, linestyle="--")
        if bool(np.any(steering_flags)):
            axes[2].scatter(
                time_s[steering_flags],
                action[steering_flags, 0],
                marker="o",
                s=26.0,
                facecolors="none",
                edgecolors="#CC0000",
                linewidths=1.2,
                label="Saturated request",
                zorder=3,
            )
            axes[2].legend(loc="upper right", framealpha=0.9)
        axes[2].set_ylabel("Steering [rad]")

        axes[3].plot(time_s, action[:, 1], color="#CC79A7", linewidth=1.5)
        axes[3].axhline(0.0, color="0.55", linewidth=0.75, linestyle="--")
        if bool(np.any(longitudinal_flags)):
            axes[3].scatter(
                time_s[longitudinal_flags],
                action[longitudinal_flags, 1],
                marker="x",
                s=28.0,
                color="#CC0000",
                linewidths=1.2,
                label="Saturated request",
                zorder=3,
            )
            axes[3].legend(loc="upper right", framealpha=0.9)
        axes[3].set_ylabel("Longitudinal [m/s²]")
        axes[3].set_xlabel("Episode time [s]")

        for axis in axes:
            axis.grid(alpha=0.2)
            axis.set_xlim(0.0, float(time_s[-1]))
        figure.suptitle(f"{_CONTROLLER_LABELS[name]} | canonical row 0 telemetry")
        figure.align_ylabels(axes)
        figure.subplots_adjust(left=0.12, right=0.98, bottom=0.08, top=0.94, hspace=0.14)
        return _png_bytes(figure)


def _track_id(value: object) -> int:
    if isinstance(value, bool) or not isinstance(value, (int, np.integer)):
        raise TypeError("track_id must be an integer")
    result = int(value)
    if not 0 <= result <= _UINT32_MAX:
        raise ValueError("track_id must fit in uint32")
    return result


def _benchmark_version(value: object) -> str:
    if not isinstance(value, str):
        raise TypeError("benchmark_version must be a string")
    if not value or value.strip() != value or any(character.isspace() for character in value):
        raise ValueError("benchmark_version must be a non-empty token")
    return value


def render_final_comparison_png(
    *,
    benchmark_version: str,
    track_id: int,
    centerline_m: object,
    left_boundary_m: object,
    right_boundary_m: object,
    track_mask: object,
    trajectories_m: Mapping[str, object],
) -> bytes:
    """Render PID, MPC, and PPO canonical row-0 paths on one Track geometry."""

    benchmark = _benchmark_version(benchmark_version)
    numeric_track_id = _track_id(track_id)
    centerline = _numeric_array(centerline_m, name="centerline_m", shape=(None, 2))
    point_count = int(centerline.shape[0])
    if point_count < 3:
        raise ValueError("centerline_m must contain at least three points")
    left_boundary = _numeric_array(
        left_boundary_m,
        name="left_boundary_m",
        shape=(point_count, 2),
    )
    right_boundary = _numeric_array(
        right_boundary_m,
        name="right_boundary_m",
        shape=(point_count, 2),
    )
    mask = _boolean_vector(track_mask, name="track_mask", length=point_count)
    if int(np.count_nonzero(mask)) < 3:
        raise ValueError("track_mask must select at least three points")
    if not isinstance(trajectories_m, Mapping):
        raise TypeError("trajectories_m must be a mapping")
    expected_names = set(FINAL_CONTROLLER_ORDER)
    actual_names = set(trajectories_m)
    if actual_names != expected_names:
        missing = sorted(expected_names - actual_names)
        extra = sorted(actual_names - expected_names, key=repr)
        raise ValueError(
            "trajectories_m must contain exactly pid, mpc, and ppo; "
            f"missing={missing}, extra={extra}"
        )
    trajectories: dict[str, NDArray[np.float64]] = {}
    for name in FINAL_CONTROLLER_ORDER:
        path = _numeric_array(
            trajectories_m[name],
            name=f"trajectories_m[{name!r}]",
            shape=(None, 2),
        )
        if path.shape[0] < 2:
            raise ValueError(f"trajectories_m[{name!r}] must contain at least two positions")
        trajectories[name] = path

    valid_centerline = centerline[mask]
    valid_left = left_boundary[mask]
    valid_right = right_boundary[mask]
    with matplotlib.rc_context(_RC_PARAMS):
        figure = Figure(figsize=(10.0, 8.0), dpi=100, facecolor="white")
        FigureCanvasAgg(figure)
        axes = figure.add_subplot(1, 1, 1)
        road = np.concatenate((valid_left, valid_right[::-1]), axis=0)
        axes.fill(road[:, 0], road[:, 1], color="0.93", zorder=0)
        axes.plot(
            valid_centerline[:, 0],
            valid_centerline[:, 1],
            color="0.55",
            linewidth=1.0,
            linestyle="--",
            label="Centerline",
            zorder=1,
        )
        axes.plot(
            valid_left[:, 0],
            valid_left[:, 1],
            color="0.25",
            linewidth=1.15,
            label="Track boundaries",
            zorder=2,
        )
        axes.plot(
            valid_right[:, 0],
            valid_right[:, 1],
            color="0.25",
            linewidth=1.15,
            zorder=2,
        )
        axes.plot(
            (valid_left[0, 0], valid_right[0, 0]),
            (valid_left[0, 1], valid_right[0, 1]),
            color="black",
            linewidth=1.5,
            linestyle=":",
            label="Start / finish",
            zorder=3,
        )
        for name in FINAL_CONTROLLER_ORDER:
            path = trajectories[name]
            axes.plot(
                path[:, 0],
                path[:, 1],
                color=_CONTROLLER_COLORS[name],
                linewidth=1.8,
                label=f"{_CONTROLLER_LABELS[name]} path",
                zorder=4,
            )

        geometry = np.concatenate(
            (valid_left, valid_right, *(trajectories[name] for name in FINAL_CONTROLLER_ORDER)),
            axis=0,
        )
        minimum = np.min(geometry, axis=0)
        maximum = np.max(geometry, axis=0)
        span = np.maximum(maximum - minimum, 1.0)
        margin = 0.05 * span
        axes.set_xlim(float(minimum[0] - margin[0]), float(maximum[0] + margin[0]))
        axes.set_ylim(float(minimum[1] - margin[1]), float(maximum[1] + margin[1]))
        axes.set_aspect("equal", adjustable="box")
        axes.set_xlabel("x [m]")
        axes.set_ylabel("y [m]")
        axes.set_title(
            f"Benchmark {benchmark} Track {numeric_track_id} | canonical row 0 comparison"
        )
        axes.grid(alpha=0.2)
        axes.legend(loc="upper right", framealpha=0.9, ncol=2)
        figure.subplots_adjust(left=0.10, right=0.98, bottom=0.09, top=0.94)
        return _png_bytes(figure)


__all__ = [
    "FINAL_CONTROLLER_ORDER",
    "render_controller_telemetry_png",
    "render_final_comparison_png",
]
