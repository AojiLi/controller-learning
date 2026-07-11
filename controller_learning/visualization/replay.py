"""Deterministic two-dimensional artifacts for public episode trajectories."""

from __future__ import annotations

import hashlib
import io
import os
import tempfile
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.figure import Figure

from controller_learning.evaluation.trajectory import EpisodeTrajectory

_TERMINATION_LABELS = {
    1: "success",
    2: "off track",
    3: "invalid action",
    4: "timeout",
}


@dataclass(frozen=True, slots=True)
class ReplayArtifact:
    """Identity and frame provenance for one deterministic replay artifact."""

    path: Path
    sha256: str
    size_bytes: int
    source_frame_count: int
    rendered_frame_indices: tuple[int, ...]

    def __post_init__(self) -> None:
        if not isinstance(self.path, Path):
            raise TypeError("path must be a Path")
        if (
            not isinstance(self.sha256, str)
            or len(self.sha256) != 64
            or any(character not in "0123456789abcdef" for character in self.sha256)
        ):
            raise ValueError("sha256 must be a lowercase 64-character digest")
        if isinstance(self.size_bytes, bool) or not isinstance(self.size_bytes, int):
            raise TypeError("size_bytes must be an integer")
        if self.size_bytes <= 0:
            raise ValueError("size_bytes must be positive")
        if isinstance(self.source_frame_count, bool) or not isinstance(
            self.source_frame_count, int
        ):
            raise TypeError("source_frame_count must be an integer")
        if self.source_frame_count < 2:
            raise ValueError("source_frame_count must be at least two")
        if not self.rendered_frame_indices:
            raise ValueError("rendered_frame_indices cannot be empty")
        if tuple(sorted(set(self.rendered_frame_indices))) != self.rendered_frame_indices:
            raise ValueError("rendered_frame_indices must be strictly increasing")
        if (
            self.rendered_frame_indices[0] < 0
            or self.rendered_frame_indices[-1] >= self.source_frame_count
        ):
            raise ValueError("rendered_frame_indices must address source trajectory frames")


def _atomic_write(path: Path, content: bytes) -> None:
    if path.is_symlink():
        raise ValueError("replay output_path cannot be a symbolic link")
    if path.exists() and not path.is_file():
        raise ValueError("replay output_path must be a regular file or absent")
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.parent.is_symlink() or not path.parent.is_dir():
        raise ValueError("replay output parent must be a real directory")
    temporary_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="wb",
            dir=path.parent,
            prefix=f".{path.name}.",
            suffix=".tmp",
            delete=False,
        ) as temporary:
            temporary_path = Path(temporary.name)
            temporary.write(content)
            temporary.flush()
            os.fchmod(temporary.fileno(), 0o644)
            os.fsync(temporary.fileno())
        os.replace(temporary_path, path)
        directory_descriptor = os.open(path.parent, os.O_RDONLY | os.O_DIRECTORY)
        try:
            os.fsync(directory_descriptor)
        finally:
            os.close(directory_descriptor)
        if path.read_bytes() != content:
            raise OSError("replay artifact readback did not match written bytes")
    finally:
        if temporary_path is not None:
            temporary_path.unlink(missing_ok=True)


def _artifact(
    path: Path,
    content: bytes,
    trajectory: EpisodeTrajectory,
    indices: tuple[int, ...],
) -> ReplayArtifact:
    _atomic_write(path, content)
    return ReplayArtifact(
        path=path,
        sha256=hashlib.sha256(content).hexdigest(),
        size_bytes=len(content),
        source_frame_count=trajectory.frame_count,
        rendered_frame_indices=indices,
    )


def _valid_track_geometry(
    trajectory: EpisodeTrajectory,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    mask = trajectory.track_mask
    return (
        trajectory.centerline_m[mask],
        trajectory.left_boundary_m[mask],
        trajectory.right_boundary_m[mask],
    )


def write_trajectory_overview_png(
    trajectory: EpisodeTrajectory,
    output_path: str | Path,
) -> ReplayArtifact:
    """Write a fixed-layout top-down trajectory overview as a deterministic PNG."""

    if not isinstance(trajectory, EpisodeTrajectory):
        raise TypeError("trajectory must be an EpisodeTrajectory")
    path = Path(output_path)
    if path.suffix.lower() != ".png":
        raise ValueError("overview output_path must use the .png suffix")

    content = render_trajectory_overview_png(trajectory)
    indices = tuple(range(trajectory.frame_count))
    return _artifact(path, content, trajectory, indices)


def render_trajectory_overview_png(trajectory: EpisodeTrajectory) -> bytes:
    """Return deterministic top-down trajectory overview PNG bytes."""

    if not isinstance(trajectory, EpisodeTrajectory):
        raise TypeError("trajectory must be an EpisodeTrajectory")

    centerline, left_boundary, right_boundary = _valid_track_geometry(trajectory)
    figure = Figure(figsize=(8.0, 6.0), dpi=100, facecolor="white")
    FigureCanvasAgg(figure)
    axes = figure.add_subplot(1, 1, 1)
    road = np.concatenate((left_boundary, right_boundary[::-1]), axis=0)
    axes.fill(road[:, 0], road[:, 1], color="0.92", zorder=0)
    axes.plot(centerline[:, 0], centerline[:, 1], "--", color="0.60", linewidth=1.0)
    axes.plot(left_boundary[:, 0], left_boundary[:, 1], color="tab:blue", linewidth=1.25)
    axes.plot(right_boundary[:, 0], right_boundary[:, 1], color="tab:blue", linewidth=1.25)
    axes.plot(
        trajectory.position_m[:, 0],
        trajectory.position_m[:, 1],
        color="tab:red",
        linewidth=1.75,
        label="Controller trajectory",
        zorder=3,
    )
    axes.scatter(
        trajectory.position_m[0, 0],
        trajectory.position_m[0, 1],
        marker="o",
        color="tab:green",
        s=36.0,
        label="Start",
        zorder=4,
    )
    axes.scatter(
        trajectory.position_m[-1, 0],
        trajectory.position_m[-1, 1],
        marker="x",
        color="black",
        s=44.0,
        label="End",
        zorder=4,
    )

    reason = int(trajectory.final_info["termination_reason"])
    status = _TERMINATION_LABELS[reason]
    track_id = int(trajectory.final_info["track_id"])
    benchmark = str(trajectory.final_info["benchmark_version"])
    axes.set_title(f"{benchmark} Track {track_id} | {status} | {trajectory.step_count} steps")
    axes.set_xlabel("x [m]")
    axes.set_ylabel("y [m]")
    axes.set_aspect("equal", adjustable="box")
    axes.grid(alpha=0.2)
    axes.legend(loc="best", framealpha=0.9)

    geometry = np.concatenate((left_boundary, right_boundary, trajectory.position_m), axis=0)
    minimum = np.min(geometry, axis=0)
    maximum = np.max(geometry, axis=0)
    span = np.maximum(maximum - minimum, 1.0)
    margin = 0.05 * span
    axes.set_xlim(float(minimum[0] - margin[0]), float(maximum[0] + margin[0]))
    axes.set_ylim(float(minimum[1] - margin[1]), float(maximum[1] + margin[1]))
    figure.subplots_adjust(left=0.11, right=0.97, bottom=0.10, top=0.91)

    output = io.BytesIO()
    figure.savefig(
        output,
        format="png",
        dpi=100,
        facecolor="white",
        metadata={"Software": "controller-learning"},
    )
    content = output.getvalue()
    figure.clear()
    if not content.startswith(b"\x89PNG\r\n\x1a\n"):
        raise RuntimeError("Matplotlib did not produce a PNG artifact")
    return content


__all__ = [
    "ReplayArtifact",
    "render_trajectory_overview_png",
    "write_trajectory_overview_png",
]
