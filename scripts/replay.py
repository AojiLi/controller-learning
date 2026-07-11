"""Inspect a canonical Controller trajectory without rerunning the simulation."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import tempfile
import time
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol

from controller_learning.evaluation.trajectory import (
    EpisodeTrajectory,
    canonical_trajectory_json_bytes,
    load_trajectory_json,
)
from controller_learning.visualization.renderer_2d import Renderer2D
from controller_learning.visualization.replay import render_trajectory_overview_png

REPLAY_SUMMARY_SCHEMA_VERSION = "controller-learning.replay-summary.v1"


class ReplayCliError(RuntimeError):
    """A replay request could not be completed safely."""


class PublicRenderer(Protocol):
    """The small Renderer2D surface used by interactive playback."""

    metadata: Mapping[str, object]

    def render(self, observation: Mapping[str, Any]) -> object:
        """Render one public observation."""

    def close(self) -> None:
        """Release renderer resources."""


RendererFactory = Callable[[str], PublicRenderer]
Clock = Callable[[], float]
Sleeper = Callable[[float], None]


@dataclass(frozen=True, slots=True)
class ReplayOptions:
    """Validated inputs for one offline trajectory replay."""

    trajectory_path: Path
    overview_path: Path | None
    play: bool
    speed: float = 1.0

    def __post_init__(self) -> None:
        object.__setattr__(self, "trajectory_path", Path(self.trajectory_path))
        if self.overview_path is not None:
            object.__setattr__(self, "overview_path", Path(self.overview_path))
        if type(self.play) is not bool:
            raise TypeError("play must be a boolean")
        if self.overview_path is None and not self.play:
            raise ValueError("request --overview, --play, or both")
        if isinstance(self.speed, bool) or not isinstance(self.speed, (int, float)):
            raise TypeError("speed must be a number")
        speed = float(self.speed)
        if not math.isfinite(speed) or speed <= 0.0:
            raise ValueError("speed must be finite and positive")
        object.__setattr__(self, "speed", speed)
        if self.overview_path is not None and self.overview_path.suffix.lower() != ".png":
            raise ValueError("overview path must use the .png suffix")


@dataclass(frozen=True, slots=True)
class PlaybackResult:
    """Deterministic playback settings and completed frame count."""

    rendered_frame_count: int
    source_fps: float
    speed: float

    def to_dict(self) -> dict[str, int | float | bool]:
        """Return a JSON-compatible record without wall-clock claims."""

        return {
            "requested": True,
            "rendered_frame_count": self.rendered_frame_count,
            "source_fps": self.source_fps,
            "speed": self.speed,
        }


def _positive_float(value: str) -> float:
    try:
        parsed = float(value)
    except ValueError as error:
        raise argparse.ArgumentTypeError("must be a number") from error
    if not math.isfinite(parsed) or parsed <= 0.0:
        raise argparse.ArgumentTypeError("must be finite and positive")
    return parsed


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "trajectory",
        type=Path,
        help="Canonical controller-learning trajectory JSON to inspect",
    )
    parser.add_argument(
        "--overview",
        type=Path,
        default=None,
        help="Write a deterministic top-down PNG; the destination must not exist",
    )
    parser.add_argument(
        "--play",
        action="store_true",
        help="Open interactive public-observation playback (requires a graphical display)",
    )
    parser.add_argument(
        "--speed",
        type=_positive_float,
        default=1.0,
        help="Interactive playback speed multiplier (default: 1.0)",
    )
    return parser


def _parse_options(argv: Sequence[str] | None = None) -> ReplayOptions:
    parser = _build_parser()
    values = parser.parse_args(argv)
    try:
        return ReplayOptions(
            trajectory_path=values.trajectory,
            overview_path=values.overview,
            play=values.play,
            speed=values.speed,
        )
    except (TypeError, ValueError) as error:
        parser.error(str(error))


def _write_new_file(path: Path, content: bytes) -> dict[str, str | int]:
    """Atomically publish bytes while refusing every pre-existing destination."""

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.parent.is_symlink() or not path.parent.is_dir():
        raise ReplayCliError("overview output parent must be a real directory")

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
        try:
            os.link(temporary_path, path, follow_symlinks=False)
        except FileExistsError as error:
            raise ReplayCliError(f"overview output already exists: {path}") from error
        directory_descriptor = os.open(path.parent, os.O_RDONLY | os.O_DIRECTORY)
        try:
            os.fsync(directory_descriptor)
        finally:
            os.close(directory_descriptor)
    finally:
        if temporary_path is not None:
            temporary_path.unlink(missing_ok=True)

    if path.read_bytes() != content:
        path.unlink(missing_ok=True)
        raise OSError("overview output readback did not match rendered bytes")
    return {
        "path": str(path),
        "sha256": hashlib.sha256(content).hexdigest(),
        "size_bytes": len(content),
    }


def play_trajectory(
    trajectory: EpisodeTrajectory,
    *,
    speed: float = 1.0,
    renderer_factory: RendererFactory = Renderer2D,
    clock: Clock = time.monotonic,
    sleeper: Sleeper = time.sleep,
) -> PlaybackResult:
    """Play every recorded public observation in order at the source frame rate."""

    if not isinstance(trajectory, EpisodeTrajectory):
        raise TypeError("trajectory must be an EpisodeTrajectory")
    if isinstance(speed, bool) or not isinstance(speed, (int, float)):
        raise TypeError("speed must be a number")
    speed = float(speed)
    if not math.isfinite(speed) or speed <= 0.0:
        raise ValueError("speed must be finite and positive")

    renderer = renderer_factory("human")
    try:
        source_fps_value = renderer.metadata.get("render_fps")
        if isinstance(source_fps_value, bool) or not isinstance(source_fps_value, (int, float)):
            raise ReplayCliError("interactive renderer does not declare a numeric render_fps")
        source_fps = float(source_fps_value)
        if not math.isfinite(source_fps) or source_fps <= 0.0:
            raise ReplayCliError("interactive renderer render_fps must be finite and positive")

        frame_period = 1.0 / (source_fps * speed)
        deadline = clock()
        for frame_index in range(trajectory.frame_count):
            renderer.render(trajectory.observation(frame_index))
            if frame_index + 1 < trajectory.frame_count:
                deadline += frame_period
                remaining = deadline - clock()
                if remaining > 0.0:
                    sleeper(remaining)
    finally:
        renderer.close()

    return PlaybackResult(
        rendered_frame_count=trajectory.frame_count,
        source_fps=source_fps,
        speed=speed,
    )


def run_replay(
    options: ReplayOptions,
    *,
    renderer_factory: RendererFactory = Renderer2D,
    clock: Clock = time.monotonic,
    sleeper: Sleeper = time.sleep,
) -> dict[str, object]:
    """Load one canonical artifact, render requested outputs, and return provenance."""

    if not isinstance(options, ReplayOptions):
        raise TypeError("options must be ReplayOptions")
    trajectory = load_trajectory_json(options.trajectory_path)
    canonical_bytes = canonical_trajectory_json_bytes(trajectory)

    overview: dict[str, str | int] | None = None
    if options.overview_path is not None:
        overview = _write_new_file(
            options.overview_path,
            render_trajectory_overview_png(trajectory),
        )

    playback: dict[str, int | float | bool] = {"requested": False}
    if options.play:
        playback = play_trajectory(
            trajectory,
            speed=options.speed,
            renderer_factory=renderer_factory,
            clock=clock,
            sleeper=sleeper,
        ).to_dict()

    return {
        "schema_version": REPLAY_SUMMARY_SCHEMA_VERSION,
        "status": "completed",
        "simulation_executed": False,
        "input": {
            "path": str(options.trajectory_path),
            "sha256": hashlib.sha256(canonical_bytes).hexdigest(),
            "size_bytes": len(canonical_bytes),
        },
        "trajectory": {
            "schema_version": trajectory.schema_version,
            "benchmark_version": trajectory.reset_info["benchmark_version"],
            "track_id": trajectory.reset_info["track_id"],
            "episode_seed": trajectory.reset_info["episode_seed"],
            "controller_seed": trajectory.reset_info["controller_seed"],
            "frame_count": trajectory.frame_count,
            "step_count": trajectory.step_count,
            "lap_completed": trajectory.final_info["lap_completed"],
            "lap_time_s": trajectory.final_info["lap_time_s"],
            "termination_reason": trajectory.final_info["termination_reason"],
        },
        "overview": overview,
        "playback": playback,
    }


def main(argv: Sequence[str] | None = None) -> None:
    """Run the replay CLI and print one machine-readable summary."""

    parser = _build_parser()
    values = parser.parse_args(argv)
    try:
        options = ReplayOptions(
            trajectory_path=values.trajectory,
            overview_path=values.overview,
            play=values.play,
            speed=values.speed,
        )
        result = run_replay(options)
    except (FileNotFoundError, OSError, ReplayCliError, TypeError, ValueError) as error:
        parser.exit(2, f"replay: error: {error}\n")
    print(json.dumps(result, allow_nan=False, sort_keys=True, separators=(",", ":")))


if __name__ == "__main__":
    main()
