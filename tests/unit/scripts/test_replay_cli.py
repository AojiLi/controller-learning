"""Tests for the offline canonical-trajectory replay command."""

from __future__ import annotations

import inspect
from pathlib import Path
from typing import Any, ClassVar

import numpy as np
import pytest

from controller_learning.evaluation.trajectory import EpisodeTrajectory, write_trajectory_json
from scripts import replay as replay_cli


def _trajectory() -> EpisodeTrajectory:
    centerline = np.zeros((6, 2), dtype=np.float32)
    centerline[:5] = ((0, 0), (5, 0), (5, 5), (0, 5), (0, 0))
    left = centerline.copy()
    left[:5, 1] += 1.0
    right = centerline.copy()
    right[:5, 1] -= 1.0
    return EpisodeTrajectory(
        reset_info={
            "episode_seed": 11,
            "controller_seed": 12,
            "track_id": 13,
            "benchmark_version": "0.1",
            "termination_reason": 0,
            "lap_completed": False,
            "lap_time_s": 0.0,
        },
        final_info={
            "episode_seed": 11,
            "controller_seed": 12,
            "track_id": 13,
            "benchmark_version": "0.1",
            "termination_reason": 1,
            "lap_completed": True,
            "lap_time_s": 0.15,
        },
        centerline_m=centerline,
        left_boundary_m=left,
        right_boundary_m=right,
        track_mask=np.asarray((1, 1, 1, 1, 1, 0), dtype=np.int8),
        track_length_m=20.0,
        position_m=np.asarray(((0, 0), (2, 0.5), (4, 1.5), (5, 3)), dtype=np.float32),
        yaw_rad=np.asarray((0.0, 0.1, 0.4, 1.0), dtype=np.float32),
        velocity_body_mps=np.asarray(((0, 0), (2, 0), (3, 0), (2, 0)), dtype=np.float32),
        yaw_rate_rad_s=np.asarray((0, 0.1, 0.2, 0.1), dtype=np.float32),
        steering_angle_rad=np.asarray((0, 0.05, 0.1, 0.05), dtype=np.float32),
        track_progress=np.asarray((0, 0.3, 0.7, 1.0), dtype=np.float32),
        action=np.asarray(((0.05, 2), (0.1, 1), (0.05, -1)), dtype=np.float32),
        reward=np.asarray((0.2, 0.3, 1.0), dtype=np.float32),
        terminated=np.asarray((False, False, True)),
        truncated=np.asarray((False, False, False)),
    )


def _trajectory_file(tmp_path: Path) -> Path:
    path = tmp_path / "trajectory.json"
    write_trajectory_json(_trajectory(), path)
    return path


def test_options_require_an_output_and_validate_speed_and_suffix() -> None:
    with pytest.raises(ValueError, match="--overview"):
        replay_cli.ReplayOptions(Path("input.json"), None, False)
    with pytest.raises(ValueError, match="finite and positive"):
        replay_cli.ReplayOptions(Path("input.json"), None, True, speed=0.0)
    with pytest.raises(ValueError, match=r"\.png suffix"):
        replay_cli.ReplayOptions(Path("input.json"), Path("output.jpg"), False)


def test_replay_writes_deterministic_overview_and_refuses_overwrite(tmp_path: Path) -> None:
    trajectory_path = _trajectory_file(tmp_path)
    output_path = tmp_path / "nested" / "overview.png"
    options = replay_cli.ReplayOptions(trajectory_path, output_path, False)

    result = replay_cli.run_replay(options)

    assert result["simulation_executed"] is False
    assert result["trajectory"] == {
        "schema_version": "controller-learning-trajectory-v1",
        "benchmark_version": "0.1",
        "track_id": 13,
        "episode_seed": 11,
        "controller_seed": 12,
        "frame_count": 4,
        "step_count": 3,
        "lap_completed": True,
        "lap_time_s": 0.15,
        "termination_reason": 1,
    }
    assert output_path.read_bytes().startswith(b"\x89PNG\r\n\x1a\n")
    assert result["overview"]["size_bytes"] == output_path.stat().st_size
    assert result["playback"] == {"requested": False}

    original = output_path.read_bytes()
    with pytest.raises(replay_cli.ReplayCliError, match="already exists"):
        replay_cli.run_replay(options)
    assert output_path.read_bytes() == original


def test_replay_refuses_symlink_destination_without_changing_target(tmp_path: Path) -> None:
    trajectory_path = _trajectory_file(tmp_path)
    target = tmp_path / "target.png"
    target.write_bytes(b"keep-me")
    link = tmp_path / "overview.png"
    link.symlink_to(target)

    with pytest.raises(replay_cli.ReplayCliError, match="already exists"):
        replay_cli.run_replay(replay_cli.ReplayOptions(trajectory_path, link, False))
    assert target.read_bytes() == b"keep-me"


def test_interactive_playback_uses_every_public_frame_and_closes() -> None:
    trajectory = _trajectory()
    rendered_positions: list[tuple[float, float]] = []
    modes: list[str] = []
    closed: list[bool] = []
    sleep_calls: list[float] = []
    now = [10.0]

    class FakeRenderer:
        metadata: ClassVar = {"render_fps": 20}

        def render(self, observation: dict[str, Any]) -> None:
            rendered_positions.append(tuple(float(value) for value in observation["position"]))

        def close(self) -> None:
            closed.append(True)

    def factory(mode: str) -> FakeRenderer:
        modes.append(mode)
        return FakeRenderer()

    def sleep(duration: float) -> None:
        sleep_calls.append(duration)
        now[0] += duration

    result = replay_cli.play_trajectory(
        trajectory,
        speed=2.0,
        renderer_factory=factory,
        clock=lambda: now[0],
        sleeper=sleep,
    )

    assert modes == ["human"]
    assert rendered_positions == [(0.0, 0.0), (2.0, 0.5), (4.0, 1.5), (5.0, 3.0)]
    assert closed == [True]
    assert sleep_calls == pytest.approx([0.025, 0.025, 0.025])
    assert result.rendered_frame_count == 4
    assert result.source_fps == 20.0
    assert result.speed == 2.0


def test_interactive_playback_closes_renderer_after_render_failure() -> None:
    closed: list[bool] = []

    class BrokenRenderer:
        metadata: ClassVar = {"render_fps": 20}

        def render(self, _observation: object) -> None:
            raise RuntimeError("display failed")

        def close(self) -> None:
            closed.append(True)

    with pytest.raises(RuntimeError, match="display failed"):
        replay_cli.play_trajectory(
            _trajectory(),
            renderer_factory=lambda _mode: BrokenRenderer(),
        )
    assert closed == [True]


def test_replay_cli_has_no_simulation_backend_or_track_asset_access() -> None:
    source = inspect.getsource(replay_cli)

    assert "controller_learning.physics" not in source
    assert "controller_learning.tracks" not in source
    assert "controller_learning.rl" not in source
    assert "controller_learning.envs" not in source
