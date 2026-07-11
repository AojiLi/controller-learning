"""Tests for deterministic public-trajectory 2D artifacts."""

from __future__ import annotations

import inspect
from pathlib import Path

import matplotlib.image as mpimg
import numpy as np
import pytest

from controller_learning.evaluation import EpisodeTrajectory
from controller_learning.visualization import (
    render_trajectory_overview_png,
    write_trajectory_overview_png,
)
from controller_learning.visualization import replay as replay_module


def _trajectory() -> EpisodeTrajectory:
    centerline = np.zeros((8, 2), dtype=np.float32)
    centerline[:5] = ((0, 0), (5, 0), (5, 5), (0, 5), (0, 0))
    left = np.zeros_like(centerline)
    left[:5] = ((0, 1), (4, 1), (4, 4), (1, 4), (0, 1))
    right = np.zeros_like(centerline)
    right[:5] = ((0, -1), (6, -1), (6, 6), (-1, 6), (0, -1))
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
        track_mask=np.asarray((1, 1, 1, 1, 1, 0, 0, 0), dtype=np.int8),
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


def test_overview_png_is_byte_deterministic_and_records_every_source_frame(
    tmp_path: Path,
) -> None:
    trajectory = _trajectory()

    first = write_trajectory_overview_png(trajectory, tmp_path / "first.png")
    second = write_trajectory_overview_png(trajectory, tmp_path / "second.png")

    assert first.path.read_bytes() == second.path.read_bytes()
    assert first.path.read_bytes() == render_trajectory_overview_png(trajectory)
    assert first.path.read_bytes().startswith(b"\x89PNG\r\n\x1a\n")
    assert first.sha256 == second.sha256
    assert first.size_bytes == second.size_bytes == first.path.stat().st_size
    assert first.source_frame_count == trajectory.frame_count
    assert first.rendered_frame_indices == tuple(range(trajectory.frame_count))
    image = mpimg.imread(first.path)
    assert image.shape == (600, 800, 4)


def test_overview_rejects_wrong_suffix_and_symlink_destination(tmp_path: Path) -> None:
    trajectory = _trajectory()
    with pytest.raises(ValueError, match=r"\.png suffix"):
        write_trajectory_overview_png(trajectory, tmp_path / "overview.jpg")

    real_path = tmp_path / "real.png"
    real_path.write_bytes(b"existing")
    link_path = tmp_path / "link.png"
    link_path.symlink_to(real_path)
    with pytest.raises(ValueError, match="symbolic link"):
        write_trajectory_overview_png(trajectory, link_path)
    assert real_path.read_bytes() == b"existing"


def test_replay_renderer_has_no_backend_or_private_state_access() -> None:
    source = inspect.getsource(replay_module)

    assert "controller_learning.physics" not in source
    assert "controller_learning.tracks" not in source
    assert "controller_learning.rl" not in source
    assert "unwrapped" not in source
