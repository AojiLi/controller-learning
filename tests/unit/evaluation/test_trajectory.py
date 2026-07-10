"""Tests for public-only Controller episode trajectory recording."""

from __future__ import annotations

import inspect
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from controller_learning.config import load_project_config
from controller_learning.envs.episode import PUBLIC_INFO_KEYS
from controller_learning.envs.observation import OBSERVATION_KEYS
from controller_learning.evaluation import (
    EpisodeTrajectory,
    load_trajectory_json,
    record_controller_episode,
    write_trajectory_json,
)
from controller_learning.evaluation import trajectory as trajectory_module

PROJECT_ROOT = Path(__file__).parents[3]


def _observation(step: int) -> dict[str, np.ndarray]:
    centerline = np.zeros((8, 2), dtype=np.float32)
    centerline[:5] = ((0, 0), (5, 0), (5, 5), (0, 5), (0, 0))
    left = np.zeros_like(centerline)
    left[:5] = ((0, 1), (4, 1), (4, 4), (1, 4), (0, 1))
    right = np.zeros_like(centerline)
    right[:5] = ((0, -1), (6, -1), (6, 6), (-1, 6), (0, -1))
    return {
        "position": np.asarray((float(step), 0.25 * step), dtype=np.float32),
        "yaw": np.asarray(0.1 * step, dtype=np.float32),
        "velocity_body": np.asarray((2.0 + step, 0.0), dtype=np.float32),
        "yaw_rate": np.asarray(0.01 * step, dtype=np.float32),
        "steering_angle": np.asarray(0.02 * step, dtype=np.float32),
        "track_progress": np.asarray(0.5 * step, dtype=np.float32),
        "centerline": centerline,
        "left_boundary": left,
        "right_boundary": right,
        "track_mask": np.asarray((1, 1, 1, 1, 1, 0, 0, 0), dtype=np.int8),
        "track_length": np.asarray(20.0, dtype=np.float32),
    }


class FakePublicEnv:
    def __init__(self, *, mutate_track: bool = False) -> None:
        self.project_config = load_project_config(PROJECT_ROOT)
        self.level_id = 1
        self.step_count = 0
        self.mutate_track = mutate_track
        self.closed = False

    @property
    def unwrapped(self) -> FakePublicEnv:
        return self

    def _info(self, *, terminal: bool) -> dict[str, Any]:
        return {
            "episode_seed": 101,
            "controller_seed": 202,
            "track_id": 303,
            "benchmark_version": "0.1",
            "termination_reason": 1 if terminal else 0,
            "lap_completed": terminal,
            "lap_time_s": 0.1 if terminal else 0.0,
            "private_backend_diagnostic": object(),
        }

    def reset(
        self,
        *,
        seed: int | None = None,
        options: dict[str, Any] | None = None,
    ) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
        assert seed == 7
        assert options == {"track_index": 4}
        self.step_count = 0
        return _observation(0), self._info(terminal=False)

    def step(
        self,
        action: object,
    ) -> tuple[dict[str, np.ndarray], float, bool, bool, dict[str, Any]]:
        np.testing.assert_allclose(action, (0.1, 1.5))
        self.step_count += 1
        terminal = self.step_count == 2
        observation = _observation(self.step_count)
        if self.mutate_track and self.step_count == 1:
            observation["centerline"][1, 0] += 0.5
        return observation, 0.25 * self.step_count, terminal, False, self._info(terminal=terminal)

    def close(self) -> None:
        self.closed = True


def _plugin(tmp_path: Path) -> Path:
    directory = tmp_path / "controller"
    directory.mkdir()
    (directory / "controller.py").write_text(
        """
import numpy as np
from controller_learning.control import Controller

class ConstantController(Controller):
    def compute_control(self, obs, info=None):
        assert tuple(info) == (
            "episode_seed", "controller_seed", "track_id", "benchmark_version",
            "termination_reason", "lap_completed", "lap_time_s",
        )
        return np.asarray((0.1, 1.5), dtype=np.float32)
""",
        encoding="utf-8",
    )
    (directory / "config.toml").write_text('name = "trajectory-test"\n', encoding="utf-8")
    return directory


def _recorded_trajectory(tmp_path: Path) -> EpisodeTrajectory:
    return record_controller_episode(
        FakePublicEnv(),
        _plugin(tmp_path),
        reset_seed=7,
        reset_options={"track_index": 4},
    ).trajectory


def test_recording_uses_normal_runner_and_copies_only_public_values(tmp_path: Path) -> None:
    env = FakePublicEnv()

    recorded = record_controller_episode(
        env,
        _plugin(tmp_path),
        reset_seed=7,
        reset_options={"track_index": 4},
    )

    trajectory = recorded.trajectory
    assert recorded.result.steps == trajectory.step_count == 2
    assert trajectory.frame_count == 3
    assert trajectory.total_reward == pytest.approx(0.75)
    assert tuple(trajectory.reset_info) == PUBLIC_INFO_KEYS
    assert tuple(trajectory.final_info) == PUBLIC_INFO_KEYS
    assert "private_backend_diagnostic" not in trajectory.final_info
    assert set(trajectory.observation(1)) == set(OBSERVATION_KEYS)
    np.testing.assert_allclose(trajectory.position_m, ((0, 0), (1, 0.25), (2, 0.5)))
    np.testing.assert_allclose(trajectory.action, ((0.1, 1.5), (0.1, 1.5)))
    np.testing.assert_array_equal(trajectory.terminated, (False, True))
    np.testing.assert_array_equal(trajectory.truncated, (False, False))
    assert env.closed is False
    assert not trajectory.position_m.flags.writeable
    with pytest.raises(ValueError, match="read-only"):
        trajectory.position_m[0, 0] = 9.0


def test_recording_rejects_track_geometry_mutation_within_episode(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match=r"track observation changed.*centerline"):
        record_controller_episode(
            FakePublicEnv(mutate_track=True),
            _plugin(tmp_path),
            reset_seed=7,
            reset_options={"track_index": 4},
        )


def test_canonical_json_roundtrip_is_deterministic_and_hash_bound(tmp_path: Path) -> None:
    trajectory = _recorded_trajectory(tmp_path)
    first_path = tmp_path / "first.json"
    second_path = tmp_path / "nested" / "second.json"

    first = write_trajectory_json(trajectory, first_path)
    second = write_trajectory_json(trajectory, second_path)

    assert first_path.read_bytes() == second_path.read_bytes()
    assert first.sha256 == second.sha256
    assert first.size_bytes == second.size_bytes == first_path.stat().st_size
    loaded = load_trajectory_json(first_path, expected_sha256=first.sha256)
    np.testing.assert_array_equal(loaded.position_m, trajectory.position_m)
    np.testing.assert_array_equal(loaded.action, trajectory.action)
    assert dict(loaded.final_info) == dict(trajectory.final_info)

    with pytest.raises(ValueError, match="does not match"):
        load_trajectory_json(first_path, expected_sha256="0" * 64)


def test_loader_rejects_noncanonical_symlink_and_oversize_inputs(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    trajectory = _recorded_trajectory(tmp_path)
    path = tmp_path / "trajectory.json"
    artifact = write_trajectory_json(trajectory, path)

    path.write_bytes(b" " + path.read_bytes())
    with pytest.raises(ValueError, match="not in canonical"):
        load_trajectory_json(path)

    write_trajectory_json(trajectory, path)
    link = tmp_path / "linked.json"
    link.symlink_to(path)
    with pytest.raises(ValueError, match="non-symlink regular file"):
        load_trajectory_json(link)
    with pytest.raises(ValueError, match="symbolic link"):
        write_trajectory_json(trajectory, link)

    monkeypatch.setattr(trajectory_module, "MAX_TRAJECTORY_JSON_BYTES", artifact.size_bytes - 1)
    with pytest.raises(ValueError, match="input size"):
        load_trajectory_json(path)


def test_trajectory_module_has_no_backend_or_track_asset_access() -> None:
    source = inspect.getsource(trajectory_module)

    assert "controller_learning.physics" not in source
    assert "controller_learning.tracks.assets" not in source
    assert "controller_learning.rl" not in source
    assert "validation" not in source.lower()
    assert "test" not in source.lower()
