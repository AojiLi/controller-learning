"""End-to-end M4 Controller, Gymnasium, DebugDraw, and renderer contract."""

from __future__ import annotations

from pathlib import Path

from controller_learning.config import load_project_config
from controller_learning.control import run_controller_episode
from controller_learning.envs import CarRacingEnv, RaceTermination
from controller_learning.tracks import (
    generate_track_candidate,
    generation_spec_from_project,
    pack_track,
    track_capacity_from_project,
)

PROJECT_ROOT = Path(__file__).parents[2]


def test_one_fresh_controller_episode_crosses_only_public_boundaries(tmp_path: Path) -> None:
    plugin = tmp_path / "controller"
    plugin.mkdir()
    (plugin / "config.toml").write_text("gain = 1.0\n", encoding="utf-8")
    (plugin / "controller.py").write_text(
        """
import numpy as np
from controller_learning.control import Controller

class OneStepController(Controller):
    def __init__(self, obs, info, config):
        assert config["controller"]["gain"] == 1.0
        assert "backend" not in config
        assert "controller_seed" in info

    def compute_control(self, obs, info=None):
        return np.asarray((np.nan, 0.0), dtype=np.float32)

    def render_callback(self, debug_draw):
        debug_draw.line((0.0, 0.0), (1.0, 0.0), color=(1.0, 0.0, 0.0))
""",
        encoding="utf-8",
    )
    project = load_project_config(PROJECT_ROOT)
    track = pack_track(
        generate_track_candidate(42, generation_spec_from_project(project)),
        track_capacity_from_project(project),
    )
    env = CarRacingEnv(
        project_config=project,
        level_id=1,
        track=track,
        backend="cpu_reference",
        render_mode="rgb_array",
    )
    try:
        result = run_controller_episode(env, plugin, reset_seed=9, render=True)

        assert result.steps == 1
        assert result.terminated is True
        assert result.truncated is False
        assert result.final_info["termination_reason"] == RaceTermination.INVALID_ACTION
        assert len(result.debug_commands) == 1
    finally:
        env.close()
