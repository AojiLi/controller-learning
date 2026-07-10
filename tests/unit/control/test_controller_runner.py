"""Tests for the fresh-per-episode single Controller Runner."""

from __future__ import annotations

from collections.abc import Mapping
from inspect import signature
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from controller_learning.config import ProjectConfig, load_project_config
from controller_learning.control import (
    ControllerExecutionError,
    EpisodeStepLimitError,
    run_controller_episode,
)
from controller_learning.control.debug_draw import DebugDrawCommand, LineCommand
from controller_learning.envs.episode import PUBLIC_INFO_KEYS

PROJECT_ROOT = Path(__file__).parents[3]


def _write_plugin(directory: Path, source: str, config: str = "gain = 2.0\n") -> Path:
    directory.mkdir()
    (directory / "controller.py").write_text(source, encoding="utf-8")
    (directory / "config.toml").write_text(config, encoding="utf-8")
    return directory


class FakeEnv:
    """Minimal deterministic single environment with inspectable lifecycle events."""

    def __init__(
        self,
        *,
        episode_steps: int = 2,
        project_config: ProjectConfig | None = None,
        level_id: int = 1,
        info_diagnostics: Mapping[str, Any] | None = None,
        omitted_info_keys: tuple[str, ...] = (),
    ) -> None:
        self.episode_steps = episode_steps
        self.project_config = project_config or load_project_config(PROJECT_ROOT)
        self.level_id = level_id
        self.info_diagnostics = dict(info_diagnostics or {})
        self.omitted_info_keys = omitted_info_keys
        self.step_count = 0
        self.events: list[Any] = []
        self.actions: list[Any] = []
        self.closed = False
        self.reset_seeds: list[int | None] = []
        self.debug_frames: list[tuple[DebugDrawCommand, ...]] = []

    @property
    def unwrapped(self) -> FakeEnv:
        return self

    def _obs(self) -> dict[str, Any]:
        return {"events": self.events, "step": self.step_count}

    def _info(self) -> dict[str, Any]:
        info = {
            "episode_seed": 11,
            "controller_seed": 29,
            "track_id": "trackgen-v1:7",
            "benchmark_version": "v0.1",
            "termination_reason": 0,
            "lap_completed": False,
            "lap_time_s": 0.0,
            **self.info_diagnostics,
        }
        for key in self.omitted_info_keys:
            info.pop(key, None)
        return info

    def reset(self, *, seed: int | None = None) -> tuple[dict[str, Any], dict[str, Any]]:
        self.step_count = 0
        self.reset_seeds.append(seed)
        self.events.append("reset")
        return self._obs(), self._info()

    def step(self, action: Any) -> tuple[dict[str, Any], float, bool, bool, dict[str, Any]]:
        self.actions.append(action)
        self.events.append(("env.step", action))
        self.step_count += 1
        terminated = self.step_count >= self.episode_steps
        info = self._info()
        info["termination_reason"] = 1 if terminated else 0
        info["lap_completed"] = terminated
        info["lap_time_s"] = self.step_count * 0.05 if terminated else 0.0
        return self._obs(), float(self.step_count), terminated, False, info

    def render(self) -> None:
        self.events.append("env.render")

    def render_debug_frame(self, commands: tuple[DebugDrawCommand, ...]) -> None:
        self.debug_frames.append(commands)
        self.events.append(("render_debug_frame", commands))

    def close(self) -> None:
        self.closed = True
        self.events.append("env.close")


ORDERED_PLUGIN = """
import numpy as np
from controller_learning.control import Controller

class Ordered(Controller):
    def __init__(self, obs, info, config):
        self.events = obs["events"]
        self.calls = 0
        self.events.append(("init", tuple(config), config["controller"]["gain"]))
    def compute_control(self, obs, info=None):
        self.calls += 1
        self.events.append(("compute", self.calls))
        return np.array([self.calls, -self.calls], dtype=np.float32)
    def step_callback(self, action, obs, reward, terminated, truncated, info):
        self.events.append(("step_callback", reward, terminated, truncated))
    def episode_callback(self):
        self.events.append(("episode_callback", self.calls))
    def render_callback(self, debug_draw):
        public = tuple(name for name in dir(debug_draw) if not name.startswith("_"))
        self.events.append(("render_callback", public))
        debug_draw.line((0, 0), (self.calls, 1))
"""


def test_runner_orders_lifecycle_and_returns_an_immutable_result(tmp_path: Path) -> None:
    plugin = _write_plugin(tmp_path / "plugin", ORDERED_PLUGIN)
    env = FakeEnv(episode_steps=2)

    result = run_controller_episode(
        env,
        plugin,
        reset_seed=73,
    )

    assert env.reset_seeds == [73]
    assert env.events == [
        "reset",
        (
            "init",
            (
                "benchmark_version",
                "level_id",
                "level_name",
                "control_dt_s",
                "vehicle",
                "action_limits",
                "track",
                "controller",
            ),
            2.0,
        ),
        ("compute", 1),
        ("env.step", env.actions[0]),
        ("step_callback", 1.0, False, False),
        ("compute", 2),
        ("env.step", env.actions[1]),
        ("step_callback", 2.0, True, False),
        ("episode_callback", 2),
    ]
    assert result.steps == 2
    assert result.total_reward == 3.0
    assert result.terminated is True
    assert result.truncated is False
    assert result.final_info["lap_completed"] is True
    assert result.debug_commands == ()
    assert isinstance(result.final_info, Mapping)
    with pytest.raises(TypeError):
        result.final_info["track_id"] = "changed"  # type: ignore[index]
    assert env.closed is False


def test_runner_constructs_fresh_controller_state_for_every_call(tmp_path: Path) -> None:
    plugin = _write_plugin(tmp_path / "plugin", ORDERED_PLUGIN)
    env = FakeEnv(episode_steps=1)
    first = run_controller_episode(env, plugin, 1)
    second = run_controller_episode(env, plugin, 2)

    assert first.steps == second.steps == 1
    np.testing.assert_array_equal(env.actions[0], [1.0, -1.0])
    np.testing.assert_array_equal(env.actions[1], [1.0, -1.0])
    assert [event for event in env.events if event == ("episode_callback", 1)] == [
        ("episode_callback", 1),
        ("episode_callback", 1),
    ]


def test_render_callback_receives_only_writer_and_precedes_environment_render(
    tmp_path: Path,
) -> None:
    plugin = _write_plugin(tmp_path / "plugin", ORDERED_PLUGIN)
    env = FakeEnv(episode_steps=2)

    result = run_controller_episode(
        env,
        plugin,
        7,
        render=True,
    )

    render_event = ("render_callback", ("line", "points", "text"))
    callback_indices = [index for index, event in enumerate(env.events) if event == render_event]
    frame_indices = [
        index
        for index, event in enumerate(env.events)
        if isinstance(event, tuple) and event[0] == "render_debug_frame"
    ]
    render_indices = [index for index, event in enumerate(env.events) if event == "env.render"]
    assert len(callback_indices) == len(frame_indices) == len(render_indices) == 2
    assert all(
        callback < frame < rendered
        for callback, frame, rendered in zip(
            callback_indices,
            frame_indices,
            render_indices,
            strict=True,
        )
    )

    assert tuple(len(frame) for frame in env.debug_frames) == (1, 1)
    assert all(isinstance(frame[0], LineCommand) for frame in env.debug_frames)
    assert [float(frame[0].end[0]) for frame in env.debug_frames] == [1.0, 2.0]
    assert result.debug_commands == env.debug_frames[0] + env.debug_frames[1]


def test_runner_filters_wrapper_diagnostics_from_every_controller_info(tmp_path: Path) -> None:
    plugin = _write_plugin(
        tmp_path / "plugin",
        """
import numpy as np
from controller_learning.control import Controller
class InfoCapture(Controller):
    def __init__(self, obs, info, config):
        self.events = obs["events"]
        self.events.append(("init_info", tuple(info)))
    def compute_control(self, obs, info=None):
        self.events.append(("compute_info", tuple(info)))
        return np.zeros(2, dtype=np.float32)
    def step_callback(self, action, obs, reward, terminated, truncated, info):
        self.events.append(("step_info", tuple(info)))
""",
    )
    private_diagnostic = object()
    env = FakeEnv(
        episode_steps=1,
        info_diagnostics={
            "wrapper_diagnostic": private_diagnostic,
            "final_observation": {"simulator": private_diagnostic},
        },
    )

    result = run_controller_episode(env, plugin, 17)

    info_events = [event for event in env.events if str(event[0]).endswith("_info")]
    assert info_events == [
        ("init_info", PUBLIC_INFO_KEYS),
        ("compute_info", PUBLIC_INFO_KEYS),
        ("step_info", PUBLIC_INFO_KEYS),
    ]
    assert tuple(result.final_info) == PUBLIC_INFO_KEYS
    assert "wrapper_diagnostic" not in result.final_info
    assert "final_observation" not in result.final_info


@pytest.mark.parametrize(
    ("environment", "error", "match"),
    [
        (
            FakeEnv(omitted_info_keys=("controller_seed",)),
            ValueError,
            "missing public field.*controller_seed",
        ),
        (
            FakeEnv(info_diagnostics={"lap_time_s": 0}),
            TypeError,
            "lap_time_s.*must have type float",
        ),
        (
            FakeEnv(info_diagnostics={"lap_completed": 0}),
            TypeError,
            "lap_completed.*must have type bool",
        ),
    ],
)
def test_runner_rejects_missing_or_invalid_public_info(
    tmp_path: Path,
    environment: FakeEnv,
    error: type[Exception],
    match: str,
) -> None:
    plugin = _write_plugin(tmp_path / "plugin", ORDERED_PLUGIN)

    with pytest.raises(error, match=match):
        run_controller_episode(environment, plugin, 1)


def test_runner_derives_project_config_and_level_only_from_environment(tmp_path: Path) -> None:
    plugin = _write_plugin(
        tmp_path / "plugin",
        """
import numpy as np
from controller_learning.control import Controller
class ConfigCapture(Controller):
    def __init__(self, obs, info, config):
        obs["events"].append(("public_level", config["level_id"], config["level_name"]))
    def compute_control(self, obs, info=None):
        return np.zeros(2, dtype=np.float32)
""",
    )
    project = load_project_config(PROJECT_ROOT)
    env = FakeEnv(episode_steps=1, project_config=project, level_id=0)

    parameters = signature(run_controller_episode).parameters
    assert "project_config" not in parameters
    assert "level_id" not in parameters
    run_controller_episode(env, plugin, 1)

    assert ("public_level", 0, project.levels[0].name) in env.events


def test_runner_rejects_nonofficial_environment_challenge_types(tmp_path: Path) -> None:
    plugin = _write_plugin(tmp_path / "plugin", ORDERED_PLUGIN)
    invalid_project = FakeEnv()
    invalid_project.project_config = object()  # type: ignore[assignment]
    invalid_level = FakeEnv()
    invalid_level.level_id = np.int64(1)  # type: ignore[assignment]

    with pytest.raises(TypeError, match="project_config must be a ProjectConfig"):
        run_controller_episode(invalid_project, plugin, 1)
    with pytest.raises(TypeError, match="level_id must be an integer"):
        run_controller_episode(invalid_level, plugin, 1)

    assert invalid_project.events == []
    assert invalid_level.events == []


@pytest.mark.parametrize(
    ("source", "phase", "render"),
    [
        ("raise RuntimeError('import failed')\n", "import", False),
        (
            "from controller_learning.control import Controller\n"
            "class Broken(Controller):\n"
            " def __init__(self, obs, info, config): raise RuntimeError('init failed')\n"
            " def compute_control(self, obs, info=None): return [0, 0]\n",
            "init",
            False,
        ),
        (
            "from controller_learning.control import Controller\n"
            "class Broken(Controller):\n"
            " def compute_control(self, obs, info=None): raise RuntimeError('compute failed')\n"
            " def episode_callback(self): self.episode_called = True\n",
            "compute",
            False,
        ),
        (
            "from controller_learning.control import Controller\n"
            "class Broken(Controller):\n"
            " def compute_control(self, obs, info=None): return [0, 0]\n"
            " def step_callback(self, *args): raise RuntimeError('step callback failed')\n",
            "step_callback",
            False,
        ),
        (
            "from controller_learning.control import Controller\n"
            "class Broken(Controller):\n"
            " def compute_control(self, obs, info=None): return [0, 0]\n"
            " def render_callback(self, debug_draw): raise RuntimeError('render failed')\n",
            "render_callback",
            True,
        ),
        (
            "from controller_learning.control import Controller\n"
            "class Broken(Controller):\n"
            " def compute_control(self, obs, info=None): return [0, 0]\n"
            " def episode_callback(self): raise RuntimeError('episode failed')\n",
            "episode_callback",
            False,
        ),
    ],
)
def test_plugin_failures_report_the_lifecycle_phase_and_cause(
    tmp_path: Path,
    source: str,
    phase: str,
    render: bool,
) -> None:
    plugin = _write_plugin(tmp_path / "plugin", source)
    env = FakeEnv(episode_steps=1)

    with pytest.raises(ControllerExecutionError) as caught:
        run_controller_episode(
            env,
            plugin,
            1,
            render=render,
        )

    assert caught.value.phase == phase
    assert isinstance(caught.value.cause, Exception)


def test_episode_callback_runs_once_after_compute_failure(tmp_path: Path) -> None:
    plugin = _write_plugin(
        tmp_path / "plugin",
        """
from controller_learning.control import Controller
class Broken(Controller):
    def __init__(self, obs, info, config): self.events = obs["events"]
    def compute_control(self, obs, info=None): raise RuntimeError("broken")
    def episode_callback(self): self.events.append("episode_callback_after_error")
""",
    )
    env = FakeEnv()

    with pytest.raises(ControllerExecutionError, match="compute"):
        run_controller_episode(env, plugin, 1)

    assert env.events.count("episode_callback_after_error") == 1


def test_max_steps_is_a_runner_error_not_a_challenge_truncation(tmp_path: Path) -> None:
    plugin = _write_plugin(tmp_path / "plugin", ORDERED_PLUGIN)
    env = FakeEnv(episode_steps=5)

    with pytest.raises(EpisodeStepLimitError) as caught:
        run_controller_episode(
            env,
            plugin,
            1,
            max_steps=2,
        )

    assert caught.value.steps == 2
    assert caught.value.max_steps == 2
    assert len(env.actions) == 2
    assert env.events.count(("episode_callback", 2)) == 1
    assert env.closed is False


def test_runner_leaves_action_validation_to_the_environment(tmp_path: Path) -> None:
    plugin = _write_plugin(
        tmp_path / "plugin",
        """
from controller_learning.control import Controller
class ArbitraryAction(Controller):
    def compute_control(self, obs, info=None): return "environment-validates-me"
""",
    )
    env = FakeEnv(episode_steps=1)

    run_controller_episode(env, plugin, 1)

    assert env.actions == ["environment-validates-me"]


@pytest.mark.parametrize("interrupt", ["KeyboardInterrupt", "SystemExit"])
def test_process_control_exceptions_are_not_wrapped(tmp_path: Path, interrupt: str) -> None:
    plugin = _write_plugin(
        tmp_path / "plugin",
        "from controller_learning.control import Controller\n"
        "class Interrupted(Controller):\n"
        f" def compute_control(self, obs, info=None): raise {interrupt}()\n",
    )
    env = FakeEnv()
    expected = KeyboardInterrupt if interrupt == "KeyboardInterrupt" else SystemExit

    with pytest.raises(expected):
        run_controller_episode(env, plugin, 1)
