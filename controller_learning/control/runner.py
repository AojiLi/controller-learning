"""Run one trusted Controller against one public Gymnasium episode."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass
from functools import partial
from pathlib import Path
from types import MappingProxyType
from typing import TYPE_CHECKING, Any, Literal, TypeAlias, TypeVar

from controller_learning.config.models import ProjectConfig
from controller_learning.control.configuration import build_public_controller_config
from controller_learning.control.debug_draw import DebugDrawCommand, _DebugDrawBuffer
from controller_learning.control.loader import load_controller, load_controller_config

if TYPE_CHECKING:
    import gymnasium as gym

ControllerExecutionPhase: TypeAlias = Literal[
    "import",
    "init",
    "compute",
    "step_callback",
    "render_callback",
    "episode_callback",
]

_Result = TypeVar("_Result")


class ControllerExecutionError(RuntimeError):
    """A Controller plugin failed during a named lifecycle phase."""

    def __init__(self, phase: ControllerExecutionPhase, cause: Exception) -> None:
        self.phase = phase
        self.cause = cause
        super().__init__(f"Controller failed during {phase}: {type(cause).__name__}: {cause}")


class EpisodeStepLimitError(RuntimeError):
    """The optional Runner safety guard fired before the Challenge ended."""

    def __init__(self, *, steps: int, max_steps: int) -> None:
        self.steps = steps
        self.max_steps = max_steps
        super().__init__(
            f"Runner max_steps safety guard reached {max_steps} steps before episode completion"
        )


@dataclass(frozen=True, slots=True)
class EpisodeRunResult:
    """Immutable outcome of one normally completed Controller episode."""

    steps: int
    total_reward: float
    terminated: bool
    truncated: bool
    final_info: Mapping[str, Any]
    debug_commands: tuple[DebugDrawCommand, ...]


def _plugin_call(
    phase: ControllerExecutionPhase,
    callback: Callable[[], _Result],
) -> _Result:
    try:
        return callback()
    except (KeyboardInterrupt, SystemExit):
        raise
    except Exception as error:
        raise ControllerExecutionError(phase, error) from error


def _public_info_keys() -> tuple[str, ...]:
    # Keep ``import controller_learning.control`` independent from the environment/JAX stack. The
    # canonical schema is loaded only when the Runner is actually used with an environment.
    from controller_learning.envs.episode import PUBLIC_INFO_KEYS

    return PUBLIC_INFO_KEYS


def _public_info(value: object, *, source: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise TypeError(f"{source} info must be a mapping")

    public_keys = _public_info_keys()
    missing = tuple(key for key in public_keys if key not in value)
    if missing:
        raise ValueError(f"{source} info is missing public field(s): {', '.join(missing)}")

    expected_types: dict[str, type] = {
        "episode_seed": int,
        "controller_seed": int,
        "track_id": int,
        "benchmark_version": str,
        "termination_reason": int,
        "lap_completed": bool,
        "lap_time_s": float,
    }
    filtered: dict[str, Any] = {}
    for key in public_keys:
        item = value[key]
        expected = expected_types[key]
        if type(item) is not expected:
            raise TypeError(
                f"{source} info field {key!r} must have type {expected.__name__}, "
                f"got {type(item).__name__}"
            )
        filtered[key] = item
    return MappingProxyType(filtered)


def _challenge_from_environment(env: object) -> tuple[object, ProjectConfig, int]:
    try:
        unwrapped = env.unwrapped  # type: ignore[attr-defined]
    except AttributeError as error:
        raise TypeError("env must expose the Gymnasium unwrapped environment") from error

    try:
        project_config = unwrapped.project_config
    except AttributeError as error:
        raise TypeError("env.unwrapped must expose project_config") from error
    if not isinstance(project_config, ProjectConfig):
        raise TypeError("env.unwrapped.project_config must be a ProjectConfig")

    try:
        level_id = unwrapped.level_id
    except AttributeError as error:
        raise TypeError("env.unwrapped must expose level_id") from error
    if type(level_id) is not int:
        raise TypeError("env.unwrapped.level_id must be an integer")
    return unwrapped, project_config, level_id


def _debug_frame_sink(unwrapped: object) -> Callable[[tuple[DebugDrawCommand, ...]], None]:
    try:
        callback = unwrapped.render_debug_frame  # type: ignore[attr-defined]
    except AttributeError as error:
        raise TypeError(
            "env.unwrapped must expose render_debug_frame(commands) when render=True"
        ) from error
    if not callable(callback):
        raise TypeError("env.unwrapped.render_debug_frame must be callable")
    return callback


def _validate_max_steps(max_steps: int | None) -> None:
    if max_steps is None:
        return
    if isinstance(max_steps, bool) or not isinstance(max_steps, int):
        raise TypeError("max_steps must be a positive integer or None")
    if max_steps <= 0:
        raise ValueError("max_steps must be positive")


def run_controller_episode(
    env: gym.Env,
    controller_directory: str | Path,
    reset_seed: int,
    render: bool = False,
    max_steps: int | None = None,
) -> EpisodeRunResult:
    """Run a fresh Controller instance until the environment ends one episode.

    Public Challenge configuration is derived exclusively from ``env.unwrapped`` so callers cannot
    pair an episode with a different ProjectConfig or Level. The Runner owns neither ``env`` nor
    its lifecycle and therefore never closes it.

    ``max_steps`` is only a host-side safety guard: reaching it raises
    :class:`EpisodeStepLimitError` and does not synthesize a Challenge truncation or mutate
    environment state.
    """

    _validate_max_steps(max_steps)
    if not isinstance(render, bool):
        raise TypeError("render must be a boolean")
    unwrapped, project_config, level_id = _challenge_from_environment(env)
    render_debug_frame = _debug_frame_sink(unwrapped) if render else None

    obs, reset_info_value = env.reset(seed=reset_seed)
    reset_info = _public_info(reset_info_value, source="reset")

    controller_class = _plugin_call("import", lambda: load_controller(controller_directory))
    controller_parameters = _plugin_call(
        "import", lambda: load_controller_config(controller_directory)
    )
    public_config = build_public_controller_config(
        project_config,
        level_id,
        controller_parameters,
    )
    controller = _plugin_call(
        "init",
        lambda: controller_class(obs, reset_info, public_config),
    )

    debug_buffer = _DebugDrawBuffer()
    episode_debug_commands: list[DebugDrawCommand] = []
    steps = 0
    total_reward = 0.0
    terminated = False
    truncated = False
    final_info = reset_info
    pending_error: BaseException | None = None

    try:
        while not (terminated or truncated):
            if max_steps is not None and steps >= max_steps:
                raise EpisodeStepLimitError(steps=steps, max_steps=max_steps)

            action = _plugin_call(
                "compute",
                partial(controller.compute_control, obs, final_info),
            )
            next_obs, reward_value, terminated_value, truncated_value, info_value = env.step(action)
            reward = float(reward_value)
            terminated = bool(terminated_value)
            truncated = bool(truncated_value)
            final_info = _public_info(info_value, source="step")
            steps += 1
            total_reward += reward

            _plugin_call(
                "step_callback",
                partial(
                    controller.step_callback,
                    action,
                    next_obs,
                    reward,
                    terminated,
                    truncated,
                    final_info,
                ),
            )
            obs = next_obs

            if render:
                _plugin_call(
                    "render_callback",
                    lambda: controller.render_callback(debug_buffer.writer),
                )
                frame_commands = debug_buffer.drain()
                episode_debug_commands.extend(frame_commands)
                if render_debug_frame is None:  # pragma: no cover - established before reset
                    raise AssertionError("render debug-frame sink is unavailable")
                render_debug_frame(frame_commands)
                env.render()
    except BaseException as error:
        pending_error = error

    try:
        _plugin_call("episode_callback", controller.episode_callback)
    except (KeyboardInterrupt, SystemExit):
        raise
    except ControllerExecutionError as callback_error:
        if pending_error is None:
            raise
        pending_error.add_note(str(callback_error))

    if pending_error is not None:
        raise pending_error

    return EpisodeRunResult(
        steps=steps,
        total_reward=total_reward,
        terminated=terminated,
        truncated=truncated,
        final_info=final_info,
        debug_commands=tuple(episode_debug_commands),
    )


__all__ = [
    "ControllerExecutionError",
    "ControllerExecutionPhase",
    "EpisodeRunResult",
    "EpisodeStepLimitError",
    "run_controller_episode",
]
