"""Run one trusted Controller on one generated Level 1 track."""

from __future__ import annotations

import argparse
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Literal, TypeAlias

import numpy as np

from controller_learning.config import ConfigError, ProjectConfig, load_project_config
from controller_learning.control import EpisodeRunResult, run_controller_episode
from controller_learning.tracks import (
    Track,
    TrackGenerationError,
    generate_track_candidate,
    generation_spec_from_project,
    pack_track,
    track_capacity_from_project,
    validate_track_candidate,
    validation_spec_from_project,
)

PROJECT_ROOT = Path(__file__).resolve().parents[1]
UINT32_MAX = int(np.iinfo(np.uint32).max)
VehicleBackend: TypeAlias = Literal["cpu_reference", "mjx_warp"]
JsonScalar: TypeAlias = str | int | float | bool | None


class SimulationCliError(RuntimeError):
    """Raised when a requested single-episode simulation cannot be prepared."""


@dataclass(frozen=True, slots=True)
class SimulationOptions:
    """Validated command-line choices for one Controller episode."""

    controller_directory: Path
    track_seed: int
    environment_seed: int
    backend: VehicleBackend
    level_id: int
    render: bool


def _uint32_argument(value: str) -> int:
    try:
        parsed = int(value)
    except ValueError as error:
        raise argparse.ArgumentTypeError("must be an integer in the uint32 range") from error
    if not 0 <= parsed <= UINT32_MAX:
        raise argparse.ArgumentTypeError(f"must be between 0 and {UINT32_MAX}")
    return parsed


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--controller",
        dest="controller_directory",
        type=Path,
        default=Path("controllers/template"),
        help="Controller plugin directory, relative to the project root by default",
    )
    parser.add_argument(
        "--track-seed",
        type=_uint32_argument,
        default=42,
        help="Exact procedural Track seed (default: 42)",
    )
    parser.add_argument(
        "--env-seed",
        dest="environment_seed",
        type=_uint32_argument,
        default=0,
        help="Environment episode seed (default: 0)",
    )
    parser.add_argument(
        "--backend",
        choices=("cpu_reference", "mjx_warp"),
        default="cpu_reference",
        help="Explicit vehicle backend (default: cpu_reference)",
    )
    parser.add_argument(
        "--level-id",
        type=int,
        choices=(1,),
        default=1,
        help="Challenge Level; M4 provides procedural Level 1 only (default: 1)",
    )
    parser.add_argument(
        "--render",
        action="store_true",
        help="Open the interactive 2D view and Controller DebugDraw output",
    )
    return parser


def _parse_args(argv: list[str] | None = None) -> SimulationOptions:
    args = _build_parser().parse_args(argv)
    return SimulationOptions(
        controller_directory=args.controller_directory,
        track_seed=args.track_seed,
        environment_seed=args.environment_seed,
        backend=args.backend,
        level_id=args.level_id,
        render=args.render,
    )


def _require_uint32(value: object, *, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or not 0 <= value <= UINT32_MAX:
        raise SimulationCliError(f"{name} must be an integer between 0 and {UINT32_MAX}")
    return value


def _resolve_project_root(value: str | Path) -> Path:
    root = Path(value).expanduser().resolve()
    if not root.is_dir():
        raise SimulationCliError(f"project root is not a directory: {root}")
    return root


def _resolve_controller_directory(value: str | Path, *, project_root: Path) -> Path:
    directory = Path(value).expanduser()
    if not directory.is_absolute():
        directory = project_root / directory
    directory = directory.resolve()
    if not directory.is_dir():
        raise SimulationCliError(f"Controller directory does not exist: {directory}")
    missing = [
        name for name in ("controller.py", "config.toml") if not (directory / name).is_file()
    ]
    if missing:
        raise SimulationCliError(
            f"Controller directory {directory} is missing required file(s): {', '.join(missing)}"
        )
    return directory


def _generate_validated_track(config: ProjectConfig, track_seed: int) -> Track:
    """Generate, validate, and pack exactly one requested seed without retrying."""

    seed = _require_uint32(track_seed, name="track seed")
    generation_spec = generation_spec_from_project(config)
    try:
        candidate = generate_track_candidate(seed, generation_spec)
    except TrackGenerationError as error:
        raise SimulationCliError(
            f"Track seed {seed} could not be generated ({error.reason}): {error}"
        ) from error

    validation = validate_track_candidate(candidate, validation_spec_from_project(config))
    if not validation.valid:
        reasons = ", ".join(validation.reasons)
        raise SimulationCliError(f"Track seed {seed} failed geometry validation: {reasons}")

    try:
        return pack_track(candidate, track_capacity_from_project(config))
    except TrackGenerationError as error:
        raise SimulationCliError(
            f"Track seed {seed} could not be packed ({error.reason}): {error}"
        ) from error


def _display_controller_path(directory: Path, project_root: Path) -> str:
    try:
        return directory.relative_to(project_root).as_posix()
    except ValueError:
        return str(directory)


def _create_environment(**kwargs):
    # Keep argument parsing and ``--help`` independent from optional GPU backend imports.
    from controller_learning.envs import CarRacingEnv

    return CarRacingEnv(**kwargs)


def _episode_summary(
    result: EpisodeRunResult,
    *,
    options: SimulationOptions,
    controller_directory: Path,
    project_root: Path,
) -> dict[str, JsonScalar]:
    info = result.final_info
    summary: dict[str, JsonScalar] = {
        "backend": options.backend,
        "benchmark_version": str(info["benchmark_version"]),
        "controller": _display_controller_path(controller_directory, project_root),
        "environment_seed": options.environment_seed,
        "lap_completed": bool(info["lap_completed"]),
        "lap_time_s": float(info["lap_time_s"]),
        "level_id": options.level_id,
        "steps": int(result.steps),
        "terminated": bool(result.terminated),
        "termination_reason": int(info["termination_reason"]),
        "total_reward": float(result.total_reward),
        "track_id": int(info["track_id"]),
        "track_seed": options.track_seed,
        "truncated": bool(result.truncated),
    }
    for key in ("lap_time_s", "total_reward"):
        value = summary[key]
        if not isinstance(value, float) or not math.isfinite(value):
            raise SimulationCliError(f"episode summary field {key!r} must be finite")
    return summary


def _run_simulation(
    options: SimulationOptions,
    *,
    project_root: str | Path = PROJECT_ROOT,
) -> dict[str, JsonScalar]:
    if options.level_id != 1:
        raise SimulationCliError(
            "M4 simulation supports only Level 1; the Level 0 asset is M5 work"
        )
    _require_uint32(options.track_seed, name="track seed")
    _require_uint32(options.environment_seed, name="environment seed")
    if options.backend not in ("cpu_reference", "mjx_warp"):
        raise SimulationCliError("backend must be 'cpu_reference' or 'mjx_warp'")
    if not isinstance(options.render, bool):
        raise SimulationCliError("render must be a boolean")

    root = _resolve_project_root(project_root)
    controller_directory = _resolve_controller_directory(
        options.controller_directory,
        project_root=root,
    )
    config = load_project_config(root)
    track = _generate_validated_track(config, options.track_seed)
    env = _create_environment(
        project_config=config,
        level_id=options.level_id,
        track=track,
        backend=options.backend,
        render_mode="human" if options.render else None,
    )
    try:
        result = run_controller_episode(
            env,
            controller_directory,
            options.environment_seed,
            render=options.render,
        )
    finally:
        env.close()
    return _episode_summary(
        result,
        options=options,
        controller_directory=controller_directory,
        project_root=root,
    )


def main(argv: list[str] | None = None) -> None:
    """Parse CLI arguments, run one episode, and print one strict JSON summary."""

    parser = _build_parser()
    args = parser.parse_args(argv)
    options = SimulationOptions(
        controller_directory=args.controller_directory,
        track_seed=args.track_seed,
        environment_seed=args.environment_seed,
        backend=args.backend,
        level_id=args.level_id,
        render=args.render,
    )
    try:
        summary = _run_simulation(options)
    except (ConfigError, SimulationCliError) as error:
        parser.exit(2, f"sim: error: {error}\n")
    print(json.dumps(summary, allow_nan=False, sort_keys=True, separators=(",", ":")))


if __name__ == "__main__":
    main()
