"""Focused tests for the M4 single-episode simulation CLI."""

from __future__ import annotations

import json
from pathlib import Path
from types import MappingProxyType
from typing import Any

import pytest

from controller_learning.config import load_project_config
from controller_learning.control import EpisodeRunResult
from controller_learning.tracks import ValidationResult
from scripts import sim

PROJECT_ROOT = Path(__file__).parents[3]


def _options(**overrides: Any) -> sim.SimulationOptions:
    values: dict[str, Any] = {
        "controller_directory": Path("controllers/template"),
        "track_seed": None,
        "environment_seed": 0,
        "backend": "cpu_reference",
        "level_id": 1,
        "render": False,
    }
    values.update(overrides)
    return sim.SimulationOptions(**values)


def _result() -> EpisodeRunResult:
    return EpisodeRunResult(
        steps=3,
        total_reward=1.25,
        terminated=True,
        truncated=False,
        final_info=MappingProxyType(
            {
                "episode_seed": 0,
                "controller_seed": 17,
                "track_id": 42,
                "benchmark_version": "0.1",
                "termination_reason": 1,
                "lap_completed": True,
                "lap_time_s": 0.15,
            }
        ),
        debug_commands=(),
    )


def test_arguments_default_to_one_cpu_level1_template_episode() -> None:
    options = sim._parse_args([])

    assert options == _options()


@pytest.mark.parametrize(
    "arguments",
    [
        ["--track-seed", "-1"],
        ["--track-seed", str(2**32)],
        ["--env-seed", "not-an-integer"],
        ["--backend", "automatic"],
    ],
)
def test_arguments_reject_out_of_contract_choices(arguments: list[str]) -> None:
    with pytest.raises(SystemExit) as caught:
        sim._parse_args(arguments)
    assert caught.value.code == 2


def test_controller_directory_is_root_relative_and_requires_plugin_files(tmp_path: Path) -> None:
    plugin = tmp_path / "controllers" / "example"
    plugin.mkdir(parents=True)
    (plugin / "controller.py").write_text("# test\n", encoding="utf-8")

    with pytest.raises(sim.SimulationCliError, match=r"config\.toml"):
        sim._resolve_controller_directory(Path("controllers/example"), project_root=tmp_path)

    (plugin / "config.toml").write_text("[controller]\n", encoding="utf-8")
    assert (
        sim._resolve_controller_directory(Path("controllers/example"), project_root=tmp_path)
        == plugin
    )


def test_requested_track_seed_is_generated_validated_and_packed() -> None:
    config = load_project_config(PROJECT_ROOT)

    track = sim._generate_validated_track(config, 42)

    assert track.seed == 42
    assert track.generator_version == config.track.generator.generator_version
    assert track.capacity.max_track_points == config.track.representation.max_track_points


def test_level0_resolves_the_fixed_official_asset() -> None:
    config = load_project_config(PROJECT_ROOT)

    track, seed = sim._resolve_track(config, level_id=0, track_seed=None)

    assert seed == sim.UINT32_MAX
    assert track.seed == sim.UINT32_MAX
    assert track.generator_version == config.track.generator.generator_version


def test_level0_rejects_a_procedural_seed() -> None:
    config = load_project_config(PROJECT_ROOT)

    with pytest.raises(sim.SimulationCliError, match="one fixed Track"):
        sim._resolve_track(config, level_id=0, track_seed=42)


def test_invalid_track_is_rejected_without_trying_another_seed(monkeypatch) -> None:
    config = load_project_config(PROJECT_ROOT)
    generated_seeds: list[int] = []
    candidate = object()

    def fake_generate(seed, spec):
        generated_seeds.append(seed)
        return candidate

    monkeypatch.setattr(sim, "generate_track_candidate", fake_generate)
    monkeypatch.setattr(
        sim,
        "validate_track_candidate",
        lambda value, spec: ValidationResult(
            valid=False,
            reasons=("curvature_exceeded",),
            primary_reason="curvature_exceeded",
            metrics={},
        ),
    )
    monkeypatch.setattr(
        sim,
        "pack_track",
        lambda *args: pytest.fail("an invalid candidate must not be packed"),
    )

    with pytest.raises(
        sim.SimulationCliError,
        match="Track seed 9 failed geometry validation: curvature_exceeded",
    ):
        sim._generate_validated_track(config, 9)
    assert generated_seeds == [9]


def test_environment_is_closed_when_controller_execution_fails(
    monkeypatch,
    tmp_path: Path,
) -> None:
    plugin = tmp_path / "plugin"
    plugin.mkdir()
    (plugin / "controller.py").write_text("# test\n", encoding="utf-8")
    (plugin / "config.toml").write_text("[controller]\n", encoding="utf-8")
    created: list[Any] = []

    class FakeEnv:
        def __init__(self, **kwargs):
            self.kwargs = kwargs
            self.closed = False
            created.append(self)

        def close(self) -> None:
            self.closed = True

    monkeypatch.setattr(sim, "load_project_config", lambda root: object())
    monkeypatch.setattr(sim, "_resolve_track", lambda *args, **kwargs: (object(), 42))
    monkeypatch.setattr(sim, "_create_environment", FakeEnv)

    def fail_runner(env, controller_directory, reset_seed, *, render):
        assert controller_directory == plugin
        assert reset_seed == 23
        assert render is True
        raise RuntimeError("controller failed")

    monkeypatch.setattr(sim, "run_controller_episode", fail_runner)

    with pytest.raises(RuntimeError, match="controller failed"):
        sim._run_simulation(
            _options(
                controller_directory=plugin,
                environment_seed=23,
                render=True,
            ),
            project_root=tmp_path,
        )

    assert len(created) == 1
    assert created[0].kwargs["backend"] == "cpu_reference"
    assert created[0].kwargs["level_id"] == 1
    assert created[0].kwargs["render_mode"] == "human"
    assert created[0].closed is True


def test_success_returns_a_strict_json_safe_summary(monkeypatch, tmp_path: Path) -> None:
    plugin = tmp_path / "plugin"
    plugin.mkdir()
    (plugin / "controller.py").write_text("# test\n", encoding="utf-8")
    (plugin / "config.toml").write_text("[controller]\n", encoding="utf-8")

    class FakeEnv:
        def __init__(self, **kwargs):
            self.closed = False

        def close(self) -> None:
            self.closed = True

    monkeypatch.setattr(sim, "load_project_config", lambda root: object())
    monkeypatch.setattr(sim, "_resolve_track", lambda *args, **kwargs: (object(), 42))
    monkeypatch.setattr(sim, "_create_environment", FakeEnv)
    monkeypatch.setattr(sim, "run_controller_episode", lambda *args, **kwargs: _result())

    summary = sim._run_simulation(
        _options(controller_directory=plugin),
        project_root=tmp_path,
    )

    assert summary == {
        "backend": "cpu_reference",
        "benchmark_version": "0.1",
        "controller": "plugin",
        "environment_seed": 0,
        "lap_completed": True,
        "lap_time_s": 0.15,
        "level_id": 1,
        "steps": 3,
        "terminated": True,
        "termination_reason": 1,
        "total_reward": 1.25,
        "track_id": 42,
        "track_seed": 42,
        "truncated": False,
    }
    assert json.loads(json.dumps(summary, allow_nan=False)) == summary


def test_main_prints_only_the_compact_episode_json(monkeypatch, capsys) -> None:
    expected = {"steps": 3, "lap_completed": True}
    monkeypatch.setattr(sim, "_run_simulation", lambda options: expected)

    sim.main([])

    output = capsys.readouterr()
    assert output.err == ""
    assert output.out == '{"lap_completed":true,"steps":3}\n'
