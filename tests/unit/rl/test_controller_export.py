"""CPU tests for PPO Controller finalization and Torch-free plugin loading."""

from __future__ import annotations

import hashlib
import json
import shutil
import tomllib
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from controller_learning.config import load_project_config
from controller_learning.control import (
    build_public_controller_config,
    load_controller,
    load_controller_config,
    run_controller_episode,
)
from controller_learning.envs.car_racing import CarRacingEnv
from controller_learning.rl.configuration import load_ppo_config
from controller_learning.rl.controller_export import (
    PpoControllerExportError,
    PpoControllerNotFinalizedError,
    SelectedCheckpointIdentity,
    export_numpy_actor_controller,
    export_ppo_controller,
    load_ppo_controller_runtime,
)
from controller_learning.rl.features import encode_local_track_features_numpy
from controller_learning.rl.numpy_actor import NumpyDeterministicActor
from controller_learning.tracks.generator import generate_track_candidate, pack_track
from controller_learning.tracks.specs import (
    generation_spec_from_project,
    track_capacity_from_project,
)

PROJECT_ROOT = Path(__file__).parents[3]
PPO_TEMPLATE = PROJECT_ROOT / "controllers" / "ppo"


def _copy_template(destination: Path) -> Path:
    destination.mkdir()
    for name in ("controller.py", "config.toml", "README.md"):
        shutil.copy2(PPO_TEMPLATE / name, destination / name)
    return destination


def _actor(seed: int = 701) -> NumpyDeterministicActor:
    generator = np.random.default_rng(seed)

    def values(shape: tuple[int, ...]) -> np.ndarray:
        return generator.normal(0.0, 0.04, size=shape).astype(np.float32)

    return NumpyDeterministicActor(
        hidden_0_weight=values((128, 100)),
        hidden_0_bias=values((128,)),
        hidden_1_weight=values((128, 128)),
        hidden_1_bias=values((128,)),
        actor_weight=values((2, 128)),
        actor_bias=values((2,)),
        action_low=np.asarray((-0.6, -8.0), dtype=np.float32),
        action_high=np.asarray((0.6, 4.0), dtype=np.float32),
    )


def _checkpoint() -> SelectedCheckpointIdentity:
    return SelectedCheckpointIdentity(
        run_id="selected-run-001",
        update_index=80,
        vector_steps=10_240,
        valid_transitions=10_000_000,
        checkpoint_sha256="a" * 64,
        source_revision="b" * 40,
        training_configuration_sha256="c" * 64,
    )


def _observation_and_info() -> tuple[dict[str, np.ndarray], dict[str, Any]]:
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
    )
    try:
        observation, info = env.reset(seed=709)
        return observation, dict(info)
    finally:
        env.close()


def _finalized_plugin(directory: Path) -> tuple[Path, NumpyDeterministicActor, Any]:
    plugin = _copy_template(directory)
    actor = _actor()
    observation = load_ppo_config(PROJECT_ROOT / "configs" / "ppo.toml").observation
    result = export_numpy_actor_controller(
        plugin,
        actor=actor,
        checkpoint=_checkpoint(),
        observation_config=observation,
        public_policy_max_bytes=5 * 1024 * 1024,
    )
    return plugin, actor, result


def _public_config(plugin: Path) -> Any:
    project = load_project_config(PROJECT_ROOT)
    return build_public_controller_config(project, 1, load_controller_config(plugin))


def test_transaction_staging_preserves_exported_endpoint_bytes(tmp_path: Path) -> None:
    direct_plugin = _copy_template(tmp_path / "direct")
    staged_plugin = _copy_template(tmp_path / "staged")
    staging = tmp_path / "transaction" / "staging"
    staging.mkdir(parents=True)
    actor = _actor()
    observation = load_ppo_config(PROJECT_ROOT / "configs" / "ppo.toml").observation

    direct = export_numpy_actor_controller(
        direct_plugin,
        actor=actor,
        checkpoint=_checkpoint(),
        observation_config=observation,
        public_policy_max_bytes=5 * 1024 * 1024,
    )
    staged = export_numpy_actor_controller(
        staged_plugin,
        actor=actor,
        checkpoint=_checkpoint(),
        observation_config=observation,
        public_policy_max_bytes=5 * 1024 * 1024,
        staging_directory=staging,
    )

    assert direct.policy == staged.policy
    assert direct.metadata_sha256 == staged.metadata_sha256
    assert direct.metadata_size_bytes == staged.metadata_size_bytes
    assert direct.config_sha256 == staged.config_sha256
    assert direct.config_size_bytes == staged.config_size_bytes
    for name in ("policy.npz", "metadata.json", "config.toml"):
        assert (direct_plugin / name).read_bytes() == (staged_plugin / name).read_bytes()
    assert tuple(staging.iterdir()) == ()
    assert not any(
        path.name.endswith((".tmp", ".recovery"))
        for plugin in (direct_plugin, staged_plugin)
        for path in plugin.iterdir()
    )


class _OneStepPublicEnv:
    def __init__(self, observation: dict[str, np.ndarray], info: dict[str, Any]) -> None:
        self.project_config = load_project_config(PROJECT_ROOT)
        self.level_id = 1
        self.observation = observation
        self.info = info
        self.actions: list[np.ndarray] = []

    @property
    def unwrapped(self) -> _OneStepPublicEnv:
        return self

    def reset(self, *, seed: int, **_kwargs: Any):
        del seed
        return self.observation, dict(self.info)

    def step(self, action: np.ndarray):
        self.actions.append(np.array(action, copy=True))
        final = dict(self.info)
        final.update(termination_reason=1, lap_completed=True, lap_time_s=0.05)
        return self.observation, 1.0, True, False, final


def test_checked_in_ppo_template_loads_but_refuses_construction_without_weights() -> None:
    assert not (PPO_TEMPLATE / "policy.npz").exists()
    config = load_controller_config(PPO_TEMPLATE)
    controller_class = load_controller(PPO_TEMPLATE)
    observation, info = _observation_and_info()
    public = build_public_controller_config(load_project_config(PROJECT_ROOT), 1, config)

    assert controller_class.__name__ == "PpoController"
    assert config["finalized"] is False
    with pytest.raises(PpoControllerNotFinalizedError, match="not finalized"):
        controller_class(observation, info, public)


def test_exported_plugin_matches_direct_numpy_actor_and_runner_uses_fresh_instances(
    tmp_path: Path,
) -> None:
    plugin, actor, export = _finalized_plugin(tmp_path / "ppo")
    observation, info = _observation_and_info()
    public = _public_config(plugin)
    first_class = load_controller(plugin)
    second_class = load_controller(plugin)
    first = first_class(observation, info, public)
    second = second_class(observation, info, public)

    first_action = first.compute_control(observation, info)
    second_action = second.compute_control(observation, info)
    project = load_project_config(PROJECT_ROOT)
    feature_config = load_ppo_config(PROJECT_ROOT / "configs" / "ppo.toml").observation
    features = encode_local_track_features_numpy(
        observation,
        preview_points=feature_config.preview_points,
        preview_distance_m=feature_config.preview_distance_m,
        max_speed_mps=feature_config.max_speed_mps,
        control_dt_s=project.vehicle.simulation.control_dt_s,
        max_steering_angle_rad=project.vehicle.actuator.max_steering_angle_rad,
    )
    expected = actor(features)

    assert first is not second
    assert first._runtime.actor is not second._runtime.actor
    np.testing.assert_array_equal(first_action, second_action)
    np.testing.assert_array_equal(first_action, expected)
    assert first_action.dtype == np.float32
    assert np.isfinite(first_action).all()
    assert np.all(first_action >= actor.action_low)
    assert np.all(first_action <= actor.action_high)
    config = load_controller_config(plugin)
    assert config["finalized"] is True
    assert config["policy"]["sha256"] == export.policy.sha256
    assert config["policy"]["size_bytes"] == export.policy.size_bytes
    metadata = json.loads((plugin / "metadata.json").read_bytes())
    assert metadata["inference_only"] == {
        "contains_environment_state": False,
        "contains_optimizer_state": False,
        "contains_value_network": False,
        "runtime": "numpy",
    }

    first_env = _OneStepPublicEnv(observation, info)
    second_env = _OneStepPublicEnv(observation, info)
    run_controller_episode(first_env, plugin, reset_seed=1)
    run_controller_episode(second_env, plugin, reset_seed=2)
    np.testing.assert_array_equal(first_env.actions[0], second_env.actions[0])


def test_runtime_rejects_missing_tampered_or_misbound_local_artifacts(tmp_path: Path) -> None:
    plugin, _actor_value, _export = _finalized_plugin(tmp_path / "ppo")
    public = _public_config(plugin)
    policy_bytes = (plugin / "policy.npz").read_bytes()
    metadata_bytes = (plugin / "metadata.json").read_bytes()

    (plugin / "policy.npz").unlink()
    with pytest.raises(PpoControllerExportError, match=r"regular local file"):
        load_ppo_controller_runtime(public, plugin_directory=plugin)
    (plugin / "policy.npz").write_bytes(policy_bytes)

    tampered = bytearray(policy_bytes)
    tampered[len(tampered) // 2] ^= 1
    (plugin / "policy.npz").write_bytes(tampered)
    with pytest.raises(PpoControllerExportError, match=r"strict local verification"):
        load_ppo_controller_runtime(public, plugin_directory=plugin)
    (plugin / "policy.npz").write_bytes(policy_bytes)

    (plugin / "metadata.json").write_bytes(metadata_bytes + b" ")
    with pytest.raises(PpoControllerExportError, match=r"size differs|SHA-256"):
        load_ppo_controller_runtime(public, plugin_directory=plugin)
    (plugin / "metadata.json").write_bytes(metadata_bytes)

    raw = tomllib.loads((plugin / "config.toml").read_text(encoding="utf-8"))
    wrong_size = {**raw, "policy": {**raw["policy"], "size_bytes": 1}}
    with pytest.raises(PpoControllerExportError, match=r"size|safe local limit"):
        load_ppo_controller_runtime(
            {"controller": wrong_size},
            plugin_directory=plugin,
        )
    wrong_schema = {**raw, "policy": {**raw["policy"], "schema_version": 999}}
    with pytest.raises(ValueError, match="schema_version"):
        load_ppo_controller_runtime(
            {"controller": wrong_schema},
            plugin_directory=plugin,
        )
    unsafe_path = {**raw, "policy": {**raw["policy"], "file": "../policy.npz"}}
    with pytest.raises(ValueError, match="local filename"):
        load_ppo_controller_runtime(
            {"controller": unsafe_path},
            plugin_directory=plugin,
        )


def test_export_is_one_way_and_controller_runtime_sources_do_not_import_torch(
    tmp_path: Path,
) -> None:
    plugin, actor, _result = _finalized_plugin(tmp_path / "ppo")
    observation = load_ppo_config(PROJECT_ROOT / "configs" / "ppo.toml").observation
    with pytest.raises(PpoControllerExportError, match="finalized"):
        export_numpy_actor_controller(
            plugin,
            actor=actor,
            checkpoint=_checkpoint(),
            observation_config=observation,
            public_policy_max_bytes=5 * 1024 * 1024,
        )

    combined = "\n".join(
        path.read_text(encoding="utf-8")
        for path in (
            PROJECT_ROOT / "controllers" / "ppo" / "controller.py",
            PROJECT_ROOT / "controller_learning" / "rl" / "controller_export.py",
            PROJECT_ROOT / "controller_learning" / "rl" / "numpy_actor.py",
        )
    )
    assert "import torch" not in combined
    assert "from torch" not in combined
    for forbidden in (
        "controller_learning.envs",
        "race_core",
        "TrackBatch",
        "import mujoco",
        "import warp",
    ):
        assert forbidden not in combined


def test_formal_export_requires_one_compound_verified_checkpoint(tmp_path: Path) -> None:
    plugin = _copy_template(tmp_path / "ppo")

    with pytest.raises(TypeError, match="verified LoadedCandidateCheckpoint"):
        export_ppo_controller(
            plugin,
            loaded_checkpoint=object(),
            training_config_path=PROJECT_ROOT / "configs" / "ppo.toml",
            public_policy_max_bytes=5 * 1024 * 1024,
        )


@pytest.mark.parametrize(
    "table",
    [None, "policy", "metadata", "feature"],
)
@pytest.mark.parametrize("schema_version", [True, 1.0])
def test_runtime_rejects_noninteger_config_schema_aliases(
    tmp_path: Path,
    table: str | None,
    schema_version: object,
) -> None:
    plugin, _actor_value, _export = _finalized_plugin(tmp_path / "ppo")
    raw = tomllib.loads((plugin / "config.toml").read_text(encoding="utf-8"))
    target = raw if table is None else raw[table]
    target["schema_version"] = schema_version

    with pytest.raises(ValueError, match="integer schema version"):
        load_ppo_controller_runtime({"controller": raw}, plugin_directory=plugin)


@pytest.mark.parametrize("field", ["schema_version", "contains_optimizer_state"])
def test_runtime_rejects_metadata_scalar_type_aliases(tmp_path: Path, field: str) -> None:
    plugin, _actor_value, _export = _finalized_plugin(tmp_path / "ppo")
    raw = tomllib.loads((plugin / "config.toml").read_text(encoding="utf-8"))
    metadata_path = plugin / "metadata.json"
    document = json.loads(metadata_path.read_bytes())
    if field == "schema_version":
        document["schema_version"] = True
    else:
        document["inference_only"][field] = 0
    content = (
        json.dumps(
            document,
            allow_nan=False,
            ensure_ascii=True,
            separators=(",", ":"),
            sort_keys=True,
        )
        + "\n"
    ).encode("utf-8")
    metadata_path.write_bytes(content)
    raw["metadata"]["sha256"] = hashlib.sha256(content).hexdigest()
    raw["metadata"]["size_bytes"] = len(content)

    with pytest.raises((ValueError, PpoControllerExportError), match=r"schema version|boolean"):
        load_ppo_controller_runtime({"controller": raw}, plugin_directory=plugin)
