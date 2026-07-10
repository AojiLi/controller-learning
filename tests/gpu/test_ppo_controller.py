"""GPU parity for selected Torch actor export into the ordinary PPO Controller."""

from __future__ import annotations

import hashlib
import importlib
import shutil
from pathlib import Path
from typing import Any

import jax.numpy as jnp
import numpy as np
import pytest

from controller_learning.config import load_project_config
from controller_learning.control import (
    build_public_controller_config,
    load_controller,
    load_controller_config,
)
from controller_learning.envs.car_racing import CarRacingEnv
from controller_learning.envs.observation import OBSERVATION_KEYS, action_space
from controller_learning.rl.configuration import load_ppo_config
from controller_learning.rl.features import (
    encode_local_track_features_jax,
    encode_local_track_features_numpy,
)
from controller_learning.tracks.generator import generate_track_candidate, pack_track
from controller_learning.tracks.specs import (
    generation_spec_from_project,
    track_capacity_from_project,
)

PROJECT_ROOT = Path(__file__).parents[2]
PPO_TEMPLATE = PROJECT_ROOT / "controllers" / "ppo"
pytestmark = pytest.mark.gpu


def _torch() -> Any:
    return importlib.import_module("torch")


def _policy_module() -> Any:
    return importlib.import_module("controller_learning.rl.policy")


def _export_module() -> Any:
    return importlib.import_module("controller_learning.rl.controller_export")


def _artifacts_module() -> Any:
    return importlib.import_module("controller_learning.rl.artifacts")


def _device() -> Any:
    torch = _torch()
    return torch.device("cuda", torch.cuda.current_device())


def _copy_template(destination: Path) -> Path:
    destination.mkdir()
    for name in ("controller.py", "config.toml", "README.md"):
        shutil.copy2(PPO_TEMPLATE / name, destination / name)
    return destination


def _checkpoint_identity(*, configuration_sha256: str) -> Any:
    artifacts = _artifacts_module()
    return artifacts.TrainingRunIdentity(
        run_id="gpu-selected-001",
        benchmark_version="0.1",
        source_revision="1" * 40,
        configuration_sha256=configuration_sha256,
        lock_sha256="3" * 64,
        train_manifest_sha256="4" * 64,
        train_cache_sha256="5" * 64,
        feature_schema_version=1,
        reward_schema_version="controller-learning.m7-public-reward.v1",
        environment_seed=7,
        policy_seed=11,
        minibatch_seed=13,
    )


def _save_and_load_checkpoint(root: Path, policy: Any, *, configuration_sha256: str) -> Any:
    torch = _torch()
    artifacts = _artifacts_module()
    identity = _checkpoint_identity(configuration_sha256=configuration_sha256)
    metadata = artifacts.TrainingCheckpointMetadata(
        run_identity=identity,
        update_index=1,
        vector_steps=1,
        valid_transitions=1024,
        elapsed_seconds=0.1,
    )
    continuation = artifacts.TrainingContinuationState(
        starting_update=1,
        num_envs=1024,
        environment_step_calls=1,
        raw_transitions=1024,
        valid_transitions=1024,
        dummy_reset_transitions=0,
        autoreset_slots=0,
        discarded_pending_reset_slots=0,
        terminal_events=0,
        terminated_events=0,
        truncated_events=0,
        episodes=0,
        successful_episodes=0,
        offtrack_episodes=0,
        invalid_action_episodes=0,
        timeout_episodes=0,
        successful_lap_time_sum_s=0.0,
        episode_length_sum_steps=0,
        cumulative_reward_sum=0.0,
        cumulative_compute_update_seconds=0.05,
        wall_elapsed_before_persistence_seconds=0.1,
    )
    generator = torch.Generator(device="cpu").manual_seed(17)
    artifacts.save_training_checkpoint(
        root,
        metadata=metadata,
        continuation_state=continuation,
        model_state_dict=policy.state_dict(),
        optimizer_state_dict={"state": {}},
        policy_rng_state=generator.get_state(),
        minibatch_rng_state=generator.get_state().clone(),
        keep_last=1,
        torch_module=torch,
    )
    return artifacts.load_published_training_checkpoint(
        root,
        expected_identity=identity,
        update_index=1,
        torch_module=torch,
    )


def _public_observation_corpus() -> tuple[list[dict[str, np.ndarray]], dict[str, Any]]:
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
    observations: list[dict[str, np.ndarray]] = []
    try:
        observation, info = env.reset(seed=811)
        observations.append(observation)
        actions = (
            (0.0, 1.0),
            (0.1, 1.5),
            (-0.1, 0.5),
            (0.2, 0.0),
            (-0.2, -0.5),
            (0.0, 2.0),
            (0.05, 1.0),
        )
        for action in actions:
            observation, _reward, terminated, truncated, _step_info = env.step(
                np.asarray(action, dtype=np.float32)
            )
            assert not terminated and not truncated
            observations.append(observation)
        return observations, dict(info)
    finally:
        env.close()


def test_torch_checkpoint_export_matches_plugin_on_real_public_observations(
    tmp_path: Path,
) -> None:
    torch = _torch()
    project = load_project_config(PROJECT_ROOT)
    ppo_config = load_ppo_config(PROJECT_ROOT / "configs" / "ppo.toml")
    physical = action_space(project)
    policy = _policy_module().PpoActorCritic(
        100,
        action_low=physical.low,
        action_high=physical.high,
        policy_seed=ppo_config.ppo.policy_seed,
        initial_log_std=ppo_config.ppo.initial_log_std,
        device=_device(),
    )
    plugin = _copy_template(tmp_path / "ppo")
    training_config_path = PROJECT_ROOT / "configs" / "ppo.toml"
    loaded_checkpoint = _save_and_load_checkpoint(
        tmp_path / "run",
        policy,
        configuration_sha256=hashlib.sha256(training_config_path.read_bytes()).hexdigest(),
    )
    wrong_config = tmp_path / "wrong-ppo.toml"
    wrong_config.write_bytes(training_config_path.read_bytes() + b"\n")
    with pytest.raises(
        _export_module().PpoControllerExportError,
        match="configuration SHA-256",
    ):
        _export_module().export_ppo_controller(
            plugin,
            loaded_checkpoint=loaded_checkpoint,
            training_config_path=wrong_config,
            public_policy_max_bytes=ppo_config.checkpoint.public_checkpoint_max_bytes,
        )
    result = _export_module().export_ppo_controller(
        plugin,
        loaded_checkpoint=loaded_checkpoint,
        training_config_path=training_config_path,
        public_policy_max_bytes=ppo_config.checkpoint.public_checkpoint_max_bytes,
    )
    observations, info = _public_observation_corpus()
    plugin_config = load_controller_config(plugin)
    public_config = build_public_controller_config(project, 1, plugin_config)
    controller_class = load_controller(plugin)
    controller = controller_class(observations[0], info, public_config)
    plugin_actions = np.stack(
        tuple(controller.compute_control(observation, info) for observation in observations)
    )
    features = np.stack(
        tuple(
            encode_local_track_features_numpy(
                observation,
                preview_points=ppo_config.observation.preview_points,
                preview_distance_m=ppo_config.observation.preview_distance_m,
                max_speed_mps=ppo_config.observation.max_speed_mps,
                control_dt_s=project.vehicle.simulation.control_dt_s,
                max_steering_angle_rad=project.vehicle.actuator.max_steering_angle_rad,
            )
            for observation in observations
        )
    )
    batched_observation = {
        name: jnp.asarray(np.stack(tuple(observation[name] for observation in observations)))
        for name in OBSERVATION_KEYS
    }
    selection_features = np.asarray(
        encode_local_track_features_jax(
            batched_observation,
            preview_points=ppo_config.observation.preview_points,
            preview_distance_m=ppo_config.observation.preview_distance_m,
            max_speed_mps=ppo_config.observation.max_speed_mps,
            control_dt_s=project.vehicle.simulation.control_dt_s,
            max_steering_angle_rad=project.vehicle.actuator.max_steering_angle_rad,
        )
    )
    # The deployment encoder recomputes public geometry in float64 before returning float32,
    # while the vector selector uses XLA float32 reductions.  A 25,700-observation official-Train
    # audit measured maxima of 3.204e-6 in features and 3.565e-5 across all eight frozen actors.
    np.testing.assert_allclose(features, selection_features, rtol=0.0, atol=5.0e-6)
    with torch.no_grad():
        expected = policy.deterministic(
            torch.as_tensor(np.array(selection_features, copy=True), device=_device())
        ).action
    expected_actions = expected.cpu().numpy()

    np.testing.assert_allclose(
        plugin_actions,
        expected_actions,
        rtol=0.0,
        atol=5.0e-5,
    )
    assert np.max(np.abs(plugin_actions - expected_actions)) <= 5.0e-5
    assert plugin_actions.shape == (len(observations), 2)
    assert plugin_actions.dtype == np.float32
    assert np.isfinite(plugin_actions).all()
    assert np.all(plugin_actions >= physical.low)
    assert np.all(plugin_actions <= physical.high)
    assert result.checkpoint.run_id == "gpu-selected-001"
    assert result.checkpoint.update_index == 1
    assert result.checkpoint.checkpoint_sha256 == loaded_checkpoint.record.sha256
    assert plugin_config["policy"]["sha256"] == result.policy.sha256
    assert not (plugin / "optimizer.pt").exists()
    assert not (plugin / "checkpoint.pt").exists()


def test_gpu_test_module_keeps_torch_import_lazy_for_cpu_collection() -> None:
    source = (PROJECT_ROOT / "tests" / "gpu" / "test_ppo_controller.py").read_text(encoding="utf-8")
    imports_and_helpers = source.split(
        "def test_gpu_test_module_keeps_torch_import_lazy_for_cpu_collection",
        maxsplit=1,
    )[0]
    assert "\nimport torch" not in imports_and_helpers
    assert "\nfrom torch" not in imports_and_helpers
