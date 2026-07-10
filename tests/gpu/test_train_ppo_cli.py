"""GPU construction test for the exact formal M7 PPO entrypoint stack."""

from __future__ import annotations

import importlib
from pathlib import Path

import pytest

from controller_learning.config import load_project_config
from controller_learning.rl.artifacts import (
    TrainingCheckpointMetadata,
    TrainingContinuationState,
    TrainingRunIdentity,
    load_training_checkpoint,
    save_training_checkpoint,
)
from controller_learning.rl.assets import load_verified_train_pool
from controller_learning.rl.configuration import load_ppo_config
from controller_learning.rl.features import LOCAL_TRACK_FEATURE_DIM, LocalTrackObservationVecEnv
from controller_learning.rl.reward import PublicRewardShapingVecEnv
from controller_learning.rl.torch_bridge import JaxToTorchVecEnv
from scripts import train_ppo

PROJECT_ROOT = Path(__file__).parents[2]
pytestmark = pytest.mark.gpu


def test_formal_factory_uses_verified_train_pool_and_exact_1024_world_public_stack(
    tmp_path: Path,
) -> None:
    torch = importlib.import_module("torch")
    project = load_project_config(PROJECT_ROOT)
    config = load_ppo_config(PROJECT_ROOT / "configs/ppo.toml")
    guard = train_ppo.OfficialTrainAssetAccessGuard(
        official_track_root=PROJECT_ROOT / "controller_learning/assets/tracks",
        train_manifest=PROJECT_ROOT / "controller_learning/assets/tracks/v0.1/train.json",
        track_cache_root=PROJECT_ROOT / ".track-cache",
        train_cache=PROJECT_ROOT / config.environment.train_cache,
    )
    guard.install()
    verified = load_verified_train_pool(
        project,
        train_cache_path=PROJECT_ROOT / config.environment.train_cache,
    )
    access = guard.evidence(loader_succeeded=True)
    assert access["opened_splits"] == ["train"]
    assert access["opened_path_categories"] == [
        "configured_train_cache",
        "official_train_manifest",
    ]
    assert access["validation_opened"] is False
    assert access["test_opened"] is False

    stack = train_ppo.build_official_training_stack(project, config, verified)
    try:
        base = stack.base_environment
        bridge = stack.environment
        featured = bridge.env
        shaped = featured.env

        assert stack.wrapper_order == train_ppo.FORMAL_WRAPPER_ORDER
        assert isinstance(bridge, JaxToTorchVecEnv)
        assert isinstance(featured, LocalTrackObservationVecEnv)
        assert isinstance(shaped, PublicRewardShapingVecEnv)
        assert shaped.env is base
        assert base.backend == "mjx_warp"
        assert base.level_id == 1
        assert base.num_envs == config.environment.num_envs == 1024
        assert base.track_pool is verified.pool
        assert base.track_pool.split == "train"
        assert base.track_pool.size == 10_000
        assert stack.policy.observation_dim == LOCAL_TRACK_FEATURE_DIM
        assert stack.collector.env is bridge
        assert stack.collector.policy is stack.policy
        assert stack.collector.rollout_steps == config.rollout.steps_per_update
        assert stack.updater.policy is stack.policy
        assert stack.updater.config is config.ppo

        identity = TrainingRunIdentity(
            run_id="gpu-roundtrip",
            benchmark_version="0.1",
            source_revision="1" * 40,
            configuration_sha256="2" * 64,
            lock_sha256="3" * 64,
            train_manifest_sha256=verified.evidence.manifest_sha256,
            train_cache_sha256=verified.evidence.cache_file_sha256,
            feature_schema_version=1,
            reward_schema_version="controller-learning.m7-public-reward.v1",
            environment_seed=7,
            policy_seed=11,
            minibatch_seed=13,
        )
        metadata = TrainingCheckpointMetadata(
            run_identity=identity,
            update_index=1,
            vector_steps=128,
            valid_transitions=131_072,
            elapsed_seconds=0.2,
        )
        continuation = TrainingContinuationState(
            starting_update=1,
            num_envs=1024,
            environment_step_calls=128,
            raw_transitions=131_072,
            valid_transitions=131_072,
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
            cumulative_compute_update_seconds=0.1,
            wall_elapsed_before_persistence_seconds=0.2,
        )
        policy_generator = torch.Generator(device=stack.policy.device).manual_seed(11)
        minibatch_generator = torch.Generator(device=stack.policy.device).manual_seed(13)
        original_log_std = stack.policy.log_std.detach().clone()
        save_training_checkpoint(
            tmp_path,
            metadata=metadata,
            continuation_state=continuation,
            model_state_dict=stack.policy.state_dict(),
            optimizer_state_dict=stack.updater.optimizer.state_dict(),
            policy_rng_state=policy_generator.get_state(),
            minibatch_rng_state=minibatch_generator.get_state(),
            keep_last=1,
            torch_module=torch,
        )
        loaded = load_training_checkpoint(
            tmp_path,
            expected_identity=identity,
            torch_module=torch,
        )
        with torch.no_grad():
            stack.policy.log_std.add_(1.0)
        train_ppo._restore_model_and_optimizer(stack, loaded)
        resume = train_ppo._trainer_resume_state(loaded)

        torch.testing.assert_close(stack.policy.log_std, original_log_std)
        assert resume.starting_update == 1
        assert resume.counts.raw_transitions == 131_072
        assert resume.discarded_pending_reset_slots == 0
        assert resume.episodes.episodes == 0
        assert resume.cumulative_compute_update_seconds == 0.1
        assert resume.wall_elapsed_before_persistence_seconds == 0.2
        assert resume.policy_rng_state.device.type == "cpu"
        assert resume.minibatch_rng_state.device.type == "cpu"
    finally:
        stack.close()

    assert stack._closed is True
    assert base._closed is True
