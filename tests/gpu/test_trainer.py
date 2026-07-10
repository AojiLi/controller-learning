"""GPU integration tests for seeded PPO training orchestration."""

from __future__ import annotations

import csv
import dataclasses
import importlib
from pathlib import Path
from typing import Any

import pytest

from controller_learning.rl.configuration import PpoTrainingConfig, load_ppo_config

PROJECT_ROOT = Path(__file__).parents[2]
OBSERVATION_DIM = 100
pytestmark = pytest.mark.gpu


def _torch() -> Any:
    return importlib.import_module("torch")


def _collector_module() -> Any:
    return importlib.import_module("controller_learning.rl.collector")


def _policy_module() -> Any:
    return importlib.import_module("controller_learning.rl.policy")


def _ppo_module() -> Any:
    return importlib.import_module("controller_learning.rl.ppo")


def _trainer_module() -> Any:
    return importlib.import_module("controller_learning.rl.trainer")


def _device() -> Any:
    torch = _torch()
    return torch.device("cuda", torch.cuda.current_device())


def _small_training_config(*, tensorboard_enabled: bool = True) -> PpoTrainingConfig:
    base = load_ppo_config(PROJECT_ROOT / "configs" / "ppo.toml")
    return dataclasses.replace(
        base,
        rollout=dataclasses.replace(
            base.rollout,
            steps_per_update=2,
            total_vector_steps=6,
        ),
        ppo=dataclasses.replace(
            base.ppo,
            learning_rate=1.0e-4,
            num_minibatches=4,
            update_epochs=1,
            target_kl=1.0e6,
        ),
        logging=dataclasses.replace(
            base.logging,
            log_interval_updates=2,
            csv_flush_interval_updates=2,
            tensorboard_enabled=tensorboard_enabled,
            memory_sample_interval_updates=2,
        ),
        checkpoint=dataclasses.replace(
            base.checkpoint,
            interval_updates=2,
            keep_last=2,
            save_optimizer_state=True,
        ),
    )


def _policy(config: PpoTrainingConfig) -> Any:
    torch = _torch()
    return _policy_module().PpoActorCritic(
        OBSERVATION_DIM,
        action_low=torch.tensor((-0.6, -8.0), dtype=torch.float32),
        action_high=torch.tensor((0.6, 4.0), dtype=torch.float32),
        policy_seed=config.ppo.policy_seed,
        initial_log_std=config.ppo.initial_log_std,
        hidden_sizes=config.ppo.hidden_sizes,
        device=_device(),
    )


class _SeededCollectorStub:
    """Small synthetic collector retaining the formal 1,024-world tensor width."""

    def __init__(self, policy: Any, config: PpoTrainingConfig) -> None:
        self.policy = policy
        self.rollout_steps = config.rollout.steps_per_update
        self.num_envs = config.environment.num_envs
        self.initialized_seed: int | None = None
        self.collect_calls = 0
        self.observed_policy_rng_states: list[Any] = []

    def initialize(self, *, seed: int) -> Any:
        torch = _torch()
        self.initialized_seed = seed
        device = self.policy.device
        world = torch.arange(self.num_envs, dtype=torch.int64, device=device)
        self._episode_seed = world + 10_000
        self._controller_seed = world + 20_000
        self._track_id = world.clone()
        return _collector_module().CollectorState(
            observation=torch.zeros(
                (self.num_envs, OBSERVATION_DIM),
                dtype=torch.float32,
                device=self.policy.device,
            ),
            pending_reset=torch.zeros(
                self.num_envs,
                dtype=torch.bool,
                device=self.policy.device,
            ),
        )

    def collect(self, state: Any, *, generator: Any) -> Any:
        torch = _torch()
        collector_module = _collector_module()
        self.collect_calls += 1
        self.observed_policy_rng_states.append(generator.get_state().clone())
        update_index = self.collect_calls
        device = self.policy.device
        shape = (self.rollout_steps, self.num_envs)

        flat = torch.arange(
            self.rollout_steps * self.num_envs * OBSERVATION_DIM,
            dtype=torch.float32,
            device=device,
        )
        observations = (flat.remainder(101.0) / 101.0).reshape(
            *shape, OBSERVATION_DIM
        ) + update_index * 0.01
        next_observation = observations[-1] + 0.005
        pre_tanh_actions = torch.empty((*shape, 2), dtype=torch.float32, device=device)
        actions = torch.empty_like(pre_tanh_actions)
        old_log_prob = torch.empty(shape, dtype=torch.float32, device=device)
        values = torch.empty((self.rollout_steps + 1, self.num_envs), device=device)
        with torch.no_grad():
            for step in range(self.rollout_steps):
                sample = self.policy.sample(observations[step], generator=generator)
                pre_tanh_actions[step].copy_(sample.pre_tanh)
                actions[step].copy_(sample.action)
                old_log_prob[step].copy_(sample.log_prob)
                values[step].copy_(sample.value)
            values[-1].copy_(self.policy.value(next_observation))

        terminated = torch.zeros(shape, dtype=torch.bool, device=device)
        truncated = torch.zeros_like(terminated)
        reason = torch.zeros(shape, dtype=torch.int32, device=device)
        lap_completed = torch.zeros_like(terminated)
        lap_time_s = torch.zeros(shape, dtype=torch.float32, device=device)
        event_world = update_index - 1
        if update_index == 1:
            terminated[-1, event_world] = True
            reason[-1, event_world] = 1
            lap_completed[-1, event_world] = True
            lap_time_s[-1, event_world] = 12.5
        elif update_index == 2:
            terminated[-1, event_world] = True
            reason[-1, event_world] = 2
        else:
            truncated[-1, event_world] = True
            reason[-1, event_world] = 4

        masks = collector_module.build_torch_rollout_transition_masks(
            state.pending_reset,
            terminated,
            truncated,
        )
        rewards = torch.full(shape, 0.25, dtype=torch.float32, device=device)
        rewards.masked_fill_(masks.reset_only, 0.0)
        reset_count = int(masks.reset_only.sum(dtype=torch.int64).to(device="cpu").tolist())
        terminated_count = int(terminated.sum(dtype=torch.int64).to(device="cpu").tolist())
        truncated_count = int(truncated.sum(dtype=torch.int64).to(device="cpu").tolist())
        raw_count = self.rollout_steps * self.num_envs
        counts = importlib.import_module("controller_learning.rl.rollout").TransitionCounts(
            num_envs=self.num_envs,
            environment_step_calls=self.rollout_steps,
            raw_transitions=raw_count,
            valid_transitions=raw_count - reset_count,
            dummy_reset_transitions=reset_count,
            autoreset_slots=reset_count,
            terminal_events=terminated_count + truncated_count,
            terminated_events=terminated_count,
            truncated_events=truncated_count,
        )
        final_state = collector_module.CollectorState(
            observation=next_observation.clone(),
            pending_reset=masks.final_pending_reset.clone(),
        )
        identity_rows: list[tuple[Any, Any, Any]] = []
        for step in range(self.rollout_steps):
            reset = masks.reset_only[step].to(dtype=torch.int64)
            self._episode_seed += reset * 100_000
            self._controller_seed += reset * 200_000
            self._track_id += reset
            identity_rows.append(
                (
                    self._episode_seed.to(dtype=torch.uint32).clone(),
                    self._controller_seed.to(dtype=torch.uint32).clone(),
                    self._track_id.to(dtype=torch.uint32).clone(),
                )
            )
        episode_seed = torch.stack(tuple(row[0] for row in identity_rows))
        controller_seed = torch.stack(tuple(row[1] for row in identity_rows))
        track_id = torch.stack(tuple(row[2] for row in identity_rows))
        return collector_module.CollectedRollout(
            observations=observations,
            pre_tanh_actions=pre_tanh_actions,
            actions=actions,
            old_log_prob=old_log_prob,
            values=values,
            rewards=rewards,
            terminated=terminated,
            truncated=truncated,
            termination_reason=reason,
            lap_completed=lap_completed,
            lap_time_s=lap_time_s,
            episode_seed=episode_seed,
            controller_seed=controller_seed,
            track_id=track_id,
            valid_transition=masks.valid_transition,
            reset_only=masks.reset_only,
            initial_pending_reset=masks.initial_pending_reset,
            final_state=final_state,
            counts=counts,
        )


class _Clock:
    def __init__(self, *values: float) -> None:
        self._values = iter(values)

    def __call__(self) -> float:
        return next(self._values)


class _SummaryWriter:
    def __init__(self, path: Path, *, close_error: BaseException | None = None) -> None:
        self.path = path
        self.close_error = close_error
        self.scalars: list[tuple[str, float | int, int]] = []
        self.flush_count = 0
        self.close_count = 0

    def add_scalar(self, tag: str, scalar_value: float | int, global_step: int) -> None:
        self.scalars.append((tag, scalar_value, global_step))

    def flush(self) -> None:
        self.flush_count += 1

    def close(self) -> None:
        self.close_count += 1
        if self.close_error is not None:
            raise self.close_error


@dataclasses.dataclass(frozen=True)
class _Run:
    summary: Any
    policy: Any
    collector: _SeededCollectorStub
    checkpoints: tuple[Any, ...]
    writer: _SummaryWriter


def _run_training(config: PpoTrainingConfig, directory: Path) -> _Run:
    trainer = _trainer_module()
    policy = _policy(config)
    collector = _SeededCollectorStub(policy, config)
    updater = _ppo_module().PpoUpdater(policy, config.ppo)
    checkpoints: list[Any] = []
    writers: list[_SummaryWriter] = []

    def writer_factory(path: Path) -> _SummaryWriter:
        writer = _SummaryWriter(path)
        writers.append(writer)
        return writer

    summary = trainer.train_ppo(
        collector,
        updater,
        config,
        run_directory=directory,
        checkpoint_callback=checkpoints.append,
        clock=_Clock(0.0, 0.0, 1.0, 1.0, 3.0, 3.0, 6.0, 8.0),
        summary_writer_factory=writer_factory,
        memory_sampler=lambda _device: trainer.TorchCudaMemoryMetrics(100, 200, 300),
    )
    return _Run(summary, policy, collector, tuple(checkpoints), writers[0])


def test_training_loop_is_seed_reproducible_accounts_budget_and_persists_records(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    torch = _torch()
    trainer = _trainer_module()
    config = _small_training_config()
    opened_paths: list[Path] = []
    original_open = Path.open

    def audited_open(path: Path, *args: Any, **kwargs: Any) -> Any:
        opened_paths.append(path)
        return original_open(path, *args, **kwargs)

    fsync_calls: list[int] = []
    monkeypatch.setattr(Path, "open", audited_open)
    monkeypatch.setattr(trainer.os, "fsync", fsync_calls.append)
    first = _run_training(config, tmp_path / "first")
    first_fsync_count = len(fsync_calls)
    second = _run_training(config, tmp_path / "second")

    assert first.collector.initialized_seed == config.environment.environment_seed
    assert first.summary == dataclasses.replace(
        second.summary,
        metrics_path=first.summary.metrics_path,
    )
    for name, first_tensor in first.policy.state_dict().items():
        torch.testing.assert_close(
            first_tensor,
            second.policy.state_dict()[name],
            rtol=0.0,
            atol=0.0,
        )

    summary = first.summary
    assert summary.configured_budget_completed
    assert summary.completed_updates == summary.configured_updates == 3
    assert summary.vector_steps == config.rollout.total_vector_steps == 6
    assert summary.counts.raw_transitions == config.world_step_slot_budget == 6_144
    assert summary.counts.valid_transitions == 6_142
    assert summary.counts.dummy_reset_transitions == 2
    assert summary.counts.autoreset_slots == 2
    assert summary.counts.terminal_events == 3
    assert summary.counts.terminated_events == 2
    assert summary.counts.truncated_events == 1
    assert summary.discarded_pending_reset_slots == 0
    assert summary.episodes.successful_episodes == 1
    assert summary.episodes.offtrack_episodes == 1
    assert summary.episodes.invalid_action_episodes == 0
    assert summary.episodes.timeout_episodes == 1
    assert summary.episodes.success_rate == pytest.approx(1.0 / 3.0)
    assert summary.episodes.mean_successful_lap_time_s == 12.5
    assert summary.episodes.episode_length_sum_steps == 12
    assert summary.episodes.mean_episode_length_steps == 4.0
    assert [record.rollout_episodes.episode_length_sum_steps for record in summary.records] == [
        2,
        4,
        6,
    ]
    assert summary.cumulative_reward_sum == pytest.approx(0.25 * 6_142)
    assert summary.compute_update_seconds == 6.0
    assert summary.compute_valid_transitions_per_second == pytest.approx(6_142 / 6.0)
    assert summary.end_to_end_elapsed_seconds == 8.0
    assert summary.end_to_end_valid_transitions_per_second == pytest.approx(6_142 / 8.0)
    assert [record.optimization.learning_rate for record in summary.records] == pytest.approx(
        (
            config.ppo.learning_rate,
            config.ppo.learning_rate * 2.0 / 3.0,
            config.ppo.learning_rate / 3.0,
        )
    )
    assert [record.compute_update_wall_seconds for record in summary.records] == [1.0, 2.0, 3.0]
    assert summary.records[0].torch_cuda_memory is None
    assert summary.records[1].torch_cuda_memory == trainer.TorchCudaMemoryMetrics(100, 200, 300)
    assert summary.records[2].torch_cuda_memory == trainer.TorchCudaMemoryMetrics(100, 200, 300)

    assert [request.update_index for request in first.checkpoints] == [2, 3]
    assert [(request.is_scheduled, request.is_final) for request in first.checkpoints] == [
        (True, False),
        (False, True),
    ]
    assert all(request.optimizer_state_dict for request in first.checkpoints)
    assert not torch.equal(
        first.checkpoints[-1].policy_rng_state,
        first.checkpoints[-1].minibatch_rng_state,
    )
    assert any(
        not torch.equal(
            first.checkpoints[0].model_state_dict[name],
            first.checkpoints[1].model_state_dict[name],
        )
        for name in first.checkpoints[0].model_state_dict
    )

    with summary.metrics_path.open(encoding="utf-8", newline="") as file:
        rows = list(csv.DictReader(file))
    assert tuple(rows[0]) == trainer.TRAINING_METRICS_COLUMNS
    assert [int(row["update_index"]) for row in rows] == [1, 2, 3]
    assert [int(row["vector_steps"]) for row in rows] == [2, 4, 6]
    assert [int(row["cumulative_discarded_pending_reset_slots"]) for row in rows] == [
        0,
        0,
        0,
    ]
    assert first_fsync_count == 2
    assert len(fsync_calls) == 4
    assert first.writer.path == tmp_path / "first"
    assert first.writer.flush_count == 2
    assert first.writer.close_count == 1
    assert {step for _tag, _value, step in first.writer.scalars} == {4, 6}
    assert all(path.name not in {"test.json", "test_pool.npz"} for path in opened_paths)


def test_update_limit_is_a_strict_prefix_and_final_checkpoint_is_not_lost(
    tmp_path: Path,
) -> None:
    trainer = _trainer_module()
    config = _small_training_config(tensorboard_enabled=False)
    policy = _policy(config)
    collector = _SeededCollectorStub(policy, config)
    updater = _ppo_module().PpoUpdater(policy, config.ppo)

    for bad_limit in (0, 4, True):
        with pytest.raises(ValueError, match="update_limit"):
            trainer.train_ppo(
                collector,
                updater,
                config,
                run_directory=tmp_path / f"bad-{bad_limit}",
                update_limit=bad_limit,
            )
        assert collector.initialized_seed is None
        assert not (tmp_path / f"bad-{bad_limit}").exists()

    checkpoints: list[Any] = []
    summary = trainer.train_ppo(
        collector,
        updater,
        config,
        run_directory=tmp_path / "prefix",
        update_limit=1,
        checkpoint_callback=checkpoints.append,
        clock=_Clock(10.0, 10.0, 12.0, 15.0),
        memory_sampler=lambda _device: trainer.TorchCudaMemoryMetrics(10, 20, 30),
    )

    assert not summary.configured_budget_completed
    assert summary.completed_updates == 1
    assert summary.vector_steps == 2
    assert summary.counts.raw_transitions == config.nominal_world_slots_per_update
    assert summary.counts.valid_transitions == config.nominal_world_slots_per_update
    assert len(summary.records) == 1
    assert summary.records[0].optimization.learning_rate == config.ppo.learning_rate
    assert len(checkpoints) == 1
    assert checkpoints[0].update_index == 1
    assert not checkpoints[0].is_scheduled
    assert checkpoints[0].is_final
    with summary.metrics_path.open(encoding="utf-8", newline="") as file:
        rows = list(csv.DictReader(file))
    assert len(rows) == 1
    assert tuple(rows[0]) == trainer.TRAINING_METRICS_COLUMNS


def test_resume_restores_rng_numbering_cadence_and_appends_validated_csv(
    tmp_path: Path,
) -> None:
    torch = _torch()
    trainer = _trainer_module()
    config = _small_training_config(tensorboard_enabled=False)
    policy = _policy(config)
    collector = _SeededCollectorStub(policy, config)
    updater = _ppo_module().PpoUpdater(policy, config.ppo)
    prefix_checkpoints: list[Any] = []
    run_directory = tmp_path / "resume-run"

    prefix = trainer.train_ppo(
        collector,
        updater,
        config,
        run_directory=run_directory,
        update_limit=2,
        checkpoint_callback=prefix_checkpoints.append,
        clock=_Clock(0.0, 0.0, 1.0, 1.0, 3.0, 5.0),
        memory_sampler=lambda _device: trainer.TorchCudaMemoryMetrics(10, 20, 30),
    )

    assert prefix.starting_update == 0
    assert prefix.completed_updates == 2
    assert prefix.end_to_end_elapsed_seconds == 5.0
    assert len(prefix_checkpoints) == 1
    checkpoint = prefix_checkpoints[0]
    assert checkpoint.update_index == 2
    assert checkpoint.is_scheduled and checkpoint.is_final
    assert checkpoint.resume_state == checkpoint.to_resume_state()
    assert checkpoint.resume_state.starting_update == 2
    assert checkpoint.resume_state.discarded_pending_reset_slots == 0
    assert checkpoint.resume_state.episodes.episode_length_sum_steps == 6
    resume_state = dataclasses.replace(
        checkpoint.resume_state,
        wall_elapsed_before_persistence_seconds=prefix.end_to_end_elapsed_seconds,
    )

    original_metrics = prefix.metrics_path.read_text(encoding="utf-8")
    corruptions = {
        "bad-header": original_metrics.replace(
            "update_index,",
            "wrong_update_index,",
            1,
        ),
        "bad-last-row": "\n".join(
            (
                *original_metrics.rstrip("\n").splitlines()[:-1],
                "1," + original_metrics.rstrip("\n").splitlines()[-1].split(",", 1)[1],
            )
        )
        + "\n",
    }
    for name, content in corruptions.items():
        corrupt_directory = tmp_path / name
        corrupt_directory.mkdir()
        (corrupt_directory / "metrics.csv").write_text(content, encoding="utf-8")
        untouched_collector = _SeededCollectorStub(policy, config)
        with pytest.raises(ValueError, match="resume metrics CSV"):
            trainer.train_ppo(
                untouched_collector,
                updater,
                config,
                run_directory=corrupt_directory,
                update_limit=3,
                resume_state=resume_state,
                clock=_Clock(0.0),
            )
        assert untouched_collector.initialized_seed is None

    resumed_policy = _policy(config)
    resumed_policy.load_state_dict(checkpoint.model_state_dict)
    resumed_updater = _ppo_module().PpoUpdater(resumed_policy, config.ppo)
    assert checkpoint.optimizer_state_dict is not None
    resumed_updater.optimizer.load_state_dict(checkpoint.optimizer_state_dict)
    observed_minibatch_states: list[Any] = []
    original_update = resumed_updater.update

    def recording_update(
        rollout: Any,
        *,
        learning_rate: float,
        minibatch_generator: Any,
    ) -> Any:
        observed_minibatch_states.append(minibatch_generator.get_state().clone())
        return original_update(
            rollout,
            learning_rate=learning_rate,
            minibatch_generator=minibatch_generator,
        )

    resumed_updater.update = recording_update
    fresh_collector = _SeededCollectorStub(resumed_policy, config)
    fresh_collector.collect_calls = 2
    resumed_checkpoints: list[Any] = []
    resumed = trainer.train_ppo(
        fresh_collector,
        resumed_updater,
        config,
        run_directory=run_directory,
        update_limit=3,
        resume_state=resume_state,
        checkpoint_callback=resumed_checkpoints.append,
        clock=_Clock(100.0, 100.0, 102.0, 107.0),
        memory_sampler=lambda _device: trainer.TorchCudaMemoryMetrics(11, 21, 31),
    )

    assert fresh_collector.initialized_seed == config.environment.environment_seed
    assert torch.equal(
        fresh_collector.observed_policy_rng_states[0],
        resume_state.policy_rng_state,
    )
    assert torch.equal(observed_minibatch_states[0], resume_state.minibatch_rng_state)
    assert resumed.starting_update == 2
    assert resumed.completed_updates == 3
    assert len(resumed.records) == 1
    assert resumed.records[0].update_index == 3
    assert resumed.records[0].optimization.learning_rate == pytest.approx(
        config.ppo.learning_rate / 3.0
    )
    assert resumed.compute_update_seconds == 2.0
    assert resumed.end_to_end_elapsed_seconds == 7.0
    assert resumed.counts.raw_transitions == config.world_step_slot_budget
    assert resumed.counts.valid_transitions == 6_143
    assert resumed.counts.dummy_reset_transitions == 1
    assert resumed.discarded_pending_reset_slots == 1
    assert resumed.episodes.episode_length_sum_steps == 8
    assert resumed.episodes.mean_episode_length_steps == pytest.approx(8.0 / 3.0)
    assert len(resumed_checkpoints) == 1
    assert not resumed_checkpoints[0].is_scheduled
    assert resumed_checkpoints[0].is_final
    assert resumed_checkpoints[0].resume_state.starting_update == 3
    assert resumed_checkpoints[0].resume_state.discarded_pending_reset_slots == 1
    with resumed.metrics_path.open(encoding="utf-8", newline="") as file:
        rows = list(csv.DictReader(file))
    assert [int(row["update_index"]) for row in rows] == [1, 2, 3]
    assert [int(row["cumulative_discarded_pending_reset_slots"]) for row in rows] == [
        0,
        0,
        1,
    ]
    assert [float(row["learning_rate"]) for row in rows] == pytest.approx(
        (
            config.ppo.learning_rate,
            config.ppo.learning_rate * 2.0 / 3.0,
            config.ppo.learning_rate / 3.0,
        )
    )


def test_resume_state_rejects_uncompensated_pending_carry_outside_vector_width() -> None:
    torch = _torch()
    trainer = _trainer_module()
    rollout = importlib.import_module("controller_learning.rl.rollout")
    counts = rollout.TransitionCounts(
        num_envs=2,
        environment_step_calls=1,
        raw_transitions=2,
        valid_transitions=2,
        dummy_reset_transitions=0,
        autoreset_slots=0,
        terminal_events=2,
        terminated_events=2,
        truncated_events=0,
    )
    episodes = trainer.EpisodeMetrics(
        episodes=2,
        successful_episodes=0,
        offtrack_episodes=2,
        invalid_action_episodes=0,
        timeout_episodes=0,
        successful_lap_time_sum_s=0.0,
        episode_length_sum_steps=2,
    )
    generator = torch.Generator(device=_device()).manual_seed(1)
    state = generator.get_state()

    with pytest.raises(ValueError, match="uncompensated pending-reset slots"):
        trainer.TrainingResumeState(
            starting_update=1,
            counts=counts,
            discarded_pending_reset_slots=3,
            episodes=episodes,
            cumulative_reward_sum=0.0,
            cumulative_compute_update_seconds=1.0,
            wall_elapsed_before_persistence_seconds=1.0,
            policy_rng_state=state,
            minibatch_rng_state=state,
        )


def test_body_and_tensorboard_cleanup_failures_are_both_reported(
    tmp_path: Path,
) -> None:
    trainer = _trainer_module()
    config = _small_training_config(tensorboard_enabled=True)
    policy = _policy(config)
    collector = _SeededCollectorStub(policy, config)
    updater = _ppo_module().PpoUpdater(policy, config.ppo)
    writers: list[_SummaryWriter] = []

    def writer_factory(path: Path) -> _SummaryWriter:
        writer = _SummaryWriter(path, close_error=OSError("tensorboard close failed"))
        writers.append(writer)
        return writer

    def fail_checkpoint(_request: Any) -> None:
        raise RuntimeError("checkpoint callback failed")

    with pytest.raises(BaseExceptionGroup) as captured:
        trainer.train_ppo(
            collector,
            updater,
            config,
            run_directory=tmp_path / "grouped-failure",
            update_limit=1,
            checkpoint_callback=fail_checkpoint,
            clock=_Clock(0.0, 0.0, 1.0),
            summary_writer_factory=writer_factory,
            memory_sampler=lambda _device: trainer.TorchCudaMemoryMetrics(1, 2, 3),
        )

    assert {type(error) for error in captured.value.exceptions} == {RuntimeError, OSError}
    assert writers[0].close_count == 1
    metrics_path = tmp_path / "grouped-failure" / "metrics.csv"
    with metrics_path.open(encoding="utf-8", newline="") as file:
        rows = list(csv.DictReader(file))
    assert len(rows) == 1
    assert int(rows[0]["update_index"]) == 1


def test_csv_cleanup_failure_is_propagated_after_other_cleanup(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    trainer = _trainer_module()
    config = _small_training_config(tensorboard_enabled=True)
    policy = _policy(config)
    collector = _SeededCollectorStub(policy, config)
    updater = _ppo_module().PpoUpdater(policy, config.ppo)
    writers: list[_SummaryWriter] = []
    original_close = trainer.FixedColumnCsvWriter.close

    def failing_csv_close(writer: Any) -> None:
        original_close(writer)
        raise OSError("metrics close failed")

    monkeypatch.setattr(trainer.FixedColumnCsvWriter, "close", failing_csv_close)

    def writer_factory(path: Path) -> _SummaryWriter:
        writer = _SummaryWriter(path)
        writers.append(writer)
        return writer

    with pytest.raises(OSError, match="metrics close failed"):
        trainer.train_ppo(
            collector,
            updater,
            config,
            run_directory=tmp_path / "csv-cleanup-failure",
            update_limit=1,
            clock=_Clock(0.0, 0.0, 1.0),
            summary_writer_factory=writer_factory,
            memory_sampler=lambda _device: trainer.TorchCudaMemoryMetrics(1, 2, 3),
        )

    assert writers[0].close_count == 1
    metrics_path = tmp_path / "csv-cleanup-failure" / "metrics.csv"
    with metrics_path.open(encoding="utf-8", newline="") as file:
        rows = list(csv.DictReader(file))
    assert len(rows) == 1


def test_trainer_has_no_asset_or_test_split_loading_path() -> None:
    source = (PROJECT_ROOT / "controller_learning" / "rl" / "trainer.py").read_text(
        encoding="utf-8"
    )
    assert "load_verified_train_pool" not in source
    assert "load_verified_asset" not in source
    assert "test.json" not in source
    assert "test_pool.npz" not in source
    assert "torch.utils.tensorboard" in source
    assert (
        "from torch.utils.tensorboard import SummaryWriter"
        not in source.split("def _create_summary_writer", maxsplit=1)[0]
    )
