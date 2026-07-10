"""CPU protocol tests for the formal M7 PPO training entrypoint."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

from controller_learning.config import load_project_config
from controller_learning.rl.artifacts import (
    ArtifactRecord,
    ArtifactValidationError,
    TrainingRunIdentity,
    atomic_write_bytes,
    atomic_write_json,
)
from controller_learning.rl.assets import TrainPoolAccessEvidence
from controller_learning.rl.configuration import load_ppo_config
from controller_learning.rl.rollout import TransitionCounts
from controller_learning.tracks.types import TrackCapacity
from scripts import train_ppo

PROJECT_ROOT = Path(__file__).parents[3]


def _train_evidence() -> TrainPoolAccessEvidence:
    return TrainPoolAccessEvidence(
        schema_version="controller-learning.m7-train-pool-access.v1",
        loaded_splits=("train",),
        benchmark_version="0.1",
        generator_version="periodic-cubic-v1",
        level_id=1,
        split="train",
        manifest_file="train.json",
        manifest_sha256="1" * 64,
        cache_file="train_pool.npz",
        manifest_asset_sha256="2" * 64,
        cache_file_sha256="2" * 64,
        track_count=10_000,
        capacity=TrackCapacity(max_track_points=640, max_checkpoints=48),
        first_track_id=0,
        last_track_id=9_999,
        track_ids_sha256="3" * 64,
        geometry_hashes_sha256="4" * 64,
        loader_accessed_validation=False,
        loader_accessed_test=False,
    )


def _identity(run_id: str = "smoke-001") -> TrainingRunIdentity:
    return TrainingRunIdentity(
        run_id=run_id,
        benchmark_version="0.1",
        source_revision="5" * 40,
        configuration_sha256="6" * 64,
        lock_sha256="7" * 64,
        train_manifest_sha256="8" * 64,
        train_cache_sha256="9" * 64,
        feature_schema_version=1,
        reward_schema_version="controller-learning.m7-public-reward.v1",
        environment_seed=7,
        policy_seed=11,
        minibatch_seed=13,
    )


def test_cli_module_import_sets_allocator_policy_without_importing_torch_training_modules() -> None:
    completed = subprocess.run(
        (
            sys.executable,
            "-c",
            (
                "import os,sys; import scripts.train_ppo; "
                "assert os.environ['CUDA_DEVICE_ORDER']=='PCI_BUS_ID'; "
                "assert os.environ['XLA_PYTHON_CLIENT_PREALLOCATE']=='false'; "
                "assert 'torch' not in sys.modules; "
                "assert 'controller_learning.rl.trainer' not in sys.modules; "
                "assert 'controller_learning.rl.policy' not in sys.modules"
            ),
        ),
        cwd=PROJECT_ROOT,
        check=False,
        capture_output=True,
        text=True,
    )
    assert completed.returncode == 0, completed.stderr


def test_options_and_parser_lock_device_run_id_and_smoke_prefix() -> None:
    options = train_ppo._parse_args(("--run-id", "train-smoke", "--smoke-updates", "2", "--resume"))
    assert options == train_ppo.TrainingOptions(
        run_id="train-smoke",
        smoke_updates=2,
        resume=True,
    )

    with pytest.raises(ValueError, match="requires device"):
        train_ppo.TrainingOptions(run_id="train-smoke", device="cuda:1")
    with pytest.raises(ArtifactValidationError, match=r"run_identity\.run_id"):
        train_ppo.TrainingOptions(run_id="Invalid Run")
    with pytest.raises(SystemExit):
        train_ppo._parse_args(("--run-id", "train-smoke", "--smoke-updates", "0"))


def test_source_preflight_requires_full_revision_and_clean_status(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    responses = iter((("a" * 40, None), ("", None)))
    monkeypatch.setattr(train_ppo, "_run_command", lambda *_args, **_kwargs: next(responses))
    snapshot = train_ppo.capture_source_snapshot(PROJECT_ROOT)
    assert snapshot.to_dict() == {"revision": "a" * 40, "worktree_clean": True}
    train_ppo.require_formal_source(snapshot)

    dirty = train_ppo.SourceSnapshot(
        revision="b" * 40,
        worktree_clean=False,
        status=" M scripts/train_ppo.py",
    )
    with pytest.raises(RuntimeError, match="clean worktree"):
        train_ppo.require_formal_source(dirty)


def test_runtime_audit_guard_allows_only_train_assets_and_records_categories(
    tmp_path: Path,
) -> None:
    official = tmp_path / "official" / "v0.1"
    cache_root = tmp_path / "cache"
    official.mkdir(parents=True)
    cache_root.mkdir()
    train_manifest = official / "train.json"
    validation_asset = official / "validation.npz"
    test_manifest = official / "test.json"
    train_cache = cache_root / "train_pool.npz"
    for path in (train_manifest, validation_asset, test_manifest, train_cache):
        path.write_bytes(b"asset")

    guard = train_ppo.OfficialTrainAssetAccessGuard(
        official_track_root=official.parent,
        train_manifest=train_manifest,
        track_cache_root=cache_root,
        train_cache=train_cache,
    )
    guard.install()

    assert train_manifest.read_bytes() == b"asset"
    assert train_cache.read_bytes() == b"asset"
    with pytest.raises(
        train_ppo.ForbiddenOfficialAssetAccessError,
        match="read-only inputs",
    ):
        train_cache.write_bytes(b"mutated")
    assert train_cache.read_bytes() == b"asset"
    with pytest.raises(
        train_ppo.ForbiddenOfficialAssetAccessError,
        match="forbids non-Train",
    ):
        validation_asset.read_bytes()
    with pytest.raises(
        train_ppo.ForbiddenOfficialAssetAccessError,
        match="forbids non-Train",
    ):
        test_manifest.read_bytes()

    evidence = guard.evidence(loader_succeeded=True)
    assert evidence["audit_hook_installed_before_asset_loader"] is True
    assert evidence["opened_splits"] == ["train"]
    assert evidence["opened_path_categories"] == [
        "configured_train_cache",
        "official_train_manifest",
    ]
    assert evidence["open_event_counts"] == {
        "configured_train_cache": 2,
        "official_train_manifest": 1,
    }
    assert [event["category"] for event in evidence["open_event_sequence"]] == [
        "official_train_manifest",
        "configured_train_cache",
        "configured_train_cache",
    ]
    assert evidence["denied_event_count"] == 3
    assert evidence["validation_opened"] is False
    assert evidence["test_opened"] is False


def test_runtime_audit_guard_denies_test_asset_in_dedicated_process(tmp_path: Path) -> None:
    official = tmp_path / "official" / "v0.1"
    cache_root = tmp_path / "cache"
    official.mkdir(parents=True)
    cache_root.mkdir()
    train_manifest = official / "train.json"
    test_manifest = official / "test.json"
    train_cache = cache_root / "train_pool.npz"
    for path in (train_manifest, test_manifest, train_cache):
        path.write_bytes(b"asset")
    program = f"""
from pathlib import Path
from scripts.train_ppo import ForbiddenOfficialAssetAccessError, OfficialTrainAssetAccessGuard
guard = OfficialTrainAssetAccessGuard(
    official_track_root=Path({str(official.parent)!r}),
    train_manifest=Path({str(train_manifest)!r}),
    track_cache_root=Path({str(cache_root)!r}),
    train_cache=Path({str(train_cache)!r}),
)
guard.install()
try:
    Path({str(test_manifest)!r}).read_bytes()
except ForbiddenOfficialAssetAccessError:
    print("blocked-before-read")
else:
    raise AssertionError("Test asset read was not blocked")
"""
    completed = subprocess.run(
        (sys.executable, "-c", program),
        cwd=PROJECT_ROOT,
        check=False,
        capture_output=True,
        text=True,
    )
    assert completed.returncode == 0, completed.stderr
    assert completed.stdout.strip() == "blocked-before-read"


def test_formal_executable_has_no_all_split_verifier_or_selection_loader_reference() -> None:
    source = (PROJECT_ROOT / "scripts/train_ppo.py").read_text(encoding="utf-8")
    for forbidden in (
        "verify_official_track_assets",
        "load_manifest_track_batch",
        "load_verified_validation_pool",
        "load_verified_test_pool",
    ):
        assert forbidden not in source


def test_run_identity_binds_config_lock_train_assets_schemas_and_three_seeds(
    tmp_path: Path,
) -> None:
    config_path = tmp_path / "ppo.toml"
    lock_path = tmp_path / "pixi.lock"
    config_path.write_bytes((PROJECT_ROOT / "configs/ppo.toml").read_bytes())
    lock_path.write_bytes(b"test lock\n")
    config = load_ppo_config(config_path)
    source = train_ppo.SourceSnapshot("c" * 40, True, "")

    identity = train_ppo.build_run_identity(
        run_id="formal-001",
        config=config,
        config_path=config_path,
        lock_path=lock_path,
        source=source,
        train_evidence=_train_evidence(),
    )

    assert identity.run_id == "formal-001"
    assert identity.source_revision == "c" * 40
    assert identity.configuration_sha256 == train_ppo.sha256_file(config_path)
    assert identity.lock_sha256 == train_ppo.sha256_file(lock_path)
    assert identity.train_manifest_sha256 == "1" * 64
    assert identity.train_cache_sha256 == "2" * 64
    assert identity.feature_schema_version == 1
    assert identity.reward_schema_version == "controller-learning.m7-public-reward.v1"
    assert (
        identity.environment_seed,
        identity.policy_seed,
        identity.minibatch_seed,
    ) == (7, 11, 13)


def test_training_accounting_recomputes_budget_next_step_and_physics_identities() -> None:
    project = load_project_config(PROJECT_ROOT)
    config = load_ppo_config(PROJECT_ROOT / "configs/ppo.toml")
    counts = TransitionCounts(
        num_envs=1024,
        environment_step_calls=256,
        raw_transitions=262_144,
        valid_transitions=262_140,
        dummy_reset_transitions=4,
        autoreset_slots=4,
        terminal_events=6,
        terminated_events=5,
        truncated_events=1,
    )
    episodes = SimpleNamespace(episodes=6, invalid_action_episodes=0)
    summary = SimpleNamespace(
        configured_updates=80,
        starting_update=1,
        completed_updates=2,
        configured_budget_completed=False,
        counts=counts,
        episodes=episodes,
    )

    evidence = train_ppo._training_accounting(
        config=config,
        project=project,
        summary=summary,
    )

    assert evidence["invocation_updates"] == 1
    assert evidence["raw_world_slots"] == 262_144
    assert evidence["valid_transitions"] + evidence["dummy_reset_transitions"] == 262_144
    assert evidence["autoreset_slots"] == evidence["dummy_reset_transitions"]
    assert evidence["terminal_events"] == (
        evidence["terminated_events"] + evidence["truncated_events"]
    )
    assert evidence["final_pending_reset_slots"] == 2
    assert evidence["physics_substeps"] == 2_621_440

    summary.episodes = SimpleNamespace(episodes=6, invalid_action_episodes=1)
    with pytest.raises(RuntimeError, match="invalid-action episode"):
        train_ppo._training_accounting(
            config=config,
            project=project,
            summary=summary,
        )


def test_resume_verifies_canonical_manifest_full_identity_and_exact_config(tmp_path: Path) -> None:
    identity = _identity()
    config_bytes = b"schema_version = 1\n"
    atomic_write_bytes(tmp_path, "config.toml", config_bytes, overwrite=False)
    atomic_write_json(
        tmp_path,
        "manifest.json",
        {
            "schema_version": train_ppo.TRAINING_MANIFEST_SCHEMA_VERSION,
            "run_identity": identity.to_dict(),
            "status": "smoke_complete",
        },
    )

    manifest = train_ppo._verify_existing_run(
        tmp_path,
        identity=identity,
        config_bytes=config_bytes,
    )
    assert manifest["status"] == "smoke_complete"

    with pytest.raises(ArtifactValidationError, match="identity differs"):
        train_ppo._verify_existing_run(
            tmp_path,
            identity=_identity("different-run"),
            config_bytes=config_bytes,
        )
    with pytest.raises(ArtifactValidationError, match="config snapshot differs"):
        train_ppo._verify_existing_run(
            tmp_path,
            identity=identity,
            config_bytes=b"different\n",
        )


def test_completed_run_artifacts_bind_csv_tensorboard_checkpoint_and_pointer(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    (tmp_path / "checkpoints").mkdir()
    files = {
        "config.toml": b"config\n",
        "metrics.csv": b"update_index\n1\n",
        "events.out.tfevents.test": b"events",
        "checkpoints/update_00000002.pt": b"checkpoint",
        "checkpoints/latest.json": b"pointer",
    }
    for relative, payload in files.items():
        (tmp_path / relative).write_bytes(payload)
    checkpoint = train_ppo._existing_artifact(
        tmp_path,
        "checkpoints/update_00000002.pt",
    )
    pointer = SimpleNamespace(update_index=2, checkpoint=checkpoint)
    monkeypatch.setattr(train_ppo, "read_latest_checkpoint_pointer", lambda _root: pointer)
    summary = SimpleNamespace(
        completed_updates=2,
        metrics_path=tmp_path / "metrics.csv",
    )

    artifacts = train_ppo._completed_run_artifacts(
        tmp_path,
        summary=summary,
        tensorboard_enabled=True,
    )

    assert ArtifactRecord(**artifacts["config"]).sha256 == train_ppo.sha256_file(
        tmp_path / "config.toml"
    )
    assert ArtifactRecord(**artifacts["metrics_csv"]).size_bytes == len(files["metrics.csv"])
    assert artifacts["final_checkpoint"] == checkpoint.to_dict()
    assert len(artifacts["tensorboard_events"]) == 1
    assert artifacts["latest_checkpoint_pointer"]["relative_path"] == ("checkpoints/latest.json")


def test_checkpoint_callback_wires_cumulative_metadata_and_all_resume_state(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured: dict[str, object] = {}

    def fake_save(root: Path, **kwargs: object) -> str:
        captured["root"] = root
        captured.update(kwargs)
        return "saved"

    monkeypatch.setattr(train_ppo, "save_training_checkpoint", fake_save)
    counts = SimpleNamespace(
        num_envs=2,
        environment_step_calls=5,
        raw_transitions=10,
        valid_transitions=9,
        dummy_reset_transitions=1,
        autoreset_slots=1,
        terminal_events=2,
        terminated_events=1,
        truncated_events=1,
    )
    episodes = SimpleNamespace(
        episodes=2,
        successful_episodes=1,
        offtrack_episodes=0,
        invalid_action_episodes=0,
        timeout_episodes=1,
        successful_lap_time_sum_s=2.5,
        episode_length_sum_steps=5,
    )
    resume_state = SimpleNamespace(
        starting_update=10,
        counts=counts,
        episodes=episodes,
        cumulative_reward_sum=-1.5,
        cumulative_compute_update_seconds=0.8,
        wall_elapsed_before_persistence_seconds=1.0,
    )
    request = SimpleNamespace(
        update_index=10,
        vector_steps=5,
        elapsed_seconds=1.0,
        counts=counts,
        resume_state=resume_state,
        model_state_dict={"model": 1},
        optimizer_state_dict={"optimizer": 2},
        policy_rng_state="policy-rng",
        minibatch_rng_state="minibatch-rng",
    )
    backend = object()
    callback = train_ppo._checkpoint_callback(
        tmp_path,
        _identity(),
        keep_last=3,
        torch_module=backend,
    )

    assert callback(request) == "saved"
    metadata = captured["metadata"]
    assert metadata.update_index == 10
    assert metadata.vector_steps == 5
    assert metadata.valid_transitions == 9
    assert metadata.elapsed_seconds == 1.0
    continuation = captured["continuation_state"]
    assert continuation.starting_update == 10
    assert continuation.raw_transitions == 10
    assert continuation.valid_transitions == 9
    assert continuation.episodes == 2
    assert continuation.episode_length_sum_steps == 5
    assert continuation.cumulative_reward_sum == -1.5
    assert continuation.cumulative_compute_update_seconds == 0.8
    assert continuation.wall_elapsed_before_persistence_seconds == 1.0
    assert captured["model_state_dict"] == {"model": 1}
    assert captured["optimizer_state_dict"] == {"optimizer": 2}
    assert captured["policy_rng_state"] == "policy-rng"
    assert captured["minibatch_rng_state"] == "minibatch-rng"
    assert captured["keep_last"] == 3
    assert captured["checkpoint_directory"] == "checkpoints"
    assert captured["torch_module"] is backend


class _FakeCuda:
    def synchronize(self, _device: object) -> None:
        return None

    def memory_allocated(self, _device: object) -> int:
        return 100

    def memory_reserved(self, _device: object) -> int:
        return 200

    def max_memory_allocated(self, _device: object) -> int:
        return 150


class _FakeJaxDevice:
    def memory_stats(self) -> dict[str, int]:
        return {"bytes_in_use": 300, "peak_bytes_in_use": 400}


def test_memory_evidence_reports_process_torch_and_jax_peaks(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(train_ppo, "_process_vram_mib", lambda _uuid: (500.0, None))
    recorder = train_ppo.MemoryEvidenceRecorder(
        torch=SimpleNamespace(cuda=_FakeCuda()),
        torch_device="cuda:0",
        jax_device=_FakeJaxDevice(),
        gpu_uuid="private",
    )
    sample = recorder.sample("boundary")
    report = recorder.report()

    assert sample["synchronized"] is True
    assert sample["process_vram_mib"] == 500.0
    assert sample["torch_cuda_allocated_bytes"] == 100
    assert sample["torch_cuda_reserved_bytes"] == 200
    assert sample["jax_allocator"]["bytes_in_use"] == 300
    assert report["peak_sampled_process_vram_mib"] == 500.0
    assert report["peak_sampled_torch_cuda_allocated_bytes"] == 100
    assert report["peak_sampled_torch_cuda_reserved_bytes"] == 200
    assert report["peak_sampled_jax_allocator_bytes"] == 400


def test_pixi_exposes_one_gpu_train_task_and_formal_project_config_is_level_one() -> None:
    pyproject = (PROJECT_ROOT / "pyproject.toml").read_text(encoding="utf-8")
    assert 'train-ppo = { cmd = "python scripts/train_ppo.py"' in pyproject
    project = load_project_config(PROJECT_ROOT)
    config = load_ppo_config(PROJECT_ROOT / "configs/ppo.toml")
    assert project.benchmark.official_level == 1
    assert config.environment.num_envs == 1024
    assert config.environment.backend == "mjx_warp"
    assert config.environment.train_cache.endswith("train_pool.npz")
