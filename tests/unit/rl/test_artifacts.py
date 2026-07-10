from __future__ import annotations

import json
import math
import pickle
import subprocess
import sys
import threading
from concurrent.futures import ThreadPoolExecutor
from dataclasses import FrozenInstanceError, replace
from pathlib import Path
from typing import Any

import pytest

import controller_learning.rl.artifacts as artifacts
from controller_learning.rl.artifacts import (
    RESUME_SEMANTICS,
    ArtifactValidationError,
    ArtifactWriteError,
    TrainingCheckpointMetadata,
    TrainingRunIdentity,
)

_CONFIGURATION_SHA256 = "1" * 64
_LOCK_SHA256 = "2" * 64
_SOURCE_REVISION = "3" * 40
_TRAIN_MANIFEST_SHA256 = "4" * 64
_TRAIN_CACHE_SHA256 = "5" * 64


class PickleTorch:
    """Small injected Torch stand-in; the CPU tests never import PyTorch."""

    def __init__(self) -> None:
        self.fail_save = False
        self.fail_load = False
        self.corrupt_metadata_update: int | None = None
        self.corrupt_continuation_update: int | None = None

    def save(self, value: Any, file: Any) -> None:
        if self.fail_save:
            raise OSError("injected save failure")
        pickle.dump(value, file, protocol=pickle.HIGHEST_PROTOCOL)

    def load(
        self,
        file: Any,
        *,
        map_location: str,
        weights_only: bool,
    ) -> Any:
        assert map_location == "cpu"
        assert weights_only is False
        if self.fail_load:
            self.fail_load = False
            raise OSError("injected load failure")
        if hasattr(file, "read"):
            value = pickle.load(file)
        else:
            with Path(file).open("rb") as stream:
                value = pickle.load(stream)
        if value["metadata"]["update_index"] == self.corrupt_metadata_update:
            value["metadata"]["vector_steps"] += 1
        if value["metadata"]["update_index"] == self.corrupt_continuation_update:
            value["continuation_state"]["timeout_episodes"] += 1
        return value


class BlockingPickleTorch(PickleTorch):
    """Expose whether another writer reaches serialization while the first holds flock."""

    def __init__(self) -> None:
        super().__init__()
        self.first_save_entered = threading.Event()
        self.release_first_save = threading.Event()
        self.second_save_entered = threading.Event()
        self._calls_lock = threading.Lock()
        self.save_calls = 0

    def save(self, value: Any, file: Any) -> None:
        with self._calls_lock:
            self.save_calls += 1
            call = self.save_calls
        if call == 1:
            self.first_save_entered.set()
            if not self.release_first_save.wait(timeout=5.0):
                raise TimeoutError("test did not release first checkpoint save")
        elif call == 2:
            self.second_save_entered.set()
        super().save(value, file)


def _identity(*, run_id: str = "run-20260710") -> TrainingRunIdentity:
    return TrainingRunIdentity(
        run_id=run_id,
        benchmark_version="0.1",
        source_revision=_SOURCE_REVISION,
        configuration_sha256=_CONFIGURATION_SHA256,
        lock_sha256=_LOCK_SHA256,
        train_manifest_sha256=_TRAIN_MANIFEST_SHA256,
        train_cache_sha256=_TRAIN_CACHE_SHA256,
        feature_schema_version=artifacts.M7_FEATURE_SCHEMA_VERSION,
        reward_schema_version=artifacts.M7_REWARD_SCHEMA_VERSION,
        environment_seed=7,
        policy_seed=11,
        minibatch_seed=13,
    )


def _metadata(
    update: int,
    *,
    identity: TrainingRunIdentity | None = None,
) -> TrainingCheckpointMetadata:
    return TrainingCheckpointMetadata(
        run_identity=_identity() if identity is None else identity,
        update_index=update,
        vector_steps=update * 128,
        valid_transitions=update * 1020,
        elapsed_seconds=float(update) / 10.0,
    )


def _continuation(update: int) -> artifacts.TrainingContinuationState:
    return artifacts.TrainingContinuationState(
        starting_update=update,
        num_envs=8,
        environment_step_calls=update * 128,
        raw_transitions=update * 1024,
        valid_transitions=update * 1020,
        dummy_reset_transitions=update * 4,
        autoreset_slots=update * 4,
        terminal_events=update * 4,
        terminated_events=update * 3,
        truncated_events=update,
        episodes=update * 4,
        successful_episodes=update,
        offtrack_episodes=update,
        invalid_action_episodes=update,
        timeout_episodes=update,
        successful_lap_time_sum_s=update * 12.5,
        episode_length_sum_steps=update * 400,
        cumulative_reward_sum=update * -2.5,
        cumulative_compute_update_seconds=update * 0.08,
        wall_elapsed_before_persistence_seconds=float(update) / 10.0,
    )


def _save(
    root: Path,
    update: int,
    *,
    backend: PickleTorch,
    keep_last: int = 2,
    identity: TrainingRunIdentity | None = None,
    continuation_state: artifacts.TrainingContinuationState | None = None,
) -> artifacts.TrainingCheckpointArtifact:
    metadata = _metadata(update, identity=identity)
    return artifacts.save_training_checkpoint(
        root,
        metadata=metadata,
        continuation_state=(
            _continuation(update) if continuation_state is None else continuation_state
        ),
        model_state_dict={"weight": [update, update + 1]},
        optimizer_state_dict={"param_groups": [{"lr": 3.0e-4}], "state": {}},
        policy_rng_state=f"policy-{update}".encode(),
        minibatch_rng_state=f"minibatch-{update}".encode(),
        keep_last=keep_last,
        torch_module=backend,
    )


def test_module_import_is_torch_free() -> None:
    project_root = Path(__file__).resolve().parents[3]
    completed = subprocess.run(
        [
            sys.executable,
            "-c",
            (
                "import sys; import controller_learning.rl.artifacts; "
                "assert 'torch' not in sys.modules"
            ),
        ],
        cwd=project_root,
        check=False,
        capture_output=True,
        text=True,
    )
    assert completed.returncode == 0, completed.stderr


def test_canonical_json_is_deterministic_strict_and_utf8() -> None:
    first = {"z": (2, "Grüße"), "a": {"value": 1.5, "enabled": True}}
    second = {"a": {"enabled": True, "value": 1.5}, "z": [2, "Grüße"]}

    expected = b'{"a":{"enabled":true,"value":1.5},"z":[2,"Gr\xc3\xbc\xc3\x9fe"]}\n'
    assert artifacts.canonical_json_bytes(first) == expected
    assert artifacts.canonical_json_bytes(second) == expected

    for invalid in (math.nan, math.inf, -math.inf):
        with pytest.raises(ArtifactValidationError, match="NaN or infinity"):
            artifacts.canonical_json_bytes({"metric": invalid})
    with pytest.raises(ArtifactValidationError, match="string object keys"):
        artifacts.canonical_json_bytes({1: "ambiguous"})  # type: ignore[dict-item]
    with pytest.raises(ArtifactValidationError, match="unsupported JSON value"):
        artifacts.canonical_json_bytes({"path": Path("run")})
    cyclic: dict[str, Any] = {}
    cyclic["self"] = cyclic
    with pytest.raises(ArtifactValidationError, match="reference cycle"):
        artifacts.canonical_json_bytes(cyclic)


def test_atomic_json_write_fsyncs_replaces_then_reads_back_exactly(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    events: list[str] = []
    real_replace = artifacts.os.replace
    real_fsync_directory = artifacts._fsync_directory
    real_readback = artifacts._readback_bytes

    def tracked_replace(source: Path, destination: Path) -> None:
        events.append("replace")
        real_replace(source, destination)

    def tracked_directory_fsync(path: Path) -> None:
        events.append("directory_fsync")
        real_fsync_directory(path)

    def tracked_readback(path: Path) -> bytes:
        events.append("readback")
        return real_readback(path)

    monkeypatch.setattr(artifacts.os, "replace", tracked_replace)
    monkeypatch.setattr(artifacts, "_fsync_directory", tracked_directory_fsync)
    monkeypatch.setattr(artifacts, "_readback_bytes", tracked_readback)

    record = artifacts.atomic_write_json(tmp_path, "metrics/report.json", {"status": "pass"})
    payload = (tmp_path / record.relative_path).read_bytes()

    assert events[-3:] == ["replace", "directory_fsync", "readback"]
    assert set(events[:-3]) == {"directory_fsync"}
    assert payload == b'{"status":"pass"}\n'
    assert record.size_bytes == len(payload)
    assert record.sha256 == artifacts.sha256_bytes(payload)
    assert record.sha256 == artifacts.sha256_file(tmp_path / record.relative_path)
    assert (tmp_path / record.relative_path).stat().st_mode & 0o777 == 0o644


def test_atomic_write_failure_preserves_prior_artifact(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    relative = "report.json"
    artifacts.atomic_write_json(tmp_path, relative, {"generation": 1})
    destination = tmp_path / relative
    prior = destination.read_bytes()

    def fail_replace(_source: Path, _destination: Path) -> None:
        raise OSError("injected replace failure")

    monkeypatch.setattr(artifacts.os, "replace", fail_replace)
    with pytest.raises(ArtifactWriteError, match="failed to commit"):
        artifacts.atomic_write_json(tmp_path, relative, {"generation": 2})

    assert destination.read_bytes() == prior
    assert not list(tmp_path.glob(".report.json.*.tmp"))


def test_atomic_staging_failure_preserves_prior_artifact(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    relative = "report.json"
    artifacts.atomic_write_json(tmp_path, relative, {"generation": 1})
    destination = tmp_path / relative
    prior = destination.read_bytes()

    def fail_staging(_parent: Path, _name: str, _payload: bytes, _mode: int) -> Path:
        raise OSError("injected temporary-write failure")

    monkeypatch.setattr(artifacts, "_write_fsynced_temporary", fail_staging)
    with pytest.raises(ArtifactWriteError, match="failed to stage"):
        artifacts.atomic_write_json(tmp_path, relative, {"generation": 2})

    assert destination.read_bytes() == prior


def test_atomic_readback_failure_rolls_back_prior_artifact(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    relative = "report.json"
    artifacts.atomic_write_json(tmp_path, relative, {"generation": 1})
    destination = tmp_path / relative
    prior = destination.read_bytes()
    real_readback = artifacts._readback_bytes
    injected = False

    def corrupt_once(path: Path) -> bytes:
        nonlocal injected
        if path == destination and not injected:
            injected = True
            return b"corrupt"
        return real_readback(path)

    monkeypatch.setattr(artifacts, "_readback_bytes", corrupt_once)
    with pytest.raises(ArtifactWriteError, match="failed to commit"):
        artifacts.atomic_write_json(tmp_path, relative, {"generation": 2})

    assert injected is True
    assert destination.read_bytes() == prior
    assert artifacts.read_strict_json(tmp_path, relative) == {"generation": 1}


@pytest.mark.parametrize(
    "relative",
    (
        "../escape.json",
        "/absolute.json",
        "not\\posix.json",
        "a//not-normal.json",
        "space dir/report.json",
    ),
)
def test_artifact_paths_reject_unsafe_or_noncanonical_names(
    tmp_path: Path,
    relative: str,
) -> None:
    with pytest.raises(ArtifactValidationError):
        artifacts.atomic_write_json(tmp_path, relative, {"safe": True})


def test_artifact_path_rejects_symlink_parent(tmp_path: Path) -> None:
    outside = tmp_path / "outside"
    outside.mkdir()
    root = tmp_path / "root"
    root.mkdir()
    (root / "link").symlink_to(outside, target_is_directory=True)

    with pytest.raises(ArtifactValidationError, match="symlinks"):
        artifacts.atomic_write_json(root, "link/report.json", {"safe": False})
    assert not (outside / "report.json").exists()


def test_strict_json_reader_rejects_noncanonical_duplicates_and_constants(tmp_path: Path) -> None:
    (tmp_path / "pretty.json").write_text('{\n  "value": 1\n}\n')
    with pytest.raises(ArtifactValidationError, match="canonical"):
        artifacts.read_strict_json(tmp_path, "pretty.json")
    assert artifacts.read_strict_json(
        tmp_path,
        "pretty.json",
        require_canonical=False,
    ) == {"value": 1}

    (tmp_path / "duplicate.json").write_text('{"value":1,"value":2}\n')
    with pytest.raises(ArtifactValidationError, match="duplicate key"):
        artifacts.read_strict_json(tmp_path, "duplicate.json", require_canonical=False)

    (tmp_path / "nan.json").write_text('{"value":NaN}\n')
    with pytest.raises(ArtifactValidationError, match="forbids NaN"):
        artifacts.read_strict_json(tmp_path, "nan.json", require_canonical=False)


def test_identity_and_checkpoint_metadata_are_immutable_and_strict() -> None:
    identity = _identity()
    metadata = _metadata(3, identity=identity)

    assert metadata.resume_semantics == RESUME_SEMANTICS
    assert TrainingRunIdentity.from_dict(identity.to_dict()) == identity
    assert TrainingCheckpointMetadata.from_dict(metadata.to_dict()) == metadata
    with pytest.raises(FrozenInstanceError):
        identity.run_id = "other"  # type: ignore[misc]
    with pytest.raises(FrozenInstanceError):
        metadata.update_index = 4  # type: ignore[misc]

    with pytest.raises(ArtifactValidationError, match="finite non-negative"):
        replace(metadata, elapsed_seconds=math.nan)
    with pytest.raises(ArtifactValidationError, match="resume_semantics"):
        replace(metadata, resume_semantics="bit_exact_environment_resume")
    with pytest.raises(ArtifactValidationError, match="pairwise distinct"):
        replace(identity, policy_seed=identity.environment_seed)
    with pytest.raises(ArtifactValidationError, match="invalid format"):
        replace(identity, configuration_sha256="not-a-digest")
    with pytest.raises(ArtifactValidationError, match="feature_schema_version"):
        replace(identity, feature_schema_version=2)
    with pytest.raises(ArtifactValidationError, match="reward_schema_version"):
        replace(identity, reward_schema_version="unknown")
    with pytest.raises(ArtifactValidationError, match="invalid format"):
        replace(identity, train_cache_sha256="not-a-digest")
    with pytest.raises(ArtifactValidationError, match="keys differ"):
        TrainingRunIdentity.from_dict({**identity.to_dict(), "unexpected": True})


def test_training_continuation_state_matches_trainer_resume_accounting() -> None:
    continuation = _continuation(2)
    metadata = _metadata(2)

    continuation.validate_checkpoint_metadata(metadata)
    assert artifacts.TrainingContinuationState.from_dict(continuation.to_dict()) == continuation
    assert set(continuation.to_dict()) == {
        "autoreset_slots",
        "cumulative_compute_update_seconds",
        "cumulative_reward_sum",
        "dummy_reset_transitions",
        "environment_step_calls",
        "episode_length_sum_steps",
        "episodes",
        "invalid_action_episodes",
        "num_envs",
        "offtrack_episodes",
        "raw_transitions",
        "schema_version",
        "starting_update",
        "successful_episodes",
        "successful_lap_time_sum_s",
        "terminal_events",
        "terminated_events",
        "timeout_episodes",
        "truncated_events",
        "valid_transitions",
        "wall_elapsed_before_persistence_seconds",
    }
    with pytest.raises(FrozenInstanceError):
        continuation.starting_update = 3  # type: ignore[misc]


def test_training_continuation_rejects_accounting_reason_and_time_violations() -> None:
    continuation = _continuation(1)

    with pytest.raises(ArtifactValidationError, match=r"num_envs \* environment_step_calls"):
        replace(continuation, raw_transitions=continuation.raw_transitions + 1)
    with pytest.raises(ArtifactValidationError, match="autoreset_slots"):
        replace(continuation, autoreset_slots=continuation.autoreset_slots + 1)
    with pytest.raises(ArtifactValidationError, match=r"terminated \+ truncated"):
        replace(continuation, terminal_events=continuation.terminal_events + 1)
    with pytest.raises(ArtifactValidationError, match="four reason counts"):
        replace(continuation, episodes=continuation.episodes + 1)
    with pytest.raises(ArtifactValidationError, match="completed-episode lengths"):
        replace(
            continuation,
            episode_length_sum_steps=continuation.valid_transitions + 1,
        )
    with pytest.raises(ArtifactValidationError, match="successful lap time"):
        replace(continuation, successful_lap_time_sum_s=0.0)
    with pytest.raises(ArtifactValidationError, match="finite number"):
        replace(continuation, cumulative_reward_sum=math.nan)
    with pytest.raises(ArtifactValidationError, match="below cumulative compute"):
        replace(
            continuation,
            cumulative_compute_update_seconds=0.2,
            wall_elapsed_before_persistence_seconds=0.1,
        )
    with pytest.raises(ArtifactValidationError, match="keys differ"):
        artifacts.TrainingContinuationState.from_dict({**continuation.to_dict(), "unexpected": 1})


@pytest.mark.parametrize(
    ("metadata_field", "value", "message"),
    (
        ("update_index", 2, "update_index"),
        ("vector_steps", 129, "vector_steps"),
        ("valid_transitions", 1019, "valid_transitions"),
        ("elapsed_seconds", 0.2, "elapsed_seconds"),
    ),
)
def test_training_continuation_binds_checkpoint_metadata(
    metadata_field: str,
    value: int | float,
    message: str,
) -> None:
    metadata = replace(_metadata(1), **{metadata_field: value})
    with pytest.raises(ArtifactValidationError, match=message):
        _continuation(1).validate_checkpoint_metadata(metadata)


def test_checkpoint_save_verifies_hash_size_payload_and_latest_pointer(tmp_path: Path) -> None:
    backend = PickleTorch()
    result = _save(tmp_path, 1, backend=backend)
    checkpoint_path = tmp_path / result.checkpoint.relative_path
    latest_path = tmp_path / result.latest_pointer.relative_path

    assert checkpoint_path.is_file()
    assert checkpoint_path.stat().st_mode & 0o777 == 0o600
    assert result.checkpoint.size_bytes == checkpoint_path.stat().st_size
    assert result.checkpoint.sha256 == artifacts.sha256_file(checkpoint_path)
    assert result.latest_pointer.size_bytes == latest_path.stat().st_size
    assert result.latest_pointer.sha256 == artifacts.sha256_file(latest_path)
    assert result.pruned_relative_paths == ()

    pointer = artifacts.read_latest_checkpoint_pointer(tmp_path)
    assert pointer is not None
    assert pointer.checkpoint == result.checkpoint
    assert pointer.update_index == 1
    assert pointer.resume_semantics == RESUME_SEMANTICS

    with checkpoint_path.open("rb") as file:
        payload = pickle.load(file)
    assert set(payload) == {
        "continuation_state",
        "metadata",
        "minibatch_rng_state",
        "model_state_dict",
        "optimizer_state_dict",
        "policy_rng_state",
        "schema_version",
    }
    assert payload["metadata"]["resume_semantics"] == RESUME_SEMANTICS
    assert payload["continuation_state"] == _continuation(1).to_dict()
    assert "environment_state" not in payload


def test_checkpoint_pointer_is_published_before_oldest_first_pruning(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    backend = PickleTorch()
    _save(tmp_path, 1, backend=backend, keep_last=4)
    _save(tmp_path, 2, backend=backend, keep_last=4)
    _save(tmp_path, 3, backend=backend, keep_last=4)

    pointer_published = False
    unlinked: list[str] = []
    real_atomic_write_json = artifacts.atomic_write_json
    real_unlink = artifacts._unlink_checkpoint

    def tracked_atomic_write_json(
        root: Path,
        relative_path: str | Path,
        value: dict[str, Any],
        *,
        overwrite: bool = True,
    ) -> artifacts.ArtifactRecord:
        nonlocal pointer_published
        result = real_atomic_write_json(
            root,
            relative_path,
            value,
            overwrite=overwrite,
        )
        if Path(relative_path).name == "latest.json":
            pointer_published = True
        return result

    def tracked_unlink(path: Path) -> None:
        assert pointer_published is True
        unlinked.append(path.name)
        real_unlink(path)

    monkeypatch.setattr(artifacts, "atomic_write_json", tracked_atomic_write_json)
    monkeypatch.setattr(artifacts, "_unlink_checkpoint", tracked_unlink)
    result = _save(tmp_path, 4, backend=backend, keep_last=1)

    assert unlinked == ["update_00000001.pt", "update_00000002.pt", "update_00000003.pt"]
    assert result.pruned_relative_paths == tuple(f"checkpoints/{name}" for name in unlinked)
    assert sorted(path.name for path in (tmp_path / "checkpoints").glob("*.pt")) == [
        "update_00000004.pt"
    ]
    pointer = artifacts.read_latest_checkpoint_pointer(tmp_path)
    assert pointer is not None and pointer.update_index == 4


def test_checkpoint_readback_failure_preserves_prior_latest(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    backend = PickleTorch()
    first = _save(tmp_path, 1, backend=backend)
    latest_path = tmp_path / first.latest_pointer.relative_path
    prior_latest = latest_path.read_bytes()
    real_readback = artifacts._readback_bytes

    def corrupt_new_checkpoint(path: Path) -> bytes:
        if path.name == "update_00000002.pt":
            return b"corrupt"
        return real_readback(path)

    monkeypatch.setattr(artifacts, "_readback_bytes", corrupt_new_checkpoint)

    with pytest.raises(ArtifactWriteError, match="exact byte readback"):
        _save(tmp_path, 2, backend=backend)

    assert latest_path.read_bytes() == prior_latest
    assert (tmp_path / first.checkpoint.relative_path).is_file()
    assert not (tmp_path / "checkpoints/update_00000002.pt").exists()
    assert artifacts.read_latest_checkpoint_pointer(tmp_path).update_index == 1  # type: ignore[union-attr]


def test_checkpoint_schema_readback_failure_preserves_prior_latest(tmp_path: Path) -> None:
    backend = PickleTorch()
    first = _save(tmp_path, 1, backend=backend)
    latest_path = tmp_path / first.latest_pointer.relative_path
    prior_latest = latest_path.read_bytes()
    backend.corrupt_metadata_update = 2

    with pytest.raises(ArtifactWriteError, match="schema readback"):
        _save(tmp_path, 2, backend=backend)

    assert latest_path.read_bytes() == prior_latest
    assert not (tmp_path / "checkpoints/update_00000002.pt").exists()


def test_checkpoint_continuation_readback_failure_preserves_prior_latest(
    tmp_path: Path,
) -> None:
    backend = PickleTorch()
    first = _save(tmp_path, 1, backend=backend)
    latest_path = tmp_path / first.latest_pointer.relative_path
    prior_latest = latest_path.read_bytes()
    backend.corrupt_continuation_update = 2

    with pytest.raises(ArtifactWriteError, match="schema readback"):
        _save(tmp_path, 2, backend=backend)

    assert latest_path.read_bytes() == prior_latest
    assert not (tmp_path / "checkpoints/update_00000002.pt").exists()


def test_latest_pointer_failure_preserves_prior_checkpoint_and_pointer(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    backend = PickleTorch()
    first = _save(tmp_path, 1, backend=backend)
    latest_path = tmp_path / first.latest_pointer.relative_path
    prior_latest = latest_path.read_bytes()
    real_atomic_write_json = artifacts.atomic_write_json

    def fail_latest(
        _root: Path,
        relative_path: str | Path,
        _value: dict[str, Any],
        *,
        overwrite: bool = True,
    ) -> artifacts.ArtifactRecord:
        assert overwrite is True
        assert Path(relative_path).name == "latest.json"
        raise ArtifactWriteError("injected latest failure")

    monkeypatch.setattr(artifacts, "atomic_write_json", fail_latest)
    with pytest.raises(ArtifactWriteError, match="injected latest failure"):
        _save(tmp_path, 2, backend=backend)

    assert latest_path.read_bytes() == prior_latest
    assert (tmp_path / first.checkpoint.relative_path).is_file()
    orphan = tmp_path / "checkpoints/update_00000002.pt"
    assert orphan.is_file()

    monkeypatch.setattr(artifacts, "atomic_write_json", real_atomic_write_json)
    backend.fail_save = True
    retried = _save(tmp_path, 2, backend=backend)
    assert retried.checkpoint.relative_path == "checkpoints/update_00000002.pt"
    assert backend.fail_save is True
    assert artifacts.read_latest_checkpoint_pointer(tmp_path).update_index == 2  # type: ignore[union-attr]


def test_same_update_orphan_with_different_continuation_is_replaced(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    backend = PickleTorch()
    _save(tmp_path, 1, backend=backend)
    real_atomic_write_json = artifacts.atomic_write_json

    def fail_latest(
        _root: Path,
        _relative_path: str | Path,
        _value: dict[str, Any],
        *,
        overwrite: bool = True,
    ) -> artifacts.ArtifactRecord:
        assert overwrite is True
        raise ArtifactWriteError("injected latest failure")

    monkeypatch.setattr(artifacts, "atomic_write_json", fail_latest)
    with pytest.raises(ArtifactWriteError, match="injected latest failure"):
        _save(tmp_path, 2, backend=backend)
    orphan = tmp_path / "checkpoints/update_00000002.pt"
    orphan_sha256 = artifacts.sha256_file(orphan)

    monkeypatch.setattr(artifacts, "atomic_write_json", real_atomic_write_json)
    changed = replace(_continuation(2), cumulative_reward_sum=123.0)
    saved = _save(
        tmp_path,
        2,
        backend=backend,
        continuation_state=changed,
    )

    assert saved.checkpoint.sha256 != orphan_sha256
    loaded = artifacts.load_training_checkpoint(
        tmp_path,
        expected_identity=_identity(),
        torch_module=backend,
    )
    assert loaded.continuation_state == changed


def test_checkpoint_rejects_unsafe_state_and_nonmonotonic_metadata(tmp_path: Path) -> None:
    backend = PickleTorch()
    _save(tmp_path, 2, backend=backend)

    with pytest.raises(ArtifactValidationError, match="increase monotonically"):
        _save(tmp_path, 1, backend=backend)
    with pytest.raises(ArtifactValidationError, match="different full run identity"):
        _save(tmp_path, 3, backend=backend, identity=_identity(run_id="another-run"))
    with pytest.raises(ArtifactValidationError, match="unsafe path component"):
        artifacts.save_training_checkpoint(
            tmp_path,
            metadata=_metadata(3),
            continuation_state=_continuation(3),
            model_state_dict={"weight": 1},
            optimizer_state_dict={"state": {}},
            policy_rng_state=b"policy",
            minibatch_rng_state=b"minibatch",
            keep_last=1,
            checkpoint_directory="../escape",
            torch_module=backend,
        )
    with pytest.raises(ArtifactValidationError, match="non-empty mapping"):
        artifacts.save_training_checkpoint(
            tmp_path / "empty-state",
            metadata=_metadata(1),
            continuation_state=_continuation(1),
            model_state_dict={},
            optimizer_state_dict={"state": {}},
            policy_rng_state=b"policy",
            minibatch_rng_state=b"minibatch",
            keep_last=1,
            torch_module=backend,
        )
    with pytest.raises(ArtifactValidationError, match="RNG states"):
        artifacts.save_training_checkpoint(
            tmp_path / "missing-rng",
            metadata=_metadata(1),
            continuation_state=_continuation(1),
            model_state_dict={"weight": 1},
            optimizer_state_dict={"state": {}},
            policy_rng_state=None,
            minibatch_rng_state=b"minibatch",
            keep_last=1,
            torch_module=backend,
        )


def test_torch_save_failure_leaves_no_checkpoint_or_latest(tmp_path: Path) -> None:
    backend = PickleTorch()
    backend.fail_save = True

    with pytest.raises(ArtifactWriteError, match="serialize and verify"):
        _save(tmp_path, 1, backend=backend)

    assert not (tmp_path / "checkpoints/update_00000001.pt").exists()
    assert not (tmp_path / "checkpoints/latest.json").exists()
    assert not list((tmp_path / "checkpoints").glob("*.tmp"))


def test_latest_pointer_json_is_canonical_and_exact_schema(tmp_path: Path) -> None:
    result = _save(tmp_path, 1, backend=PickleTorch())
    pointer_path = tmp_path / result.latest_pointer.relative_path
    parsed = json.loads(pointer_path.read_bytes())

    assert pointer_path.read_bytes() == artifacts.canonical_json_bytes(parsed)
    assert parsed["schema_version"] == artifacts.LATEST_CHECKPOINT_SCHEMA_VERSION
    assert parsed["checkpoint"]["sha256"] == result.checkpoint.sha256
    assert parsed["checkpoint"]["size_bytes"] == result.checkpoint.size_bytes


def test_latest_pointer_rejects_checkpoint_path_or_update_tampering(tmp_path: Path) -> None:
    result = _save(tmp_path, 1, backend=PickleTorch())
    latest_relative = result.latest_pointer.relative_path
    pointer = artifacts.read_strict_json(tmp_path, latest_relative)

    pointer["checkpoint"]["relative_path"] = "checkpoints/update_00000002.pt"
    artifacts.atomic_write_json(tmp_path, latest_relative, pointer)
    with pytest.raises(ArtifactValidationError, match="filename must match"):
        artifacts.read_latest_checkpoint_pointer(tmp_path)

    pointer["checkpoint"]["relative_path"] = "other/update_00000001.pt"
    artifacts.atomic_write_json(tmp_path, latest_relative, pointer)
    with pytest.raises(ArtifactValidationError, match="checkpoint_directory"):
        artifacts.read_latest_checkpoint_pointer(tmp_path)


def test_load_training_checkpoint_verifies_payload_and_full_expected_identity(
    tmp_path: Path,
) -> None:
    backend = PickleTorch()
    identity = _identity()
    saved = _save(tmp_path, 1, backend=backend, identity=identity)

    loaded = artifacts.load_training_checkpoint(
        tmp_path,
        expected_identity=identity,
        torch_module=backend,
    )

    assert loaded.pointer.checkpoint == saved.checkpoint
    assert loaded.pointer.run_identity_sha256 == artifacts.run_identity_sha256(identity)
    assert loaded.pointer.published_updates == (1,)
    assert loaded.metadata == _metadata(1, identity=identity)
    assert loaded.continuation_state == _continuation(1)
    assert loaded.payload["model_state_dict"] == {"weight": [1, 2]}
    assert loaded.payload["optimizer_state_dict"]["param_groups"][0]["lr"] == 3.0e-4
    with pytest.raises(TypeError):
        loaded.payload["unexpected"] = True  # type: ignore[index]

    changed_cache = replace(identity, train_cache_sha256="6" * 64)
    with pytest.raises(ArtifactValidationError, match="expected_identity"):
        artifacts.load_training_checkpoint(
            tmp_path,
            expected_identity=changed_cache,
            torch_module=backend,
        )


def test_load_rejects_in_memory_continuation_tampering(tmp_path: Path) -> None:
    backend = PickleTorch()
    identity = _identity()
    _save(tmp_path, 1, backend=backend, identity=identity)
    backend.corrupt_continuation_update = 1

    with pytest.raises(ArtifactValidationError, match="four reason counts"):
        artifacts.load_training_checkpoint(
            tmp_path,
            expected_identity=identity,
            torch_module=backend,
        )


def test_load_training_checkpoint_rejects_size_hash_and_symlink_tampering(
    tmp_path: Path,
) -> None:
    backend = PickleTorch()
    identity = _identity()
    saved = _save(tmp_path, 1, backend=backend, identity=identity)
    checkpoint = tmp_path / saved.checkpoint.relative_path
    original = checkpoint.read_bytes()

    checkpoint.write_bytes(original + b"tamper")
    with pytest.raises(ArtifactValidationError, match="size differs"):
        artifacts.load_training_checkpoint(
            tmp_path,
            expected_identity=identity,
            torch_module=backend,
        )

    checkpoint.write_bytes(b"x" * len(original))
    with pytest.raises(ArtifactValidationError, match="SHA-256 differs"):
        artifacts.load_training_checkpoint(
            tmp_path,
            expected_identity=identity,
            torch_module=backend,
        )

    external = tmp_path / "external.pt"
    external.write_bytes(original)
    checkpoint.unlink()
    checkpoint.symlink_to(external)
    with pytest.raises(ArtifactValidationError, match="symbolic link"):
        artifacts.load_training_checkpoint(
            tmp_path,
            expected_identity=identity,
            torch_module=backend,
        )


def test_read_paths_do_not_create_missing_directories(tmp_path: Path) -> None:
    missing = tmp_path / "never-create"
    identity = _identity()

    assert artifacts.read_latest_checkpoint_pointer(missing) is None
    assert not missing.exists()
    with pytest.raises(ArtifactValidationError, match="does not exist"):
        artifacts.read_strict_json(missing, "report.json")
    assert not missing.exists()
    with pytest.raises(ArtifactValidationError, match="root does not exist"):
        artifacts.load_training_checkpoint(
            missing,
            expected_identity=identity,
            torch_module=PickleTorch(),
        )
    assert not missing.exists()


def test_save_continuity_compares_full_identity_not_only_run_id(tmp_path: Path) -> None:
    backend = PickleTorch()
    identity = _identity()
    _save(tmp_path, 1, backend=backend, identity=identity)
    changed_manifest = replace(identity, train_manifest_sha256="7" * 64)

    with pytest.raises(ArtifactValidationError, match="different full run identity"):
        _save(tmp_path, 2, backend=backend, identity=changed_manifest)

    pointer = artifacts.read_latest_checkpoint_pointer(tmp_path)
    assert pointer is not None and pointer.update_index == 1


def test_invalid_same_update_orphan_is_cleaned_and_replaced_under_lock(tmp_path: Path) -> None:
    backend = PickleTorch()
    checkpoint_directory = tmp_path / "checkpoints"
    checkpoint_directory.mkdir()
    orphan = checkpoint_directory / "update_00000001.pt"
    orphan.write_bytes(b"not a checkpoint")

    saved = _save(tmp_path, 1, backend=backend)

    assert orphan.is_file()
    assert orphan.read_bytes() != b"not a checkpoint"
    assert artifacts.sha256_file(orphan) == saved.checkpoint.sha256
    loaded = artifacts.load_training_checkpoint(
        tmp_path,
        expected_identity=_identity(),
        torch_module=backend,
    )
    assert loaded.metadata.update_index == 1


def test_pruning_uses_published_updates_and_ignores_lower_and_higher_orphans(
    tmp_path: Path,
) -> None:
    backend = PickleTorch()
    first = _save(tmp_path, 1, backend=backend, keep_last=4)
    checkpoint_directory = tmp_path / "checkpoints"
    lower_orphan = checkpoint_directory / "update_00000002.pt"
    higher_orphan = checkpoint_directory / "update_99999999.pt"
    lower_orphan.write_bytes(b"lower orphan")
    higher_orphan.write_bytes(b"higher orphan")

    third = _save(tmp_path, 3, backend=backend, keep_last=1)

    assert not (tmp_path / first.checkpoint.relative_path).exists()
    assert lower_orphan.read_bytes() == b"lower orphan"
    assert higher_orphan.read_bytes() == b"higher orphan"
    assert (tmp_path / third.checkpoint.relative_path).is_file()
    pointer = artifacts.read_latest_checkpoint_pointer(tmp_path)
    assert pointer is not None
    assert pointer.update_index == 3
    assert pointer.published_updates == (1, 3)


def test_advisory_lock_serializes_concurrent_checkpoint_transactions(tmp_path: Path) -> None:
    backend = BlockingPickleTorch()
    identity = _identity()

    with ThreadPoolExecutor(max_workers=2) as executor:
        first = executor.submit(
            _save,
            tmp_path,
            1,
            backend=backend,
            keep_last=2,
            identity=identity,
        )
        assert backend.first_save_entered.wait(timeout=2.0)
        second = executor.submit(
            _save,
            tmp_path,
            2,
            backend=backend,
            keep_last=2,
            identity=identity,
        )
        try:
            assert not backend.second_save_entered.wait(timeout=0.1)
        finally:
            backend.release_first_save.set()
        assert first.result(timeout=5.0).metadata.update_index == 1
        assert second.result(timeout=5.0).metadata.update_index == 2

    assert backend.save_calls == 2
    pointer = artifacts.read_latest_checkpoint_pointer(tmp_path)
    assert pointer is not None
    assert pointer.update_index == 2
    assert pointer.published_updates == (1, 2)


def test_shared_load_lock_waits_for_exclusive_save_transaction(tmp_path: Path) -> None:
    identity = _identity()
    _save(tmp_path, 1, backend=PickleTorch(), identity=identity)
    saving_backend = BlockingPickleTorch()
    loading_backend = PickleTorch()

    with ThreadPoolExecutor(max_workers=2) as executor:
        saving = executor.submit(
            _save,
            tmp_path,
            2,
            backend=saving_backend,
            keep_last=2,
            identity=identity,
        )
        assert saving_backend.first_save_entered.wait(timeout=2.0)
        loading = executor.submit(
            artifacts.load_training_checkpoint,
            tmp_path,
            expected_identity=identity,
            torch_module=loading_backend,
        )
        try:
            assert not loading.done()
            assert not threading.Event().wait(timeout=0.1)
            assert not loading.done()
        finally:
            saving_backend.release_first_save.set()
        assert saving.result(timeout=5.0).metadata.update_index == 2
        assert loading.result(timeout=5.0).metadata.update_index == 2


def test_load_rejects_pointer_identity_digest_tampering(tmp_path: Path) -> None:
    backend = PickleTorch()
    identity = _identity()
    saved = _save(tmp_path, 1, backend=backend, identity=identity)
    pointer = artifacts.read_strict_json(tmp_path, saved.latest_pointer.relative_path)
    pointer["run_identity_sha256"] = "8" * 64
    artifacts.atomic_write_json(tmp_path, saved.latest_pointer.relative_path, pointer)

    with pytest.raises(ArtifactValidationError, match="expected_identity"):
        artifacts.load_training_checkpoint(
            tmp_path,
            expected_identity=identity,
            torch_module=backend,
        )


@pytest.mark.gpu
def test_real_torch_checkpoint_round_trip_in_gpu_environment(tmp_path: Path) -> None:
    import torch

    identity = _identity()
    generator = torch.Generator(device="cpu")
    generator.manual_seed(17)
    rng_state = generator.get_state()
    saved = artifacts.save_training_checkpoint(
        tmp_path,
        metadata=_metadata(1, identity=identity),
        continuation_state=_continuation(1),
        model_state_dict={"weight": torch.tensor([1.0, -2.0])},
        optimizer_state_dict={"state": {}, "param_groups": [{"lr": 3.0e-4}]},
        policy_rng_state=rng_state,
        minibatch_rng_state=rng_state.clone(),
        keep_last=1,
        torch_module=torch,
    )
    loaded = artifacts.load_training_checkpoint(
        tmp_path,
        expected_identity=identity,
        torch_module=torch,
    )

    assert saved.checkpoint.size_bytes > 0
    assert loaded.continuation_state == _continuation(1)
    assert torch.equal(
        loaded.payload["model_state_dict"]["weight"],
        torch.tensor([1.0, -2.0]),
    )
    assert torch.equal(loaded.payload["policy_rng_state"], rng_state)
