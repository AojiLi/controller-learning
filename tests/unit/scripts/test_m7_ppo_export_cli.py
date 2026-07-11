"""CPU tests for the strict, asset-free M7 PPO Controller export process."""

from __future__ import annotations

import copy
import hashlib
import json
import os
import shutil
import stat
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from controller_learning.rl.artifacts import (
    ArtifactRecord,
    TrainingRunIdentity,
    canonical_json_bytes,
    read_strict_json,
)
from controller_learning.rl.export_protocol import (
    EXPORT_REPORT_PATH,
    EXPORT_REPORT_SCHEMA_VERSION,
    ExportProtocolError,
    selected_export_candidate,
    validate_export_report,
)
from controller_learning.rl.schema import (
    LOCAL_TRACK_FEATURE_SCHEMA_VERSION,
    PUBLIC_REWARD_SCHEMA_VERSION,
)
from scripts import export_m7_ppo_controller as export_cli

PROJECT_ROOT = Path(__file__).parents[3]


def test_run_command_preserves_git_porcelain_leading_status_column() -> None:
    output = export_cli._run_command(
        (sys.executable, "-c", "import sys; sys.stdout.write(' M tracked\\n')"),
        cwd=PROJECT_ROOT,
    )

    assert output == " M tracked"


def _record(path: str, digest: str, size: int = 10) -> dict[str, Any]:
    return ArtifactRecord(relative_path=path, sha256=digest, size_bytes=size).to_dict()


def _identity(*, configuration_sha256: str = "c" * 64) -> TrainingRunIdentity:
    return TrainingRunIdentity(
        run_id="m7-formal-v0-1-001",
        benchmark_version="0.1",
        source_revision="1" * 40,
        configuration_sha256=configuration_sha256,
        lock_sha256="2" * 64,
        train_manifest_sha256="3" * 64,
        train_cache_sha256="4" * 64,
        feature_schema_version=LOCAL_TRACK_FEATURE_SCHEMA_VERSION,
        reward_schema_version=PUBLIC_REWARD_SCHEMA_VERSION,
        environment_seed=7,
        policy_seed=11,
        minibatch_seed=13,
    )


def _selection_stub(
    *,
    checkpoint_sha256: str = "5" * 64,
    parameter_sha256: str = "6" * 64,
    policy_sha256: str = "7" * 64,
    identity: TrainingRunIdentity | None = None,
) -> dict[str, Any]:
    identity = _identity() if identity is None else identity
    update = 20
    checkpoint = {
        "checkpoint": _record("checkpoints/update_00000020.pt", checkpoint_sha256, 123),
        "inference_policy": {
            "schema_version": 1,
            "sha256": policy_sha256,
            "size_bytes": 456,
        },
        "parameter_sha256": parameter_sha256,
        "update_index": update,
        "valid_transitions": 2_000,
        "vector_steps": 2_560,
    }
    return {
        "artifacts": {},
        "configuration": {"checkpoint_directory": "checkpoints"},
        "evaluations": {
            "candidates": [
                {
                    "parameter_sha256_after": parameter_sha256,
                    "parameter_sha256_before": parameter_sha256,
                    "parameter_unchanged": True,
                    "policy_id": "checkpoint_update_00000020",
                    "update_index": update,
                }
            ]
        },
        "gates": {"passed": True},
        "schema_version": "controller-learning.m7-ppo-selection-report.v1",
        "selection": {"selected_update": update},
        "status": "passed",
        "training_run": {
            "candidate_checkpoints": [checkpoint],
            "identity": identity.to_dict(),
        },
    }


def _export_report() -> dict[str, Any]:
    identity = _identity()
    checkpoint_sha = "5" * 64
    policy_sha = "7" * 64
    selection_config_sha = "8" * 64
    selection_report_sha = "9" * 64
    training_config_sha = identity.configuration_sha256
    outputs = sorted(
        [
            EXPORT_REPORT_PATH,
            "controllers/ppo/config.toml",
            "controllers/ppo/metadata.json",
            "controllers/ppo/policy.npz",
        ]
    )
    return {
        "asset_access": {
            "audit_hook_installed_before_project_imports": True,
            "denied_event_count": 0,
            "denied_mutation_event_count": 0,
            "denied_open_event_count": 0,
            "official_track_open_count": 0,
            "official_track_mutation_count": 0,
            "opened_path_categories": [],
            "track_cache_open_count": 0,
            "track_cache_mutation_count": 0,
            "mutation_event_counts": {},
            "unaudited_mutation_wrappers": ["os.mkfifo", "os.mknod"],
        },
        "controller": {
            "artifacts": {
                "config": _record("controllers/ppo/config.toml", "a" * 64),
                "metadata": _record("controllers/ppo/metadata.json", "b" * 64),
                "policy": _record("controllers/ppo/policy.npz", policy_sha, 456),
            },
            "checkpoint": {
                "checkpoint_sha256": checkpoint_sha,
                "run_id": identity.run_id,
                "source_revision": identity.source_revision,
                "training_configuration_sha256": identity.configuration_sha256,
                "update_index": 20,
                "valid_transitions": 2_000,
                "vector_steps": 2_560,
            },
            "inference_only": {
                "contains_environment_state": False,
                "contains_optimizer_state": False,
                "contains_value_network": False,
            },
            "plugin_directory": "controllers/ppo",
            "runtime": "numpy",
        },
        "input_stability": {
            "all_inputs_unchanged": True,
            "post_export_sha256": {
                "selected_checkpoint": checkpoint_sha,
                "selection_config": selection_config_sha,
                "selection_report": selection_report_sha,
                "training_config": training_config_sha,
            },
            "pre_export_sha256": {
                "selected_checkpoint": checkpoint_sha,
                "selection_config": selection_config_sha,
                "selection_report": selection_report_sha,
                "training_config": training_config_sha,
            },
        },
        "protocol": {
            "canonical_inference_policy_verified": True,
            "canonical_selection_report_required": True,
            "exact_published_checkpoint_loader": "v2_explicit_update",
            "formal_export_function": (
                "controller_learning.rl.controller_export.export_ppo_controller"
            ),
            "full_parameter_sha256_verified": True,
            "no_gradient_or_optimizer_operations": True,
            "one_time_unfinalized_template_activation": True,
            "passed_selection_gate_required": True,
            "persistent_crash_recovery": {
                "commit_transition": "READY_to_COMMITTED_then_cleanup",
                "exporter_starts_only_after_ready": True,
                "original_config_bytes_and_mode_fsynced": True,
                "startup_ready_action": "restore_config_delete_outputs_then_cleanup",
                "startup_unready_action": "cleanup_staging_only",
                "temporary_file_location": "transaction_staging_only",
                "transaction_directory": "runs/ppo/.m7-controller-export-transaction",
            },
            "selection_outputs_committed_before_export": True,
        },
        "schema_version": EXPORT_REPORT_SCHEMA_VERSION,
        "selection": {
            "config": _record("configs/ppo_selection.toml", selection_config_sha),
            "gate_passed": True,
            "report": _record("benchmarks/v0.1/m7_ppo_selection_report.json", selection_report_sha),
            "report_schema_version": "controller-learning.m7-ppo-selection-report.v1",
            "report_status": "passed",
            "selected_candidate": {
                "checkpoint": _record("checkpoints/update_00000020.pt", checkpoint_sha, 123),
                "inference_policy": {
                    "schema_version": 1,
                    "sha256": policy_sha,
                    "size_bytes": 456,
                },
                "parameter_sha256": "6" * 64,
                "update_index": 20,
                "valid_transitions": 2_000,
                "vector_steps": 2_560,
            },
        },
        "source": {
            "post_export_worktree": {
                "allowed_generated_output_paths": outputs,
                "observed_changed_paths": outputs,
                "only_allowed_generated_outputs": True,
                "revision": "f" * 40,
                "unexpected_changed_paths": [],
            },
            "preflight": {"revision": "f" * 40, "worktree_clean": True},
        },
        "status": "passed",
        "training": {
            "checkpoint_directory": "checkpoints",
            "identity": identity.to_dict(),
            "run_directory": "runs/ppo/m7-formal-v0-1-001",
            "training_config": _record("configs/ppo.toml", training_config_sha),
        },
    }


def test_cli_import_is_torch_jax_and_environment_asset_free() -> None:
    completed = subprocess.run(
        (
            sys.executable,
            "-c",
            (
                "import sys; import scripts.export_m7_ppo_controller; "
                "assert 'torch' not in sys.modules; assert 'jax' not in sys.modules; "
                "assert 'controller_learning.rl.artifacts' not in sys.modules; "
                "assert 'controller_learning.rl.selection' not in sys.modules"
            ),
        ),
        cwd=PROJECT_ROOT,
        check=False,
        capture_output=True,
        text=True,
    )
    assert completed.returncode == 0, completed.stderr
    source = (PROJECT_ROOT / "scripts/export_m7_ppo_controller.py").read_text(encoding="utf-8")
    for forbidden in (
        "controller_learning.envs",
        "load_verified_train_pool",
        "load_verified_validation_pool",
        "load_verified_test_pool",
        ".backward(",
        "optimizer.step(",
    ):
        assert forbidden not in source


def test_selected_candidate_joins_ranking_checkpoint_and_evaluated_parameters() -> None:
    report = _selection_stub()
    selected = selected_export_candidate(report)

    assert selected.update_index == 20
    assert selected.checkpoint.relative_path == "checkpoints/update_00000020.pt"
    assert selected.parameter_sha256 == "6" * 64
    assert selected.inference_policy == {
        "schema_version": 1,
        "sha256": "7" * 64,
        "size_bytes": 456,
    }

    tampered = copy.deepcopy(report)
    tampered["evaluations"]["candidates"][0]["parameter_sha256_after"] = "0" * 64
    with pytest.raises(ExportProtocolError, match="does not bind"):
        selected_export_candidate(tampered)
    failed = copy.deepcopy(report)
    failed["status"] = "gate_failed"
    failed["gates"]["passed"] = False
    with pytest.raises(ExportProtocolError, match="passed selection gate"):
        selected_export_candidate(failed)


def test_export_report_recomputes_policy_checkpoint_training_and_source_bindings() -> None:
    report = _export_report()
    validate_export_report(report)

    tampered_policy = copy.deepcopy(report)
    tampered_policy["controller"]["artifacts"]["policy"]["sha256"] = "0" * 64
    with pytest.raises(ExportProtocolError, match="exported policy differs"):
        validate_export_report(tampered_policy)
    tampered_checkpoint = copy.deepcopy(report)
    tampered_checkpoint["controller"]["checkpoint"]["update_index"] = 30
    with pytest.raises(ExportProtocolError, match="checkpoint identity differs"):
        validate_export_report(tampered_checkpoint)
    missing_output = copy.deepcopy(report)
    missing_output["source"]["post_export_worktree"]["observed_changed_paths"].pop()
    with pytest.raises(ExportProtocolError, match="worktree evidence differs"):
        validate_export_report(missing_output)


def test_unfinalized_template_preflight_rejects_stale_or_prior_outputs(tmp_path: Path) -> None:
    plugin = tmp_path / "controllers" / "ppo"
    plugin.mkdir(parents=True)
    (plugin / "controller.py").write_text("class Controller: pass\n", encoding="utf-8")
    (plugin / "config.toml").write_text("finalized = false\n", encoding="utf-8")
    report = tmp_path / EXPORT_REPORT_PATH
    export_cli._require_unfinalized_template(plugin, report_path=report)

    (plugin / "policy.npz").write_bytes(b"stale")
    with pytest.raises(RuntimeError, match="already exists"):
        export_cli._require_unfinalized_template(plugin, report_path=report)


def test_deny_all_asset_guard_blocks_an_open_and_records_the_attempt(tmp_path: Path) -> None:
    official = tmp_path / "official"
    cache = tmp_path / "cache"
    guard = export_cli.ExportAssetAccessGuard(
        official_track_root=official,
        track_cache_root=cache,
    )
    # Synthetic dispatch test: avoid installing a permanent process-wide hook in pytest.
    guard._installed = True

    with pytest.raises(export_cli.ForbiddenExportAssetAccessError, match="official_track"):
        guard._audit("open", (official / "v0.1/validation.npz", "rb", 0))

    evidence = guard.evidence()
    assert evidence["audit_hook_installed_before_project_imports"] is True
    assert evidence["denied_event_count"] == 1
    assert evidence["denied_open_event_count"] == 1
    assert evidence["denied_mutation_event_count"] == 0
    assert evidence["official_track_open_count"] == 1
    assert evidence["mutation_event_counts"] == {}


def test_deny_all_asset_guard_blocks_synthetic_mutations_of_sources_and_targets(
    tmp_path: Path,
) -> None:
    official = tmp_path / "official"
    cache = tmp_path / "cache"
    outside = tmp_path / "outside"
    official_path = official / "v0.1/asset.npz"
    cache_path = cache / "v0.1/train_pool.npz"
    guard = export_cli.ExportAssetAccessGuard(
        official_track_root=official,
        track_cache_root=cache,
    )
    # Synthetic dispatch test: avoid installing permanent process-wide wrappers in pytest.
    guard._installed = True
    events: tuple[tuple[str, tuple[Any, ...]], ...] = (
        ("os.rename", (official_path, outside, -1, -1)),
        ("os.replace", (outside, official_path, -1, -1)),
        ("os.remove", (official_path, -1)),
        ("os.unlink", (cache_path, -1)),
        ("os.rmdir", (official / "v0.1", -1)),
        ("os.mkdir", (cache / "v0.1", 0o755, -1)),
        ("os.link", (outside, official_path, -1, -1)),
        ("os.symlink", ("target", cache_path, -1)),
        ("os.truncate", (official_path, 0)),
        ("os.chmod", (cache_path, 0o600, -1)),
        ("os.chown", (official_path, 1_000, 1_000, -1)),
        ("os.utime", (cache_path, None, None, -1)),
        ("shutil.rmtree", (official / "v0.1", None)),
        ("os.mkfifo", (cache_path, 0o600, -1)),
        ("os.mknod", (official_path, 0o600, 0, -1)),
        ("os.setxattr", (cache_path, "user.test", b"value", 0)),
        ("os.removexattr", (official_path, "user.test")),
    )

    for event, arguments in events:
        with pytest.raises(export_cli.ForbiddenExportAssetAccessError, match="mutation"):
            guard._audit(event, arguments)

    evidence = guard.evidence()
    assert evidence["denied_event_count"] == len(events)
    assert evidence["denied_open_event_count"] == 0
    assert evidence["denied_mutation_event_count"] == len(events)
    assert evidence["official_track_mutation_count"] == 10
    assert evidence["track_cache_mutation_count"] == 7
    assert evidence["mutation_event_counts"] == {event: 1 for event, _arguments in events}


def test_mutation_guard_resolves_relative_destination_against_dir_fd(tmp_path: Path) -> None:
    parent = tmp_path / "parent"
    parent.mkdir()
    descriptor = os.open(parent, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
    try:
        guard = export_cli.ExportAssetAccessGuard(
            official_track_root=parent / "official",
            track_cache_root=parent / "cache",
        )
        guard._installed = True
        with pytest.raises(export_cli.ForbiddenExportAssetAccessError, match=r"os\.mkdir"):
            guard._audit("os.mkdir", ("official/new", 0o755, descriptor))
    finally:
        os.close(descriptor)

    assert guard.evidence()["official_track_mutation_count"] == 1


def test_install_wraps_real_unaudited_mkfifo_and_mknod_in_isolated_process(
    tmp_path: Path,
) -> None:
    official = tmp_path / "official"
    cache = tmp_path / "cache"
    outside = tmp_path / "outside"
    for directory in (official, cache, outside):
        directory.mkdir()
    program = f"""
import json
import os
import stat
from pathlib import Path
from scripts.export_m7_ppo_controller import (
    ExportAssetAccessGuard,
    ForbiddenExportAssetAccessError,
)
official = Path({str(official)!r})
cache = Path({str(cache)!r})
outside = Path({str(outside)!r})
official_fd = os.open(official, os.O_RDONLY | getattr(os, 'O_DIRECTORY', 0))
cache_fd = os.open(cache, os.O_RDONLY | getattr(os, 'O_DIRECTORY', 0))
guard = ExportAssetAccessGuard(official_track_root=official, track_cache_root=cache)
guard.install()
try:
    try:
        os.mkfifo('blocked.fifo', dir_fd=official_fd)
    except ForbiddenExportAssetAccessError:
        pass
    else:
        raise AssertionError('protected mkfifo was not denied')
    try:
        os.mknod('blocked.node', stat.S_IFREG | 0o600, dir_fd=cache_fd)
    except ForbiddenExportAssetAccessError:
        pass
    else:
        raise AssertionError('protected mknod was not denied')
    os.mkfifo(outside / 'allowed.fifo')
    os.mknod(outside / 'allowed.node', stat.S_IFREG | 0o600)
finally:
    os.close(official_fd)
    os.close(cache_fd)
print(json.dumps(guard.evidence(), sort_keys=True))
"""
    completed = subprocess.run(
        (sys.executable, "-c", program),
        cwd=PROJECT_ROOT,
        check=False,
        capture_output=True,
        text=True,
    )

    assert completed.returncode == 0, completed.stderr
    assert not (official / "blocked.fifo").exists()
    assert not (cache / "blocked.node").exists()
    assert stat.S_ISFIFO((outside / "allowed.fifo").stat().st_mode)
    assert (outside / "allowed.node").is_file()
    evidence = json.loads(completed.stdout)
    assert evidence["denied_mutation_event_count"] == 2
    assert evidence["mutation_event_counts"] == {"os.mkfifo": 1, "os.mknod": 1}
    assert evidence["unaudited_mutation_wrappers"] == ["os.mkfifo", "os.mknod"]


def test_output_transaction_rolls_back_a_failure_after_report_publication(
    tmp_path: Path,
) -> None:
    plugin = tmp_path / "controllers/ppo"
    plugin.mkdir(parents=True)
    report = tmp_path / EXPORT_REPORT_PATH
    report.parent.mkdir(parents=True)
    config = plugin / "config.toml"
    original = b"finalized = false\n"
    config.write_bytes(original)

    with (
        pytest.raises(RuntimeError, match="injected post-publication failure"),
        export_cli._ExportOutputTransaction(
            project_root=tmp_path,
            plugin_directory=plugin,
            report_path=report,
        ),
    ):
        config.write_bytes(b"finalized = true\n")
        (plugin / "metadata.json").write_bytes(b"metadata")
        (plugin / "policy.npz").write_bytes(b"policy")
        report.write_bytes(b"report")
        raise RuntimeError("injected post-publication failure")

    assert config.read_bytes() == original
    assert not (plugin / "metadata.json").exists()
    assert not (plugin / "policy.npz").exists()
    assert not report.exists()


def test_startup_recovers_ready_transaction_after_simulated_process_loss(
    tmp_path: Path,
) -> None:
    plugin = tmp_path / "controllers/ppo"
    plugin.mkdir(parents=True)
    report = tmp_path / EXPORT_REPORT_PATH
    report.parent.mkdir(parents=True)
    controller = plugin / "controller.py"
    config = plugin / "config.toml"
    controller.write_text("class Controller: pass\n", encoding="utf-8")
    original = b'name = "ppo"\nfinalized = false\n'
    config.write_bytes(original)
    config.chmod(0o640)
    (tmp_path / ".gitignore").write_text("/runs/\n", encoding="utf-8")
    commands = (
        ("git", "init", "-q"),
        ("git", "config", "user.name", "Test"),
        ("git", "config", "user.email", "test@example.com"),
        (
            "git",
            "add",
            ".gitignore",
            "controllers/ppo/controller.py",
            "controllers/ppo/config.toml",
        ),
        ("git", "commit", "-qm", "test: seed export recovery fixture"),
    )
    for command in commands:
        subprocess.run(command, cwd=tmp_path, check=True, capture_output=True, text=True)

    interrupted = export_cli._ExportOutputTransaction(
        project_root=tmp_path,
        plugin_directory=plugin,
        report_path=report,
    )
    interrupted.__enter__()
    transaction_directory = tmp_path / export_cli.EXPORT_TRANSACTION_DIRECTORY
    assert (transaction_directory / "READY").read_bytes() == (
        b"controller-learning.m7-export-ready.v1\n"
    )
    staging = transaction_directory / "staging"
    (staging / ".policy.npz.interrupted.tmp").write_bytes(b"staged policy residue")
    (staging / ".config.toml.interrupted.recovery").write_bytes(b"staged recovery residue")
    (staging / ".m7_ppo_export_report.json.interrupted.tmp").write_bytes(b"staged report residue")
    config.write_bytes(b'name = "ppo"\nfinalized = true\n')
    config.chmod(0o600)
    (plugin / "policy.npz").write_bytes(b"partial policy")
    (plugin / "metadata.json").write_bytes(b"partial metadata")
    report.write_bytes(b"partial report")
    # Simulate SIGKILL/power loss: deliberately do not invoke __exit__ on the old object.
    del interrupted

    recovery = export_cli._recover_persistent_export_transaction(tmp_path)

    assert recovery == "ready_rolled_back"
    assert config.read_bytes() == original
    assert stat.S_IMODE(config.stat().st_mode) == 0o640
    assert not (plugin / "policy.npz").exists()
    assert not (plugin / "metadata.json").exists()
    assert not report.exists()
    assert not transaction_directory.exists()
    assert not tuple(plugin.glob("*.tmp"))
    assert not tuple(plugin.glob("*.recovery"))
    assert not tuple(report.parent.glob("*.tmp"))
    assert not tuple(report.parent.glob("*.recovery"))
    export_cli._require_unfinalized_template(plugin, report_path=report)
    assert export_cli._source_snapshot(tmp_path) == {
        "revision": subprocess.run(
            ("git", "rev-parse", "HEAD"),
            cwd=tmp_path,
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip(),
        "worktree_clean": True,
    }


def test_startup_cleans_unready_staging_without_touching_controller(tmp_path: Path) -> None:
    plugin = tmp_path / "controllers/ppo"
    plugin.mkdir(parents=True)
    config = plugin / "config.toml"
    config.write_bytes(b"finalized = false\n")
    transaction = tmp_path / export_cli.EXPORT_TRANSACTION_DIRECTORY
    transaction.mkdir(parents=True)
    (transaction / "original_config.bin").write_bytes(config.read_bytes())

    recovery = export_cli._recover_persistent_export_transaction(tmp_path)

    assert recovery == "unready_staging_cleaned"
    assert config.read_bytes() == b"finalized = false\n"
    assert not transaction.exists()


def test_run_export_binds_temp_canonical_report_checkpoint_and_fake_exporter(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root = tmp_path
    for directory in (
        root / "configs",
        root / "benchmarks" / "v0.1",
        root / "runs" / "ppo" / "m7-formal-v0-1-001" / "checkpoints",
        root / "controllers" / "ppo",
    ):
        directory.mkdir(parents=True, exist_ok=True)
    shutil.copy2(PROJECT_ROOT / "configs/ppo_selection.toml", root / "configs/ppo_selection.toml")
    shutil.copy2(PROJECT_ROOT / "configs/ppo.toml", root / "configs/ppo.toml")
    shutil.copy2(
        PROJECT_ROOT / "controllers/ppo/controller.py", root / "controllers/ppo/controller.py"
    )
    shutil.copy2(
        PROJECT_ROOT / "tests/fixtures/ppo_unfinalized_config.toml",
        root / "controllers/ppo/config.toml",
    )

    training_config = root / "configs/ppo.toml"
    training_sha = hashlib.sha256(training_config.read_bytes()).hexdigest()
    identity = _identity(configuration_sha256=training_sha)
    checkpoint_path = (
        root / "runs" / "ppo" / "m7-formal-v0-1-001" / "checkpoints" / "update_00000020.pt"
    )
    checkpoint_path.write_bytes(b"retained checkpoint bytes")
    checkpoint_sha = hashlib.sha256(checkpoint_path.read_bytes()).hexdigest()
    parameter_sha = "6" * 64
    policy_bytes = b"canonical fake policy"
    policy_sha = hashlib.sha256(policy_bytes).hexdigest()
    selection = _selection_stub(
        checkpoint_sha256=checkpoint_sha,
        parameter_sha256=parameter_sha,
        policy_sha256=policy_sha,
        identity=identity,
    )
    selection["training_run"]["candidate_checkpoints"][0]["checkpoint"]["size_bytes"] = (
        checkpoint_path.stat().st_size
    )
    selection["training_run"]["candidate_checkpoints"][0]["inference_policy"]["size_bytes"] = len(
        policy_bytes
    )
    selection_config_path = root / "configs/ppo_selection.toml"
    selection["artifacts"] = {
        "selection_config": _record(
            "configs/ppo_selection.toml",
            hashlib.sha256(selection_config_path.read_bytes()).hexdigest(),
            selection_config_path.stat().st_size,
        ),
        "training_config": _record(
            "configs/ppo.toml", training_sha, training_config.stat().st_size
        ),
    }
    selection_report_path = root / "benchmarks/v0.1/m7_ppo_selection_report.json"
    selection_report_path.write_bytes(canonical_json_bytes(selection))

    revision = "f" * 40
    monkeypatch.setattr(
        export_cli,
        "_source_snapshot",
        lambda _root: {"revision": revision, "worktree_clean": True},
    )
    import controller_learning.rl.artifacts as artifacts_module
    import controller_learning.rl.controller_export as controller_export_module
    import controller_learning.rl.numpy_actor as numpy_actor_module
    import controller_learning.rl.selection as selection_module

    monkeypatch.setattr(
        selection_module, "validate_selection_report", lambda *_args, **_kwargs: None
    )
    loaded = SimpleNamespace(
        record=ArtifactRecord(
            relative_path="checkpoints/update_00000020.pt",
            sha256=checkpoint_sha,
            size_bytes=checkpoint_path.stat().st_size,
        ),
        metadata=SimpleNamespace(
            run_identity=identity,
            update_index=20,
            vector_steps=2_560,
            valid_transitions=2_000,
        ),
        payload={"model_state_dict": object()},
    )
    calls: dict[str, Any] = {}

    def fake_load(run_directory: Path, **kwargs: Any) -> Any:
        calls["load"] = (run_directory, kwargs)
        return loaded

    monkeypatch.setattr(artifacts_module, "load_published_training_checkpoint", fake_load)
    monkeypatch.setattr(
        selection_module, "torch_state_dict_sha256", lambda *_args, **_kwargs: parameter_sha
    )
    monkeypatch.setattr(
        numpy_actor_module, "numpy_actor_from_ppo_state_dict", lambda _state: object()
    )
    monkeypatch.setattr(
        numpy_actor_module, "canonical_numpy_actor_bytes", lambda _actor: policy_bytes
    )
    policy_evidence = SimpleNamespace(
        schema_version=1, sha256=policy_sha, size_bytes=len(policy_bytes)
    )
    result = SimpleNamespace(policy=policy_evidence)

    def fake_export(plugin_directory: Path, **kwargs: Any) -> Any:
        calls["export"] = (plugin_directory, kwargs)
        return result

    monkeypatch.setattr(controller_export_module, "export_ppo_controller", fake_export)

    def fake_verify(**kwargs: Any) -> dict[str, Any]:
        plugin = kwargs["plugin_directory"]
        (plugin / "config.toml").write_text("finalized = true\n", encoding="utf-8")
        (plugin / "metadata.json").write_bytes(b"{}\n")
        (plugin / "policy.npz").write_bytes(policy_bytes)

        def output_record(name: str) -> dict[str, Any]:
            path = plugin / name
            return _record(
                f"controllers/ppo/{name}",
                hashlib.sha256(path.read_bytes()).hexdigest(),
                path.stat().st_size,
            )

        return {
            "artifacts": {
                "config": output_record("config.toml"),
                "metadata": output_record("metadata.json"),
                "policy": output_record("policy.npz"),
            },
            "checkpoint": {
                "checkpoint_sha256": checkpoint_sha,
                "run_id": identity.run_id,
                "source_revision": identity.source_revision,
                "training_configuration_sha256": identity.configuration_sha256,
                "update_index": 20,
                "valid_transitions": 2_000,
                "vector_steps": 2_560,
            },
            "inference_only": {
                "contains_environment_state": False,
                "contains_optimizer_state": False,
                "contains_value_network": False,
            },
            "plugin_directory": "controllers/ppo",
            "runtime": "numpy",
        }

    monkeypatch.setattr(export_cli, "_verify_exported_controller", fake_verify)
    outputs = sorted(
        [
            EXPORT_REPORT_PATH,
            "controllers/ppo/config.toml",
            "controllers/ppo/metadata.json",
            "controllers/ppo/policy.npz",
        ]
    )
    monkeypatch.setattr(
        export_cli,
        "_source_snapshot_allowing_outputs",
        lambda *_args, **_kwargs: {
            "allowed_generated_output_paths": outputs,
            "observed_changed_paths": outputs,
            "only_allowed_generated_outputs": True,
            "revision": revision,
            "unexpected_changed_paths": [],
        },
    )

    guard = export_cli.ExportAssetAccessGuard(
        official_track_root=root / "controller_learning/assets/tracks",
        track_cache_root=root / ".track-cache",
    )
    # The isolated subprocess test below proves the real install path and wrappers.
    guard._installed = True
    guard._unaudited_mutation_wrappers_installed = True
    outcome = export_cli.run_export(
        export_cli.ExportOptions(),
        access_guard=guard,
        project_root=root,
        torch_module=object(),
    )

    assert outcome == {
        "checkpoint_sha256": checkpoint_sha,
        "export_report": EXPORT_REPORT_PATH,
        "policy_sha256": policy_sha,
        "selected_update": 20,
    }
    assert calls["load"][1]["update_index"] == 20
    assert calls["load"][1]["expected_identity"] == identity
    assert calls["export"][1]["loaded_checkpoint"] is loaded
    assert calls["export"][1]["training_config_path"] == training_config
    assert calls["export"][1]["staging_directory"] == (
        root / export_cli.EXPORT_TRANSACTION_DIRECTORY / "staging"
    )
    assert not (root / export_cli.EXPORT_TRANSACTION_DIRECTORY).exists()
    assert not any(
        path.name.endswith((".tmp", ".recovery"))
        for directory in (root / "controllers/ppo", root / "benchmarks/v0.1")
        for path in directory.iterdir()
    )
    published = read_strict_json(root, EXPORT_REPORT_PATH)
    validate_export_report(published)
