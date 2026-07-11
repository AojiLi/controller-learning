"""CPU protocol tests for the post-selection ordinary Controller CLI."""

from __future__ import annotations

import stat
import subprocess
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).parents[3]


def _initialize_clean_git_repository(root: Path) -> dict[str, object]:
    from scripts import benchmark_m7_ppo_controller as benchmark

    (root / ".gitignore").write_text("/runs/\n", encoding="utf-8")
    for command in (
        ("git", "init", "-q"),
        ("git", "config", "user.email", "tests@example.invalid"),
        ("git", "config", "user.name", "Controller Learning Tests"),
        ("git", "add", "."),
        ("git", "commit", "-qm", "test baseline"),
    ):
        subprocess.run(command, cwd=root, check=True, capture_output=True, text=True)
    return benchmark._source_snapshot(root)


def test_cli_import_sets_allocator_policy_without_importing_gpu_stacks() -> None:
    completed = subprocess.run(
        (
            sys.executable,
            "-c",
            (
                "import os,sys; import scripts.benchmark_m7_ppo_controller; "
                "assert os.environ['CUDA_DEVICE_ORDER']=='PCI_BUS_ID'; "
                "assert os.environ['XLA_PYTHON_CLIENT_PREALLOCATE']=='false'; "
                "assert 'jax' not in sys.modules; assert 'torch' not in sys.modules"
            ),
        ),
        cwd=PROJECT_ROOT,
        check=False,
        capture_output=True,
        text=True,
    )
    assert completed.returncode == 0, completed.stderr


def test_run_command_preserves_git_porcelain_leading_status_column() -> None:
    from scripts import benchmark_m7_ppo_controller as benchmark

    output = benchmark._run_command(
        (sys.executable, "-c", "import sys; sys.stdout.write(' M tracked\\n')"),
        cwd=PROJECT_ROOT,
    )

    assert output == " M tracked"


def test_guard_blocks_non_validation_assets_before_read(tmp_path: Path) -> None:
    official = tmp_path / "official" / "v0.1"
    cache = tmp_path / "cache"
    official.mkdir(parents=True)
    cache.mkdir()
    validation_manifest = official / "validation.json"
    validation_asset = official / "validation.npz"
    forbidden_test = official / "test.npz"
    for path in (validation_manifest, validation_asset, forbidden_test):
        path.write_bytes(b"asset")
    program = f"""
import os
from pathlib import Path
from scripts.benchmark_m7_ppo_controller import (
    ForbiddenControllerEvaluationAssetAccessError,
    OfficialValidationAssetAccessGuard,
)
preopened_fd = os.open(Path({str(validation_asset)!r}), os.O_RDONLY)
guard = OfficialValidationAssetAccessGuard(
    official_track_root=Path({str(official.parent)!r}),
    validation_manifest=Path({str(validation_manifest)!r}),
    validation_asset=Path({str(validation_asset)!r}),
    track_cache_root=Path({str(cache)!r}),
)
guard.install()
try:
    guard._audit('open', (preopened_fd, 'w', os.O_WRONLY))
except ForbiddenControllerEvaluationAssetAccessError:
    pass
else:
    raise AssertionError('protected open fd was not blocked')
finally:
    os.close(preopened_fd)
try:
    Path({str(validation_manifest)!r}).read_bytes()
except ForbiddenControllerEvaluationAssetAccessError:
    pass
else:
    raise AssertionError('Validation opened before one-way phase transition')
try:
    Path({str(validation_asset)!r}).rename(Path({str(official / "renamed.npz")!r}))
except ForbiddenControllerEvaluationAssetAccessError:
    pass
else:
    raise AssertionError('protected rename was not blocked')
try:
    Path({str(validation_asset)!r}).unlink()
except ForbiddenControllerEvaluationAssetAccessError:
    pass
else:
    raise AssertionError('protected unlink was not blocked')
relative_asset = os.path.relpath(Path({str(validation_asset)!r}), Path.cwd())
try:
    os.remove(relative_asset)
except ForbiddenControllerEvaluationAssetAccessError:
    pass
else:
    raise AssertionError('relative protected remove with default dir_fd was not blocked')
try:
    os.setxattr(Path({str(validation_asset)!r}), 'user.controller_learning_test', b'x')
except ForbiddenControllerEvaluationAssetAccessError:
    pass
else:
    raise AssertionError('protected setxattr was not blocked')
try:
    os.removexattr(Path({str(validation_asset)!r}), 'user.controller_learning_test')
except ForbiddenControllerEvaluationAssetAccessError:
    pass
else:
    raise AssertionError('protected removexattr was not blocked')
blocked_fifo = Path({str(official / "blocked.fifo")!r})
try:
    os.mkfifo(blocked_fifo)
except ForbiddenControllerEvaluationAssetAccessError:
    pass
else:
    raise AssertionError('protected mkfifo was not blocked')
assert not blocked_fifo.exists()
blocked_node = Path({str(official / "blocked.node")!r})
try:
    os.mknod(blocked_node)
except ForbiddenControllerEvaluationAssetAccessError:
    pass
else:
    raise AssertionError('protected mknod was not blocked')
assert not blocked_node.exists()
try:
    Path({str(forbidden_test)!r}).read_bytes()
except ForbiddenControllerEvaluationAssetAccessError:
    pass
else:
    raise AssertionError('forbidden asset opened')
evidence = guard.evidence(validation_loaded=False)
assert evidence['validation_reads_enabled'] is False
assert evidence['denied_event_count'] == 10
assert evidence['denied_mutation_event_count'] == 7
assert evidence['denied_mutation_event_types'] == {{
    'os.mkfifo': 1,
    'os.mknod': 1,
    'os.remove': 2,
    'os.removexattr': 1,
    'os.rename': 1,
    'os.setxattr': 1,
}}
try:
    guard.enable_validation_reads()
except RuntimeError:
    pass
else:
    raise AssertionError('denied activity did not permanently close the Validation phase')
assert guard.evidence(validation_loaded=False)['validation_reads_enabled'] is False
print('blocked')
"""
    completed = subprocess.run(
        (sys.executable, "-c", program),
        cwd=PROJECT_ROOT,
        check=False,
        capture_output=True,
        text=True,
    )
    assert completed.returncode == 0, completed.stderr
    assert completed.stdout.strip() == "blocked"


def test_guard_allows_validation_only_after_clean_preflight(tmp_path: Path) -> None:
    official = tmp_path / "official" / "v0.1"
    cache = tmp_path / "cache"
    official.mkdir(parents=True)
    cache.mkdir()
    validation_manifest = official / "validation.json"
    validation_asset = official / "validation.npz"
    validation_manifest.write_bytes(b"manifest")
    validation_asset.write_bytes(b"asset")
    program = f"""
from pathlib import Path
from scripts.benchmark_m7_ppo_controller import OfficialValidationAssetAccessGuard
guard = OfficialValidationAssetAccessGuard(
    official_track_root=Path({str(official.parent)!r}),
    validation_manifest=Path({str(validation_manifest)!r}),
    validation_asset=Path({str(validation_asset)!r}),
    track_cache_root=Path({str(cache)!r}),
)
guard.install()
before = guard.evidence(validation_loaded=False)
assert before['denied_event_count'] == 0
assert before['denied_mutation_event_count'] == 0
assert before['validation_reads_enabled'] is False
guard.enable_validation_reads()
assert Path({str(validation_manifest)!r}).read_bytes() == b'manifest'
assert Path({str(validation_asset)!r}).read_bytes() == b'asset'
after = guard.evidence(validation_loaded=True)
assert after['denied_event_count'] == 0
assert after['denied_mutation_event_count'] == 0
assert after['validation_reads_enabled'] is True
print('allowed')
"""
    completed = subprocess.run(
        (sys.executable, "-c", program),
        cwd=PROJECT_ROOT,
        check=False,
        capture_output=True,
        text=True,
    )
    assert completed.returncode == 0, completed.stderr
    assert completed.stdout.strip() == "allowed"


def test_persistent_transaction_recovers_interrupted_publication_for_absent_outputs(
    tmp_path: Path,
) -> None:
    from scripts import benchmark_m7_ppo_controller as benchmark

    for relative in benchmark.FORMAL_OUTPUT_PATHS:
        (tmp_path / relative).parent.mkdir(parents=True, exist_ok=True)
    before = _initialize_clean_git_repository(tmp_path)
    transaction = benchmark._PersistentOutputTransaction(tmp_path)
    transaction.__enter__()
    assert (transaction.transaction_directory / "READY").is_file()
    for relative in benchmark.FORMAL_OUTPUT_PATHS[:2]:
        transaction.publish_bytes(relative, f"partial:{relative}".encode())
    staged = transaction.prepare_staged_output(
        benchmark.FORMAL_OUTPUT_PATHS[2],
        b"fsynced but not replaced",
    )
    assert staged.parent == transaction.staging_directory
    assert staged.is_file()
    benchmark_directory = tmp_path / "benchmarks/v0.1"
    assert {path.name for path in benchmark_directory.iterdir()} == {
        Path(relative).name for relative in benchmark.FORMAL_OUTPUT_PATHS[:2]
    }

    recovery = benchmark._PersistentOutputTransaction(tmp_path)
    assert recovery.recover_startup() == "restored_ready_transaction"
    assert all(not (tmp_path / relative).exists() for relative in benchmark.FORMAL_OUTPUT_PATHS)
    assert not any(path.is_file() for path in benchmark_directory.iterdir())
    assert not recovery.transaction_directory.exists()
    assert not recovery.cleanup_directory.exists()
    assert benchmark._source_snapshot(tmp_path) == before


def test_persistent_transaction_recovers_interrupted_publication_for_existing_outputs(
    tmp_path: Path,
) -> None:
    from scripts import benchmark_m7_ppo_controller as benchmark

    originals: dict[str, tuple[bytes, int]] = {}
    for index, relative in enumerate(benchmark.FORMAL_OUTPUT_PATHS):
        output = tmp_path / relative
        output.parent.mkdir(parents=True, exist_ok=True)
        content = f"original:{index}".encode()
        mode = (0o600, 0o640, 0o644)[index]
        output.write_bytes(content)
        output.chmod(mode)
        originals[relative] = (content, mode)
    before = _initialize_clean_git_repository(tmp_path)

    transaction = benchmark._PersistentOutputTransaction(tmp_path)
    transaction.__enter__()
    for relative in benchmark.FORMAL_OUTPUT_PATHS[:2]:
        transaction.publish_bytes(relative, b"partial replacement", mode=0o777)
    staged = transaction.prepare_staged_output(
        benchmark.FORMAL_OUTPUT_PATHS[2],
        b"prepared report without replace",
    )
    assert staged.is_file()
    benchmark_directory = tmp_path / "benchmarks/v0.1"
    assert {path.name for path in benchmark_directory.iterdir()} == {
        Path(relative).name for relative in benchmark.FORMAL_OUTPUT_PATHS
    }

    recovery = benchmark._PersistentOutputTransaction(tmp_path)
    assert recovery.recover_startup() == "restored_ready_transaction"
    for relative, (content, mode) in originals.items():
        output = tmp_path / relative
        assert output.read_bytes() == content
        assert stat.S_IMODE(output.stat().st_mode) == mode
    assert not recovery.transaction_directory.exists()
    assert not recovery.cleanup_directory.exists()
    assert {path.name for path in benchmark_directory.iterdir()} == {
        Path(relative).name for relative in benchmark.FORMAL_OUTPUT_PATHS
    }
    assert benchmark._source_snapshot(tmp_path) == before


def test_canonical_evaluation_config_path_rejects_an_equivalent_alternate(tmp_path: Path) -> None:
    from scripts import benchmark_m7_ppo_controller as benchmark

    canonical = tmp_path / benchmark.DEFAULT_CONFIG
    canonical.parent.mkdir(parents=True)
    canonical.write_bytes(b"frozen")
    alternate = tmp_path / "configs/equivalent.toml"
    alternate.write_bytes(b"frozen")
    assert (
        benchmark._canonical_evaluation_config_path(tmp_path, benchmark.DEFAULT_CONFIG) == canonical
    )
    try:
        benchmark._canonical_evaluation_config_path(tmp_path, alternate.relative_to(tmp_path))
    except RuntimeError:
        pass
    else:
        raise AssertionError("an alternate formal config path was accepted")


def test_source_uses_only_public_controller_paths_and_one_pixi_task() -> None:
    source = (PROJECT_ROOT / "scripts/benchmark_m7_ppo_controller.py").read_text(encoding="utf-8")
    for required in (
        "evaluate_track_batch(",
        "record_controller_episode(",
        "validate_export_report(export_report)",
        "max_steps=config.max_episode_steps",
        'reset_options={"track_index": selected_replay_index}',
        "transaction.publish_bytes(",
    ):
        assert required in source
    for forbidden in (
        "load_verified_train_pool",
        "PpoUpdater",
        "controller_learning.rl.trainer",
        ".backward(",
        "optimizer.step",
        "atomic_write_bytes(root",
        "atomic_write_json(root",
        "_restore_output_snapshots",
    ):
        assert forbidden not in source
    assert source.index("access_guard.enable_validation_reads()") < source.index(
        "load_verified_validation_pool(project)"
    )
    assert source.index('memory.sample("before_environment_create")') < source.index(
        "access_guard.enable_validation_reads()"
    )
    assert "TemporaryDirectory" in source
    run_source = source[source.index("def run_benchmark(") :]
    assert run_source.index("_canonical_evaluation_config_path") < run_source.index("import jax")
    assert run_source.index("recover_startup()") < run_source.index("_source_snapshot(root)")
    assert "_PersistentOutputTransaction" in run_source

    pyproject = (PROJECT_ROOT / "pyproject.toml").read_text(encoding="utf-8")
    assert pyproject.count("benchmark-m7-ppo-controller =") == 1
    assert "python scripts/benchmark_m7_ppo_controller.py" in pyproject
    assert pyproject.count("export-m7-ppo-controller =") == 1
    assert (
        pyproject.index("benchmark-m7-ppo =")
        < pyproject.index("export-m7-ppo-controller =")
        < pyproject.index("benchmark-m7-ppo-controller =")
    )
