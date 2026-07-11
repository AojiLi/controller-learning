"""Subprocess tests for the irreversible M8 Test-only filesystem guard."""

from __future__ import annotations

import os
import subprocess
import sys
import threading
from pathlib import Path
from types import SimpleNamespace

import pytest

import controller_learning.evaluation.test_access as test_access_module
from controller_learning.evaluation.test_access import (
    ForbiddenFinalEvaluationAssetAccessError,
    M8TestAssetAccessGuard,
)


def _asset_tree(root: Path) -> tuple[Path, Path, Path, Path, Path]:
    official = root / "official" / "v0.1"
    cache = root / "cache"
    official.mkdir(parents=True)
    cache.mkdir()
    test_manifest = official / "test.json"
    test_asset = official / "test.npz"
    validation = official / "validation.npz"
    for path in (test_manifest, test_asset, validation):
        path.write_bytes(b"asset")
    return official.parent, cache, test_manifest, test_asset, validation


def _fake_nvidia_smi(root: Path) -> Path:
    executable = root / "nvidia-smi"
    executable.write_bytes(b"#!/bin/sh\nexit 0\n")
    executable.chmod(0o755)
    return executable.resolve(strict=True)


def _locked_guard(tmp_path: Path) -> tuple[M8TestAssetAccessGuard, Path]:
    official, cache, manifest, asset, _validation = _asset_tree(tmp_path)
    executable = _fake_nvidia_smi(tmp_path)
    guard = M8TestAssetAccessGuard(
        official_track_root=official,
        test_manifest=manifest,
        test_asset=asset,
        track_cache_root=cache,
    )
    guard._installed = True
    guard._test_reads_enabled = True
    guard._all_track_reads_forbidden = True
    guard._nvidia_smi_identity = guard._capture_executable_identity(
        executable,
        require_trusted_ownership=False,
    )
    return guard, executable


def test_post_load_process_gate_rejects_direct_fixed_vram_query(tmp_path: Path) -> None:
    guard, executable = _locked_guard(tmp_path)
    command = (str(executable), *guard._POST_LOAD_NVIDIA_SMI_ARGUMENTS)
    environment = dict(guard._POST_LOAD_NVIDIA_SMI_ENVIRONMENT)

    with pytest.raises(ForbiddenFinalEvaluationAssetAccessError, match="private memory-query"):
        guard._audit("subprocess.Popen", (str(executable), command, None, environment))

    guard._denied_event_count = 0
    with pytest.raises(ForbiddenFinalEvaluationAssetAccessError, match="private memory-query"):
        guard._audit("os.posix_spawn", (str(executable), command, environment))


def test_private_memory_query_uses_one_fixed_process_capability(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    guard, executable = _locked_guard(tmp_path)
    command = (str(executable), *guard._POST_LOAD_NVIDIA_SMI_ARGUMENTS)
    captured: dict[str, object] = {}

    monkeypatch.setattr(M8TestAssetAccessGuard, "assert_audit_hook_active", lambda _self: None)

    def run(argv: tuple[str, ...], **kwargs: object) -> SimpleNamespace:
        captured["argv"] = tuple(argv)
        captured.update(kwargs)
        guard._audit(
            "subprocess.Popen",
            (str(executable), tuple(argv), kwargs["cwd"], kwargs["env"]),
        )
        return SimpleNamespace(stdout="query output\n")

    monkeypatch.setattr(test_access_module.subprocess, "run", run)

    assert guard.run_frozen_memory_query(command) == "query output\n"
    assert set(captured) == {
        "argv",
        "capture_output",
        "check",
        "close_fds",
        "cwd",
        "env",
        "shell",
        "start_new_session",
        "stdin",
        "text",
        "timeout",
    }
    assert captured["argv"] == command
    assert captured["stdin"] == subprocess.DEVNULL
    assert captured["capture_output"] is True
    assert captured["cwd"] is None
    assert captured["env"] == dict(guard._POST_LOAD_NVIDIA_SMI_ENVIRONMENT)
    assert captured["check"] is True
    assert captured["close_fds"] is True
    assert captured["shell"] is False
    assert captured["start_new_session"] is True
    assert captured["text"] is True
    assert captured["timeout"] == guard._POST_LOAD_NVIDIA_SMI_TIMEOUT_S
    assert guard._process_context.capability is None


def test_private_memory_query_capability_is_thread_local(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    guard, executable = _locked_guard(tmp_path)
    command = (str(executable), *guard._POST_LOAD_NVIDIA_SMI_ARGUMENTS)
    environment = dict(guard._POST_LOAD_NVIDIA_SMI_ENVIRONMENT)
    denied: list[BaseException] = []
    monkeypatch.setattr(M8TestAssetAccessGuard, "assert_audit_hook_active", lambda _self: None)

    def run(argv: tuple[str, ...], **_kwargs: object) -> SimpleNamespace:
        def direct_other_thread() -> None:
            try:
                guard._audit(
                    "subprocess.Popen",
                    (str(executable), command, None, environment),
                )
            except BaseException as error:
                denied.append(error)

        thread = threading.Thread(target=direct_other_thread)
        thread.start()
        thread.join()
        guard._audit("subprocess.Popen", (str(executable), tuple(argv), None, environment))
        return SimpleNamespace(stdout="query output\n")

    monkeypatch.setattr(test_access_module.subprocess, "run", run)

    assert guard.run_frozen_memory_query(command) == "query output\n"
    assert len(denied) == 1
    assert isinstance(denied[0], ForbiddenFinalEvaluationAssetAccessError)


@pytest.mark.parametrize("event", ("subprocess.Popen", "os.posix_spawn"))
@pytest.mark.parametrize("environment_kind", ("inherited", "ambient", "extra"))
def test_post_load_process_gate_rejects_nonminimal_environment(
    tmp_path: Path,
    event: str,
    environment_kind: str,
) -> None:
    guard, executable = _locked_guard(tmp_path)
    command = (str(executable), *guard._POST_LOAD_NVIDIA_SMI_ARGUMENTS)
    environment: object
    if environment_kind == "inherited":
        environment = None
    elif environment_kind == "ambient":
        environment = dict(os.environ)
    else:
        environment = dict(guard._POST_LOAD_NVIDIA_SMI_ENVIRONMENT)
        environment["LD_PRELOAD"] = "/tmp/untrusted.so"
    if event == "subprocess.Popen":
        arguments = (str(executable), command, None, environment)
    else:
        arguments = (str(executable), command, environment)

    with pytest.raises(ForbiddenFinalEvaluationAssetAccessError):
        guard._audit(event, arguments)

    assert guard._denied_event_count == 1


@pytest.mark.parametrize(
    ("event", "arguments"),
    [
        ("subprocess.Popen", ("git", ("git", "status"), None, None)),
        (
            "subprocess.Popen",
            (
                "nvidia-smi",
                ("nvidia-smi", "--query-gpu=uuid"),
                None,
                None,
            ),
        ),
        ("subprocess.Popen", ("nvidia-smi", (), "/tmp", None)),
        ("os.system", (b"git status",)),
        ("os.fork", ()),
        ("os.exec", ("git", ("git", "status"), None)),
        ("os.posix_spawn", ("git", ("git", "status"), dict(os.environ))),
        ("os.startfile", ("malware.exe",)),
        ("os.startfile/2", ("malware.exe", None, None, None, None)),
    ],
)
def test_post_load_process_gate_rejects_every_other_process_path(
    tmp_path: Path,
    event: str,
    arguments: tuple[object, ...],
) -> None:
    guard, _executable = _locked_guard(tmp_path)
    with pytest.raises(ForbiddenFinalEvaluationAssetAccessError):
        guard._audit(event, arguments)
    assert guard._denied_event_count == 1


def test_guard_allows_only_test_after_one_way_transition(tmp_path: Path) -> None:
    official, cache, manifest, asset, validation = _asset_tree(tmp_path)
    nvidia_smi = _fake_nvidia_smi(tmp_path)
    program = f"""
from pathlib import Path
from controller_learning.evaluation.test_access import (
    ForbiddenFinalEvaluationAssetAccessError,
    M8TestAssetAccessGuard,
)
guard = M8TestAssetAccessGuard(
    official_track_root=Path({str(official)!r}),
    test_manifest=Path({str(manifest)!r}),
    test_asset=Path({str(asset)!r}),
    track_cache_root=Path({str(cache)!r}),
)
guard.install()
guard._nvidia_smi_identity = guard._capture_executable_identity(
    Path({str(nvidia_smi)!r}), require_trusted_ownership=False
)
assert guard.evidence(test_loaded=False)['test_reads_enabled'] is False
assert guard.evidence(test_loaded=False)['all_track_reads_forbidden'] is False
try:
    Path({str(manifest)!r}).read_bytes()
except ForbiddenFinalEvaluationAssetAccessError:
    pass
else:
    raise AssertionError('Test opened before TEST_BOUND')
try:
    guard.enable_test_reads()
except RuntimeError:
    pass
else:
    raise AssertionError('a denied event did not latch the guard')
print('latched')
"""
    completed = subprocess.run(
        (sys.executable, "-c", program),
        check=False,
        capture_output=True,
        text=True,
    )
    assert completed.returncode == 0, completed.stderr
    assert completed.stdout.strip() == "latched"

    clean_program = f"""
from pathlib import Path
from controller_learning.evaluation.test_access import (
    ForbiddenFinalEvaluationAssetAccessError,
    M8TestAssetAccessGuard,
)
guard = M8TestAssetAccessGuard(
    official_track_root=Path({str(official)!r}),
    test_manifest=Path({str(manifest)!r}),
    test_asset=Path({str(asset)!r}),
    track_cache_root=Path({str(cache)!r}),
)
guard.install()
guard._nvidia_smi_identity = guard._capture_executable_identity(
    Path({str(nvidia_smi)!r}), require_trusted_ownership=False
)
guard.enable_test_reads()
assert Path({str(manifest)!r}).read_bytes() == b'asset'
assert Path({str(asset)!r}).read_bytes() == b'asset'
try:
    Path({str(validation)!r}).read_bytes()
except ForbiddenFinalEvaluationAssetAccessError:
    pass
else:
    raise AssertionError('Validation was not blocked')
evidence = guard.evidence(test_loaded=True)
assert evidence['opened_splits'] == ['test']
assert evidence['denied_event_count'] == 1
assert evidence['train_opened'] is False
assert evidence['validation_opened'] is False
print('test-only')
"""
    completed = subprocess.run(
        (sys.executable, "-c", clean_program),
        check=False,
        capture_output=True,
        text=True,
    )
    assert completed.returncode == 0, completed.stderr
    assert completed.stdout.strip() == "test-only"

    locked_program = f"""
from pathlib import Path
from controller_learning.evaluation.test_access import (
    ForbiddenFinalEvaluationAssetAccessError,
    M8TestAssetAccessGuard,
)
guard = M8TestAssetAccessGuard(
    official_track_root=Path({str(official)!r}),
    test_manifest=Path({str(manifest)!r}),
    test_asset=Path({str(asset)!r}),
    track_cache_root=Path({str(cache)!r}),
)
guard.install()
guard._nvidia_smi_identity = guard._capture_executable_identity(
    Path({str(nvidia_smi)!r}), require_trusted_ownership=False
)
guard.enable_test_reads()
Path({str(manifest)!r}).read_bytes()
Path({str(asset)!r}).read_bytes()
guard.forbid_all_track_reads()
evidence = guard.evidence(test_loaded=True)
assert evidence['all_track_reads_forbidden'] is True
assert evidence['denied_event_count'] == 0
try:
    Path({str(asset)!r}).read_bytes()
except ForbiddenFinalEvaluationAssetAccessError:
    pass
else:
    raise AssertionError('post-load Test read was not blocked')
print('reads-closed')
"""
    completed = subprocess.run(
        (sys.executable, "-c", locked_program),
        check=False,
        capture_output=True,
        text=True,
    )
    assert completed.returncode == 0, completed.stderr
    assert completed.stdout.strip() == "reads-closed"


def test_guard_blocks_protected_mutations(tmp_path: Path) -> None:
    official, cache, manifest, asset, _validation = _asset_tree(tmp_path)
    program = f"""
import os
from pathlib import Path
from controller_learning.evaluation.test_access import (
    ForbiddenFinalEvaluationAssetAccessError,
    M8TestAssetAccessGuard,
)
guard = M8TestAssetAccessGuard(
    official_track_root=Path({str(official)!r}),
    test_manifest=Path({str(manifest)!r}),
    test_asset=Path({str(asset)!r}),
    track_cache_root=Path({str(cache)!r}),
)
guard.install()
for operation in (
    lambda: Path({str(asset)!r}).unlink(),
    lambda: Path({str(asset)!r}).rename(Path({str(asset.with_suffix(".other"))!r})),
    lambda: os.mkfifo(Path({str(asset.with_suffix(".fifo"))!r})),
    lambda: os.mknod(Path({str(asset.with_suffix(".node"))!r})),
):
    try:
        operation()
    except ForbiddenFinalEvaluationAssetAccessError:
        pass
    else:
        raise AssertionError('protected mutation was not blocked')
evidence = guard.evidence(test_loaded=False)
assert evidence['denied_event_count'] == 4
assert evidence['denied_mutation_event_count'] == 4
assert not Path({str(asset.with_suffix(".fifo"))!r}).exists()
assert not Path({str(asset.with_suffix(".node"))!r}).exists()
print('mutations-blocked')
"""
    completed = subprocess.run(
        (sys.executable, "-c", program),
        check=False,
        capture_output=True,
        text=True,
    )
    assert completed.returncode == 0, completed.stderr
    assert completed.stdout.strip() == "mutations-blocked"


def test_guard_blocks_absolute_and_descriptor_protected_directory_enumeration(
    tmp_path: Path,
) -> None:
    official, cache, manifest, asset, _validation = _asset_tree(tmp_path)
    program = f"""
import os
from pathlib import Path
from controller_learning.evaluation.test_access import (
    ForbiddenFinalEvaluationAssetAccessError,
    M8TestAssetAccessGuard,
)
track_directory = Path({str(manifest.parent)!r})
track_fd = os.open(track_directory, os.O_RDONLY | os.O_DIRECTORY)
cache_fd = os.open(Path({str(cache)!r}), os.O_RDONLY | os.O_DIRECTORY)
guard = M8TestAssetAccessGuard(
    official_track_root=Path({str(official)!r}),
    test_manifest=Path({str(manifest)!r}),
    test_asset=Path({str(asset)!r}),
    track_cache_root=Path({str(cache)!r}),
)
guard.install()
for operation in (
    lambda: os.listdir(track_directory),
    lambda: os.scandir(track_directory),
    lambda: os.listdir(track_fd),
    lambda: os.scandir(cache_fd),
    lambda: os.chdir(track_directory),
    lambda: os.fchdir(track_fd),
):
    try:
        operation()
    except ForbiddenFinalEvaluationAssetAccessError:
        pass
    else:
        raise AssertionError('protected directory enumeration was not blocked')
assert guard.evidence(test_loaded=False)['denied_event_count'] == 6
os.close(track_fd)
os.close(cache_fd)
print('enumeration-blocked')
"""
    completed = subprocess.run(
        (sys.executable, "-c", program),
        check=False,
        capture_output=True,
        text=True,
    )
    assert completed.returncode == 0, completed.stderr
    assert completed.stdout.strip() == "enumeration-blocked"


def test_guard_resolves_default_enumeration_path_from_protected_cwd(tmp_path: Path) -> None:
    official, cache, manifest, asset, _validation = _asset_tree(tmp_path)
    program = f"""
import os
from pathlib import Path
from controller_learning.evaluation.test_access import (
    ForbiddenFinalEvaluationAssetAccessError,
    M8TestAssetAccessGuard,
)
original = Path.cwd()
os.chdir(Path({str(manifest.parent)!r}))
guard = M8TestAssetAccessGuard(
    official_track_root=Path({str(official)!r}),
    test_manifest=Path({str(manifest)!r}),
    test_asset=Path({str(asset)!r}),
    track_cache_root=Path({str(cache)!r}),
)
guard.install()
for operation in (lambda: os.listdir(), lambda: os.scandir()):
    try:
        operation()
    except ForbiddenFinalEvaluationAssetAccessError:
        pass
    else:
        raise AssertionError('default-cwd protected enumeration was not blocked')
os.chdir(original)
assert guard.evidence(test_loaded=False)['denied_event_count'] == 2
print('default-enumeration-blocked')
"""
    completed = subprocess.run(
        (sys.executable, "-c", program),
        check=False,
        capture_output=True,
        text=True,
    )
    assert completed.returncode == 0, completed.stderr
    assert completed.stdout.strip() == "default-enumeration-blocked"


def test_guard_rejects_freezing_executable_below_untrusted_temp_parent(
    tmp_path: Path,
) -> None:
    official, cache, manifest, asset, _validation = _asset_tree(tmp_path)
    nvidia_smi = _fake_nvidia_smi(tmp_path)
    program = f"""
from pathlib import Path
from controller_learning.evaluation.test_access import M8TestAssetAccessGuard
guard = M8TestAssetAccessGuard(
    official_track_root=Path({str(official)!r}),
    test_manifest=Path({str(manifest)!r}),
    test_asset=Path({str(asset)!r}),
    track_cache_root=Path({str(cache)!r}),
)
guard.install()
try:
    guard.freeze_nvidia_smi_executable(Path({str(nvidia_smi)!r}))
except ValueError as error:
    assert 'owned by root' in str(error) or 'group- or world-writable' in str(error)
else:
    raise AssertionError('an executable below an untrusted temporary parent was frozen')
print('untrusted-executable-rejected')
"""
    completed = subprocess.run(
        (sys.executable, "-c", program),
        check=False,
        capture_output=True,
        text=True,
    )
    assert completed.returncode == 0, completed.stderr
    assert completed.stdout.strip() == "untrusted-executable-rejected"


def test_guard_self_check_rejects_synthetic_installed_state_without_hook(
    tmp_path: Path,
) -> None:
    official, cache, manifest, asset, _validation = _asset_tree(tmp_path)
    guard = M8TestAssetAccessGuard(
        official_track_root=official,
        test_manifest=manifest,
        test_asset=asset,
        track_cache_root=cache,
    )
    guard._installed = True

    with pytest.raises(RuntimeError, match="did not answer"):
        guard.assert_audit_hook_active()


def test_descriptor_relative_opens_receive_the_same_test_asset_policy(
    tmp_path: Path,
) -> None:
    official, cache, manifest, asset, validation = _asset_tree(tmp_path)
    nvidia_smi = _fake_nvidia_smi(tmp_path)
    program = f"""
import os
from pathlib import Path
from controller_learning.evaluation.test_access import (
    ForbiddenFinalEvaluationAssetAccessError,
    M8TestAssetAccessGuard,
)
directory_fd = os.open(Path({str(manifest.parent)!r}), os.O_RDONLY | os.O_DIRECTORY)
guard = M8TestAssetAccessGuard(
    official_track_root=Path({str(official)!r}),
    test_manifest=Path({str(manifest)!r}),
    test_asset=Path({str(asset)!r}),
    track_cache_root=Path({str(cache)!r}),
)
guard.install()
guard._nvidia_smi_identity = guard._capture_executable_identity(
    Path({str(nvidia_smi)!r}), require_trusted_ownership=False
)
guard.enable_test_reads()
for name in ({manifest.name!r}, {asset.name!r}):
    descriptor = os.open(name, os.O_RDONLY, dir_fd=directory_fd)
    try:
        assert os.read(descriptor, 5) == b'asset'
    finally:
        os.close(descriptor)
try:
    os.open({validation.name!r}, os.O_RDONLY, dir_fd=directory_fd)
except ForbiddenFinalEvaluationAssetAccessError:
    pass
else:
    raise AssertionError('descriptor-relative Validation access was not blocked')
evidence = guard.evidence(test_loaded=True)
assert evidence['open_event_counts'] == {{
    'official_test_asset': 1,
    'official_test_manifest': 1,
}}
os.close(directory_fd)
print('descriptor-policy-enforced')
"""
    completed = subprocess.run(
        (sys.executable, "-c", program),
        check=False,
        capture_output=True,
        text=True,
    )
    assert completed.returncode == 0, completed.stderr
    assert completed.stdout.strip() == "descriptor-policy-enforced"


def test_process_creation_is_disabled_throughout_test_asset_loading(
    tmp_path: Path,
) -> None:
    official, cache, manifest, asset, _validation = _asset_tree(tmp_path)
    nvidia_smi = _fake_nvidia_smi(tmp_path)
    program = f"""
import subprocess
from pathlib import Path
from controller_learning.evaluation.test_access import (
    ForbiddenFinalEvaluationAssetAccessError,
    M8TestAssetAccessGuard,
)
guard = M8TestAssetAccessGuard(
    official_track_root=Path({str(official)!r}),
    test_manifest=Path({str(manifest)!r}),
    test_asset=Path({str(asset)!r}),
    track_cache_root=Path({str(cache)!r}),
)
guard.install()
guard._nvidia_smi_identity = guard._capture_executable_identity(
    Path({str(nvidia_smi)!r}), require_trusted_ownership=False
)
guard.enable_test_reads()
try:
    subprocess.run(
        (str(Path({str(nvidia_smi)!r})), *guard._POST_LOAD_NVIDIA_SMI_ARGUMENTS),
        check=True,
        env=dict(guard._POST_LOAD_NVIDIA_SMI_ENVIRONMENT),
    )
except ForbiddenFinalEvaluationAssetAccessError:
    pass
else:
    raise AssertionError('process creation was allowed during Test asset loading')
assert guard._denied_event_count == 1
print('load-process-blocked')
"""
    completed = subprocess.run(
        (sys.executable, "-c", program),
        check=False,
        capture_output=True,
        text=True,
    )
    assert completed.returncode == 0, completed.stderr
    assert completed.stdout.strip() == "load-process-blocked"


def test_post_load_process_creation_uses_only_frozen_absolute_query(
    tmp_path: Path,
) -> None:
    official, cache, manifest, asset, _validation = _asset_tree(tmp_path)
    nvidia_smi = _fake_nvidia_smi(tmp_path)
    program = f"""
import subprocess
from pathlib import Path
from controller_learning.evaluation.test_access import (
    ForbiddenFinalEvaluationAssetAccessError,
    M8TestAssetAccessGuard,
)
guard = M8TestAssetAccessGuard(
    official_track_root=Path({str(official)!r}),
    test_manifest=Path({str(manifest)!r}),
    test_asset=Path({str(asset)!r}),
    track_cache_root=Path({str(cache)!r}),
)
guard.install()
guard._nvidia_smi_identity = guard._capture_executable_identity(
    Path({str(nvidia_smi)!r}), require_trusted_ownership=False
)
guard.enable_test_reads()
Path({str(manifest)!r}).read_bytes()
Path({str(asset)!r}).read_bytes()
guard.forbid_all_track_reads()
absolute = str(Path({str(nvidia_smi)!r}))
command = (absolute, *guard._POST_LOAD_NVIDIA_SMI_ARGUMENTS)
environment = dict(guard._POST_LOAD_NVIDIA_SMI_ENVIRONMENT)
guard.run_frozen_memory_query(command)
try:
    subprocess.run(command, check=True, env=environment)
except ForbiddenFinalEvaluationAssetAccessError:
    pass
else:
    raise AssertionError('a direct exact query bypassed the private capability')
try:
    subprocess.run(
        ('nvidia-smi', *guard._POST_LOAD_NVIDIA_SMI_ARGUMENTS),
        check=True,
        env=environment,
    )
except ForbiddenFinalEvaluationAssetAccessError:
    pass
else:
    raise AssertionError('a basename executable bypassed the frozen absolute identity')
print('absolute-query-only')
"""
    completed = subprocess.run(
        (sys.executable, "-c", program),
        check=False,
        capture_output=True,
        text=True,
    )
    assert completed.returncode == 0, completed.stderr
    assert completed.stdout.strip() == "absolute-query-only"


def test_post_load_process_creation_rejects_frozen_executable_drift(
    tmp_path: Path,
) -> None:
    official, cache, manifest, asset, _validation = _asset_tree(tmp_path)
    nvidia_smi = _fake_nvidia_smi(tmp_path)
    program = f"""
import subprocess
from pathlib import Path
from controller_learning.evaluation.test_access import (
    ForbiddenFinalEvaluationAssetAccessError,
    M8TestAssetAccessGuard,
)
executable = Path({str(nvidia_smi)!r})
guard = M8TestAssetAccessGuard(
    official_track_root=Path({str(official)!r}),
    test_manifest=Path({str(manifest)!r}),
    test_asset=Path({str(asset)!r}),
    track_cache_root=Path({str(cache)!r}),
)
guard.install()
guard._nvidia_smi_identity = guard._capture_executable_identity(
    executable, require_trusted_ownership=False
)
guard.enable_test_reads()
Path({str(manifest)!r}).read_bytes()
Path({str(asset)!r}).read_bytes()
guard.forbid_all_track_reads()
executable.write_bytes(b'#!/bin/sh\\nexit 1\\n')
try:
    guard.run_frozen_memory_query(
        (str(executable), *guard._POST_LOAD_NVIDIA_SMI_ARGUMENTS),
    )
except ForbiddenFinalEvaluationAssetAccessError:
    pass
else:
    raise AssertionError('a changed frozen executable was launched')
print('drift-blocked')
"""
    completed = subprocess.run(
        (sys.executable, "-c", program),
        check=False,
        capture_output=True,
        text=True,
    )
    assert completed.returncode == 0, completed.stderr
    assert completed.stdout.strip() == "drift-blocked"


def test_deterministic_recovery_disables_process_creation_and_track_reads(
    tmp_path: Path,
) -> None:
    official, cache, manifest, asset, _validation = _asset_tree(tmp_path)
    nvidia_smi = _fake_nvidia_smi(tmp_path)
    program = f"""
import subprocess
from pathlib import Path
from controller_learning.evaluation.test_access import (
    ForbiddenFinalEvaluationAssetAccessError,
    M8TestAssetAccessGuard,
)
guard = M8TestAssetAccessGuard(
    official_track_root=Path({str(official)!r}),
    test_manifest=Path({str(manifest)!r}),
    test_asset=Path({str(asset)!r}),
    track_cache_root=Path({str(cache)!r}),
)
guard.install()
guard.enter_deterministic_recovery()
for operation in (
    lambda: subprocess.run((str(Path({str(nvidia_smi)!r})),), check=True),
    lambda: Path({str(manifest)!r}).read_bytes(),
):
    try:
        operation()
    except ForbiddenFinalEvaluationAssetAccessError:
        pass
    else:
        raise AssertionError('deterministic recovery allowed forbidden external input')
assert guard._denied_event_count == 2
print('recovery-sealed')
"""
    completed = subprocess.run(
        (sys.executable, "-c", program),
        check=False,
        capture_output=True,
        text=True,
    )
    assert completed.returncode == 0, completed.stderr
    assert completed.stdout.strip() == "recovery-sealed"
