"""Run the frozen M8 PID/MPC/PPO replacement comparison on the official Test pool.

This entry point is intentionally fail closed.  It installs the Test-asset audit guard before
importing any project module, durably records every canonical episode before starting the next
one, and publishes only the 24 paths frozen by ``configs/final_evaluation.toml``.  A Test-bound
failure is evidence, never an invitation to rerun the workload.
"""

from __future__ import annotations

import argparse
import contextlib
import hashlib
import importlib
import importlib.abc
import importlib.machinery
import importlib.util
import json
import os
import re
import secrets
import stat
import subprocess
import sys
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType, ModuleType, SimpleNamespace
from typing import Any, Final

# These process policies must exist before JAX, MuJoCo, or any project module can be imported.
os.environ.setdefault("CUDA_DEVICE_ORDER", "PCI_BUS_ID")
os.environ.setdefault("MUJOCO_GL", "egl")
os.environ.setdefault("PYOPENGL_PLATFORM", "egl")
os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
os.environ.setdefault("PYTHONDONTWRITEBYTECODE", "1")
sys.dont_write_bytecode = True

PROJECT_ROOT: Final = Path(__file__).resolve().parents[1]
CANONICAL_CONFIG: Final = Path("configs/final_evaluation.toml")
ATTEMPT_RELATIVE_PATH: Final = "runs/m8_final_attempt_002_transaction"
SNAPSHOT_RELATIVE_PATH: Final = "runs/m8_final_controller_snapshot_002"
COMMITTED_SNAPSHOT_RELATIVE_PATH: Final = SNAPSHOT_RELATIVE_PATH + ".committed"
FAILURE_EVIDENCE_BLOB: Final = "failures/final-workload.json"
_PRIVATE_GUARD_MODULE: Final = "_controller_learning_m8_test_access_guard"
_PRIVATE_ATTEMPT_MODULE: Final = "_controller_learning_m8_attempt_transaction"
_ATTEMPT_TRANSACTION_SCHEMA: Final = "controller-learning.m8-attempt-transaction.v3"
_ATTEMPT_PHASE_INDEX: Final = {
    "PREPARED": 0,
    "TEST_BOUND": 1,
    "EVALUATION_COMPLETE": 2,
    "ARTIFACTS_VALIDATED": 3,
    "COMMITTED": 4,
}
_FAILURE_EVIDENCE_SCHEMA: Final = "controller-learning.m8-workload-failure.v1"
_FAILURE_DETAIL_MAX_CHARS: Final = 512
_FAILURE_TRACEBACK_MAX_CHARS: Final = 4096
_POSIX_ABSOLUTE_PATH: Final = re.compile(r"(?<![A-Za-z0-9_.-])/(?:[^\s\"'<>]|\\ )+")
_WINDOWS_ABSOLUTE_PATH: Final = re.compile(r"(?<![A-Za-z0-9_.-])[A-Za-z]:[\\/][^\s\"'<>]+")
_SECRET_SHAPE: Final = re.compile(
    r"(?i)(?:ghp_[A-Za-z0-9]{20,}|github_pat_[A-Za-z0-9_]{20,}|"
    r"(?:password|access[_-]?token|api[_-]?key)\s*[:=]\s*[^\s,}]+)"
)

_BOOTSTRAP_STATE: _BootstrapState | None = None
_BOOTSTRAP_NONCE: Final = object()
_FORMAL_ENTRY_CONSUMED = False
_EARLY_RECOVERY_LOCKDOWN = False
_POST_BIND_COMMANDS_FORBIDDEN = False
_PROJECT_SOURCE_FINDER: _ProjectSourceFinder | None = None


@dataclass(frozen=True, slots=True)
class BenchmarkOptions:
    """The canonical frozen config is the only command-line input."""

    config: Path = CANONICAL_CONFIG


@dataclass(frozen=True, slots=True)
class _BootstrapState:
    """Unforgeable process-local identity of the installed private Test guard."""

    guard: object
    guard_class: type
    process_id: int
    project_root: Path
    recovery_lockdown: bool
    nonce: object


class PreparedRecoveryRerunRequired(RuntimeError):
    """A PREPARED attempt was cleaned safely and requires a fresh formal process."""


class _ProjectSourceFinder(importlib.abc.MetaPathFinder):
    """Resolve project modules only from exact regular source files in the canonical tree."""

    def __init__(self, project_root: Path) -> None:
        self.project_root = project_root

    @staticmethod
    def _entry_exists(path: Path) -> bool:
        try:
            path.lstat()
        except FileNotFoundError:
            return False
        except OSError as error:
            raise ImportError("project import path is unreadable") from error
        return True

    def find_spec(
        self,
        fullname: str,
        path: Sequence[str] | None = None,
        target: ModuleType | None = None,
    ) -> importlib.machinery.ModuleSpec | None:
        del path, target
        if fullname != "controller_learning" and not fullname.startswith("controller_learning."):
            return None
        parts = fullname.split(".")
        if any(not part.isidentifier() for part in parts):
            raise ImportError("project module name is not a dotted Python identifier")
        package_directory = self.project_root.joinpath(*parts)
        package_init = package_directory / "__init__.py"
        module_source = self.project_root.joinpath(*parts).with_suffix(".py")
        package_exists = self._entry_exists(package_directory)
        module_exists = self._entry_exists(module_source)
        if package_exists:
            _require_real_project_directory(self.project_root, package_directory)
            if not self._entry_exists(package_init):
                raise ImportError(f"project package {fullname!r} cannot be a namespace package")
            _require_regular_project_python_path(self.project_root, package_init)
            if module_exists:
                raise ImportError(f"project import {fullname!r} has ambiguous source entries")
            loader = importlib.machinery.SourceFileLoader(fullname, str(package_init))
            spec = importlib.util.spec_from_file_location(
                fullname,
                package_init,
                loader=loader,
                submodule_search_locations=[str(package_directory)],
            )
        elif module_exists:
            _require_regular_project_python_path(self.project_root, module_source)
            loader = importlib.machinery.SourceFileLoader(fullname, str(module_source))
            spec = importlib.util.spec_from_file_location(
                fullname,
                module_source,
                loader=loader,
            )
        else:
            raise ModuleNotFoundError(
                f"project import {fullname!r} has no exact regular Python source"
            )
        if spec is None or spec.loader is not loader:
            raise ImportError(f"could not create the exact project source spec for {fullname!r}")
        return spec


@dataclass(frozen=True, slots=True)
class _StaticInputs:
    """All non-Test identities needed to construct or recover one attempt."""

    config_path: Path
    config: Any
    config_bytes: bytes
    config_sha256: str
    pixi_lock_bytes: bytes
    pixi_lock_sha256: str
    reports: Mapping[str, Any]
    reports_digest: str
    source_revision: str
    source_tree_identity: Any
    source_tree_sha256: str


@dataclass(frozen=True, slots=True)
class _DurableEvidence:
    """Typed evidence reconstructed from transaction-local canonical JSON."""

    test_pool_access: Any
    test_access_audit: Any
    runtime: Any
    memory: Any
    execution: Any
    durable_execution_evidence: Any


def _parse_args(argv: Sequence[str] | None = None) -> BenchmarkOptions:
    parser = argparse.ArgumentParser(
        description="Run the frozen M8 PID, MPC, and PPO Test comparison"
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=CANONICAL_CONFIG,
        help="Must be configs/final_evaluation.toml",
    )
    return BenchmarkOptions(config=parser.parse_args(argv).config)


def _canonical_config_path(project_root: Path, requested: Path) -> Path:
    root = project_root.resolve(strict=True)
    candidate = requested if requested.is_absolute() else root / requested
    if candidate.is_symlink():
        raise RuntimeError("the formal M8 config cannot be a symbolic link")
    try:
        resolved = candidate.resolve(strict=True)
    except FileNotFoundError as error:
        raise RuntimeError("the formal M8 config is missing") from error
    expected = (root / CANONICAL_CONFIG).resolve(strict=True)
    if resolved != expected or not resolved.is_file():
        raise RuntimeError("formal M8 accepts only configs/final_evaluation.toml")
    return resolved


def _assert_project_not_imported() -> None:
    imported = sorted(
        name
        for name in sys.modules
        if name == "controller_learning" or name.startswith("controller_learning.")
    )
    if imported:
        raise RuntimeError(
            "the formal M8 Test guard must be installed before project imports: "
            + ", ".join(imported[:5])
        )


def _canonical_project_root(project_root: Path) -> Path:
    try:
        root = Path(project_root).resolve(strict=True)
        canonical = PROJECT_ROOT.resolve(strict=True)
    except (FileNotFoundError, OSError) as error:
        raise RuntimeError("the canonical formal project root is missing or unreadable") from error
    if root != canonical:
        raise RuntimeError("formal M8 is bound to the project root containing this entry point")
    try:
        metadata = canonical.lstat()
    except OSError as error:
        raise RuntimeError("the canonical formal project root is unreadable") from error
    if stat.S_ISLNK(metadata.st_mode) or not stat.S_ISDIR(metadata.st_mode):
        raise RuntimeError("the canonical formal project root must be a real directory")
    return canonical


def _active_attempt_transaction_exists(project_root: Path) -> bool:
    """Detect the fixed transaction entry using only no-follow stdlib metadata."""

    root = _canonical_project_root(project_root)
    runs = root / "runs"
    try:
        runs_metadata = runs.lstat()
    except FileNotFoundError:
        return False
    except OSError as error:
        raise RuntimeError("the formal runs directory is unreadable") from error
    if stat.S_ISLNK(runs_metadata.st_mode) or not stat.S_ISDIR(runs_metadata.st_mode):
        raise RuntimeError("formal recovery requires a real runs directory")
    transaction = root / ATTEMPT_RELATIVE_PATH
    try:
        transaction.lstat()
    except FileNotFoundError:
        return False
    except OSError as error:
        raise RuntimeError("the formal attempt transaction is unreadable") from error
    return True


def _load_private_module(name: str, path: Path) -> ModuleType:
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError("could not load the M8 Test access guard")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    try:
        spec.loader.exec_module(module)
    except BaseException:
        sys.modules.pop(name, None)
        raise
    return module


def _bootstrap_test_guard(project_root: Path) -> object:
    """File-load and install the stdlib-only guard before any package import."""

    global _BOOTSTRAP_STATE
    if _BOOTSTRAP_STATE is not None or _FORMAL_ENTRY_CONSUMED:
        raise RuntimeError("the formal M8 Test guard may be bootstrapped only once per process")
    _assert_project_not_imported()
    root = _canonical_project_root(project_root)
    guard_module = _load_private_module(
        _PRIVATE_GUARD_MODULE,
        root / "controller_learning/evaluation/test_access.py",
    )
    official_root = root / "controller_learning/assets/tracks/v0.1"
    guard = guard_module.M8TestAssetAccessGuard(
        official_track_root=official_root,
        test_manifest=official_root / "test.json",
        test_asset=official_root / "test.npz",
        track_cache_root=root / ".track-cache/v0.1",
    )
    guard.install()
    if (
        type(guard) is not guard_module.M8TestAssetAccessGuard
        or type(guard).__module__ != _PRIVATE_GUARD_MODULE
    ):
        raise RuntimeError("the installed Test guard does not have the private module identity")
    recovery_lockdown = _active_attempt_transaction_exists(root)
    if recovery_lockdown:
        guard.enter_deterministic_recovery()
    _BOOTSTRAP_STATE = _BootstrapState(
        guard=guard,
        guard_class=guard_module.M8TestAssetAccessGuard,
        process_id=os.getpid(),
        project_root=root,
        recovery_lockdown=recovery_lockdown,
        nonce=_BOOTSTRAP_NONCE,
    )
    return guard


def _consume_bootstrap_guard(project_root: Path) -> object:
    """Consume the unique same-process guard identity immediately before project imports."""

    global _BOOTSTRAP_STATE, _EARLY_RECOVERY_LOCKDOWN, _FORMAL_ENTRY_CONSUMED
    state = _BOOTSTRAP_STATE
    if _FORMAL_ENTRY_CONSUMED or not isinstance(state, _BootstrapState):
        raise RuntimeError("run_benchmark requires the unconsumed private guard bootstrap")
    root = _canonical_project_root(project_root)
    if (
        state.nonce is not _BOOTSTRAP_NONCE
        or state.process_id != os.getpid()
        or state.project_root != root
        or type(state.guard) is not state.guard_class
        or state.guard_class.__module__ != _PRIVATE_GUARD_MODULE
        or state.guard_class.__name__ != "M8TestAssetAccessGuard"
    ):
        raise RuntimeError("the private Test guard identity or process binding differs")
    _assert_project_not_imported()
    transaction_exists = _active_attempt_transaction_exists(root)
    if state.recovery_lockdown and not transaction_exists:
        raise RuntimeError("the active formal transaction disappeared before dependency imports")
    recovery_lockdown = state.recovery_lockdown or transaction_exists
    if recovery_lockdown and not state.recovery_lockdown:
        state.guard.enter_deterministic_recovery()
    try:
        state.guard._audit_identity_response = None
        audit_token = state.guard._audit_identity_token
        audit_event = state.guard._AUDIT_SELF_CHECK_EVENT
    except AttributeError as error:
        raise RuntimeError("the private Test guard lacks its audit-hook identity state") from error
    sys.audit(audit_event, audit_token)
    if state.guard._audit_identity_response is not audit_token:
        raise RuntimeError("the private Test guard audit hook did not answer its identity check")
    evidence = state.guard.evidence(test_loaded=False)
    try:
        deterministic_recovery = state.guard._deterministic_recovery
    except AttributeError as error:
        raise RuntimeError("the private Test guard lacks its recovery-lockdown state") from error
    clean_state = (
        deterministic_recovery is recovery_lockdown
        and evidence["audit_hook_installed_before_preflight"] is True
        and evidence["test_reads_enabled"] is False
        and evidence["all_track_reads_forbidden"] is False
        and evidence["open_event_counts"] == {}
        and evidence["denied_event_count"] == 0
    )
    if not clean_state:
        raise RuntimeError("the private Test guard is not in its clean pre-import state")
    _EARLY_RECOVERY_LOCKDOWN = recovery_lockdown
    _FORMAL_ENTRY_CONSUMED = True
    _BOOTSTRAP_STATE = None
    return state.guard


def _pin_project_import_root(project_root: Path) -> None:
    root = project_root.resolve(strict=True)
    retained: list[str] = []
    for entry in sys.path:
        if not isinstance(entry, str):
            continue
        try:
            resolved = Path(entry or os.curdir).resolve(strict=True)
        except (FileNotFoundError, OSError):
            retained.append(entry)
            continue
        if resolved != root:
            retained.append(entry)
    sys.path[:] = [str(root), *retained]


def _remove_project_import_root(project_root: Path) -> None:
    root = project_root.resolve(strict=True)
    retained: list[str] = []
    for entry in sys.path:
        if not isinstance(entry, str):
            continue
        try:
            resolved = Path(entry or os.curdir).resolve(strict=True)
        except (FileNotFoundError, OSError):
            retained.append(entry)
            continue
        if resolved != root:
            retained.append(entry)
    sys.path[:] = retained
    for entry in sys.path:
        if not isinstance(entry, str):
            continue
        try:
            if Path(entry or os.curdir).resolve(strict=True) == root:
                raise RuntimeError("the project root remains on the ordinary import path")
        except (FileNotFoundError, OSError):
            continue


def _create_isolated_import_cache(project_root: Path) -> Path:
    root = project_root.resolve(strict=True)
    root_metadata = root.lstat()
    flags = (
        os.O_RDONLY
        | getattr(os, "O_CLOEXEC", 0)
        | getattr(os, "O_DIRECTORY", 0)
        | getattr(os, "O_NOFOLLOW", 0)
    )
    root_descriptor = os.open(root, flags)
    try:
        opened_root = os.fstat(root_descriptor)
        if (
            stat.S_ISLNK(root_metadata.st_mode)
            or not stat.S_ISDIR(root_metadata.st_mode)
            or root_metadata.st_dev != opened_root.st_dev
            or root_metadata.st_ino != opened_root.st_ino
        ):
            raise RuntimeError("formal project root changed during import isolation")
        try:
            runs_metadata = os.stat("runs", dir_fd=root_descriptor, follow_symlinks=False)
        except FileNotFoundError:
            os.mkdir("runs", 0o700, dir_fd=root_descriptor)
            os.fsync(root_descriptor)
            runs_metadata = os.stat("runs", dir_fd=root_descriptor, follow_symlinks=False)
        if stat.S_ISLNK(runs_metadata.st_mode) or not stat.S_ISDIR(runs_metadata.st_mode):
            raise RuntimeError("formal M8 bytecode isolation requires a real runs directory")
        runs_descriptor = os.open("runs", flags, dir_fd=root_descriptor)
        try:
            opened_runs = os.fstat(runs_descriptor)
            if (
                runs_metadata.st_dev != opened_runs.st_dev
                or runs_metadata.st_ino != opened_runs.st_ino
            ):
                raise RuntimeError("formal runs directory changed during import isolation")
            for _attempt in range(128):
                name = "m8_import_cache." + secrets.token_hex(16)
                try:
                    os.mkdir(name, 0o700, dir_fd=runs_descriptor)
                except FileExistsError:
                    continue
                os.fsync(runs_descriptor)
                return root / "runs" / name
            raise RuntimeError("could not allocate an isolated formal import cache")
        finally:
            os.close(runs_descriptor)
    finally:
        os.close(root_descriptor)


def _prepare_isolated_python_runtime(project_root: Path) -> None:
    """Validate the exact GPU interpreter before installing the Test guard."""

    if (
        sys.flags.isolated != 1
        or sys.flags.ignore_environment != 1
        or sys.flags.no_user_site != 1
        or sys.flags.no_site != 1
        or sys.flags.dont_write_bytecode != 1
        or sys.flags.safe_path is not True
    ):
        raise RuntimeError("formal M8 must run with Python -I -B -S")
    root = _canonical_project_root(project_root)
    gpu_prefix = root / ".pixi/envs/gpu"
    expected_executable = gpu_prefix / "bin/python3.11"
    try:
        executable = Path(sys.executable).resolve(strict=True)
        expected_executable_metadata = expected_executable.lstat()
    except (FileNotFoundError, OSError) as error:
        raise RuntimeError("the canonical Pixi GPU interpreter is missing or unreadable") from error
    if (
        executable != expected_executable
        or not stat.S_ISREG(expected_executable_metadata.st_mode)
        or stat.S_ISLNK(expected_executable_metadata.st_mode)
        or Path(sys.prefix) != gpu_prefix
        or Path(sys.base_prefix) != gpu_prefix
    ):
        raise RuntimeError("formal M8 requires the exact canonical Pixi GPU Python 3.11")
    forbidden_environment = sorted(
        name
        for name in os.environ
        if name in {"LD_LIBRARY_PATH", "LD_PRELOAD"}
        or (
            (name.startswith("PYTHON") or name.startswith("_PYTHON"))
            and name != "PYTHONDONTWRITEBYTECODE"
        )
    )
    if forbidden_environment:
        raise RuntimeError(
            "formal M8 rejects startup environment overrides: " + ", ".join(forbidden_environment)
        )
    if os.environ.get("PYTHONDONTWRITEBYTECODE") != "1":
        raise RuntimeError("formal M8 requires its fixed bytecode environment policy")
    if os.environ.get("MUJOCO_GL") != "egl" or os.environ.get("PYOPENGL_PLATFORM") != "egl":
        raise RuntimeError("formal M8 requires the fixed headless EGL import route")
    if any(name in sys.modules for name in ("site", "sitecustomize", "usercustomize")):
        raise RuntimeError("formal M8 rejects site and startup customization modules")
    cache = _create_isolated_import_cache(root)
    sys.pycache_prefix = str(cache)
    _pin_project_import_root(root)


def _require_real_project_directory(project_root: Path, expected: Path) -> None:
    root = project_root.resolve(strict=True)
    if not expected.is_relative_to(root):
        raise RuntimeError("project directory escaped the canonical source tree")
    try:
        root_metadata = root.lstat()
    except OSError as error:
        raise RuntimeError("project root is unreadable") from error
    if stat.S_ISLNK(root_metadata.st_mode) or not stat.S_ISDIR(root_metadata.st_mode):
        raise RuntimeError("project root must be a real directory")
    current = root
    metadata = root_metadata
    for part in expected.relative_to(root).parts:
        current /= part
        try:
            metadata = current.lstat()
        except (FileNotFoundError, OSError) as error:
            raise RuntimeError("project directory is missing or unreadable") from error
        if stat.S_ISLNK(metadata.st_mode):
            raise RuntimeError("project directory path contains a symbolic link")
    if not stat.S_ISDIR(metadata.st_mode):
        raise RuntimeError("project package path must be a real directory")


def _require_regular_project_python_path(project_root: Path, expected: Path) -> None:
    root = project_root.resolve(strict=True)
    if not expected.is_relative_to(root) or expected.suffix != ".py":
        raise RuntimeError("project module source path escaped the Python source tree")
    current = root
    for part in expected.relative_to(root).parts:
        current /= part
        try:
            metadata = current.lstat()
        except (FileNotFoundError, OSError) as error:
            raise RuntimeError("project module source path is missing or unreadable") from error
        if stat.S_ISLNK(metadata.st_mode):
            raise RuntimeError("project module source path contains a symbolic link")
    try:
        metadata = expected.lstat()
    except (FileNotFoundError, OSError) as error:
        raise RuntimeError("project module source is missing or unreadable") from error
    if not stat.S_ISREG(metadata.st_mode):
        raise RuntimeError("project module source must be a regular .py file")


def _install_project_source_finder(project_root: Path) -> _ProjectSourceFinder:
    """Install the sole project resolver before any ordinary project import."""

    global _PROJECT_SOURCE_FINDER
    root = _canonical_project_root(project_root)
    _assert_project_not_imported()
    if _PROJECT_SOURCE_FINDER is not None:
        raise RuntimeError("the formal project source finder may be installed only once")
    finder = _ProjectSourceFinder(root)
    sys.meta_path.insert(0, finder)
    _PROJECT_SOURCE_FINDER = finder
    _assert_project_source_finder_active(root)
    return finder


def _load_private_attempt_transaction_module(project_root: Path) -> ModuleType:
    """Load the stdlib-only transaction code from its exact regular source file."""

    root = _canonical_project_root(project_root)
    _assert_project_not_imported()
    source = root / "controller_learning/evaluation/attempt_transaction.py"
    _require_regular_project_python_path(root, source)
    if _PRIVATE_ATTEMPT_MODULE in sys.modules:
        raise RuntimeError("the private PREPARED recovery module was already loaded")
    module = _load_private_module(_PRIVATE_ATTEMPT_MODULE, source)
    spec = module.__spec__
    loader = module.__loader__
    if (
        type(module) is not ModuleType
        or module.__file__ != str(source)
        or spec is None
        or spec.origin != str(source)
        or type(loader) is not importlib.machinery.SourceFileLoader
        or loader.path != str(source)
    ):
        sys.modules.pop(_PRIVATE_ATTEMPT_MODULE, None)
        raise RuntimeError("the private PREPARED recovery module provenance differs")
    _assert_project_not_imported()
    return module


def _prepared_transaction_from_durable_manifest(
    project_root: Path,
) -> tuple[ModuleType, object, object]:
    """Reconstruct an existing transaction from its own pre-Test durable identity."""

    root = _canonical_project_root(project_root)
    module = _load_private_attempt_transaction_module(root)
    transaction_directory = root / ATTEMPT_RELATIVE_PATH
    manifest = module._read_canonical_json(
        transaction_directory / "manifest.json",
        field_name="transaction manifest",
    )
    identity_mapping = manifest.get("identity")
    output_allowlist = manifest.get("output_allowlist")
    if not isinstance(identity_mapping, Mapping) or set(identity_mapping) != {
        "config_sha256",
        "input_sha256",
        "pixi_lock_sha256",
        "source_revision",
        "source_tree_sha256",
    }:
        raise RuntimeError("PREPARED transaction identity keys differ")
    if (
        not isinstance(output_allowlist, list)
        or len(output_allowlist) != 24
        or any(type(value) is not str for value in output_allowlist)
    ):
        raise RuntimeError("PREPARED transaction output allowlist must contain 24 paths")
    identity = module.AttemptIdentity(**dict(identity_mapping))
    transaction = module.M8AttemptTransaction(
        root,
        transaction_relative_path=ATTEMPT_RELATIVE_PATH,
        output_allowlist=output_allowlist,
        identity=identity,
    )
    inspection = transaction.inspect()
    if not inspection.exists:
        raise RuntimeError("the locked formal transaction disappeared during early recovery")
    return module, transaction, inspection


def _read_canonical_json_no_follow(
    path: Path,
    *,
    field_name: str,
    parent_descriptor: int | None = None,
) -> tuple[Mapping[str, Any], bytes]:
    """Read one canonical JSON object without following links or mutating its directory."""

    try:
        before = (
            path.lstat()
            if parent_descriptor is None
            else os.stat(path.name, dir_fd=parent_descriptor, follow_symlinks=False)
        )
    except (FileNotFoundError, OSError) as error:
        raise RuntimeError(f"{field_name} is missing or unreadable") from error
    if stat.S_ISLNK(before.st_mode) or not stat.S_ISREG(before.st_mode):
        raise RuntimeError(f"{field_name} must be a non-symlink regular file")
    descriptor = os.open(
        path if parent_descriptor is None else path.name,
        os.O_RDONLY | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0),
        dir_fd=parent_descriptor,
    )
    try:
        opened = os.fstat(descriptor)
        chunks: list[bytes] = []
        while chunk := os.read(descriptor, 1024 * 1024):
            chunks.append(chunk)
        opened_after = os.fstat(descriptor)
    finally:
        os.close(descriptor)
    try:
        after = (
            path.lstat()
            if parent_descriptor is None
            else os.stat(path.name, dir_fd=parent_descriptor, follow_symlinks=False)
        )
    except (FileNotFoundError, OSError) as error:
        raise RuntimeError(f"{field_name} changed while it was read") from error
    identity_fields = (
        "st_dev",
        "st_ino",
        "st_mode",
        "st_uid",
        "st_gid",
        "st_size",
        "st_ctime_ns",
        "st_mtime_ns",
    )
    if any(
        getattr(left, name) != getattr(right, name)
        for left, right in ((before, opened), (opened, opened_after), (opened_after, after))
        for name in identity_fields
    ):
        raise RuntimeError(f"{field_name} changed while it was read")
    payload = b"".join(chunks)
    if len(payload) != opened.st_size:
        raise RuntimeError(f"{field_name} size changed while it was read")

    def reject_constant(value: str) -> None:
        raise ValueError(f"forbidden JSON constant {value}")

    def reject_duplicates(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in pairs:
            if key in result:
                raise ValueError(f"duplicate JSON key {key!r}")
            result[key] = value
        return result

    try:
        value = json.loads(
            payload,
            parse_constant=reject_constant,
            object_pairs_hook=reject_duplicates,
        )
    except (UnicodeDecodeError, json.JSONDecodeError, ValueError) as error:
        raise RuntimeError(f"{field_name} is not strict JSON") from error
    if not isinstance(value, Mapping) or _canonical_json_bytes(value) != payload:
        raise RuntimeError(f"{field_name} is not canonical JSON")
    return value, payload


def _peek_attempt_phase_read_only(project_root: Path) -> str:
    """Validate enough durable identity to classify phase without transaction-tree writes."""

    root = _canonical_project_root(project_root)
    directory_flags = (
        os.O_RDONLY
        | getattr(os, "O_CLOEXEC", 0)
        | getattr(os, "O_DIRECTORY", 0)
        | getattr(os, "O_NOFOLLOW", 0)
    )
    root_metadata = root.lstat()
    root_descriptor = os.open(root, directory_flags)
    try:
        opened_root = os.fstat(root_descriptor)
        if (root_metadata.st_dev, root_metadata.st_ino) != (
            opened_root.st_dev,
            opened_root.st_ino,
        ):
            raise RuntimeError("the project root changed during the read-only phase peek")
        try:
            runs_metadata = os.stat("runs", dir_fd=root_descriptor, follow_symlinks=False)
            runs_descriptor = os.open("runs", directory_flags, dir_fd=root_descriptor)
        except (FileNotFoundError, OSError) as error:
            raise RuntimeError("the formal runs directory is missing or unreadable") from error
        try:
            opened_runs = os.fstat(runs_descriptor)
            if (
                stat.S_ISLNK(runs_metadata.st_mode)
                or not stat.S_ISDIR(runs_metadata.st_mode)
                or (runs_metadata.st_dev, runs_metadata.st_ino)
                != (opened_runs.st_dev, opened_runs.st_ino)
            ):
                raise RuntimeError("the formal runs directory changed during phase peek")
            transaction_name = Path(ATTEMPT_RELATIVE_PATH).name
            try:
                transaction_metadata = os.stat(
                    transaction_name,
                    dir_fd=runs_descriptor,
                    follow_symlinks=False,
                )
                transaction_descriptor = os.open(
                    transaction_name,
                    directory_flags,
                    dir_fd=runs_descriptor,
                )
            except (FileNotFoundError, OSError) as error:
                raise RuntimeError(
                    "the locked formal transaction is missing or unreadable"
                ) from error
            try:
                opened_transaction = os.fstat(transaction_descriptor)
                if (
                    stat.S_ISLNK(transaction_metadata.st_mode)
                    or not stat.S_ISDIR(transaction_metadata.st_mode)
                    or (transaction_metadata.st_dev, transaction_metadata.st_ino)
                    != (opened_transaction.st_dev, opened_transaction.st_ino)
                ):
                    raise RuntimeError("the locked formal transaction changed during phase peek")
                manifest, manifest_bytes = _read_canonical_json_no_follow(
                    Path("manifest.json"),
                    field_name="transaction manifest",
                    parent_descriptor=transaction_descriptor,
                )
                state, _state_bytes = _read_canonical_json_no_follow(
                    Path("state.json"),
                    field_name="transaction state",
                    parent_descriptor=transaction_descriptor,
                )
            finally:
                os.close(transaction_descriptor)
        finally:
            os.close(runs_descriptor)
    finally:
        os.close(root_descriptor)
    expected_state_keys = {
        "evidence",
        "identity",
        "manifest_sha256",
        "phase",
        "phase_index",
        "schema_version",
    }
    phase = state.get("phase")
    if (
        set(state) != expected_state_keys
        or not isinstance(phase, str)
        or phase not in _ATTEMPT_PHASE_INDEX
        or state.get("phase_index") != _ATTEMPT_PHASE_INDEX[phase]
        or state.get("schema_version") != _ATTEMPT_TRANSACTION_SCHEMA
        or manifest.get("schema_version") != _ATTEMPT_TRANSACTION_SCHEMA
        or manifest.get("transaction_relative_path") != ATTEMPT_RELATIVE_PATH
        or state.get("identity") != manifest.get("identity")
        or state.get("manifest_sha256") != hashlib.sha256(manifest_bytes).hexdigest()
        or (phase in {"PREPARED", "TEST_BOUND"} and state.get("evidence") is not None)
    ):
        raise RuntimeError("the read-only formal transaction phase identity differs")
    return phase


def _isolate_prepared_controller_snapshot(project_root: Path) -> Path | None:
    """Quarantine the active PREPARED snapshot through one same-parent dirfd rename."""

    root = _canonical_project_root(project_root)
    flags = (
        os.O_RDONLY
        | getattr(os, "O_CLOEXEC", 0)
        | getattr(os, "O_DIRECTORY", 0)
        | getattr(os, "O_NOFOLLOW", 0)
    )
    root_descriptor = os.open(root, flags)
    try:
        root_metadata = root.lstat()
        opened_root = os.fstat(root_descriptor)
        if (
            stat.S_ISLNK(root_metadata.st_mode)
            or not stat.S_ISDIR(opened_root.st_mode)
            or (root_metadata.st_dev, root_metadata.st_ino)
            != (opened_root.st_dev, opened_root.st_ino)
        ):
            raise RuntimeError("the formal project root changed during PREPARED recovery")
        runs_metadata = os.stat("runs", dir_fd=root_descriptor, follow_symlinks=False)
        if stat.S_ISLNK(runs_metadata.st_mode) or not stat.S_ISDIR(runs_metadata.st_mode):
            raise RuntimeError("PREPARED snapshot recovery requires a real runs directory")
        runs_descriptor = os.open("runs", flags, dir_fd=root_descriptor)
        try:
            opened_runs = os.fstat(runs_descriptor)
            if (runs_metadata.st_dev, runs_metadata.st_ino) != (
                opened_runs.st_dev,
                opened_runs.st_ino,
            ):
                raise RuntimeError("the runs directory changed during PREPARED recovery")
            active_name = Path(SNAPSHOT_RELATIVE_PATH).name
            committed_name = Path(COMMITTED_SNAPSHOT_RELATIVE_PATH).name
            try:
                os.stat(committed_name, dir_fd=runs_descriptor, follow_symlinks=False)
            except FileNotFoundError:
                pass
            else:
                raise RuntimeError("PREPARED recovery found a COMMITTED snapshot quarantine")
            try:
                active_metadata = os.stat(
                    active_name,
                    dir_fd=runs_descriptor,
                    follow_symlinks=False,
                )
            except FileNotFoundError:
                os.fsync(runs_descriptor)
                return None
            if stat.S_ISLNK(active_metadata.st_mode) or not stat.S_ISDIR(active_metadata.st_mode):
                raise RuntimeError("the active PREPARED snapshot must be a real directory")
            quarantine_name = active_name + ".abort." + secrets.token_hex(16)
            try:
                os.stat(quarantine_name, dir_fd=runs_descriptor, follow_symlinks=False)
            except FileNotFoundError:
                pass
            else:  # pragma: no cover - 128-bit collision is not practically reachable
                raise RuntimeError("the PREPARED snapshot quarantine name collided")
            os.rename(
                active_name,
                quarantine_name,
                src_dir_fd=runs_descriptor,
                dst_dir_fd=runs_descriptor,
            )
            quarantine_metadata = os.stat(
                quarantine_name,
                dir_fd=runs_descriptor,
                follow_symlinks=False,
            )
            try:
                os.stat(active_name, dir_fd=runs_descriptor, follow_symlinks=False)
            except FileNotFoundError:
                pass
            else:
                raise RuntimeError("the active PREPARED snapshot remained after quarantine")
            identity_fields = ("st_dev", "st_ino", "st_mode", "st_uid", "st_gid")
            if any(
                getattr(active_metadata, field) != getattr(quarantine_metadata, field)
                for field in identity_fields
            ):
                raise RuntimeError("the PREPARED snapshot inode changed during quarantine")
            os.fsync(runs_descriptor)
            return root / "runs" / quarantine_name
        finally:
            os.close(runs_descriptor)
    finally:
        os.close(root_descriptor)


def _recover_prepared_attempt_before_dependencies(
    project_root: Path,
    *,
    recovery_lockdown: bool,
) -> None:
    """Retire only PREPARED state before site, GPU, MuJoCo, or project imports."""

    if not recovery_lockdown:
        return
    if _peek_attempt_phase_read_only(project_root) != "PREPARED":
        return
    module, transaction, inspection = _prepared_transaction_from_durable_manifest(project_root)
    if inspection.phase is not module.AttemptPhase.PREPARED:
        return
    _isolate_prepared_controller_snapshot(project_root)
    recovery = transaction.recover()
    if recovery.action != "pre_test_restored":
        raise RuntimeError("early PREPARED recovery did not restore the pre-Test state")
    raise PreparedRecoveryRerunRequired(
        "PREPARED formal state was cleaned before dependency imports; rerun is required"
    )


def _assert_project_source_finder_active(project_root: Path) -> None:
    root = _canonical_project_root(project_root)
    finder = _PROJECT_SOURCE_FINDER
    if (
        not isinstance(finder, _ProjectSourceFinder)
        or finder.project_root != root
        or not sys.meta_path
        or sys.meta_path[0] is not finder
        or sum(candidate is finder for candidate in sys.meta_path) != 1
    ):
        raise RuntimeError("the canonical project-only source finder is not first and unique")


def _configure_gpu_site_packages(project_root: Path) -> Path:
    """Append one exact site-packages path without executing any .pth file."""

    root = _canonical_project_root(project_root)
    gpu_prefix = root / ".pixi/envs/gpu"
    site_packages = gpu_prefix / "lib/python3.11/site-packages"
    _require_real_project_directory(root, site_packages)
    equivalent_entries: list[str] = []
    for entry in sys.path:
        if not isinstance(entry, str):
            continue
        try:
            if Path(entry or os.curdir).resolve(strict=True) == site_packages:
                equivalent_entries.append(entry)
        except (FileNotFoundError, OSError):
            continue
    if equivalent_entries:
        raise RuntimeError("GPU site-packages was present before its guarded explicit append")
    sys.path.append(str(site_packages))

    module_name = "_cuda_bindings_redirector"
    if module_name in sys.modules:
        raise RuntimeError("CUDA bindings redirector was imported before guarded site setup")
    redirector = site_packages / f"{module_name}.py"
    _require_regular_project_python_path(root, redirector)
    loader = importlib.machinery.SourceFileLoader(module_name, str(redirector))
    spec = importlib.util.spec_from_file_location(module_name, redirector, loader=loader)
    if spec is None or spec.loader is not loader:
        raise RuntimeError("could not build the exact CUDA bindings redirector spec")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    try:
        loader.exec_module(module)
    except BaseException:
        sys.modules.pop(module_name, None)
        raise
    if (
        type(module) is not ModuleType
        or module.__spec__ is not spec
        or module.__loader__ is not loader
        or module.__file__ != str(redirector)
        or spec.origin != str(redirector)
        or loader.path != str(redirector)
    ):
        raise RuntimeError("CUDA bindings redirector provenance differs from its exact .py file")
    return site_packages


def _validate_project_module_provenance(project_root: Path) -> None:
    root = project_root.resolve(strict=True)
    project_modules = sorted(
        (name, module)
        for name, module in sys.modules.items()
        if name == "controller_learning" or name.startswith("controller_learning.")
    )
    if not project_modules:
        raise RuntimeError("formal M8 loaded no project modules")
    for name, module in project_modules:
        if not isinstance(module, ModuleType):
            raise RuntimeError(f"project module {name!r} is not a real module")
        parts = name.split(".")
        package_paths = getattr(module, "__path__", None)
        if package_paths is None:
            expected = root.joinpath(*parts).with_suffix(".py")
        else:
            paths = tuple(package_paths)
            expected_directory = root.joinpath(*parts)
            if len(paths) != 1 or Path(paths[0]) != expected_directory:
                raise RuntimeError(f"project package {name!r} is namespace, zip, or shadowed")
            expected = expected_directory / "__init__.py"
        _require_regular_project_python_path(root, expected)
        module_file = getattr(module, "__file__", None)
        module_spec = getattr(module, "__spec__", None)
        origin = getattr(module_spec, "origin", None)
        loader = getattr(module_spec, "loader", None)
        if (
            not isinstance(module_file, str)
            or not isinstance(origin, str)
            or Path(module_file) != expected
            or Path(origin) != expected
            or type(loader) is not importlib.machinery.SourceFileLoader
            or Path(loader.path) != expected
        ):
            raise RuntimeError(f"project module {name!r} did not load from its exact source file")


def _load_project_api(project_root: Path) -> SimpleNamespace:
    """Import all project/GPU dependencies only after the guard is active."""

    root = _canonical_project_root(project_root)
    _assert_project_source_finder_active(root)
    modules = {
        name: importlib.import_module(module_name)
        for name, module_name in {
            "attempt": "controller_learning.evaluation.attempt_transaction",
            "benchmark": "controller_learning.evaluation.final_benchmark",
            "config": "controller_learning.config",
            "control": "controller_learning.control",
            "controller_identity": "controller_learning.evaluation.controller_identity",
            "execution": "controller_learning.evaluation.final_execution",
            "metrics": "controller_learning.evaluation.final_metrics",
            "physics_mjx_warp": "controller_learning.physics.mjx_warp",
            "preflight": "controller_learning.evaluation.final_preflight",
            "replacement": "controller_learning.evaluation.replacement",
            "report": "controller_learning.evaluation.final_report",
            "results": "controller_learning.evaluation.final_results",
            "runtime": "controller_learning.evaluation.final_runtime",
            "source_tree": "controller_learning.evaluation.source_tree_identity",
            "test_assets": "controller_learning.evaluation.test_assets",
            "trajectory": "controller_learning.evaluation.trajectory",
            "visual_final": "controller_learning.visualization.final_results",
            "visual_replay": "controller_learning.visualization.replay",
            "environment": "controller_learning.envs.car_racing",
        }.items()
    }
    modules["jax"] = importlib.import_module("jax")
    modules["warp"] = importlib.import_module("warp")
    _assert_project_source_finder_active(root)
    _validate_project_module_provenance(root)
    return SimpleNamespace(**modules)


def _run_command(command: Sequence[str], *, cwd: Path) -> str:
    if _POST_BIND_COMMANDS_FORBIDDEN:
        raise RuntimeError("formal CLI subprocess commands are forbidden after TEST_BOUND")
    try:
        completed = subprocess.run(
            tuple(command),
            cwd=cwd,
            check=True,
            capture_output=True,
            text=True,
            timeout=30,
        )
    except (OSError, subprocess.SubprocessError) as error:
        raise RuntimeError(f"formal command failed: {' '.join(command)}") from error
    return completed.stdout.rstrip("\r\n")


def _secure_git_executable() -> Path:
    executable = Path("/usr/bin/git")
    current = Path("/")
    for part in executable.parts[1:]:
        current /= part
        metadata = current.lstat()
        if stat.S_ISLNK(metadata.st_mode) or metadata.st_uid != 0 or metadata.st_mode & 0o022:
            raise RuntimeError("formal M8 requires a root-owned, non-writable /usr/bin/git path")
    metadata = executable.lstat()
    if not stat.S_ISREG(metadata.st_mode) or metadata.st_mode & 0o6000:
        raise RuntimeError("formal M8 requires a regular non-privileged /usr/bin/git executable")
    return executable


def _secure_git_command_runner(command: Sequence[str], cwd: Path) -> str:
    if _POST_BIND_COMMANDS_FORBIDDEN:
        raise RuntimeError("Git clean checks are forbidden after TEST_BOUND")
    if not command or command[0] != "git":
        raise RuntimeError("secure Git runner accepts only canonical git commands")
    executable = _secure_git_executable()
    safe_command = (
        str(executable),
        "--no-optional-locks",
        "-c",
        "core.fsmonitor=false",
        "-c",
        "core.untrackedCache=false",
        *command[1:],
    )
    environment = {
        "GIT_CONFIG_GLOBAL": "/dev/null",
        "GIT_CONFIG_NOSYSTEM": "1",
        "GIT_PAGER": "cat",
        "GIT_TERMINAL_PROMPT": "0",
        "HOME": "/nonexistent",
        "LANG": "C",
        "LC_ALL": "C",
        "PAGER": "cat",
        "PATH": "/usr/bin:/bin",
        "XDG_CONFIG_HOME": "/nonexistent",
    }
    try:
        completed = subprocess.run(
            safe_command,
            cwd=cwd,
            env=environment,
            stdin=subprocess.DEVNULL,
            check=True,
            capture_output=True,
            text=True,
            timeout=30,
        )
    except (OSError, subprocess.SubprocessError) as error:
        raise RuntimeError("formal secure Git command failed") from error
    return completed.stdout.rstrip("\r\n")


def _enter_post_bind_phase() -> None:
    global _POST_BIND_COMMANDS_FORBIDDEN
    if _POST_BIND_COMMANDS_FORBIDDEN:
        raise RuntimeError("the formal process already crossed the TEST_BOUND command latch")
    _POST_BIND_COMMANDS_FORBIDDEN = True


def _initialize_warp_runtime(api: SimpleNamespace) -> None:
    """Initialize Warp exactly once while pre-Test process creation remains available."""

    api.warp.init()
    if getattr(api.warp._src.context, "runtime", None) is None:
        raise RuntimeError("Warp runtime did not initialize before TEST_BOUND")


def _python_git_revision(project_root: Path) -> str:
    """Resolve HEAD with Python-only reads so recovery never spawns Git after Test binding."""

    marker = project_root / ".git"
    if marker.is_symlink():
        raise RuntimeError("the Git metadata marker cannot be a symbolic link")
    if marker.is_dir():
        git_directory = marker
    elif marker.is_file():
        marker_text = marker.read_text(encoding="utf-8").strip()
        prefix = "gitdir: "
        if not marker_text.startswith(prefix):
            raise RuntimeError("the Git worktree marker is malformed")
        candidate = Path(marker_text[len(prefix) :])
        git_directory = candidate if candidate.is_absolute() else project_root / candidate
        git_directory = git_directory.resolve(strict=True)
    else:
        raise RuntimeError("formal M8 requires Git repository metadata")
    head = (git_directory / "HEAD").read_text(encoding="ascii").strip()
    if head.startswith("ref: "):
        reference = head[5:]
        if not reference.startswith("refs/") or ".." in Path(reference).parts:
            raise RuntimeError("the Git HEAD reference is malformed")
        reference_path = git_directory / reference
        if reference_path.is_file() and not reference_path.is_symlink():
            revision = reference_path.read_text(encoding="ascii").strip()
        else:
            revision = ""
            packed_refs = git_directory / "packed-refs"
            if packed_refs.is_file() and not packed_refs.is_symlink():
                for line in packed_refs.read_text(encoding="ascii").splitlines():
                    if line.startswith(("#", "^")):
                        continue
                    candidate_revision, separator, candidate_ref = line.partition(" ")
                    if separator and candidate_ref == reference:
                        revision = candidate_revision
                        break
    else:
        revision = head
    if len(revision) != 40 or any(character not in "0123456789abcdef" for character in revision):
        raise RuntimeError("formal M8 requires one full lowercase Git revision")
    return revision


def _sha256(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _canonical_json_bytes(value: Mapping[str, object]) -> bytes:
    return (
        json.dumps(
            value,
            allow_nan=False,
            ensure_ascii=True,
            separators=(",", ":"),
            sort_keys=True,
        )
        + "\n"
    ).encode("ascii")


def _assert_torch_absent() -> None:
    if "torch" in sys.modules or any(name.startswith("torch.") for name in sys.modules):
        raise RuntimeError("formal ordinary-Controller evaluation must remain Torch-free")


def _load_static_inputs(
    api: SimpleNamespace,
    project_root: Path,
    config_path: Path,
) -> _StaticInputs:
    config_sha, _config_size, config_bytes = api.preflight.sha256_regular_file(config_path)
    config = api.benchmark.load_m8_final_evaluation_config(config_path)
    lock_sha, _lock_size, lock_bytes = api.preflight.sha256_regular_file(project_root / "pixi.lock")
    reports = api.preflight.load_frozen_input_reports(project_root, config)
    reports_digest = api.preflight.frozen_input_digest(reports)
    source_tree_identity = api.source_tree.capture_source_tree_identity(project_root)
    return _StaticInputs(
        config_path=config_path,
        config=config,
        config_bytes=config_bytes,
        config_sha256=config_sha,
        pixi_lock_bytes=lock_bytes,
        pixi_lock_sha256=lock_sha,
        reports=reports,
        reports_digest=reports_digest,
        source_revision=_python_git_revision(project_root),
        source_tree_identity=source_tree_identity,
        source_tree_sha256=source_tree_identity.aggregate_sha256,
    )


def _attempt_transaction(
    api: SimpleNamespace,
    project_root: Path,
    static: _StaticInputs,
) -> object:
    identity = api.attempt.AttemptIdentity(
        source_revision=static.source_revision,
        source_tree_sha256=static.source_tree_sha256,
        config_sha256=static.config_sha256,
        pixi_lock_sha256=static.pixi_lock_sha256,
        input_sha256=static.reports_digest,
    )
    return api.attempt.M8AttemptTransaction(
        project_root,
        transaction_relative_path=ATTEMPT_RELATIVE_PATH,
        output_allowlist=api.benchmark.formal_output_paths(static.config),
        identity=identity,
    )


def _snapshot_from_existing(
    api: SimpleNamespace,
    project_root: Path,
    config: object,
) -> object:
    snapshot_root = project_root / SNAPSHOT_RELATIVE_PATH
    if snapshot_root.is_symlink() or not snapshot_root.is_dir():
        raise RuntimeError("the durable formal Controller snapshot is missing or unsafe")
    identities = {
        name: api.controller_identity.capture_frozen_controller_identity(snapshot_root, name)
        for name in api.benchmark.M8_CONTROLLER_ORDER
    }
    snapshot = api.preflight.FrozenControllerSnapshot(
        root=snapshot_root,
        directories={
            name: snapshot_root / "controllers" / name for name in api.benchmark.M8_CONTROLLER_ORDER
        },
        identities=identities,
    )
    api.preflight.validate_frozen_controller_snapshot(snapshot, config)
    return snapshot


def _assert_active_snapshot_absent(project_root: Path) -> None:
    try:
        (project_root / SNAPSHOT_RELATIVE_PATH).lstat()
    except FileNotFoundError:
        return
    raise RuntimeError("an active Controller snapshot without its transaction is unsafe")


def _recover_pre_test_attempt(
    api: SimpleNamespace,
    transaction: object,
    project_root: Path,
    *,
    recovery_lockdown: bool,
) -> bool:
    """Recover PREPARED only; incomplete Test-bound state always raises and remains."""

    inspection = transaction.inspect()
    if not inspection.exists:
        if recovery_lockdown:
            raise RuntimeError("the locked formal transaction disappeared before recovery")
        return False
    if not recovery_lockdown:
        raise RuntimeError("existing formal transaction was not locked before dependency imports")
    if inspection.phase is not api.attempt.AttemptPhase.COMMITTED:
        api.preflight.require_controller_snapshot_quarantine_absent(
            project_root,
            committed_relative_path=COMMITTED_SNAPSHOT_RELATIVE_PATH,
        )
    if inspection.phase is api.attempt.AttemptPhase.PREPARED:
        api.preflight.isolate_aborted_controller_snapshot(
            project_root,
            relative_path=SNAPSHOT_RELATIVE_PATH,
        )
        recovery = transaction.recover()
        if recovery.action != "pre_test_restored":
            raise RuntimeError("PREPARED recovery did not restore the pre-Test state")
        raise PreparedRecoveryRerunRequired(
            "PREPARED formal state was cleaned under recovery lockdown; rerun is required"
        )
    if (
        inspection.phase is api.attempt.AttemptPhase.TEST_BOUND
        and inspection.journal_record_count < api.attempt.FORMAL_EPISODE_COUNT
    ):
        raise api.attempt.IncompleteTestAttemptError(inspection)
    return False


def _validate_static_stability(
    api: SimpleNamespace,
    project_root: Path,
    static: _StaticInputs,
    source_identities: Mapping[str, object],
    *,
    require_clean: bool,
) -> None:
    _assert_project_source_finder_active(project_root)
    _validate_project_module_provenance(project_root)
    revision = _python_git_revision(project_root)
    if revision != static.source_revision:
        raise RuntimeError("source revision changed during formal M8")
    source_tree_identity = api.source_tree.capture_source_tree_identity(project_root)
    if (
        source_tree_identity != static.source_tree_identity
        or source_tree_identity.aggregate_sha256 != static.source_tree_sha256
    ):
        raise RuntimeError("the deterministic formal source tree changed during M8")
    config_sha, _size, config_bytes = api.preflight.sha256_regular_file(static.config_path)
    lock_sha, _lock_size, lock_bytes = api.preflight.sha256_regular_file(project_root / "pixi.lock")
    if (
        config_sha != static.config_sha256
        or config_bytes != static.config_bytes
        or lock_sha != static.pixi_lock_sha256
        or lock_bytes != static.pixi_lock_bytes
    ):
        raise RuntimeError("the formal config or Pixi lock changed during M8")
    reports = api.preflight.load_frozen_input_reports(project_root, static.config)
    if api.preflight.frozen_input_digest(reports) != static.reports_digest:
        raise RuntimeError("a frozen M5/M6/M7 input report changed during M8")
    for name in api.benchmark.M8_CONTROLLER_ORDER:
        observed = api.controller_identity.capture_frozen_controller_identity(project_root, name)
        if (
            observed != source_identities[name]
            or observed.aggregate_sha256 != static.config.controller_aggregate_sha256[name]
            or observed.config_sha256 != static.config.controller_config_sha256[name]
        ):
            raise RuntimeError(f"source Controller {name!r} changed during M8")
    if require_clean:
        if _POST_BIND_COMMANDS_FORBIDDEN:
            raise RuntimeError("clean Git subprocess checks are forbidden after TEST_BOUND")
        clean = api.preflight.capture_clean_source(
            project_root,
            command_runner=_secure_git_command_runner,
        )
        if clean.revision != static.source_revision:
            raise RuntimeError("clean-source evidence changed during M8")
    _assert_project_source_finder_active(project_root)
    _validate_project_module_provenance(project_root)
    _assert_torch_absent()


def _pool_access_from_mapping(api: SimpleNamespace, value: Mapping[str, object]) -> object:
    return api.test_assets.TestPoolAccessEvidence.from_mapping(value)


def _evidence_from_mapping(
    evidence_type: type, value: Mapping[str, object], *, label: str
) -> object:
    factory = getattr(evidence_type, "from_mapping", None)
    if not callable(factory):
        raise RuntimeError(f"{label} report evidence does not expose from_mapping()")
    return factory(value)


def _load_durable_evidence(api: SimpleNamespace, transaction: object) -> _DurableEvidence:
    seal_mapping = transaction.read_execution_evidence()
    expected = {
        "asset_access",
        "execution",
        "memory",
        "runtime",
        "schema_version",
        "test_assets",
    }
    if set(seal_mapping) != expected:
        raise RuntimeError("durable execution-evidence seal keys differ")
    mappings = {
        "test_pool_access": seal_mapping["test_assets"],
        "test_access_audit": seal_mapping["asset_access"],
        "runtime": seal_mapping["runtime"],
        "memory": seal_mapping["memory"],
        "execution": seal_mapping["execution"],
    }
    if any(not isinstance(value, Mapping) for value in mappings.values()):
        raise RuntimeError("durable report evidence contains a non-object value")
    audit_mapping = dict(mappings["test_access_audit"])
    audit_schema = audit_mapping.pop("schema_version", None)
    if audit_schema != api.report.M8_TEST_ACCESS_AUDIT_SCHEMA_VERSION:
        raise RuntimeError("durable Test access audit schema differs")
    return _DurableEvidence(
        test_pool_access=_pool_access_from_mapping(api, mappings["test_pool_access"]),
        test_access_audit=api.report.TestAccessAuditEvidence.from_mapping(audit_mapping),
        runtime=_evidence_from_mapping(
            api.report.RuntimeEvidence,
            mappings["runtime"],
            label="runtime",
        ),
        memory=_evidence_from_mapping(
            api.report.MemoryEvidence,
            mappings["memory"],
            label="memory",
        ),
        execution=api.report.ExecutionEvidence.from_mapping(mappings["execution"]),
        durable_execution_evidence=api.report.DurableExecutionEvidenceSeal.from_mapping(
            seal_mapping
        ),
    )


def _journal_record(
    api: SimpleNamespace,
    controller_name: str,
    row_index: int,
    recorded: object,
    trajectory_payload: bytes,
) -> object:
    final_info = recorded.trajectory.final_info
    reset_info = recorded.trajectory.reset_info
    outcomes = {1: "success", 2: "off_track", 3: "invalid_action", 4: "timeout"}
    try:
        outcome = outcomes[int(final_info["termination_reason"])]
    except (KeyError, TypeError, ValueError) as error:
        raise RuntimeError("canonical episode has an invalid terminal outcome") from error
    return api.attempt.EpisodeJournalRecord(
        controller=controller_name,
        row_index=row_index,
        track_id=int(reset_info["track_id"]),
        reset_seed=row_index,
        episode_seed=int(reset_info["episode_seed"]),
        controller_seed=int(reset_info["controller_seed"]),
        outcome=outcome,
        steps=recorded.result.steps,
        trajectory_blob_path=(f"episodes/{controller_name}/row_{row_index:03d}_trajectory.json"),
        trajectory_blob_sha256=_sha256(trajectory_payload),
        trajectory_blob_size_bytes=len(trajectory_payload),
        data={
            "compute_times_s": [float(value) for value in recorded.result.compute_times_s],
            "controller_import_time_s": float(recorded.result.controller_import_time_s),
            "controller_init_time_s": float(recorded.result.controller_init_time_s),
        },
    )


def _episode_bundle_sink(
    api: SimpleNamespace,
    transaction: object,
    snapshot: object,
    config: object,
    memory_recorder: object,
):
    """Return the sink that durably commits each canonical bundle before the next row."""

    def persist(
        controller_name: str,
        row_index: int,
        recorded: object,
        _metric_samples: object,
    ) -> None:
        api.preflight.validate_frozen_controller_snapshot(snapshot, config)
        trajectory_payload = api.trajectory.canonical_trajectory_json_bytes(recorded.trajectory)
        record = _journal_record(
            api,
            controller_name,
            row_index,
            recorded,
            trajectory_payload,
        )
        transaction.append_episode_bundle(record, trajectory_payload)
        if row_index == config.test_track_count - 1:
            memory_recorder.sample(f"after_{controller_name}_controller")

    return persist


def _sanitize_failure_text(value: object, *, maximum: int) -> str:
    text = "".join(
        character if character in "\n\t" or character.isprintable() else "?"
        for character in str(value)
    )
    text = _WINDOWS_ABSOLUTE_PATH.sub("<path>", text)
    text = _POSIX_ABSOLUTE_PATH.sub("<path>", text)
    text = _SECRET_SHAPE.sub("<redacted>", text).strip()
    return text[:maximum]


def _write_sanitized_infrastructure_failure(
    transaction: object,
    error: BaseException,
    *,
    infrastructure_phase: str,
) -> None:
    if re.fullmatch(r"[a-z][a-z0-9_]{0,63}", infrastructure_phase) is None:
        raise ValueError("infrastructure failure phase must be a canonical identifier")
    controller = getattr(error, "controller_name", None)
    row_index = getattr(error, "row_index", None)
    workload_phase = getattr(error, "phase", None)
    sanitized_traceback = getattr(error, "sanitized_traceback", None)
    workload = None
    if (
        isinstance(controller, str)
        and type(row_index) is int
        and isinstance(workload_phase, str)
        and isinstance(sanitized_traceback, str)
    ):
        workload = {
            "controller": _sanitize_failure_text(controller, maximum=32),
            "phase": _sanitize_failure_text(workload_phase, maximum=64),
            "row_index": row_index,
            "sanitized_traceback": _sanitize_failure_text(
                sanitized_traceback,
                maximum=_FAILURE_TRACEBACK_MAX_CHARS,
            ),
        }
    payload = _canonical_json_bytes(
        {
            "cause_type": type(error).__name__,
            "detail": _sanitize_failure_text(error, maximum=_FAILURE_DETAIL_MAX_CHARS),
            "infrastructure_phase": infrastructure_phase,
            "schema_version": _FAILURE_EVIDENCE_SCHEMA,
            "workload": workload,
        }
    )
    with contextlib.suppress(BaseException):
        transaction.write_blob(FAILURE_EVIDENCE_BLOB, payload)


def _recorded_episode_from_journal(
    api: SimpleNamespace,
    transaction: object,
    record: object,
) -> object:
    """Reconstruct a Runner result and recording without executing or replaying an environment."""

    trajectory_payload = transaction.read_blob(record.trajectory_blob_path)
    trajectory = api.trajectory.load_trajectory_json_bytes(
        trajectory_payload,
        expected_sha256=record.trajectory_blob_sha256,
    )
    result = api.control.EpisodeRunResult(
        steps=record.steps,
        total_reward=trajectory.total_reward,
        terminated=bool(trajectory.terminated[-1]),
        truncated=bool(trajectory.truncated[-1]),
        final_info=trajectory.final_info,
        debug_commands=(),
        controller_import_time_s=record.data["controller_import_time_s"],
        controller_init_time_s=record.data["controller_init_time_s"],
        compute_times_s=tuple(record.data["compute_times_s"]),
    )
    return api.trajectory.RecordedControllerEpisode(result=result, trajectory=trajectory)


def _results_from_durable_journal(
    api: SimpleNamespace, transaction: object
) -> Mapping[str, object]:
    records = transaction.episode_records()
    if len(records) != api.attempt.FORMAL_EPISODE_COUNT:
        raise RuntimeError("formal result reconstruction requires exactly 60 journal records")
    grouped: dict[str, list[tuple[object, object]]] = {
        name: [] for name in api.benchmark.M8_CONTROLLER_ORDER
    }
    for record in records:
        recorded = _recorded_episode_from_journal(api, transaction, record)
        samples = api.metrics.compute_episode_metric_samples(
            recorded,
            reset_seed=record.row_index,
            action_limits=api.results.FINAL_RESULTS_ACTION_LIMITS,
        )
        grouped[record.controller].append((recorded, samples))
    results: dict[str, object] = {}
    for name in api.benchmark.M8_CONTROLLER_ORDER:
        pairs = grouped[name]
        if len(pairs) != api.benchmark.M8_TEST_TRACK_COUNT:
            raise RuntimeError("durable journal does not contain 20 rows per Controller")
        metrics = api.metrics.build_final_metrics_data(
            name,
            tuple(samples for _recorded, samples in pairs),
        )
        episodes = tuple(
            api.results.FinalEpisodeResult(name, row_index, recorded, samples)
            for row_index, (recorded, samples) in enumerate(pairs)
        )
        results[name] = api.results.FinalControllerResult(
            controller_name=name,
            episodes=episodes,
            metrics=metrics,
        )
    return MappingProxyType(results)


def _input_artifact_records(
    api: SimpleNamespace,
    static: _StaticInputs,
) -> tuple[object, Mapping[str, object]]:
    pixi_lock = api.report.ArtifactRecord.from_bytes(
        "pixi.lock",
        static.pixi_lock_bytes,
        "application/yaml",
    )
    reports: dict[str, object] = {}
    for name, report in static.reports.items():
        schema = report.payload.get("schema_version")
        reports[name] = api.report.ArtifactRecord(
            relative_path=report.relative_path,
            sha256=report.sha256,
            size_bytes=report.size_bytes,
            media_type="application/json",
            schema_version=schema if isinstance(schema, (str, int)) else None,
        )
    return pixi_lock, MappingProxyType(reports)


def _artifact_record(api: SimpleNamespace, path: str, payload: bytes) -> object:
    if path.endswith("/metrics.npz"):
        metadata = ("application/x-npz", api.metrics.FINAL_METRICS_SCHEMA_VERSION)
    elif path.endswith("/results.csv"):
        metadata = ("text/csv", api.results.FINAL_CONTROLLER_RESULTS_SCHEMA_VERSION)
    elif path.endswith("/summary.json"):
        metadata = ("application/json", api.results.FINAL_CONTROLLER_SUMMARY_SCHEMA_VERSION)
    elif path.endswith("/run_manifest.json"):
        metadata = ("application/json", api.report.M8_CONTROLLER_RUN_MANIFEST_SCHEMA_VERSION)
    elif path.endswith("/selected_replays/test_row_000_trajectory.json"):
        metadata = ("application/json", api.trajectory.TRAJECTORY_SCHEMA_VERSION)
    elif path.endswith(".png"):
        metadata = ("image/png", None)
    elif path.endswith("m8_final_results.csv"):
        metadata = ("text/csv", api.results.FINAL_COMPARISON_SCHEMA_VERSION)
    elif path.endswith("m8_final_evaluation_report.json"):
        metadata = ("application/json", api.benchmark.M8_FINAL_REPORT_SCHEMA_VERSION)
    else:
        raise RuntimeError(f"unknown formal artifact path {path!r}")
    return api.report.ArtifactRecord.from_bytes(path, payload, *metadata)


def _build_outputs(
    api: SimpleNamespace,
    transaction: object,
    static: _StaticInputs,
    results: Mapping[str, object],
    durable: _DurableEvidence,
    identities_before: Mapping[str, object],
    identities_after: Mapping[str, object],
) -> Mapping[str, bytes]:
    config = static.config
    source = api.report.SourceEvidence(revision=static.source_revision)
    config_evidence = api.report.FinalConfigEvidence.from_bytes(config, static.config_bytes)
    pixi_lock, input_reports = _input_artifact_records(api, static)
    transaction_evidence = api.report.TransactionEvidence()
    privacy = api.report.PrivacyEvidence()
    outputs: dict[str, bytes] = {}

    row_zero_trajectories: dict[str, object] = {}
    for name in api.benchmark.M8_CONTROLLER_ORDER:
        result = results[name]
        paths = api.benchmark.controller_output_paths(config, name)
        row_zero_record = next(
            record
            for record in transaction.episode_records()
            if record.controller == name and record.row_index == config.replay_test_row_index
        )
        row_zero_payload = transaction.read_blob(row_zero_record.trajectory_blob_path)
        trajectory = api.trajectory.load_trajectory_json_bytes(
            row_zero_payload,
            expected_sha256=row_zero_record.trajectory_blob_sha256,
        )
        row_zero_trajectories[name] = trajectory
        samples = result.metrics.episode(config.replay_test_row_index)
        outputs[paths["metrics"]] = api.metrics.canonical_final_metrics_bytes(result.metrics)
        outputs[paths["replay_trajectory"]] = row_zero_payload
        outputs[paths["results"]] = api.results.canonical_controller_results_csv_bytes(result)
        outputs[paths["summary"]] = api.results.canonical_controller_summary_json_bytes(result)
        outputs[paths["trajectory"]] = api.visual_replay.render_trajectory_overview_png(trajectory)
        outputs[paths["telemetry"]] = api.visual_final.render_controller_telemetry_png(
            controller_name=name,
            control_dt_s=config.control_dt_s,
            speed_mps=samples.speed_mps,
            lateral_error_m=samples.lateral_error_m,
            requested_action=samples.requested_action,
            steering_saturated=samples.steering_saturated,
            longitudinal_saturated=samples.longitudinal_saturated,
        )

    outputs[config.comparison_csv_path] = api.results.canonical_final_comparison_csv_bytes(results)
    first = row_zero_trajectories[api.benchmark.M8_CONTROLLER_ORDER[0]]
    outputs[config.comparison_png_path] = api.visual_final.render_final_comparison_png(
        benchmark_version=config.benchmark_version,
        track_id=int(first.reset_info["track_id"]),
        centerline_m=first.centerline_m,
        left_boundary_m=first.left_boundary_m,
        right_boundary_m=first.right_boundary_m,
        track_mask=first.track_mask,
        trajectories_m={
            name: row_zero_trajectories[name].position_m
            for name in api.benchmark.M8_CONTROLLER_ORDER
        },
    )

    for name in api.benchmark.M8_CONTROLLER_ORDER:
        paths = api.benchmark.controller_output_paths(config, name)
        artifact_records = {
            key: _artifact_record(api, path, outputs[path])
            for key, path in paths.items()
            if key != "run_manifest"
        }
        outputs[paths["run_manifest"]] = api.report.canonical_controller_run_manifest_json_bytes(
            results[name],
            source=source,
            protocol_config=config,
            config_evidence=config_evidence,
            pixi_lock=pixi_lock,
            input_reports=input_reports,
            controller_identity=identities_after[name],
            test_pool_access=durable.test_pool_access,
            test_access_audit=durable.test_access_audit,
            runtime=durable.runtime,
            memory=durable.memory,
            execution=durable.execution,
            durable_execution_evidence=durable.durable_execution_evidence,
            output_artifacts=artifact_records,
        )

    global_artifacts = {
        path: _artifact_record(api, path, payload)
        for path, payload in outputs.items()
        if path != config.report_path
    }
    clean_identities_after = {
        name: identities_after[name] for name in api.benchmark.M8_CONTROLLER_ORDER
    }
    outputs[config.report_path] = api.report.canonical_m8_final_report_json_bytes(
        results,
        source=source,
        protocol_config=config,
        config_evidence=config_evidence,
        pixi_lock=pixi_lock,
        input_reports=input_reports,
        replacement_failure_report=static.reports["m8_attempt_001_failure_report"].payload,
        controller_identities_before=identities_before,
        controller_identities_after=clean_identities_after,
        test_pool_access=durable.test_pool_access,
        test_access_audit=durable.test_access_audit,
        runtime=durable.runtime,
        memory=durable.memory,
        execution=durable.execution,
        durable_execution_evidence=durable.durable_execution_evidence,
        transaction=transaction_evidence,
        privacy=privacy,
        output_artifacts=global_artifacts,
    )
    expected = set(api.benchmark.formal_output_paths(config))
    if set(outputs) != expected or len(outputs) != 24:
        raise RuntimeError("formal output construction did not produce exactly 24 paths")
    return MappingProxyType(outputs)


def _publication_evidence_kwargs(
    api: SimpleNamespace,
    static: _StaticInputs,
    durable: _DurableEvidence,
    identities_before: Mapping[str, object],
    identities_after: Mapping[str, object],
) -> dict[str, object]:
    pixi_lock, input_reports = _input_artifact_records(api, static)
    return {
        "source": api.report.SourceEvidence(revision=static.source_revision),
        "protocol_config": static.config,
        "config_evidence": api.report.FinalConfigEvidence.from_bytes(
            static.config, static.config_bytes
        ),
        "pixi_lock": pixi_lock,
        "input_reports": input_reports,
        "replacement_failure_report": static.reports["m8_attempt_001_failure_report"].payload,
        "controller_identities_before": identities_before,
        "controller_identities_after": identities_after,
        "test_pool_access": durable.test_pool_access,
        "test_access_audit": durable.test_access_audit,
        "runtime": durable.runtime,
        "memory": durable.memory,
        "execution": durable.execution,
        "durable_execution_evidence": durable.durable_execution_evidence,
        "transaction": api.report.TransactionEvidence(),
        "privacy": api.report.PrivacyEvidence(),
    }


def _validate_stage_then_publish(
    api: SimpleNamespace,
    transaction: object,
    static: _StaticInputs,
    results: Mapping[str, object],
    durable: _DurableEvidence,
    identities_before: Mapping[str, object],
    identities_after: Mapping[str, object],
) -> tuple[object, ...]:
    staged = transaction.read_staged_outputs()
    semantic_digest = api.report.validate_m8_publication(
        staged,
        results,
        **_publication_evidence_kwargs(
            api,
            static,
            durable,
            identities_before,
            identities_after,
        ),
    )
    inspection = transaction.inspect()
    if inspection.phase is api.attempt.AttemptPhase.EVALUATION_COMPLETE:
        transaction.mark_artifacts_validated(
            semantic_validation_sha256=semantic_digest,
        )
    elif inspection.phase is not api.attempt.AttemptPhase.ARTIFACTS_VALIDATED:
        raise RuntimeError("semantic publication validation ran in an invalid phase")
    # Revalidate the exact staged bytes even when recovering ARTIFACTS_VALIDATED.
    second_digest = api.report.validate_m8_publication(
        transaction.read_staged_outputs(),
        results,
        **_publication_evidence_kwargs(
            api,
            static,
            durable,
            identities_before,
            identities_after,
        ),
    )
    if second_digest != semantic_digest:
        raise RuntimeError("semantic publication digest changed before publish")
    return transaction.publish_and_commit(retain_committed_transaction=True)


def _resume_attempt(
    api: SimpleNamespace,
    project_root: Path,
    static: _StaticInputs,
    transaction: object,
) -> Mapping[str, object]:
    _enter_post_bind_phase()
    inspection = transaction.inspect()
    if not inspection.exists:
        raise RuntimeError("attempt recovery requested without a transaction")
    if inspection.phase is not api.attempt.AttemptPhase.COMMITTED:
        api.preflight.require_controller_snapshot_quarantine_absent(
            project_root,
            committed_relative_path=COMMITTED_SNAPSHOT_RELATIVE_PATH,
        )
    if (
        inspection.phase is api.attempt.AttemptPhase.TEST_BOUND
        and inspection.journal_record_count < api.attempt.FORMAL_EPISODE_COUNT
    ):
        raise api.attempt.IncompleteTestAttemptError(inspection)
    if inspection.journal_record_count != api.attempt.FORMAL_EPISODE_COUNT:
        raise RuntimeError("recoverable Test-bound attempt must contain exactly 60 records")
    if inspection.phase is api.attempt.AttemptPhase.TEST_BOUND:
        recovery = transaction.recover()
        if recovery.action != "evaluation_complete_ready":
            raise RuntimeError("sealed TEST_BOUND recovery action differs")
    elif inspection.phase is api.attempt.AttemptPhase.ARTIFACTS_VALIDATED:
        recovery = transaction.recover()
        if recovery.action not in {"artifacts_validated_ready", "partial_publication_restored"}:
            raise RuntimeError("ARTIFACTS_VALIDATED recovery action differs")
    source_identities = {
        name: api.controller_identity.capture_frozen_controller_identity(project_root, name)
        for name in api.benchmark.M8_CONTROLLER_ORDER
    }
    _validate_static_stability(
        api,
        project_root,
        static,
        source_identities,
        require_clean=False,
    )
    snapshot = None
    if inspection.phase is api.attempt.AttemptPhase.COMMITTED:
        # The snapshot may already have been removed before a crash in the outer-gate window.
        # Source identities are independently hash-bound by the frozen config and transaction.
        identities_before = dict(source_identities)
        identities_after = dict(source_identities)
    else:
        snapshot = _snapshot_from_existing(api, project_root, static.config)
        identities_before = dict(snapshot.identities)
        identities_after = {
            name: api.controller_identity.capture_frozen_controller_identity(snapshot.root, name)
            for name in api.benchmark.M8_CONTROLLER_ORDER
        }
    results = _results_from_durable_journal(api, transaction)
    durable = _load_durable_evidence(api, transaction)

    if inspection.phase is api.attempt.AttemptPhase.TEST_BOUND:
        outputs = _build_outputs(
            api,
            transaction,
            static,
            results,
            durable,
            identities_before,
            identities_after,
        )
        transaction.complete_evaluation(outputs)
    elif inspection.phase is api.attempt.AttemptPhase.COMMITTED:
        staged = transaction.read_staged_outputs()
        digest = api.report.validate_m8_publication(
            staged,
            results,
            **_publication_evidence_kwargs(
                api,
                static,
                durable,
                identities_before,
                identities_after,
            ),
        )
        if not digest:
            raise RuntimeError("committed semantic validation did not return a digest")
        api.preflight.retire_committed_controller_snapshot(
            project_root,
            relative_path=SNAPSHOT_RELATIVE_PATH,
            committed_relative_path=COMMITTED_SNAPSHOT_RELATIVE_PATH,
        )
        return results
    elif inspection.phase is api.attempt.AttemptPhase.ARTIFACTS_VALIDATED:
        pass
    elif inspection.phase is not api.attempt.AttemptPhase.EVALUATION_COMPLETE:
        raise RuntimeError("attempt phase is not recoverable without rerunning Test")

    _validate_static_stability(
        api,
        project_root,
        static,
        source_identities,
        require_clean=False,
    )
    if snapshot is None:  # pragma: no cover - all non-COMMITTED phases require it
        raise RuntimeError("recoverable pre-publication attempt lost its Controller snapshot")
    api.preflight.validate_frozen_controller_snapshot(snapshot, static.config)
    _validate_stage_then_publish(
        api,
        transaction,
        static,
        results,
        durable,
        identities_before,
        identities_after,
    )
    api.preflight.retire_committed_controller_snapshot(
        project_root,
        relative_path=SNAPSHOT_RELATIVE_PATH,
        committed_relative_path=COMMITTED_SNAPSHOT_RELATIVE_PATH,
    )
    return results


def _fresh_attempt(
    api: SimpleNamespace,
    guard: object,
    project_root: Path,
    static: _StaticInputs,
    transaction: object,
) -> Mapping[str, object]:
    config = static.config
    api.preflight.require_controller_snapshot_quarantine_absent(
        project_root,
        committed_relative_path=COMMITTED_SNAPSHOT_RELATIVE_PATH,
    )
    api.benchmark.validate_formal_output_tree(project_root, config, expected_present=False)
    clean_source = api.preflight.capture_clean_source(
        project_root,
        command_runner=_secure_git_command_runner,
    )
    if clean_source.revision != static.source_revision:
        raise RuntimeError("clean source differs from transaction identity")
    source_identities = {
        name: api.controller_identity.capture_frozen_controller_identity(project_root, name)
        for name in api.benchmark.M8_CONTROLLER_ORDER
    }
    _assert_torch_absent()

    predecessor_report_path = project_root / config.replacement_failure_report_path
    predecessor_validation = api.replacement.validate_local_predecessor(
        project_root,
        predecessor_report_path,
        expected_sha256=config.replacement_failure_report_sha256,
    )
    if (
        predecessor_validation.eligible is not True
        or predecessor_validation.report_sha256 != config.replacement_failure_report_sha256
        or predecessor_validation.successor_run_id != config.run_id
    ):
        raise RuntimeError("attempt 001 is not eligible for the frozen replacement attempt")

    transaction.prepare()
    try:
        snapshot = api.preflight.create_frozen_controller_snapshot(
            project_root,
            config,
            relative_path=SNAPSHOT_RELATIVE_PATH,
        )
    except BaseException:
        api.preflight.isolate_aborted_controller_snapshot(
            project_root,
            relative_path=SNAPSHOT_RELATIVE_PATH,
        )
        transaction.recover()
        raise
    api.preflight.validate_frozen_controller_snapshot(snapshot, config)
    _validate_static_stability(
        api,
        project_root,
        static,
        source_identities,
        require_clean=True,
    )

    project_config = api.config.load_project_config(project_root)
    if (
        project_config.benchmark.version != config.benchmark_version
        or project_config.benchmark.test_track_count != config.test_track_count
    ):
        raise RuntimeError("project config differs from the frozen final benchmark")
    nvidia_smi_executable = api.runtime.resolve_nvidia_smi_executable()
    frozen_nvidia_smi = guard.freeze_nvidia_smi_executable(nvidia_smi_executable)
    runtime_mapping, private_gpu_uuid = api.runtime.collect_final_runtime_evidence(
        api.jax,
        frozen_nvidia_smi,
    )
    _initialize_warp_runtime(api)
    device = api.jax.devices("gpu")[0]
    memory_recorder = api.runtime.FinalMemoryRecorder(
        api.jax,
        device,
        private_gpu_uuid,
        frozen_nvidia_smi,
        command_runner=guard.run_frozen_memory_query,
    )

    pre_access = guard.evidence(test_loaded=False)
    if (
        pre_access["pre_test_open_event_count"] != 0
        or pre_access["denied_event_count"] != 0
        or pre_access["denied_mutation_event_count"] != 0
        or pre_access["open_event_counts"] != {}
    ):
        raise RuntimeError("Test guard evidence is nonzero before durable TEST_BOUND")
    _enter_post_bind_phase()
    infrastructure_phase = "bind_test"
    # The sanitized failure blob covers the TEST_BOUND interval through durable artifact staging.
    # Once EVALUATION_COMPLETE is durable, later failures recover deterministically from staged
    # bytes and do not require another mutable failure blob.
    try:
        transaction.bind_test()
        infrastructure_phase = "enable_test_reads"
        guard.enable_test_reads()
        infrastructure_phase = "load_test_pool"
        verified_test = api.test_assets.load_verified_test_pool(project_config)
        if (
            verified_test.evidence.manifest_sha256 != config.test_manifest_sha256
            or verified_test.evidence.asset_file_sha256 != config.test_asset_sha256
            or verified_test.evidence.track_count != config.test_track_count
        ):
            raise RuntimeError("loaded Test pool differs from the precommitted M5/config identity")
        infrastructure_phase = "forbid_track_reads"
        guard.forbid_all_track_reads()
        access_mapping = guard.evidence(test_loaded=True)
        api.report.TestAccessAuditEvidence.from_mapping(access_mapping)

        infrastructure_phase = "memory_before_environment"
        memory_recorder.sample("before_environment_create")
        measured_factory = api.runtime.MeasuredFinalEnvironmentFactory(api.environment.CarRacingEnv)
        environment = None
        workload = None
        try:
            infrastructure_phase = "environment_create"
            environment = measured_factory.create(
                project_config=project_config,
                level_id=config.level_id,
                backend=config.backend,
                track_pool=verified_test.pool,
                render_mode=None,
            )
            infrastructure_phase = "controller_workload"
            workload = api.execution.execute_controller_workload(
                environment=environment,
                controller_directories=snapshot.directories,
                track_ids=verified_test.evidence.track_ids,
                action_limits=api.results.FINAL_RESULTS_ACTION_LIMITS,
                max_episode_steps=config.max_episode_steps,
                episode_sink=_episode_bundle_sink(
                    api,
                    transaction,
                    snapshot,
                    config,
                    memory_recorder,
                ),
                expected_track_count=config.test_track_count,
            )
        finally:
            if environment is not None:
                prior_phase = infrastructure_phase
                infrastructure_phase = "environment_close"
                environment.close()
                infrastructure_phase = prior_phase
        if workload is None:
            raise RuntimeError("formal workload returned no complete execution evidence")

        infrastructure_phase = "environment_lifecycle"
        total_steps = sum(
            value.result.steps
            for controller in workload.controller_results.values()
            for value in controller.recorded_episodes
        )
        lifecycle_mapping = measured_factory.evidence(
            expected_resets=api.attempt.FORMAL_EPISODE_COUNT,
            expected_steps=total_steps,
        )
        infrastructure_phase = "final_memory"
        memory_recorder.sample("after_environment_close")
        memory_mapping = memory_recorder.evidence()
        infrastructure_phase = "final_asset_audit"
        final_access_mapping = guard.evidence(test_loaded=True)
        if final_access_mapping != access_mapping:
            raise RuntimeError("official Test assets were reopened after the one Test-pool load")

        infrastructure_phase = "typed_execution_evidence"
        test_access_audit = api.report.TestAccessAuditEvidence.from_mapping(final_access_mapping)
        results = _results_from_durable_journal(api, transaction)
        runtime = api.report.RuntimeEvidence.from_mapping(runtime_mapping)
        memory = api.report.MemoryEvidence.from_mapping(memory_mapping)
        lifecycle = api.report.EnvironmentLifecycleEvidence.from_mapping(lifecycle_mapping)
        execution = api.report.ExecutionEvidence.from_workload(
            workload,
            results,
            measured_environment_lifecycle=lifecycle,
        )
        identities_after = {
            name: api.controller_identity.capture_frozen_controller_identity(snapshot.root, name)
            for name in api.benchmark.M8_CONTROLLER_ORDER
        }
        if identities_after != dict(snapshot.identities):
            raise RuntimeError("frozen Controller snapshot changed during Test execution")
        _assert_torch_absent()
        _validate_static_stability(
            api,
            project_root,
            static,
            source_identities,
            require_clean=False,
        )

        infrastructure_phase = "execution_evidence_seal"
        durable_seal = api.report.DurableExecutionEvidenceSeal.from_evidence(
            test_access_audit=test_access_audit,
            execution=execution,
            memory=memory,
            runtime=runtime,
            test_pool_access=verified_test.evidence,
        )
        transaction.write_execution_evidence(
            api.attempt.canonical_execution_evidence_bytes(durable_seal.to_dict())
        )
        durable = _load_durable_evidence(api, transaction)

        infrastructure_phase = "artifact_construction"
        outputs = _build_outputs(
            api,
            transaction,
            static,
            results,
            durable,
            snapshot.identities,
            identities_after,
        )
        infrastructure_phase = "artifact_staging"
        transaction.complete_evaluation(outputs)
    except BaseException as error:
        _write_sanitized_infrastructure_failure(
            transaction,
            error,
            infrastructure_phase=infrastructure_phase,
        )
        raise
    api.preflight.validate_frozen_controller_snapshot(snapshot, config)
    _validate_static_stability(
        api,
        project_root,
        static,
        source_identities,
        require_clean=False,
    )
    # Reconstruct again after staging so validation cannot accidentally trust workload objects.
    reconstructed_results = _results_from_durable_journal(api, transaction)
    reconstructed_evidence = _load_durable_evidence(api, transaction)
    _validate_stage_then_publish(
        api,
        transaction,
        static,
        reconstructed_results,
        reconstructed_evidence,
        snapshot.identities,
        identities_after,
    )
    api.preflight.retire_committed_controller_snapshot(
        project_root,
        relative_path=SNAPSHOT_RELATIVE_PATH,
        committed_relative_path=COMMITTED_SNAPSHOT_RELATIVE_PATH,
    )
    return reconstructed_results


def _success_summary(api: SimpleNamespace, results: Mapping[str, object]) -> Mapping[str, object]:
    rank_order = api.results.rank_final_controller_results(results)
    return {
        "benchmark_version": "0.1",
        "controller_order": list(api.benchmark.M8_CONTROLLER_ORDER),
        "rank_order": list(rank_order),
        "run_id": api.benchmark.M8_FINAL_RUN_ID,
        "status": "passed",
        "success_count": {
            name: results[name].summary.success_count for name in api.benchmark.M8_CONTROLLER_ORDER
        },
    }


def run_benchmark(
    options: BenchmarkOptions,
    *,
    project_root: Path = PROJECT_ROOT,
) -> Mapping[str, object]:
    root = _canonical_project_root(project_root)
    guard = _consume_bootstrap_guard(root)
    _remove_project_import_root(root)
    _recover_prepared_attempt_before_dependencies(
        root,
        recovery_lockdown=_EARLY_RECOVERY_LOCKDOWN,
    )
    _install_project_source_finder(root)
    _configure_gpu_site_packages(root)
    config_path = _canonical_config_path(root, options.config)
    api = _load_project_api(root)
    _remove_project_import_root(root)
    _assert_project_source_finder_active(root)
    _validate_project_module_provenance(root)
    _assert_torch_absent()
    static = _load_static_inputs(api, root, config_path)
    transaction = _attempt_transaction(api, root, static)

    _recover_pre_test_attempt(
        api,
        transaction,
        root,
        recovery_lockdown=_EARLY_RECOVERY_LOCKDOWN,
    )
    inspection = transaction.inspect()
    if inspection.exists:
        results = _resume_attempt(api, root, static, transaction)
    else:
        api.preflight.require_controller_snapshot_quarantine_absent(
            root,
            committed_relative_path=COMMITTED_SNAPSHOT_RELATIVE_PATH,
        )
        _assert_active_snapshot_absent(root)
        results = _fresh_attempt(api, guard, root, static, transaction)

    api.benchmark.validate_formal_output_tree(root, static.config, expected_present=True)
    post_publication_identities = {
        name: api.controller_identity.capture_frozen_controller_identity(root, name)
        for name in api.benchmark.M8_CONTROLLER_ORDER
    }
    _validate_static_stability(
        api,
        root,
        static,
        post_publication_identities,
        require_clean=False,
    )
    api.preflight.validate_committed_controller_snapshot_quarantine(
        root,
        relative_path=SNAPSHOT_RELATIVE_PATH,
        committed_relative_path=COMMITTED_SNAPSHOT_RELATIVE_PATH,
    )
    _assert_torch_absent()
    committed = transaction.inspect()
    if (
        not committed.exists
        or committed.phase is not api.attempt.AttemptPhase.COMMITTED
        or committed.output_state != "published"
    ):
        raise RuntimeError("outer formal gates require a durable COMMITTED transaction")
    return _success_summary(api, results)


def main(argv: Sequence[str] | None = None) -> int:
    options = _parse_args(argv)
    _prepare_isolated_python_runtime(PROJECT_ROOT)
    _bootstrap_test_guard(PROJECT_ROOT)
    summary = run_benchmark(options, project_root=PROJECT_ROOT)
    print(json.dumps(summary, allow_nan=False, ensure_ascii=True, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
