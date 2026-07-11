"""Process-wide Test-only filesystem guard for the one-shot M8 evaluation."""

from __future__ import annotations

import hashlib
import os
import stat
import subprocess
import sys
import threading
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, ClassVar


class ForbiddenFinalEvaluationAssetAccessError(RuntimeError):
    """A formal M8 process attempted an unapproved Track-asset operation."""


@dataclass(frozen=True, slots=True)
class _FrozenExecutableIdentity:
    """Private identity of the sole executable allowed after the Test pool is in memory."""

    path: Path
    device: int
    inode: int
    mode: int
    owner_uid: int
    owner_gid: int
    size_bytes: int
    changed_ns: int
    modified_ns: int
    sha256: str


@dataclass(slots=True)
class M8TestAssetAccessGuard:
    """Allow only read-only Test manifest/NPZ access after a one-way phase transition."""

    official_track_root: Path
    test_manifest: Path
    test_asset: Path
    track_cache_root: Path
    _installed: bool = False
    _test_reads_enabled: bool = False
    _all_track_reads_forbidden: bool = False
    _allowed_event_counts: dict[str, int] = field(default_factory=dict)
    _allowed_event_sequence: list[dict[str, str | int | None]] = field(default_factory=list)
    _denied_event_count: int = 0
    _denied_mutation_event_count: int = 0
    _denied_mutation_event_types: dict[str, int] = field(default_factory=dict)
    _deterministic_recovery: bool = False
    _nvidia_smi_identity: _FrozenExecutableIdentity | None = None
    _audit_identity_token: object = field(default_factory=object, init=False, repr=False)
    _audit_identity_response: object | None = field(default=None, init=False, repr=False)
    _open_context: threading.local = field(default_factory=threading.local, init=False, repr=False)
    _process_context: threading.local = field(
        default_factory=threading.local, init=False, repr=False
    )

    _MUTATION_PATH_ARGUMENTS: ClassVar[Mapping[str, tuple[tuple[int, int | None], ...]]] = {
        "os.chmod": ((0, 2),),
        "os.chown": ((0, 3),),
        "os.link": ((0, 2), (1, 3)),
        "os.mkfifo": ((0, 2),),
        "os.mkdir": ((0, 2),),
        "os.mknod": ((0, 3),),
        "os.remove": ((0, 1),),
        "os.removexattr": ((0, None),),
        "os.rename": ((0, 2), (1, 3)),
        "os.replace": ((0, 2), (1, 3)),
        "os.rmdir": ((0, 1),),
        "os.setxattr": ((0, None),),
        "os.symlink": ((1, 2),),
        "os.truncate": ((0, None),),
        "os.unlink": ((0, 1),),
        "os.utime": ((0, 3),),
        "shutil.rmtree": ((0, 1),),
    }
    _AUDIT_SELF_CHECK_EVENT: ClassVar[str] = "controller_learning.m8_test_guard_self_check"
    _POST_LOAD_NVIDIA_SMI_ARGUMENTS: ClassVar[tuple[str, ...]] = (
        "--query-compute-apps=gpu_uuid,pid,used_gpu_memory",
        "--format=csv,noheader,nounits",
    )
    _POST_LOAD_NVIDIA_SMI_ENVIRONMENT: ClassVar[tuple[tuple[str, str], ...]] = (
        ("HOME", "/nonexistent"),
        ("LANG", "C"),
        ("LC_ALL", "C"),
    )
    _POST_LOAD_NVIDIA_SMI_TIMEOUT_S: ClassVar[int] = 15
    _POST_LOAD_PROCESS_EVENTS: ClassVar[frozenset[str]] = frozenset(
        {
            "os.exec",
            "os.fork",
            "os.forkpty",
            "os.posix_spawn",
            "os.spawn",
            "os.startfile",
            "os.startfile/2",
            "os.system",
            "pty.spawn",
        }
    )

    def __post_init__(self) -> None:
        for name in ("official_track_root", "test_manifest", "test_asset", "track_cache_root"):
            object.__setattr__(self, name, Path(getattr(self, name)).resolve(strict=False))
        for allowed in (self.test_manifest, self.test_asset):
            if not allowed.is_relative_to(self.official_track_root):
                raise ValueError("allowed Test assets must be inside official_track_root")
        if self.test_manifest == self.test_asset:
            raise ValueError("Test manifest and NPZ paths must differ")

    @staticmethod
    def _capture_executable_identity(
        executable: Path,
        *,
        require_trusted_ownership: bool = True,
    ) -> _FrozenExecutableIdentity:
        candidate = Path(executable)
        if not candidate.is_absolute():
            raise ValueError("nvidia-smi executable must be an absolute pre-resolved path")
        try:
            metadata = candidate.lstat()
            resolved = candidate.resolve(strict=True)
        except (FileNotFoundError, OSError) as error:
            raise ValueError("nvidia-smi executable is unavailable") from error
        if resolved != candidate or stat.S_ISLNK(metadata.st_mode):
            raise ValueError("nvidia-smi executable must be a pre-resolved non-symlink path")
        if not stat.S_ISREG(metadata.st_mode):
            raise ValueError("nvidia-smi executable must be a regular file")
        if metadata.st_mode & 0o111 == 0:
            raise ValueError("nvidia-smi executable must have an executable mode")
        if require_trusted_ownership:
            if metadata.st_uid != 0:
                raise ValueError("nvidia-smi executable must be owned by root")
            if metadata.st_mode & (stat.S_ISUID | stat.S_ISGID):
                raise ValueError("nvidia-smi executable cannot use setuid or setgid mode bits")
            if metadata.st_mode & 0o022:
                raise ValueError("nvidia-smi executable cannot be group- or world-writable")
            for parent in candidate.parents:
                try:
                    parent_metadata = parent.lstat()
                except OSError as error:
                    raise ValueError(
                        "nvidia-smi executable parent metadata is unavailable"
                    ) from error
                if (
                    stat.S_ISLNK(parent_metadata.st_mode)
                    or not stat.S_ISDIR(parent_metadata.st_mode)
                    or parent_metadata.st_uid != 0
                    or parent_metadata.st_mode & 0o022
                ):
                    raise ValueError(
                        "nvidia-smi executable parents must be root-owned, non-symlink, and not "
                        "group- or world-writable"
                    )
        digest = hashlib.sha256()
        try:
            with candidate.open("rb") as stream:
                while chunk := stream.read(1024 * 1024):
                    digest.update(chunk)
            after = candidate.lstat()
        except OSError as error:
            raise ValueError("nvidia-smi executable could not be read") from error
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
        if any(getattr(metadata, name) != getattr(after, name) for name in identity_fields):
            raise ValueError("nvidia-smi executable changed while its identity was captured")
        return _FrozenExecutableIdentity(
            path=candidate,
            device=metadata.st_dev,
            inode=metadata.st_ino,
            mode=metadata.st_mode,
            owner_uid=metadata.st_uid,
            owner_gid=metadata.st_gid,
            size_bytes=metadata.st_size,
            changed_ns=metadata.st_ctime_ns,
            modified_ns=metadata.st_mtime_ns,
            sha256=digest.hexdigest(),
        )

    def _verify_nvidia_smi_identity(self) -> _FrozenExecutableIdentity:
        frozen = self._nvidia_smi_identity
        if frozen is None:
            self._deny("post-load nvidia-smi executable identity was not frozen before Test access")
        try:
            observed = self._capture_executable_identity(
                frozen.path,
                require_trusted_ownership=False,
            )
        except ValueError:
            self._deny("the frozen post-load nvidia-smi executable identity changed")
        if observed != frozen:
            self._deny("the frozen post-load nvidia-smi executable identity changed")
        return frozen

    def assert_audit_hook_active(self) -> None:
        """Emit a private event and require this exact installed hook to acknowledge it."""

        if not self._installed:
            raise RuntimeError("M8 Test asset guard must be installed first")
        self._audit_identity_response = None
        sys.audit(self._AUDIT_SELF_CHECK_EVENT, self._audit_identity_token)
        if self._audit_identity_response is not self._audit_identity_token:
            raise RuntimeError("M8 Test asset guard audit hook did not answer its identity check")

    def freeze_nvidia_smi_executable(self, executable: str | os.PathLike[str]) -> str:
        """Freeze the private executable used by post-load process-VRAM sampling."""

        self.assert_audit_hook_active()
        if (
            self._test_reads_enabled
            or self._all_track_reads_forbidden
            or self._deterministic_recovery
        ):
            raise RuntimeError("nvidia-smi identity must be frozen before Test or recovery mode")
        if self._nvidia_smi_identity is not None:
            raise RuntimeError("nvidia-smi executable identity is already frozen")
        identity = self._capture_executable_identity(Path(executable))
        self._nvidia_smi_identity = identity
        return str(identity.path)

    def enter_deterministic_recovery(self) -> None:
        """Forbid every child process and Track read during durable artifact-only recovery."""

        self.assert_audit_hook_active()
        if self._test_reads_enabled or self._all_track_reads_forbidden:
            raise RuntimeError(
                "deterministic recovery must begin from the clean pre-Test guard state"
            )
        if self._deterministic_recovery:
            raise RuntimeError("deterministic recovery mode is already active")
        if self._allowed_event_counts or self._denied_event_count:
            raise RuntimeError("deterministic recovery requires a clean asset-access history")
        self._deterministic_recovery = True

    def run_frozen_memory_query(self, command: Sequence[str]) -> str:
        """Run the sole post-load process through one private thread-local capability."""

        self.assert_audit_hook_active()
        if isinstance(command, (str, bytes)) or not isinstance(command, Sequence):
            raise TypeError("memory query command must be a sequence of strings")
        frozen = self._verify_nvidia_smi_identity()
        expected = (str(frozen.path), *self._POST_LOAD_NVIDIA_SMI_ARGUMENTS)
        if tuple(command) != expected or any(type(value) is not str for value in command):
            raise ValueError("memory query must use the exact frozen nvidia-smi argv")
        if not self._all_track_reads_forbidden or self._deterministic_recovery:
            raise RuntimeError("frozen memory queries require the sealed post-load phase")
        if getattr(self._process_context, "capability", None) is not None:
            raise RuntimeError("frozen memory query capability is already active on this thread")

        capability = {"popen_count": 0}
        self._process_context.capability = capability
        try:
            completed = subprocess.run(
                expected,
                stdin=subprocess.DEVNULL,
                capture_output=True,
                cwd=None,
                env=dict(self._POST_LOAD_NVIDIA_SMI_ENVIRONMENT),
                check=True,
                close_fds=True,
                shell=False,
                start_new_session=True,
                text=True,
                timeout=self._POST_LOAD_NVIDIA_SMI_TIMEOUT_S,
            )
            if capability["popen_count"] != 1:
                raise RuntimeError("frozen memory query did not consume exactly one process token")
            if not isinstance(completed.stdout, str):
                raise RuntimeError("frozen memory query did not return text")
            return completed.stdout
        finally:
            self._process_context.capability = None

    @staticmethod
    def _audit_path(source: object, directory_descriptor: object = None) -> Path | None:
        if source is None:
            try:
                return Path.cwd().resolve(strict=True)
            except (FileNotFoundError, OSError):
                return None
        if type(source) is int:
            if source < 0:
                return None
            try:
                return Path(f"/proc/self/fd/{source}").resolve(strict=True)
            except (FileNotFoundError, OSError):
                return None
        if not isinstance(source, (str, bytes, os.PathLike)):
            return None
        path = Path(os.fsdecode(os.fspath(source)))
        if (
            not path.is_absolute()
            and type(directory_descriptor) is int
            and directory_descriptor >= 0
        ):
            try:
                path = Path(f"/proc/self/fd/{directory_descriptor}").resolve(strict=True) / path
            except (FileNotFoundError, OSError):
                return None
        return path.resolve(strict=False)

    def _is_protected(self, candidate: Path) -> bool:
        return candidate.is_relative_to(self.official_track_root) or candidate.is_relative_to(
            self.track_cache_root
        )

    def _category(self, candidate: Path) -> str | None:
        if candidate == self.test_manifest:
            return "official_test_manifest"
        if candidate == self.test_asset:
            return "official_test_asset"
        return None

    def _deny(self, message: str, *, mutation_event: str | None = None) -> None:
        self._denied_event_count += 1
        if mutation_event is not None:
            self._denied_mutation_event_count += 1
            self._denied_mutation_event_types[mutation_event] = (
                self._denied_mutation_event_types.get(mutation_event, 0) + 1
            )
        raise ForbiddenFinalEvaluationAssetAccessError(message)

    def _block_mutation(self, event: str, arguments: tuple[Any, ...]) -> None:
        specifications = self._MUTATION_PATH_ARGUMENTS.get(event)
        if specifications is None:
            return
        for path_index, descriptor_index in specifications:
            if path_index >= len(arguments):
                continue
            descriptor = (
                arguments[descriptor_index]
                if descriptor_index is not None and descriptor_index < len(arguments)
                else None
            )
            candidate = self._audit_path(arguments[path_index], descriptor)
            if candidate is not None and self._is_protected(candidate):
                self._deny(
                    f"M8 final evaluation forbids protected filesystem mutation {event}",
                    mutation_event=event,
                )

    def _block_enumeration(self, event: str, arguments: tuple[Any, ...]) -> None:
        if event not in {"os.listdir", "os.scandir"} or not arguments:
            return
        candidate = self._audit_path(arguments[0])
        if candidate is not None and self._is_protected(candidate):
            self._deny(f"M8 final evaluation forbids protected directory enumeration {event}")

    def _block_cwd_change(self, event: str, arguments: tuple[Any, ...]) -> None:
        if event not in {"os.chdir", "os.fchdir"} or not arguments:
            return
        candidate = self._audit_path(arguments[0])
        if candidate is not None and self._is_protected(candidate):
            self._deny(f"M8 final evaluation forbids protected working-directory change {event}")

    def _is_exact_nvidia_smi_environment(self, value: object) -> bool:
        return type(value) is dict and value == dict(self._POST_LOAD_NVIDIA_SMI_ENVIRONMENT)

    def _allow_only_frozen_memory_query(self, event: str, arguments: tuple[Any, ...]) -> None:
        capability = getattr(self._process_context, "capability", None)
        if not isinstance(capability, dict) or capability.get("popen_count") != 0:
            self._deny("post-load process creation requires the private memory-query capability")
        if event != "subprocess.Popen":
            self._deny("direct posix_spawn is forbidden after the in-memory Test pool load")
        frozen = self._verify_nvidia_smi_identity()
        expected_command = (str(frozen.path), *self._POST_LOAD_NVIDIA_SMI_ARGUMENTS)
        executable = arguments[0] if len(arguments) > 0 else None
        command = arguments[1] if len(arguments) > 1 else None
        cwd = arguments[2] if len(arguments) > 2 else None
        environment = arguments[3] if len(arguments) > 3 else None
        allowed = (
            executable == str(frozen.path)
            and isinstance(command, (tuple, list))
            and tuple(command) == expected_command
            and cwd is None
            and self._is_exact_nvidia_smi_environment(environment)
        )
        if not allowed:
            self._deny(
                "only the frozen absolute nvidia-smi process-VRAM query is allowed after the "
                "in-memory Test pool load"
            )
        capability["popen_count"] = 1

    def _audit(self, event: str, arguments: tuple[Any, ...]) -> None:
        if event == self._AUDIT_SELF_CHECK_EVENT:
            if len(arguments) == 1 and arguments[0] is self._audit_identity_token:
                self._audit_identity_response = self._audit_identity_token
            return
        self._block_mutation(event, arguments)
        self._block_enumeration(event, arguments)
        self._block_cwd_change(event, arguments)
        process_event = event == "subprocess.Popen" or event in self._POST_LOAD_PROCESS_EVENTS
        if process_event and (
            self._deterministic_recovery
            or (self._test_reads_enabled and not self._all_track_reads_forbidden)
        ):
            phase = (
                "deterministic recovery"
                if self._deterministic_recovery
                else "the in-memory Test asset-load interval"
            )
            self._deny(f"process creation event {event} is forbidden during {phase}")
        if process_event and self._all_track_reads_forbidden:
            self._allow_only_frozen_memory_query(event, arguments)
        if event != "open" or not arguments:
            return
        directory_descriptor = None
        stack = getattr(self._open_context, "stack", ())
        if stack:
            source, descriptor = stack[-1]
            try:
                same_source = os.fspath(source) == os.fspath(arguments[0])
            except TypeError:
                same_source = False
            if same_source:
                directory_descriptor = descriptor
        candidate = self._audit_path(arguments[0], directory_descriptor)
        if candidate is None or not self._is_protected(candidate):
            return
        if self._all_track_reads_forbidden:
            self._deny("all Track-asset reads are forbidden after the in-memory Test pool load")
        category = self._category(candidate)
        if category is None:
            self._deny(
                "M8 final evaluation forbids Level0, Train, Validation, and Track-cache access"
            )
        if not self._test_reads_enabled:
            self._deny("Test reads remain disabled until the durable TEST_BOUND transition")
        mode = arguments[1] if len(arguments) > 1 else None
        flags = arguments[2] if len(arguments) > 2 else None
        write_mode = isinstance(mode, str) and any(token in mode for token in "wax+")
        write_flags = type(flags) is int and bool(
            flags & (os.O_WRONLY | os.O_RDWR | os.O_CREAT | os.O_TRUNC | os.O_APPEND)
        )
        if write_mode or write_flags:
            self._deny("official Test assets are read-only")
        self._allowed_event_counts[category] = self._allowed_event_counts.get(category, 0) + 1
        self._allowed_event_sequence.append(
            {
                "category": category,
                "flags": flags if type(flags) is int else None,
                "mode": mode if isinstance(mode, str) else None,
            }
        )

    def install(self) -> None:
        """Install the irreversible process-local audit guard before project imports."""

        if self._installed:
            raise RuntimeError("M8 Test asset guard is already installed")
        sys.addaudithook(self._audit)
        original_open = os.open
        original_mkfifo = os.mkfifo
        original_mknod = os.mknod

        def guarded_open(
            path: str | bytes | os.PathLike[str] | os.PathLike[bytes],
            flags: int,
            mode: int = 0o777,
            *,
            dir_fd: int | None = None,
        ) -> int:
            stack = getattr(self._open_context, "stack", None)
            if stack is None:
                stack = []
                self._open_context.stack = stack
            stack.append((path, dir_fd))
            try:
                return original_open(path, flags, mode, dir_fd=dir_fd)
            finally:
                stack.pop()

        def guarded_mkfifo(
            path: str | bytes | os.PathLike[str] | os.PathLike[bytes],
            mode: int = 0o666,
            *,
            dir_fd: int | None = None,
        ) -> None:
            self._block_mutation("os.mkfifo", (path, mode, -1 if dir_fd is None else dir_fd))
            original_mkfifo(path, mode, dir_fd=dir_fd)

        def guarded_mknod(
            path: str | bytes | os.PathLike[str] | os.PathLike[bytes],
            mode: int = 0o600,
            device: int = 0,
            *,
            dir_fd: int | None = None,
        ) -> None:
            self._block_mutation(
                "os.mknod",
                (path, mode, device, -1 if dir_fd is None else dir_fd),
            )
            original_mknod(path, mode, device, dir_fd=dir_fd)

        # The open audit event omits dir_fd.  Preserve it in thread-local context while CPython
        # emits the synchronous event so relative descriptor paths receive the normal policy.
        os.open = guarded_open  # type: ignore[assignment]
        # CPython emits no audit event for these two functions.
        os.mkfifo = guarded_mkfifo  # type: ignore[assignment]
        os.mknod = guarded_mknod  # type: ignore[assignment]
        self._installed = True
        self.assert_audit_hook_active()

    def enable_test_reads(self) -> None:
        """Open the one-way Test phase only after the durable transaction is TEST_BOUND."""

        if not self._installed:
            raise RuntimeError("M8 Test asset guard must be installed first")
        self.assert_audit_hook_active()
        if self._test_reads_enabled:
            raise RuntimeError("Test reads are already enabled")
        if self._deterministic_recovery:
            raise RuntimeError("Test reads cannot be enabled during deterministic recovery")
        if self._nvidia_smi_identity is None:
            raise RuntimeError("nvidia-smi executable identity must be frozen before Test reads")
        if self._denied_event_count != 0 or self._denied_mutation_event_count != 0:
            raise RuntimeError("a denied asset event permanently closed the Test phase")
        if self._allowed_event_counts:
            raise RuntimeError("Test assets were opened before the phase transition")
        self._test_reads_enabled = True

    def forbid_all_track_reads(self) -> None:
        """Irreversibly close every Track-asset read path after the Test pool is in memory."""

        if not self._installed or not self._test_reads_enabled:
            raise RuntimeError("Test reads must be enabled before they can be closed")
        self.assert_audit_hook_active()
        if self._all_track_reads_forbidden:
            raise RuntimeError("all Track-asset reads are already forbidden")
        if self._denied_event_count != 0 or self._denied_mutation_event_count != 0:
            raise RuntimeError("a denied asset event permanently invalidated the Test process")
        if set(self._allowed_event_counts) != {
            "official_test_manifest",
            "official_test_asset",
        }:
            raise RuntimeError("both official Test files must be audited before reads are closed")
        self._all_track_reads_forbidden = True

    def evidence(self, *, test_loaded: bool) -> dict[str, Any]:
        """Return strict path-sanitized access evidence for the final report."""

        if type(test_loaded) is not bool:
            raise TypeError("test_loaded must be a boolean")
        self.assert_audit_hook_active()
        observed = set(self._allowed_event_counts)
        expected = {"official_test_manifest", "official_test_asset"}
        if test_loaded and observed != expected:
            raise RuntimeError("successful Test loading did not audit both allowed files")
        if not test_loaded and observed:
            raise RuntimeError("Test assets were opened before the preflight boundary")
        return {
            "all_track_reads_forbidden": self._all_track_reads_forbidden,
            "audit_hook_installed_before_preflight": self._installed,
            "denied_event_count": self._denied_event_count,
            "denied_mutation_event_count": self._denied_mutation_event_count,
            "denied_mutation_event_types": dict(sorted(self._denied_mutation_event_types.items())),
            "open_event_counts": dict(sorted(self._allowed_event_counts.items())),
            "open_event_sequence": list(self._allowed_event_sequence),
            "opened_path_categories": sorted(observed),
            "opened_splits": ["test"] if test_loaded else [],
            "pre_test_open_event_count": 0,
            "test_loaded": test_loaded,
            "test_reads_enabled": self._test_reads_enabled,
            "track_cache_opened": False,
            "train_opened": False,
            "validation_opened": False,
        }


__all__ = ["ForbiddenFinalEvaluationAssetAccessError", "M8TestAssetAccessGuard"]
