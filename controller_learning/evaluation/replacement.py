"""Read-only lineage and eligibility gates for the single M8 replacement attempt.

This module deliberately uses only the Python standard library.  It never imports the formal
benchmark, Track, environment, Controller, JAX, or NumPy modules, and it never opens an official
Track asset.  The local validator accepts only the retained zero-episode infrastructure failure
from attempt 001 and proves that its bytes, output state, frozen Controllers, and public hash
bindings still match the canonical public failure report.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import re
import stat
import tomllib
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from types import MappingProxyType
from typing import Any, Final

M8_ATTEMPT_001_FAILURE_REPORT_SCHEMA_VERSION: Final = (
    "controller-learning.m8-attempt-001-failure.v1"
)
M8_ATTEMPT_001_FAILURE_REPORT_RELATIVE_PATH: Final = (
    "benchmarks/v0.1/m8_attempt_001_failure_report.json"
)
M8_ATTEMPT_001_TRANSACTION_RELATIVE_PATH: Final = "runs/m8_final_attempt_transaction"
M8_ATTEMPT_001_CONTROLLER_SNAPSHOT_RELATIVE_PATH: Final = "runs/m8_final_controller_snapshot"
M8_ATTEMPT_001_RUN_ID: Final = "m8-final-v0-1-001"
M8_REPLACEMENT_RUN_ID: Final = "m8-final-v0-1-002"
M8_REPLACEMENT_MAX_ATTEMPTS: Final = 1

_TRANSACTION_SCHEMA_VERSION: Final = "controller-learning.m8-attempt-transaction.v2"
_FAILURE_SCHEMA_VERSION: Final = "controller-learning.m8-workload-failure.v1"
_EXPECTED_EPISODE_COUNT: Final = 60
_EXPECTED_CONTROLLER_ORDER: Final = ("pid", "mpc", "ppo")
_EXPECTED_ALLOWED_CHANGES: Final = (
    "pre_bind_warp_initialization",
    "replacement_lineage_evidence",
    "replacement_eligibility_gates",
    "replacement_documentation",
)
_EXPECTED_TRANSACTION_DIRECTORIES: Final = (
    (".", 0o700),
    ("backups", 0o700),
    ("blobs", 0o700),
    ("blobs/failures", 0o700),
)
_EXPECTED_TRANSACTION_FILES: Final = (
    "blob-index.jsonl",
    "blobs/failures/final-workload.json",
    "episode-journal.jsonl",
    "manifest.json",
    "state.json",
)
_FAILURE_BLOB_RELATIVE_PATH: Final = "failures/final-workload.json"
_EXECUTION_SEAL_RELATIVE_PATH: Final = "execution/final_evidence.json"
_EXPECTED_FAILURE: Final = MappingProxyType(
    {
        "cause_type": "ForbiddenFinalEvaluationAssetAccessError",
        "detail": "post-load process creation requires the private memory-query capability",
        "infrastructure_phase": "environment_create",
        "schema_version": _FAILURE_SCHEMA_VERSION,
        "workload": None,
    }
)
_CONTROLLER_FILES: Final = MappingProxyType(
    {
        "pid": ("README.md", "config.toml", "controller.py", "helpers.py"),
        "mpc": ("README.md", "config.toml", "controller.py", "helpers.py", "solver.py"),
        "ppo": ("README.md", "config.toml", "controller.py", "metadata.json", "policy.npz"),
    }
)
_EXPECTED_CONTROLLER_HASHES: Final = MappingProxyType(
    {
        "pid": MappingProxyType(
            {
                "aggregate_sha256": (
                    "4f9f63eb2b6c0862fcf3c584f73ae25e9a721fac4cf916e738657b3f6c9c0d71"
                ),
                "config_sha256": (
                    "10d661604ad1cab25bb2073d29aafb16003df3cae59026baef10a10e5e737e47"
                ),
            }
        ),
        "mpc": MappingProxyType(
            {
                "aggregate_sha256": (
                    "f0a288515b48ec360e72939e65184d58234b491e917c55bb5dd4e9466150c9bb"
                ),
                "config_sha256": (
                    "0aef6eacb4f9882adf0a97d728b210d9c99b09bb694f6792a2ad53d7802281fd"
                ),
            }
        ),
        "ppo": MappingProxyType(
            {
                "aggregate_sha256": (
                    "55720b360d6780704135da4670a1ac35cc13045bb11cb4866e256c494be14f2e"
                ),
                "config_sha256": (
                    "ee9f09deb5b55f21df90f234d251b79d6dfcdfaf80f7fbf2b7b488c489acf5dc"
                ),
            }
        ),
    }
)
_EXPECTED_TEST_HASHES: Final = MappingProxyType(
    {
        "asset_sha256": "0d654395630ec0b64952b076a2595de96f3926ea208fac3796a50be37df29c71",
        "manifest_sha256": ("2230e29f3e13029d4ca09de32a703e9a80c070e654386563b9ef4f7a2d197f8b"),
    }
)
_SHA256_PATTERN = re.compile(r"^[0-9a-f]{64}$")
_REVISION_PATTERN = re.compile(r"^[0-9a-f]{40}$")


class ReplacementEligibilityError(ValueError):
    """Attempt 001 or its public lineage is not eligible for replacement."""


@dataclass(frozen=True, slots=True)
class LocalPredecessorValidation:
    """Successful local proof for the single authorized replacement attempt."""

    eligible: bool
    report_sha256: str
    transaction_tree_sha256: str
    predecessor_source_revision: str
    successor_run_id: str

    def __post_init__(self) -> None:
        if self.eligible is not True:
            raise ValueError("a LocalPredecessorValidation can represent only an eligible attempt")
        _require_sha256(self.report_sha256, field="report_sha256")
        _require_sha256(self.transaction_tree_sha256, field="transaction_tree_sha256")
        if _REVISION_PATTERN.fullmatch(self.predecessor_source_revision) is None:
            raise ValueError("predecessor_source_revision must be a full lowercase Git revision")
        if self.successor_run_id != M8_REPLACEMENT_RUN_ID:
            raise ValueError("successor_run_id differs from the authorized replacement")


@dataclass(frozen=True, slots=True)
class _StableFile:
    content: bytes
    mode: int
    sha256: str
    size_bytes: int

    def report_record(self, relative_path: str) -> dict[str, object]:
        return {
            "mode": self.mode,
            "path": relative_path,
            "sha256": self.sha256,
            "size_bytes": self.size_bytes,
        }


def _require_sha256(value: object, *, field: str) -> str:
    if not isinstance(value, str) or _SHA256_PATTERN.fullmatch(value) is None:
        raise ReplacementEligibilityError(f"{field} must be a lowercase SHA-256 digest")
    return value


def _exact_keys(value: Mapping[str, Any], expected: set[str], *, field: str) -> None:
    if any(type(key) is not str for key in value) or set(value) != expected:
        raise ReplacementEligibilityError(f"{field} keys differ")


def _safe_relative_path(value: object, *, field: str) -> str:
    if not isinstance(value, str) or not value or "\\" in value:
        raise ReplacementEligibilityError(f"{field} must be a normalized relative POSIX path")
    path = PurePosixPath(value)
    if (
        path.is_absolute()
        or path.as_posix() != value
        or any(part in {"", ".", ".."} for part in path.parts)
    ):
        raise ReplacementEligibilityError(f"{field} must be a normalized relative POSIX path")
    return value


def _json_snapshot(value: Any, *, field: str = "$", active: set[int] | None = None) -> Any:
    if value is None or type(value) in {bool, int, str}:
        if isinstance(value, str):
            try:
                value.encode("utf-8")
            except UnicodeEncodeError as error:
                raise ReplacementEligibilityError(f"{field} is not valid UTF-8") from error
        return value
    if type(value) is float:
        if not math.isfinite(value):
            raise ReplacementEligibilityError(f"{field} contains NaN or infinity")
        return value
    active = set() if active is None else active
    identity = id(value)
    if identity in active:
        raise ReplacementEligibilityError(f"{field} contains a reference cycle")
    if isinstance(value, Mapping):
        active.add(identity)
        try:
            result: dict[str, Any] = {}
            for key, item in value.items():
                if type(key) is not str:
                    raise ReplacementEligibilityError(f"{field} contains a non-string key")
                result[key] = _json_snapshot(item, field=f"{field}.{key}", active=active)
            return result
        finally:
            active.remove(identity)
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        active.add(identity)
        try:
            return [
                _json_snapshot(item, field=f"{field}[{index}]", active=active)
                for index, item in enumerate(value)
            ]
        finally:
            active.remove(identity)
    raise ReplacementEligibilityError(f"{field} contains unsupported JSON data")


def canonical_failure_report_bytes(report: Mapping[str, Any]) -> bytes:
    """Return strict deterministic UTF-8 JSON bytes for one validated report mapping."""

    if not isinstance(report, Mapping):
        raise TypeError("failure report must be a mapping")
    snapshot = _json_snapshot(report)
    if not isinstance(snapshot, dict):  # pragma: no cover - Mapping is checked above
        raise TypeError("failure report must be a mapping")
    return (
        json.dumps(
            snapshot,
            allow_nan=False,
            ensure_ascii=True,
            separators=(",", ":"),
            sort_keys=True,
        )
        + "\n"
    ).encode("ascii")


def _strict_json_object(payload: bytes, *, field: str) -> Mapping[str, Any]:
    def reject_constant(value: str) -> None:
        raise ReplacementEligibilityError(f"{field} contains forbidden JSON constant {value}")

    def reject_duplicates(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in pairs:
            if key in result:
                raise ReplacementEligibilityError(f"{field} contains duplicate key {key!r}")
            result[key] = value
        return result

    try:
        value = json.loads(
            payload,
            parse_constant=reject_constant,
            object_pairs_hook=reject_duplicates,
        )
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ReplacementEligibilityError(f"{field} is not strict JSON") from error
    if not isinstance(value, Mapping):
        raise ReplacementEligibilityError(f"{field} must contain a JSON object")
    return value


def _stable_regular_file(path: Path, *, field: str) -> _StableFile:
    try:
        before = path.lstat()
    except FileNotFoundError as error:
        raise ReplacementEligibilityError(f"{field} is missing") from error
    if stat.S_ISLNK(before.st_mode) or not stat.S_ISREG(before.st_mode):
        raise ReplacementEligibilityError(f"{field} must be a non-symlink regular file")
    flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0)
    try:
        descriptor = os.open(path, flags)
    except OSError as error:
        raise ReplacementEligibilityError(
            f"{field} could not be opened without following"
        ) from error
    try:
        opened = os.fstat(descriptor)
        if not stat.S_ISREG(opened.st_mode):
            raise ReplacementEligibilityError(f"{field} descriptor is not regular")
        chunks: list[bytes] = []
        while True:
            chunk = os.read(descriptor, 1024 * 1024)
            if not chunk:
                break
            chunks.append(chunk)
    finally:
        os.close(descriptor)
    try:
        after = path.lstat()
    except FileNotFoundError as error:
        raise ReplacementEligibilityError(f"{field} disappeared while being read") from error
    stable_fields = (
        "st_dev",
        "st_ino",
        "st_mode",
        "st_nlink",
        "st_uid",
        "st_gid",
        "st_size",
        "st_mtime_ns",
        "st_ctime_ns",
    )
    if any(
        getattr(before, name) != getattr(opened, name)
        or getattr(opened, name) != getattr(after, name)
        for name in stable_fields
    ):
        raise ReplacementEligibilityError(f"{field} changed while it was read")
    content = b"".join(chunks)
    if len(content) != opened.st_size:
        raise ReplacementEligibilityError(f"{field} size changed while it was read")
    return _StableFile(
        content=content,
        mode=stat.S_IMODE(opened.st_mode),
        sha256=hashlib.sha256(content).hexdigest(),
        size_bytes=len(content),
    )


def _require_real_directory(path: Path, *, field: str, mode: int | None = None) -> os.stat_result:
    try:
        metadata = path.lstat()
    except FileNotFoundError as error:
        raise ReplacementEligibilityError(f"{field} is missing") from error
    if stat.S_ISLNK(metadata.st_mode) or not stat.S_ISDIR(metadata.st_mode):
        raise ReplacementEligibilityError(f"{field} must be a non-symlink directory")
    if mode is not None and stat.S_IMODE(metadata.st_mode) != mode:
        raise ReplacementEligibilityError(f"{field} mode differs")
    return metadata


def _real_project_root(project_root: str | Path) -> Path:
    supplied = Path(project_root)
    _require_real_directory(supplied, field="project root")
    return supplied.absolute()


def _directory_entries(path: Path, *, field: str) -> Mapping[str, os.stat_result]:
    _require_real_directory(path, field=field)
    try:
        with os.scandir(path) as iterator:
            entries = list(iterator)
    except OSError as error:
        raise ReplacementEligibilityError(f"{field} could not be enumerated") from error
    result: dict[str, os.stat_result] = {}
    for entry in entries:
        if entry.name in result or entry.name in {"", ".", ".."}:
            raise ReplacementEligibilityError(f"{field} contains an invalid entry")
        try:
            metadata = entry.stat(follow_symlinks=False)
        except OSError as error:
            raise ReplacementEligibilityError(
                f"{field} entry changed during enumeration"
            ) from error
        if stat.S_ISLNK(metadata.st_mode):
            raise ReplacementEligibilityError(f"{field} contains a symlink")
        result[entry.name] = metadata
    return MappingProxyType(result)


def _capture_transaction_tree(
    root: Path,
) -> tuple[tuple[dict[str, object], ...], tuple[dict[str, object], ...]]:
    transaction = root / M8_ATTEMPT_001_TRANSACTION_RELATIVE_PATH
    _require_real_directory(transaction, field="attempt 001 transaction", mode=0o700)
    observed_directories: list[dict[str, object]] = []
    observed_files: list[dict[str, object]] = []

    def visit(directory: Path, relative: PurePosixPath) -> None:
        metadata = _require_real_directory(directory, field=f"transaction directory {relative}")
        relative_text = "." if not relative.parts else relative.as_posix()
        observed_directories.append({"mode": stat.S_IMODE(metadata.st_mode), "path": relative_text})
        entries = _directory_entries(directory, field=f"transaction directory {relative_text}")
        for name in sorted(entries):
            child = directory / name
            child_relative = relative / name
            metadata = entries[name]
            if stat.S_ISDIR(metadata.st_mode):
                visit(child, child_relative)
            elif stat.S_ISREG(metadata.st_mode):
                stable = _stable_regular_file(
                    child, field=f"transaction file {child_relative.as_posix()}"
                )
                observed_files.append(stable.report_record(child_relative.as_posix()))
            else:
                raise ReplacementEligibilityError("attempt 001 transaction contains a special file")

    visit(transaction, PurePosixPath())
    observed_directories.sort(key=lambda item: str(item["path"]))
    observed_files.sort(key=lambda item: str(item["path"]))
    return tuple(observed_directories), tuple(observed_files)


def _transaction_tree_sha256(
    directories: Sequence[Mapping[str, object]], files: Sequence[Mapping[str, object]]
) -> str:
    payload = canonical_failure_report_bytes(
        {"directories": list(directories), "files": list(files)}
    )
    return hashlib.sha256(payload).hexdigest()


def _parse_canonical_json_file(file: _StableFile, *, field: str) -> Mapping[str, Any]:
    value = _strict_json_object(file.content, field=field)
    if canonical_failure_report_bytes(value) != file.content:
        raise ReplacementEligibilityError(f"{field} is not canonical JSON")
    return value


def _path_is_absent(root: Path, relative_path: str, *, field: str) -> bool:
    relative = _safe_relative_path(relative_path, field=field)
    current = root
    for index, part in enumerate(PurePosixPath(relative).parts):
        candidate = current / part
        try:
            metadata = candidate.lstat()
        except FileNotFoundError:
            return True
        if stat.S_ISLNK(metadata.st_mode):
            raise ReplacementEligibilityError(f"{field} traverses a symlink")
        if index < len(PurePosixPath(relative).parts) - 1:
            if not stat.S_ISDIR(metadata.st_mode):
                raise ReplacementEligibilityError(f"{field} parent is not a directory")
            current = candidate
            continue
        return False
    return False  # pragma: no cover - non-empty path is enforced


def _controller_identity(
    directory: Path,
    controller: str,
    *,
    read_only_snapshot: bool,
) -> dict[str, str]:
    expected_files = _CONTROLLER_FILES[controller]
    expected_directory_mode = 0o555 if read_only_snapshot else None
    expected_file_mode = 0o444 if read_only_snapshot else None
    _require_real_directory(
        directory,
        field=f"{controller} Controller directory",
        mode=expected_directory_mode,
    )
    entries = dict(_directory_entries(directory, field=f"{controller} Controller directory"))
    if not read_only_snapshot and "__pycache__" in entries:
        if not stat.S_ISDIR(entries["__pycache__"].st_mode):
            raise ReplacementEligibilityError(
                f"{controller} Controller __pycache__ is not a directory"
            )
        del entries["__pycache__"]
    if tuple(sorted(entries)) != tuple(sorted(expected_files)):
        raise ReplacementEligibilityError(f"{controller} Controller file manifest differs")
    records: list[dict[str, object]] = []
    config_sha256: str | None = None
    for name in expected_files:
        metadata = entries[name]
        if not stat.S_ISREG(metadata.st_mode):
            raise ReplacementEligibilityError(f"{controller}/{name} is not a regular file")
        file = _stable_regular_file(directory / name, field=f"{controller}/{name}")
        if expected_file_mode is not None and file.mode != expected_file_mode:
            raise ReplacementEligibilityError(f"{controller}/{name} snapshot mode differs")
        records.append({"path": name, "sha256": file.sha256, "size_bytes": file.size_bytes})
        if name == "config.toml":
            config_sha256 = file.sha256
    canonical = json.dumps(
        records,
        allow_nan=False,
        ensure_ascii=True,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("ascii")
    if config_sha256 is None:  # pragma: no cover - fixed manifests all contain config.toml
        raise ReplacementEligibilityError("Controller config identity is missing")
    return {
        "aggregate_sha256": hashlib.sha256(canonical).hexdigest(),
        "config_sha256": config_sha256,
    }


def _capture_controllers(root: Path) -> Mapping[str, Mapping[str, str]]:
    snapshot_root = root / M8_ATTEMPT_001_CONTROLLER_SNAPSHOT_RELATIVE_PATH
    _require_real_directory(snapshot_root, field="active Controller snapshot", mode=0o555)
    snapshot_root_entries = _directory_entries(snapshot_root, field="active Controller snapshot")
    if set(snapshot_root_entries) != {"controllers"} or not stat.S_ISDIR(
        snapshot_root_entries["controllers"].st_mode
    ):
        raise ReplacementEligibilityError("active Controller snapshot root differs")
    snapshot_controllers = snapshot_root / "controllers"
    _require_real_directory(
        snapshot_controllers, field="snapshot Controllers directory", mode=0o555
    )
    snapshot_entries = _directory_entries(
        snapshot_controllers, field="snapshot Controllers directory"
    )
    if set(snapshot_entries) != set(_EXPECTED_CONTROLLER_ORDER) or any(
        not stat.S_ISDIR(metadata.st_mode) for metadata in snapshot_entries.values()
    ):
        raise ReplacementEligibilityError("active snapshot must contain exactly pid, mpc, and ppo")

    result: dict[str, Mapping[str, str]] = {}
    for controller in _EXPECTED_CONTROLLER_ORDER:
        snapshot_identity = _controller_identity(
            snapshot_controllers / controller,
            controller,
            read_only_snapshot=True,
        )
        live_identity = _controller_identity(
            root / "controllers" / controller,
            controller,
            read_only_snapshot=False,
        )
        expected = dict(_EXPECTED_CONTROLLER_HASHES[controller])
        if snapshot_identity != expected or live_identity != expected:
            raise ReplacementEligibilityError(
                f"{controller} live/snapshot identity differs from attempt 001"
            )
        result[controller] = MappingProxyType(snapshot_identity)
    return MappingProxyType(result)


def _validate_current_hash_bindings(root: Path) -> None:
    config_file = _stable_regular_file(
        root / "configs" / "final_evaluation.toml",
        field="replacement final_evaluation.toml",
    )
    try:
        config = tomllib.loads(config_file.content.decode("utf-8"))
    except (UnicodeDecodeError, tomllib.TOMLDecodeError) as error:
        raise ReplacementEligibilityError("replacement final_evaluation.toml is invalid") from error
    test_assets = config.get("test_assets")
    controllers = config.get("controllers")
    if not isinstance(test_assets, Mapping) or not isinstance(controllers, Mapping):
        raise ReplacementEligibilityError("replacement config lacks frozen hash bindings")
    if dict(test_assets) != dict(_EXPECTED_TEST_HASHES):
        raise ReplacementEligibilityError("official Test hash bindings changed")
    for controller in _EXPECTED_CONTROLLER_ORDER:
        table = controllers.get(controller)
        if not isinstance(table, Mapping):
            raise ReplacementEligibilityError(f"replacement config lacks {controller} hashes")
        if (
            table.get("aggregate_sha256")
            != _EXPECTED_CONTROLLER_HASHES[controller]["aggregate_sha256"]
            or table.get("config_sha256")
            != _EXPECTED_CONTROLLER_HASHES[controller]["config_sha256"]
        ):
            raise ReplacementEligibilityError(f"replacement config changed {controller} hashes")


def _expected_output_paths() -> tuple[str, ...]:
    paths = {
        "benchmarks/v0.1/m8_final_evaluation_report.json",
        "benchmarks/v0.1/m8_final_results.csv",
        "benchmarks/v0.1/m8_test_row_000_comparison.png",
    }
    for controller in _EXPECTED_CONTROLLER_ORDER:
        base = f"results/0.1/{controller}/{M8_ATTEMPT_001_RUN_ID}"
        paths.update(
            {
                f"{base}/metrics.npz",
                f"{base}/results.csv",
                f"{base}/run_manifest.json",
                f"{base}/selected_replays/test_row_000_trajectory.json",
                f"{base}/summary.json",
                f"{base}/telemetry.png",
                f"{base}/trajectory.png",
            }
        )
    return tuple(sorted(paths))


def _inspect_attempt(root: Path) -> dict[str, Any]:
    directories_before, files_before = _capture_transaction_tree(root)
    expected_directories = tuple(
        {"mode": mode, "path": path} for path, mode in _EXPECTED_TRANSACTION_DIRECTORIES
    )
    if directories_before != expected_directories:
        raise ReplacementEligibilityError("attempt 001 transaction directories differ")
    if tuple(item["path"] for item in files_before) != _EXPECTED_TRANSACTION_FILES:
        raise ReplacementEligibilityError("attempt 001 transaction files differ")
    if any(item["mode"] != 0o600 for item in files_before):
        raise ReplacementEligibilityError("attempt 001 transaction file mode differs")

    transaction = root / M8_ATTEMPT_001_TRANSACTION_RELATIVE_PATH
    files = {
        str(item["path"]): _stable_regular_file(
            transaction / str(item["path"]), field=f"transaction {item['path']}"
        )
        for item in files_before
    }
    manifest = _parse_canonical_json_file(files["manifest.json"], field="attempt manifest")
    state = _parse_canonical_json_file(files["state.json"], field="attempt state")
    failure = _parse_canonical_json_file(
        files["blobs/failures/final-workload.json"], field="attempt failure blob"
    )
    if dict(failure) != dict(_EXPECTED_FAILURE):
        raise ReplacementEligibilityError("attempt 001 failure is not the authorized failure")

    if files["episode-journal.jsonl"].content != b"":
        raise ReplacementEligibilityError("attempt 001 journal is not empty")
    index_content = files["blob-index.jsonl"].content
    if not index_content.endswith(b"\n") or index_content.count(b"\n") != 1:
        raise ReplacementEligibilityError("attempt 001 blob index must contain exactly one record")
    blob_record = _strict_json_object(index_content, field="attempt blob index")
    if canonical_failure_report_bytes(blob_record) != index_content:
        raise ReplacementEligibilityError("attempt 001 blob index is not canonical")
    expected_blob_record = {
        "mode": 0o600,
        "relative_path": _FAILURE_BLOB_RELATIVE_PATH,
        "sha256": files["blobs/failures/final-workload.json"].sha256,
        "size_bytes": files["blobs/failures/final-workload.json"].size_bytes,
    }
    if dict(blob_record) != expected_blob_record:
        raise ReplacementEligibilityError("attempt 001 contains an extra or changed blob")
    if blob_record["relative_path"] == _EXECUTION_SEAL_RELATIVE_PATH:
        raise ReplacementEligibilityError("attempt 001 unexpectedly contains an execution seal")

    _exact_keys(
        manifest,
        {
            "created_output_directories",
            "episode_protocol",
            "identity",
            "output_allowlist",
            "outputs",
            "recovery_policy",
            "schema_version",
            "transaction_relative_path",
        },
        field="attempt manifest",
    )
    _exact_keys(
        state,
        {
            "evidence",
            "identity",
            "manifest_sha256",
            "phase",
            "phase_index",
            "schema_version",
        },
        field="attempt state",
    )
    if (
        manifest.get("schema_version") != _TRANSACTION_SCHEMA_VERSION
        or manifest.get("transaction_relative_path") != M8_ATTEMPT_001_TRANSACTION_RELATIVE_PATH
        or state.get("schema_version") != _TRANSACTION_SCHEMA_VERSION
        or state.get("phase") != "TEST_BOUND"
        or state.get("phase_index") != 1
        or state.get("evidence") is not None
        or state.get("identity") != manifest.get("identity")
        or state.get("manifest_sha256") != files["manifest.json"].sha256
    ):
        raise ReplacementEligibilityError("attempt 001 state/manifest binding differs")
    protocol = manifest.get("episode_protocol")
    if not isinstance(protocol, Mapping) or dict(protocol) != {
        "controller_order": list(_EXPECTED_CONTROLLER_ORDER),
        "expected_record_count": _EXPECTED_EPISODE_COUNT,
        "ordering": "controller_major_then_row_index",
        "rows_per_controller": 20,
    }:
        raise ReplacementEligibilityError("attempt 001 episode protocol differs")
    recovery = manifest.get("recovery_policy")
    if not isinstance(recovery, Mapping) or dict(recovery) != {
        "accepted_result": "first_complete_protocol_passing_attempt",
        "automatic_retry_after_test_bound": False,
        "completed_attempt_finalizes_from_durable_bytes_only": True,
        "low_performance_can_trigger_retry": False,
        "partial_publication_restores_originals_before_republish": True,
    }:
        raise ReplacementEligibilityError("attempt 001 original recovery policy differs")
    identity = manifest.get("identity")
    if not isinstance(identity, Mapping):
        raise ReplacementEligibilityError("attempt 001 identity is missing")
    _exact_keys(
        identity,
        {
            "config_sha256",
            "input_sha256",
            "pixi_lock_sha256",
            "source_revision",
            "source_tree_sha256",
        },
        field="attempt identity",
    )
    if _REVISION_PATTERN.fullmatch(str(identity.get("source_revision"))) is None:
        raise ReplacementEligibilityError("attempt 001 source revision is invalid")
    for key in ("config_sha256", "input_sha256", "pixi_lock_sha256", "source_tree_sha256"):
        _require_sha256(identity.get(key), field=f"attempt identity {key}")

    expected_output_paths = _expected_output_paths()
    output_allowlist = manifest.get("output_allowlist")
    outputs = manifest.get("outputs")
    if output_allowlist != list(expected_output_paths) or not isinstance(outputs, list):
        raise ReplacementEligibilityError("attempt 001 output allowlist differs")
    if len(outputs) != 24:
        raise ReplacementEligibilityError("attempt 001 must bind exactly 24 outputs")
    output_report: list[dict[str, object]] = []
    for path, output in zip(expected_output_paths, outputs, strict=True):
        if not isinstance(output, Mapping) or dict(output) != {
            "backup_relative_path": None,
            "existed": False,
            "mode": None,
            "relative_path": path,
            "sha256": None,
            "size_bytes": 0,
        }:
            raise ReplacementEligibilityError(f"attempt 001 output snapshot differs for {path}")
        if not _path_is_absent(root, path, field=f"attempt output {path}"):
            raise ReplacementEligibilityError(f"attempt 001 output exists: {path}")
        output_report.append(
            {
                "local_state": "absent",
                "manifest_original_state": "absent",
                "path": path,
            }
        )

    controllers_before = _capture_controllers(root)
    _validate_current_hash_bindings(root)
    controllers_after = _capture_controllers(root)
    directories_after, files_after = _capture_transaction_tree(root)
    if directories_after != directories_before or files_after != files_before:
        raise ReplacementEligibilityError("attempt 001 transaction changed during validation")
    if controllers_after != controllers_before:
        raise ReplacementEligibilityError("active Controller snapshot changed during validation")

    file_report = [files[path].report_record(path) for path in _EXPECTED_TRANSACTION_FILES]
    tree_sha256 = _transaction_tree_sha256(directories_before, file_report)
    return {
        "authorization": {
            "allowed_changes": list(_EXPECTED_ALLOWED_CHANGES),
            "authorization_source": "explicit_repository_owner_approval",
            "authorized": True,
            "authorized_on": "2026-07-11",
            "max_replacement_attempts": M8_REPLACEMENT_MAX_ATTEMPTS,
            "performance_outcome_can_trigger_replacement": False,
            "scope": "single_zero_episode_infrastructure_replacement",
            "successor_run_id": M8_REPLACEMENT_RUN_ID,
            "third_attempt_allowed": False,
        },
        "controllers": {
            "active_snapshot_relative_path": (M8_ATTEMPT_001_CONTROLLER_SNAPSHOT_RELATIVE_PATH),
            "identities": {
                name: dict(controllers_before[name]) for name in _EXPECTED_CONTROLLER_ORDER
            },
        },
        "failure": {
            **dict(failure),
            "blob_relative_path": _FAILURE_BLOB_RELATIVE_PATH,
        },
        "official_test_assets": dict(_EXPECTED_TEST_HASHES),
        "predecessor": {
            "execution_evidence": None,
            "expected_episode_count": _EXPECTED_EPISODE_COUNT,
            "identity": dict(identity),
            "journal_record_count": 0,
            "performance_observed": False,
            "run_id": M8_ATTEMPT_001_RUN_ID,
            "transaction_phase": "TEST_BOUND",
            "transaction_relative_path": M8_ATTEMPT_001_TRANSACTION_RELATIVE_PATH,
        },
        "privacy": {
            "absolute_paths_recorded": False,
            "controller_performance_observed": False,
            "host_identifiers_recorded": False,
            "official_test_geometry_opened_by_lineage_builder": False,
        },
        "schema_version": M8_ATTEMPT_001_FAILURE_REPORT_SCHEMA_VERSION,
        "status": "eligible_for_one_replacement_attempt",
        "transaction": {
            "artifact_validation_present": False,
            "blob_count": 1,
            "directories": list(directories_before),
            "episode_blob_count": 0,
            "execution_seal_present": False,
            "files": file_report,
            "final_staged_present": False,
            "output_count": len(output_report),
            "outputs": output_report,
            "publication_present": False,
            "tree_sha256": tree_sha256,
        },
    }


def build_failure_report(project_root: str | Path) -> Mapping[str, Any]:
    """Build the canonical public attempt-001 report from read-only local evidence."""

    root = _real_project_root(project_root)
    return MappingProxyType(_inspect_attempt(root))


def _validate_report_mapping(report: Mapping[str, Any]) -> None:
    _exact_keys(
        report,
        {
            "authorization",
            "controllers",
            "failure",
            "official_test_assets",
            "predecessor",
            "privacy",
            "schema_version",
            "status",
            "transaction",
        },
        field="failure report",
    )
    if (
        report.get("schema_version") != M8_ATTEMPT_001_FAILURE_REPORT_SCHEMA_VERSION
        or report.get("status") != "eligible_for_one_replacement_attempt"
    ):
        raise ReplacementEligibilityError("failure report schema or status differs")
    authorization = report.get("authorization")
    if not isinstance(authorization, Mapping) or dict(authorization) != {
        "allowed_changes": list(_EXPECTED_ALLOWED_CHANGES),
        "authorization_source": "explicit_repository_owner_approval",
        "authorized": True,
        "authorized_on": "2026-07-11",
        "max_replacement_attempts": 1,
        "performance_outcome_can_trigger_replacement": False,
        "scope": "single_zero_episode_infrastructure_replacement",
        "successor_run_id": M8_REPLACEMENT_RUN_ID,
        "third_attempt_allowed": False,
    }:
        raise ReplacementEligibilityError("failure report authorization differs")
    failure = report.get("failure")
    expected_failure = {
        **dict(_EXPECTED_FAILURE),
        "blob_relative_path": _FAILURE_BLOB_RELATIVE_PATH,
    }
    if not isinstance(failure, Mapping) or dict(failure) != expected_failure:
        raise ReplacementEligibilityError("failure report failure evidence differs")
    if report.get("official_test_assets") != dict(_EXPECTED_TEST_HASHES):
        raise ReplacementEligibilityError("failure report official hash bindings differ")
    controllers = report.get("controllers")
    if (
        not isinstance(controllers, Mapping)
        or controllers.get("active_snapshot_relative_path")
        != M8_ATTEMPT_001_CONTROLLER_SNAPSHOT_RELATIVE_PATH
        or controllers.get("identities")
        != {name: dict(_EXPECTED_CONTROLLER_HASHES[name]) for name in _EXPECTED_CONTROLLER_ORDER}
    ):
        raise ReplacementEligibilityError("failure report Controller identities differ")
    predecessor = report.get("predecessor")
    if not isinstance(predecessor, Mapping):
        raise ReplacementEligibilityError("failure report predecessor is missing")
    _exact_keys(
        predecessor,
        {
            "execution_evidence",
            "expected_episode_count",
            "identity",
            "journal_record_count",
            "performance_observed",
            "run_id",
            "transaction_phase",
            "transaction_relative_path",
        },
        field="failure report predecessor",
    )
    if (
        predecessor.get("execution_evidence") is not None
        or predecessor.get("expected_episode_count") != 60
        or predecessor.get("journal_record_count") != 0
        or predecessor.get("performance_observed") is not False
        or predecessor.get("run_id") != M8_ATTEMPT_001_RUN_ID
        or predecessor.get("transaction_phase") != "TEST_BOUND"
        or predecessor.get("transaction_relative_path") != M8_ATTEMPT_001_TRANSACTION_RELATIVE_PATH
    ):
        raise ReplacementEligibilityError("failure report predecessor eligibility differs")
    identity = predecessor.get("identity")
    if not isinstance(identity, Mapping):
        raise ReplacementEligibilityError("failure report predecessor identity is missing")
    _exact_keys(
        identity,
        {
            "config_sha256",
            "input_sha256",
            "pixi_lock_sha256",
            "source_revision",
            "source_tree_sha256",
        },
        field="failure report predecessor identity",
    )
    if _REVISION_PATTERN.fullmatch(str(identity.get("source_revision"))) is None:
        raise ReplacementEligibilityError("failure report source revision differs")
    for key in ("config_sha256", "input_sha256", "pixi_lock_sha256", "source_tree_sha256"):
        _require_sha256(identity.get(key), field=f"failure report identity {key}")
    if report.get("privacy") != {
        "absolute_paths_recorded": False,
        "controller_performance_observed": False,
        "host_identifiers_recorded": False,
        "official_test_geometry_opened_by_lineage_builder": False,
    }:
        raise ReplacementEligibilityError("failure report privacy attestation differs")

    transaction = report.get("transaction")
    if not isinstance(transaction, Mapping):
        raise ReplacementEligibilityError("failure report transaction is missing")
    _exact_keys(
        transaction,
        {
            "artifact_validation_present",
            "blob_count",
            "directories",
            "episode_blob_count",
            "execution_seal_present",
            "files",
            "final_staged_present",
            "output_count",
            "outputs",
            "publication_present",
            "tree_sha256",
        },
        field="failure report transaction",
    )
    if (
        any(
            transaction.get(key) is not False
            for key in (
                "artifact_validation_present",
                "execution_seal_present",
                "final_staged_present",
                "publication_present",
            )
        )
        or transaction.get("blob_count") != 1
        or transaction.get("episode_blob_count") != 0
    ):
        raise ReplacementEligibilityError("failure report transaction eligibility differs")
    directories = transaction.get("directories")
    files = transaction.get("files")
    outputs = transaction.get("outputs")
    if directories != [
        {"mode": mode, "path": path} for path, mode in _EXPECTED_TRANSACTION_DIRECTORIES
    ]:
        raise ReplacementEligibilityError("failure report transaction directories differ")
    if not isinstance(files, list) or len(files) != len(_EXPECTED_TRANSACTION_FILES):
        raise ReplacementEligibilityError("failure report transaction files differ")
    for expected_path, record in zip(_EXPECTED_TRANSACTION_FILES, files, strict=True):
        if not isinstance(record, Mapping):
            raise ReplacementEligibilityError("failure report file record is malformed")
        _exact_keys(record, {"mode", "path", "sha256", "size_bytes"}, field="file record")
        if (
            record.get("path") != expected_path
            or record.get("mode") != 0o600
            or type(record.get("size_bytes")) is not int
            or record["size_bytes"] < 0
        ):
            raise ReplacementEligibilityError("failure report file identity differs")
        _require_sha256(record.get("sha256"), field=f"failure report file {expected_path}")
    tree_sha256 = _transaction_tree_sha256(directories, files)
    if transaction.get("tree_sha256") != tree_sha256:
        raise ReplacementEligibilityError("failure report transaction tree digest differs")
    expected_paths = _expected_output_paths()
    if transaction.get("output_count") != 24 or not isinstance(outputs, list) or len(outputs) != 24:
        raise ReplacementEligibilityError("failure report output count differs")
    for path, output in zip(expected_paths, outputs, strict=True):
        if not isinstance(output, Mapping) or dict(output) != {
            "local_state": "absent",
            "manifest_original_state": "absent",
            "path": path,
        }:
            raise ReplacementEligibilityError("failure report output identity differs")


def validate_failure_report_bytes(
    payload: bytes,
    *,
    expected_sha256: str | None = None,
) -> Mapping[str, Any]:
    """Validate canonical report bytes without consulting local attempt state."""

    if not isinstance(payload, bytes):
        raise TypeError("failure report payload must be bytes")
    digest = hashlib.sha256(payload).hexdigest()
    if expected_sha256 is not None and digest != _require_sha256(
        expected_sha256, field="expected_sha256"
    ):
        raise ReplacementEligibilityError("failure report SHA-256 differs")
    report = _strict_json_object(payload, field="failure report")
    if canonical_failure_report_bytes(report) != payload:
        raise ReplacementEligibilityError("failure report must use canonical JSON bytes")
    _validate_report_mapping(report)
    return MappingProxyType(dict(report))


def validate_failure_report_file(
    path: str | Path,
    *,
    expected_sha256: str | None = None,
) -> Mapping[str, Any]:
    """Read one no-follow stable report file and validate its canonical bytes."""

    report_file = _stable_regular_file(Path(path), field="failure report file")
    return validate_failure_report_bytes(
        report_file.content,
        expected_sha256=expected_sha256,
    )


def validate_local_predecessor(
    project_root: str | Path,
    report: str | Path | Mapping[str, Any],
    *,
    expected_sha256: str | None = None,
) -> LocalPredecessorValidation:
    """Prove that the retained local attempt exactly matches the authorized public report.

    All checks are read-only.  Failure raises :class:`ReplacementEligibilityError`; a returned
    value therefore always has ``eligible=True``.
    """

    root = _real_project_root(project_root)
    if isinstance(report, Mapping):
        payload = canonical_failure_report_bytes(report)
        validated = validate_failure_report_bytes(payload, expected_sha256=expected_sha256)
    else:
        path = Path(report)
        if not path.is_absolute():
            path = root / path
        report_file = _stable_regular_file(path, field="failure report file")
        payload = report_file.content
        validated = validate_failure_report_bytes(payload, expected_sha256=expected_sha256)
    local = build_failure_report(root)
    if canonical_failure_report_bytes(local) != payload:
        raise ReplacementEligibilityError(
            "local attempt 001 differs from the canonical public failure report"
        )
    transaction = validated["transaction"]
    predecessor = validated["predecessor"]
    return LocalPredecessorValidation(
        eligible=True,
        report_sha256=hashlib.sha256(payload).hexdigest(),
        transaction_tree_sha256=transaction["tree_sha256"],
        predecessor_source_revision=predecessor["identity"]["source_revision"],
        successor_run_id=M8_REPLACEMENT_RUN_ID,
    )


__all__ = [
    "M8_ATTEMPT_001_CONTROLLER_SNAPSHOT_RELATIVE_PATH",
    "M8_ATTEMPT_001_FAILURE_REPORT_RELATIVE_PATH",
    "M8_ATTEMPT_001_FAILURE_REPORT_SCHEMA_VERSION",
    "M8_ATTEMPT_001_RUN_ID",
    "M8_ATTEMPT_001_TRANSACTION_RELATIVE_PATH",
    "M8_REPLACEMENT_MAX_ATTEMPTS",
    "M8_REPLACEMENT_RUN_ID",
    "LocalPredecessorValidation",
    "ReplacementEligibilityError",
    "build_failure_report",
    "canonical_failure_report_bytes",
    "validate_failure_report_bytes",
    "validate_failure_report_file",
    "validate_local_predecessor",
]
