from __future__ import annotations

import ast
import hashlib
import importlib.util
import os
import sys
import tomllib
from pathlib import Path
from types import ModuleType, SimpleNamespace

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[3]
SCRIPT_PATH = PROJECT_ROOT / "scripts/benchmark_m8_controllers.py"
SOURCE = SCRIPT_PATH.read_text(encoding="utf-8")
TREE = ast.parse(SOURCE)


@pytest.fixture(scope="module")
def cli_module():
    name = "_test_benchmark_m8_controllers"
    spec = importlib.util.spec_from_file_location(name, SCRIPT_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    try:
        spec.loader.exec_module(module)
        yield module
    finally:
        sys.modules.pop(name, None)


@pytest.fixture(autouse=True)
def reset_formal_process_latches(cli_module, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(cli_module, "_BOOTSTRAP_STATE", None)
    monkeypatch.setattr(cli_module, "_EARLY_RECOVERY_LOCKDOWN", False)
    monkeypatch.setattr(cli_module, "_FORMAL_ENTRY_CONSUMED", False)
    monkeypatch.setattr(cli_module, "_POST_BIND_COMMANDS_FORBIDDEN", False)
    monkeypatch.setattr(cli_module, "_PROJECT_SOURCE_FINDER", None)


def _function(name: str) -> ast.FunctionDef:
    return next(
        node for node in TREE.body if isinstance(node, ast.FunctionDef) and node.name == name
    )


def _call_names(function: ast.FunctionDef) -> list[str]:
    names: list[str] = []
    for node in ast.walk(function):
        if not isinstance(node, ast.Call):
            continue
        target = node.func
        if isinstance(target, ast.Name):
            names.append(target.id)
        elif isinstance(target, ast.Attribute):
            names.append(target.attr)
    return names


def test_top_level_is_stdlib_only_and_sets_import_policy_first() -> None:
    imports = [node for node in TREE.body if isinstance(node, (ast.Import, ast.ImportFrom))]
    imported_names = []
    for node in imports:
        if isinstance(node, ast.Import):
            imported_names.extend(alias.name for alias in node.names)
        elif node.module is not None:
            imported_names.append(node.module)
    assert not any(
        name == "controller_learning" or name.startswith("controller_learning.")
        for name in imported_names
    )

    policy_lines = {
        token: SOURCE.index(f'os.environ.setdefault("{token}"')
        for token in (
            "CUDA_DEVICE_ORDER",
            "XLA_PYTHON_CLIENT_PREALLOCATE",
            "PYTHONDONTWRITEBYTECODE",
        )
    }
    assert max(policy_lines.values()) < SOURCE.index("PROJECT_ROOT: Final")
    assert max(policy_lines.values()) < SOURCE.index("def _load_project_api")


def test_canonical_cli_rejects_any_other_config(cli_module, tmp_path: Path) -> None:
    assert cli_module._parse_args([]).config == Path("configs/final_evaluation.toml")
    assert cli_module._parse_args(["--config", "configs/final_evaluation.toml"]).config == Path(
        "configs/final_evaluation.toml"
    )
    with pytest.raises(SystemExit):
        cli_module._parse_args(["--output", "elsewhere.json"])

    other = tmp_path / "other.toml"
    other.write_text("schema_version = 1\n", encoding="utf-8")
    with pytest.raises(RuntimeError, match="accepts only"):
        cli_module._canonical_config_path(PROJECT_ROOT, other)


def test_bootstrap_installs_private_guard_before_project_imports() -> None:
    top_level_project_calls = [
        node
        for node in TREE.body
        if isinstance(node, ast.Expr)
        and isinstance(node.value, ast.Call)
        and "controller_learning" in ast.unparse(node.value)
    ]
    assert top_level_project_calls == []

    bootstrap = _function("_bootstrap_test_guard")
    bootstrap_calls = _call_names(bootstrap)
    assert bootstrap_calls.index("_assert_project_not_imported") < bootstrap_calls.index(
        "_load_private_module"
    )
    assert bootstrap_calls.index("_load_private_module") < bootstrap_calls.index("install")

    main = _function("main")
    main_source = ast.get_source_segment(SOURCE, main)
    assert main_source is not None
    assert main_source.index("_prepare_isolated_python_runtime") < main_source.index(
        "_bootstrap_test_guard"
    )
    assert main_source.index("_bootstrap_test_guard") < main_source.index("run_benchmark")
    assert "_load_project_api" not in main_source


def test_formal_startup_requires_exact_isolated_no_site_gpu_python() -> None:
    imports = [
        alias.name for node in TREE.body if isinstance(node, ast.Import) for alias in node.names
    ]
    assert "site" not in imports
    prepare_source = ast.get_source_segment(SOURCE, _function("_prepare_isolated_python_runtime"))
    assert prepare_source is not None
    for requirement in (
        "sys.flags.isolated",
        "sys.flags.ignore_environment",
        "sys.flags.no_user_site",
        "sys.flags.no_site",
        "sys.flags.dont_write_bytecode",
        "sys.flags.safe_path",
        "Path(sys.executable).resolve(strict=True)",
        "Path(sys.prefix) != gpu_prefix",
        "Path(sys.base_prefix) != gpu_prefix",
        '"LD_LIBRARY_PATH"',
        '"LD_PRELOAD"',
    ):
        assert requirement in prepare_source


def test_run_installs_exact_source_finder_and_site_route_before_project_api() -> None:
    source = ast.get_source_segment(SOURCE, _function("run_benchmark"))
    assert source is not None
    consume = source.index("_consume_bootstrap_guard")
    finder = source.index("_install_project_source_finder")
    first_remove = source.index("_remove_project_import_root")
    site_setup = source.index("_configure_gpu_site_packages")
    load_api = source.index("_load_project_api")
    second_remove = source.index("_remove_project_import_root", first_remove + 1)
    assert consume < finder < first_remove < site_setup < load_api < second_remove
    assert "site.addsitedir" not in SOURCE
    assert "controller_learning.physics.mjx_warp" in SOURCE


def test_project_source_finder_accepts_only_exact_regular_python_sources(
    cli_module,
    tmp_path: Path,
) -> None:
    package = tmp_path / "controller_learning"
    package.mkdir()
    (package / "probe.py").write_text("VALUE = 7\n", encoding="utf-8")
    finder = cli_module._ProjectSourceFinder(tmp_path)

    spec = finder.find_spec("controller_learning.probe")
    assert spec is not None
    assert type(spec.loader) is cli_module.importlib.machinery.SourceFileLoader
    module = cli_module.importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    assert module.VALUE == 7
    assert Path(spec.origin) == package / "probe.py"

    (package / "namespace_probe").mkdir()
    with pytest.raises(ImportError, match="namespace"):
        finder.find_spec("controller_learning.namespace_probe")

    (package / "extension_probe.so").write_bytes(b"not an extension")
    with pytest.raises(ModuleNotFoundError, match="no exact regular Python source"):
        finder.find_spec("controller_learning.extension_probe")

    (package / "link_probe.py").symlink_to(package / "probe.py")
    with pytest.raises(RuntimeError, match="symbolic link"):
        finder.find_spec("controller_learning.link_probe")


def test_canonical_root_binding_rejects_an_alternate_project(cli_module, tmp_path: Path) -> None:
    with pytest.raises(RuntimeError, match="bound to the project root"):
        cli_module._canonical_project_root(tmp_path)


def test_project_module_provenance_rejects_shadow_and_namespace_modules(
    cli_module,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    import controller_learning

    assert controller_learning.__file__ is not None
    for name in tuple(sys.modules):
        if name.startswith("controller_learning."):
            monkeypatch.delitem(sys.modules, name)
    cli_module._validate_project_module_provenance(PROJECT_ROOT)

    shadow = ModuleType("controller_learning.shadow_probe")
    shadow.__file__ = str(tmp_path / "shadow_probe.py")
    shadow.__spec__ = SimpleNamespace(origin=shadow.__file__)
    monkeypatch.setitem(sys.modules, shadow.__name__, shadow)
    with pytest.raises(RuntimeError, match=r"missing|exact source"):
        cli_module._validate_project_module_provenance(PROJECT_ROOT)
    monkeypatch.delitem(sys.modules, shadow.__name__)

    namespace = ModuleType("controller_learning.namespace_probe")
    namespace.__path__ = [str(tmp_path)]
    namespace.__spec__ = SimpleNamespace(origin=None)
    monkeypatch.setitem(sys.modules, namespace.__name__, namespace)
    with pytest.raises(RuntimeError, match="namespace, zip, or shadowed"):
        cli_module._validate_project_module_provenance(PROJECT_ROOT)


def test_secure_git_runner_uses_fixed_binary_and_clean_environment(
    cli_module,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    captured: dict[str, object] = {}

    def run(command, **kwargs):
        captured["command"] = command
        captured.update(kwargs)
        return SimpleNamespace(stdout="a" * 40 + "\n")

    monkeypatch.setattr(cli_module.subprocess, "run", run)
    output = cli_module._secure_git_command_runner(
        ("git", "rev-parse", "--verify", "HEAD"),
        tmp_path,
    )

    command = captured["command"]
    assert isinstance(command, tuple)
    assert command[0] == "/usr/bin/git"
    assert command[-3:] == ("rev-parse", "--verify", "HEAD")
    assert captured["cwd"] == tmp_path
    assert captured["env"] == {
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
    assert output == "a" * 40


def test_private_guard_requires_audit_response_and_is_consumed_once(
    cli_module,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    guard_type = type(
        "M8TestAssetAccessGuard",
        (),
        {
            "__module__": cli_module._PRIVATE_GUARD_MODULE,
            "_deterministic_recovery": False,
            "evidence": lambda self, *, test_loaded: {
                "all_track_reads_forbidden": False,
                "audit_hook_installed_before_preflight": True,
                "denied_event_count": 0,
                "open_event_counts": {},
                "test_reads_enabled": False,
            },
        },
    )
    guard = guard_type()
    monkeypatch.setattr(cli_module, "_assert_project_not_imported", lambda: None)
    cli_module._BOOTSTRAP_STATE = cli_module._BootstrapState(
        guard=guard,
        guard_class=guard_type,
        process_id=cli_module.os.getpid(),
        project_root=PROJECT_ROOT,
        recovery_lockdown=False,
        nonce=cli_module._BOOTSTRAP_NONCE,
    )

    with pytest.raises(RuntimeError, match="audit-hook identity state"):
        cli_module._consume_bootstrap_guard(PROJECT_ROOT)

    token = object()
    guard._audit_identity_token = token
    guard._audit_identity_response = None
    guard._AUDIT_SELF_CHECK_EVENT = "synthetic.guard.self_check"

    def respond(event: str, supplied: object) -> None:
        if event == guard._AUDIT_SELF_CHECK_EVENT and supplied is token:
            guard._audit_identity_response = token

    monkeypatch.setattr(cli_module.sys, "audit", respond)
    assert cli_module._consume_bootstrap_guard(PROJECT_ROOT) is guard
    with pytest.raises(RuntimeError, match="unconsumed"):
        cli_module._consume_bootstrap_guard(PROJECT_ROOT)


def test_private_guard_consumer_performs_the_audit_identity_exchange_directly() -> None:
    function = _function("_consume_bootstrap_guard")
    source = ast.get_source_segment(SOURCE, function)
    assert source is not None
    assert "sys.audit(audit_event, audit_token)" in source
    assert "_audit_identity_response is not audit_token" in source
    assert "assert_audit_hook_active" not in source


def test_transaction_appearing_before_dependency_import_enters_recovery_lockdown(
    cli_module,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    events: list[str] = []

    def evidence(_self, *, test_loaded: bool):
        assert test_loaded is False
        return {
            "all_track_reads_forbidden": False,
            "audit_hook_installed_before_preflight": True,
            "denied_event_count": 0,
            "open_event_counts": {},
            "test_reads_enabled": False,
        }

    def enter_recovery(self) -> None:
        events.append("lockdown")
        self._deterministic_recovery = True

    guard_type = type(
        "M8TestAssetAccessGuard",
        (),
        {
            "__module__": cli_module._PRIVATE_GUARD_MODULE,
            "_deterministic_recovery": False,
            "enter_deterministic_recovery": enter_recovery,
            "evidence": evidence,
        },
    )
    guard = guard_type()
    token = object()
    guard._audit_identity_token = token
    guard._audit_identity_response = None
    guard._AUDIT_SELF_CHECK_EVENT = "synthetic.guard.recovery"
    cli_module._BOOTSTRAP_STATE = cli_module._BootstrapState(
        guard=guard,
        guard_class=guard_type,
        process_id=cli_module.os.getpid(),
        project_root=PROJECT_ROOT,
        recovery_lockdown=False,
        nonce=cli_module._BOOTSTRAP_NONCE,
    )
    monkeypatch.setattr(cli_module, "_assert_project_not_imported", lambda: None)
    monkeypatch.setattr(cli_module, "_active_attempt_transaction_exists", lambda _root: True)

    def respond(event: str, supplied: object) -> None:
        if event == guard._AUDIT_SELF_CHECK_EVENT and supplied is token:
            guard._audit_identity_response = token

    monkeypatch.setattr(cli_module.sys, "audit", respond)
    assert cli_module._consume_bootstrap_guard(PROJECT_ROOT) is guard
    assert events == ["lockdown"]
    assert cli_module._EARLY_RECOVERY_LOCKDOWN is True


def test_private_guard_rejects_fake_module_and_forked_pid(
    cli_module,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    guard_type = type("M8TestAssetAccessGuard", (), {"__module__": "fake_guard"})
    cli_module._BOOTSTRAP_STATE = cli_module._BootstrapState(
        guard=guard_type(),
        guard_class=guard_type,
        process_id=cli_module.os.getpid(),
        project_root=PROJECT_ROOT,
        recovery_lockdown=False,
        nonce=cli_module._BOOTSTRAP_NONCE,
    )
    monkeypatch.setattr(cli_module, "_assert_project_not_imported", lambda: None)
    with pytest.raises(RuntimeError, match="identity"):
        cli_module._consume_bootstrap_guard(PROJECT_ROOT)

    private_type = type(
        "M8TestAssetAccessGuard",
        (),
        {"__module__": cli_module._PRIVATE_GUARD_MODULE},
    )
    cli_module._BOOTSTRAP_STATE = cli_module._BootstrapState(
        guard=private_type(),
        guard_class=private_type,
        process_id=cli_module.os.getpid() + 1,
        project_root=PROJECT_ROOT,
        recovery_lockdown=False,
        nonce=cli_module._BOOTSTRAP_NONCE,
    )
    with pytest.raises(RuntimeError, match="process"):
        cli_module._consume_bootstrap_guard(PROJECT_ROOT)


def test_transaction_uses_exact_frozen_output_allowlist(cli_module, tmp_path: Path) -> None:
    output_paths = tuple(f"results/file-{index:02d}" for index in range(24))
    captured: dict[str, object] = {}

    class Identity:
        def __init__(self, **kwargs):
            captured["identity_kwargs"] = kwargs

    class Transaction:
        def __init__(self, project_root, **kwargs):
            captured["project_root"] = project_root
            captured.update(kwargs)

    api = SimpleNamespace(
        attempt=SimpleNamespace(AttemptIdentity=Identity, M8AttemptTransaction=Transaction),
        benchmark=SimpleNamespace(formal_output_paths=lambda _config: output_paths),
    )
    static = SimpleNamespace(
        source_revision="a" * 40,
        source_tree_sha256="e" * 64,
        config_sha256="b" * 64,
        pixi_lock_sha256="c" * 64,
        reports_digest="d" * 64,
        config=object(),
    )
    cli_module._attempt_transaction(api, tmp_path, static)
    assert captured["identity_kwargs"] == {
        "config_sha256": "b" * 64,
        "input_sha256": "d" * 64,
        "pixi_lock_sha256": "c" * 64,
        "source_revision": "a" * 40,
        "source_tree_sha256": "e" * 64,
    }
    assert captured["transaction_relative_path"] == "runs/m8_final_attempt_transaction"
    assert captured["output_allowlist"] == output_paths
    assert len(captured["output_allowlist"]) == 24


def test_episode_sink_validates_then_appends_bundle_before_phase_sample(
    cli_module,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    events: list[object] = []
    api = SimpleNamespace(
        preflight=SimpleNamespace(
            validate_frozen_controller_snapshot=lambda *_args: events.append("validated")
        ),
        trajectory=SimpleNamespace(
            canonical_trajectory_json_bytes=lambda _trajectory: b"trajectory\n"
        ),
    )
    transaction = SimpleNamespace(
        append_episode_bundle=lambda record, payload: events.append(("bundle", record, payload))
    )
    memory = SimpleNamespace(sample=lambda label: events.append(("memory", label)))
    config = SimpleNamespace(test_track_count=20)
    recorded = SimpleNamespace(trajectory=object())
    monkeypatch.setattr(cli_module, "_journal_record", lambda *_args: "journal-record")
    sink = cli_module._episode_bundle_sink(
        api,
        transaction,
        object(),
        config,
        memory,
    )
    sink("pid", 19, recorded, object())
    assert events == [
        "validated",
        ("bundle", "journal-record", b"trajectory\n"),
        ("memory", "after_pid_controller"),
    ]


def test_recovery_refuses_incomplete_test_bound_before_reconstruction(
    cli_module,
    tmp_path: Path,
) -> None:
    class Incomplete(RuntimeError):
        pass

    test_bound = object()
    committed = object()
    inspection = SimpleNamespace(
        exists=True,
        phase=test_bound,
        journal_record_count=59,
    )
    transaction = SimpleNamespace(inspect=lambda: inspection)
    api = SimpleNamespace(
        attempt=SimpleNamespace(
            AttemptPhase=SimpleNamespace(TEST_BOUND=test_bound, COMMITTED=committed),
            FORMAL_EPISODE_COUNT=60,
            IncompleteTestAttemptError=Incomplete,
        ),
        preflight=SimpleNamespace(require_controller_snapshot_quarantine_absent=lambda _root: None),
    )
    with pytest.raises(Incomplete):
        cli_module._resume_attempt(api, tmp_path, SimpleNamespace(), transaction)


def test_prepared_recovery_isolates_snapshot_before_transaction_retirement(
    cli_module,
    tmp_path: Path,
) -> None:
    prepared = object()
    committed = object()
    inspection = SimpleNamespace(exists=True, phase=prepared, journal_record_count=0)
    events: list[str] = []
    transaction = SimpleNamespace(
        inspect=lambda: inspection,
        recover=lambda: events.append("recover") or SimpleNamespace(action="pre_test_restored"),
    )
    api = SimpleNamespace(
        attempt=SimpleNamespace(
            AttemptPhase=SimpleNamespace(PREPARED=prepared, COMMITTED=committed),
        ),
        preflight=SimpleNamespace(
            require_controller_snapshot_quarantine_absent=lambda _root: events.append("gate"),
            isolate_aborted_controller_snapshot=lambda _root: events.append("isolate"),
        ),
    )

    with pytest.raises(cli_module.PreparedRecoveryRerunRequired, match="rerun is required"):
        cli_module._recover_pre_test_attempt(
            api,
            transaction,
            tmp_path,
            recovery_lockdown=True,
        )
    assert events == ["gate", "isolate", "recover"]


def test_prepared_recovery_preserves_transaction_when_snapshot_isolation_fails(
    cli_module,
    tmp_path: Path,
) -> None:
    prepared = object()
    committed = object()
    inspection = SimpleNamespace(exists=True, phase=prepared, journal_record_count=0)
    recover_called = False

    def recover():
        nonlocal recover_called
        recover_called = True
        return SimpleNamespace(action="pre_test_restored")

    transaction = SimpleNamespace(inspect=lambda: inspection, recover=recover)
    api = SimpleNamespace(
        attempt=SimpleNamespace(
            AttemptPhase=SimpleNamespace(PREPARED=prepared, COMMITTED=committed),
        ),
        preflight=SimpleNamespace(
            require_controller_snapshot_quarantine_absent=lambda _root: None,
            isolate_aborted_controller_snapshot=lambda _root: (_ for _ in ()).throw(
                OSError("isolation failed")
            ),
        ),
    )

    with pytest.raises(OSError, match="isolation failed"):
        cli_module._recover_pre_test_attempt(
            api,
            transaction,
            tmp_path,
            recovery_lockdown=True,
        )
    assert recover_called is False


def test_existing_attempt_recovery_requires_early_lockdown(cli_module, tmp_path: Path) -> None:
    transaction = SimpleNamespace(
        inspect=lambda: SimpleNamespace(exists=True, phase=object(), journal_record_count=0)
    )
    with pytest.raises(RuntimeError, match="not locked before dependency imports"):
        cli_module._recover_pre_test_attempt(
            SimpleNamespace(),
            transaction,
            tmp_path,
            recovery_lockdown=False,
        )


def test_fresh_snapshot_creation_failure_isolates_before_recovering_transaction(
    cli_module,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    class CreationFailed(RuntimeError):
        pass

    events: list[str] = []
    api = SimpleNamespace(
        benchmark=SimpleNamespace(
            M8_CONTROLLER_ORDER=("pid", "mpc", "ppo"),
            validate_formal_output_tree=lambda *_args, **_kwargs: events.append("outputs"),
        ),
        controller_identity=SimpleNamespace(
            capture_frozen_controller_identity=lambda _root, name: (
                events.append(f"identity:{name}") or object()
            )
        ),
        preflight=SimpleNamespace(
            require_controller_snapshot_quarantine_absent=lambda _root: events.append("gate"),
            capture_clean_source=lambda _root, **_kwargs: SimpleNamespace(revision="a" * 40),
            create_frozen_controller_snapshot=lambda *_args, **_kwargs: (
                events.append("create") or (_ for _ in ()).throw(CreationFailed("creation failed"))
            ),
            isolate_aborted_controller_snapshot=lambda _root: events.append("isolate"),
        ),
    )
    transaction = SimpleNamespace(
        prepare=lambda: events.append("prepare"),
        recover=lambda: events.append("recover"),
    )
    monkeypatch.setattr(cli_module, "_assert_torch_absent", lambda: None)

    with pytest.raises(CreationFailed, match="creation failed"):
        cli_module._fresh_attempt(
            api,
            object(),
            tmp_path,
            SimpleNamespace(config=object(), source_revision="a" * 40),
            transaction,
        )
    assert events[-4:] == ["prepare", "create", "isolate", "recover"]


def test_fresh_snapshot_isolation_failure_does_not_recover_prepared_transaction(
    cli_module,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    recover_called = False

    def recover():
        nonlocal recover_called
        recover_called = True

    api = SimpleNamespace(
        benchmark=SimpleNamespace(
            M8_CONTROLLER_ORDER=("pid", "mpc", "ppo"),
            validate_formal_output_tree=lambda *_args, **_kwargs: None,
        ),
        controller_identity=SimpleNamespace(
            capture_frozen_controller_identity=lambda *_args: object()
        ),
        preflight=SimpleNamespace(
            require_controller_snapshot_quarantine_absent=lambda _root: None,
            capture_clean_source=lambda _root, **_kwargs: SimpleNamespace(revision="a" * 40),
            create_frozen_controller_snapshot=lambda *_args, **_kwargs: (_ for _ in ()).throw(
                RuntimeError("creation failed")
            ),
            isolate_aborted_controller_snapshot=lambda _root: (_ for _ in ()).throw(
                OSError("isolation failed")
            ),
        ),
    )
    transaction = SimpleNamespace(prepare=lambda: None, recover=recover)
    monkeypatch.setattr(cli_module, "_assert_torch_absent", lambda: None)

    with pytest.raises(OSError, match="isolation failed"):
        cli_module._fresh_attempt(
            api,
            object(),
            tmp_path,
            SimpleNamespace(config=object(), source_revision="a" * 40),
            transaction,
        )
    assert recover_called is False


def test_transaction_absent_rejects_any_active_snapshot_entry(
    cli_module,
    tmp_path: Path,
) -> None:
    active = tmp_path / cli_module.SNAPSHOT_RELATIVE_PATH
    active.parent.mkdir()
    active.write_bytes(b"unexpected")
    with pytest.raises(RuntimeError, match="without its transaction"):
        cli_module._assert_active_snapshot_absent(tmp_path)

    active.unlink()
    active.symlink_to(tmp_path / "missing-target")
    with pytest.raises(RuntimeError, match="without its transaction"):
        cli_module._assert_active_snapshot_absent(tmp_path)


def test_recovery_requires_post_close_seal_even_with_all_60_rows(
    cli_module,
    tmp_path: Path,
) -> None:
    class MissingSeal(RuntimeError):
        pass

    test_bound = object()
    committed = object()
    inspection = SimpleNamespace(
        exists=True,
        phase=test_bound,
        journal_record_count=60,
    )

    def refuse_unsealed():
        raise MissingSeal("execution evidence sealed=false")

    transaction = SimpleNamespace(inspect=lambda: inspection, recover=refuse_unsealed)
    api = SimpleNamespace(
        attempt=SimpleNamespace(
            AttemptPhase=SimpleNamespace(TEST_BOUND=test_bound, COMMITTED=committed),
            FORMAL_EPISODE_COUNT=60,
        ),
        preflight=SimpleNamespace(require_controller_snapshot_quarantine_absent=lambda _root: None),
    )
    with pytest.raises(MissingSeal, match="sealed=false"):
        cli_module._resume_attempt(api, tmp_path, SimpleNamespace(), transaction)


def test_committed_recovery_uses_verified_source_when_snapshot_is_absent(
    cli_module,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    test_bound = object()
    committed = object()
    artifacts_validated = object()
    evaluation_complete = object()
    inspection = SimpleNamespace(
        exists=True,
        phase=committed,
        journal_record_count=60,
    )
    transaction = SimpleNamespace(
        inspect=lambda: inspection,
        read_staged_outputs=lambda: {"artifact": b"bytes"},
    )
    identity = object()
    results = {"pid": object(), "mpc": object(), "ppo": object()}
    api = SimpleNamespace(
        attempt=SimpleNamespace(
            AttemptPhase=SimpleNamespace(
                TEST_BOUND=test_bound,
                COMMITTED=committed,
                ARTIFACTS_VALIDATED=artifacts_validated,
                EVALUATION_COMPLETE=evaluation_complete,
            ),
            FORMAL_EPISODE_COUNT=60,
        ),
        benchmark=SimpleNamespace(M8_CONTROLLER_ORDER=("pid", "mpc", "ppo")),
        controller_identity=SimpleNamespace(
            capture_frozen_controller_identity=lambda _root, _name: identity
        ),
        report=SimpleNamespace(validate_m8_publication=lambda *_args, **_kwargs: "a" * 64),
    )
    monkeypatch.setattr(cli_module, "_validate_static_stability", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(cli_module, "_results_from_durable_journal", lambda *_args: results)
    monkeypatch.setattr(cli_module, "_load_durable_evidence", lambda *_args: object())
    monkeypatch.setattr(cli_module, "_publication_evidence_kwargs", lambda *_args: {})
    cleanup_events: list[Path] = []
    api.preflight = SimpleNamespace(
        retire_committed_controller_snapshot=lambda root: cleanup_events.append(root)
    )
    monkeypatch.setattr(
        cli_module,
        "_snapshot_from_existing",
        lambda *_args: pytest.fail("COMMITTED recovery must not require the removed snapshot"),
    )

    recovered = cli_module._resume_attempt(
        api,
        tmp_path,
        SimpleNamespace(config=object()),
        transaction,
    )
    assert recovered is results
    assert cleanup_events == [tmp_path]


def test_partial_publication_is_restored_before_clean_source_gate(
    cli_module,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    class StopAfterRecovery(RuntimeError):
        pass

    test_bound = object()
    artifacts_validated = object()
    committed = object()
    inspection = SimpleNamespace(
        exists=True,
        phase=artifacts_validated,
        journal_record_count=60,
    )
    events: list[str] = []
    transaction = SimpleNamespace(
        inspect=lambda: inspection,
        recover=lambda: (
            events.append("restored") or SimpleNamespace(action="partial_publication_restored")
        ),
    )
    api = SimpleNamespace(
        attempt=SimpleNamespace(
            AttemptPhase=SimpleNamespace(
                TEST_BOUND=test_bound,
                ARTIFACTS_VALIDATED=artifacts_validated,
                COMMITTED=committed,
            ),
            FORMAL_EPISODE_COUNT=60,
        ),
        benchmark=SimpleNamespace(M8_CONTROLLER_ORDER=("pid", "mpc", "ppo")),
        controller_identity=SimpleNamespace(
            capture_frozen_controller_identity=lambda _root, _name: object()
        ),
        preflight=SimpleNamespace(require_controller_snapshot_quarantine_absent=lambda _root: None),
    )

    def stop_after_restore(*_args):
        assert events == ["restored"]
        raise StopAfterRecovery

    monkeypatch.setattr(cli_module, "_snapshot_from_existing", stop_after_restore)
    monkeypatch.setattr(cli_module, "_validate_static_stability", lambda *_args, **_kwargs: None)
    with pytest.raises(StopAfterRecovery):
        cli_module._resume_attempt(api, tmp_path, SimpleNamespace(config=object()), transaction)


def test_journal_reconstruction_uses_blob_without_rollout(cli_module) -> None:
    events: list[str] = []
    trajectory = SimpleNamespace(
        total_reward=1.25,
        terminated=[True],
        truncated=[False],
        final_info={"termination_reason": 1},
    )

    class EpisodeRunResult:
        def __init__(self, **kwargs):
            self.kwargs = kwargs

    class Recorded:
        def __init__(self, *, result, trajectory):
            self.result = result
            self.trajectory = trajectory

    api = SimpleNamespace(
        control=SimpleNamespace(EpisodeRunResult=EpisodeRunResult),
        trajectory=SimpleNamespace(
            load_trajectory_json_bytes=lambda payload, expected_sha256: (
                events.append("load_trajectory") or trajectory
            ),
            RecordedControllerEpisode=Recorded,
        ),
    )
    transaction = SimpleNamespace(
        read_blob=lambda path: events.append(f"read:{path}") or b"trajectory\n"
    )
    record = SimpleNamespace(
        trajectory_blob_path="episodes/pid/row_000_trajectory.json",
        trajectory_blob_sha256="a" * 64,
        steps=1,
        data={
            "controller_import_time_s": 0.1,
            "controller_init_time_s": 0.2,
            "compute_times_s": [0.003],
        },
    )
    reconstructed = cli_module._recorded_episode_from_journal(api, transaction, record)
    assert reconstructed.trajectory is trajectory
    assert reconstructed.result.kwargs["compute_times_s"] == (0.003,)
    assert events == ["read:episodes/pid/row_000_trajectory.json", "load_trajectory"]


def test_unified_infrastructure_failure_blob_is_sanitized(cli_module) -> None:
    captured: dict[str, bytes] = {}
    transaction = SimpleNamespace(
        write_blob=lambda path, payload: captured.setdefault(path, payload)
    )
    error = RuntimeError("failed at /home/private/controller_learning password=super-secret-value")

    cli_module._write_sanitized_infrastructure_failure(
        transaction,
        error,
        infrastructure_phase="final_memory",
    )

    payload = captured["failures/final-workload.json"]
    decoded = payload.decode("ascii")
    assert "/home/private" not in decoded
    assert "super-secret-value" not in decoded
    assert "<path>" in decoded
    assert "<redacted>" in decoded
    assert '"infrastructure_phase":"final_memory"' in decoded


def test_post_bind_command_latch_rejects_cli_subprocess(
    cli_module,
    tmp_path: Path,
) -> None:
    cli_module._enter_post_bind_phase()
    with pytest.raises(RuntimeError, match="forbidden after TEST_BOUND"):
        cli_module._run_command(("git", "status"), cwd=tmp_path)


def test_post_bind_static_gate_recomputes_source_identity_without_git_or_track_open(
    cli_module,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    from controller_learning.evaluation import source_tree_identity

    (tmp_path / ".git").mkdir()
    (tmp_path / ".git/HEAD").write_text("a" * 40 + "\n", encoding="ascii")
    config_path = tmp_path / "config.toml"
    config_bytes = b"config\n"
    config_path.write_bytes(config_bytes)
    lock_path = tmp_path / "pixi.lock"
    lock_bytes = b"lock\n"
    lock_path.write_bytes(lock_bytes)
    source = tmp_path / "source.py"
    source.write_bytes(b"VALUE = 1\n")
    protected = tmp_path / "controller_learning/assets/tracks/v0.1"
    protected.mkdir(parents=True)
    (protected / "test.npz").write_bytes(b"protected")
    baseline = source_tree_identity.capture_source_tree_identity(tmp_path)

    def sha256_regular_file(path: Path):
        payload = path.read_bytes()
        return hashlib.sha256(payload).hexdigest(), len(payload), payload

    api = SimpleNamespace(
        benchmark=SimpleNamespace(M8_CONTROLLER_ORDER=()),
        preflight=SimpleNamespace(
            sha256_regular_file=sha256_regular_file,
            load_frozen_input_reports=lambda *_args: {},
            frozen_input_digest=lambda _reports: "d" * 64,
        ),
        source_tree=source_tree_identity,
    )
    static = SimpleNamespace(
        config_path=config_path,
        config=object(),
        config_bytes=config_bytes,
        config_sha256=hashlib.sha256(config_bytes).hexdigest(),
        pixi_lock_bytes=lock_bytes,
        pixi_lock_sha256=hashlib.sha256(lock_bytes).hexdigest(),
        reports_digest="d" * 64,
        source_revision="a" * 40,
        source_tree_identity=baseline,
        source_tree_sha256=baseline.aggregate_sha256,
    )
    original_open = source_tree_identity.os.open
    original_scandir = source_tree_identity.os.scandir

    def guarded_open(path, *args, **kwargs):
        name = os.fsdecode(os.fspath(path))
        if name in {"v0.1", "test.npz"} or "assets/tracks/v0.1" in name:
            raise AssertionError("post-bind source identity opened protected Track")
        return original_open(path, *args, **kwargs)

    def guarded_scandir(path):
        if not isinstance(path, int) and "assets/tracks/v0.1" in os.fsdecode(os.fspath(path)):
            raise AssertionError("post-bind source identity enumerated protected Track")
        return original_scandir(path)

    monkeypatch.setattr(source_tree_identity.os, "open", guarded_open)
    monkeypatch.setattr(source_tree_identity.os, "scandir", guarded_scandir)
    monkeypatch.setattr(
        cli_module.subprocess,
        "run",
        lambda *_args, **_kwargs: pytest.fail("post-bind static gate spawned a subprocess"),
    )
    monkeypatch.setattr(cli_module, "_assert_project_source_finder_active", lambda _root: None)
    monkeypatch.setattr(cli_module, "_validate_project_module_provenance", lambda _root: None)
    monkeypatch.setattr(cli_module, "_assert_torch_absent", lambda: None)
    cli_module._enter_post_bind_phase()

    cli_module._validate_static_stability(
        api,
        tmp_path,
        static,
        {},
        require_clean=False,
    )
    source.write_bytes(b"VALUE = 2\n")
    with pytest.raises(RuntimeError, match="source tree changed"):
        cli_module._validate_static_stability(
            api,
            tmp_path,
            static,
            {},
            require_clean=False,
        )


def test_semantic_validator_precedes_mark_and_publish() -> None:
    function = _function("_validate_stage_then_publish")
    source = ast.get_source_segment(SOURCE, function)
    assert source is not None
    first_validate = source.index("validate_m8_publication")
    mark = source.index("mark_artifacts_validated")
    publish = source.index("publish_and_commit")
    assert first_validate < mark < publish
    assert source.count("validate_m8_publication") == 2
    assert "retain_committed_transaction=True" in source


def test_every_static_stability_gate_revalidates_finder_and_loaded_project_modules() -> None:
    source = ast.get_source_segment(SOURCE, _function("_validate_static_stability"))
    assert source is not None
    assert source.count("_assert_project_source_finder_active") == 2
    assert source.count("_validate_project_module_provenance") == 2


def test_fresh_path_seals_post_close_evidence_before_staging() -> None:
    function = _function("_fresh_attempt")
    source = ast.get_source_segment(SOURCE, function)
    assert source is not None
    close = source.index("environment.close()")
    seal = source.index("write_execution_evidence")
    build = source.index("_build_outputs")
    complete = source.index("complete_evaluation")
    assert close < seal < build < complete
    assert "canonical_execution_evidence_bytes" in source
    assert "except BaseException as error" in source
    assert "_write_sanitized_infrastructure_failure" in source
    assert source.index("forbid_all_track_reads") < source.index("environment_create")


def test_process_seal_is_frozen_before_test_load_and_entered_for_recovery() -> None:
    fresh_source = ast.get_source_segment(SOURCE, _function("_fresh_attempt"))
    bootstrap_source = ast.get_source_segment(SOURCE, _function("_bootstrap_test_guard"))
    run_source = ast.get_source_segment(SOURCE, _function("run_benchmark"))
    assert fresh_source is not None
    assert bootstrap_source is not None
    assert run_source is not None
    resolve = fresh_source.index("resolve_nvidia_smi_executable")
    freeze = fresh_source.index("freeze_nvidia_smi_executable")
    enable = fresh_source.index("enable_test_reads")
    load = fresh_source.index("load_verified_test_pool")
    close = fresh_source.index("forbid_all_track_reads")
    assert resolve < freeze < enable < load < close
    assert "command_runner=guard.run_frozen_memory_query" in fresh_source
    assert bootstrap_source.index("install()") < bootstrap_source.index(
        "_active_attempt_transaction_exists"
    )
    assert bootstrap_source.index("_active_attempt_transaction_exists") < (
        bootstrap_source.index("enter_deterministic_recovery")
    )
    assert "enter_deterministic_recovery" not in run_source


def test_success_path_keeps_durable_committed_transaction() -> None:
    function = _function("run_benchmark")
    source = ast.get_source_segment(SOURCE, function)
    assert source is not None
    assert "AttemptPhase.COMMITTED" in source
    assert "output_state" in source
    assert "validate_committed_controller_snapshot_quarantine" in source
    assert "retire_committed" not in source


def test_formal_script_has_no_replay_rollout_path() -> None:
    call_names = _call_names(_function("_fresh_attempt"))
    assert "record_controller_episode" not in call_names
    assert "run_controller_episode" not in call_names
    assert call_names.count("create") == 1
    assert "execute_controller_workload" in call_names
    assert "render_trajectory_overview_png" in SOURCE


def test_pixi_exposes_the_formal_task_exactly_once() -> None:
    with (PROJECT_ROOT / "pyproject.toml").open("rb") as file:
        project = tomllib.load(file)
    tasks = project["tool"]["pixi"]["feature"]["gpu"]["tasks"]
    assert list(tasks).count("benchmark-m8-controllers") == 1
    assert tasks["benchmark-m8-controllers"]["cmd"] == (
        "python -I -B -S scripts/benchmark_m8_controllers.py"
    )
    assert "smoke-m8-controllers" not in tasks
