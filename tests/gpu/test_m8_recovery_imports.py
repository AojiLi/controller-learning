"""Recovery-lockdown import checks for the formal M8 entry point."""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[2]
SCRIPT_PATH = PROJECT_ROOT / "scripts/benchmark_m8_controllers.py"
pytestmark = pytest.mark.gpu


def test_recovery_lockdown_imports_full_project_api_without_process_creation() -> None:
    """The fixed EGL route must avoid GLFW/ctypes helper subprocesses during recovery."""

    code = r"""
import importlib.util
import sys
from pathlib import Path

root, script = map(Path, sys.argv[1:])
spec = importlib.util.spec_from_file_location("_m8_recovery_import_probe", script)
module = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = module
spec.loader.exec_module(module)
module._prepare_isolated_python_runtime(root)
module._bootstrap_test_guard(root)
guard = module._consume_bootstrap_guard(root)
if not guard._deterministic_recovery:
    guard.enter_deterministic_recovery()
module._remove_project_import_root(root)
module._install_project_source_finder(root)
module._configure_gpu_site_packages(root)
api = module._load_project_api(root)
evidence = guard.evidence(test_loaded=False)
if evidence["open_event_counts"] != {} or evidence["denied_event_count"] != 0:
    raise RuntimeError("recovery imports touched Track assets or attempted a denied operation")
print("recovery-imports-sealed", api.environment.__name__)
"""
    environment = {
        name: value
        for name, value in os.environ.items()
        if name not in {"LD_LIBRARY_PATH", "LD_PRELOAD"}
        and not name.startswith(("PYTHON", "_PYTHON"))
    }
    environment["MUJOCO_GL"] = "egl"
    environment["PYOPENGL_PLATFORM"] = "egl"
    completed = subprocess.run(
        (
            sys.executable,
            "-I",
            "-B",
            "-S",
            "-c",
            code,
            str(PROJECT_ROOT),
            str(SCRIPT_PATH),
        ),
        check=True,
        capture_output=True,
        env=environment,
        text=True,
        timeout=30,
    )

    assert completed.stdout.strip() == (
        "recovery-imports-sealed controller_learning.envs.car_racing"
    )
