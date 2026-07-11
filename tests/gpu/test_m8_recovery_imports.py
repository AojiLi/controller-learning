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


def test_warp_initializes_once_while_fake_guard_is_still_pre_bind() -> None:
    """Real Warp initialization must finish before the post-bind process latch."""

    code = r"""
import importlib
import importlib.util
import sys
from pathlib import Path
from types import SimpleNamespace

root, script = map(Path, sys.argv[1:])
spec = importlib.util.spec_from_file_location("_m8_warp_init_probe", script)
module = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = module
spec.loader.exec_module(module)
module._prepare_isolated_python_runtime(root)
module._remove_project_import_root(root)
module._configure_gpu_site_packages(root)

class FakeGuard:
    def __init__(self):
        self.phase = "pre_bind"
        self.process_events = []

    def audit(self, event, _arguments):
        if event == "subprocess.Popen" or event.startswith("os.posix_spawn"):
            self.process_events.append((self.phase, event))
            if self.phase != "pre_bind":
                raise RuntimeError("fake guard denied post-bind process creation")

guard = FakeGuard()
sys.addaudithook(guard.audit)
warp = importlib.import_module("warp")

class WarpProxy:
    def __init__(self):
        self._src = warp._src
        self.init_count = 0

    def init(self):
        self.init_count += 1
        warp.init()

proxy = WarpProxy()
module._initialize_warp_runtime(SimpleNamespace(warp=proxy))
guard.phase = "post_bind"
if proxy.init_count != 1 or warp._src.context.runtime is None:
    raise RuntimeError("Warp did not initialize exactly once before post-bind")
if any(phase != "pre_bind" for phase, _event in guard.process_events):
    raise RuntimeError("Warp process creation crossed the fake post-bind guard")
print("warp-init-pre-bind", proxy.init_count)
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

    assert completed.stdout.strip().endswith("warp-init-pre-bind 1")


def test_post_seal_generated_track_environment_needs_no_helper_process() -> None:
    """A generated-Track environment must run after sealing without official assets."""

    code = r"""
import importlib
import importlib.util
import sys
from pathlib import Path
from types import SimpleNamespace

root, script = map(Path, sys.argv[1:])
spec = importlib.util.spec_from_file_location("_m8_post_seal_environment_probe", script)
module = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = module
spec.loader.exec_module(module)
module._prepare_isolated_python_runtime(root)

guard_module = module._load_private_module(
    "_m8_fake_asset_guard",
    root / "controller_learning/evaluation/test_access.py",
)
official_root = root / "controller_learning/assets/tracks/v0.1"
fake_manifest = official_root / "__m8_fake_test_manifest_never_exists__.json"
fake_asset = official_root / "__m8_fake_test_asset_never_exists__.npz"
if fake_manifest.exists() or fake_asset.exists():
    raise RuntimeError("fake audit-only Track paths unexpectedly exist")
guard = guard_module.M8TestAssetAccessGuard(
    official_track_root=official_root,
    test_manifest=fake_manifest,
    test_asset=fake_asset,
    track_cache_root=root / ".track-cache/v0.1",
)
guard.install()

module._remove_project_import_root(root)
module._install_project_source_finder(root)
module._configure_gpu_site_packages(root)
config_module = importlib.import_module("controller_learning.config")
environment_module = importlib.import_module("controller_learning.envs.car_racing")
generator_module = importlib.import_module("controller_learning.tracks.generator")
runtime_module = importlib.import_module("controller_learning.evaluation.final_runtime")
specs_module = importlib.import_module("controller_learning.tracks.specs")
warp = importlib.import_module("warp")

project = config_module.load_project_config(root)
generation = specs_module.generation_spec_from_project(project)
capacity = specs_module.track_capacity_from_project(project)
track = generator_module.pack_track(
    generator_module.generate_track_candidate(42, generation),
    capacity,
)
nvidia_smi = runtime_module.resolve_nvidia_smi_executable()
guard.freeze_nvidia_smi_executable(nvidia_smi)
module._initialize_warp_runtime(SimpleNamespace(warp=warp))
module._enter_post_bind_phase()

guard.enable_test_reads()
for fake_path in (fake_manifest, fake_asset):
    try:
        fake_path.read_bytes()
    except FileNotFoundError:
        pass
    else:
        raise RuntimeError("audit-only fake Track read unexpectedly returned bytes")
guard.forbid_all_track_reads()

environment = environment_module.CarRacingEnv(
    project_config=project,
    level_id=1,
    backend="mjx_warp",
    track=track,
    render_mode=None,
)
try:
    observation, info = environment.reset(seed=123)
    transition = environment.step((0.0, 0.0))
    if not observation or info["track_id"] != 42:
        raise RuntimeError("generated-Track reset evidence differs")
    if not transition[0] or not isinstance(transition[1], float):
        raise RuntimeError("generated-Track step evidence differs")
finally:
    environment.close()

evidence = guard.evidence(test_loaded=True)
if evidence["denied_event_count"] != 0:
    raise RuntimeError("post-seal environment triggered a denied operation")
if evidence["open_event_counts"] != {
    "official_test_asset": 1,
    "official_test_manifest": 1,
}:
    raise RuntimeError("fake asset-read seal evidence differs")
print("post-seal-generated-track-pass", evidence["denied_event_count"])
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
        timeout=60,
    )

    assert completed.stdout.strip().endswith("post-seal-generated-track-pass 0")
