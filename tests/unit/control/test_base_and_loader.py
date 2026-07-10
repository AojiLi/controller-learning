"""Tests for the trusted directory Controller contract."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path
from types import MappingProxyType

import numpy as np
import pytest

from controller_learning.control import (
    Controller,
    ControllerLoadError,
    load_controller,
    load_controller_config,
)

PROJECT_ROOT = Path(__file__).parents[3]
TEMPLATE_DIRECTORY = PROJECT_ROOT / "controllers" / "template"


def _write_plugin(
    directory: Path, source: str, config: str = "[controller]\nname = 'test'\n"
) -> Path:
    directory.mkdir()
    (directory / "controller.py").write_text(source, encoding="utf-8")
    (directory / "config.toml").write_text(config, encoding="utf-8")
    return directory


def test_template_loads_as_class_and_returns_float32_action() -> None:
    controller_class = load_controller(TEMPLATE_DIRECTORY)
    controller_config = load_controller_config(TEMPLATE_DIRECTORY)

    assert issubclass(controller_class, Controller)
    assert tuple(controller_config) == ("name", "description")
    assert controller_config["name"] == "template"
    controller = controller_class({}, {"controller_seed": 7}, controller_config)
    action = controller.compute_control({})

    assert action.shape == (2,)
    assert action.dtype == np.float32
    np.testing.assert_array_equal(action, np.zeros(2, dtype=np.float32))


@pytest.mark.parametrize("missing_name", ["controller.py", "config.toml"])
def test_loader_requires_both_directory_files(tmp_path: Path, missing_name: str) -> None:
    plugin = tmp_path / "plugin"
    plugin.mkdir()
    for name in {"controller.py", "config.toml"} - {missing_name}:
        (plugin / name).write_text("", encoding="utf-8")

    with pytest.raises(ControllerLoadError, match=missing_name):
        load_controller(plugin)


def test_loader_rejects_a_non_directory_path(tmp_path: Path) -> None:
    plugin = tmp_path / "controller.py"
    plugin.write_text("", encoding="utf-8")

    with pytest.raises(ControllerLoadError, match="must be a directory"):
        load_controller(plugin)


def test_loader_rejects_malformed_toml_before_import(tmp_path: Path) -> None:
    plugin = _write_plugin(
        tmp_path / "plugin",
        "raise RuntimeError('module should not be imported')\n",
        "broken = [\n",
    )

    with pytest.raises(ControllerLoadError, match="Invalid TOML") as error:
        load_controller(plugin)

    assert "module should not be imported" not in str(error.value)


def test_loader_rejects_zero_controller_classes(tmp_path: Path) -> None:
    plugin = _write_plugin(tmp_path / "plugin", "VALUE = 1\n")

    with pytest.raises(ControllerLoadError, match="exactly one Controller subclass"):
        load_controller(plugin)


def test_loader_rejects_multiple_controller_classes(tmp_path: Path) -> None:
    plugin = _write_plugin(
        tmp_path / "plugin",
        """
from controller_learning.control import Controller
class First(Controller):
    def compute_control(self, obs, info=None):
        return None
class Second(Controller):
    def compute_control(self, obs, info=None):
        return None
""",
    )

    with pytest.raises(ControllerLoadError, match="multiple Controller subclasses"):
        load_controller(plugin)


def test_loader_rejects_an_abstract_controller_class(tmp_path: Path) -> None:
    plugin = _write_plugin(
        tmp_path / "plugin",
        """
from controller_learning.control import Controller
class Incomplete(Controller):
    pass
""",
    )

    with pytest.raises(ControllerLoadError, match="is abstract"):
        load_controller(plugin)


def test_imported_controller_subclass_is_not_counted(tmp_path: Path) -> None:
    plugin = _write_plugin(
        tmp_path / "plugin",
        """
from controller_learning.control import Controller
from .helper import ImportedController
class LocalController(Controller):
    def compute_control(self, obs, info=None):
        return ImportedController.VALUE
""",
    )
    (plugin / "helper.py").write_text(
        """
import numpy as np
from controller_learning.control import Controller
class ImportedController(Controller):
    VALUE = np.array([0.0, 0.0], dtype=np.float32)
    def compute_control(self, obs, info=None):
        return self.VALUE
""",
        encoding="utf-8",
    )

    controller_class = load_controller(plugin)

    assert controller_class.__name__ == "LocalController"
    assert controller_class({}, {}, {}).compute_control({}).dtype == np.float32


def test_relative_helpers_are_isolated_between_controller_directories(tmp_path: Path) -> None:
    source = """
from controller_learning.control import Controller
from .helper import ACTION
class LocalController(Controller):
    def compute_control(self, obs, info=None):
        return ACTION.copy()
"""
    first = _write_plugin(tmp_path / "first", source)
    second = _write_plugin(tmp_path / "second", source)
    (first / "helper.py").write_text(
        "import numpy as np\nACTION = np.array([1.0, 2.0], dtype=np.float32)\n",
        encoding="utf-8",
    )
    (second / "helper.py").write_text(
        "import numpy as np\nACTION = np.array([3.0, 4.0], dtype=np.float32)\n",
        encoding="utf-8",
    )

    unrelated_helper = sys.modules.get("helper")
    first_class = load_controller(first)
    second_class = load_controller(second)

    assert first_class.__module__ != second_class.__module__
    assert sys.modules.get("helper") is unrelated_helper
    np.testing.assert_array_equal(first_class({}, {}, {}).compute_control({}), [1.0, 2.0])
    np.testing.assert_array_equal(second_class({}, {}, {}).compute_control({}), [3.0, 4.0])


def test_module_name_is_deterministic_for_the_same_directory(tmp_path: Path) -> None:
    plugin = _write_plugin(
        tmp_path / "plugin",
        "from controller_learning.control import Controller\n"
        "class Valid(Controller):\n"
        "    def compute_control(self, obs, info=None): return None\n",
    )

    first_name = load_controller(plugin).__module__
    second_name = load_controller(plugin).__module__

    assert first_name == second_name


def test_one_controller_class_alias_is_not_treated_as_two_classes(tmp_path: Path) -> None:
    plugin = _write_plugin(
        tmp_path / "plugin",
        "from controller_learning.control import Controller\n"
        "class Valid(Controller):\n"
        "    def compute_control(self, obs, info=None): return None\n"
        "Alias = Valid\n",
    )

    assert load_controller(plugin).__name__ == "Valid"


def test_controller_config_is_recursively_immutable(tmp_path: Path) -> None:
    plugin = _write_plugin(
        tmp_path / "plugin",
        "from controller_learning.control import Controller\n"
        "class Valid(Controller):\n"
        "    def compute_control(self, obs, info=None): return None\n",
        """
name = "nested"
gains = [1.0, 2.0]
[[stages]]
name = "first"
[stages.limits]
values = [3, 4]
""",
    )

    config = load_controller_config(plugin)

    assert isinstance(config, MappingProxyType)
    assert config["gains"] == (1.0, 2.0)
    assert isinstance(config["stages"], tuple)
    stage = config["stages"][0]
    assert isinstance(stage, MappingProxyType)
    assert stage["limits"]["values"] == (3, 4)
    with pytest.raises(TypeError):
        config["name"] = "changed"  # type: ignore[index]
    with pytest.raises(TypeError):
        stage["name"] = "changed"


def test_fresh_instances_do_not_share_episode_state(tmp_path: Path) -> None:
    plugin = _write_plugin(
        tmp_path / "plugin",
        """
import numpy as np
from controller_learning.control import Controller
class Stateful(Controller):
    def __init__(self, obs, info, config):
        self.calls = 0
    def compute_control(self, obs, info=None):
        self.calls += 1
        return np.array([self.calls, 0], dtype=np.float32)
""",
    )
    controller_class = load_controller(plugin)

    first = controller_class({}, {}, {})
    second = controller_class({}, {}, {})

    assert first is not second
    assert first.compute_control({})[0] == 1.0
    assert first.compute_control({})[0] == 2.0
    assert second.compute_control({})[0] == 1.0


def test_optional_callbacks_default_to_no_op() -> None:
    controller = load_controller(TEMPLATE_DIRECTORY)({}, {}, {})
    action = controller.compute_control({})

    assert controller.step_callback(action, {}, 0.0, False, False, {}) is None
    assert controller.episode_callback() is None


def test_plugin_can_override_all_callbacks(tmp_path: Path) -> None:
    plugin = _write_plugin(
        tmp_path / "plugin",
        """
import numpy as np
from controller_learning.control import Controller
class CallbackController(Controller):
    def __init__(self, obs, info, config):
        self.events = []
    def compute_control(self, obs, info=None):
        return np.zeros(2, dtype=np.float32)
    def step_callback(self, action, obs, reward, terminated, truncated, info):
        self.events.append(("step", reward))
    def episode_callback(self):
        self.events.append(("episode",))
    def render_callback(self, debug_draw):
        debug_draw.text((0, 0), "target")
""",
    )
    controller = load_controller(plugin)({}, {}, {})

    controller.step_callback(controller.compute_control({}), {}, 1.5, False, False, {})
    controller.episode_callback()

    assert controller.events == [("step", 1.5), ("episode",)]


def test_control_package_does_not_import_environment_or_simulator_stacks() -> None:
    command = (
        "import sys; import controller_learning.control; "
        "assert all(name not in sys.modules for name in "
        "('controller_learning.envs', 'controller_learning.physics', 'jax', 'mujoco', 'warp'))"
    )

    subprocess.run([sys.executable, "-c", command], check=True)
