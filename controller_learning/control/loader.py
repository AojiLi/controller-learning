"""Load one trusted Controller class and its immutable directory configuration."""

from __future__ import annotations

import hashlib
import importlib.machinery
import importlib.util
import inspect
import sys
import tomllib
from collections.abc import Mapping
from pathlib import Path
from threading import RLock
from types import MappingProxyType, ModuleType
from typing import Any, TypeAlias

from controller_learning.control.base import Controller

ControllerConfig: TypeAlias = Mapping[str, Any]

_LOAD_LOCK = RLock()


class ControllerLoadError(RuntimeError):
    """A Controller directory cannot be loaded through the public plugin contract."""


def _required_paths(directory: str | Path) -> tuple[Path, Path, Path]:
    path = Path(directory).expanduser()
    if not path.exists():
        raise ControllerLoadError(f"Controller directory does not exist: {path}")
    if not path.is_dir():
        raise ControllerLoadError(f"Controller path must be a directory: {path}")

    resolved = path.resolve()
    controller_path = resolved / "controller.py"
    config_path = resolved / "config.toml"
    missing = [item.name for item in (controller_path, config_path) if not item.is_file()]
    if missing:
        names = ", ".join(missing)
        raise ControllerLoadError(f"Controller directory is missing required file(s): {names}")
    return resolved, controller_path, config_path


def _freeze_config(value: Any) -> Any:
    if isinstance(value, dict):
        return MappingProxyType({key: _freeze_config(item) for key, item in value.items()})
    if isinstance(value, list):
        return tuple(_freeze_config(item) for item in value)
    return value


def _parse_config(config_path: Path) -> ControllerConfig:
    try:
        with config_path.open("rb") as file:
            parsed = tomllib.load(file)
    except tomllib.TOMLDecodeError as error:
        raise ControllerLoadError(
            f"Invalid TOML in Controller config {config_path}: {error}"
        ) from error
    except OSError as error:
        raise ControllerLoadError(
            f"Cannot read Controller config {config_path}: {error}"
        ) from error
    frozen = _freeze_config(parsed)
    if not isinstance(frozen, Mapping):
        raise ControllerLoadError(f"Controller config root must be a TOML table: {config_path}")
    return frozen


def load_controller_config(directory: str | Path) -> ControllerConfig:
    """Parse arbitrary ``config.toml`` values into recursively read-only containers."""
    _, _, config_path = _required_paths(directory)
    return _parse_config(config_path)


def _module_names(directory: Path) -> tuple[str, str]:
    digest = hashlib.sha256(str(directory).encode("utf-8")).hexdigest()[:20]
    package_name = f"_controller_learning_plugin_{digest}"
    return package_name, f"{package_name}.controller"


def _purge_package(package_name: str) -> None:
    prefix = f"{package_name}."
    for name in tuple(sys.modules):
        if name == package_name or name.startswith(prefix):
            del sys.modules[name]


def _plugin_package(package_name: str, directory: Path) -> ModuleType:
    package = ModuleType(package_name)
    package.__file__ = str(directory)
    package.__package__ = package_name
    package.__path__ = [str(directory)]  # type: ignore[attr-defined]
    spec = importlib.machinery.ModuleSpec(package_name, loader=None, is_package=True)
    spec.submodule_search_locations = [str(directory)]
    package.__spec__ = spec
    return package


def _controller_classes(module: ModuleType) -> list[type[Controller]]:
    classes: list[type[Controller]] = []
    seen: set[type[Controller]] = set()
    for value in vars(module).values():
        if not inspect.isclass(value) or value is Controller:
            continue
        if not issubclass(value, Controller):
            continue
        if value.__module__ == module.__name__ and value not in seen:
            classes.append(value)
            seen.add(value)
    return classes


def _load_controller_module(directory: Path, controller_path: Path) -> ModuleType:
    package_name, module_name = _module_names(directory)
    _purge_package(package_name)
    sys.modules[package_name] = _plugin_package(package_name, directory)

    spec = importlib.util.spec_from_file_location(module_name, controller_path)
    if spec is None or spec.loader is None:
        _purge_package(package_name)
        raise ControllerLoadError(f"Cannot create an import specification for {controller_path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    try:
        spec.loader.exec_module(module)
    except Exception as error:
        _purge_package(package_name)
        raise ControllerLoadError(
            f"Failed to import Controller module {controller_path}: {type(error).__name__}: {error}"
        ) from error
    return module


def load_controller(directory: str | Path) -> type[Controller]:
    """Load and return the one concrete Controller class defined by a plugin directory.

    ``config.toml`` is parsed as part of loading so a malformed or unreadable required config can
    never accompany an otherwise loadable class. The runner remains responsible for loading the
    immutable config and constructing a fresh class instance for each episode.
    """
    resolved, controller_path, config_path = _required_paths(directory)
    _parse_config(config_path)

    with _LOAD_LOCK:
        module = _load_controller_module(resolved, controller_path)
        classes = _controller_classes(module)
        if not classes:
            _purge_package(_module_names(resolved)[0])
            raise ControllerLoadError(
                f"Controller module must define exactly one Controller subclass: {controller_path}"
            )
        if len(classes) > 1:
            names = ", ".join(sorted(controller.__name__ for controller in classes))
            _purge_package(_module_names(resolved)[0])
            raise ControllerLoadError(
                f"Controller module defines multiple Controller subclasses ({names}): "
                f"{controller_path}"
            )

        controller_class = classes[0]
        if inspect.isabstract(controller_class):
            _purge_package(_module_names(resolved)[0])
            raise ControllerLoadError(
                f"Controller class {controller_class.__name__} is abstract: {controller_path}"
            )
        return controller_class


__all__ = [
    "ControllerConfig",
    "ControllerLoadError",
    "load_controller",
    "load_controller_config",
]
