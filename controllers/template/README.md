# Template Controller

This directory is the smallest valid Controller Learning plugin. `controller.py` defines exactly
one concrete `Controller` subclass, and `config.toml` contains only Controller-owned settings.
The complete TOML document is exposed as the recursively read-only `config["controller"]`
mapping; it cannot override Challenge-owned configuration.

The template intentionally returns zero steering and zero acceleration. It demonstrates the API;
it is not a driving baseline. A larger plugin may add helper modules or assets in this directory.
Use package-relative imports such as `from .helpers import Planner` so independently loaded plugins
cannot collide in Python's global module namespace.
