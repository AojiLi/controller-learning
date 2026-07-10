# Controllers

Each trusted Controller plugin is a directory containing `controller.py`, `config.toml`, and
optional helpers or assets. The module must define exactly one concrete `Controller` subclass. The
complete TOML document appears under the read-only `config["controller"]` mapping, while Challenge
configuration remains outside the directory and cannot be overridden by a Controller.

Start with `controllers/template`, then run one episode from the repository root:

```console
pixi run sim -- --controller controllers/template --track-seed 42
```

The template deliberately returns a neutral action and reaches the normal Challenge timeout. It is
an API example, not a driving baseline.
