# Stability and Versioning

Controller Learning versions the software and the benchmark separately. This policy defines what a
Controller author can rely on during the v0.1.x software series and how later changes will be
signaled.

## Benchmark versions are immutable

Benchmark `0.1` binds the vehicle and Level configuration, Track manifests and packed geometry,
split roles, seed derivation, public observation/action/info contracts, termination rules, ranking,
metric definitions, Controller identities, and accepted M8 artifacts.

A software maintenance release may improve documentation, development tooling, or reporting around
that evidence. It cannot silently replace the accepted result, tune a frozen Controller from Test,
change a benchmark `0.1` Track, or redefine a reported metric. A changed performance protocol must
use a new benchmark version and retain the old evidence.

## Supported v0.1.x Controller contract

The following documented surfaces are stable throughout v0.1.x:

- a trusted plugin is a directory containing `controller.py` and `config.toml`;
- `controller.py` defines exactly one concrete subclass of `controller_learning.control.Controller`;
- the constructor and lifecycle methods documented in [Gymnasium and Controller Platform](environment.md)
  retain their names, argument roles, and callback order;
- the public action remains float32 `[steering_angle_rad, longitudinal_acceleration_mps2]` with the
  configured physical bounds;
- the observation field names, meanings, dtypes, single-world shapes, and fixed Track capacity
  remain those documented for `ControllerLearning/CarRacing-v0`;
- reset/step info retains its seven-field whitelist and separates environment and Controller seeds;
- Challenge-owned configuration remains immutable and plugin TOML remains under
  `config["controller"]`; and
- `DebugDraw` remains write-only and provides no simulator read path.

Compatible clarifications, stricter rejection of already-invalid inputs, bug fixes that restore the
documented behavior, and additive optional tooling may ship in v0.1.x. They must include tests and
must not change accepted benchmark `0.1` evidence.

## Command and artifact status

The supported user entry points are Pixi tasks documented in this site. `sim`,
`evaluate-controller`, and `replay` are the normal Controller workflow. The development evaluation
schemas identify informal local evidence and are not formal benchmark schemas.

The public canonical M8 CSV, JSON, NPZ, PNG, and trajectory schemas are immutable as benchmark
`0.1` artifacts. Their Python implementation modules are not thereby promised as a general stable
library API.

## Internal APIs

Unless a symbol is explicitly documented on this page or the Controller-contract page, module
paths under `controller_learning.physics`, `tracks`, `envs`, `evaluation`, `rl`, and
`visualization` are internal implementation details. Tests and example Controllers may use some of
them to preserve the frozen repository, but external code should not assume their import paths are
stable across v0.2.

The inference-only v0.1 PPO plugin is hash-frozen and intentionally retains its historical imports.
A future public learned-controller runtime surface will require a versioned compatibility plan; it
will not be retroactively substituted into the accepted plugin.

## Moving to v0.2

A v0.2 proposal that changes a public Controller surface must:

1. name every affected field, method, command, or schema;
2. explain why an additive v0.1.x change is insufficient;
3. provide an explicit migration path and compatibility tests;
4. preserve benchmark `0.1` artifacts and reproducibility instructions; and
5. introduce a new benchmark version when evaluation semantics or comparable inputs change.

Current v0.1 non-goals—Level 2/3, MPCC, perception, multi-car racing, real vehicles, sim-to-real,
untrusted online execution, and broad physics-backend abstraction—do not become implied promises
through this policy.

## Platform claims

Linux x86-64 is the only tested v0.1.x platform. GPU claims additionally require the recorded
NVIDIA/CUDA environment. macOS, native Windows, and WSL2 support will be claimed only after their
installation, test, and numerical evidence exists. Package compatibility and measured performance
are separate claims.
