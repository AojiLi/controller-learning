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

The repository also includes three example Controllers:

| Directory | Approach |
| --- | --- |
| `controllers/pid` | Curvature-aware speed PID with cascaded lateral and heading control |
| `controllers/mpc` | Warm-started, constrained Frenet NMPC built with CasADi and IPOPT |
| `controllers/ppo` | Finalized, Torch-free NumPy actor selected from the M7 PPO run |

Run them on the fixed Level 0 Track with optional public-observation rendering:

```console
pixi run sim -- --controller controllers/pid --level-id 0 --render
pixi run sim -- --controller controllers/mpc --level-id 0 --render
```

See the [Classical Controllers tutorial](../docs/controllers.md) for their shared geometry and
speed planner, Controller lifecycle, PID anti-windup, MPC model and fallback policy, DebugDraw, and
formal timing interpretation.

The PPO plugin follows the same lifecycle and public observation boundary. Training uses the
official GPU-batched `VecCarRacingEnv`; deployment loads a hash-bound 120,968-byte NumPy actor and
does not import PyTorch. Run it from the GPU Pixi environment:

```console
pixi run -e gpu sim -- \
  --controller controllers/ppo \
  --level-id 1 \
  --track-seed 42 \
  --backend mjx_warp \
  --render
```

See the [PPO tutorial](../docs/ppo.md) for Train/Validation separation, NEXT_STEP rollout masks,
DLPack exchange, checkpoint selection, export provenance, formal Validation evidence, and replay.
