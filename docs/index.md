# Controller Learning

**A GPU-parallel race-car control benchmark for learning by building and comparing Controllers.**

Controller Learning keeps one physical four-wheel car, Challenge, Track distribution, observation,
action, and evaluation contract fixed while the Controller changes. PID, nonlinear MPC, and PPO
are examples; the reusable environment and plugin boundary are the product.

![PID, MPC, and PPO on accepted benchmark 0.1 row 0](https://raw.githubusercontent.com/AojiLi/controller-learning/main/benchmarks/v0.1/m8_test_row_000_comparison.png)

## Start here

```bash
git clone https://github.com/AojiLi/controller-learning.git
cd controller-learning
pixi install
pixi run sim -- --controller controllers/pid --level-id 0 --render
```

The [Controller workflow](getting-started.md) continues from a copied template through simulation,
informal Level 0/Validation evaluation, exact same-rollout trajectory capture, and offline replay.
The evaluator deliberately has no Test option.

## Accepted benchmark 0.1 result

| Rank | Controller | Success | Mean successful lap | Mean speed | Lateral RMS |
| ---: | --- | ---: | ---: | ---: | ---: |
| 1 | PID | 20/20 | 88.085 s | 4.974 m/s | 0.0211 m |
| 2 | MPC | 20/20 | 102.563 s | 4.273 m/s | 0.0381 m |
| 3 | PPO | 19/20 | 23.913 s | 18.324 m/s | 0.2205 m |

Success rate is ranked before successful lap time. The
[result interpretation](analysis.md) shows the speed and lateral-error distributions and explains
what this particular comparison does—and does not—establish. The [Evaluation Protocol](evaluation.md)
contains the fixed execution, ranking, metrics, attempt lineage, and canonical artifacts.

## Design principles

- One physical four-wheel simulation truth for every Controller.
- One official Challenge for classical control, reinforcement learning, and evaluation.
- Native GPU batching instead of one simulator per CPU process.
- A narrow Controller interface with no simulator-internal shortcuts.
- Public, deterministic benchmark geometry, seeds, configuration, manifests, metrics, and replays.

## Documentation map

- [Controller workflow](getting-started.md): install, author, simulate, evaluate, and replay.
- [CPU Vehicle](vehicle.md) and [MJX-Warp GPU Vehicle](gpu.md): plant and backend evidence.
- [Tracks and Race Core](tracks.md): procedural geometry, splits, assets, and TrackPool.
- [Gymnasium and Controller Platform](environment.md): public runtime contracts.
- [Classical Controllers](controllers.md): observation-only PID and MPC design.
- [PPO Training and Export](ppo.md): official vector-environment learning path.
- [Evaluation Protocol](evaluation.md): immutable benchmark `0.1` comparison.
- [Reproducibility](reproducibility.md): identity and numerical reproducibility boundaries.
- [Stability Policy](stability.md): supported v0.1.x contracts and version evolution.
- [Development](development.md): maintainer commands and milestone workflows.

Claims are limited to the platforms and workloads recorded in the linked evidence. Linux x86-64 is
the only tested v0.1.x platform; macOS, native Windows, and WSL2 are not claimed as supported.
