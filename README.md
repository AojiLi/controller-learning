# Controller Learning

**A GPU-parallel race car control benchmark with procedurally generated tracks,
pluggable controllers, and reproducible evaluation.**

Controller Learning is a benchmark and teaching platform for developing and comparing race-car
controllers under one environment, vehicle, task, and evaluation protocol. PID, MPC, and PPO are
provided as examples; the reusable Challenge and Controller interface are the core product.

> **Project status:** M1 is complete: the CPU four-wheel vehicle and measured timestep report are
> available. M2 MJX-Warp GPU validation is active. The racing benchmark, Controllers, and GPU
> performance results are not available yet.

The reviewed CPU evidence is available in
[the M1 machine-readable report](benchmarks/v0.1/m1_cpu_report.json).

## Why This Project Exists

Control approaches are difficult to compare when each example uses a different vehicle model,
track, observation, action, or success definition. This project is designed to make those choices
explicit and reproducible:

- a physical four-wheel race car as the simulation truth;
- native GPU-batched simulation for reinforcement learning;
- fixed and procedurally generated closed-loop tracks;
- a small directory-based Controller plugin interface;
- the same official environment for classical control, training, and evaluation; and
- public benchmark tracks, seeds, manifests, metrics, and replays.

## Planned v0.1 Stack

- MuJoCo MJCF and MJX-Warp
- JAX and Gymnasium
- CasADi/IPOPT for MPC
- PyTorch for PPO
- Pixi on Linux with Python 3.11

GPU scale and Controller success rates will only be documented after the corresponding milestone
benchmarks pass.

## Development Setup

Pixi is the only supported environment workflow for v0.1.

```bash
pixi install
pixi run tests
pixi run lint
pixi run docs
```

The NVIDIA environment is installed separately so CPU development and CI do not resolve or install
CUDA/PyTorch dependencies:

```bash
pixi install -e gpu
pixi run -e gpu gpu-check
```

These commands are verified as part of M0. Linux x86-64 with glibc 2.28 or newer is the only
supported v0.1 platform; macOS, native Windows, and WSL2 are future work.

## Architecture

The repository separates five responsibilities:

1. **Physics** advances the four-wheel vehicle.
2. **Track** owns deterministic geometry, validation, and benchmark pools.
3. **Challenge** defines observations, actions, progress, reward, reset, and termination.
4. **Controller** contains trusted plugins that only use the public interface.
5. **Evaluation** produces reproducible metrics, manifests, plots, and replays.

PPO will train directly against the official `VecCarRacingEnv`; the project will not maintain a
second simplified training environment.

## Roadmap

The implementation follows strict milestone gates:

- M0: repository, Pixi, package, tests, CI, and configuration schemas
- M1: stable CPU MuJoCo four-wheel car
- M2: MJX-Warp 1/64/256/1024-world GPU go/no-go
- M3: batched tracks and Race Core
- M4: Gymnasium environments and Controller platform
- M5: Level 0/1 and versioned track pools
- M6: PID and MPC
- M7: PPO on the official vector environment
- M8: evaluation, documentation, and public v0.1 release

The detailed confirmed design is recorded in [PROJECT_PLAN.md](PROJECT_PLAN.md).

## Inspiration

The Challenge-layer design is inspired by
[learnsyslab/lsy_drone_racing](https://github.com/learnsyslab/lsy_drone_racing). This repository is
an independent race-car implementation and does not vendor the reference source.

## License

Controller Learning is released under the [MIT License](LICENSE).
