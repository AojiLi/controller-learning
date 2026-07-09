# Controller Learning — Durable Project Context

## Purpose

Controller Learning is a portfolio-oriented, benchmark-structured, tutorial-presented platform for learning and comparing race-car Controllers. It provides a reproducible Challenge—vehicle, tracks, observations, actions, termination, metrics, and evaluation—then demonstrates it with PID, MPC, and PPO Controllers.

Public one-line description:

> A GPU-parallel race car control benchmark with procedurally generated tracks, pluggable controllers, and reproducible evaluation.

The project is not a collection of disconnected demos and is not a full autonomous-driving stack. The environment and evaluation protocol are the product; example Controllers prove that the product is usable and teach different control approaches.

## Repository State and Identity

- Repository name: `controller-learning`.
- Python package: `controller_learning`.
- License: MIT.
- Development visibility: private until the v0.1 release checklist is satisfied, then public.
- Current state is milestone-driven and recorded in `STATUS.md`; architecture and v0.1 scope are
  confirmed.
- Canonical detailed specification: `PROJECT_PLAN.md`.
- Current work direction: `ACTIVE_CONTEXT.md`.
- Human-readable progress: `STATUS.md`.

## Target Users and Success

Primary users are students, control/RL practitioners, and reviewers evaluating the project as an engineering portfolio. A user should be able to:

1. Install the project with Pixi on Linux.
2. run a fixed-track tutorial environment;
3. add a Controller through a small documented plugin interface;
4. train with hundreds or thousands of GPU worlds;
5. evaluate PID, MPC, PPO, or a new Controller under the same public protocol; and
6. reproduce published metrics and replays from versioned configs, tracks, seeds, dependencies, and manifests.

v0.1 is complete only when the full definition in `PROJECT_PLAN.md` section 21 is met. The most important technical proof is a stable 1024-world MJX-Warp run for 10,000 environment steps with independent random tracks and autoreset.

## v0.1 Stack

- Python 3.11.
- Pixi for dependency management, tasks, and locking.
- MuJoCo MJCF for the physical four-wheel vehicle and CPU reference/debugging.
- MJX-Warp for formal Linux/NVIDIA GPU simulation.
- JAX for batched track, Challenge, reward, termination, and reset computations.
- Gymnasium for `ControllerLearning/CarRacing-v0`, `CarRacingEnv`, and `VecCarRacingEnv` interfaces.
- CasADi + IPOPT for the example MPC.
- PyTorch with a CleanRL-style loop for PPO.
- Ruff and pytest for formatting, linting, and tests.
- GitHub Actions for CPU-only CI; local NVIDIA hardware for versioned GPU reports.

v0.1 supports Linux x86-64. macOS, native Windows, and WSL2 are future targets and must not be advertised as supported before testing.

## System Architecture

The system has five responsibility boundaries:

1. **Physics** — four-wheel vehicle, actuators, wheel-ground contact, state transition.
2. **Track** — procedural closed-loop geometry, validation, fixed-capacity arrays, pools, benchmark versions.
3. **Challenge** — observation, action, progress, checkpoints, reward, termination, reset, Level definitions.
4. **Controller** — trusted plugins and PID/MPC/PPO examples using only public interfaces.
5. **Evaluation** — fixed protocol, metrics, manifests, reports, and 2D replay.

The intended dependency direction is Physics + Track -> Challenge -> Gymnasium/Controller Runner -> Evaluation. Controllers do not own or reach through the Challenge.

The reference project's useful pattern is its Challenge layering, not its drone domain or simulator. Controller Learning intentionally improves the pattern by training PPO directly against the official vector environment and by exposing a write-only `DebugDraw` rather than simulator internals.

## Durable Invariants

### Physics and GPU

- The default simulation truth is one rigid 6-DoF chassis with four physical wheels, four wheel rotation joints, and two front steering joints.
- Controllers may use a simplified kinematic or bicycle prediction model internally.
- Formal v0.1 training and evaluation use MJX-Warp; CPU MuJoCo is for development and bounded consistency tests.
- Native leading-dimension GPU batching and independent masked autoreset are hard requirements.
- The M2 go/no-go precedes Track/Controller/RL expansion. If contact tuning and vehicle simplification fail, the allowed fallback is pure-JAX planar four-wheel tire-force dynamics, not CPU multiprocessing or a bicycle-model simulation truth.

### Track and Benchmark

- Physics runs on a uniform plane; track truth is fixed-capacity JAX geometry, not a random physical road mesh.
- Level 0 uses a fixed track. Level 1 is the formal v0.1 benchmark and randomizes only closed-loop track geometry.
- Track generation uses deterministic seeds and a generator version, geometry validation, and conservative four-wheel driveability validation.
- Training uses a large pre-generated pool; validation and test geometry are fixed, public, versioned, and disjoint.
- Published benchmark geometry and protocol are immutable within a benchmark version.

### Environment and API

- Coordinates and units are SI: world/body +x forward, +y left, yaw counter-clockwise, positive steering left.
- Action is `[steering_angle_rad, longitudinal_acceleration_mps2]` and passes through one standardized actuator layer.
- Observation includes vehicle state, progress, and full fixed-capacity track geometry. It does not directly expose lateral error, heading error, target speed, nearest centerline index, future state, or simulator objects.
- Finite out-of-range actions are clipped and counted. Invalid shape/dtype conversion, NaN, or Inf ends the episode as `invalid_action`.
- `VecCarRacingEnv` uses a leading `num_envs` dimension and Gymnasium NEXT_STEP masked autoreset.
- Single-environment and batch-size-one behavior must be tested for consistency.

### Controller Boundary

- A Controller is a trusted directory plugin containing `controller.py`, `config.toml`, and optional helpers/assets.
- The Runner creates a fresh Controller instance per episode through `Controller(obs, info, config)`.
- Controllers receive only public observations, restricted info, read-only public config, and write-only debug drawing.
- Challenge and Controller configs are separate. Controllers cannot change Level rules, test tracks, actuator limits, or evaluation protocol.
- Environment and Controller seeds are independently and deterministically derived.
- PPO training consumes batched arrays directly; it does not instantiate one Python Controller per world or use a separate simplified environment.

### Evaluation

- An episode starts from rest and succeeds after ordered checkpoints and one completed lap.
- Official ordering is success rate descending, then mean successful lap time ascending. Reward is never the ranking score.
- Formal evaluation also records error, action saturation/smoothness, Controller timing/deadline misses, per-track results, and failure causes.
- Formal multi-track runs are headless. Selected episodes are replayed afterward with the 2D renderer.
- Every published run records code revision, benchmark version, lock/dependencies, hardware/software, configs, seeds, and track IDs.

## Scope Boundaries

v0.1 includes Level 0/1, a physical four-wheel car, GPU vector simulation, procedural tracks, the Controller platform, PID/MPC/PPO examples, evaluation, reports, 2D replay, Linux Pixi setup, CPU CI, and a local GPU benchmark report.

v0.1 excludes Level 2/3, MPCC, perception/SLAM, multi-car racing, ROS/real vehicles, sim-to-real, detailed suspension/Pacejka/aerodynamics, full 3D random roads, online untrusted submissions, multiple general-purpose physics backends, Docker/devcontainers, and formal macOS/Windows/WSL2 support.

## Planned Repository Map

- `controller_learning/physics/`: MJCF adapters, MJX-Warp path, CPU reference, allowed fallback.
- `controller_learning/tracks/`: track types, generation, geometry, validation, pools, benchmark assets.
- `controller_learning/envs/`: Race Core, Gymnasium single/vector environments, observation/reward/termination.
- `controller_learning/control/`: Controller base, loader, and write-only debug drawing.
- `controller_learning/evaluation/`: evaluator, metrics, manifests, and reports.
- `controller_learning/visualization/`: 2D rendering and replay.
- `controller_learning/assets/vehicle/`: four-wheel MJCF and necessary meshes.
- `controllers/`: template, PID, MPC, and PPO plugins.
- `configs/`: Level, vehicle, and benchmark TOML files.
- `scripts/`: simulation, evaluation, track, GPU benchmark, PPO training, and replay entry points.
- `tests/unit/`, `tests/integration/`, `tests/gpu/`: verification layers.
- `benchmarks/`: versioned benchmark evidence, including the v0.1 GPU report.
- `docs/`: English tutorials and API documentation.
- `reference/lsy_drone_racing/`: local read-only design reference; never committed.

Directories above are planned and may not exist until their milestone is implemented.

## Milestone Order

- **M0** — repository skeleton, private Git, MIT, ignore rules, Pixi default/gpu environments, package/tests/Ruff/CPU CI, schemas/config loading.
- **M1** — stable CPU MuJoCo four-wheel vehicle and physics timestep measurements.
- **M2** — MJX-Warp 1/64/256/1024 GPU go/no-go and versioned evidence.
- **M3** — batched Track representation/generation/validation and Race Core.
- **M4** — Gymnasium environments, Controller platform, config boundaries, `DebugDraw`, simulation CLI.
- **M5** — Level 0/1, training pool, fixed validation/test geometry, benchmark manifest.
- **M6** — PID and CasADi/IPOPT MPC with tutorials and timing evidence.
- **M7** — official-environment PPO training, logs, checkpoint Controller.
- **M8** — evaluation artifacts, English public documentation, cleanup, and public release.

Do not bypass a milestone stop condition simply to add visible features. Resolve the blocked foundation or explicitly revise the architecture first.

## Planned Commands

These commands are part of the confirmed interface but do not exist until M0 implements the relevant Pixi tasks:

```bash
pixi install
pixi run tests
pixi run sim

pixi install -e gpu
pixi run -e gpu gpu-tests
pixi run -e gpu benchmark-gpu
pixi run -e gpu train-ppo
```

CPU CI is expected to run a locked Pixi install, Ruff format/lint, unit tests, track and Controller checks, CPU model load/short rollout, and docs build. GPU verification remains local for v0.1 and produces `benchmarks/v0.1/gpu_report.json`.

## Experimental Decisions

The exact physics timestep, solver/contact capacities, actuator mapping, track array capacity/resolution, track scale bounds, stable world count, PPO hyperparameters, MPC horizon/weights, and CPU/GPU consistency tolerance must be selected from M1–M3 measurements. They are intentionally not durable constants yet.

## External Reference

- Upstream inspiration: <https://github.com/learnsyslab/lsy_drone_racing>
- Local study copy: `reference/lsy_drone_racing/`

The local copy is evidence and inspiration only. It is not a vendored dependency and must not enter the public Git history.
