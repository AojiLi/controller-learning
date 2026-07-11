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
- Repository visibility: public since the v0.1 release checklist was satisfied on 2026-07-11.
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
- The CPU `CpuVehicle` state is expressed at the rear-axle reference site. One public control step
  is 0.05 seconds and advances an integer number of MuJoCo physics substeps.
- The standardized actuator layer clips and rate-limits steering targets, applies equal positive
  drive torque to all four wheels, and makes negative acceleration a non-reversing brake command.
- M1 selected 0.005 seconds as the largest passing CPU physics timestep, and M2 subsequently
  validated the same timestep in MJX-Warp through 1024 native worlds and 10,000 environment steps.
- The reviewed M2 flat-ground vehicle capacities are 16 global contact entries per world and 64
  constraints per world; observed peak usage was 50% and 37.5%, respectively.
- Controllers may use a simplified kinematic or bicycle prediction model internally.
- Formal v0.1 training and evaluation use MJX-Warp; CPU MuJoCo is for development and bounded consistency tests.
- Native leading-dimension GPU batching and independent masked autoreset are hard requirements.
- The M2 go/no-go passed on the MJX-Warp path. The pure-JAX planar four-wheel fallback was not
  activated; CPU multiprocessing and a bicycle-model simulation truth remain invalid fallbacks.

### Track and Benchmark

- Physics runs on a uniform plane; track truth is fixed-capacity JAX geometry, not a random physical road mesh.
- Level 0 uses a fixed track. Level 1 is the formal v0.1 benchmark and randomizes only closed-loop track geometry.
- Track generation uses deterministic seeds and a generator version, geometry validation, and conservative four-wheel driveability validation.
- The v0.1 Track representation is locked at 1.0 m nominal arc spacing, 640 points including an
  explicit closure point, 15 m checkpoint spacing, and 48 checkpoints. The corresponding 600 m
  theoretical requirements are 601 points and 40 checkpoints.
- Runtime Track arrays occupy 26.640625 MiB for 1,024 worlds and 260.162 MiB for a 10,000-track pool.
- Track projection is topology-local to the prior segment, with four backward and twelve forward
  candidates in v0.1. Race Core owns ordered checkpoint progress, effective boundary, reward,
  timeout, termination priority, and per-world masked reset independently from the physics backend.
- Training uses a large pre-generated pool; validation and test geometry are fixed, public, versioned, and disjoint.
- Benchmark `0.1` fixes one Level 0 Track plus 10,000 Train, 100 Validation, and 20 Test Level 1
  Tracks. Level 1 seed namespaces are `[0, 1_000_000)`, `[1_000_000, 2_000_000)`, and
  `[2_000_000, 3_000_000)` respectively; manifest seeds are strictly increasing and packed geometry
  hashes are globally disjoint.
- Level 0 is the deterministic ellipse with reserved Track seed/ID `UINT32_MAX`. Level 0,
  Validation, and Test NPZ assets ship in the package. The 272,800,000-byte Train NPZ is regenerated
  from its committed seed/hash manifest into `.track-cache/v0.1/` and is never committed.
- Official admission is first-N ascending-seed selection after generation, geometry validation,
  packing, geometry-hash isolation, and conservative four-wheel driveability. Infrastructure-level
  numerical/contact faults invalidate the whole run rather than becoming per-Track rejections.
- Published benchmark geometry and protocol are immutable within a benchmark version.

### Environment and API

- Coordinates and units are SI: world/body +x forward, +y left, yaw counter-clockwise, positive steering left.
- Action is `[steering_angle_rad, longitudinal_acceleration_mps2]` and passes through one standardized actuator layer.
- Observation includes vehicle state, progress, and full fixed-capacity track geometry. It does not directly expose lateral error, heading error, target speed, nearest centerline index, future state, or simulator objects.
- Finite out-of-range actions are clipped and counted. Invalid shape/dtype conversion, NaN, or Inf ends the episode as `invalid_action`.
- `VecCarRacingEnv` uses a leading `num_envs` dimension and Gymnasium NEXT_STEP masked autoreset.
- `VecCarRacingEnv` is the sole Challenge state machine. `CarRacingEnv` is a host/NumPy batch-one
  adapter and requires an explicit reset after termination.
- `VecCarRacingEnv` accepts exactly one fixed Track sequence or one immutable `TrackPool`. Pool
  selection is device-native, deterministic sampling with replacement from SeedSequence domain 2.
  It atomically replaces Track, vehicle, Race Core, observation, and numeric Track ID on NEXT_STEP
  reset without creating a training-only environment.
- Single-environment and batch-size-one behavior is tested for consistency. Warm MJX-Warp active
  and mixed-autoreset steps must pass JAX transfer guards with no host/device transfer.
- Reset and step info use the fixed public whitelist: episode seed, Controller seed, Track ID,
  benchmark version, termination reason, lap-completed flag, and lap time. Neutral terminal values
  keep the vector schema fixed.
- Public `track_id` is uint32 `Track.seed`; its stable identity is
  `(benchmark_version, level_id, track_id)`. A terminal transition reports the old Track, and only
  the following NEXT_STEP reset reports the newly selected Track.
- The CPU backend is restricted to one world and remains a development/reference path. Formal
  vector training and evaluation use explicit `backend="mjx_warp"`; there is no silent fallback.

### Controller Boundary

- A Controller is a trusted directory plugin containing `controller.py`, `config.toml`, and optional helpers/assets.
- The Runner creates a fresh Controller instance per episode through `Controller(obs, info, config)`.
- Controllers receive only public observations, restricted info, read-only public config, and write-only debug drawing.
- Challenge and Controller configs are separate. Controllers cannot change Level rules, test tracks, actuator limits, or evaluation protocol.
- Environment and Controller seeds are independently and deterministically derived.
- The Runner derives Challenge-owned public configuration from the actual unwrapped environment,
  filters info to the canonical whitelist, drains `DebugDraw` per rendered frame, and never passes
  an Environment or renderer reference to a Controller.
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

## Commands

The following commands exist now:

```bash
pixi install
pixi run tests
pixi run ci
pixi run benchmark-cpu-vehicle
pixi run benchmark-track-capacity
pixi run sim
pixi run verify-track-assets
pixi run materialize-track-pool
pixi run view-cpu-vehicle -- --scenario demo --duration 12

pixi install -e gpu
pixi run -e gpu gpu-tests
pixi run -e gpu benchmark-gpu
pixi run -e gpu benchmark-racing-env
pixi run -e gpu validate-track-driveability
pixi run -e gpu build-track-assets
pixi run -e gpu benchmark-track-pool
pixi run -e gpu benchmark-controllers
pixi run -e gpu train-ppo -- --run-id my-ppo-run
pixi run -e gpu benchmark-m7-ppo
pixi run -e gpu export-m7-ppo-controller
pixi run -e gpu benchmark-m7-ppo-controller
pixi run -e gpu benchmark-m8-controllers
```

The four M7 commands above implement formal Train-only optimization, frozen Validation selection,
hash-bound inference-only Controller export, and ordinary Controller evaluation/replay respectively.
The M8 command is a release-maintainer workflow guarded by a clean revision, immutable Controller
snapshots, Test-only process access, durable episode and post-close evidence, semantic validation
of exactly 24 outputs, and no automatic retry after Test binding. Attempt 001 loaded Test but
stopped during Environment creation with zero journal rows and no performance observation. The
owner-authorized attempt 002 is its sole replacement, pre-initializes Warp before Test binding,
requires exact read-only predecessor lineage, and forbids a third attempt. It is not routine
development or tuning tooling.
CPU CI currently checks formatting, lint, CPU tests, installed wheel contents, strict docs, GitHub
Actions syntax, and package metadata. GPU verification remains local for v0.1 and produces
versioned evidence under `benchmarks/v0.1/`.

Reviewed M1 CPU evidence is stored at `benchmarks/v0.1/m1_cpu_report.json`. The report records the
source revision, dirty-worktree gate, dependency lock, model/config/protocol hashes, runtime, all
candidate results, and the selected M2 candidate.

Reviewed M2 GPU evidence is stored at `benchmarks/v0.1/gpu_report.json`. It records fresh-process
1/64/256/1024-world results, the 1024-world × 10,000-step endurance run, CPU/GPU numerical and
contact agreement, capacity headroom, physical gates, compile/throughput/VRAM measurements, native
warnings, masked resets, runtime versions, and stable pre/post Git and source hashes. The persisted
report redacts GPU UUIDs and machine-specific filesystem paths.

Reviewed M3 capacity evidence is stored at `benchmarks/v0.1/track_capacity_report.json`. It records
10,000 contiguous seeds at each of 0.75/1.0/1.25 m spacing, generation and validation rejections,
distribution percentiles, reproducibility, theoretical bounds, capacity selection, and TrackBatch
memory. Reviewed low-speed physical admission evidence is stored at
`benchmarks/v0.1/track_driveability_report.json`; 16/16 generated tracks completed on the formal
four-wheel backend at a 4 m/s target with no recorded failure outcome or numerical/buffer fault.

Reviewed M4 environment evidence is stored at `benchmarks/v0.1/m4_environment_report.json`. The
formal run used 1,024 different valid Tracks for 10,000 environment steps, measured 165,633
transitions/s, passed active and mixed-autoreset no-transfer guards, observed timeout and independent
autoreset in all worlds, recorded no non-finite public value, and used 556 MiB peak sampled process
VRAM with 10 MiB steady growth against a 64 MiB gate.

Reviewed M5 admission evidence is stored at `benchmarks/v0.1/m5_track_admission_report.json`. It
binds every official manifest and asset hash to clean revision `9d9d178`, records all attempted seed
outcomes, and verifies the written artifacts. Train selected 10,000 Tracks from 11,306 attempts
after 42 geometry and 1,220 physical rejections. The fixed-shape 1,024-world admission executed
54,161,408 transitions at 48,523 transitions/s; every selected official Track passed.

Reviewed M5 runtime evidence is stored at `benchmarks/v0.1/m5_track_pool_report.json`. Its v2
protocol keeps the complete 272,800,000-byte Train pool on GPU, verifies exact domain-2 selection,
transfer-free active/mixed reset, 65,536 reset-heavy events, all-world timeout/autoreset, numerical
health, JIT-cache stability, source/privacy, and a post-stabilization allocator plateau. The headline
1,024-world × 10,000-step epoch measured 210,372 transitions/s, 0.958 of the matched fixed baseline.
Peak sampled process VRAM was 1,334 MiB; after the disclosed one-time allocator expansion, process,
pool, and peak growth were zero through three distinct-seed 10,000-step epochs.

Reviewed M6 Controller evidence is stored at `benchmarks/v0.1/m6_controller_report.json`. Its
locked formal run used four batch-one MJX-Warp environment backends and 112 fresh Controller
instances. PID completed Level 0 and 10/10 fixed Validation-prefix Tracks. MPC completed Level 0
and 95/100 fixed Validation Tracks; all five failures were timeouts. Combined MPC compute timing
was 32.373/39.892/44.347 ms at P50/P95/P99 with a 0.0967% soft-deadline miss rate. All 234,358
public transitions were finite, invalid-action count was zero, peak sampled process VRAM was
396 MiB, and post-group JAX live bytes were zero. The report passed 34/34 gates, loaded only Level 0
and Validation, and did not access Test.

Reviewed M7 PPO evidence is split across the training run and three committed reports. The formal
training source was clean revision `86f8f384`; one long-lived 1,024-world official Train environment
completed 80 updates and 10,466,653 valid transitions at 56,245.788 end-to-end valid transitions/s.
Peak sampled process VRAM was 1,180 MiB and no numerical error was recorded. Frozen candidate
updates `[10, 20, 30, 40, 50, 60, 70, 80]` were evaluated once on Validation; update 70 was selected
with 95/100 successes against 0/100 for the seeded random baseline. The exported inference-only
NumPy policy is SHA-256
`f3054e95c6d357f571425ad69b9ac16c713e24b9f09b7768e7a648af84731a4b`.

The ordinary Controller evaluation at clean source `1b434f4` instantiated 100 fresh Controllers and
completed 99/100 fixed Validation Tracks with a 24.316667 s mean successful lap time over 48,709
environment steps. Controller compute timing was 0.260/0.305/0.332 ms at P50/P95/P99 with zero
50 ms deadline misses. Peak sampled process VRAM was 364 MiB and final JAX live bytes were zero.
Protocol v2 captured the selected row-0 replay inline from that same evaluation trajectory, with no
cherry-picking and no second rollout. A preceding formal v1 replay attempt exposed MJX-Warp atomic
nondeterminism, failed its gate, and fully rolled back before v2 was frozen and run.

M7 performance paths never accessed Test. Routine official-asset verification may hash Test assets
but does not instantiate a Test environment, run a policy, or observe Test performance. Before the
formal selection, one capacity-only diagnostic loaded the Validation asset to inspect its fixed
shape; it created no environment, ran no policy, and observed no performance. This access is
disclosed separately from the formal selection, whose own pre-Validation access count was zero.

Reviewed M8 evidence is stored in `benchmarks/v0.1/m8_final_evaluation_report.json`, the central
comparison CSV/PNG, and the three `results/0.1/<controller>/m8-final-v0-1-002/` directories. The
accepted clean source is `6095481`. PID completed 20/20 Test Tracks, MPC 20/20, and PPO 19/20; the
official ranking is PID, MPC, PPO. The run executed 85,874 Environment steps in 2,873.186 seconds,
recorded zero numerical failures, used 360 MiB peak sampled process VRAM, and ended with zero JAX
live bytes. Its retained transaction is `COMMITTED` with 60 journal rows, 60 trajectory blobs, a
typed execution seal, semantic validation, and exactly 24 outputs.

Attempt 001 had loaded Test but stopped during Environment creation before reset, step, Controller
construction, or performance. Its canonical failure report and unchanged retained transaction bind
the explicit one-replacement authorization. Attempt 002 is the accepted benchmark `0.1` result;
later executions are reproductions and no third official attempt is permitted.

## Experimental Decisions

M1 fixed the physics timestep at 0.005 seconds and proved the standardized actuator mapping on CPU;
M2 retained that timestep on MJX-Warp and locked the reviewed flat-ground contact/constraint
capacities. M3 fixed 1.0 m Track sampling, 640 points, 48 checkpoints, the current generator and
validation ranges, and the 4/12 local projection window from measured evidence. M4 fixed the
single/vector Gymnasium schema, NEXT_STEP autoreset, restricted Controller boundary, domain-separated
device-native episode identities, and transfer-free MJX-Warp hot path. M5 fixed the official split
quotas/namespaces/assets, canonical geometry hashes and manifests, domain-2 TrackPool selection,
numeric Track ID contract, local Train cache, bounded physical admission, and v2 full-pool
memory/performance protocol. M6 fixed the public-only classical-Controller examples, shared
geometry/speed-planning utilities, bounded-iteration CasADi/IPOPT MPC configuration, reusable
batch-one formal evaluator semantics, and the reviewed Level 0/Validation timing protocol. M7 fixed
the public PPO feature/reward wrappers, NEXT_STEP-aware training semantics, frozen Train-only PPO
configuration, Validation-only checkpoint selection, inference-only NumPy export, and ordinary
Controller replay protocol. M8's final Test-only PID/MPC/PPO comparison fixes one shared
environment, 60 Controller-major episodes, same-row Track/seed identity, typed transition metrics,
row-zero same-rollout replay, post-close runtime/memory/access evidence, and transactional
publication. Attempt 001 reached Test loading but failed in Environment creation before reset,
step, Controller construction, or performance. Attempt 002 retained every performance-affecting
input, added only the authorized pre-bind Warp initialization and lineage gates, and completed as
the accepted result. The M8 performance protocol is now immutable within benchmark `0.1`.

## External Reference

- Upstream inspiration: <https://github.com/learnsyslab/lsy_drone_racing>
- Local study copy: `reference/lsy_drone_racing/`

The local copy is evidence and inspiration only. It is not a vendored dependency and must not enter the public Git history.
