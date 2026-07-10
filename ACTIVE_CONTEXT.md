# Active Context

Last updated: 2026-07-10

## Current Direction

Execute M4: Gymnasium environments and the Controller platform. M0 repository infrastructure, M1
CPU four-wheel validation, M2 native MJX-Warp batching, and M3 fixed-capacity tracks plus Race Core
are complete. Do not start Level pools, PID/MPC, or PPO until the single/vector environment and
Controller boundary pass the M4 gates.

## M3 Handoff Evidence

- Reviewed capacity report: `benchmarks/v0.1/track_capacity_report.json`.
- Reviewed driveability report: `benchmarks/v0.1/track_driveability_report.json`.
- The deterministic capacity protocol tested 10,000 contiguous seeds at each of 0.75 m, 1.0 m, and
  1.25 m arc spacing. At the selected 1.0 m spacing, 9,994/10,000 candidates were generated and
  9,965 passed validation; six failed the length gate and 29 failed the curvature gate. All eight
  sampled seed-reproducibility checks passed.
- The v0.1 representation is locked at 1.0 m spacing, 640 track points, and 48 checkpoints. The 600 m
  theoretical bounds are 601 points including closure and 40 checkpoints. Runtime numerical arrays
  occupy 26.640625 MiB for 1,024 worlds and 260.162 MiB for a 10,000-track pool.
- `Track` is an immutable fixed-capacity host value; `TrackBatch` gives every numerical leaf a leading
  world dimension. Generation uses a deterministic periodic spline and one candidate per seed;
  validation covers schema, topology, curvature, separation, boundaries, start, and checkpoints.
- Race Core implements topology-local projection, ordered checkpoints, legal progress, reward,
  effective boundaries, length-dependent timeout, termination, and masked reset in fixed-shape JAX.
- The GPU suite passed 1,024 different tracks through the same compiled Race Core executables.
  Masked replacement/reset preserved unselected worlds, and a 16-step one-world perturbation left
  the other 1,023 worlds bit-exact. Observed peak JAX allocation was about 140.4 MB.
- The formal low-speed MJX-Warp run completed 16/16 generated tracks at a 4 m/s target over 46,400
  transitions. Maximum lateral error was 0.2387 m and maximum speed was 3.9847 m/s, with no off-track,
  timeout, invalid-action, numerical, overflow, or unexpected-contact failure.
- Current local validation passes 153 default-environment tests. The complete local GPU suite passes
  17 tests: one environment check, 12 vehicle tests, three Race Core tests, and one driveability test.

These measurements establish the Track and backend-independent Race Core layers. They do not yet
establish Gymnasium API compliance, Controller loading, public info/config restrictions, rendering,
or a complete simulation CLI.

## Current Narrow Focus

- Implement `ControllerLearning/CarRacing-v0`, `CarRacingEnv`, and `VecCarRacingEnv` on the same
  MJX-Warp vehicle, TrackBatch, and Race Core path.
- Define fixed Gymnasium observation and action spaces with documented dtypes, shapes, bounds, and
  conversion between the single and leading-batch interfaces.
- Implement deterministic seed handling, independent track selection, terminal observations, and
  Gymnasium NEXT_STEP masked autoreset without changing Race Core semantics.
- Enforce invalid-action versus finite out-of-range action behavior at the public environment
  boundary and expose only the confirmed restricted `info` fields.
- Implement the trusted directory Controller base, loader, template, fresh-per-episode lifecycle,
  read-only public config, independent Controller seed, and no simulator references.
- Add the write-only `DebugDraw` boundary and the minimal single-run simulation CLI.
- Pass Gymnasium checker, batch-size-one consistency, masked-autoreset isolation, plugin loading,
  fresh-instance, config-boundary, and state-leakage tests.

## Scope Boundaries

In scope:

- single and vector Gymnasium-compatible wrappers over the existing official backend;
- public observation/action encoding and restricted reset/step info;
- environment and Controller seed separation;
- trusted local Controller plugin loading and a non-performing template Controller;
- write-only debug drawing and a minimal simulation/debug CLI;
- CPU-contract tests and local GPU integration tests proportional to the backend boundary.

Out of scope:

- Level 0 fixed geometry, the 10,000-track training pool, fixed validation/test sets, and benchmark
  manifest (M5);
- PID, MPC, PPO, MPCC, Controller performance claims, and training;
- evaluator, leaderboard ordering, plots, replay publication, or release cleanup;
- alternate physics backends, platform-support expansion, perception, multiple cars, or real-time
  vehicle integration.

## Confirmed Judgments

- Single and vector environments are adapters over one official Challenge path; PPO must later train
  the same `VecCarRacingEnv`, not a simplified alternative.
- Controllers only receive public observations, restricted info, read-only public config, callbacks,
  and write-only `DebugDraw`. They never receive Environment, MJX Data, Simulator, or mutable
  Challenge configuration.
- A fresh Python Controller instance is created for every evaluation episode. Batched PPO training
  later consumes arrays directly and does not create one Controller object per world.
- Finite out-of-range actions are clipped and counted. Invalid shape/dtype conversion, NaN, and Inf
  terminate as `invalid_action`.
- Environment and Controller seeds remain independently and deterministically derived.
- `VecCarRacingEnv` keeps a leading `num_envs` dimension and uses Gymnasium NEXT_STEP masked
  autoreset. Batch-size-one behavior must agree with the single-environment contract.

## Open M4 Engineering Questions

- The smallest observation container and dtype policy that is both Gymnasium-compliant and efficient
  for direct JAX-to-PyTorch PPO consumption in M7.
- How terminal observations and restricted per-world info are represented without Python-object work
  in the vector hot path.
- The narrowest loader validation that keeps trusted plugins ergonomic while proving one exported
  Controller subclass and fresh episode state.
- The minimal rendering/debug surface needed for M4 without implementing the full M8 replay system.

## Next Step

Define and test the public observation/action codec and deterministic single/vector reset contract,
then place the Gymnasium wrappers over the existing MJX-Warp + TrackBatch + Race Core transition
without creating a second environment path.
