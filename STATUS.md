# Project Status

Last updated: 2026-07-10

**Status:** M4 is complete. M5 Level assets and Track pools are active.

## Main Line

Turn the proven M4 Gymnasium/Controller path into a versioned Level 0/1 benchmark by adding fixed
assets, deterministic train/validation/test splits, and device-native pool reselection. PID/MPC and
PPO remain gated on M5.

## Completed Evidence

- M0 established the Linux/Pixi package, typed configuration, tests, docs, and private GitHub CPU CI.
- M1 selected the stable 0.005 s CPU MuJoCo physics timestep and validated the four-wheel vehicle.
- M2 established native MJX-Warp batching through 1,024 worlds × 10,000 steps at 77,751
  transitions/s without numerical, capacity, or memory-growth failure.
- M3 locked deterministic 1.0 m Track sampling, 640 points, 48 checkpoints, batched Race Core,
  1,024-world isolation, and conservative four-wheel driveability.
- M4 added the registered `ControllerLearning/CarRacing-v0` single environment and the native
  `VecCarRacingEnv` without a second Challenge implementation.
- Observation/action, restricted info, invalid-action, terminal, batch-one, and strict NEXT_STEP
  autoreset semantics pass CPU and GPU tests.
- Device episode identities reproduce the NumPy `SeedSequence` contract, and warm active/mixed-reset
  GPU steps pass transfer guards with no host/device transfer.
- Trusted Controller plugins load in isolated packages, receive immutable public config/info, and
  start from a fresh instance every episode. The renderer receives only public observations and
  write-only `DebugDraw` commands.
- `pixi run sim` completes the neutral template episode on CPU and MJX-Warp; this is an interface
  test, not a driving baseline.
- The reviewed `benchmarks/v0.1/m4_environment_report.json` passed all formal gates. It used 1,024
  distinct valid Tracks for 10,000 steps, measured 165,633 transitions/s, observed independent
  timeout/autoreset in all worlds, recorded zero non-finite public values, and sampled 556 MiB peak
  process VRAM with 10 MiB steady growth.
- Current local validation passes 298 CPU/default tests and all 21 GPU tests.

M4 therefore clears the public environment, native GPU hot path, Controller isolation, rendering,
and simulation CLI gates. It does not claim that any driving Controller succeeds.

## Current Thinking

M5 should extend Track ownership rather than environment architecture. A pool-aware Track source
must feed the same fixed-shape `VecCarRacingEnv`, select replacement geometry on device during
masked reset, and preserve deterministic episode/Controller seed domains. Fixed validation/test
geometry and hashes should be committed and immutable, while the roughly 260 MiB training pool
needs a reproducible materialization/cache strategy instead of an unreviewed large Git artifact.

## Next Step

Lock the M5 manifest and split schema, public device-native Track ID, per-world pool RNG rule, and
Level 0 asset before generating the formal 10,000-Track pool.

## Risks and Blockers

- The M4 string Track ID is constant for injected Tracks; dynamic pool selection needs a numerical
  device representation without breaking the public single-environment API.
- A convenient host-side pool sampler would reintroduce the transfer bottleneck removed in M4.
- Committing the full training arrays would create an unsuitable repository artifact; regenerating
  without hashes would weaken reproducibility.
- Physical admission of roughly 10,000 Tracks must be batched and bounded so it remains practical
  while still rejecting undriveable geometry.
