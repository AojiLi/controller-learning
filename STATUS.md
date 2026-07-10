# Project Status

Last updated: 2026-07-10

**Status:** M6 is complete. M7 PPO and GPU training are active.

## Main Line

Train one PyTorch PPO directly on the official 1,024-world `VecCarRacingEnv`, record reproducible
local training evidence, export the selected checkpoint as a normal Controller plugin, and prove it
learns beyond seeded random actions without touching Test.

## Completed Evidence

- M0 established the private Linux/Pixi repository, package, typed config, CPU CI, and docs.
- M1 selected the stable 0.005 s CPU MuJoCo timestep for the physical four-wheel vehicle.
- M2 proved native MJX-Warp batching through 1,024 worlds × 10,000 steps.
- M3 locked deterministic Track generation, 640/48 fixed capacity, Race Core, and physical
  driveability.
- M4 added the registered single environment, native vector Challenge, trusted Controller plugins,
  renderer, and template simulation CLI.
- M5 published fixed Level 0 plus disjoint 10,000/100/20 Level 1 Train/Validation/Test manifests,
  verified the reproducible Train cache, and measured 210,372 transitions/s on the full GPU pool.
- M6 added observation-only geometry and speed planning, PID, and constrained CasADi/IPOPT MPC with
  configs, DebugDraw, tests, and an English tutorial.
- The formal M6 report passed 34/34 gates. PID completed Level 0 and 10/10 Validation-prefix Tracks;
  MPC completed Level 0 and 95/100 full Validation Tracks. All five MPC failures were timeouts.
- MPC compute timing was 32.373/39.892/44.347 ms at P50/P95/P99 with a 0.0967% soft-deadline miss
  rate. PID P99 was 0.401 ms with no misses.
- The M6 run checked 234,358 public transitions through four MJX-Warp backends and 112 fresh
  Controller instances. It recorded no non-finite output or invalid action, 396 MiB peak process
  VRAM, zero post-group JAX live bytes, fixed row ordering/seeds, and no Test access.
- Current local validation passes 547 CPU/default tests, all 23 GPU tests, strict documentation,
  official-asset verification, Actions linting, and release-package checks.

M6 therefore clears the public classical-Controller, fixed Validation success, timing, lifecycle,
and formal evaluation gates. PPO learning remains unproven until M7.

## Current Work

- strict PPO config and a verified Train-only asset loader;
- public local-track observation and reward wrappers;
- numeric JAX-to-Torch DLPack compatibility with public string info preserved;
- NEXT_STEP-aware rollout masks and GAE;
- CleanRL-style PPO, CSV/TensorBoard logging, and atomic checkpoints;
- 1,024-world smoke/full training, Validation selection, random baseline, replay, and Controller
  export.

## Next Step

Implement the M7 configuration and Train-only pool loader with mutation and Test-access guards,
then add the public wrapper stack before the PPO optimizer.

## Risks and Blockers

- Stock Gymnasium `JaxToTorch` fails on the public NumPy string `benchmark_version`; the compatible
  bridge must preserve the full info whitelist without host-copying numeric hot-path values.
- NEXT_STEP reset-only slots must be excluded from GAE and PPO losses or training semantics are
  wrong across episode boundaries.
- The all-split asset verifier reads Test and is forbidden in M7 training/selection paths.
- Reward/feature weights, PPO hyperparameters, and training budget need Train-only evidence before
  they are frozen for Validation selection.
- The 272.8 MB Train NPZ remains local and ignored; run directories and intermediate checkpoints
  must not enter Git.
