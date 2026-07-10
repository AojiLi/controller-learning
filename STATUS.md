# Project Status

Last updated: 2026-07-10

**Status:** M5 is complete. M6 PID and MPC are active.

## Main Line

Use the published Level 0/1 benchmark to implement and explain two classical Controller examples:
an interpretable PID baseline and a constrained CasADi/IPOPT MPC. PPO remains gated on M6.

## Completed Evidence

- M0 established the private Linux/Pixi repository, package, typed config, CPU CI, and docs.
- M1 selected the stable 0.005 s CPU MuJoCo timestep for the physical four-wheel vehicle.
- M2 proved native MJX-Warp batching through 1,024 worlds × 10,000 steps.
- M3 locked deterministic Track generation, 640/48 fixed capacity, Race Core, and initial physical
  driveability.
- M4 added the registered single environment, native vector Challenge, trusted Controller plugins,
  renderer, and template simulation CLI.
- M5 published fixed Level 0 plus disjoint 10,000/100/20 Level 1 train/validation/test manifests.
  Fixed assets are packaged; the 260.162 MiB Train arrays are reproducibly materialized into an
  ignored local cache and verified against the committed manifest.
- Formal M5 admission selected every official Track only after geometry and four-wheel driveability.
  Train required 11,306 attempts: 42 geometry and 1,220 driveability rejections were retained in the
  report. All artifact hashes, official paths, split isolation, readback, source, and runtime gates
  passed.
- The M5 GPU report passed all 62 gates with the full 10,000-Track pool resident. Its headline
  10,240,000-transition epoch measured 210,372 transitions/s, 0.958 of the matched fixed-track
  baseline. Exact domain-2 reset selection, no-transfer active/mixed steps, 65,536 reset-heavy
  events, and all-world timeout/autoreset passed.
- The strengthened E0–E3 memory protocol showed a post-stabilization plateau: 1,334 MiB peak process
  VRAM, zero process/pool/peak growth after E0, bounded live-buffer variation, stable host RSS, and
  no JIT recompilation.
- Current local validation passes 404 CPU/default tests, all 22 GPU tests, strict documentation,
  official-asset verification, Actions linting, and release-package checks.

M5 therefore clears Level assets, versioned manifests, physical admission, reproducible Train cache,
device-native pool autoreset, and full-pool GPU performance. It makes no PID/MPC/PPO success claim.

## Current Work

- observation-only geometry and speed-planning helpers;
- PID Controller, config, tests, DebugDraw, Level 0 completion, and Level 1 portability;
- CasADi/IPOPT dependency integration and warm-started constrained MPC;
- fixed-validation success and Controller deadline evidence;
- English PID/MPC tutorials.

## Next Step

Implement and validate shared public geometry helpers and the PID Controller on the fixed Level 0
asset, then use that tested path as the reference for MPC integration.

## Risks and Blockers

- Controller utilities must not expose Race Core projection/checkpoint internals as shortcuts.
- PID parameters must not be retuned per Track; validation must use the published fixed split.
- Nonlinear MPC may exceed the soft 50 ms compute deadline; horizon/warm-start tradeoffs require
  measurement before considering the planned linearized fallback.
- M5 evidence files are large but reviewed; the 272.8 MB Train NPZ remains local and ignored.
