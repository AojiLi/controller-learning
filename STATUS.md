# Project Status

Last updated: 2026-07-10

**Status:** M3 is complete. M4 Gymnasium environments and Controller platform is active.

## Main Line

Expose the proven MJX-Warp vehicle, fixed-capacity tracks, and batched Race Core through one public
single/vector Gymnasium contract, then add the trusted Controller plugin boundary. Level pools,
PID/MPC, and PPO remain gated on later milestones.

## Completed Evidence

- M0 established the Linux/Pixi package, typed configuration, tests, docs, and private GitHub CPU CI.
- M1 established the physical CPU MuJoCo four-wheel reference and selected a 0.005 second physics
  timestep in `benchmarks/v0.1/m1_cpu_report.json`.
- M2 established native leading-dimension MJX-Warp simulation. Its reviewed
  `benchmarks/v0.1/gpu_report.json` passed 1/64/256/1024-world gates and the 1024-world × 10,000-step
  endurance run at 77,751 transitions/s without numerical, contact-capacity, or memory-growth failure.
- M3 added immutable fixed-capacity Track arrays, deterministic periodic-spline generation, strict
  geometry validation, project-config adapters, and rear-axle pose reset.
- The reviewed `benchmarks/v0.1/track_capacity_report.json` swept 10,000 seeds per spacing. The
  selected 1.0 m representation generated 9,994 candidates, accepted 9,965, and passed 8/8 sampled
  reproducibility checks; the only primary rejections were six length and 29 curvature failures.
- The locked capacity is 640 track points and 48 checkpoints against theoretical maxima of 601 and
  40. Runtime arrays use 26.640625 MiB for 1,024 worlds and 260.162 MiB for a 10,000-track pool.
- M3 Race Core implements local topology-aware projection, ordered checkpoint progress, reward,
  effective boundary, timeout, termination priority, and masked reset.
- GPU Race Core tests reused one compiled executable across 1,024 distinct tracks, preserved every
  unselected world during masked replacement/reset, and left 1,023 worlds bit-exact after a 16-step
  perturbation to one world. Peak JAX allocation was about 140.4 MB.
- The reviewed `benchmarks/v0.1/track_driveability_report.json` passed 16/16 generated tracks at a
  4 m/s target over 46,400 transitions. Maximum lateral error was 0.2387 m and maximum speed was
  3.9847 m/s; no off-track, timeout, invalid-action, numerical, overflow, or unexpected-contact
  failure occurred.
- Current local validation passes 153 CPU/default tests and all 17 GPU tests: one environment check,
  12 vehicle tests, three Race Core tests, and one driveability test.

M3 therefore clears the fixed-shape geometry, independent race state, memory, and conservative
four-wheel driveability gates. It does not claim Gymnasium compliance or Controller availability.

## Current Thinking

M4 should remain an adapter milestone. The environment wrappers must compose the existing physical
step and Race Core without duplicating reward, termination, reset, or track logic. The same vector
path must remain suitable for direct batched PPO training, while the Controller loader stays outside
the hot path and creates one fresh trusted plugin instance per single evaluation episode.

## Next Step

Implement the public observation/action codec and deterministic single/vector reset semantics, then
add `CarRacingEnv`, `VecCarRacingEnv`, and Gymnasium registration on that common transition path.
After those contracts pass, add the Controller base/loader/template, restricted config/info,
write-only `DebugDraw`, and minimal simulation CLI.

## Risks and Blockers

- A convenient Python observation structure could create avoidable host transfers or object work in
  the later PPO vector path; shapes and dtypes need to be fixed at the boundary.
- Single and vector wrappers can drift if they own separate transition logic; batch-size-one
  equivalence and shared implementation are required.
- Gymnasium terminal-observation and NEXT_STEP autoreset semantics must preserve per-world outcomes
  without changing the already-tested Race Core reset behavior.
- Controller configuration, info, callbacks, and rendering must not expose simulator or Challenge
  internals as accidental control shortcuts.
