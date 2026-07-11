# Project Status

Last updated: 2026-07-11

**Status:** M7 is complete. M8 final evaluation and release work are active.

## Main Line

Freeze and commit the Test-only final protocol before any Test performance access, then evaluate
PID, MPC, and PPO once on the same fixed-order 20-Track Test split. Finish public documentation and
release audits afterward; the repository is still private.

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
- M7 trained PPO at clean source `86f8f384` through one official 1,024-world environment for 80
  updates and 10,466,653 valid transitions. It measured 56,245.788 valid transitions/s, 1,180 MiB
  peak sampled process VRAM, and no numerical errors.
- Frozen candidate updates `[10, 20, 30, 40, 50, 60, 70, 80]` were evaluated once on Validation.
  Update 70 achieved 95/100 successes and was selected; the seeded random baseline achieved 0/100.
- The exported inference-only policy is SHA-256
  `f3054e95c6d357f571425ad69b9ac16c713e24b9f09b7768e7a648af84731a4b`.
- The ordinary Controller run at clean source `1b434f4` completed 99/100 Validation Tracks with a
  24.316667 s mean successful lap time over 48,709 steps. Compute P50/P95/P99 was
  0.260/0.305/0.332 ms with zero deadline misses; peak sampled process VRAM was 364 MiB and final
  JAX live bytes were zero.
- Replay v2 captured fixed-order row 0 inline from that evaluation, with no cherry-pick or second
  rollout. The earlier formal v1 attempt failed on MJX-Warp atomic nondeterminism and fully rolled
  back before v2 was frozen.
- One pre-formal capacity-only Validation-loader diagnostic inspected fixed shape without creating
  an environment, running a policy, or observing performance. Formal selection began with zero
  prior Validation opens in its process.
- M7 performance paths never accessed Test. Routine asset verification may hash Test assets but
  does not run Controllers or reveal Test performance.

M7 therefore clears PPO learning, frozen Validation selection, inference-only export, ordinary
Controller timing/lifecycle, and replay gates. Final Test comparison and release proof remain M8.

## Current Work

- commit and push the complete Test-only M8 protocol and rejection tests before performance access;
- run one formal same-order/same-seed PID/MPC/PPO comparison on all 20 Test Tracks;
- publish strict result and replay artifacts without Test-informed tuning or checkpoint changes;
- complete English README/tutorial/API/reproduction docs, package/privacy cleanup, and release
  audits;
- make the repository public only after every v0.1 release gate passes.

## Next Step

Commit and push the Test-only M8 protocol, which now passes the full 1,055-test CPU CI plus Linux GPU
and Validation-only smoke checks, then execute the single formal PID/MPC/PPO 20-Track run from that
clean revision. No formal Test Controller performance has been opened yet.

## Risks and Blockers

- Any Test performance access before the complete protocol is committed would invalidate the
  one-shot comparison boundary.
- Infrastructure failure handling and publication rollback must be predeclared so a failed run
  cannot become a performance-motivated rerun.
- PID, MPC, and PPO artifact/config identities must remain frozen throughout the Test run.
- Public-release claims remain blocked on final documentation, privacy, package, evidence, and
  repository-visibility checks.
