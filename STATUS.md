# Project Status

Last updated: 2026-07-11

**Status:** M0 through M8 are complete, and the v0.1 repository is public.

## Main Line

Maintain the released v0.1 evidence and interfaces. Attempt 001's zero-episode infrastructure
failure is retained and disclosed. Attempt 002 completed the fixed 20-Track comparison and all
transaction/artifact gates. The accepted artifacts, documentation, and source are public.

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
Controller timing/lifecycle, and replay gates.

- M8 attempt 002 completed from clean source `6095481` after the explicitly authorized zero-episode
  infrastructure replacement. PID completed 20/20 Test Tracks, MPC 20/20, and PPO 19/20; ranking is
  PID, MPC, PPO.
- The run executed 85,874 Environment steps in 2,873.186 seconds with zero numerical failures,
  360 MiB peak sampled process VRAM, and zero final JAX live bytes.
- The retained transaction is `COMMITTED` with 60 journal rows, 60 trajectory blobs, a typed
  execution seal, semantic validation, and exactly 24 published artifacts. Attempt 001's original
  transaction hashes remain unchanged.

## Current Work

- No v0.1 implementation work remains.
- Preserve benchmark `0.1`, the accepted Test result, and the public Controller boundary.
- Treat maintenance fixes as non-performance-changing unless a future benchmark version is
  explicitly planned.

## Next Step

Plan any post-v0.1 work as a separate version without changing the published benchmark `0.1`
result.

## Risks and Blockers

- The accepted Test result is immutable; no later reproduction can replace it.
- Linux x86-64 remains the only supported v0.1 platform; macOS, native Windows, and WSL2 require
  future test evidence before support claims.
