# Active Context

Last updated: 2026-07-10

## Current Direction

Execute M6: add the educational PID and CasADi/IPOPT MPC Controller examples on top of the now
published Level 0/1 benchmark. M0 through M5 are complete. Do not start PPO until PID/MPC, their
public geometry utilities, documentation, timing evidence, and M6 success gates are complete.

## M5 Handoff Evidence

- Level 0 is one fixed smooth ellipse with reserved numeric Track ID `UINT32_MAX`. Level 1 uses
  versioned train/validation/test manifests with 10,000 / 100 / 20 Tracks in disjoint seed
  namespaces. All selected IDs and packed-geometry hashes are disjoint.
- Validation/test/Level 0 assets are committed and packaged. The 272,800,000-byte training pool is
  reconstructed from the committed seed/hash manifest into `.track-cache/v0.1/train_pool.npz`; it
  is verified but never committed.
- Formal admission scanned seeds in ascending order without retry. It selected 10,000 Train Tracks
  from 11,306 attempts after 42 geometry and 1,220 physical rejections. Every selected official
  Track passed geometry and conservative four-wheel MJX-Warp driveability.
- `VecCarRacingEnv` remains the only Challenge state machine. It accepts either fixed injected
  Tracks or a `TrackPool`, derives pool selection from SeedSequence domain 2, and atomically replaces
  Track/vehicle/Race/observation state during NEXT_STEP reset.
- Public `track_id` is the device-native uint32 generator seed. Its stable namespace is
  `(benchmark_version, level_id, track_id)`. Terminal steps expose the old ID; the following reset
  step exposes the deterministically selected new ID.
- The formal M5 pool report passed 62 gates. The headline 1,024-world × 10,000-step epoch measured
  210,372 transitions/s; fixed-track baseline was 219,605 transitions/s, ratio 0.958. Active and
  mixed-reset transfer guards, exact domain-2 selection, all-world timeout/autoreset, JIT-cache
  stability, and numerical checks passed.
- The v2 memory protocol ran E0 plus three distinct-seed 10,000-step epochs on one environment.
  After the disclosed one-time 524 MiB allocator expansion, process VRAM and allocator pool/peak
  growth were zero through E3; live JAX growth stayed below 4.94 MB and host RSS drift was 0.027 MiB.
  Peak sampled process VRAM was 1,334 MiB.

M5 proves benchmark assets, physical admission, reproducible local materialization, device-native
pool sampling, and large-pool GPU execution. It does not claim that a public Controller can drive.

## Current Narrow Focus

1. Add public, observation-only geometry helpers needed by both classical Controllers without
   exposing Race Core or simulator internals.
2. Implement the longitudinal curvature-speed planner and anti-windup speed PID.
3. Implement the lateral cascade/PD PID with one parameter set that completes Level 0 and runs on
   Level 1 without per-track tuning.
4. Add CasADi + IPOPT through the existing Pixi lock, then implement a warm-started kinematic-car
   MPC with action, speed, and effective-track constraints.
5. Add Controller configs, DebugDraw, tutorials, timing/deadline measurements, deterministic
   evaluation scripts, and focused tests.
6. Demonstrate PID and MPC Level 0 completion, then tune MPC toward approximately 80% success on the
   fixed 100-Track Level 1 validation set without touching test geometry.

## Scope Boundaries

In scope:

- PID longitudinal/lateral loops, curvature speed planning, anti-windup, and interpretable config;
- CasADi/IPOPT kinematic-car MPC, warm start, constraints, and bounded fallback behavior;
- public geometry helpers derived only from observations;
- Level 0/validation evaluation and Controller compute-time/deadline evidence;
- English tutorials and DebugDraw examples.

Out of scope:

- PPO, reward shaping, or training code (M7);
- MPCC, acados, perception, multi-car racing, sim-to-real, or alternative simulation truth;
- tuning on the fixed Test split or changing any M5 manifest/asset/protocol;
- claims of macOS, Windows, WSL2, or real-time support without evidence.

## Confirmed Judgments

- The four-wheel simulator remains truth; MPC may use a kinematic car only as its internal model.
- Controllers consume only observation/info/public config and do not receive Environment, TrackPool,
  Race Core, MJX, or hidden projection state.
- Level 0 completion is required for both PID and MPC. The approximately 80% Level 1 validation
  target belongs to MPC and must be measured over the fixed 100-Track validation manifest.
- If nonlinear MPC misses the soft 50 ms deadline, first shorten the horizon and improve warm start;
  then evaluate linearized MPC + OSQP. Do not silently introduce a second Challenge path.
- Test Tracks stay untouched until M8 formal evaluation.

## Next Step

Inspect the existing public observation/controller boundary and implement the smallest shared
geometry/speed-planning utilities plus a tested PID Controller before adding CasADi or MPC.
