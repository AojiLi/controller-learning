# Active Context

Last updated: 2026-07-10

## Current Direction

Execute M3: fixed-capacity batched Track geometry and Race Core. M0 repository infrastructure, M1
CPU four-wheel validation, and the M2 MJX-Warp GPU go/no-go are complete. Do not start Gymnasium or
Controller work until M3 geometry, progress, termination, and masked race reset pass their gates.

## M2 Handoff Evidence

- Reviewed report: `benchmarks/v0.1/gpu_report.json`.
- Adapter commit: `00200696304a46226bd71e2de979fa35fbc6af0b`; benchmark implementation
  commit: `7f538055299962e92845794ad2ed033b43219632`.
- The report records a clean and unchanged Git revision plus matching pre/post hashes for the model,
  config, lock, adapter, protocol, worker, and launcher.
- The same packaged MJCF and standardized physical actions run on CPU MuJoCo and MJX-Warp; only the
  unsupported Warp conversion copy of `mjDSBL_AUTORESET` is normalized, with explicit diagnostics
  replacing hidden recovery.
- Formal fresh workers passed at 1, 64, 256, and 1024 worlds. The 1024-world worker completed 10,000
  measured environment steps with finite independent state, 1,255 masked resets, no overflow,
  unexpected contact, invalid action, or native warning.
- 1024-world measured execution: 131.703 s, 77,751 transitions/s, 777,506 world-physics steps/s,
  346 MiB peak process VRAM, and no long-window process growth.
- Contact capacities are locked for the current flat-ground vehicle at 16 entries/world globally and
  64 constraints/world: observed headroom fractions were 0.50 contacts, 0.3125 broad-phase pairs,
  and 0.375 constraints.
- The fixed five-second batch-one comparison passed pose, attitude, velocity, steering, wheel-speed,
  contact-participation, contact-gap, and penetration tolerances.
- Current local validation: 66 CPU tests and 11 GPU tests pass.

These measurements prove the M2 physics layer only. They do not prove procedural-track geometry,
race progress, independent episode termination, Gymnasium compliance, Controller performance, or
PPO learning.

## Current Narrow Focus

- Define an immutable Track value with fixed-capacity centerline, left/right boundaries, masks,
  checkpoints, start pose, length, width, seed, and generator version.
- Spike the confirmed Level 1 generation distribution to choose one fixed arc-length spacing,
  `max_track_points`, and `max_checkpoints`; reject overflow instead of changing resolution.
- Implement deterministic 8–16 control-point periodic cubic-spline generation, fixed-arc-length
  resampling, tangents, normals, curvature, boundaries, start line, and ordered checkpoints.
- Validate closure, self-intersection, boundary crossing, non-adjacent spacing, curvature/turning
  radius, length, width, start straight, checkpoint direction/order, and seed reproducibility.
- Implement batched projection, legal progress, effective inward boundary, checkpoint crossing,
  lap success, off-track, length-dependent timeout, base reward, and masked race-state reset.
- Prove fixed shapes do not trigger JIT recompilation across seeds and different worlds remain
  independent.
- Add conservative low-speed driveability validation using the formal four-wheel backend.

## Scope Boundaries

In scope:

- offline deterministic Track generation and validation;
- fixed-capacity NumPy/JAX track batches and masks;
- public geometry helpers such as angle wrapping, frame transforms, and track projection;
- batched Race Core state, reward, success/off-track/timeout semantics, and masked reset;
- property, determinism, independence, and low-speed driveability tests.

Out of scope:

- Gymnasium registration and `CarRacingEnv`/`VecCarRacingEnv` wrappers (M4);
- Controller base/loader/template, PID, MPC, PPO, and MPCC;
- train/validation/test pool publication (M5, after generator and validator stabilize);
- visual road collision geometry, walls, cones, obstacles, multiple cars, or perception;
- platform claims beyond the measured Linux CPU/NVIDIA paths.

## Confirmed Judgments

- Track is independent JAX geometry; the physical world remains the common plane and four-wheel car.
- Fixed arc-length spacing is invariant. Invalid tails are zero-filled and masked; over-capacity
  tracks are rejected.
- Level 1 randomizes only closed planar geometry. Vehicle, start state, width, friction, plane, and
  obstacle count remain fixed.
- Progress requires ordered checkpoints and may not jump to a spatially close non-adjacent segment.
- Off-track uses the rear-axle reference against boundaries shrunk by half vehicle width plus the
  configured safety margin.
- Timeout is `max(60 s, track_length / 3 m/s)`; success/off-track/invalid action are terminated and
  timeout is truncated.
- Base reward is normalized forward progress, +1 success, and -1 off-track/invalid action. Ranking
  does not use reward.
- M3 masked reset must preserve every unmasked world and must not recompile for track seed or state.

## Open Experimental Questions

- Fixed arc-length spacing, `max_track_points`, and `max_checkpoints` for the confirmed distribution.
- Exact generator parameter ranges that reliably satisfy separation, curvature, length, and start
  straight constraints without changing the Level 1 definition.
- Efficient batched nearest-segment/progress logic that prevents topological shortcuts at 1024
  worlds within the local GPU memory budget.
- The conservative speed and steering policy used only to admit driveable generated tracks.

## Next Step

Create the minimal Track schema and a deterministic generator-distribution spike that reports point
and checkpoint capacity percentiles, rejection reasons, and reproducibility. Lock capacities from
that evidence before implementing the full batched Race Core.
