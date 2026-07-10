# Project Status

Last updated: 2026-07-10

**Status:** M2 is complete. M3 batched Track and Race Core is active.

## Main Line

Build deterministic fixed-capacity closed-track geometry and the batched JAX race-state operations
on top of the now-proven MJX-Warp four-wheel backend. Gymnasium, Controller plugins, PID/MPC, and PPO
remain gated on later milestones.

## Completed Evidence

- M0 established the Linux/Pixi package, typed configuration, tests, docs, and private GitHub CPU CI.
- M1 established the physical CPU MuJoCo four-wheel reference and selected a 0.005 second physics
  timestep in `benchmarks/v0.1/m1_cpu_report.json`.
- The current-state correction in `753fa0d39b9109db771a526e3092e309062f64e0` derives rear-axle pose
  and velocity directly from integrated `qpos/qvel`.
- M2 added the native leading-dimension adapter in `00200696304a46226bd71e2de979fa35fbc6af0b`
  and the isolated formal protocol in `7f538055299962e92845794ad2ed033b43219632`.
- Local CPU CI passes 66 tests plus strict formatting, lint, docs, Actions, and package checks. The
  local NVIDIA suite passes 11 GPU tests.
- The clean, hash-backed `benchmarks/v0.1/gpu_report.json` passed every M2 gate:
  - fresh 1/64/256/1024-world workers all passed;
  - 1024 worlds completed 10,000 measured environment steps, 10,240,000 transitions, and
    102,400,000 world-physics steps;
  - measured throughput was 77,751 transitions/s and 777,506 world-physics steps/s;
  - peak process VRAM was 346 MiB with no long-window growth and stable JAX live allocation;
  - no non-finite state, overflow, unexpected contact, invalid action, or native warning occurred;
  - the minimum 100-step per-world/per-wheel mean contact participation was 84.3%, maximum contact
    gap 65 ms, maximum penetration 1.361 mm, and 1,255 masked resets were exercised;
  - CPU/GPU maximum planar-position and body-velocity errors were 0.034 mm and 0.009 mm/s; contact
    participation, contact gap, and penetration also agreed within the recorded tolerances.

M2 therefore keeps the approved MJX-Warp path. The pure-JAX four-wheel fallback was not activated.

## Current Thinking

The largest early technical risk is cleared: one physical four-wheel model is stable on CPU and in
native 1024-world GPU batching. M3 should now keep track geometry as fixed-shape JAX data, independent
from MuJoCo collision geometry, so different worlds can use different closed tracks without forcing
recompilation or duplicating physics models.

## Next Step

Measure and lock the M3 fixed-capacity Track representation, then implement the deterministic
periodic-spline generator and geometry validator before adding checkpoint progress, effective
boundaries, timeout, reward, termination, and masked race-state reset.

## Risks and Blockers

- `max_track_points`, arc-length spacing, and `max_checkpoints` still require a distribution spike;
  undersizing rejects valid tracks while oversizing wastes GPU memory.
- Periodic resampling and boundary offsets must preserve closure, orientation, minimum separation,
  and curvature constraints deterministically across seeds.
- Projection onto spatially adjacent but non-adjacent segments must not allow progress jumps.
- Low-speed driveability validation must use the formal four-wheel backend without becoming a hidden
  Controller shortcut or changing the benchmark distribution.
