# Project Status

Last updated: 2026-07-10

**Status:** M0 is complete. M1 CPU MuJoCo four-wheel vehicle implementation is active.

## Main Line

Prove a stable four-wheel CPU MuJoCo vehicle at a measured physics timestep, then use the same MJCF
for the mandatory M2 MJX-Warp GPU gate.

## Timeline

- Confirmed the product identity, v0.1 scope, Challenge architecture, public Controller API, benchmark protocol, platform policy, and M0–M8 milestone gates in `PROJECT_PLAN.md`.
- Added project-specific Codex routing, durable context, active direction, and status files on 2026-07-10.
- Initialized local Git on `main`, excluded `reference/` and generated artifacts, and added the MIT
  License, English README/docs, package skeleton, typed immutable TOML schemas, Pixi lock, tests,
  and CPU workflow.
- Verified both locked Pixi environments on 2026-07-10:
  - CPU: Python 3.11.15; 14 tests passed and 1 GPU test deselected; Ruff, strict docs,
    `actionlint`, wheel/sdist builds, and Twine metadata checks passed.
  - GPU: JAX 0.10.2, MuJoCo/MJX-Warp 3.10.0, Warp 1.13.0, PyTorch 2.11.0+cu128; one finite
    MJX-Warp JIT step and the GPU pytest passed on an RTX 5070 Ti Laptop GPU.
- Created the private repository <https://github.com/AojiLi/controller-learning> and pushed `main`.
- Completed M0 after hosted CPU CI run
  [29054661176](https://github.com/AojiLi/controller-learning/actions/runs/29054661176) passed in 33
  seconds using the locked environment. No platform, throughput, or Controller-performance claim
  is inferred from this infrastructure result.

## Current Thinking

M1 must keep the plant physically four-wheeled while remaining simple enough for M2 throughput:
rigid chassis, four wheel-spin joints, two front steering joints, and wheel-ground contact, without
suspension or detailed drivetrain scope. Timestep and contact parameters remain measurements, not
assumptions.

## Next Step

Implement the M1 MJCF vehicle and CPU reference API, then verify rest, straight driving, steering,
braking, action limits, contact stability, coordinates, and the 100/200/500 Hz timestep candidates.

## Risks and Blockers

- Wheel-ground contact may be unstable or too sensitive at larger timesteps; M1 must measure it.
- A model that is stable in CPU MuJoCo may still fail or scale poorly in MJX-Warp; M2 remains a
  separate mandatory gate.
- GPU scale and vehicle stability remain unverified until M2; no performance claim should be published yet.
