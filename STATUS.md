# Project Status

Last updated: 2026-07-10

**Status:** M1 is complete. M2 MJX-Warp GPU go/no-go is active.

## Main Line

Move the proven CPU four-wheel MJCF into MJX-Warp, establish CPU/GPU short-rollout agreement, and
measure native 1/64/256/1024-world GPU stability before building tracks or Controllers.

## Timeline

- Confirmed the product identity, v0.1 scope, Challenge architecture, public Controller API,
  benchmark protocol, platform policy, and M0–M8 milestone gates in `PROJECT_PLAN.md`.
- Completed M0 repository/Pixi/package/configuration infrastructure and created the private GitHub
  repository at <https://github.com/AojiLi/controller-learning>. Hosted CPU CI run
  [29054661176](https://github.com/AojiLi/controller-learning/actions/runs/29054661176) passed.
- Verified the locked GPU environment on the local RTX 5070 Ti Laptop GPU with JAX 0.10.2,
  MuJoCo/MJX-Warp 3.10.0, Warp 1.13.0, and PyTorch 2.11.0+cu128. One finite MJX-Warp smoke step
  passed; this was dependency/device evidence, not a vehicle-scale result.
- Completed the M1 physical vehicle in implementation commit
  `237f5046dc369095e4247efefe80e2b728254044`: one rigid 6-DoF chassis, four physical rotating
  wheels, two front steering joints, four-wheel drive/brake mapping, rear-axle state extraction,
  substep contact diagnostics, CPU viewer, installed-wheel asset validation, and formal benchmark.
- Local M1 CPU CI passed 44 tests with one GPU test deselected, strict docs, Actions lint,
  sdist-to-wheel construction, installed-wheel MJCF loading, and package metadata checks.
- Generated the clean, hash-backed `benchmarks/v0.1/m1_cpu_report.json`:
  - 0.010 s failed long-stress penetration, vertical-motion, and convergence gates;
  - 0.005 s and 0.002 s passed, so 0.005 s is the largest passing CPU candidate;
  - the selected 0.005 s stress run completed 60 seconds with no warnings or unexpected contacts,
    80.15% minimum per-wheel substep contact participation, a 55 ms maximum continuous contact gap,
    and 0.791 mm steady penetration P99;
  - rest, straight, mirrored steering, braking, action clipping/rate limiting, determinism, symmetry,
    and convergence checks all passed.

## Current Thinking

The CPU model is now sufficiently stable and reproducible to attempt M2. Its 0.005 s selection is a
starting candidate, not a GPU result. The next risk is whether the same contact-rich MJCF remains
stable, consistent, and memory-efficient under MJX-Warp native batching.

## Next Step

Implement batch-size-one MJX-Warp load/reset/step and a short CPU comparison. If that passes, scale
to 64, 256, and 1024 worlds, then run the required 1024-world × 10,000-step endurance benchmark and
write `benchmarks/v0.1/gpu_report.json`.

## Risks and Blockers

- CPU MuJoCo stability does not prove MJX-Warp compatibility or numerical agreement.
- Fixed contact/constraint capacities may overflow or consume too much memory at 1024 worlds.
- The local GPU memory budget may require buffer tuning or the smaller 0.002 s timestep.
- M2 may trigger the approved pure-JAX planar four-wheel fallback if measured tuning cannot make the
  formal vehicle stable and scalable; CPU multiprocessing and a bicycle truth model remain invalid
  fallbacks.
