# Active Context

Last updated: 2026-07-10

## Current Direction

Execute the M2 MJX-Warp GPU go/no-go using the same four-wheel MJCF proven by M1. M0 repository
infrastructure and M1 CPU vehicle validation are complete. Track, Challenge, Controller, and RL
feature work remains gated on M2.

## M1 Handoff Evidence

- Implementation commit: `237f5046dc369095e4247efefe80e2b728254044`.
- Reviewed report: `benchmarks/v0.1/m1_cpu_report.json`.
- The report was generated from a clean worktree and records matching source, model, config, lock,
  protocol, and Git hashes.
- 0.010 s failed the long stress penetration, vertical-motion, and convergence gates.
- 0.005 s and 0.002 s passed; 0.005 s is the largest passing CPU candidate and M2 starting point.
- At 0.005 s, the 60-second stress run had no warnings or unexpected contacts, 80.15% minimum
  per-wheel substep contact participation, a 55 ms maximum contact gap, and 0.791 mm steady
  penetration P99.
- Local CPU CI passed 44 tests with one GPU test deselected, strict docs, Actions lint, sdist-to-wheel
  construction, installed-wheel asset loading, and metadata checks.

These facts do not prove MJX-Warp compatibility, CPU/GPU agreement, 1024-world stability, or GPU
throughput.

## Current Narrow Focus

- Put the packaged `car.xml` through the MJX-Warp conversion path without maintaining a second
  vehicle definition.
- Establish batch-size-one stepping and compare a short fixed rollout with CPU MuJoCo.
- Scale only after batch one passes: 1, 64, 256, then 1024 independent worlds.
- Measure compile time, steady-state steps and transitions per second, peak VRAM, numerical warnings,
  contact/constraint overflows, and long-run memory behavior.
- Tune fixed contact/constraint capacities and graph/buffer choices without weakening the public
  vehicle or native leading-dimension batching requirement.
- Produce `benchmarks/v0.1/gpu_report.json` from a clean revision on the local NVIDIA GPU.

## Current Goal

Pass 1024 worlds for 10,000 environment steps with finite independent state, no warning or buffer
overflow, no sustained VRAM growth, and a documented CPU/MJX-Warp short-rollout tolerance. If the
formal four-wheel model cannot pass after measured tuning and permitted simplification, trigger the
approved pure-JAX planar four-wheel fallback review.

## Scope Boundaries

In scope:

- MJX-Warp conversion and step code for the M1 model;
- batch initialization, stepping, deterministic reset inputs, and finite-state diagnostics;
- CPU/GPU short-rollout comparison;
- contact/constraint capacity and GPU memory tuning;
- 1/64/256/1024-world benchmarks and versioned M2 evidence.

Out of scope:

- procedural tracks or Race Core (M3);
- Gymnasium environments and Controller plugins (M4);
- PID, MPC, PPO, or MPCC;
- suspension, detailed tire models, a second training plant, or CPU multiprocessing fallback;
- macOS, Windows, WSL2, or public GPU performance claims before M2 passes.

## Confirmed Judgments

- `PROJECT_PLAN.md` remains the detailed decision source.
- M1 proves the physical CPU plant and selects 0.005 s only as the M2 starting candidate.
- The same MJCF and standardized action semantics must be used by CPU and GPU paths.
- M2 is an early stop condition: do not begin visible Track or Controller features to bypass it.
- Native NVIDIA GPU batching is a product requirement, not an optional optimization.
- GPU tests remain local for v0.1 and require reviewed, versioned evidence.

## Open Experimental Questions

- Whether MJX-Warp accepts every current MJCF feature without semantic modification.
- The smallest fixed contact and constraint capacities that avoid overflow at 1024 worlds.
- Whether 0.005 s remains stable and sufficiently close to CPU, or M2 must select 0.002 s.
- The measured CPU/GPU rollout tolerance for position, velocity, yaw, wheel speed, and contact state.
- Whether 1024 worlds fit the local GPU memory budget with stable throughput and no growth.

## Next Step

Implement the thinnest batch-size-one MJX-Warp adapter for the packaged model, verify one finite
reset/step and a short CPU comparison, then scale in the required 1/64/256/1024 order. Do not begin
M3 until the versioned M2 report passes or the approved fallback decision is explicitly activated.
