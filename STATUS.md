# Project Status

Last updated: 2026-07-10

**Status:** M0 is implemented and verified locally; private GitHub publication and an actual green
CPU Actions run remain before M0 can be marked complete.

## Main Line

Finish the hosted M0 CI gate, then prove the four-wheel MuJoCo vehicle and MJX-Warp 1024-world GPU
path before expanding into tracks, Controller examples, and RL.

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

## Current Thinking

Dependency resolution and a minimal joint GPU step work, but they do not prove four-wheel contact
stability, 1024-world memory use, or throughput. Those claims remain gated by M1/M2. The public API
must remain narrow enough that the pure-JAX four-wheel fallback can preserve it if MJX-Warp fails
after measured tuning and simplification.

## Next Step

With user approval, create the private `AojiLi/controller-learning` GitHub repository, push the M0
commit, and verify the CPU Actions workflow. Advance to M1 only after it is green.

## Risks and Blockers

- The remaining M0 evidence requires an external GitHub repository operation and user approval.
- The local dependency matrix is proven installable and smoke-tested; large-scale GPU behavior is
  still unverified.
- GPU scale and vehicle stability remain unverified until M2; no performance claim should be published yet.
