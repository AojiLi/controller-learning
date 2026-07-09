# Active Context

Last updated: 2026-07-10

## Current Direction

Finish M0 before starting vehicle work. The local repository foundation and both locked Pixi
environments are implemented and verified; the remaining M0 acceptance evidence is a successful
CPU workflow run in the private GitHub repository.

## Current Narrow Focus

Publish the verified M0 foundation to a private GitHub repository and confirm that its locked CPU
Actions workflow passes. This is an external GitHub operation and requires the user's approval.

## Current Goal

Complete M0 by obtaining a green GitHub CPU CI run for the exact local lockfile and commit. Do not
start M1 until that evidence exists.

## Scope for the Next Implementation Change

In scope:

- create `AojiLi/controller-learning` as a private GitHub repository after approval;
- push the verified local M0 commit;
- monitor and, if necessary, fix the CPU CI workflow;
- record the successful Actions run and advance `ACTIVE_CONTEXT.md` to M1.

Out of scope:

- beginning the final four-wheel MJCF before M0 is accepted;
- track generation, Gymnasium environments, Controller algorithms, or PPO;
- advertising GPU throughput before M2 evidence;
- macOS/Windows/WSL2 support;
- changing confirmed v0.1 architecture for convenience.

## Confirmed Judgments

- `PROJECT_PLAN.md` is approved and is the detailed decision source.
- Python 3.11.15 is locked in both Pixi environments.
- `pixi run ci` passes 14 CPU tests, Ruff format/lint, and strict documentation build.
- The GPU environment jointly runs JAX 0.10.2, MuJoCo/MJX-Warp 3.10.0, Warp 1.13.0, and
  PyTorch 2.11.0+cu128 on the local RTX 5070 Ti; one MJX-Warp smoke test passes.
- Linux/NVIDIA native GPU batching is a product requirement, not an optional optimization.
- M1 proves the physical vehicle; M2 is the early GPU go/no-go before feature expansion.
- PID, MPC, and PPO are examples built after the Challenge foundation, not the initial implementation target.
- Public project content is English; these internal context files may remain Chinese or bilingual.

## Open Experimental Questions

There are no open product-direction questions. The only remaining M0 dependency is authorization
for the external private GitHub repository operation and its hosted CI run. Physics, contact,
track-capacity, MPC, and PPO numeric parameters remain deliberately open until their planned
benchmarks.

## Next Step

After user approval, create the private GitHub repository, push the local M0 commit, and monitor the
CPU workflow to completion. If it passes, record M0 complete and advance to M1.
