# Active Context

Last updated: 2026-07-10

## Current Direction

Implement and prove the M1 CPU MuJoCo four-wheel vehicle. M0 is complete: the private GitHub
repository exists, both Pixi environments are locked and locally verified, and the hosted CPU
workflow passes on `main`.

## Current Narrow Focus

Build the actual simulation-truth vehicle and its CPU reference interface before any vector or
track work:

- one free 6-DoF chassis;
- four independently rotating physical wheels;
- two front steering joints;
- shared steering-angle and longitudinal-acceleration actuator mapping;
- typed state extraction in the confirmed SI/world/body convention;
- measured stability and behavior at the 100/200/500 Hz timestep candidates.

## Current Goal

Complete M1 with a stable CPU MuJoCo vehicle that passes rest, straight-line, steering, braking,
action-limit, contact, coordinate, and timestep-scan tests. Record measured results and select a
candidate physics timestep for M2 without claiming GPU equivalence yet.

## Scope for the Next Implementation Change

In scope:

- `controller_learning/assets/vehicle/car.xml` and only the assets it actually needs;
- CPU model loading, reset, actuator conversion, stepping, and state extraction;
- front steering and four wheel-spin joint semantics;
- deterministic CPU rollout tests and a versioned M1 measurement report;
- headless tests as the required gate, with the MuJoCo viewer only as an optional debug CLI.

Out of scope:

- MJX-Warp batching or GPU performance claims;
- adding suspension, a detailed differential, Pacejka tires, or aerodynamics;
- track generation, Gymnasium environments, Controller algorithms, or PPO;
- advertising GPU throughput before M2 evidence;
- macOS/Windows/WSL2 support;
- changing confirmed v0.1 architecture for convenience.

## Confirmed Judgments

- `PROJECT_PLAN.md` is approved and is the detailed decision source.
- M0 is complete. Hosted CPU CI run `29054661176` passed for `main` with the locked environment.
- Python 3.11.15 is locked in both Pixi environments.
- `pixi run ci` passes 14 CPU tests, Ruff format/lint, and strict documentation build.
- The GPU environment jointly runs JAX 0.10.2, MuJoCo/MJX-Warp 3.10.0, Warp 1.13.0, and
  PyTorch 2.11.0+cu128 on the local RTX 5070 Ti; one MJX-Warp smoke test passes.
- Linux/NVIDIA native GPU batching is a product requirement, not an optional optimization.
- M1 proves the physical vehicle; M2 is the early GPU go/no-go before feature expansion.
- PID, MPC, and PPO are examples built after the Challenge foundation, not the initial implementation target.
- Public project content is English; these internal context files may remain Chinese or bilingual.

## Open Experimental Questions

There are no open product-direction questions. M1 must experimentally select among the confirmed
physics timestep candidates and tune ordinary MuJoCo contact/solver parameters. Those measurements
may change candidate numeric configuration values but must not change the four-wheel truth model.

## Next Step

Implement the smallest complete CPU four-wheel vehicle slice: MJCF structure, loader/state schema,
actuator mapping, and rest/straight/steer/brake tests. Then run the timestep scan and record M1
evidence before entering M2.
