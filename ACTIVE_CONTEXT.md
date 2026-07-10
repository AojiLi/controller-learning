# Active Context

Last updated: 2026-07-10

## Current Direction

Execute M7: train one PyTorch PPO directly against the official GPU-batched
`VecCarRacingEnv`, export the selected checkpoint as a normal single-environment Controller, and
produce reproducible training, comparison, and replay evidence. M0 through M6 are complete. Do not
touch the Test split or start M8 publication work until the M7 training and Controller gates pass.

## M6 Handoff Evidence

- PID and MPC are ordinary directory plugins that consume only the public observation, restricted
  info, immutable config, callbacks, and write-only `DebugDraw`. Shared geometry and speed-planning
  utilities derive every control quantity from the public observation.
- The physical four-wheel MJX-Warp car remains simulation truth. MPC uses a three-state Frenet
  kinematic model only inside its prediction problem and applies public action, rate, speed, and
  effective-track constraints.
- `benchmarks/v0.1/m6_controller_report.json` passed 34/34 gates at clean revision `add0a9a`.
  PID completed Level 0 and 10/10 Validation-prefix Tracks. MPC completed Level 0 and 95/100 full
  Validation Tracks; all five failures were timeouts.
- Combined MPC compute timing was 32.373/39.892/44.347 ms at P50/P95/P99 with a 0.0967% miss rate
  against the 50 ms soft deadline. PID P99 was 0.401 ms with no misses. Both passed the diagnostic
  real-time qualification; this is measured local evidence, not a platform support claim.
- Formal M6 used four batch-one MJX-Warp backends and 112 fresh Controller instances. All 234,358
  public transitions were finite, invalid-action count was zero, peak sampled process VRAM was
  396 MiB, and JAX live bytes returned to zero after each group.
- Validation rows were selected explicitly from verified pools while preserving row-index reset
  seeds. The report loaded only Level 0 and Validation; Test was not accessed.
- Environment teardown now severs instance-owned JIT, driver, and device-state references without
  process-global cache clearing. Public evaluation retains its original per-Track environment path
  when no reusable pool is supplied.
- Current local validation passes 547 CPU/default tests, all 23 GPU tests, strict documentation,
  official-asset verification, Actions linting, and release-package checks.

M6 proves that public classical Controllers can solve the Challenge and that the same evaluator can
produce fixed-order, source-bound timing and success evidence. It does not prove PPO learning.

## Current Narrow Focus

1. Add one strict PPO configuration and split-specific asset loaders. Training may read only the
   verified 10,000-Track Train cache; checkpoint selection may read Validation in a separate phase.
2. Build public observation/reward wrappers and a narrow JAX-to-Torch DLPack bridge. Preserve the
   public string `benchmark_version` info field instead of passing it to DLPack.
3. Implement a CleanRL-style actor/critic and NEXT_STEP-aware rollout/GAE logic. Reset-only world
   slots after terminal transitions must be excluded from learning and must break advantage
   recursion.
4. Train through one long-lived 1,024-world official `VecCarRacingEnv`, with local CSV/TensorBoard
   logging, atomic checkpoints, memory/timing/numerical evidence, and deterministic seeds.
5. Select a checkpoint on Validation without further gradient updates, compare it with a seeded
   random policy, and export a small inference-only checkpoint as `controllers/ppo`.
6. Run the exported plugin through the existing batch-one Evaluator, generate the required replay
   and manifest, document the method in English, and persist a strict M7 report.

## Scope Boundaries

In scope:

- one PyTorch PPO with an MLP policy/value network and state/local-track observation;
- public reward shaping and observation compression layered over the official Challenge;
- numeric JAX/Torch DLPack exchange without a second environment implementation;
- Train-only optimization, Validation-only selection, and comparison with random actions;
- 1,024-world smoke/full training, local artifacts, checkpoint Controller, replay, and evidence.

Out of scope:

- Test access or final Test evaluation before M8;
- a simplified PPO physics/Challenge path, CPU multiprocessing, or one Controller per GPU world;
- private Race Core indices, TrackPool rows, MJX state, or simulator objects as policy features;
- SAC/TD3, MPCC, perception, multi-car racing, sim-to-real, or broader backend abstractions;
- macOS, Windows, WSL2, multi-GPU, or public-release claims without later evidence.

## Confirmed Judgments

- PPO trains the exact `VecCarRacingEnv` used by formal evaluation, with only public observation and
  reward wrappers. There is no training-only transition function.
- Gymnasium's stock `JaxToTorch` conversion cannot handle the public NumPy string
  `benchmark_version`; M7 needs a small compatible wrapper that converts numeric leaves by DLPack
  and preserves the whitelisted string field.
- With Gymnasium NEXT_STEP autoreset, the call after a terminal transition is a reset-only slot for
  that world. PPO must mask it from rewards, GAE, advantage normalization, losses, and valid-sample
  counts while other worlds continue normally.
- The general all-split verifier is not a training loader because it reads Test. Train and
  Validation loaders must be explicit and guarded independently.
- PPO hyperparameters, feature/reward weights, and the formal training budget must first be
  measured on Train-only smoke runs and frozen before formal Validation selection.
- Test Tracks remain untouched until M8 formal evaluation.

## Next Step

Implement and test the strict PPO configuration plus a Train-only verified pool loader before
adding Torch wrappers or the optimization loop.
