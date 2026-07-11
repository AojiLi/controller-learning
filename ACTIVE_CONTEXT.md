# Active Context

Last updated: 2026-07-11

## Current Direction

Execute M8: freeze and commit the Test-only final evaluation protocol before any Test performance
access, then run one formal fixed-order 20-Track comparison of PID, MPC, and PPO. M0 through M7 are
complete. Public documentation, cleanup, and repository publication remain pending and must not be
claimed before the v0.1 release checklist passes.

## M7 Handoff Evidence

- Formal PPO training used clean source `86f8f384`, one long-lived 1,024-world official
  `VecCarRacingEnv`, 80 updates, and 10,466,653 valid transitions. End-to-end throughput was
  56,245.788 valid transitions/s, peak sampled process VRAM was 1,180 MiB, and no numerical error
  was recorded.
- The retained frozen candidates were updates `[10, 20, 30, 40, 50, 60, 70, 80]`. A single formal
  Validation selection chose update 70 at 95/100 successes; the seeded random baseline achieved
  0/100. No candidate received further gradient updates.
- The selected inference-only NumPy Controller policy has SHA-256
  `f3054e95c6d357f571425ad69b9ac16c713e24b9f09b7768e7a648af84731a4b`.
- At clean source `1b434f4`, the ordinary Controller path completed 99/100 Validation Tracks with a
  24.316667 s mean successful lap time over 48,709 environment steps. Compute timing was
  0.260/0.305/0.332 ms at P50/P95/P99, with zero 50 ms deadline misses, 364 MiB peak sampled
  process VRAM, and zero final JAX live bytes.
- Replay protocol v2 captured the fixed-order row-0 success inline from the same evaluation
  trajectory. It did not cherry-pick or execute a second rollout. The preceding formal v1 attempt
  failed because MJX-Warp atomics were not rollout-bit-deterministic and fully rolled back before
  the v2 protocol was frozen.
- Before formal selection, one capacity-only diagnostic loaded Validation to inspect fixed shape;
  it created no environment, ran no policy, and observed no performance. The formal selection's own
  pre-Validation access count was zero.
- M7 performance workflows did not access Test. Routine official-asset verification may hash Test,
  but it does not instantiate a Test environment, execute a Controller, or reveal performance.

M7 proves end-to-end PPO learning, frozen Validation selection, portable inference-only export, and
ordinary Controller evaluation/replay. It does not provide final Test comparison or release proof.

## Current Narrow Focus

1. Specify the one-shot M8 Test protocol completely: fixed 20-Track order, reset and Controller
   seeds, PID/MPC/PPO artifact identities, environment lifecycle, metrics, timing, replay selection,
   memory/numerical gates, output paths, and rollback behavior.
2. Add validators and tests that prove the protocol rejects dirty source, configuration drift,
   split leakage, reordered rows, Controller mutation, incomplete metrics, and partial publication.
3. Freeze and commit the protocol plus configuration while Test performance remains unopened.
4. Run one formal PID/MPC/PPO evaluation over the same 20 fixed Test Tracks and persist the strict
   report and selected replay artifacts without tuning from Test results.
5. Finish English public documentation, release/package/privacy audits, and the v0.1 cleanup
   checklist. Make the repository public only after all release gates pass.

## Scope Boundaries

In scope:

- one frozen, source-bound Test-only protocol for PID, MPC, and PPO;
- the same fixed 20 Test Tracks, ordering, seeds, public Controller boundary, and formal MJX-Warp
  backend for all three Controllers;
- strict success/lap/error/action/timing/failure metrics plus selected 2D replay artifacts;
- English tutorials/API/reproduction documentation, package and privacy audit, and public-release
  readiness evidence.

Out of scope:

- any Test-informed tuning, checkpoint selection, Controller/config change, or rerun chosen from
  performance results;
- changing benchmark `0.1`, the physical simulation truth, or the public Controller boundary;
- SAC/TD3, MPCC, perception, multi-car racing, sim-to-real, or broader backend abstractions;
- macOS, Windows, WSL2, multi-GPU, or support claims without corresponding evidence.

## Confirmed Judgments

- The final comparison is evidence for the already frozen Controllers, not a new optimization or
  selection phase. Test results may be reported but may not feed back into v0.1 Controller changes.
- PID, MPC, and PPO must use the ordinary Controller interface and the same formal four-wheel
  MJX-Warp Challenge; no Controller-specific environment path is permitted.
- Test performance remains unopened until the complete protocol and its rejection tests are in a
  clean committed revision.
- The final Test protocol runs once across all three Controllers and 20 Tracks. Any infrastructure
  failure must be reported and handled by the predeclared policy, not silently converted into a
  performance-motivated rerun.
- Repository visibility remains private until final docs, privacy, package, evidence, and release
  checks pass.

## Next Step

Freeze, test, and commit the Test-only M8 final evaluation protocol before opening Test for any
Controller performance run; then execute one formal PID/MPC/PPO 20-Track comparison.
