# Active Context

Last updated: 2026-07-11

## Current Direction

Build v0.1.1 as a maintenance release that closes the public Controller-author workflow and improves
the portfolio presentation without changing benchmark `0.1`. The accepted M8 result, Test assets,
formal configuration, and PID/MPC/PPO Controller identities are frozen. No formal Test execution is
permitted during this work.

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

The v0.1.1 implementation is complete: informal Level 0/Validation evaluation, same-rollout
trajectory capture, strict offline replay, hash-pinned M8 interpretation, README restructuring,
Pages configuration, contribution/citation metadata, and the public stability policy are present.
The current work is release verification:

1. run the complete CPU and GPU checks without formal Test;
2. prove every v0.1.0 Controller and accepted M8 artifact identity is unchanged; and
3. publish only after GitHub CPU CI and Pages deployment are green, then verify v0.1.1 from an
   anonymous fresh checkout.

The v0.1.0 baseline passes the complete 1,086-test local CPU suite, all 69 GPU tests,
strict documentation/package checks, and GitHub Actions syntax validation. The public release
commit also passed clean-checkout GitHub CPU CI. Independent red-team review found no unresolved
protocol P0/P1 issue. The remaining accepted P2 threat boundary concerns hostile concurrent
replacement of intermediate parent directories; the trusted single-process release-maintainer
model and CLI symlink gates remain the declared v0.1 boundary.

The attempt 002 boundary installs a Test-only audit guard before project imports, captures a
read-only hash-bound snapshot of every Controller, uses one environment for the fixed 60-episode
order, fsyncs each canonical trajectory/journal pair, and requires a typed post-close execution
seal before deterministic artifact construction can recover. Exactly 24 outputs must pass semantic
recomputation before transactional publication. The durable `COMMITTED` transaction is retained,
and the runtime Controller snapshot is atomically quarantined under ignored `runs/` after
publication so an interrupted cleanup remains recoverable without rerunning Test. Attempt 002
initializes Warp before the one-way Test binding; Test-pool loading then closes all Track reads and
all process creation except the fixed `nvidia-smi` VRAM query.

Attempt 001 retained `TEST_BOUND`, a 0/60 journal, null execution evidence, and exactly one
sanitized `environment_create` failure with `workload=null`. It loaded Test geometry but did not
create an environment, reset, step, instantiate a Controller, or observe performance. Its
transaction and original Controller snapshot remain read-only. The canonical failure report binds
their hashes and authorizes only attempt 002.

Attempt 002 completed 60/60 durable episodes from clean source `6095481`: PID succeeded on 20/20
Test Tracks, MPC on 20/20, and PPO on 19/20. The accepted ranking is PID, MPC, PPO. The run used
85,874 Environment steps over 2,873.186 seconds, recorded zero numerical failures, peaked at
360 MiB sampled process VRAM, and returned JAX live bytes to zero. Its transaction contains 60
journal rows, 60 trajectory blobs, a typed execution seal, 24 exact outputs, semantic validation,
and durable `COMMITTED` state. No further official Test attempt is permitted.

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
- Attempt 002 is the accepted benchmark `0.1` result. Later invocations are reproductions and may
  not replace it; no third official attempt is allowed.
- Repository visibility is public; the v0.1 documentation, privacy, package, evidence, and release
  checks passed before the visibility change.

## Next Step

Run the complete local CPU/GPU, privacy, package, and protected-identity verification suite. Do not
invoke the formal M8 command.
