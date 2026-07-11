# Active Context

Last updated: 2026-07-11

## Current Direction

Execute the sole authorized M8 replacement attempt, then finish release work. Attempt 001 loaded
the fixed Test pool but stopped during Environment creation before reset, step, Controller
construction, or performance observation. Attempt 002 retains the frozen comparison and adds only
pre-bind Warp initialization, predecessor lineage/eligibility gates, and disclosure. M0 through M7
are complete. Public documentation, cleanup, and repository publication remain pending.

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

1. Freeze, validate, commit, and push attempt 002 plus the canonical attempt 001 failure report.
   Prove the retained predecessor remains byte-identical and eligible before any new Test access.
2. Run attempt 002 exactly once over the same 20 fixed Test Tracks and persist the strict report
   and selected replay artifacts without tuning from Test results. A third attempt is forbidden.
3. Finish English public documentation, release/package/privacy audits, and the v0.1 cleanup
   checklist. Make the repository public only after all release gates pass.

The attempt 002 implementation passes the complete 1,086-test CPU suite, all 69 GPU tests, strict
documentation, package, and GitHub Actions checks. Independent red-team review found no unresolved
P0/P1 issue. The remaining accepted P2 threat boundary concerns hostile concurrent replacement of
intermediate parent directories; the trusted single-process release-maintainer model and CLI
symlink gates remain the declared v0.1 boundary.

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
- No Test performance has been observed. Attempt 001's Test load and zero-episode failure are
  disclosed; attempt 002 must be in a clean committed revision before it runs.
- Attempt 002 runs once across all three Controllers and 20 Tracks. Its post-bind failure must be
  retained and cannot be converted into a retry; no third formal attempt is allowed.
- Repository visibility remains private until final docs, privacy, package, evidence, and release
  checks pass.

## Next Step

Complete the final privacy/lineage/allowlist audit, commit and push attempt 002, then execute it
exactly once from that clean revision.
