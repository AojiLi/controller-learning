# Evaluation Protocol

Controller Learning benchmark `0.1` compares trusted Controller plugins under one physical
four-wheel car, one Level 1 Challenge, and one versioned set of Tracks. Reward is useful for
training diagnostics, but it is never the ranking score.

For routine work on a new Controller, use the informal Level 0/Validation command documented in the
[Controller workflow](getting-started.md). It writes clearly labeled local CSV/JSON evidence and can
capture a trajectory from the measured rollout. It has no Test option and cannot replace the
accepted result described below.

The formal M8 Test result is published. Attempt 001 loaded the fixed Test pool, then stopped in
Environment creation before reset, stepping, Controller construction, or any performance
observation. The sole authorized replacement, attempt 002, completed the unchanged protocol and
passed all transaction and artifact-integrity gates.

## Benchmark 0.1 split contract

Benchmark `0.1` contains 10,000 Train Tracks, 100 Validation Tracks, and 20 Test Tracks generated
from the same Level 1 distribution. The splits use disjoint seed namespaces and geometry hashes.
Their roles are deliberately different:

| Split | Role | Controller decisions allowed |
| --- | --- | --- |
| Train | Optimization, reward design, and learning diagnostics | Training and iteration |
| Validation | Hyperparameter and checkpoint selection | Selection before Test |
| Test | One final comparison | Reporting only |

The Test geometry is public and packaged for reproducibility. Public availability does not make it
tuning data: a Controller, checkpoint, feature definition, or configuration changed after seeing
Test performance is not the benchmark `0.1` submission described here.

PID and MPC were frozen after their M6 Validation evidence. PPO was trained on Train, selected on
Validation, and exported as the frozen inference-only plugin documented in the
[PPO tutorial](ppo.md). The M8 comparison is an evaluation of those artifacts, not another
optimization phase.

## Fixed formal execution

The complete protocol is stored in `configs/final_evaluation.toml`. Its execution order is fixed:

1. PID on Test manifest rows 0 through 19.
2. MPC on Test manifest rows 0 through 19.
3. PPO on Test manifest rows 0 through 19.

One shared `CarRacingEnv` with `backend="mjx_warp"` and batch size one serves all 60 episodes. This
keeps the ordinary Controller plugin boundary identical for every algorithm. Native GPU batching
is used for PPO training and environment benchmarks; the final plugin comparison is intentionally
sequential and host-synchronized.

For Test row `i`:

- the Environment is reset with root seed `i` and fixed `track_index=i`;
- every Controller therefore receives the same Track and reset seed;
- episode and Controller seeds are independently derived with NumPy `SeedSequence`, world index
  zero, episode counter zero, and separate randomness domains; and
- the Runner imports the frozen plugin and creates a fresh Controller instance.

Because the derivation inputs are identical for a corresponding row, PID, MPC, and PPO receive the
same Controller seed on that row. The Controller seed remains distinct from the episode seed.

The row order, root seeds `0..19`, Controller order, Controller directories, full plugin file
identities, source revision, Pixi lock, input reports, and protocol configuration are bound before
Test is opened. A plugin cannot retain state between episodes or change Challenge settings.

Attempt 002 initializes Warp before the one-way Test binding. This eagerly performs the runtime's
platform probe while ordinary pre-Test process creation is still available; after binding, the
only permitted child process is the hash-bound `nvidia-smi` memory query. The change does not alter
the Environment, vehicle, Tracks, Controller boundary, episode order, seeds, metrics, or ranking.

Each episode starts from rest, runs at 20 Hz, and ends on success, off-track, invalid action, or the
Challenge timeout. The formal cap is 4,000 Controller steps. Controller initialization and each
complete `compute_control` call are timed separately; the per-step soft deadline is 50 ms.

An unexpected plugin, recorder, or callback exception invalidates attempt 002. The
transaction preserves a bounded, path-sanitized failure record and does not invent a terminal row,
continue with a different execution path, or retry automatically. Initialization above 30 seconds
is recorded as a soft diagnostic and does not by itself change ranking or protocol status.

## Ranking

There is no combined score. Controllers are ordered by:

1. success rate, descending; then
2. mean lap time across successful episodes, ascending.

If both values are exactly tied, the declared `PID -> MPC -> PPO` order provides a deterministic
final ordering. A Controller with no successful episode has no mean successful lap time. Reward,
tracking error, smoothness, and compute timing do not alter the ranking.

M8 has no minimum success-rate gate. The report may pass its protocol and integrity checks even if
a Controller performs poorly. This separation prevents an undesirable result from becoming a
reason to tune on Test or repeat the official attempt.

## Metric definitions

All dynamic metrics are reconstructed from the public observation, requested action, and Runner
timing captured during the canonical episode. They do not read MJX state, Race Core internals, or
private Track projection state.

One sample is recorded after every Environment transition:

- **Speed** is the Euclidean norm of the post-step body-frame longitudinal and lateral velocity.
- **Lateral error** is the signed distance from the post-step rear-axle position to the observed
  centerline. The first projection searches globally; later projections search four segments
  backward and twelve forward from the previous segment to respect Track topology.
- **Requested action** is the raw Controller request before the common actuator layer clips or
  rate-limits it.
- **Steering saturation** is true only when requested steering is strictly below `-0.6 rad` or
  strictly above `0.6 rad`. A request exactly on a bound is not saturated.
- **Longitudinal saturation** is true only when requested acceleration is strictly below
  `-8.0 m/s²` or strictly above `4.0 m/s²`.
- **Action smoothness** uses successive requested-action differences divided by `0.05 s`.
  The first action of each episode and cross-episode differences are excluded. The reported values
  are steering-rate RMS in `rad/s` and acceleration-rate RMS in `m/s³`.
- **Controller timing** stores each complete `compute_control` duration. P50, P95, and P99 use the
  raw samples. A deadline miss is a call strictly longer than `0.05 s`.

Speed, lateral-error, saturation, and timing aggregates are transition-weighted across all 20
episodes. Smoothness aggregates are weighted across valid within-episode action deltas. The report
also includes successful lap times, per-Track outcomes, transition counts, failure-reason counts,
maximum lateral error, and deadline-miss count and rate.

Each Controller summary reports a diagnostic real-time qualification: P99 must be at most `0.05 s`
and the deadline-miss rate must be at most `1%`. The boolean is published explicitly, but it is not
a protocol pass gate and does not alter the success/lap-time ranking.

Timing is a diagnostic, not a hard real-time guarantee. Python scheduling, host synchronization,
hardware, and solver behavior are part of the measured context, so the run manifest must accompany
the percentiles.

## Same-rollout replay rule

Replay selection is fixed before evaluation: Test row 0 is retained for PID, MPC, and PPO whether
the episode succeeds or fails. The evaluator records every canonical episode and keeps the row-0
public trajectory from that same measured rollout.

It does not run a replay Environment, repeat an episode, choose the first success, or replace a
trajectory because a later outcome looks better. This rule avoids both outcome cherry-picking and
false parity assumptions across repeated MJX-Warp rollouts.

## Published artifacts

Each Controller owns one directory:

```text
results/0.1/<controller>/m8-final-v0-1-002/
├── results.csv
├── summary.json
├── run_manifest.json
├── metrics.npz
├── trajectory.png
├── telemetry.png
└── selected_replays/
    └── test_row_000_trajectory.json
```

`results.csv` contains the fixed 20 per-Track rows. `summary.json` contains the aggregate outcome,
metric, and timing values. `run_manifest.json` binds the source, dependencies, hardware/software,
configuration, Controller files, seeds, Track identities, and artifact hashes.

`metrics.npz` is the canonical non-pickled sample artifact. It stores benchmark and Controller
metadata, ordered Track IDs and reset seeds, episode offsets, compute durations, post-step speed,
lateral error, raw requested actions, and both saturation flags. Episode offsets preserve the
boundary needed to exclude cross-episode action deltas.

`trajectory.png` and `telemetry.png` render the predeclared row-0 trajectory and samples. The JSON
beside them preserves the corresponding public trajectory. Three comparison artifacts live under
`benchmarks/v0.1/`: the strict M8 report, the Controller comparison CSV, and a row-0 path comparison
PNG. The formal publication allowlist contains exactly these 24 files.

## Attempt lineage and crash policy

Attempt 001 (`m8-final-v0-1-001`) crossed `TEST_BOUND` and loaded the verified Test pool. Its first
Environment creation triggered Warp's lazy platform probe, which attempted an unauthorized helper
process after the process guard was sealed. The guard rejected it. The retained transaction has an
empty episode journal, one canonical failure blob with `workload=null`, no reset or step, no
Controller construction, no execution-evidence seal, no staged artifacts, and no published
outputs. Consequently, it exposed no Controller outcome, success, lap time, or other performance
signal. The canonical
[attempt 001 failure report](https://github.com/AojiLi/controller-learning/blob/main/benchmarks/v0.1/m8_attempt_001_failure_report.json)
binds those facts, the retained transaction tree, the unchanged Controller identities, and the
replacement authorization. The final global JSON repeats this evidence in a strictly recomputed
`replacement_lineage` block, so its current-transaction attempt count cannot be read as the total
formal-attempt count.

The repository owner authorized exactly one zero-episode infrastructure replacement:
`m8-final-v0-1-002`. Its accepted result is the first complete attempt 002 result that passes the
otherwise unchanged frozen protocol and artifact-integrity checks. The only implementation changes
permitted by that authorization are pre-bind Warp initialization, replacement lineage evidence,
replacement eligibility gates, and replacement documentation. Controller performance is never a
retry condition, and a third attempt is forbidden.

Before Test access is durably bound, a failed preparation may restore prior outputs and restart.
After attempt 002's one-way `TEST_BOUND` transition:

- automatic retry is forbidden;
- an incomplete episode journal is preserved for explicit investigation rather than silently
  rerunning the Controllers;
- deterministic artifact construction may resume without another rollout only when all 60
  episode records and the typed post-close execution-evidence seal are both durable; a journal
  without that seal remains incomplete and non-retryable; and
- outputs become public only after the exact allowlist is staged and validated.

The transaction advances monotonically through `PREPARED`, `TEST_BOUND`,
`EVALUATION_COMPLETE`, `ARTIFACTS_VALIDATED`, and `COMMITTED`. Partial publication is restored from
durable backups. A successful process retains the ignored attempt 002 `COMMITTED` transaction as
recovery evidence and atomically moves the runtime Controller snapshot to an ignored quarantine;
it does not recursively delete that snapshot during the formal process. A crash is not permission
to tune, change a Controller, select a different checkpoint, or discard an unfavorable completed
result.

## Published result

The accepted run is `m8-final-v0-1-002`, bound to clean source
`609548199bf1872185d5f9dc5741f3b7795ce77e`.

| Rank | Controller | Success | Mean successful lap | Lateral RMS | Compute P99 | Deadline misses |
| ---: | --- | ---: | ---: | ---: | ---: | ---: |
| 1 | PID | 20/20 | 88.085 s | 0.0211 m | 0.340 ms | 0/35,234 |
| 2 | MPC | 20/20 | 102.563 s | 0.0381 m | 43.902 ms | 40/41,025 |
| 3 | PPO | 19/20 | 23.913 s | 0.2205 m | 0.281 ms | 0/9,615 |

| Controller | Detailed artifacts |
| --- | --- |
| PID | [summary](https://github.com/AojiLi/controller-learning/blob/main/results/0.1/pid/m8-final-v0-1-002/summary.json) · [per-Track CSV](https://github.com/AojiLi/controller-learning/blob/main/results/0.1/pid/m8-final-v0-1-002/results.csv) · [trajectory](https://raw.githubusercontent.com/AojiLi/controller-learning/main/results/0.1/pid/m8-final-v0-1-002/trajectory.png) · [telemetry](https://raw.githubusercontent.com/AojiLi/controller-learning/main/results/0.1/pid/m8-final-v0-1-002/telemetry.png) |
| MPC | [summary](https://github.com/AojiLi/controller-learning/blob/main/results/0.1/mpc/m8-final-v0-1-002/summary.json) · [per-Track CSV](https://github.com/AojiLi/controller-learning/blob/main/results/0.1/mpc/m8-final-v0-1-002/results.csv) · [trajectory](https://raw.githubusercontent.com/AojiLi/controller-learning/main/results/0.1/mpc/m8-final-v0-1-002/trajectory.png) · [telemetry](https://raw.githubusercontent.com/AojiLi/controller-learning/main/results/0.1/mpc/m8-final-v0-1-002/telemetry.png) |
| PPO | [summary](https://github.com/AojiLi/controller-learning/blob/main/results/0.1/ppo/m8-final-v0-1-002/summary.json) · [per-Track CSV](https://github.com/AojiLi/controller-learning/blob/main/results/0.1/ppo/m8-final-v0-1-002/results.csv) · [trajectory](https://raw.githubusercontent.com/AojiLi/controller-learning/main/results/0.1/ppo/m8-final-v0-1-002/trajectory.png) · [telemetry](https://raw.githubusercontent.com/AojiLi/controller-learning/main/results/0.1/ppo/m8-final-v0-1-002/telemetry.png) |

PPO's only failure was an off-track termination on Test row 14, Track ID `2000016`. It was faster
than PID and MPC on successful laps, but ranking considers success rate first. All three Controllers
passed the diagnostic real-time criterion. The run executed 85,874 Environment steps in 2,873.186
seconds with zero numerical failures, 360 MiB peak sampled process VRAM, and zero final JAX live
bytes.

The canonical machine-readable evidence is the
[global report](https://github.com/AojiLi/controller-learning/blob/main/benchmarks/v0.1/m8_final_evaluation_report.json)
and
[comparison CSV](https://github.com/AojiLi/controller-learning/blob/main/benchmarks/v0.1/m8_final_results.csv).
The [result interpretation](analysis.md) deterministically derives descriptive plots and paired
comparisons from the frozen CSV/NPZ artifacts without another Test execution.
The predeclared same-rollout Test row-0 comparison is shown below.

![Benchmark 0.1 canonical Test row 0 comparison](https://raw.githubusercontent.com/AojiLi/controller-learning/main/benchmarks/v0.1/m8_test_row_000_comparison.png)

The release-maintainer command remains available for reproduction:

```bash
pixi run -e gpu benchmark-m8-controllers
```

The published attempt 002 remains the accepted benchmark `0.1` result. A later invocation is a
reproduction attempt and cannot replace it. See [Reproducibility](reproducibility.md) for the
identity and artifact workflow.
