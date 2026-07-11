# Evaluation Protocol

Controller Learning benchmark `0.1` compares trusted Controller plugins under one physical
four-wheel car, one Level 1 Challenge, and one versioned set of Tracks. Reward is useful for
training diagnostics, but it is never the ranking score.

The formal M8 Test result is still pending. This page documents the protocol before the Test run;
it does not contain or imply PID, MPC, or PPO Test performance.

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

Each episode starts from rest, runs at 20 Hz, and ends on success, off-track, invalid action, or the
Challenge timeout. The formal cap is 4,000 Controller steps. Controller initialization and each
complete `compute_control` call are timed separately; the per-step soft deadline is 50 ms.

An unexpected plugin, recorder, or callback exception invalidates the one-shot attempt. The
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
results/0.1/<controller>/m8-final-v0-1-001/
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

## One-shot attempt and crash policy

The official result is the first complete attempt that passes the frozen protocol and artifact
integrity checks. Controller performance is never a retry condition.

Before Test access is durably bound, a failed preparation may restore prior outputs and restart.
After the one-way `TEST_BOUND` transition:

- automatic retry is forbidden;
- an incomplete episode journal is preserved for explicit investigation rather than silently
  rerunning the Controllers;
- deterministic artifact construction may resume without another rollout only when all 60
  episode records and the typed post-close execution-evidence seal are both durable; a journal
  without that seal remains incomplete and non-retryable; and
- outputs become public only after the exact allowlist is staged and validated.

The transaction advances monotonically through `PREPARED`, `TEST_BOUND`,
`EVALUATION_COMPLETE`, `ARTIFACTS_VALIDATED`, and `COMMITTED`. Partial publication is restored from
durable backups. A successful process retains the ignored `COMMITTED` transaction as recovery
evidence and atomically moves the runtime Controller snapshot to an ignored quarantine; it does not
recursively delete that snapshot during the formal process. A crash is not permission to tune,
change a Controller, select a different checkpoint, or discard an unfavorable completed result.

## Current status

The formal Test comparison has not run, and no Test success rate or lap time is currently claimed.
The release-maintainer command is:

```bash
pixi run -e gpu benchmark-m8-controllers
```

It is present in the locked Pixi task set, but it must be invoked only from the clean committed
protocol revision. A result is official only after the strict report and all 24 artifacts pass the
publication gates. See [Reproducibility](reproducibility.md) for the development and release
workflows.
