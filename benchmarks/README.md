# Benchmark Evidence

Only reviewed, versioned benchmark evidence belongs here. Ad-hoc local output goes in
`benchmarks/local/`, which is ignored by Git.

## M1 CPU Vehicle

`v0.1/m1_cpu_report.json` records the deterministic 100/200/500 Hz CPU scan, scenario checks,
physics-substep contact diagnostics, convergence against the 500 Hz reference, runtime versions,
source/config/model/lock hashes, Git revision, and selected M2 candidate timestep. Generate it with
`pixi run benchmark-cpu-vehicle` from a clean worktree. A dirty worktree can never produce a report
that is marked ready for M2.

## M2 GPU Vehicle

`v0.1/gpu_report.json` records the passed 1/64/256/1024-world MJX-Warp scale sweep, the
1,024-world × 10,000-step endurance run, numerical/contact agreement, throughput, memory, capacity,
runtime, and source gates.

## M3 Tracks and Race Core

- `v0.1/track_capacity_report.json` records the 10,000-seed representation sweep that selected
  1.0 m spacing, 640 points, and 48 checkpoints.
- `v0.1/track_driveability_report.json` records the formal four-wheel low-speed admission check.

## M4 Environment

`v0.1/m4_environment_report.json` records the 1,024-world official Challenge run, transfer guards,
autoreset behavior, numerical health, throughput, and memory evidence.

## M5 Official Track Assets

- `v0.1/m5_track_admission_report.json` binds the fixed Level 0, Train, Validation, and Test
  manifests/assets to their deterministic admission process.
- `v0.1/m5_track_pool_report.json` records full Train-pool GPU sampling, reset, transfer,
  throughput, and allocator-stability evidence.

## M6 Classical Controllers

`v0.1/m6_controller_report.json` records PID/MPC Level 0 and Validation results, Controller timing,
memory, public-interface integrity, and source gates.

## M7 PPO

- `v0.1/m7_ppo_selection_report.json` records Train-only optimization and frozen Validation
  selection.
- `v0.1/m7_ppo_export_report.json` binds the selected inference-only NumPy policy.
- `v0.1/m7_ppo_controller_evaluation_report.json` records the ordinary Controller Runner result.
- `v0.1/m7_training_curve.png` and `v0.1/m7_ppo_replay_overview.png` are reviewed presentation
  artifacts derived from those workflows.

## M8 Final Evaluation

- `v0.1/m8_attempt_001_failure_report.json` discloses the retained zero-episode infrastructure
  failure and the exact one-replacement authorization.
- `v0.1/m8_final_evaluation_report.json` is the canonical accepted Test report.
- `v0.1/m8_final_results.csv` is the compact PID/MPC/PPO comparison.
- `v0.1/m8_test_row_000_comparison.png` compares the predeclared same-rollout row-0 trajectories.

The accepted run is `m8-final-v0-1-002` at source revision `6095481`: PID completed 20/20 Test
Tracks, MPC 20/20, and PPO 19/20. Per-Controller CSV, summary, manifest, metrics, and replay files
live under `results/0.1/<controller>/m8-final-v0-1-002/`.

Do not place local diagnostics, private paths, raw run directories, or unreviewed benchmark output
in this directory.
