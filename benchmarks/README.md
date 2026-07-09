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

M2 will produce `v0.1/gpu_report.json` after the required 1/64/256/1024-world measurements pass.
