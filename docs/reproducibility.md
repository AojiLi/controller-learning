# Reproducibility

Controller Learning treats reproducibility as a chain of identities: source revision, `pixi.lock`,
configuration, Controller files, benchmark manifests, Track IDs, seeds, runtime inventory, raw
metric samples, and derived reports. A matching command without those identities is only a similar
experiment.

The formal Test result is published. Attempt 001 loaded Test but failed during Environment creation
before reset, stepping, Controller construction, or any performance observation. The sole
authorized replacement completed from clean source
`609548199bf1872185d5f9dc5741f3b7795ce77e`; the commands and artifact layout below document that
evidence chain.

## Supported platform

v0.1 supports Linux x86-64 and Python 3.11 through Pixi. The CPU development environment requires
glibc 2.28 or newer. Formal training, GPU tests, and benchmark evaluation require an NVIDIA GPU and
the locked GPU Pixi environment.

macOS, native Windows, and WSL2 are not supported or tested for v0.1. The project does not claim
that MJX-Warp, GPU throughput, numerical behavior, or the formal evaluator works on those systems.
Docker, Conda, Poetry, and a second dependency lock are not supported setup paths.

## CPU development workflow

Install the default environment and run the same complete CPU validation used by GitHub Actions:

```bash
pixi install
pixi run ci
```

For a narrower edit loop:

```bash
pixi run format-check
pixi run lint
pixi run tests
pixi run docs
```

Run a deterministic Level 0 development episode with the template or an example Controller:

```bash
pixi run sim
pixi run sim -- --controller controllers/pid --level-id 0 --render
pixi run sim -- --controller controllers/mpc --level-id 0 --render
```

The CPU backend is for development and bounded consistency checks. A CPU episode is not a formal
benchmark result, and CPU multiprocessing is not a substitute for the native GPU batching
requirement.

`pixi run verify-track-assets` verifies all committed official manifests and packaged fixed assets.
That routine may read and hash the Test files, but it does not create a Test Environment, execute a
Controller, or observe Test performance. Keep this distinction explicit when recording access.

## NVIDIA GPU workflow

Install and verify the separate GPU environment:

```bash
pixi install -e gpu
pixi run -e gpu gpu-check
pixi run -e gpu gpu-tests
```

`gpu-check` proves only that dependencies and one MJX-Warp step are available. `gpu-tests` covers
the local GPU test suite. Neither command reproduces a versioned benchmark report by itself.

Existing formal workflows include:

```bash
pixi run -e gpu benchmark-gpu
pixi run -e gpu benchmark-racing-env
pixi run -e gpu benchmark-track-pool
pixi run -e gpu benchmark-controllers
pixi run -e gpu benchmark-m7-ppo-controller
```

Read each command's versioned report before comparing numbers. Native batched throughput and the
sequential batch-one Controller evaluator measure different paths and must not be compared as if
they were the same workload.

The M8 release-maintainer task is:

```bash
pixi run -e gpu benchmark-m8-controllers
```

Attempt 002 is the accepted result and the sole authorized zero-episode infrastructure
replacement. Its transaction is `COMMITTED` with 60 journal records, 60 trajectory blobs, a typed
post-close execution seal, and 24 semantically validated outputs. A local invocation is now a
reproduction attempt; it cannot replace the accepted result, and no third official attempt is
allowed.

## Author a Controller without Test leakage

Create a new trusted directory plugin from the template:

```bash
cp -R controllers/template controllers/my_controller
```

Then:

1. keep the concrete `Controller` subclass in `controller.py` and algorithm settings in
   `config.toml`;
2. use only public observations, restricted info, immutable public config, and write-only
   `DebugDraw`;
3. create all mutable algorithm state inside each fresh Controller instance;
4. develop on Level 0 and generated Level 1 Train-namespace seeds;
5. use Train for optimization and Validation for model or parameter selection; and
6. freeze code, config, dependencies, and learned assets before any final Test evaluation.

Run the new plugin on development Tracks:

```bash
pixi run sim -- --controller controllers/my_controller --level-id 0 --render
pixi run sim -- --controller controllers/my_controller --level-id 1 --track-seed 42
pixi run evaluate-controller -- \
  --controller controllers/my_controller \
  --run-id my-validation-10 \
  --split validation \
  --count 10 \
  --capture-row 0
pixi run tests
```

The informal evaluator uses deterministic manifest-order rows and reset seeds, labels CPU and
MJX-Warp development results separately, and exposes no Test option. Its selected trajectory is
captured from the measured rollout and can be inspected with `pixi run replay`; see the
[Controller workflow](getting-started.md).

The Controller may implement PID, MPC, RL, or another method internally, including a simplified
prediction model. The official simulation truth remains the same physical four-wheel car. Do not
read Environment, TrackPool, Race Core, MJX, renderer, or simulator internals from the plugin.

Do not repeatedly evaluate the packaged Test rows while developing. Do not use Test outcomes to
choose gains, features, rewards, architectures, checkpoints, fallbacks, or seeds. If Test informs a
later design, report it as a new experiment or benchmark version rather than overwriting benchmark
`0.1` evidence.

## Reproduce the M8 protocol identity

For a source-matched reproduction, begin from the accepted source revision and record a clean
state:

```bash
git status --short
git rev-parse HEAD
pixi run ci
pixi run -e gpu gpu-check
pixi run -e gpu gpu-tests
```

The formal evaluator additionally binds:

- `configs/final_evaluation.toml` and `pixi.lock` hashes;
- every declared file in `controllers/pid`, `controllers/mpc`, and `controllers/ppo`;
- the M5 Track-admission, M6 Controller, M7 PPO, and canonical M8 attempt 001 failure reports,
  including the exported policy identity and zero-episode predecessor transaction;
- the Test manifest and fixed asset identity after the one-way Test transition; and
- OS, CPU, GPU model, driver, CUDA, Python, JAX, MuJoCo/MJX-Warp, Warp, CasADi, and PyTorch
  versions relevant to the run.

The evaluator uses one shared MJX-Warp batch-one Environment for all 60 canonical episodes, fixed
Controller-major order `PID -> MPC -> PPO`, Test rows and reset seeds `0..19`, and a fresh plugin
instance for each episode. See [Evaluation Protocol](evaluation.md) for ranking, metrics, replay,
and crash policy.

The accepted global report has SHA-256
`4867035ceb7532f3dca190c344e9d2cc82b3aa3c3098d2f7169a5acb75c49e69`. Verify the committed bytes
before comparing a reproduction:

```bash
sha256sum benchmarks/v0.1/m8_final_evaluation_report.json
```

## Inspect published artifacts

For each Controller, begin with these files:

- `results.csv` for the ordered 20 per-Track outcomes;
- `summary.json` for success, successful-lap, metric, failure, and timing aggregates;
- `run_manifest.json` for provenance and artifact identities; and
- `metrics.npz` for the canonical transition samples used to recompute aggregates.

Load the NPZ without pickle support:

```python
from pathlib import Path

import numpy as np

path = Path("results/0.1/pid/m8-final-v0-1-002/metrics.npz")
with np.load(path, allow_pickle=False) as metrics:
    print(metrics.files)
    print(metrics["track_id"])
    print(metrics["episode_offsets"])
```

The artifact stores fixed metadata plus `compute_time_s`, `speed_mps`, `lateral_error_m`,
`requested_action`, `steering_saturated`, and `longitudinal_saturated`. Use `episode_offsets` when
recomputing smoothness so no difference crosses an episode boundary.

Accept a formal result only when the strict M8 report, all three Controller directories, and the
three central comparison artifacts are present; the report status and integrity gates pass; and
the hashes, Track order, seeds, Controller identities, summaries, CSV rows, NPZ samples, plots, and
same-rollout row-0 trajectories agree. Performance values without that evidence chain are not the
published benchmark.

## Numerical reproducibility boundary

Fixed inputs make the experiment auditable, but they do not promise bit-identical closed-loop
trajectories across repeated GPU runs or different machines. MJX-Warp contact and constraint
atomics can introduce small ordering differences that compound over a long episode. Preserve the
exact hardware/software inventory, compare protocol-level outcomes and toleranced numerical gates,
and never regenerate a preferred official result by repeating Test.
