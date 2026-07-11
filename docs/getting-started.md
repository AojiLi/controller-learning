# Controller Workflow

This page closes the everyday Controller-development loop: copy a plugin, simulate one Track,
evaluate an ordered development set, and inspect the exact measured rollout. None of these commands
can access benchmark Test.

## Prerequisites

The supported v0.1.x setup is Linux x86-64 with glibc 2.28 or newer. Pixi installs the locked Python
3.11 environment and is the only supported dependency workflow.

```bash
git clone https://github.com/AojiLi/controller-learning.git
cd controller-learning
pixi install
```

Run the narrow CPU checks after setup:

```bash
pixi run format-check
pixi run lint
pixi run tests
```

`cpu_reference` is the development/reference backend. Formal training and benchmark evidence use
MJX-Warp on an NVIDIA GPU; install that environment separately only when needed:

```bash
pixi install -e gpu
pixi run -e gpu gpu-check
```

## 1. Copy the template

```bash
cp -R controllers/template controllers/my_controller
```

A plugin directory requires `controller.py` and `config.toml`. It may add relative helper modules
and local assets. The loader executes trusted local Python; this is an interface boundary, not a
security sandbox.

Implement exactly one concrete `Controller` subclass. The lifecycle receives public observations,
restricted info, immutable Challenge configuration plus `config["controller"]`, and write-only
`DebugDraw`. It never receives an Environment, TrackPool, physics state, or simulator object. See
[Gymnasium and Controller Platform](environment.md) for every field and method.

## 2. Simulate one Track

Begin with the deterministic Level 0 ellipse:

```bash
pixi run sim -- --controller controllers/my_controller --level-id 0 --render
```

Then use exact procedural Level 1 seeds for repeatable development:

```bash
pixi run sim -- \
  --controller controllers/my_controller \
  --level-id 1 \
  --track-seed 42
```

`--render` requires a graphical display. The JSON result still reports the explicit backend,
Track identity, seeds, terminal reason, steps, lap time, and reward when rendering is disabled.

## 3. Evaluate without Test

`evaluate-controller` is an explicitly informal development command. It accepts only `level0` and
`validation`; it cannot load Train or Test through its split-specific loaders.

Measure Level 0 and capture row 0 from that same rollout:

```bash
pixi run evaluate-controller -- \
  --controller controllers/my_controller \
  --run-id my-level0 \
  --split level0 \
  --capture-row 0
```

Measure the first ten fixed Validation rows in manifest order:

```bash
pixi run evaluate-controller -- \
  --controller controllers/my_controller \
  --run-id my-validation-10 \
  --split validation \
  --count 10 \
  --capture-row 0
```

Run IDs are lowercase, path-safe, and never overwritten. Rows use deterministic reset seeds
`0..N-1`. The command records the full Controller-directory identity and a path-sanitized Git
source identity before evaluation, then requires both to remain unchanged.

Output is written transactionally under the ignored directory:

```text
runs/evaluations/<run-id>/
├── episodes.csv
├── summary.json
└── selected_replays/
    └── row_NNN_trajectory.json  # only with --capture-row
```

`summary.json` labels the run `informal_development_evaluation`, sets
`formal_benchmark_result=false`, records the backend scope and Track-source hashes, and reports
ordered success, successful-lap, and Controller-timing aggregates. A selected trajectory comes
from the measured evaluation rollout; the command never executes a second episode for replay.

For an informal MJX-Warp development measurement:

```bash
pixi run -e gpu evaluate-controller -- \
  --controller controllers/my_controller \
  --run-id my-validation-gpu-10 \
  --split validation \
  --count 10 \
  --backend mjx_warp
```

CPU and MJX-Warp output is labeled separately. An informal result is not comparable to or a
replacement for the accepted benchmark `0.1` Test result.

## 4. Replay the measured trajectory

Create a deterministic 800×600 top-down PNG without loading an Environment:

```bash
pixi run replay -- \
  runs/evaluations/my-level0/selected_replays/row_000_trajectory.json \
  --overview runs/evaluations/my-level0/overview.png
```

Open interactive playback at four times the recorded display rate:

```bash
pixi run replay -- \
  runs/evaluations/my-level0/selected_replays/row_000_trajectory.json \
  --play \
  --speed 4
```

The loader rejects symbolic links, oversized files, non-canonical JSON, schema violations, and
hash-inconsistent reconstruction. PNG output refuses every existing destination. Interactive
playback reconstructs each frame solely from the stored public observation and requires a graphical
display; it does not rerun vehicle dynamics or Controller code.

The same command can inspect the published canonical trajectories under
`results/0.1/<controller>/m8-final-v0-1-002/selected_replays/`. Reading an accepted artifact does not
change it and does not execute Test again.

## Split discipline

| Split | Development use | Available in `evaluate-controller` |
| --- | --- | --- |
| Level 0 | API and basic closed-loop debugging | Yes |
| Train | Optimization and learning | No; PPO uses the dedicated training path |
| Validation | Controller and checkpoint selection | Yes |
| Test | Frozen final reporting | No |

Do not change a Controller or select a checkpoint based on the published Test outcome and then
describe it as the accepted benchmark `0.1` submission. Read [Reproducibility](reproducibility.md)
and the [Stability Policy](stability.md) before publishing comparable work.
