# Development

## CPU Workflow

```bash
pixi install
pixi run format-check
pixi run lint
pixi run tests
pixi run docs
pixi run actions-lint
pixi run package-check
```

`pixi run ci` executes the complete CPU validation used by GitHub Actions.

## M1 Vehicle Workflow

Run the deterministic CPU vehicle tests during development:

```bash
pixi run pytest tests/unit/physics tests/integration/physics
```

Open the optional MuJoCo viewer on a machine with a display:

```bash
pixi run view-cpu-vehicle -- --scenario demo --duration 12
```

Generate the formal CPU timestep report only from a clean Git worktree:

```bash
pixi run benchmark-cpu-vehicle
```

The benchmark still runs in a dirty worktree for diagnosis, but marks the report invalid and exits
with a failure. See [CPU Vehicle](vehicle.md) for the physical contract and gate definitions.

## GPU Environment

```bash
pixi install -e gpu
pixi run -e gpu gpu-check
pixi run -e gpu gpu-tests
```

Installing the GPU environment only proves dependency and device availability. It is not evidence
for simulator throughput or numerical stability.

Run the formal isolated M2 protocol only from a clean Git worktree:

```bash
pixi run -e gpu benchmark-gpu
```

The command launches a fresh process for every scale and writes
`benchmarks/v0.1/gpu_report.json`. It exits unsuccessfully if the worktree or benchmark sources
change during the run, any numerical or physical gate fails, memory does not stabilize, or the
CPU/GPU comparison exceeds its tolerances. See [MJX-Warp GPU Vehicle](gpu.md) for the frozen protocol
and reviewed measurements.

## M3 Track Capacity Workflow

Run the deterministic offline capacity sweep in the default environment:

```bash
pixi run benchmark-track-capacity
```

The command generates one candidate per seed for 10,000 contiguous seeds at each configured arc
spacing and writes `benchmarks/v0.1/track_capacity_report.json`. It records generation and validation
rejections, distribution percentiles, reproducibility samples, theoretical capacity bounds, and
runtime-array memory estimates. The reviewed result locks 1.0 m arc spacing, 640 track points, and 48
checkpoints.

## M3 Race Core and Driveability Workflow

The complete local GPU suite includes the vehicle, 1,024-track Race Core, masked-reset isolation,
and generated-track driveability tests:

```bash
pixi run -e gpu gpu-tests
```

Run the formal low-speed admission protocol separately:

```bash
pixi run -e gpu validate-track-driveability
```

The command uses the formal MJX-Warp four-wheel backend, the production Race Core, and a private
conservative 4 m/s reference policy. It writes
`benchmarks/v0.1/track_driveability_report.json` and fails if any accepted track does not complete a
lap or if a numerical, contact-capacity, unexpected-contact, invalid-action, off-track, or timeout
gate fails. This reference policy is an offline track-admission tool, not a public Controller or a
performance baseline. See [Tracks and Race Core](tracks.md) for the reviewed measurements.

## M4 Environment and Controller Workflow

Run one complete template Controller episode through the single-world Gymnasium adapter:

```bash
pixi run sim
```

The template is deliberately neutral and reaches the Challenge timeout. Add `--render` for the 2D
public-observation renderer, or select the formal backend from the GPU environment:

```bash
pixi run -e gpu sim -- --backend mjx_warp
```

The CPU suite runs the Gymnasium checker, batch-one agreement, restricted info/config, plugin
isolation, fresh Controller lifecycle, and renderer integration. The GPU suite adds native batched
steps, mixed-world NEXT_STEP autoreset, and transfer guards that reject any host/device transfer in
the warmed JAX action path.

Run the formal 1,024-world wrapper benchmark only from a clean relevant-source worktree:

```bash
pixi run -e gpu benchmark-racing-env
```

The default protocol executes 10,000 timed environment steps without per-step host synchronization,
then performs separate transfer, timeout/autoreset, public numerical-health, source, and memory
checks. It writes `benchmarks/v0.1/m4_environment_report.json`.

The reviewed 1,024-world run completed 10,240,000 transitions at 165,633 transitions/s. Both
transfer guards passed, every world timed out and autoreset independently in the separate health
run, no non-finite public value was observed, and steady process VRAM grew by 10 MiB against a
64 MiB gate. See [Gymnasium and Controller Platform](environment.md) for the full contract.

## M5 Official Track Assets

The normal consumer workflow verifies the committed fixed assets and materializes the ignored
Train cache from its published manifest:

```bash
pixi run verify-track-assets
pixi run materialize-track-pool
pixi run verify-track-assets -- --require-train-cache
```

The formal asset-admission and full-pool GPU workflows are retained for source-matched
reproduction, not routine setup:

```bash
pixi run -e gpu build-track-assets
pixi run -e gpu benchmark-track-pool
```

See [Tracks and Race Core](tracks.md) for the fixed split contract and reviewed M5 evidence.

## M6 Example Controllers

Develop PID and MPC through the same public simulation entry point used by a new plugin:

```bash
pixi run sim -- --controller controllers/pid --level-id 0 --render
pixi run sim -- --controller controllers/mpc --level-id 0 --render
```

Measure any trusted plugin on Level 0 or an ordered Validation prefix, then render the selected
same-rollout trajectory without another simulation:

```bash
pixi run evaluate-controller -- \
  --controller controllers/pid \
  --run-id pid-validation-10 \
  --split validation \
  --count 10 \
  --capture-row 0
pixi run replay -- \
  runs/evaluations/pid-validation-10/selected_replays/row_000_trajectory.json \
  --overview runs/evaluations/pid-validation-10/overview.png
```

These outputs are informal development evidence and cannot access or replace Test. The complete
workflow and output contracts are documented in [Controller Workflow](getting-started.md).

The retained formal Validation workflow is `pixi run -e gpu benchmark-controllers`. Its accepted
report is already published; a later invocation is a reproduction. See
[Classical Controllers](controllers.md) for the Controller API, design, and timing evidence.

## M7 PPO Training and Export

The four formal stages are deliberately separate:

```bash
pixi run -e gpu train-ppo -- --run-id my-ppo-run
pixi run -e gpu benchmark-m7-ppo
pixi run -e gpu export-m7-ppo-controller
pixi run -e gpu benchmark-m7-ppo-controller
```

`--run-id` is required and must be a new lowercase local identifier. Training also requires a clean
source revision; the accepted historical run ID is evidence, not a destination for new output.

They implement Train-only optimization, one frozen Validation selection, hash-bound NumPy export,
and ordinary Controller evaluation. The accepted artifacts are already published and must not be
reselected from Test evidence. See [PPO Training and Export](ppo.md).

## M8 Accepted Test Evaluation

The release-maintainer task remains available for exact-protocol reproduction:

```bash
pixi run -e gpu benchmark-m8-controllers
```

Attempt 002 is the accepted benchmark `0.1` result. A later invocation cannot replace it, and no
third official attempt is allowed. Read [Evaluation Protocol](evaluation.md) and
[Reproducibility](reproducibility.md) before using the command; routine Controller development must
stay on Train and Validation.
