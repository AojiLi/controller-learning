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
