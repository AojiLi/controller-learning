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
```

Installing the GPU environment only proves dependency and device availability. It is not evidence
for simulator throughput or numerical stability; those measurements are produced by M2.
