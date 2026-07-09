# Development

## CPU Workflow

```bash
pixi install
pixi run format-check
pixi run lint
pixi run tests
pixi run docs
```

`pixi run ci` executes the complete CPU validation used by GitHub Actions.

## GPU Environment

```bash
pixi install -e gpu
pixi run -e gpu gpu-check
```

Installing the GPU environment only proves dependency and device availability. It is not evidence
for simulator throughput or numerical stability; those measurements are produced by M2.
