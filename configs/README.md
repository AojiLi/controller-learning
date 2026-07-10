# Configuration

Challenge, vehicle, and benchmark configuration are intentionally separate. Controller-specific
configuration lives under `controllers/<name>/config.toml` and cannot override these files.

M1 and M2 locked the vehicle timestep, actuator mapping, and formal GPU physics path from measured
CPU/GPU evidence. M3 locked `track.toml`: fixed 1.0 m arc-length resolution, 640-point/48-checkpoint
capacity, deterministic generator inputs, validation limits, and topology-local race rules. Track
width remains a Level rule in `levels/`; it is intentionally not duplicated in `track.toml`.

M5 completed the Level and asset contract. `levels/level0.toml` selects one fixed smooth ellipse
with reserved uint32 Track ID `UINT32_MAX`. `levels/level1.toml` selects procedurally generated
geometry with three immutable namespaces: Train has 10,000 Tracks from `[0, 1,000,000)`, Validation
has 100 from `[1,000,000, 2,000,000)`, and Test has 20 from `[2,000,000, 3,000,000)`. The stable
numeric identity is `(benchmark_version, level_id, track_id)`.

The official manifests are under `controller_learning/assets/tracks/v0.1/`. Level 0, Validation,
and Test NPZ files are packaged. The Train arrays are reproducibly materialized into the ignored
`.track-cache/v0.1/train_pool.npz` and verified against the committed Train manifest; they must not
be committed.

```bash
pixi run verify-track-assets
pixi run materialize-track-pool
pixi run verify-track-assets -- --require-train-cache
pixi run -e gpu build-track-assets
pixi run -e gpu benchmark-track-pool
```

`build-track-assets` is the formal, expensive GPU publication workflow. The normal consumer path is
`verify-track-assets`, followed by `materialize-track-pool` when the Train cache is needed. Reviewed
evidence is in the
[M5 admission report](../benchmarks/v0.1/m5_track_admission_report.json) and
[M5 TrackPool report](../benchmarks/v0.1/m5_track_pool_report.json). M5 is complete; M6 PID and MPC
Controller work is active and must consume these published values rather than override them.
