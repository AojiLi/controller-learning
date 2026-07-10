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
[M5 TrackPool report](../benchmarks/v0.1/m5_track_pool_report.json). M6 PID and MPC are also
complete and use these published values through the public Challenge. M7 PPO training is active;
its Controller and training configuration must not override Challenge, Level, or benchmark values.

`ppo.toml` is the single strict M7 training document. It covers the official environment identity,
public observation compression, public reward shaping, rollout budget, PPO optimizer, local
logging, and checkpoint policy. The committed values are an initial Train-only candidate: M7 may
adjust them using Train evidence, but the entire file must be frozen before any Validation-based
checkpoint selection. Vehicle steering limits and control timing continue to come from the
Challenge configuration and are intentionally not duplicated in PPO configuration.
Environment episode selection, policy sampling, and minibatch shuffling use three explicit,
distinct seeds so each randomness domain can be reproduced independently.

PPO optimization loads `.track-cache/v0.1/train_pool.npz` through a dedicated Train-only loader.
That path verifies the Train manifest, cache digest, count, capacity, Track order, and geometry
contract without calling the general all-split verifier. Validation has a separate later selection
phase, and no M7 path may load or evaluate Test geometry.
