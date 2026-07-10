# Tracks and Race Core

M3 established deterministic closed-track geometry and the pure-JAX batched race-state layer. The
physical vehicle still runs on a uniform MuJoCo plane: tracks are numerical Challenge data rather
than collision meshes. This lets every GPU world use different geometry without duplicating or
recompiling the vehicle model.

M4 exposes these contracts through the Gymnasium and Controller interfaces described in
[Gymnasium and Controller Platform](environment.md). M5 now publishes the fixed Level assets,
versioned split manifests, reproducible training cache, and device-resident TrackPool.

## Published Levels and Splits

Level 0 is a deterministic smooth ellipse, with 70 m and 50 m semi-axes and the same fixed 7 m width
used by Level 1. Its start is normalized to `(0, 0, 0)` with a `+x` tangent. It uses the reserved
maximum uint32 Track ID, `4,294,967,295`, so it cannot collide with an ordinary Level 1 generator
seed.

Level 1 uses three immutable half-open seed namespaces. Admission scans each namespace in ascending
order and does not hide retries inside Track generation:

| Split | Track count | Seed namespace | Use |
| --- | ---: | --- | --- |
| Train | 10,000 | `[0, 1,000,000)` | Controller training and pool sampling |
| Validation | 100 | `[1,000,000, 2,000,000)` | Development and tuning evidence |
| Test | 20 | `[2,000,000, 3,000,000)` | Held-out formal evaluation |

The stable public namespace of a numeric `track_id` is the composite
`(benchmark_version, level_id, track_id)`. The ID itself remains a device-native uint32 generator
seed rather than a host string. The manifests bind each accepted seed to its packed-geometry
SHA-256 digest, protocol versions, capacity, and exact asset digest. Selected seeds and geometry
hashes are disjoint across Level 0, Train, Validation, and Test.

## Official Asset Workflow

All manifests live under `controller_learning/assets/tracks/v0.1/`. The small fixed
`level0.npz`, `validation.npz`, and `test.npz` assets are committed and packaged. The Train manifest
is committed, but its 272,800,000-byte `train_pool.npz` is reconstructed into the ignored local
cache `.track-cache/v0.1/train_pool.npz`; it is not a repository asset.

Verify the manifests and fixed assets without requiring a local Train cache:

```bash
pixi run verify-track-assets
```

Reproduce or verify the Train cache from the committed seed/hash manifest, then verify all assets
including that cache:

```bash
pixi run materialize-track-pool
pixi run verify-track-assets -- --require-train-cache
```

The formal GPU admission command regenerates the official manifests, fixed assets, local Train
cache, and admission report. It is the expensive publication workflow, not a normal setup step:

```bash
pixi run -e gpu build-track-assets
```

Run the formal 1,024-world TrackPool protocol after the Train cache is present:

```bash
pixi run -e gpu benchmark-track-pool
```

## Fixed-Capacity Track Contract

An immutable host `Track` contains the centerline, left and right boundaries, unit tangents,
curvature, cumulative arc length, ordered checkpoints, canonical rear-axle start pose, masks, counts,
length, width, seed, and generator version. `TrackBatch` stacks only fixed-shape numerical leaves for
JAX. Valid values occupy a prefix; unused capacity is zero-filled and masked.

The v0.1 representation is locked to:

| Field | Shape per track | Rule |
| --- | --- | --- |
| Centerline and boundaries | `(640, 2)` each | 1.0 m nominal arc spacing; final point closes the loop |
| Tangent | `(640, 2)` | Unit tangent for each valid point |
| Curvature and cumulative length | `(640,)` each | SI units; cumulative length ends at lap length |
| Track mask | `(640,)` | Prefix mask including the explicit closure point |
| Checkpoint center and tangent | `(48, 2)` each | Ordered at 15 m spacing, including the finish |
| Checkpoint mask | `(48,)` | Prefix mask |

The generator uses one deterministic candidate per seed with no hidden retry. It samples 8–16
ordered control points, fits a periodic cubic spline, resamples by arc length, canonicalizes the
start to rear-axle pose `(0, 0, 0)`, and derives boundaries and checkpoints. The validator checks
schema and finite values, closure and orientation, length, width, curvature, start straight,
self-intersection, boundary crossing, nonlocal clearance, checkpoint order, and reproducibility.

## Measured Capacity Selection

The capacity protocol evaluated 10,000 contiguous seeds independently at each resolution:

| Arc spacing | Generated | Accepted | Capacity candidate | 1,024-world batch | 10,000-track pool |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 0.75 m | 9,994 | 9,964 | 896 points / 48 checkpoints | 36.891 MiB | 360.260 MiB |
| **1.00 m** | **9,994** | **9,965** | **640 points / 48 checkpoints** | **26.641 MiB** | **260.162 MiB** |
| 1.25 m | 9,994 | 9,965 | 512 points / 48 checkpoints | 21.516 MiB | 210.114 MiB |

For the selected 1.0 m spacing, six seeds failed the 300–600 m generation length gate and 29
generated candidates exceeded the `1 / 15 m` curvature limit. All eight sampled reproducibility
checks matched exactly. A 600 m track theoretically requires 601 points including explicit closure,
and 15 m checkpoint spacing requires 40 checkpoints. Capacities of 640 and 48 add measured headroom
without changing spatial resolution.

The 1.0 m selection is a resolution/memory balance: 0.75 m costs about 10.25 MiB more per 1,024-world
batch, while 1.25 m saves only about 5.13 MiB by reducing the geometry resolution. The complete
machine-readable evidence is the
[track-capacity report](https://github.com/AojiLi/controller-learning/blob/main/benchmarks/v0.1/track_capacity_report.json).

## Batched Race Core

Race Core consumes public rear-axle positions and `TrackBatch`; it never reads MuJoCo or MJX data.
Its fixed-shape JAX operations implement:

- topology-local projection around the previous segment, preventing a nearby hairpin from creating
  a progress shortcut;
- ordered checkpoint crossing and legal, non-duplicated forward progress;
- effective boundaries reduced by half the vehicle width and a safety margin;
- normalized progress reward, success and off-track outcomes, and a length-dependent timeout;
- deterministic termination priority and per-world masked reset.

The configured projection window examines four segments behind and twelve ahead. Timeout is
`max(60 s, track_length / 3 m/s)`. Success, off-track, and invalid action are terminal outcomes;
timeout is truncation. Reward remains a learning signal and is not an evaluation ranking score.

The GPU integration suite placed 1,024 distinct tracks in one batch and reused the same compiled
projection and step executables for different track values. Masked track replacement and Race Core
reset preserved every unselected world. In a 16-step randomized rollout, changing one world's track
and position sequence left all outputs for the other 1,023 worlds bit-exact. Peak JAX allocation was
about 140.4 MB in this test.

## Four-Wheel Driveability Gate

Geometry validity is followed by a conservative physical admission check. The formal M3 run used
the same MJX-Warp four-wheel backend and production Race Core as later environments. A private
reference policy targeted 4 m/s; it is not exposed as a Controller and does not participate in
benchmark ranking.

All 16 accepted tracks completed one lap: 16/16 success over 46,400 transitions. Maximum lateral
error was 0.2387 m and maximum planar speed was 3.9847 m/s. There were no off-track, timeout,
invalid-action, numerical-failure, contact/constraint-overflow, or unexpected-contact outcomes. The
complete machine-readable evidence is the
[track-driveability report](https://github.com/AojiLi/controller-learning/blob/main/benchmarks/v0.1/track_driveability_report.json).

## Measured M5 Admission Result

The formal M5 admission run completed in 1,266.411 seconds. Its batched four-wheel MJX-Warp work
took 1,116.205 seconds and executed 54,161,408 transitions at 48,522.822 transitions/s.

| Split | Candidate rows | Geometry rejected | Driveability rejected | Quota extras | Selected |
| --- | ---: | ---: | ---: | ---: | ---: |
| Train | 11,306 | 42 | 1,220 | 44 | 10,000 |
| Validation | 1,027 | 3 | 13 | 911 | 100 |
| Test | 1,026 | 2 | 4 | 1,000 | 20 |

Admission uses fixed 1,024-world GPU batches. `Quota extras` are candidates retained in the report
after the required count was reached within the final batch; they are not part of the published
split. This accounts for every candidate row in the table.

Level 0 and every selected Level 1 Track passed geometry and conservative physical driveability.
All official locations and hashes, cross-split seed/hash disjointness, serialized artifact readback,
and source/runtime gates passed. The complete evidence is the
[M5 Track admission report](https://github.com/AojiLi/controller-learning/blob/main/benchmarks/v0.1/m5_track_admission_report.json).

## Measured M5 TrackPool Result

The formal pool contained 10,000 Tracks and 17 GPU-resident leaves occupying exactly 272,800,000
bytes, matching the verified host representation. Its E1 headline epoch used 1,024 worlds for
10,000 steps: 10,240,000 transitions in 48.6758 seconds, or 210,371.5 transitions/s. The matched
fixed-Track baseline measured 219,604.7 transitions/s, so pool sampling retained a 0.958 throughput
ratio.

The first long run exposed a one-time 524 MiB process-VRAM allocator expansion. The v2 protocol
therefore fixes its memory baseline only after a separate 10,000-step E0 stabilization epoch, then
runs three distinct-seed 10,000-step epochs on the same environment without clearing JAX caches.
Across E0–E3, 40,960,000 transitions were executed. Post-E0 process-VRAM, allocator-pool, and
allocator-peak growth were all zero. Live JAX bytes had 4,936,960-byte maximum growth and ended
2,191,360 bytes above E0; host RSS ended only 0.027 MiB above E0. Peak sampled process VRAM was
1,334 MiB.

The separate 3,998-step health run observed exactly 1,024 timeout and next-step autoreset events,
with no unexpected termination or non-finite value. The reset-heavy protocol requested 65,536
resets and matched the host domain-2 reference. Active/mixed transfer guards, E0–E3 JIT-cache
stability, source binding, and privacy gates all passed. The complete 62-gate evidence is the
[M5 TrackPool report](https://github.com/AojiLi/controller-learning/blob/main/benchmarks/v0.1/m5_track_pool_report.json).
