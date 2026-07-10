# Tracks and Race Core

M3 establishes deterministic closed-track geometry and the pure-JAX batched race-state layer. The
physical vehicle still runs on a uniform MuJoCo plane: tracks are numerical Challenge data rather
than collision meshes. This lets every GPU world use different geometry without duplicating or
recompiling the vehicle model.

M3 does not provide a Gymnasium environment or a Controller interface. Those are M4 responsibilities
built on top of the contracts described here.

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
