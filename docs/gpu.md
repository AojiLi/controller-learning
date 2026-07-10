# MJX-Warp GPU Vehicle

M2 establishes the native GPU-batched physics backend for the physical four-wheel vehicle. It uses
the same packaged MJCF, vehicle configuration, action semantics, and rear-axle state convention as
the CPU reference. M2 is a physics gate only; tracks and race logic begin in M3.

## Backend Contract

`MjxWarpVehicle` keeps a leading world dimension in one MJX-Warp `Data` tree. It does not create one
Python simulator or process per world. One environment step is 0.05 seconds and contains ten 0.005
second physics substeps.

MuJoCo 3.10.0 MJX-Warp does not accept the model's disabled `AUTORESET` flag. The GPU adapter clears
only that flag on its private conversion model, records the source and effective flags, and replaces
hidden recovery with explicit finite-state, contact, constraint, pose, and velocity checks. The CPU
model and packaged MJCF remain unchanged.

## Formal Protocol

Each scale runs in a fresh process so compilation, allocator state, and VRAM from an earlier scale
cannot affect the next result. The protocol uses 1, 64, and 256 worlds for 1,000 measured environment
steps and 1,024 worlds for 10,000 measured steps. Compilation is timed separately, followed by eight
unmeasured warmup chunks before throughput and memory sampling.

The worker checks finite state, monotonic time, independent terminal signatures, masked resets,
contact and constraint capacity, contact participation, contact gaps, penetration, quaternion norm,
roll/pitch, vertical speed, generalized velocity, and native warnings. CPU/GPU agreement is measured
over a fixed five-second drive, steering, and braking rollout. The report records the clean Git
revision and hashes the lock file, model, config, adapter, protocol, worker, and launcher before and
after the run.

## Measured M2 Result

The reviewed report was generated on an NVIDIA GeForce RTX 5070 Ti Laptop GPU with JAX/JAXLIB
0.10.2, MuJoCo/MJX-Warp 3.10.0, Warp 1.13.0, and NVIDIA driver 590.48.01.

| Worlds | Measured steps | Compile (s) | Transitions/s | Peak process VRAM (MiB) | VRAM growth (MiB) | Minimum wheel coverage |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 1,000 | 0.581 | 307 | 358 | 1 | 0.923 |
| 64 | 1,000 | 0.650 | 10,864 | 356 | 2 | 0.862 |
| 256 | 1,000 | 0.743 | 34,196 | 356 | 1 | 0.857 |
| 1,024 | 10,000 | 0.899 | 77,751 | 346 | 0 | 0.843 |

The 1,024-world run completed 10,240,000 transitions and 102,400,000 world-physics steps in 131.703
seconds of measured execution. It observed no non-finite state, contact/constraint overflow,
unexpected contact, invalid action, or native warning. Peak global contact and collision counts were
8,192 and 5,120 against a 16,384-entry capacity; peak per-world constraints were 24 against 64.

The batch-one CPU/GPU rollout had 0.034 mm maximum planar-position error, 0.010 mm/s maximum body
velocity error, 0.0000005 rad maximum yaw error, and 0.000029 rad/s maximum wheel-speed error. Contact
participation, contact gap, and penetration also agreed within their recorded tolerances.

The complete evidence is stored at `benchmarks/v0.1/gpu_report.json`.
