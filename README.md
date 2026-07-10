# Controller Learning

**A GPU-parallel race car control benchmark with procedurally generated tracks,
pluggable controllers, and reproducible evaluation.**

Controller Learning is a benchmark and teaching platform for developing and comparing race-car
controllers under one environment, vehicle, task, and evaluation protocol. PID, MPC, and PPO are
planned as examples; the reusable Challenge and Controller interface are the core product.

> **Project status:** M5 is complete. Fixed Level 0/1 assets, versioned Track manifests, reproducible
> training-pool materialization, and the 10,000-Track GPU pool have passed their formal gates. M6
> PID and MPC Controller examples are now active.

Reviewed machine-readable evidence is available in the
[M1 CPU report](benchmarks/v0.1/m1_cpu_report.json) and
[M2 GPU report](benchmarks/v0.1/gpu_report.json). M3 evidence is in the
[track-capacity report](benchmarks/v0.1/track_capacity_report.json) and
[track-driveability report](benchmarks/v0.1/track_driveability_report.json). The complete M4
environment path is measured in the
[M4 environment report](benchmarks/v0.1/m4_environment_report.json). M5 evidence is in the
[Track admission report](benchmarks/v0.1/m5_track_admission_report.json) and
[TrackPool GPU report](benchmarks/v0.1/m5_track_pool_report.json).

## Why This Project Exists

Control approaches are difficult to compare when each example uses a different vehicle model,
track, observation, action, or success definition. This project is designed to make those choices
explicit and reproducible:

- a physical four-wheel race car as the simulation truth;
- native GPU-batched simulation for reinforcement learning;
- fixed and procedurally generated closed-loop tracks;
- a small directory-based Controller plugin interface;
- the same official environment for classical control, training, and evaluation; and
- public benchmark tracks, seeds, manifests, metrics, and replays.

## Planned v0.1 Stack

- MuJoCo MJCF and MJX-Warp
- JAX and Gymnasium
- CasADi/IPOPT for MPC
- PyTorch for PPO
- Pixi on Linux with Python 3.11

Controller success rates will only be documented after the corresponding milestone benchmarks
pass.

## Development Setup

Pixi is the only supported environment workflow for v0.1.

```bash
pixi install
pixi run tests
pixi run lint
pixi run docs
```

Run the template Controller through one complete development episode:

```bash
pixi run sim
```

The NVIDIA environment is installed separately so CPU development and CI do not resolve or install
CUDA/PyTorch dependencies:

```bash
pixi install -e gpu
pixi run -e gpu gpu-check
pixi run -e gpu gpu-tests
```

These commands are verified as part of M0. Linux x86-64 with glibc 2.28 or newer is the only
supported v0.1 platform; macOS, native Windows, and WSL2 are future work.

## Architecture

The repository separates five responsibilities:

1. **Physics** advances the four-wheel vehicle.
2. **Track** owns deterministic geometry, validation, and benchmark pools.
3. **Challenge** defines observations, actions, progress, reward, reset, and termination.
4. **Controller** contains trusted plugins that only use the public interface.
5. **Evaluation** produces reproducible metrics, manifests, plots, and replays.

PPO will train directly against the official `VecCarRacingEnv`; the project will not maintain a
second simplified training environment.

## Verified GPU Result

The formal M2 run used an NVIDIA GeForce RTX 5070 Ti Laptop GPU and the locked Pixi environment. It
completed 10,000 environment steps with 1,024 native worlds: 10,240,000 transitions and 102,400,000
world-physics steps. The measured rate was 77,751 transitions/s with 346 MiB peak process VRAM and
no long-window process-VRAM growth. All states remained finite, all four wheel contacts stayed
within the physical gates, and no buffer overflow, unexpected contact, or runtime warning occurred.

This is the M2 physics-layer result. M3 subsequently validated track geometry and independent Race
Core state, and M4 exposed them through Gymnasium. PPO remains a later milestone and is not implied
by these results.

## Verified M3 Track and Race Core Result

The M3 capacity sweep evaluated 10,000 contiguous seeds at each of 0.75 m, 1.0 m, and 1.25 m arc
spacing. The selected 1.0 m representation generated 9,994 candidates, accepted 9,965 after
validation, and reproduced all eight sampled seeds exactly. Six candidates were outside the length
range and 29 exceeded the curvature limit. The 600 m length bound requires at most 601 stored points
and 40 checkpoints; the locked capacities are 640 points and 48 checkpoints.

One 1,024-world `TrackBatch` occupies 26.641 MiB and a 10,000-track numerical pool occupies 260.162
MiB. The 1.0 m spacing preserves more geometry resolution than 1.25 m while avoiding the additional
memory cost of 0.75 m, so it is the measured resolution/memory balance for v0.1.

GPU tests passed with 1,024 distinct tracks using the same compiled Race Core executable. Masked
track replacement and race reset preserved unselected worlds, and perturbing one world through a
16-step rollout left the other 1,023 worlds bit-exact. The observed peak JAX allocation was about
140.4 MB. A separate formal MJX-Warp driveability run completed all 16 generated tracks at a 4 m/s
target with 0.239 m maximum lateral error, no failure outcome, and no numerical or buffer fault over
46,400 transitions. See [Tracks and Race Core](docs/tracks.md) for the contract and protocol.

## Verified M4 Environment Result

`CarRacingEnv` and `VecCarRacingEnv` now share one Challenge state machine. The registered ID is
`ControllerLearning/CarRacing-v0`; the vector path retains leading JAX arrays and strict Gymnasium
NEXT_STEP masked autoreset. Controllers are loaded from trusted directories, instantiated fresh for
every episode, and receive only public observations, restricted info, immutable public config, and
write-only `DebugDraw`.

The formal M4 run placed 1,024 different validated tracks in one MJX-Warp environment and executed
10,000 environment steps: 10,240,000 transitions in 61.824 seconds, or 165,633 transitions/s. The
separate health run observed timeout and subsequent independent autoreset for all 1,024 worlds, with
no unexpected termination or non-finite public output. Warm active and mixed-autoreset steps passed
JAX transfer guards that disallow both host-to-device and device-to-host transfers.

Peak sampled process VRAM was 556 MiB. The measured steady segment grew by 10 MiB against a 64 MiB
gate. First-step compilation took 1.698 seconds on the recorded NVIDIA GeForce RTX 5070 Ti Laptop
GPU. These are zero-action environment-throughput measurements, not Controller performance claims.
See [Gymnasium and Controller Platform](docs/environment.md) for the API and protocol.

## Verified M5 Level Assets and TrackPool Result

M5 publishes one deterministic Level 0 ellipse with reserved numeric Track ID `UINT32_MAX`, plus
three disjoint Level 1 namespaces:

| Split | Published Tracks | Allowed seed range |
| --- | ---: | --- |
| Train | 10,000 | `[0, 1,000,000)` |
| Validation | 100 | `[1,000,000, 2,000,000)` |
| Test | 20 | `[2,000,000, 3,000,000)` |

All four manifests are committed. Level 0, Validation, and Test NPZ assets are also committed and
packaged; the 272,800,000-byte Train pool is reconstructed into the ignored local cache
`.track-cache/v0.1/train_pool.npz` and verified against its manifest hash. Formal admission selected
the 10,000 Train Tracks from 11,306 ascending-seed candidates after 42 geometry and 1,220 physical
driveability rejections; 44 additional valid candidates in the final fixed-size GPU batch were
recorded as quota extras. The complete admission took 1,266.411 seconds, including 1,116.205 seconds
and 54,161,408 transitions on the four-wheel GPU backend at 48,522.822 transitions/s. All official
Tracks, split-disjointness checks, artifact hashes, and serialized readback gates passed.

The formal TrackPool headline epoch ran 1,024 worlds for 10,000 steps: 10,240,000 transitions in
48.6758 seconds, or 210,371.5 transitions/s. The matched fixed-Track baseline measured 219,604.7
transitions/s, giving a 0.958 throughput ratio. The strengthened memory protocol ran E0 through E3
for 40,960,000 total transitions on one environment. The first long run exposed a one-time 524 MiB
allocator expansion; after E0, process VRAM, allocator pool size, and allocator peak growth were all
zero through E3. Peak sampled process VRAM was 1,334 MiB. Health, reset-heavy, transfer, JIT-cache,
source, and privacy gates also passed. These are environment and asset results, not Controller
success claims. See [Tracks and Race Core](docs/tracks.md) for the asset and sampling contracts.

## Roadmap

The implementation follows strict milestone gates:

- M0: repository, Pixi, package, tests, CI, and configuration schemas — complete
- M1: stable CPU MuJoCo four-wheel car — complete
- M2: MJX-Warp 1/64/256/1024-world GPU go/no-go — complete
- M3: batched tracks and Race Core — complete
- M4: Gymnasium environments and Controller platform — complete
- M5: Level 0/1 and versioned track pools — complete
- M6: PID and MPC — active
- M7: PPO on the official vector environment
- M8: evaluation, documentation, and public v0.1 release

The detailed confirmed design is recorded in [PROJECT_PLAN.md](PROJECT_PLAN.md).

## Inspiration

The Challenge-layer design is inspired by
[learnsyslab/lsy_drone_racing](https://github.com/learnsyslab/lsy_drone_racing). This repository is
an independent race-car implementation and does not vendor the reference source.

## License

Controller Learning is released under the [MIT License](LICENSE).
