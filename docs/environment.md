# Gymnasium and Controller Platform

M4 exposed the physical four-wheel Challenge through one vector state machine and one thin
single-world adapter. Classical Controllers, later PPO training, and evaluation all use this same
transition path; there is no simplified RL environment. M5 extends that same vector state machine
with deterministic device-native TrackPool selection and masked replacement.

## Quick Simulation

Run the non-performing template Controller on the deterministic development track:

```bash
pixi run sim
```

The command generates and validates exactly track seed `42`, runs one CPU-reference episode, and
prints a JSON result. It does not retry a rejected seed. Use the formal GPU backend from the GPU
Pixi environment:

```bash
pixi run -e gpu sim -- --backend mjx_warp
```

Add `--render` for the interactive 2D public-observation view. The template intentionally returns a
neutral action and reaches the normal Challenge timeout; it demonstrates the API rather than a
driving result.

## Environment Contract

The registered single-world ID is:

```text
ControllerLearning/CarRacing-v0
```

`CarRacingEnv` is a host/NumPy batch-one adapter over `VecCarRacingEnv`. The vector environment owns
all vehicle, Race Core, seed, termination, and NEXT_STEP autoreset state. Both constructors require
an explicit immutable `ProjectConfig`, Level, and backend. `CarRacingEnv` accepts one fixed `Track`;
`VecCarRacingEnv` accepts exactly one of fixed injected Tracks or the published M5 Train `TrackPool`.
All modes use the same Challenge path.

The public action is a float32 vector in physical units:

| Index | Meaning | Unit | Configured range |
| ---: | --- | --- | --- |
| 0 | Steering angle | rad | `[-0.6, 0.6]` |
| 1 | Longitudinal acceleration | m/s² | `[-8.0, 4.0]` |

Finite out-of-range actions are clipped and counted internally. A wrong shape, failed numeric
conversion, NaN, or infinity terminates only the affected active world as `invalid_action`; pending
NEXT_STEP-reset worlds ignore the supplied action.

The public observation is a Gymnasium `Dict`. A vector observation adds a leading `num_envs`
dimension to every field.

| Field | Single shape | Dtype | Meaning |
| --- | ---: | --- | --- |
| `position` | `(2,)` | float32 | Rear-axle world `(x, y)` |
| `yaw` | `()` | float32 | World yaw in `[-π, π]` |
| `velocity_body` | `(2,)` | float32 | Body longitudinal/lateral velocity |
| `yaw_rate` | `()` | float32 | Body yaw rate |
| `steering_angle` | `()` | float32 | Measured front steering angle |
| `track_progress` | `()` | float32 | Monotonic legal lap progress in `[0, 1]` |
| `centerline` | `(640, 2)` | float32 | Fixed-capacity world centerline |
| `left_boundary` | `(640, 2)` | float32 | Fixed-capacity left boundary |
| `right_boundary` | `(640, 2)` | float32 | Fixed-capacity right boundary |
| `track_mask` | `(640,)` | int8 | Valid Track prefix |
| `track_length` | `()` | float32 | Lap length in metres |

No projection index, lateral error, target speed, checkpoint state, physics diagnostics, MJX data,
or simulator object is exposed.

## Episode and Info Semantics

Reset and step info use one fixed seven-field whitelist so vector code does not create object-valued
terminal dictionaries:

- `episode_seed`
- `controller_seed`
- `track_id`
- `benchmark_version`
- `termination_reason`
- `lap_completed`
- `lap_time_s`

Reset and non-terminal rows use neutral terminal values. Environment and Controller seeds are
domain-separated and deterministic for `(root seed, world index, episode counter)`. The GPU path
implements the same NumPy `SeedSequence` result in pure JAX, so masked autoreset does not transfer
state to the host.

`track_id` is the selected Track's uint32 generator seed. Its stable identity is the composite
`(benchmark_version, level_id, track_id)`; the numeric value is not globally meaningful without the
published benchmark and Level namespace. Level 0 reserves `UINT32_MAX`. The Level 1 Train,
Validation, and Test namespaces are disjoint, and each manifest binds its IDs to exact geometry
hashes.

TrackPool choice uses SeedSequence domain 2, separate from episode domain 0 and Controller domain 1,
then maps the uint32 selection seed to a pool row with replacement. On a terminal call, info still
contains the old Track ID. On the following NEXT_STEP call, affected worlds first advance their
episode counters, derive their new domain-2 choice, and atomically replace Track, vehicle, Race Core,
and observation state. That reset transition returns the new Track ID with zero reward and false
termination flags; unaffected worlds advance normally.

Termination reason values are `0=none`, `1=success`, `2=off_track`, `3=invalid_action`, and
`4=timeout`. Timeout sets `truncated`; success, off-track, and invalid action set `terminated`.

`VecCarRacingEnv` uses strict Gymnasium NEXT_STEP semantics. A terminal transition returns its final
observation and outcome. On the next call, that world returns its new initial observation, zero
reward, false termination flags, and new seeds while every other world advances normally. The
single environment instead requires an explicit `reset()` after a terminal transition.

## Controller Plugins

A trusted plugin directory contains:

```text
controllers/example/
├── controller.py
├── config.toml
└── optional_helpers.py
```

`controller.py` must define exactly one concrete subclass of `Controller`. Relative helper imports
are isolated under a directory-specific package name. The Runner imports a fresh class and creates
a fresh instance for every episode, preventing accidental cross-episode state.

The constructor receives only the initial public observation, restricted info, and an immutable
configuration. The complete plugin TOML document appears under `config["controller"]`; it cannot
shadow Challenge-owned vehicle, action, Track, Level, or benchmark values. The Runner derives those
values from the actual unwrapped environment rather than accepting a second potentially mismatched
Level configuration.

Controller lifecycle methods are:

```python
class Controller:
    def __init__(self, obs, info, config): ...
    def compute_control(self, obs, info=None): ...
    def step_callback(self, action, obs, reward, terminated, truncated, info): ...
    def episode_callback(self): ...
    def render_callback(self, debug_draw): ...
```

`render_callback` receives a write-only surface with `line`, `points`, and `text`. Commands are
drained per frame and rendered together with the latest public observation. The Controller never
receives an Environment, renderer, vehicle backend, or simulator reference. Plugins are trusted
local code in v0.1; they are not sandboxed.

## Backend and GPU Boundary

`cpu_reference` is a batch-one development and Gymnasium-checker path. `mjx_warp` is the formal
training and evaluation backend and retains JAX arrays for actions, observations, pending-reset
state, episode identities, and numeric info. After warmup, tests disallow both host-to-device and
device-to-host transfers across an active step and a mixed-world autoreset step.

Run the contracts locally with:

```bash
pixi run tests
pixi run -e gpu gpu-tests
```

The formal M4 environment-throughput protocol is run separately:

```bash
pixi run -e gpu benchmark-racing-env
```

It writes `benchmarks/v0.1/m4_environment_report.json` and passes only when its source, transfer,
autoreset, numerical, memory, and formal-protocol gates all pass.

## Measured M4 Result

The reviewed run used 1,024 different validated Tracks. It executed 10,000 timed environment steps
without per-step host synchronization:

| Measurement | Result |
| --- | ---: |
| World transitions | 10,240,000 |
| Steady execution | 61.824 s |
| Environment steps/s | 161.751 |
| Transitions/s | 165,633 |
| Environment creation | 1.275 s |
| Reset compilation | 0.560 s |
| First-step compilation | 1.698 s |
| Peak sampled process VRAM | 556 MiB |
| Steady process-VRAM growth | 10 MiB / 64 MiB gate |

The separate 3,972-step health run observed timeout and next-step autoreset for every world. It
recorded zero success, off-track, or invalid-action outcomes from the zero-action schedule, zero
non-finite public values, and exactly 1,024 timeout/autoreset events. Both the active-step and
mixed-autoreset transfer guards passed.

The run used an NVIDIA GeForce RTX 5070 Ti Laptop GPU, NVIDIA driver 590.48.01, JAX/JAXLIB 0.10.2,
MuJoCo/MJX-Warp 3.10.0, and Warp 1.13.0. This is an environment-throughput and correctness result;
it is not a driving Controller benchmark. The complete source hashes, Track IDs, configuration,
checks, versions, and memory samples are in the
[M4 environment report](https://github.com/AojiLi/controller-learning/blob/main/benchmarks/v0.1/m4_environment_report.json).

## Measured M5 TrackPool Result

Run the reproducible local-cache and formal GPU workflows with:

```bash
pixi run materialize-track-pool
pixi run -e gpu benchmark-track-pool
```

The reviewed E1 headline epoch ran 1,024 worlds against the full 10,000-Track Train pool for 10,000
steps: 10,240,000 transitions in 48.6758 seconds, or 210,371.5 transitions/s. The fixed-Track
baseline measured 219,604.7 transitions/s, giving a 0.958 ratio. E0 plus three distinct-seed
measurement epochs ran 40,960,000 transitions on the same environment. After the disclosed one-time
524 MiB allocator expansion, process VRAM, allocator-pool size, and allocator peak showed zero
post-E0 growth; peak sampled process VRAM was 1,334 MiB.

The health protocol ran 3,998 steps and observed exactly 1,024 timeouts followed by 1,024 independent
autoresets, with no unexpected termination or non-finite public value. Another 65,536 requested
reset events matched the host domain-2 reference. Active and mixed-reset transfer guards, JIT-cache
stability, source binding, and privacy gates passed. These measurements establish the environment
and pool path; they do not claim that a Controller can drive. See the
[M5 TrackPool report](https://github.com/AojiLi/controller-learning/blob/main/benchmarks/v0.1/m5_track_pool_report.json)
for all 62 gates and [Tracks and Race Core](tracks.md) for admission and asset details.
