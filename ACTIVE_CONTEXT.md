# Active Context

Last updated: 2026-07-10

## Current Direction

Execute M5: fixed Level 0/1 assets, the Level 1 training pool, fixed validation/test geometry, and a
versioned benchmark manifest. M0 repository infrastructure, M1 CPU four-wheel validation, M2 native
MJX-Warp batching, M3 Track/Race Core, and M4 Gymnasium/Controller Platform are complete. Do not
start PID/MPC or PPO until the Level assets and pool-reset path pass the M5 gates.

## M4 Handoff Evidence

- `ControllerLearning/CarRacing-v0`, `CarRacingEnv`, and `VecCarRacingEnv` are implemented. The
  single environment is a host batch-one adapter over the sole vector Challenge state machine.
- Gymnasium checker, exact observation/action schemas, batch-one agreement, invalid-action
  behavior, explicit post-terminal reset, strict NEXT_STEP autoreset, and Gymnasium registration
  pass their tests.
- Environment and Controller seeds are domain-separated. The device implementation is bit-exact
  with the locked NumPy `SeedSequence` contract across masked episode updates.
- Warm MJX-Warp active and mixed-autoreset steps pass `jax.transfer_guard("disallow")`; the valid
  JAX action path performs no host-to-device or device-to-host transfer after warmup.
- Trusted Controller directories load under isolated package names, expose exactly one concrete
  class, receive a recursively immutable config/info whitelist, and create a fresh instance per
  episode. The Runner derives Level config from the actual environment.
- `DebugDraw` is write-only and drained per frame. The 2D renderer consumes only public observations
  and debug commands. `pixi run sim` completes the template episode on CPU and formal GPU paths.
- The reviewed `benchmarks/v0.1/m4_environment_report.json` passed all gates with 1,024 distinct
  valid Tracks and 10,000 timed steps: 10,240,000 transitions in 61.824 s (165,633 transitions/s).
  All worlds timed out and autoreset independently in the health run, numerical failures were zero,
  peak sampled process VRAM was 556 MiB, and steady growth was 10 MiB against a 64 MiB gate.
- Current local validation passes 298 CPU/default tests and all 21 GPU tests.

M4 proves the public execution and plugin boundaries. It does not yet provide fixed Level assets,
runtime Track-pool sampling, PID/MPC/PPO performance, or formal evaluation.

## Current Narrow Focus

- Define a versioned asset/manifest schema that preserves the published generator, validation,
  capacity, seed, split, geometry hash, and driveability evidence.
- Create one fixed, teachable Level 0 Track and bind Level 0 resets to that immutable asset.
- Select approximately 10,000 disjoint valid Level 1 training Tracks plus fixed validation and at
  least 20 fixed test Tracks with deterministic, auditable split rules.
- Keep the full training Track pool resident on GPU and select replacement Tracks during masked
  NEXT_STEP reset without host synchronization or shape-dependent recompilation.
- Preserve one official `VecCarRacingEnv`; extend its injected M4 Track source into the M5 pool
  source rather than creating a training-only environment.
- Validate split disjointness, geometry hashes, reproducible loading/generation, pool memory, and
  1,024-world independent sampling/autoreset.

## Scope Boundaries

In scope:

- fixed Level 0 geometry and configuration binding;
- deterministic Level 1 train/validation/test selection and versioned manifest;
- compact committed validation/test assets and a reproducible strategy for the large training pool;
- device-native Track-pool indexing and masked replacement in the official vector environment;
- geometry, split, reproducibility, memory, GPU autoreset, and bounded physical driveability gates.

Out of scope:

- PID, MPC, PPO, reward shaping, checkpoint selection, or Controller success claims;
- hidden test infrastructure, online submissions, multi-car racing, perception, or sim-to-real;
- alternate physics backends, macOS/Windows/WSL2 support, or broader environment abstractions.

## Confirmed Judgments

- Validation/test geometry is fixed and immutable inside benchmark version `0.1`; generator changes
  require a new benchmark version rather than silently regenerating published assets.
- Training, validation, and test splits must be disjoint by both Track ID and geometry hash.
- The 10,000-Track numerical representation is about 260.162 MiB and is intended to reside on the
  GPU. CPU multiprocessing is not an acceptable substitute for pool batching.
- PPO in M7 must still train the same `VecCarRacingEnv`; M5 may add a Track-source/pool input but not
  a parallel RL environment.
- Fixed public validation/test files may be committed when reasonably sized. A roughly 260 MiB
  training array must not be committed blindly; its reproducible seed manifest/cache strategy must
  be explicit and verifiable.

## Open M5 Engineering Questions

- The exact validation split size and the smallest useful Level 0 geometry.
- Whether committed fixed assets use NPZ plus JSON manifest, another deterministic binary format,
  or seed+hash records with reproducible materialization.
- How public `track_id` remains device-native when a pool reset changes a world's Track; the current
  M4 string ID is static per injected Track and must not force a per-step host synchronization.
- How to sample pool indices deterministically from per-world episode identity while keeping
  environment and Controller RNG domains independent.
- The bounded physical driveability protocol and batching strategy for admitting the full pool.

## Next Step

Review the current Track/episode/environment data flow and decide the M5 manifest, split, Track-ID,
and device pool-selection contracts before generating or committing any large asset.
