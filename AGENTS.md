# Repository Instructions

## Start Here

Before making changes, read these files in order:

1. `CONTEXT.md` for durable project facts and invariants.
2. `ACTIVE_CONTEXT.md` for the current direction and next step.
3. `STATUS.md` for the short human-facing project state.
4. `PROJECT_PLAN.md` when the task touches scope, architecture, milestones, or acceptance criteria.

`PROJECT_PLAN.md` records the confirmed v0.1 product and architecture decisions. Do not silently change those decisions. If implementation evidence invalidates one, record the evidence and discuss the smallest viable revision with the user.

## Always-On Rules

- Work milestone by milestone. The current implementation target is recorded in `ACTIVE_CONTEXT.md`.
- Keep the Challenge layer independent from Controller implementations. Controllers may only use the documented observation, action, config, callback, and `DebugDraw` interfaces.
- Keep one official environment path: PPO trains the same `VecCarRacingEnv` used by the benchmark, with only public observation/reward wrappers.
- Preserve native GPU batching as a hard requirement. Do not replace it with CPU multiprocessing and still claim the M2 requirement is met.
- The simulation truth is a physical four-wheel car. Simplified bicycle/kinematic models are allowed inside Controllers, not as the default simulation truth.
- Treat CPU MuJoCo as a development/reference path and MJX-Warp as the formal v0.1 training and evaluation backend.
- Use Pixi as the only supported environment workflow for v0.1. Do not add Poetry, Conda setup files, Docker, or a second lockfile unless the project direction explicitly changes.
- Public repository content—README, docs, API names, code comments, examples, and user-facing CLI text—must be English. Internal planning and status files may be Chinese.
- Do not commit `reference/`, secrets, generated runs, local results, large checkpoints, or unreviewed benchmark artifacts. Add exclusions before initializing or publishing Git history.
- Do not claim macOS, Windows, WSL2, GPU scale, real-time performance, or Controller success rates without the corresponding test evidence.
- Keep environment and Controller seeds separate and deterministic. Preserve fixed public validation/test tracks once a benchmark version is published.
- Avoid broad backend abstractions, perception, multi-car racing, MPCC, sim-to-real, and other v0.1 non-goals unless the user explicitly changes scope.

## Change and Verification Discipline

- Inspect nearby code, tests, configs, and relevant plan sections before editing.
- Preserve unrelated user changes and avoid destructive Git operations.
- Pair behavior changes with tests at the lowest useful level; add integration or GPU checks when behavior crosses those boundaries.
- Run the narrowest relevant Pixi task first, then the broader task when available. Commands listed as planned in `CONTEXT.md` are not valid until M0 creates them.
- GPU work must report environment count, steps, compile time, throughput, VRAM, numerical failures, and hardware/software versions.
- Update `ACTIVE_CONTEXT.md` when the active direction changes. Update `STATUS.md` after meaningful milestone progress, a decision change, or a newly discovered blocker.
- Keep `CONTEXT.md` durable: place temporary progress in `ACTIVE_CONTEXT.md` or `STATUS.md`, not there.

## Reference Repository

`reference/lsy_drone_racing/` is read-only local research material derived from the upstream project. Use it to study Challenge layering, Controller loading, vector-environment semantics, evaluation, and Pixi patterns. Reimplement concepts for this project; do not copy reference source wholesale or let its drone-specific assumptions define the car API.
