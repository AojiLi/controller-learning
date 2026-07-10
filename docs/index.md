# Controller Learning

Controller Learning is a GPU-parallel race-car control benchmark with procedurally generated
tracks, pluggable Controllers, and reproducible evaluation.

The project is being implemented through explicit milestone gates. Published documentation will
only claim features and performance that have passed their corresponding tests and benchmarks.

M6 is complete: PID and MPC use the public Controller boundary and formal MJX-Warp evaluator. PID
completed its fixed 10-Track Validation prefix, and MPC completed 95 of 100 fixed Validation
Tracks. M7 PPO training on the official vector environment is active.

## Design Principles

- One physical four-wheel simulation truth for every Controller.
- One official Challenge for classical control, reinforcement learning, and evaluation.
- Native GPU batching rather than one simulator per CPU process.
- Public, deterministic benchmark geometry, seeds, configuration, and run manifests.
- A narrow Controller interface with no simulator-internal shortcuts.

See the repository README for the current implementation status and verified commands.

The measured M3 representation and Challenge semantics are documented in
[Tracks and Race Core](tracks.md). The public M4 interfaces and measured 165,633 transitions/s
environment run are documented in [Gymnasium and Controller Platform](environment.md). The
observation-only PID and MPC examples are explained in
[Classical Controllers: PID and MPC](controllers.md), together with the measured M6 success,
timing, and memory evidence.
