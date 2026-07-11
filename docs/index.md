# Controller Learning

Controller Learning is a GPU-parallel race-car control benchmark with procedurally generated
tracks, pluggable Controllers, and reproducible evaluation.

The project is being implemented through explicit milestone gates. Published documentation will
only claim features and performance that have passed their corresponding tests and benchmarks.

M7 is complete: PPO trained on the official 1,024-world vector environment, a frozen Validation
selection chose update 70 at 95/100 successes against a 0/100 seeded-random baseline, and the
Torch-free exported plugin completed 99/100 Validation Tracks through the ordinary batch-one
Controller Runner. No Controller has been evaluated on Test and Test performance remains unopened;
M8 final evaluation and release remain pending.

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
timing, and memory evidence. The official RL path, NEXT_STEP masks, DLPack bridge, frozen
checkpoint selection, NumPy export, and replay are documented in
[PPO: GPU Training to Controller Plugin](ppo.md).
