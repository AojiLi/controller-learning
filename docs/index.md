# Controller Learning

Controller Learning is a GPU-parallel race-car control benchmark with procedurally generated
tracks, pluggable Controllers, and reproducible evaluation.

The project is being implemented through explicit milestone gates. Published documentation will
only claim features and performance that have passed their corresponding tests and benchmarks.

M4 is complete: the Gymnasium single/vector environments, trusted Controller platform, renderer,
single-run CLI, native transfer-free GPU hot path, and 1,024-world environment benchmark have passed
their gates. M5 Level assets and versioned Track pools are now active.

## Design Principles

- One physical four-wheel simulation truth for every Controller.
- One official Challenge for classical control, reinforcement learning, and evaluation.
- Native GPU batching rather than one simulator per CPU process.
- Public, deterministic benchmark geometry, seeds, configuration, and run manifests.
- A narrow Controller interface with no simulator-internal shortcuts.

See the repository README for the current implementation status and verified commands.

The measured M3 representation and Challenge semantics are documented in
[Tracks and Race Core](tracks.md). The public M4 interfaces and measured 165,633 transitions/s
environment run are documented in [Gymnasium and Controller Platform](environment.md).
