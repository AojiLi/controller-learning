# Controller Learning

Controller Learning is a GPU-parallel race-car control benchmark with procedurally generated
tracks, pluggable Controllers, and reproducible evaluation.

The project is being implemented through explicit milestone gates. Published documentation will
only claim features and performance that have passed their corresponding tests and benchmarks.

M3 is complete: deterministic fixed-capacity tracks, batched Race Core state, 1,024-world isolation,
and low-speed four-wheel driveability have passed their gates. M4 Gymnasium environments and the
Controller platform are now active; neither interface is available yet.

## Design Principles

- One physical four-wheel simulation truth for every Controller.
- One official Challenge for classical control, reinforcement learning, and evaluation.
- Native GPU batching rather than one simulator per CPU process.
- Public, deterministic benchmark geometry, seeds, configuration, and run manifests.
- A narrow Controller interface with no simulator-internal shortcuts.

See the repository README for the current implementation status and verified commands.

The measured M3 representation and Challenge semantics are documented in
[Tracks and Race Core](tracks.md).
