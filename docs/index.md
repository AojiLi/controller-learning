# Controller Learning

Controller Learning is a GPU-parallel race-car control benchmark with procedurally generated
tracks, pluggable Controllers, and reproducible evaluation.

The project is being implemented through explicit milestone gates. Published documentation will
only claim features and performance that have passed their corresponding tests and benchmarks.

M2 is complete: the physical four-wheel model passed native MJX-Warp validation through 1,024
worlds and a 10,000-step endurance run. M3 batched tracks and Race Core are now active.

## Design Principles

- One physical four-wheel simulation truth for every Controller.
- One official Challenge for classical control, reinforcement learning, and evaluation.
- Native GPU batching rather than one simulator per CPU process.
- Public, deterministic benchmark geometry, seeds, configuration, and run manifests.
- A narrow Controller interface with no simulator-internal shortcuts.

See the repository README for the current implementation status and verified commands.
