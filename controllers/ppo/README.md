# PPO Controller

This directory contains the finalized M7 inference-only PPO Controller. It is an ordinary
single-environment Controller plugin: every episode receives only the documented public
observation, info, and immutable public configuration. At runtime it uses NumPy to encode the
versioned 100-dimensional local-track feature vector and evaluate a fixed `100 -> 128 -> 128 -> 2`
deterministic actor. It does not import PyTorch and contains no optimizer, value-network, or
simulator state.

The policy was exported from update 70 of training run `m7-formal-v0-1-001` after the frozen
Validation selection. The selection completed 95 of 100 Tracks; the seeded random baseline
completed none. The canonical evidence is recorded in
[`m7_ppo_selection_report.json`](../../benchmarks/v0.1/m7_ppo_selection_report.json) and
[`m7_ppo_export_report.json`](../../benchmarks/v0.1/m7_ppo_export_report.json).

- `policy.npz` contains the canonical, hash-bound NumPy actor weights.
- `metadata.json` binds the actor to the selected checkpoint and feature schema.
- `config.toml` activates the plugin and repeats the expected hashes, sizes, and checkpoint
  identity.

Do not edit these generated files by hand. Hash, size, schema, feature parameters, physical action
bounds, training checkpoint identity, and local filenames are verified whenever a fresh Controller
instance is constructed.
