# PPO Controller

This directory is the source template for the M7 inference-only PPO Controller. It is an ordinary
single-environment Controller plugin: every episode receives only the documented public
observation, info, and immutable public configuration. At runtime it uses NumPy to encode the
versioned 100-dimensional local-track feature vector and evaluate a fixed `100 -> 128 -> 128 -> 2`
deterministic actor. It does not import PyTorch and contains no optimizer or simulator state.

The repository intentionally does not contain placeholder weights. `config.toml` therefore has
`finalized = false`, and constructing this Controller fails with a clear error. After Train-only
optimization and separate Validation-only checkpoint selection, the export workflow writes:

- `policy.npz`: canonical, hash-bound NumPy actor weights;
- `metadata.json`: checkpoint and feature identity with explicit inference-only contents;
- a finalized `config.toml`, committed last so partially staged files cannot activate the plugin.

Do not edit those finalized files by hand. Hash, size, schema, feature parameters, physical action
bounds, training checkpoint identity, and local filenames are verified whenever a fresh Controller
instance is constructed.
