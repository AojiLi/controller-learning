# Changelog

All notable user-facing changes are recorded here. Benchmark versions and software release versions
are separate: a maintenance release does not replace an accepted benchmark result.

## [0.1.1] - 2026-07-11

### Added

- Informal `evaluate-controller` workflow for trusted Controllers on Level 0 or Validation, with
  deterministic rows/seeds, explicit backend scope, transactional outputs, and optional
  same-rollout trajectory capture.
- Offline `replay` workflow for strict canonical trajectory loading, deterministic PNG output, and
  interactive public-observation playback.
- Hash-pinned deterministic benchmark interpretation generated only from accepted M8 CSV/NPZ data.
- Hosted MkDocs workflow, Controller quick start, stability policy, contribution guide, and citation
  metadata.

### Changed

- Restructured the README around the product, visual result, quick start, Controller loop, and a
  compact evidence index.
- Corrected the public PPO training example to include its required run ID.

### Compatibility

- Benchmark `0.1`, the accepted `m8-final-v0-1-002` artifacts, and the frozen PID/MPC/PPO Controller
  identities are unchanged.
- Formal Test was not rerun for this release.

## [0.1.0] - 2026-07-11

- Initial public release of the four-wheel CPU/MJX-Warp Challenge, procedural Track assets,
  GPU-vector environment, Controller plugin platform, PID/MPC/PPO examples, and accepted benchmark
  `0.1` evidence.

[0.1.1]: https://github.com/AojiLi/controller-learning/compare/v0.1.0...v0.1.1
[0.1.0]: https://github.com/AojiLi/controller-learning/releases/tag/v0.1.0
