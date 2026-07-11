# Contributing to Controller Learning

Thank you for helping improve Controller Learning. The project values small, evidence-backed
changes that preserve the Controller/Challenge boundary and the immutable benchmark `0.1` record.

## Before opening a change

Read the [Controller workflow](https://aojili.github.io/controller-learning/getting-started/),
[stability policy](https://aojili.github.io/controller-learning/stability/), and the relevant design
page. For architecture or benchmark changes, open an issue first and identify whether the proposal
is a compatible v0.1.x improvement, a v0.2 migration, or a new benchmark version.

Good v0.1.x contributions include focused bug fixes, tests, documentation, Controller-author
ergonomics, and reporting improvements. New Controller families, Levels, physics backends, real
vehicle integration, online execution, and changes to the accepted M8 protocol require separate
scope decisions; they should not arrive as incidental pull-request changes.

## Development setup

Pixi is the only supported environment workflow:

```bash
git clone https://github.com/AojiLi/controller-learning.git
cd controller-learning
pixi install
pixi run ci
```

Use a focused test first, then the complete CPU validation:

```bash
pixi run pytest tests/unit/path/to/test_file.py
pixi run ci
```

The CPU CI task checks formatting, lint, tests, official Track assets, deterministic result-analysis
outputs, strict MkDocs, GitHub Actions syntax, and package metadata. Regenerate the checked analysis
page only from its frozen inputs:

```bash
pixi run build-result-analysis
pixi run check-result-analysis
```

Do not add Conda, Poetry, Docker, or a second lockfile as an alternative setup path.

## Controller and benchmark discipline

- Controllers may use only public observations, restricted info, immutable configuration,
  callbacks, and write-only `DebugDraw`.
- Keep environment and Controller seeds separate and deterministic.
- Use Level 0 and Validation through `evaluate-controller` for ordinary iteration. Do not use Test
  outcomes for Controller changes or selection.
- Do not modify the frozen PID, MPC, or PPO directories in a v0.1.x maintenance change.
- Do not modify accepted M8 artifacts. Derived reporting must identify its frozen inputs and must
  never execute another formal Test run.
- Keep generated runs, checkpoints, large artifacts, `reference/`, and secrets out of Git.

## GPU changes

Install and test the explicit GPU environment only on a compatible NVIDIA Linux host:

```bash
pixi install -e gpu
pixi run -e gpu gpu-check
pixi run -e gpu gpu-tests
```

A GPU performance or scale claim must report the environment count, steps, compilation time,
throughput, VRAM, numerical failures, hardware, driver, and locked software versions. CPU
multiprocessing is not evidence for the native GPU-batching requirement. Avoid running formal M8
Test during routine development.

## Pull-request checklist

- Explain the user-visible outcome and the smallest affected contract.
- Add or update tests at the lowest useful level.
- Keep public documentation, API names, comments, examples, and CLI text in English.
- Run `pixi run ci`; report any GPU checks separately with their hardware context.
- Confirm `git status --ignored` contains no secret, local run, reference source, or unreviewed
  benchmark artifact intended for commit.
- Use a concise conventional commit such as `fix(replay): reject existing output`.

By contributing, you agree that your contribution is licensed under the repository's MIT License.
