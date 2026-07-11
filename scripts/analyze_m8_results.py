"""Build the descriptive M8 analysis page from frozen published CSV/NPZ evidence."""

from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import math
import os
import stat
import tempfile
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Final

import numpy as np
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.figure import Figure

PROJECT_ROOT = Path(__file__).resolve().parents[1]
OUTPUT_MARKDOWN = Path("docs/analysis.md")
OUTPUT_FIGURE = Path("docs/assets/m8_result_analysis.png")
CONTROLLERS: Final = ("pid", "mpc", "ppo")
LABELS: Final = {"pid": "PID", "mpc": "MPC", "ppo": "PPO"}
COLORS: Final = {"pid": "#1f77b4", "mpc": "#ff7f0e", "ppo": "#2ca02c"}
EXPECTED_INPUT_SHA256: Final = {
    "benchmarks/v0.1/m8_final_results.csv": (
        "a6d5a7425d1c1091ba3111c722b607f9f60398d8e444e9943c80d27042de7a04"
    ),
    "results/0.1/pid/m8-final-v0-1-002/results.csv": (
        "bce37414510ef1f4c865c19fbc69e45a74c475b94137eda9c1f9a6ba3fa0f44d"
    ),
    "results/0.1/pid/m8-final-v0-1-002/metrics.npz": (
        "5b09f33ffebf2268ae02dbbd43ac08daf8fded8fe3a7e294b6973eb363d31466"
    ),
    "results/0.1/mpc/m8-final-v0-1-002/results.csv": (
        "bd561a80f5eb7653361c5306fe67c9a240d99a23a718c0d0d2d450155a2d27ed"
    ),
    "results/0.1/mpc/m8-final-v0-1-002/metrics.npz": (
        "fcb70ab25e5413abe3da7787a9677f06944532a17c9b4cfaa4fb2065c31286c8"
    ),
    "results/0.1/ppo/m8-final-v0-1-002/results.csv": (
        "9dad0feab09c7df1bba5bd627c3d02d81707f00fa4283f49e592637d671b7fd8"
    ),
    "results/0.1/ppo/m8-final-v0-1-002/metrics.npz": (
        "c1117be7dabbec489d468a288cbecbff6d99b3c238a28542f8d1940beadadd6f"
    ),
}

_EXPECTED_METRIC_KEYS: Final = {
    "benchmark_version",
    "compute_time_s",
    "controller_name",
    "episode_offsets",
    "lateral_error_m",
    "longitudinal_saturated",
    "requested_action",
    "reset_seed",
    "schema_version",
    "speed_mps",
    "steering_saturated",
    "track_id",
}


class AnalysisError(RuntimeError):
    """Published analysis evidence or generated output failed a strict check."""


@dataclass(frozen=True, slots=True)
class ControllerEvidence:
    """One Controller's accepted aggregate, episode rows, and public samples."""

    aggregate: Mapping[str, str]
    episodes: tuple[Mapping[str, str], ...]
    metrics: Mapping[str, np.ndarray]


@dataclass(frozen=True, slots=True)
class AnalysisEvidence:
    """All validated inputs used to derive the public interpretation page."""

    controllers: Mapping[str, ControllerEvidence]
    input_sha256: Mapping[str, str]


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _verify_inputs(project_root: Path) -> dict[str, str]:
    identities: dict[str, str] = {}
    for relative_path, expected_sha256 in EXPECTED_INPUT_SHA256.items():
        path = project_root / relative_path
        try:
            metadata = path.lstat()
        except FileNotFoundError as error:
            raise AnalysisError(f"missing frozen M8 input: {relative_path}") from error
        if stat.S_ISLNK(metadata.st_mode) or not stat.S_ISREG(metadata.st_mode):
            raise AnalysisError(
                f"frozen M8 input must be a non-symlink regular file: {relative_path}"
            )
        actual_sha256 = _sha256_file(path)
        if actual_sha256 != expected_sha256:
            raise AnalysisError(
                f"frozen M8 input identity changed for {relative_path}: "
                f"expected {expected_sha256}, got {actual_sha256}"
            )
        identities[relative_path] = actual_sha256
    return identities


def _read_csv(path: Path) -> tuple[dict[str, str], ...]:
    with path.open("r", encoding="utf-8", newline="") as stream:
        reader = csv.DictReader(stream)
        if reader.fieldnames is None or len(reader.fieldnames) != len(set(reader.fieldnames)):
            raise AnalysisError(f"CSV has missing or duplicate fields: {path}")
        rows = tuple(dict(row) for row in reader)
    if not rows or any(set(row) != set(reader.fieldnames) for row in rows):
        raise AnalysisError(f"CSV rows do not match one non-empty schema: {path}")
    return rows


def _scalar_text(value: np.ndarray, *, name: str) -> str:
    array = np.asarray(value)
    if array.shape != ():
        raise AnalysisError(f"{name} must be a scalar")
    scalar = array.item()
    if isinstance(scalar, bytes):
        try:
            return scalar.decode("ascii")
        except UnicodeDecodeError as error:
            raise AnalysisError(f"{name} must contain ASCII text") from error
    if isinstance(scalar, str):
        return scalar
    raise AnalysisError(f"{name} must contain text")


def _float(row: Mapping[str, str], field: str) -> float:
    try:
        value = float(row[field])
    except (KeyError, ValueError) as error:
        raise AnalysisError(f"invalid floating-point field {field!r}") from error
    if not math.isfinite(value):
        raise AnalysisError(f"field {field!r} must be finite")
    return value


def _int(row: Mapping[str, str], field: str) -> int:
    try:
        return int(row[field])
    except (KeyError, ValueError) as error:
        raise AnalysisError(f"invalid integer field {field!r}") from error


def _bool(row: Mapping[str, str], field: str) -> bool:
    try:
        value = row[field]
    except KeyError as error:
        raise AnalysisError(f"missing boolean field {field!r}") from error
    if value not in ("true", "false"):
        raise AnalysisError(f"field {field!r} must be canonical lowercase boolean text")
    return value == "true"


def _require_close(actual: float, expected: float, *, name: str) -> None:
    if not math.isclose(actual, expected, rel_tol=1.0e-12, abs_tol=1.0e-12):
        raise AnalysisError(f"{name} does not recompute: expected {expected}, got {actual}")


def _load_metrics(path: Path) -> dict[str, np.ndarray]:
    try:
        with np.load(path, allow_pickle=False) as archive:
            if set(archive.files) != _EXPECTED_METRIC_KEYS:
                raise AnalysisError(f"unexpected metrics.npz members: {path}")
            values = {name: np.array(archive[name], copy=True) for name in archive.files}
    except (OSError, ValueError) as error:
        raise AnalysisError(f"could not load non-pickled metrics archive: {path}") from error
    if any(value.dtype.hasobject for value in values.values()):
        raise AnalysisError(f"metrics archive contains an object array: {path}")
    return values


def _validate_controller(
    name: str,
    aggregate: Mapping[str, str],
    episodes: tuple[Mapping[str, str], ...],
    metrics: Mapping[str, np.ndarray],
) -> None:
    if aggregate.get("schema_version") != "controller-learning.m8-comparison.v1":
        raise AnalysisError(f"unexpected central schema for {name}")
    if aggregate.get("benchmark_version") != "0.1" or aggregate.get("controller_name") != name:
        raise AnalysisError(f"unexpected central identity for {name}")
    if aggregate.get("ranking_rule") != "success_rate_desc_then_mean_successful_lap_time_asc":
        raise AnalysisError("accepted ranking rule changed")
    if len(episodes) != _int(aggregate, "track_count") or len(episodes) != 20:
        raise AnalysisError(f"{name} must contain exactly 20 ordered episode rows")

    expected_schema = "controller-learning.m8-controller-results.v1"
    for row_index, row in enumerate(episodes):
        if (
            row.get("schema_version") != expected_schema
            or row.get("benchmark_version") != "0.1"
            or row.get("controller_name") != name
            or _int(row, "row_index") != row_index
            or _int(row, "reset_seed") != row_index
        ):
            raise AnalysisError(f"invalid {name} identity at episode row {row_index}")

    if _scalar_text(metrics["benchmark_version"], name="benchmark_version") != "0.1":
        raise AnalysisError(f"unexpected metrics benchmark version for {name}")
    if _scalar_text(metrics["controller_name"], name="controller_name") != name:
        raise AnalysisError(f"unexpected metrics Controller name for {name}")
    if np.asarray(metrics["schema_version"]).shape != () or int(metrics["schema_version"]) != 1:
        raise AnalysisError(f"unexpected metrics schema version for {name}")

    offsets = np.asarray(metrics["episode_offsets"])
    track_id = np.asarray(metrics["track_id"])
    reset_seed = np.asarray(metrics["reset_seed"])
    compute_time = np.asarray(metrics["compute_time_s"])
    speed = np.asarray(metrics["speed_mps"])
    lateral_error = np.asarray(metrics["lateral_error_m"])
    action = np.asarray(metrics["requested_action"])
    steering_saturated = np.asarray(metrics["steering_saturated"])
    longitudinal_saturated = np.asarray(metrics["longitudinal_saturated"])
    transition_count = _int(aggregate, "metric_transition_count")

    if (
        offsets.shape != (21,)
        or offsets.dtype.kind not in "iu"
        or int(offsets[0]) != 0
        or np.any(np.diff(offsets) <= 0)
        or int(offsets[-1]) != transition_count
    ):
        raise AnalysisError(f"invalid episode offsets for {name}")
    if track_id.shape != (20,) or reset_seed.shape != (20,):
        raise AnalysisError(f"invalid Track identity arrays for {name}")
    expected_track_ids = np.asarray([_int(row, "track_id") for row in episodes], dtype=np.uint32)
    expected_reset_seeds = np.asarray(
        [_int(row, "reset_seed") for row in episodes], dtype=np.uint32
    )
    if not np.array_equal(track_id, expected_track_ids) or not np.array_equal(
        reset_seed, expected_reset_seeds
    ):
        raise AnalysisError(f"CSV and NPZ Track identities differ for {name}")
    expected_steps = np.asarray([_int(row, "environment_steps") for row in episodes])
    if not np.array_equal(np.diff(offsets), expected_steps):
        raise AnalysisError(f"CSV and NPZ episode boundaries differ for {name}")

    one_dimensional = (
        compute_time,
        speed,
        lateral_error,
        steering_saturated,
        longitudinal_saturated,
    )
    if any(values.shape != (transition_count,) for values in one_dimensional):
        raise AnalysisError(f"invalid transition metric shapes for {name}")
    if action.shape != (transition_count, 2):
        raise AnalysisError(f"invalid requested_action shape for {name}")
    if not all(
        np.isfinite(values).all() for values in (compute_time, speed, lateral_error, action)
    ):
        raise AnalysisError(f"non-finite metric sample for {name}")

    successes = tuple(row for row in episodes if _bool(row, "success"))
    successful_laps = np.asarray([_float(row, "lap_time_s") for row in successes])
    if len(successes) != _int(aggregate, "success_count"):
        raise AnalysisError(f"success count does not recompute for {name}")
    _require_close(
        len(successes) / len(episodes),
        _float(aggregate, "success_rate"),
        name=f"{name} success rate",
    )
    _require_close(
        float(np.mean(successful_laps)),
        _float(aggregate, "mean_successful_lap_time_s"),
        name=f"{name} mean successful lap",
    )
    _require_close(
        float(np.mean(speed, dtype=np.float64)),
        _float(aggregate, "mean_speed_mps"),
        name=f"{name} mean speed",
    )
    _require_close(
        float(np.sqrt(np.mean(np.square(lateral_error), dtype=np.float64))),
        _float(aggregate, "lateral_error_rms_m"),
        name=f"{name} lateral RMS",
    )
    _require_close(
        float(np.quantile(np.abs(lateral_error), 0.95, method="linear")),
        _float(aggregate, "lateral_error_abs_p95_m"),
        name=f"{name} lateral absolute P95",
    )
    _require_close(
        float(np.max(np.abs(lateral_error))),
        _float(aggregate, "lateral_error_abs_max_m"),
        name=f"{name} lateral absolute maximum",
    )
    _require_close(
        float(np.mean(steering_saturated)),
        _float(aggregate, "steering_saturation_rate"),
        name=f"{name} steering saturation rate",
    )
    _require_close(
        float(np.mean(longitudinal_saturated)),
        _float(aggregate, "longitudinal_saturation_rate"),
        name=f"{name} longitudinal saturation rate",
    )

    deadline = _float(aggregate, "compute_deadline_s")
    for percentile, field in (
        (0.50, "compute_p50_s"),
        (0.95, "compute_p95_s"),
        (0.99, "compute_p99_s"),
    ):
        _require_close(
            float(np.quantile(compute_time, percentile, method="linear")),
            _float(aggregate, field),
            name=f"{name} {field}",
        )
    miss_count = int(np.count_nonzero(compute_time > deadline))
    if miss_count != _int(aggregate, "compute_deadline_miss_count"):
        raise AnalysisError(f"compute deadline misses do not recompute for {name}")
    _require_close(
        miss_count / transition_count,
        _float(aggregate, "compute_deadline_miss_rate"),
        name=f"{name} compute deadline miss rate",
    )


def load_analysis_evidence(project_root: str | Path = PROJECT_ROOT) -> AnalysisEvidence:
    """Load only the seven frozen M8 CSV/NPZ inputs and cross-check their aggregates."""

    root = Path(project_root).resolve(strict=True)
    identities = _verify_inputs(root)
    central_rows = _read_csv(root / "benchmarks/v0.1/m8_final_results.csv")
    if len(central_rows) != len(CONTROLLERS):
        raise AnalysisError("central comparison must contain exactly three Controller rows")
    central = {row.get("controller_name", ""): row for row in central_rows}
    if tuple(row.get("controller_name") for row in central_rows) != CONTROLLERS or set(
        central
    ) != set(CONTROLLERS):
        raise AnalysisError("central comparison Controller order or identity changed")
    if tuple(_int(row, "rank") for row in central_rows) != (1, 2, 3):
        raise AnalysisError("central comparison ranks changed")

    controllers: dict[str, ControllerEvidence] = {}
    for name in CONTROLLERS:
        base = root / f"results/0.1/{name}/m8-final-v0-1-002"
        episodes = _read_csv(base / "results.csv")
        metrics = _load_metrics(base / "metrics.npz")
        _validate_controller(name, central[name], episodes, metrics)
        controllers[name] = ControllerEvidence(
            aggregate=central[name],
            episodes=episodes,
            metrics=metrics,
        )
    return AnalysisEvidence(controllers=controllers, input_sha256=identities)


def _successful_laps(evidence: ControllerEvidence) -> np.ndarray:
    return np.asarray(
        [_float(row, "lap_time_s") for row in evidence.episodes if _bool(row, "success")],
        dtype=np.float64,
    )


def _paired_advantage(
    faster: ControllerEvidence,
    slower: ControllerEvidence,
) -> np.ndarray:
    differences: list[float] = []
    for faster_row, slower_row in zip(faster.episodes, slower.episodes, strict=True):
        if _int(faster_row, "track_id") != _int(slower_row, "track_id"):
            raise AnalysisError("paired Controller rows do not share Track identity")
        if _bool(faster_row, "success") and _bool(slower_row, "success"):
            differences.append(_float(slower_row, "lap_time_s") - _float(faster_row, "lap_time_s"))
    return np.asarray(differences, dtype=np.float64)


def render_analysis_figure(evidence: AnalysisEvidence) -> bytes:
    """Render fixed descriptive views of the accepted aggregates and sample distributions."""

    figure = Figure(figsize=(13.0, 4.8), dpi=120, facecolor="white")
    FigureCanvasAgg(figure)
    axes = figure.subplots(1, 3)

    ranking_axes = axes[0]
    for name in CONTROLLERS:
        row = evidence.controllers[name].aggregate
        lap_time = _float(row, "mean_successful_lap_time_s")
        success_rate = 100.0 * _float(row, "success_rate")
        ranking_axes.scatter(
            lap_time,
            success_rate,
            s=72.0,
            color=COLORS[name],
            edgecolor="white",
            linewidth=0.8,
            zorder=3,
        )
        offset = (6, -13) if name == "pid" else (6, 6)
        ranking_axes.annotate(
            LABELS[name],
            (lap_time, success_rate),
            xytext=offset,
            textcoords="offset points",
            color=COLORS[name],
            weight="bold",
        )
    ranking_axes.set_title("Outcome ranking inputs")
    ranking_axes.set_xlabel("Mean successful lap time [s] (lower is better)")
    ranking_axes.set_ylabel("Success rate [%] (ranked first)")
    ranking_axes.set_ylim(93.5, 100.7)
    ranking_axes.grid(alpha=0.22)

    percentiles = np.linspace(0.0, 100.0, 1001, dtype=np.float64)
    for name in CONTROLLERS:
        controller = evidence.controllers[name]
        speed = np.asarray(controller.metrics["speed_mps"], dtype=np.float64)
        lateral_error = np.abs(np.asarray(controller.metrics["lateral_error_m"], dtype=np.float64))
        axes[1].plot(
            np.quantile(speed, percentiles / 100.0, method="linear"),
            percentiles,
            color=COLORS[name],
            linewidth=1.8,
            label=LABELS[name],
        )
        axes[2].plot(
            np.quantile(lateral_error, percentiles / 100.0, method="linear"),
            percentiles,
            color=COLORS[name],
            linewidth=1.8,
            label=LABELS[name],
        )

    axes[1].set_title("Transition-weighted speed distribution")
    axes[1].set_xlabel("Post-step speed [m/s]")
    axes[1].set_ylabel("Percentile [%]")
    axes[1].set_ylim(0.0, 100.0)
    axes[1].grid(alpha=0.22)
    axes[1].legend(loc="lower right", frameon=True)

    axes[2].set_title("Transition-weighted tracking distribution")
    axes[2].set_xlabel("Absolute lateral error [m] (symmetric-log scale)")
    axes[2].set_ylabel("Percentile [%]")
    axes[2].set_xscale("symlog", linthresh=0.01, linscale=0.8)
    axes[2].set_ylim(0.0, 100.0)
    axes[2].grid(alpha=0.22)

    figure.suptitle("Accepted benchmark 0.1 M8 result — descriptive views", weight="bold")
    figure.text(
        0.5,
        0.015,
        "Source: frozen m8-final-v0-1-002 CSV/NPZ artifacts; no simulation or Test rerun",
        ha="center",
        fontsize=8.5,
        color="0.35",
    )
    figure.tight_layout(rect=(0.0, 0.05, 1.0, 0.93), w_pad=2.0)
    buffer = io.BytesIO()
    figure.canvas.print_png(buffer, metadata={"Software": "Controller Learning v0.1.1"})
    return buffer.getvalue()


def render_analysis_markdown(evidence: AnalysisEvidence, *, figure_sha256: str) -> bytes:
    """Render the checked descriptive narrative from validated M8 values."""

    rows = {name: evidence.controllers[name].aggregate for name in CONTROLLERS}
    laps = {name: _successful_laps(evidence.controllers[name]) for name in CONTROLLERS}
    pid_mpc = _paired_advantage(evidence.controllers["pid"], evidence.controllers["mpc"])
    ppo_pid = _paired_advantage(evidence.controllers["ppo"], evidence.controllers["pid"])
    ppo_mpc = _paired_advantage(evidence.controllers["ppo"], evidence.controllers["mpc"])
    ppo_failure = tuple(
        row for row in evidence.controllers["ppo"].episodes if not _bool(row, "success")
    )
    if len(ppo_failure) != 1:
        raise AnalysisError("the accepted PPO evidence must contain exactly one unsuccessful row")

    table_lines: list[str] = []
    for name in CONTROLLERS:
        row = rows[name]
        table_lines.append(
            "| {rank} | {label} | {success}/20 | {mean_lap:.3f} s | {median_lap:.3f} s | "
            "{speed:.3f} m/s | {lateral:.4f} m | {p99:.3f} ms |".format(
                rank=_int(row, "rank"),
                label=LABELS[name],
                success=_int(row, "success_count"),
                mean_lap=_float(row, "mean_successful_lap_time_s"),
                median_lap=float(np.median(laps[name])),
                speed=_float(row, "mean_speed_mps"),
                lateral=_float(row, "lateral_error_rms_m"),
                p99=1000.0 * _float(row, "compute_p99_s"),
            )
        )

    identity_lines = [
        f"| `{path}` | `{digest}` |" for path, digest in evidence.input_sha256.items()
    ]
    ppo_time_fraction = _float(rows["ppo"], "mean_successful_lap_time_s") / _float(
        rows["pid"], "mean_successful_lap_time_s"
    )
    speed_ratio = _float(rows["ppo"], "mean_speed_mps") / _float(rows["pid"], "mean_speed_mps")
    lateral_ratio = _float(rows["ppo"], "lateral_error_rms_m") / _float(
        rows["pid"], "lateral_error_rms_m"
    )
    failure = ppo_failure[0]
    result_table_header = (
        "| Rank | Controller | Success | Mean successful lap | Median successful lap | "
        "Mean speed | Lateral RMS | Compute P99 |"
    )

    markdown = f"""<!-- Generated by scripts/analyze_m8_results.py. Do not edit by hand. -->
# Reading the benchmark 0.1 result

!!! info "Interpretation, not new benchmark evidence"
    This page deterministically describes the already accepted `m8-final-v0-1-002` artifacts.
    Its build path reads only the seven hash-pinned CSV/NPZ files listed below. It does not load a
    Track asset, create an Environment, run a Controller, or access Test again.

The benchmark ranks **success rate first**, then mean lap time over successful episodes. That rule
is essential to reading the result: PPO recorded much shorter successful laps, but its 19/20
completion rate places it behind the two 20/20 Controllers.

{result_table_header}
| ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: |
{chr(10).join(table_lines)}

![Descriptive M8 success, speed, and lateral-error views](assets/m8_result_analysis.png)

## What the accepted samples show

- **PID and MPC occupy the reliable, lower-speed end of this particular comparison.** Both
  completed all 20 Tracks. PID was faster than MPC on every paired Track, by
  {float(np.mean(pid_mpc)):.3f} s on average (paired median {float(np.median(pid_mpc)):.3f} s).
- **PPO occupies the higher-speed, lower-tracking-margin end.** Its mean successful lap time was
  {100.0 * ppo_time_fraction:.1f}% of PID's and its transition-weighted mean speed was
  {speed_ratio:.2f} times PID's. It was faster on all {ppo_pid.size} Tracks where both PPO and PID
  succeeded, and on all {ppo_mpc.size} shared successes with MPC.
- **The speed came with a visibly wider tracking-error distribution.** PPO's transition-weighted
  lateral RMS was {lateral_ratio:.1f} times PID's. Its single unsuccessful episode was an
  `{failure["termination_reason_name"]}` termination on row {_int(failure, "row_index")}, Track ID
  `{_int(failure, "track_id")}`; that episode is included in the transition distributions.
- **Runtime cost differs sharply by implementation.** PID and exported NumPy PPO had sub-millisecond
  compute P99 values. MPC's P99 was {_float(rows["mpc"], "compute_p99_s") * 1000.0:.3f} ms against
  the 50 ms diagnostic deadline, with {_int(rows["mpc"], "compute_deadline_miss_count")} misses in
  {_int(rows["mpc"], "compute_sample_count"):,} calls. Timing is hardware- and run-specific; it is
  not a general real-time guarantee.

These observations make the table useful for teaching: the ranking deliberately refuses to trade
one failure for faster successful laps, while the auxiliary metrics expose the behavior hidden by
rank alone. They do **not** show that PID is generally superior to reinforcement learning, that PPO
is generally faster, or that these three points establish a controller-family Pareto frontier.

## Scope and limitations

This is a descriptive comparison of three frozen example implementations and configurations on one
fixed 20-Track benchmark set. It is not a matched-speed ablation, a causal study of algorithm
families, or a confidence claim about a broader population. Transition distributions weight every
recorded simulation step, so longer episodes contribute more samples. Lap-time summaries contain
successful episodes only. Read the [Evaluation Protocol](evaluation.md) for the exact metric,
ranking, split, and attempt-lineage definitions.

No Controller or benchmark configuration was changed from this interpretation. Future Controller
work must use Train and Validation; these accepted Test outcomes are reporting data, not tuning
feedback.

## Deterministic derivation

Regenerate the page and figure, then verify that the committed bytes are current:

```bash
pixi run build-result-analysis
pixi run check-result-analysis
```

The figure SHA-256 is `{figure_sha256}`. The generator recomputes central success, lap-time, speed,
lateral-error, saturation, timing, Track-order, reset-seed, and episode-boundary claims from the
underlying rows and arrays before it writes anything.

| Frozen input | SHA-256 |
| --- | --- |
{chr(10).join(identity_lines)}
"""
    return markdown.encode("utf-8")


def build_analysis(project_root: str | Path = PROJECT_ROOT) -> tuple[bytes, bytes]:
    """Return deterministic Markdown and PNG bytes after complete evidence validation."""

    evidence = load_analysis_evidence(project_root)
    figure = render_analysis_figure(evidence)
    figure_sha256 = hashlib.sha256(figure).hexdigest()
    markdown = render_analysis_markdown(evidence, figure_sha256=figure_sha256)
    return markdown, figure


def _atomic_write(path: Path, content: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="wb",
            dir=path.parent,
            prefix=f".{path.name}.",
            suffix=".tmp",
            delete=False,
        ) as temporary:
            temporary_path = Path(temporary.name)
            temporary.write(content)
            temporary.flush()
            os.fchmod(temporary.fileno(), 0o644)
            os.fsync(temporary.fileno())
        os.replace(temporary_path, path)
    finally:
        if temporary_path is not None:
            temporary_path.unlink(missing_ok=True)


def write_analysis_outputs(
    markdown: bytes,
    figure: bytes,
    *,
    project_root: str | Path = PROJECT_ROOT,
) -> None:
    """Atomically replace the two derived, non-benchmark documentation outputs."""

    root = Path(project_root).resolve(strict=True)
    _atomic_write(root / OUTPUT_FIGURE, figure)
    _atomic_write(root / OUTPUT_MARKDOWN, markdown)


def check_analysis_outputs(
    markdown: bytes,
    figure: bytes,
    *,
    project_root: str | Path = PROJECT_ROOT,
) -> None:
    """Require committed documentation bytes to match a fresh deterministic derivation."""

    root = Path(project_root).resolve(strict=True)
    for relative_path, expected in ((OUTPUT_MARKDOWN, markdown), (OUTPUT_FIGURE, figure)):
        path = root / relative_path
        if not path.is_file() or path.is_symlink():
            raise AnalysisError(f"generated output is missing or unsafe: {relative_path}")
        if path.read_bytes() != expected:
            raise AnalysisError(
                f"generated output is stale: {relative_path}; run pixi run build-result-analysis"
            )


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--write", action="store_true", help="replace the two derived doc outputs")
    mode.add_argument("--check", action="store_true", help="verify committed output bytes")
    return parser


def main(argv: Sequence[str] | None = None) -> None:
    """Write or check deterministic analysis outputs without executing a benchmark."""

    parser = _build_parser()
    values = parser.parse_args(argv)
    try:
        markdown, figure = build_analysis()
        if values.write:
            write_analysis_outputs(markdown, figure)
            mode = "written"
        else:
            check_analysis_outputs(markdown, figure)
            mode = "checked"
    except (AnalysisError, OSError, ValueError) as error:
        parser.exit(2, f"analyze-m8-results: error: {error}\n")
    print(
        json.dumps(
            {
                "status": mode,
                "simulation_executed": False,
                "input_count": len(EXPECTED_INPUT_SHA256),
                "markdown": {
                    "path": OUTPUT_MARKDOWN.as_posix(),
                    "sha256": hashlib.sha256(markdown).hexdigest(),
                    "size_bytes": len(markdown),
                },
                "figure": {
                    "path": OUTPUT_FIGURE.as_posix(),
                    "sha256": hashlib.sha256(figure).hexdigest(),
                    "size_bytes": len(figure),
                },
            },
            allow_nan=False,
            sort_keys=True,
            separators=(",", ":"),
        )
    )


if __name__ == "__main__":
    main()
