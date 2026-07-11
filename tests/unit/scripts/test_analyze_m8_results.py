"""Tests for evidence-only deterministic M8 result interpretation."""

from __future__ import annotations

import hashlib
import inspect
from pathlib import Path

from scripts import analyze_m8_results as analysis

PROJECT_ROOT = Path(__file__).resolve().parents[3]


def test_analysis_build_is_deterministic_and_descriptive_only() -> None:
    first_markdown, first_figure = analysis.build_analysis(PROJECT_ROOT)
    second_markdown, second_figure = analysis.build_analysis(PROJECT_ROOT)

    assert first_markdown == second_markdown
    assert first_figure == second_figure
    assert first_figure.startswith(b"\x89PNG\r\n\x1a\n")
    text = first_markdown.decode("utf-8")
    assert "Interpretation, not new benchmark evidence" in text
    assert "do **not** show that PID is generally superior" in text
    assert "create an Environment, run a Controller, or access Test again" in text
    assert hashlib.sha256(first_figure).hexdigest() in text
    assert all(digest in text for digest in analysis.EXPECTED_INPUT_SHA256.values())


def test_committed_analysis_outputs_match_frozen_inputs() -> None:
    markdown, figure = analysis.build_analysis(PROJECT_ROOT)
    analysis.check_analysis_outputs(markdown, figure, project_root=PROJECT_ROOT)


def test_analysis_script_cannot_execute_or_import_benchmark_layers() -> None:
    source = inspect.getsource(analysis)

    assert "controller_learning.control" not in source
    assert "controller_learning.envs" not in source
    assert "controller_learning.evaluation" not in source
    assert "controller_learning.physics" not in source
    assert "controller_learning.rl" not in source
    assert "controller_learning.tracks" not in source
    assert "subprocess" not in source
