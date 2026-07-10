"""Synthetic CUDA tests for first-terminal M7 selection collection; no official assets are read."""

from __future__ import annotations

import importlib
from types import SimpleNamespace
from typing import Any

import numpy as np
import pytest

from scripts.benchmark_m7_ppo import evaluate_first_terminal_rows

pytestmark = pytest.mark.gpu


def _torch() -> Any:
    return importlib.import_module("torch")


class _SyntheticSelectionEnv:
    num_envs = 100

    def __init__(self, torch: Any, *, initial_progress: float = 0.0) -> None:
        self.torch = torch
        self.device = torch.device("cuda:0")
        self.single_action_space = SimpleNamespace(
            low=np.asarray((-0.5, -4.0), dtype=np.float32),
            high=np.asarray((0.5, 3.0), dtype=np.float32),
        )
        self.initial_progress = initial_progress
        self.step_calls = 0
        self.actions: list[Any] = []
        self.reset_seed: int | None = None
        self.reset_indices: np.ndarray[Any, Any] | None = None
        self.track_ids = torch.tensor(
            list(range(2_000, 2_100)),
            dtype=torch.uint32,
            device=self.device,
        )
        self.terminal_step = (
            torch.arange(100, dtype=torch.int32, device=self.device).remainder(3) + 1
        )

    def _observation(self, progress: Any) -> dict[str, Any]:
        return {
            "features": self.torch.zeros((100, 8), dtype=self.torch.float32, device=self.device),
            "track_progress": progress,
        }

    def reset(self, *, seed: int, options: dict[str, Any]) -> tuple[dict[str, Any], dict[str, Any]]:
        self.reset_seed = seed
        self.reset_indices = options["track_indices"]
        self.step_calls = 0
        progress = self.torch.full(
            (100,),
            self.initial_progress,
            dtype=self.torch.float32,
            device=self.device,
        )
        return self._observation(progress), {"track_id": self.track_ids.clone()}

    def step(self, action: Any) -> tuple[Any, Any, Any, Any, Any]:
        self.step_calls += 1
        self.actions.append(action.clone())
        terminated = self.terminal_step == self.step_calls
        truncated = self.torch.zeros(100, dtype=self.torch.bool, device=self.device)
        successes = terminated & (
            self.torch.arange(100, dtype=self.torch.int32, device=self.device).remainder(2) == 0
        )
        reasons = self.torch.where(
            successes,
            self.torch.ones(100, dtype=self.torch.int32, device=self.device),
            self.torch.where(
                terminated,
                self.torch.full((100,), 2, dtype=self.torch.int32, device=self.device),
                self.torch.zeros(100, dtype=self.torch.int32, device=self.device),
            ),
        )
        lap_times = self.torch.where(
            successes,
            self.torch.full(
                (100,),
                self.step_calls * 0.05,
                dtype=self.torch.float32,
                device=self.device,
            ),
            self.torch.zeros(100, dtype=self.torch.float32, device=self.device),
        )
        progress = self.torch.full(
            (100,),
            self.step_calls / 3.0,
            dtype=self.torch.float32,
            device=self.device,
        )
        return (
            self._observation(progress),
            self.torch.zeros(100, dtype=self.torch.float32, device=self.device),
            terminated,
            truncated,
            {
                "lap_completed": successes,
                "lap_time_s": lap_times,
                "termination_reason": reasons,
                "track_id": self.track_ids.clone(),
            },
        )


def test_first_terminal_collection_preserves_order_and_masks_finished_worlds() -> None:
    torch = _torch()
    environment = _SyntheticSelectionEnv(torch)
    expected_track_ids = tuple(range(2_000, 2_100))

    def action_provider(features: Any) -> Any:
        action = torch.empty((features.shape[0], 2), dtype=torch.float32, device=features.device)
        action[:, 0] = 0.25
        action[:, 1] = 1.0
        return action

    rows = evaluate_first_terminal_rows(
        environment,
        action_provider,
        expected_track_ids=expected_track_ids,
        reset_seed=7,
        max_vector_steps=4,
        control_dt_s=0.05,
        torch_module=torch,
    )
    assert environment.reset_seed == 7
    assert isinstance(environment.reset_indices, np.ndarray)
    assert environment.reset_indices.dtype == np.int32
    assert np.array_equal(environment.reset_indices, np.arange(100, dtype=np.int32))
    assert environment.step_calls == 3
    assert [row.track_id for row in rows] == list(expected_track_ids)
    assert [row.steps for row in rows] == [index % 3 + 1 for index in range(100)]
    assert [row.termination_reason for row in rows] == [
        1 if index % 2 == 0 else 2 for index in range(100)
    ]
    assert all(0.0 <= row.max_progress <= 1.0 for row in rows)

    terminal_after_first = torch.arange(100, device="cuda").remainder(3) == 0
    assert torch.all(environment.actions[1][terminal_after_first] == 0.0)
    terminal_after_second = torch.arange(100, device="cuda").remainder(3) <= 1
    assert torch.all(environment.actions[2][terminal_after_second] == 0.0)


@pytest.mark.parametrize("bad_action", [float("nan"), 0.75])
def test_first_terminal_collection_rejects_nonfinite_or_out_of_bounds_actions(
    bad_action: float,
) -> None:
    torch = _torch()
    environment = _SyntheticSelectionEnv(torch)

    def action_provider(features: Any) -> Any:
        action = torch.zeros((100, 2), dtype=torch.float32, device=features.device)
        action[0, 0] = bad_action
        return action

    with pytest.raises(ValueError, match="finite and inside physical bounds"):
        evaluate_first_terminal_rows(
            environment,
            action_provider,
            expected_track_ids=tuple(range(2_000, 2_100)),
            reset_seed=7,
            max_vector_steps=4,
            control_dt_s=0.05,
            torch_module=torch,
        )


def test_first_terminal_collection_rejects_progress_instead_of_clamping() -> None:
    torch = _torch()
    environment = _SyntheticSelectionEnv(torch, initial_progress=1.01)
    with pytest.raises(ValueError, match=r"track_progress.*\[0, 1\]"):
        evaluate_first_terminal_rows(
            environment,
            lambda features: torch.zeros((100, 2), dtype=torch.float32, device=features.device),
            expected_track_ids=tuple(range(2_000, 2_100)),
            reset_seed=7,
            max_vector_steps=4,
            control_dt_s=0.05,
            torch_module=torch,
        )


def test_first_terminal_collection_rejects_inconsistent_public_terminal_info() -> None:
    torch = _torch()
    environment = _SyntheticSelectionEnv(torch)
    original_step = environment.step

    def inconsistent_step(action: Any) -> tuple[Any, Any, Any, Any, Any]:
        observation, reward, terminated, truncated, info = original_step(action)
        info["lap_completed"] = info["lap_completed"].clone()
        info["lap_completed"][0] = False
        return observation, reward, terminated, truncated, info

    environment.step = inconsistent_step  # type: ignore[method-assign]
    with pytest.raises(ValueError, match="terminal reason, flags, success"):
        evaluate_first_terminal_rows(
            environment,
            lambda features: torch.zeros((100, 2), dtype=torch.float32, device=features.device),
            expected_track_ids=tuple(range(2_000, 2_100)),
            reset_seed=7,
            max_vector_steps=4,
            control_dt_s=0.05,
            torch_module=torch,
        )
