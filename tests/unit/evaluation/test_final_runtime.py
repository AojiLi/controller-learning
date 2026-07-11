"""Tests for measured and privacy-safe M8 runtime evidence."""

from __future__ import annotations

import copy
import os
import subprocess
from collections.abc import Sequence
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

import controller_learning.evaluation.final_runtime as runtime_module
from controller_learning.evaluation.final_runtime import (
    FINAL_RUNTIME_PACKAGE_NAMES,
    FinalMemoryRecorder,
    FinalRuntimeEvidenceError,
    MeasuredFinalEnvironmentFactory,
    collect_final_runtime_evidence,
    resolve_nvidia_smi_executable,
    runtime_evidence_is_private_safe,
    validate_final_memory_evidence,
    validate_final_runtime_evidence,
)

GPU_0_UUID = "GPU-12345678-1234-1234-1234-123456789abc"
GPU_1_UUID = "GPU-abcdef12-abcd-abcd-abcd-abcdef123456"


def _fake_nvidia_smi(tmp_path: Path) -> str:
    executable = tmp_path / "nvidia-smi"
    executable.write_bytes(b"#!/bin/sh\nexit 0\n")
    executable.chmod(0o755)
    return str(executable.resolve(strict=True))


class _FakeEnvironment:
    def __init__(self) -> None:
        self.unwrapped = object()
        self.events: list[tuple[Any, ...]] = []

    def reset(self, *args: Any, **kwargs: Any) -> tuple[str, dict[str, int]]:
        self.events.append(("reset", args, kwargs))
        return "observation", {"track_id": 2_000_000}

    def step(self, action: object) -> tuple[str, float, bool, bool, dict[str, int]]:
        self.events.append(("step", action))
        return "next", 1.0, True, False, {"track_id": 2_000_000}

    def render(self, *args: Any, **kwargs: Any) -> str:
        self.events.append(("render", args, kwargs))
        return "frame"

    def close(self) -> None:
        self.events.append(("close",))


def test_measured_environment_enforces_one_instance_and_exact_closed_lifecycle(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    clock = iter((10.0, 10.25, 20.0, 21.5, 30.0, 32.25))
    monkeypatch.setattr(runtime_module.time, "perf_counter", lambda: next(clock))
    created: list[dict[str, object]] = []
    raw = _FakeEnvironment()

    def factory(**kwargs: object) -> _FakeEnvironment:
        created.append(dict(kwargs))
        return raw

    measured_factory = MeasuredFinalEnvironmentFactory(factory)
    environment = measured_factory.create(backend="mjx_warp", num_envs=1)
    assert created == [{"backend": "mjx_warp", "num_envs": 1}]
    assert environment.unwrapped is raw.unwrapped
    assert environment.reset(seed=0, options={"track_index": 0})[0] == "observation"
    assert environment.step([0.0, 1.0])[0] == "next"
    assert environment.render(mode="rgb_array") == "frame"

    with pytest.raises(RuntimeError, match="only after close"):
        measured_factory.evidence(expected_resets=1, expected_steps=1)
    environment.close()

    evidence = measured_factory.evidence(expected_resets=1, expected_steps=1)
    assert evidence["environment_instance_count"] == 1
    assert evidence["environment_create_wall_time_s"] == pytest.approx(0.25)
    assert evidence["first_reset_wall_time_including_lazy_compilation_s"] == pytest.approx(1.5)
    assert evidence["first_step_wall_time_including_lazy_compilation_s"] == pytest.approx(2.25)
    assert evidence["reset_count"] == evidence["expected_reset_count"] == 1
    assert evidence["step_count"] == evidence["expected_step_count"] == 1
    assert evidence["close_count"] == 1
    assert "include any lazy compilation" in evidence["method"]
    assert raw.events[-1] == ("close",)

    with pytest.raises(RuntimeError, match="exactly one environment"):
        measured_factory.create()
    with pytest.raises(RuntimeError, match="closed exactly once"):
        environment.close()
    with pytest.raises(RuntimeError, match="already closed"):
        environment.reset()


def test_measured_environment_rejects_missing_calls_and_count_drift() -> None:
    measured_factory = MeasuredFinalEnvironmentFactory(_FakeEnvironment)
    environment = measured_factory.create()
    environment.close()
    with pytest.raises(RuntimeError, match="reset count"):
        measured_factory.evidence(expected_resets=60, expected_steps=0)

    second_factory = MeasuredFinalEnvironmentFactory(_FakeEnvironment)
    second = second_factory.create()
    second.reset()
    second.step([0.0, 0.0])
    second.close()
    with pytest.raises(RuntimeError, match="step count"):
        second_factory.evidence(expected_resets=1, expected_steps=2)


class _FakeJax:
    def __init__(self, device: object | None = None) -> None:
        self.barrier_count = 0
        self._device = device

    def effects_barrier(self) -> None:
        self.barrier_count += 1

    def devices(self, platform_name: str) -> list[object]:
        assert platform_name == "gpu"
        return [] if self._device is None else [self._device]


class _MemoryDevice:
    def __init__(self, statistics: Sequence[dict[str, int]]) -> None:
        self._statistics = iter(statistics)

    def memory_stats(self) -> dict[str, int]:
        return next(self._statistics)


def test_final_memory_recorder_measures_selected_process_and_allocator_peaks(
    tmp_path: Path,
) -> None:
    jax = _FakeJax()
    device = _MemoryDevice(
        (
            {"bytes_in_use": 10, "peak_bytes_in_use": 20},
            {"bytes_in_use": 0, "peak_bytes_in_use": 30},
        )
    )
    calls: list[tuple[str, ...]] = []
    outputs = iter(
        (
            f"{GPU_1_UUID}, {os.getpid()}, 999\n{GPU_0_UUID}, {os.getpid()}, 100",
            f"{GPU_0_UUID}, 7, 999\n{GPU_0_UUID}, {os.getpid()}, 120",
        )
    )

    def command_runner(command: Sequence[str]) -> str:
        calls.append(tuple(command))
        return next(outputs)

    executable = _fake_nvidia_smi(tmp_path)
    recorder = FinalMemoryRecorder(
        jax,
        device,
        GPU_0_UUID,
        executable,
        command_runner=command_runner,
    )
    first = recorder.sample("after_environment_create")
    second = recorder.sample("after_environment_close")
    evidence = recorder.evidence()

    assert first["process_vram_mib"] == 100.0
    assert second["jax_bytes_in_use"] == 0
    assert jax.barrier_count == 2
    assert len(calls) == 2
    assert all(call[0] == executable and Path(call[0]).is_absolute() for call in calls)
    assert all("--query-compute-apps=gpu_uuid,pid,used_gpu_memory" in call for call in calls)
    assert evidence["sample_count"] == 2
    assert evidence["peak_sampled_process_vram_mib"] == 120.0
    assert evidence["peak_jax_allocator_bytes"] == 30
    assert evidence["final_jax_live_bytes"] == 0
    assert runtime_evidence_is_private_safe(evidence)
    assert GPU_0_UUID not in repr(evidence)


def test_final_memory_recorder_rejects_missing_process_and_invalid_allocator(
    tmp_path: Path,
) -> None:
    executable = _fake_nvidia_smi(tmp_path)
    recorder = FinalMemoryRecorder(
        _FakeJax(),
        _MemoryDevice(({"bytes_in_use": 1, "peak_bytes_in_use": 2},)),
        GPU_0_UUID,
        executable,
        command_runner=lambda _command: f"{GPU_0_UUID}, {os.getpid() + 1}, 100",
    )
    with pytest.raises(RuntimeError, match="did not report"):
        recorder.sample("missing_process")

    invalid = FinalMemoryRecorder(
        _FakeJax(),
        _MemoryDevice(({"bytes_in_use": 3, "peak_bytes_in_use": 2},)),
        GPU_0_UUID,
        executable,
        command_runner=lambda _command: f"{GPU_0_UUID}, {os.getpid()}, 100",
    )
    with pytest.raises(RuntimeError, match="smaller than live"):
        invalid.sample("invalid_allocator")

    empty = FinalMemoryRecorder(
        _FakeJax(),
        _MemoryDevice(()),
        GPU_0_UUID,
        executable,
        command_runner=lambda _command: "",
    )
    with pytest.raises(RuntimeError, match="at least one"):
        empty.evidence()

    zero = FinalMemoryRecorder(
        _FakeJax(),
        _MemoryDevice(({"bytes_in_use": 0, "peak_bytes_in_use": 0},)),
        GPU_0_UUID,
        executable,
        command_runner=lambda _command: f"{GPU_0_UUID}, {os.getpid()}, 0",
    )
    with pytest.raises(RuntimeError, match="must be positive"):
        zero.sample("zero_process_memory")


def test_default_nvidia_smi_runner_uses_exact_minimal_environment(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    executable = _fake_nvidia_smi(tmp_path)
    captured: dict[str, object] = {}
    monkeypatch.setenv("LD_PRELOAD", "/tmp/untrusted-preload.so")
    monkeypatch.setenv("LD_LIBRARY_PATH", "/tmp/untrusted-libraries")

    def run(command: Sequence[str], **kwargs: object) -> SimpleNamespace:
        captured["command"] = tuple(command)
        captured.update(kwargs)
        return SimpleNamespace(stdout="memory evidence\n")

    monkeypatch.setattr(runtime_module.subprocess, "run", run)

    output = runtime_module._default_command_runner((executable, "--fixed-query"))

    assert output == "memory evidence\n"
    assert set(captured) == {
        "capture_output",
        "check",
        "close_fds",
        "command",
        "cwd",
        "env",
        "shell",
        "start_new_session",
        "stdin",
        "text",
        "timeout",
    }
    assert captured["command"] == (executable, "--fixed-query")
    assert captured["env"] == {
        "HOME": "/nonexistent",
        "LANG": "C",
        "LC_ALL": "C",
    }
    assert captured["stdin"] == subprocess.DEVNULL
    assert captured["capture_output"] is True
    assert captured["cwd"] is None
    assert captured["check"] is True
    assert captured["close_fds"] is True
    assert captured["shell"] is False
    assert captured["start_new_session"] is True
    assert captured["text"] is True
    assert captured["timeout"] == 15


def _runtime_device() -> SimpleNamespace:
    return SimpleNamespace(
        id=0,
        platform="gpu",
        device_kind="NVIDIA Test GPU",
        client=SimpleNamespace(platform_version="CUDA 12.8.0"),
    )


def _collect_runtime(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> tuple[dict[str, Any], str]:
    monkeypatch.setenv("CUDA_DEVICE_ORDER", "PCI_BUS_ID")
    monkeypatch.setenv("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", GPU_1_UUID)
    monkeypatch.setattr(runtime_module.platform, "system", lambda: "Linux")
    monkeypatch.setattr(runtime_module.platform, "machine", lambda: "x86_64")
    monkeypatch.setattr(runtime_module.platform, "python_version", lambda: "3.11.9")
    monkeypatch.setattr(runtime_module.platform, "release", lambda: "6.8.0-test")
    monkeypatch.setattr(runtime_module, "_cpu_model", lambda: "Test CPU")

    executable = _fake_nvidia_smi(tmp_path)

    def command_runner(command: Sequence[str]) -> str:
        assert command[0] == executable
        assert Path(command[0]).is_absolute()
        assert "--query-gpu=index,uuid,name,driver_version,memory.total" in command
        return (
            f"0, {GPU_0_UUID}, NVIDIA First GPU, 570.1, 24000\n"
            f"1, {GPU_1_UUID}, NVIDIA Test GPU, 570.2, 48000"
        )

    versions: list[str] = []

    def package_version(name: str) -> str:
        versions.append(name)
        return "1.2.3"

    result = collect_final_runtime_evidence(
        _FakeJax(_runtime_device()),
        executable,
        command_runner=command_runner,
        package_version=package_version,
    )
    assert tuple(versions) == FINAL_RUNTIME_PACKAGE_NAMES
    return result


def test_collect_final_runtime_evidence_is_strict_mapped_and_redacted(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    public, private_uuid = _collect_runtime(monkeypatch, tmp_path)

    assert private_uuid == GPU_1_UUID
    assert public["selected_gpu"] == {
        "index": 1,
        "uuid": "redacted",
        "name": "NVIDIA Test GPU",
        "driver_version": "570.2",
        "memory_total_mib": 48000.0,
    }
    assert public["cuda_runtime"] == "CUDA 12.8.0"
    assert public["cuda_driver"] == "570.2"
    assert public["jax_device"]["id"] == 0
    assert public["cpu_model"] == "Test CPU"
    assert runtime_evidence_is_private_safe(public)
    assert GPU_1_UUID not in repr(public)
    validate_final_runtime_evidence(public)


def test_cpu_model_prefers_linux_model_name_over_architecture_fallback(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        runtime_module.Path,
        "read_text",
        lambda _self, **_kwargs: "processor: 0\nmodel name: Test Xeon CPU\n",
    )
    monkeypatch.setattr(runtime_module.platform, "processor", lambda: "x86_64")
    assert runtime_module._cpu_model() == "Test Xeon CPU"


@pytest.mark.parametrize(
    "mutation",
    (
        lambda report: report["selected_gpu"].update(uuid=GPU_0_UUID),
        lambda report: report["selected_gpu"].update(uuid=GPU_0_UUID.lower()),
        lambda report: report.update(cpu_model="/home/user/private/cpu.txt"),
        lambda report: report.update(cpu_model="file:///home/user/private/cpu.txt"),
        lambda report: report["selected_gpu"].update(memory_total_mib=float("nan")),
        lambda report: report.update(schema_version=True),
        lambda report: report["jax_device"].update(id=False),
        lambda report: report.pop("kernel"),
        lambda report: report["packages"].pop("torch"),
    ),
)
def test_runtime_validator_rejects_private_nonfinite_or_missing_evidence(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    mutation: Any,
) -> None:
    public, _private_uuid = _collect_runtime(monkeypatch, tmp_path)
    changed = copy.deepcopy(public)
    mutation(changed)
    with pytest.raises(FinalRuntimeEvidenceError):
        validate_final_runtime_evidence(changed)


def test_memory_validator_rejects_private_nonfinite_or_missing_evidence() -> None:
    evidence = {
        "schema_version": 1,
        "sampling_method": "synchronized process and allocator sampling",
        "sample_count": 1,
        "samples": [
            {
                "label": "final",
                "synchronized": True,
                "process_vram_mib": 10.0,
                "jax_bytes_in_use": 0,
                "jax_peak_bytes_in_use": 20,
            }
        ],
        "peak_sampled_process_vram_mib": 10.0,
        "peak_jax_allocator_bytes": 20,
        "final_jax_live_bytes": 0,
    }
    validate_final_memory_evidence(evidence)
    for mutate in (
        lambda value: value["samples"][0].update(label=GPU_0_UUID),
        lambda value: value["samples"][0].update(process_vram_mib=float("inf")),
        lambda value: value.pop("samples"),
    ):
        changed = copy.deepcopy(evidence)
        mutate(changed)
        with pytest.raises(FinalRuntimeEvidenceError):
            validate_final_memory_evidence(changed)


def test_collect_runtime_rejects_cuda_policy_or_non_gpu_device(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    monkeypatch.setattr(runtime_module.platform, "system", lambda: "Linux")
    monkeypatch.setattr(runtime_module.platform, "machine", lambda: "x86_64")
    monkeypatch.setattr(runtime_module.platform, "python_version", lambda: "3.11.9")
    monkeypatch.delenv("CUDA_DEVICE_ORDER", raising=False)
    monkeypatch.setenv("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
    with pytest.raises(RuntimeError, match="CUDA environment policy"):
        collect_final_runtime_evidence(
            _FakeJax(_runtime_device()),
            _fake_nvidia_smi(tmp_path),
        )


def test_nvidia_smi_resolver_returns_one_absolute_regular_executable(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    executable = _fake_nvidia_smi(tmp_path)
    monkeypatch.setattr(runtime_module.shutil, "which", lambda _value: executable)

    resolved = resolve_nvidia_smi_executable()

    assert resolved == executable
    assert Path(resolved).is_absolute()


def test_runtime_consumers_reject_relative_or_symlink_executable(
    tmp_path: Path,
) -> None:
    executable = Path(_fake_nvidia_smi(tmp_path))
    symlink = tmp_path / "nvidia-smi-link"
    symlink.symlink_to(executable)

    with pytest.raises(ValueError, match="absolute pre-resolved"):
        FinalMemoryRecorder(
            _FakeJax(),
            _MemoryDevice(()),
            GPU_0_UUID,
            "nvidia-smi",
        )
    with pytest.raises(ValueError, match="non-symlink regular"):
        FinalMemoryRecorder(
            _FakeJax(),
            _MemoryDevice(()),
            GPU_0_UUID,
            symlink,
        )
