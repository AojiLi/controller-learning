"""Zero-copy JAX-to-Torch bridge for the official vector Challenge.

The bridge deliberately converts only numerical device arrays through DLPack.  The public
``benchmark_version`` info field is a read-only NumPy string array and remains on the host because
DLPack has no string dtype.  Importing this module does not import PyTorch; PyTorch is loaded only
when :class:`JaxToTorchVecEnv` is constructed in the GPU environment.
"""

from __future__ import annotations

import importlib
from collections.abc import Mapping
from typing import Any

import gymnasium as gym
import jax
import numpy as np
from gymnasium.vector import AutoresetMode, VectorEnv

from controller_learning.envs.episode import PUBLIC_INFO_KEYS
from controller_learning.envs.vector_racing import VecCarRacingEnv

_BENCHMARK_VERSION_KEY = "benchmark_version"
_NUMERIC_INFO_DTYPES = {
    "episode_seed": np.dtype(np.uint32),
    "controller_seed": np.dtype(np.uint32),
    "track_id": np.dtype(np.uint32),
    "termination_reason": np.dtype(np.int32),
    "lap_completed": np.dtype(np.bool_),
    "lap_time_s": np.dtype(np.float32),
}


def _load_torch() -> Any:
    """Import PyTorch only when the GPU bridge is actually constructed."""

    try:
        return importlib.import_module("torch")
    except ModuleNotFoundError as error:
        raise gym.error.DependencyNotInstalled(
            "PyTorch is required for JaxToTorchVecEnv; use the Pixi gpu environment"
        ) from error


def _matches_device(actual: Any, expected: Any, torch: Any) -> bool:
    expected_device = torch.device(expected)
    return actual.type == expected_device.type and (
        expected_device.index is None or actual.index == expected_device.index
    )


def _jax_array_to_torch(value: Any, *, torch: Any, device: Any) -> Any:
    """Share one numerical JAX buffer with Torch through DLPack."""

    if not isinstance(value, jax.Array):
        raise TypeError(f"expected a JAX array, got {type(value).__name__}")
    result = torch.from_dlpack(value)
    if not _matches_device(result.device, device, torch):
        raise ValueError(
            f"JAX output is on {result.device}, but the Torch bridge requires device {device!r}"
        )
    return result


def _torch_array_to_jax(value: Any, *, torch: Any, device: Any) -> jax.Array:
    """Share one Torch action buffer with JAX through DLPack."""

    if not isinstance(value, torch.Tensor):
        raise TypeError(f"actions must be a torch.Tensor, got {type(value).__name__}")
    if not _matches_device(value.device, device, torch):
        raise ValueError(
            f"Torch actions are on {value.device}, but the bridge requires device {device!r}"
        )
    # Environment transitions are intentionally outside the policy autograd graph. ``detach`` is a
    # metadata-only view and therefore preserves the zero-copy DLPack path.
    source = value.detach() if value.requires_grad else value
    try:
        return jax.dlpack.from_dlpack(source)
    except (BufferError, RuntimeError, TypeError) as error:
        raise TypeError(
            "Torch actions must expose a DLPack-compatible numerical tensor on the JAX device"
        ) from error


def _jax_tree_to_torch(value: Any, *, torch: Any, device: Any) -> Any:
    """Convert a numerical JAX pytree without accepting implicit host-array copies."""

    if isinstance(value, jax.Array):
        return _jax_array_to_torch(value, torch=torch, device=device)
    if isinstance(value, Mapping):
        return {
            key: _jax_tree_to_torch(item, torch=torch, device=device) for key, item in value.items()
        }
    if isinstance(value, tuple):
        return tuple(_jax_tree_to_torch(item, torch=torch, device=device) for item in value)
    if isinstance(value, list):
        return [_jax_tree_to_torch(item, torch=torch, device=device) for item in value]
    raise TypeError(
        "JAX-to-Torch observations must contain only numerical JAX array leaves; "
        f"got {type(value).__name__}"
    )


def _validate_device_vector(
    value: Any,
    *,
    name: str,
    num_envs: int,
    dtype: np.dtype[Any],
) -> jax.Array:
    """Validate one JAX vector using shape/dtype metadata only."""

    if not isinstance(value, jax.Array):
        raise TypeError(f"public info field {name!r} must be a JAX array")
    if value.shape != (num_envs,):
        raise ValueError(
            f"public info field {name!r} must have shape {(num_envs,)}, got {value.shape}"
        )
    if np.dtype(value.dtype) != dtype:
        raise TypeError(f"public info field {name!r} must have dtype {dtype}, got {value.dtype}")
    return value


def _validate_benchmark_versions(
    value: Any,
    *,
    num_envs: int,
    expected_version: str,
) -> np.ndarray:
    """Validate the sole host/string public info value without copying it."""

    if not isinstance(value, np.ndarray):
        raise TypeError("public info field 'benchmark_version' must be a NumPy string array")
    if value.shape != (num_envs,):
        raise ValueError(
            "public info field 'benchmark_version' must have shape "
            f"{(num_envs,)}, got {value.shape}"
        )
    if value.dtype.kind != "U":
        raise TypeError(
            "public info field 'benchmark_version' must have a Unicode string dtype, "
            f"got {value.dtype}"
        )
    if value.flags.writeable:
        raise ValueError("public info field 'benchmark_version' must be read-only")
    if not np.all(value == expected_version):
        raise ValueError(
            "public info field 'benchmark_version' does not match the official environment"
        )
    return value


def _public_info_to_torch(
    info: Any,
    *,
    num_envs: int,
    benchmark_version: str,
    torch: Any,
    device: Any,
) -> dict[str, Any]:
    """Convert exactly the canonical public info schema and preserve its string leaf."""

    if not isinstance(info, Mapping):
        raise TypeError("vector environment info must be a mapping")
    actual = set(info)
    expected = set(PUBLIC_INFO_KEYS)
    if actual != expected:
        missing = sorted(expected - actual)
        extra = sorted(actual - expected)
        raise ValueError(
            "public info keys do not match the canonical whitelist; "
            f"missing={missing}, extra={extra}"
        )

    converted: dict[str, Any] = {}
    for name in PUBLIC_INFO_KEYS:
        value = info[name]
        if name == _BENCHMARK_VERSION_KEY:
            converted[name] = _validate_benchmark_versions(
                value,
                num_envs=num_envs,
                expected_version=benchmark_version,
            )
            continue
        validated = _validate_device_vector(
            value,
            name=name,
            num_envs=num_envs,
            dtype=_NUMERIC_INFO_DTYPES[name],
        )
        converted[name] = _jax_array_to_torch(validated, torch=torch, device=device)
    return converted


class JaxToTorchVecEnv(gym.vector.VectorWrapper):
    """Expose an official JAX ``VectorEnv`` through same-device Torch tensors.

    Numerical observations, rewards, flags, actions, and numeric public info use DLPack and remain
    on their existing device. Reset options are control-plane host values and are passed through
    unchanged. The wrapper never performs a numeric ``device_get``/NumPy conversion in ``reset`` or
    ``step``.
    """

    def __init__(self, env: VectorEnv, *, device: Any) -> None:
        super().__init__(env)
        base = env.unwrapped
        if not isinstance(base, VecCarRacingEnv):
            raise TypeError("JaxToTorchVecEnv requires the official VecCarRacingEnv")
        if base.backend != "mjx_warp":
            raise ValueError("JaxToTorchVecEnv requires the formal 'mjx_warp' backend")
        if base.level_id != 1:
            raise ValueError("JaxToTorchVecEnv requires the formal Level 1 environment")
        benchmark_version = base.project_config.benchmark.version
        if benchmark_version != "0.1":
            raise ValueError("JaxToTorchVecEnv requires benchmark version '0.1'")
        if env.metadata.get("autoreset_mode") != AutoresetMode.NEXT_STEP:
            raise ValueError("JaxToTorchVecEnv requires NEXT_STEP autoreset semantics")
        if env.num_envs != base.num_envs:
            raise ValueError("wrapped vector environment width must match VecCarRacingEnv")

        self._torch = _load_torch()
        if device is None:
            raise ValueError("an explicit CUDA device is required")
        try:
            requested_device = self._torch.device(device)
        except (RuntimeError, TypeError, ValueError) as error:
            raise ValueError(f"invalid Torch device {device!r}") from error
        if requested_device.type != "cuda":
            raise ValueError("JaxToTorchVecEnv requires an explicit CUDA device")
        if requested_device.index is None:
            requested_device = self._torch.device(
                "cuda",
                self._torch.cuda.current_device(),
            )
        self.device = requested_device
        self.benchmark_version = benchmark_version

    def _observation(self, observation: Any) -> Any:
        return _jax_tree_to_torch(
            observation,
            torch=self._torch,
            device=self.device,
        )

    def _info(self, info: Any) -> dict[str, Any]:
        return _public_info_to_torch(
            info,
            num_envs=self.num_envs,
            benchmark_version=self.benchmark_version,
            torch=self._torch,
            device=self.device,
        )

    def reset(
        self,
        *,
        seed: int | None = None,
        options: dict[str, Any] | None = None,
    ) -> tuple[Any, dict[str, Any]]:
        """Reset the JAX environment and expose public outputs as Torch/string leaves."""

        observation, info = self.env.reset(seed=seed, options=options)
        return self._observation(observation), self._info(info)

    def step(self, actions: Any) -> tuple[Any, Any, Any, Any, dict[str, Any]]:
        """Convert actions to JAX, advance once, and convert public numerical outputs to Torch."""

        jax_actions = _torch_array_to_jax(
            actions,
            torch=self._torch,
            device=self.device,
        )
        observation, reward, terminated, truncated, info = self.env.step(jax_actions)

        reward = _validate_device_vector(
            reward,
            name="reward",
            num_envs=self.num_envs,
            dtype=np.dtype(np.float32),
        )
        terminated = _validate_device_vector(
            terminated,
            name="terminated",
            num_envs=self.num_envs,
            dtype=np.dtype(np.bool_),
        )
        truncated = _validate_device_vector(
            truncated,
            name="truncated",
            num_envs=self.num_envs,
            dtype=np.dtype(np.bool_),
        )
        return (
            self._observation(observation),
            _jax_array_to_torch(reward, torch=self._torch, device=self.device),
            _jax_array_to_torch(terminated, torch=self._torch, device=self.device),
            _jax_array_to_torch(truncated, torch=self._torch, device=self.device),
            self._info(info),
        )


__all__ = ["JaxToTorchVecEnv"]
