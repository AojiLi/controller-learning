"""Torch-free deterministic NumPy actor and canonical inference artifact format."""

from __future__ import annotations

import hashlib
import importlib
import io
import os
import re
import tempfile
import zipfile
from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Final

import numpy as np
from numpy.typing import NDArray

NUMPY_ACTOR_SCHEMA_VERSION: Final = 1
NUMPY_ACTOR_OBSERVATION_DIM: Final = 100
NUMPY_ACTOR_HIDDEN_DIM: Final = 128
NUMPY_ACTOR_ACTION_DIM: Final = 2
NUMPY_ACTOR_MAX_BYTES: Final = 256 * 1024

_FLOAT32 = np.dtype(np.float32)
_UINT32 = np.dtype(np.uint32)
_SHA256_PATTERN = re.compile(r"^[0-9a-f]{64}$")
_ZIP_TIMESTAMP = (1980, 1, 1, 0, 0, 0)
_ARRAY_NAMES = (
    "schema_version",
    "action_high",
    "action_low",
    "actor_bias",
    "actor_weight",
    "hidden_0_bias",
    "hidden_0_weight",
    "hidden_1_bias",
    "hidden_1_weight",
)
_PPO_STATE_KEYS = {
    "action_bias",
    "action_high",
    "action_low",
    "action_scale",
    "actor_mean.bias",
    "actor_mean.weight",
    "critic.bias",
    "critic.weight",
    "log_action_scale",
    "log_std",
    "trunk.0.bias",
    "trunk.0.weight",
    "trunk.2.bias",
    "trunk.2.weight",
}


class NumpyActorArtifactError(RuntimeError):
    """Raised when a deterministic actor artifact is invalid or cannot be persisted."""


def _readonly_float32(value: object, *, name: str, shape: tuple[int, ...]) -> NDArray[np.float32]:
    if not isinstance(value, np.ndarray):
        raise TypeError(f"{name} must be a numpy.ndarray")
    if value.dtype != _FLOAT32:
        raise TypeError(f"{name} must use float32")
    if value.shape != shape:
        raise ValueError(f"{name} must have shape {shape}, got {value.shape}")
    if not np.isfinite(value).all():
        raise ValueError(f"{name} must contain only finite values")
    result = np.array(value, dtype=np.float32, order="C", copy=True)
    result.setflags(write=False)
    return result


def _readonly_result(value: NDArray[np.float32]) -> NDArray[np.float32]:
    result = np.array(value, dtype=np.float32, order="C", copy=True)
    if not np.isfinite(result).all():
        raise FloatingPointError("NumPy actor produced a non-finite value")
    result.setflags(write=False)
    return result


@dataclass(frozen=True, slots=True)
class NumpyDeterministicAction:
    """Physical action and pre-tanh actor output for one NumPy inference call."""

    action: NDArray[np.float32]
    pre_tanh: NDArray[np.float32]


@dataclass(frozen=True, slots=True)
class NumpyDeterministicActor:
    """Fixed ``100 -> 128 -> 128 -> 2`` tanh actor for deployment without Torch."""

    hidden_0_weight: NDArray[np.float32]
    hidden_0_bias: NDArray[np.float32]
    hidden_1_weight: NDArray[np.float32]
    hidden_1_bias: NDArray[np.float32]
    actor_weight: NDArray[np.float32]
    actor_bias: NDArray[np.float32]
    action_low: NDArray[np.float32]
    action_high: NDArray[np.float32]
    _action_scale: NDArray[np.float32] = field(init=False, repr=False, compare=False)
    _action_bias: NDArray[np.float32] = field(init=False, repr=False, compare=False)

    def __post_init__(self) -> None:
        shapes = {
            "hidden_0_weight": (NUMPY_ACTOR_HIDDEN_DIM, NUMPY_ACTOR_OBSERVATION_DIM),
            "hidden_0_bias": (NUMPY_ACTOR_HIDDEN_DIM,),
            "hidden_1_weight": (NUMPY_ACTOR_HIDDEN_DIM, NUMPY_ACTOR_HIDDEN_DIM),
            "hidden_1_bias": (NUMPY_ACTOR_HIDDEN_DIM,),
            "actor_weight": (NUMPY_ACTOR_ACTION_DIM, NUMPY_ACTOR_HIDDEN_DIM),
            "actor_bias": (NUMPY_ACTOR_ACTION_DIM,),
            "action_low": (NUMPY_ACTOR_ACTION_DIM,),
            "action_high": (NUMPY_ACTOR_ACTION_DIM,),
        }
        for name, shape in shapes.items():
            object.__setattr__(
                self,
                name,
                _readonly_float32(getattr(self, name), name=name, shape=shape),
            )
        if not np.all(self.action_high > self.action_low):
            raise ValueError("action_high must be greater than action_low")
        scale = np.asarray((self.action_high - self.action_low) * np.float32(0.5))
        bias = np.asarray((self.action_high + self.action_low) * np.float32(0.5))
        object.__setattr__(
            self,
            "_action_scale",
            _readonly_float32(scale, name="action_scale", shape=(NUMPY_ACTOR_ACTION_DIM,)),
        )
        object.__setattr__(
            self,
            "_action_bias",
            _readonly_float32(bias, name="action_bias", shape=(NUMPY_ACTOR_ACTION_DIM,)),
        )

    @property
    def action_scale(self) -> NDArray[np.float32]:
        """Return the immutable physical-action scale."""

        return self._action_scale

    @property
    def action_bias(self) -> NDArray[np.float32]:
        """Return the immutable physical-action midpoint."""

        return self._action_bias

    def deterministic(self, observation: object) -> NumpyDeterministicAction:
        """Evaluate one observation or a batch with arbitrary leading dimensions."""

        if not isinstance(observation, np.ndarray):
            raise TypeError("observation must be a numpy.ndarray")
        if observation.dtype != _FLOAT32:
            raise TypeError("observation must use float32")
        if observation.ndim < 1 or observation.shape[-1] != NUMPY_ACTOR_OBSERVATION_DIM:
            raise ValueError("observation must end with the fixed 100-dimensional feature schema")
        if not np.isfinite(observation).all():
            raise ValueError("observation must contain only finite values")

        hidden_0 = np.tanh(np.matmul(observation, self.hidden_0_weight.T) + self.hidden_0_bias)
        hidden_1 = np.tanh(np.matmul(hidden_0, self.hidden_1_weight.T) + self.hidden_1_bias)
        pre_tanh = np.asarray(
            np.matmul(hidden_1, self.actor_weight.T) + self.actor_bias,
            dtype=np.float32,
        )
        action = np.asarray(
            self.action_bias + self.action_scale * np.tanh(pre_tanh),
            dtype=np.float32,
        )
        return NumpyDeterministicAction(
            action=_readonly_result(action),
            pre_tanh=_readonly_result(pre_tanh),
        )

    def __call__(self, observation: object) -> NDArray[np.float32]:
        """Return only the physical deterministic action."""

        return self.deterministic(observation).action


@dataclass(frozen=True, slots=True)
class NumpyActorFileEvidence:
    """Content identity of one strict canonical actor NPZ."""

    schema_version: int
    sha256: str
    size_bytes: int

    def __post_init__(self) -> None:
        if self.schema_version != NUMPY_ACTOR_SCHEMA_VERSION:
            raise ValueError("unexpected NumPy actor schema version")
        if not isinstance(self.sha256, str) or _SHA256_PATTERN.fullmatch(self.sha256) is None:
            raise ValueError("sha256 must be a lowercase SHA-256 digest")
        if type(self.size_bytes) is not int or not 0 < self.size_bytes <= NUMPY_ACTOR_MAX_BYTES:
            raise ValueError("size_bytes must fit the public NumPy actor size limit")


@dataclass(frozen=True, slots=True)
class LoadedNumpyActor:
    """A validated Torch-free actor and the identity of its source bytes."""

    actor: NumpyDeterministicActor
    evidence: NumpyActorFileEvidence

    def __post_init__(self) -> None:
        if not isinstance(self.actor, NumpyDeterministicActor):
            raise TypeError("actor must be NumpyDeterministicActor")
        if not isinstance(self.evidence, NumpyActorFileEvidence):
            raise TypeError("evidence must be NumpyActorFileEvidence")


def _actor_arrays(actor: NumpyDeterministicActor) -> dict[str, NDArray[Any]]:
    if not isinstance(actor, NumpyDeterministicActor):
        raise TypeError("actor must be NumpyDeterministicActor")
    return {
        "schema_version": np.asarray(NUMPY_ACTOR_SCHEMA_VERSION, dtype=np.uint32),
        "action_high": actor.action_high,
        "action_low": actor.action_low,
        "actor_bias": actor.actor_bias,
        "actor_weight": actor.actor_weight,
        "hidden_0_bias": actor.hidden_0_bias,
        "hidden_0_weight": actor.hidden_0_weight,
        "hidden_1_bias": actor.hidden_1_bias,
        "hidden_1_weight": actor.hidden_1_weight,
    }


def _npy_bytes(array: NDArray[Any]) -> bytes:
    output = io.BytesIO()
    contiguous = array if array.ndim == 0 else np.ascontiguousarray(array)
    np.lib.format.write_array(
        output,
        contiguous,
        version=(1, 0),
        allow_pickle=False,
    )
    return output.getvalue()


def canonical_numpy_actor_bytes(actor: NumpyDeterministicActor) -> bytes:
    """Return the unique ZIP_STORED NPZ byte representation for ``actor``."""

    arrays = _actor_arrays(actor)
    output = io.BytesIO()
    with zipfile.ZipFile(output, mode="w", compression=zipfile.ZIP_STORED) as archive:
        for name in _ARRAY_NAMES:
            information = zipfile.ZipInfo(f"{name}.npy", date_time=_ZIP_TIMESTAMP)
            information.compress_type = zipfile.ZIP_STORED
            information.create_system = 3
            information.external_attr = 0o100600 << 16
            archive.writestr(information, _npy_bytes(arrays[name]))
    result = output.getvalue()
    if not result or len(result) > NUMPY_ACTOR_MAX_BYTES:
        raise NumpyActorArtifactError("canonical NumPy actor exceeds its public size limit")
    return result


def _evidence(data: bytes) -> NumpyActorFileEvidence:
    return NumpyActorFileEvidence(
        schema_version=NUMPY_ACTOR_SCHEMA_VERSION,
        sha256=hashlib.sha256(data).hexdigest(),
        size_bytes=len(data),
    )


def _fsync_directory(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def save_numpy_actor_npz(
    actor: NumpyDeterministicActor,
    path: str | Path,
    *,
    staging_directory: str | Path | None = None,
) -> NumpyActorFileEvidence:
    """Atomically persist and read back one canonical deterministic actor NPZ.

    ``staging_directory`` lets a crash-recovery owner contain pre-replace temporary files. The
    default preserves the original same-directory atomic-write behavior.
    """

    destination = Path(path)
    if destination.suffix != ".npz":
        raise ValueError("NumPy actor path must use the .npz suffix")
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.is_symlink():
        raise NumpyActorArtifactError("refusing to replace a symbolic-link actor path")
    staging = destination.parent if staging_directory is None else Path(staging_directory)
    if staging.is_symlink() or not staging.is_dir():
        raise NumpyActorArtifactError("staging_directory must be a regular existing directory")
    staging = staging.resolve(strict=True)
    data = canonical_numpy_actor_bytes(actor)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{destination.name}.",
        suffix=".tmp",
        dir=staging,
    )
    temporary = Path(temporary_name)
    try:
        os.fchmod(descriptor, 0o600)
        with os.fdopen(descriptor, "wb") as file:
            descriptor = -1
            file.write(data)
            file.flush()
            os.fsync(file.fileno())
        os.replace(temporary, destination)
        _fsync_directory(staging)
        _fsync_directory(destination.parent)
        if destination.read_bytes() != data:
            raise NumpyActorArtifactError("NumPy actor failed exact byte readback")
    except BaseException as error:
        if descriptor >= 0:
            os.close(descriptor)
        temporary.unlink(missing_ok=True)
        if isinstance(error, (NumpyActorArtifactError, ValueError, TypeError)):
            raise
        raise NumpyActorArtifactError("failed to persist canonical NumPy actor") from error
    finally:
        temporary.unlink(missing_ok=True)
    return _evidence(data)


def _actor_from_arrays(arrays: Mapping[str, NDArray[Any]]) -> NumpyDeterministicActor:
    if set(arrays) != set(_ARRAY_NAMES):
        raise NumpyActorArtifactError("NumPy actor array names differ from the strict schema")
    schema = arrays["schema_version"]
    if schema.shape != () or schema.dtype != _UINT32 or int(schema) != NUMPY_ACTOR_SCHEMA_VERSION:
        raise NumpyActorArtifactError("NumPy actor schema_version is invalid")
    try:
        return NumpyDeterministicActor(
            hidden_0_weight=arrays["hidden_0_weight"],
            hidden_0_bias=arrays["hidden_0_bias"],
            hidden_1_weight=arrays["hidden_1_weight"],
            hidden_1_bias=arrays["hidden_1_bias"],
            actor_weight=arrays["actor_weight"],
            actor_bias=arrays["actor_bias"],
            action_low=arrays["action_low"],
            action_high=arrays["action_high"],
        )
    except (TypeError, ValueError) as error:
        raise NumpyActorArtifactError(
            "NumPy actor arrays violate shape, dtype, or value rules"
        ) from error


def load_numpy_actor_npz(
    path: str | Path,
    *,
    expected_sha256: str | None = None,
    expected_size_bytes: int | None = None,
) -> LoadedNumpyActor:
    """Load only an exact canonical actor NPZ with optional bound hash and size."""

    source = Path(path)
    if source.suffix != ".npz":
        raise ValueError("NumPy actor path must use the .npz suffix")
    if source.is_symlink():
        raise NumpyActorArtifactError("refusing to load a symbolic-link actor path")
    try:
        data = source.read_bytes()
    except OSError as error:
        raise NumpyActorArtifactError(f"cannot read NumPy actor: {source}") from error
    if not data or len(data) > NUMPY_ACTOR_MAX_BYTES:
        raise NumpyActorArtifactError("NumPy actor file size is outside the public limit")
    evidence = _evidence(data)
    if expected_sha256 is not None:
        if _SHA256_PATTERN.fullmatch(expected_sha256) is None:
            raise ValueError("expected_sha256 must be a lowercase SHA-256 digest")
        if evidence.sha256 != expected_sha256:
            raise NumpyActorArtifactError("NumPy actor SHA-256 differs from expected_sha256")
    if expected_size_bytes is not None:
        if type(expected_size_bytes) is not int or expected_size_bytes < 1:
            raise ValueError("expected_size_bytes must be a positive integer")
        if evidence.size_bytes != expected_size_bytes:
            raise NumpyActorArtifactError("NumPy actor size differs from expected_size_bytes")

    try:
        with zipfile.ZipFile(io.BytesIO(data), mode="r") as archive:
            information = archive.infolist()
            expected_entries = tuple(f"{name}.npy" for name in _ARRAY_NAMES)
            if tuple(entry.filename for entry in information) != expected_entries:
                raise NumpyActorArtifactError(
                    "NumPy actor ZIP entries differ from the strict schema"
                )
            if any(
                entry.compress_type != zipfile.ZIP_STORED
                or entry.date_time != _ZIP_TIMESTAMP
                or entry.create_system != 3
                or entry.external_attr != 0o100600 << 16
                for entry in information
            ):
                raise NumpyActorArtifactError("NumPy actor ZIP metadata is not canonical")
        with np.load(io.BytesIO(data), allow_pickle=False) as archive:
            if tuple(archive.files) != _ARRAY_NAMES:
                raise NumpyActorArtifactError("NumPy actor NPZ keys differ from the strict schema")
            arrays = {name: np.array(archive[name], copy=True) for name in _ARRAY_NAMES}
    except NumpyActorArtifactError:
        raise
    except (OSError, ValueError, zipfile.BadZipFile) as error:
        raise NumpyActorArtifactError("NumPy actor is not a valid non-pickled NPZ") from error

    actor = _actor_from_arrays(arrays)
    if canonical_numpy_actor_bytes(actor) != data:
        raise NumpyActorArtifactError("NumPy actor bytes are not the canonical representation")
    return LoadedNumpyActor(actor=actor, evidence=evidence)


def _torch_state_array(
    torch_module: Any,
    value: object,
    *,
    name: str,
    shape: tuple[int, ...],
) -> NDArray[np.float32]:
    tensor_type = getattr(torch_module, "Tensor", None)
    if tensor_type is None or not isinstance(value, tensor_type):
        raise TypeError(f"state_dict[{name!r}] must be a torch.Tensor")
    if value.dtype is not torch_module.float32:
        raise TypeError(f"state_dict[{name!r}] must use torch.float32")
    if tuple(value.shape) != shape:
        raise ValueError(f"state_dict[{name!r}] must have shape {shape}")
    result = value.detach().to(device="cpu").numpy()
    return _readonly_float32(result, name=f"state_dict[{name!r}]", shape=shape)


def numpy_actor_from_ppo_state_dict(state_dict: object) -> NumpyDeterministicActor:
    """Lazily convert one exact :class:`PpoActorCritic` state mapping into NumPy."""

    if not isinstance(state_dict, Mapping):
        raise TypeError("state_dict must be a mapping")
    if set(state_dict) != _PPO_STATE_KEYS:
        missing = sorted(_PPO_STATE_KEYS - set(state_dict))
        extra = sorted(set(state_dict) - _PPO_STATE_KEYS)
        raise ValueError(f"PPO state_dict keys differ; missing={missing}, extra={extra}")
    try:
        torch = importlib.import_module("torch")
    except ImportError as error:  # pragma: no cover - exercised by environments without Torch.
        raise RuntimeError("PPO state conversion requires Torch only at conversion time") from error

    shapes = {
        "trunk.0.weight": (NUMPY_ACTOR_HIDDEN_DIM, NUMPY_ACTOR_OBSERVATION_DIM),
        "trunk.0.bias": (NUMPY_ACTOR_HIDDEN_DIM,),
        "trunk.2.weight": (NUMPY_ACTOR_HIDDEN_DIM, NUMPY_ACTOR_HIDDEN_DIM),
        "trunk.2.bias": (NUMPY_ACTOR_HIDDEN_DIM,),
        "actor_mean.weight": (NUMPY_ACTOR_ACTION_DIM, NUMPY_ACTOR_HIDDEN_DIM),
        "actor_mean.bias": (NUMPY_ACTOR_ACTION_DIM,),
        "critic.weight": (1, NUMPY_ACTOR_HIDDEN_DIM),
        "critic.bias": (1,),
        "log_std": (NUMPY_ACTOR_ACTION_DIM,),
        "action_low": (NUMPY_ACTOR_ACTION_DIM,),
        "action_high": (NUMPY_ACTOR_ACTION_DIM,),
        "action_scale": (NUMPY_ACTOR_ACTION_DIM,),
        "action_bias": (NUMPY_ACTOR_ACTION_DIM,),
        "log_action_scale": (NUMPY_ACTOR_ACTION_DIM,),
    }
    arrays = {
        name: _torch_state_array(torch, state_dict[name], name=name, shape=shape)
        for name, shape in shapes.items()
    }
    actor = NumpyDeterministicActor(
        hidden_0_weight=arrays["trunk.0.weight"],
        hidden_0_bias=arrays["trunk.0.bias"],
        hidden_1_weight=arrays["trunk.2.weight"],
        hidden_1_bias=arrays["trunk.2.bias"],
        actor_weight=arrays["actor_mean.weight"],
        actor_bias=arrays["actor_mean.bias"],
        action_low=arrays["action_low"],
        action_high=arrays["action_high"],
    )
    if not np.array_equal(arrays["action_scale"], actor.action_scale):
        raise ValueError("PPO action_scale differs from action_low/action_high")
    if not np.array_equal(arrays["action_bias"], actor.action_bias):
        raise ValueError("PPO action_bias differs from action_low/action_high")
    expected_log_scale = np.log(actor.action_scale).astype(np.float32)
    if not np.allclose(
        arrays["log_action_scale"],
        expected_log_scale,
        rtol=0.0,
        atol=np.float32(1.0e-7),
    ):
        raise ValueError("PPO log_action_scale differs from the physical action scale")
    return actor


__all__ = [
    "NUMPY_ACTOR_ACTION_DIM",
    "NUMPY_ACTOR_HIDDEN_DIM",
    "NUMPY_ACTOR_MAX_BYTES",
    "NUMPY_ACTOR_OBSERVATION_DIM",
    "NUMPY_ACTOR_SCHEMA_VERSION",
    "LoadedNumpyActor",
    "NumpyActorArtifactError",
    "NumpyActorFileEvidence",
    "NumpyDeterministicAction",
    "NumpyDeterministicActor",
    "canonical_numpy_actor_bytes",
    "load_numpy_actor_npz",
    "numpy_actor_from_ppo_state_dict",
    "save_numpy_actor_npz",
]
