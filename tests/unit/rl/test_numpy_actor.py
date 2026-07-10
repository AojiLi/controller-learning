"""CPU tests for the Torch-free NumPy actor and canonical NPZ format."""

from __future__ import annotations

import hashlib
import io
import zipfile
from pathlib import Path

import numpy as np
import pytest

from controller_learning.rl.numpy_actor import (
    NUMPY_ACTOR_ACTION_DIM,
    NUMPY_ACTOR_HIDDEN_DIM,
    NUMPY_ACTOR_MAX_BYTES,
    NUMPY_ACTOR_OBSERVATION_DIM,
    NumpyActorArtifactError,
    NumpyDeterministicActor,
    canonical_numpy_actor_bytes,
    load_numpy_actor_npz,
    save_numpy_actor_npz,
)

PROJECT_ROOT = Path(__file__).parents[3]


def _actor(seed: int = 7) -> NumpyDeterministicActor:
    generator = np.random.default_rng(seed)

    def values(shape: tuple[int, ...], scale: float = 0.05) -> np.ndarray:
        return np.asarray(generator.normal(0.0, scale, size=shape), dtype=np.float32)

    return NumpyDeterministicActor(
        hidden_0_weight=values((NUMPY_ACTOR_HIDDEN_DIM, NUMPY_ACTOR_OBSERVATION_DIM)),
        hidden_0_bias=values((NUMPY_ACTOR_HIDDEN_DIM,)),
        hidden_1_weight=values((NUMPY_ACTOR_HIDDEN_DIM, NUMPY_ACTOR_HIDDEN_DIM)),
        hidden_1_bias=values((NUMPY_ACTOR_HIDDEN_DIM,)),
        actor_weight=values((NUMPY_ACTOR_ACTION_DIM, NUMPY_ACTOR_HIDDEN_DIM)),
        actor_bias=values((NUMPY_ACTOR_ACTION_DIM,)),
        action_low=np.asarray((-0.6, -8.0), dtype=np.float32),
        action_high=np.asarray((0.6, 4.0), dtype=np.float32),
    )


def test_numpy_actor_is_owned_immutable_batched_and_seed_reproducible() -> None:
    actor = _actor(11)
    repeated = _actor(11)
    observations = np.linspace(
        -1.0,
        1.0,
        3 * NUMPY_ACTOR_OBSERVATION_DIM,
        dtype=np.float32,
    ).reshape(3, NUMPY_ACTOR_OBSERVATION_DIM)

    result = actor.deterministic(observations)
    repeated_result = repeated.deterministic(observations)

    assert result.action.shape == result.pre_tanh.shape == (3, NUMPY_ACTOR_ACTION_DIM)
    assert result.action.dtype == result.pre_tanh.dtype == np.dtype(np.float32)
    assert not result.action.flags.writeable
    assert not result.pre_tanh.flags.writeable
    np.testing.assert_array_equal(repeated_result.action, result.action)
    np.testing.assert_array_equal(actor(observations), result.action)
    assert np.all(result.action >= actor.action_low)
    assert np.all(result.action <= actor.action_high)
    for array in (
        actor.hidden_0_weight,
        actor.hidden_0_bias,
        actor.hidden_1_weight,
        actor.hidden_1_bias,
        actor.actor_weight,
        actor.actor_bias,
        actor.action_low,
        actor.action_high,
        actor.action_scale,
        actor.action_bias,
    ):
        assert array.dtype == np.dtype(np.float32)
        assert array.flags.c_contiguous
        assert not array.flags.writeable

    single = actor.deterministic(observations[0])
    assert single.action.shape == single.pre_tanh.shape == (NUMPY_ACTOR_ACTION_DIM,)
    # BLAS implementations may choose different reduction kernels for vector and matrix inputs.
    # Keep the shape contract strict while allowing only a handful of float32 rounding steps.
    np.testing.assert_array_max_ulp(single.action, result.action[0], maxulp=8)


def test_numpy_actor_rejects_wrong_shapes_dtypes_bounds_and_observations() -> None:
    actor = _actor()
    fields = {
        "hidden_0_weight": actor.hidden_0_weight,
        "hidden_0_bias": actor.hidden_0_bias,
        "hidden_1_weight": actor.hidden_1_weight,
        "hidden_1_bias": actor.hidden_1_bias,
        "actor_weight": actor.actor_weight,
        "actor_bias": actor.actor_bias,
        "action_low": actor.action_low,
        "action_high": actor.action_high,
    }
    with pytest.raises(TypeError, match="hidden_0_weight must use float32"):
        NumpyDeterministicActor(
            **{**fields, "hidden_0_weight": actor.hidden_0_weight.astype(np.float64)}
        )
    with pytest.raises(ValueError, match="hidden_0_weight must have shape"):
        NumpyDeterministicActor(**{**fields, "hidden_0_weight": actor.hidden_0_weight[:, :-1]})
    with pytest.raises(ValueError, match="action_high"):
        NumpyDeterministicActor(
            **{**fields, "action_high": np.asarray((-1.0, 4.0), dtype=np.float32)}
        )

    with pytest.raises(TypeError, match=r"numpy\.ndarray"):
        actor([0.0] * NUMPY_ACTOR_OBSERVATION_DIM)
    with pytest.raises(TypeError, match="float32"):
        actor(np.zeros(NUMPY_ACTOR_OBSERVATION_DIM, dtype=np.float64))
    with pytest.raises(ValueError, match="100-dimensional"):
        actor(np.zeros(99, dtype=np.float32))
    nonfinite = np.zeros(NUMPY_ACTOR_OBSERVATION_DIM, dtype=np.float32)
    nonfinite[0] = np.nan
    with pytest.raises(ValueError, match="finite"):
        actor(nonfinite)


def test_canonical_npz_export_is_byte_stable_hash_bound_and_round_trips(
    tmp_path: Path,
) -> None:
    actor = _actor(17)
    first_bytes = canonical_numpy_actor_bytes(actor)
    second_bytes = canonical_numpy_actor_bytes(actor)
    first_path = tmp_path / "first.npz"
    second_path = tmp_path / "second.npz"

    first_evidence = save_numpy_actor_npz(actor, first_path)
    second_evidence = save_numpy_actor_npz(actor, second_path)

    assert first_bytes == second_bytes == first_path.read_bytes() == second_path.read_bytes()
    assert first_evidence == second_evidence
    assert first_evidence.sha256 == hashlib.sha256(first_bytes).hexdigest()
    assert first_evidence.size_bytes == len(first_bytes) < NUMPY_ACTOR_MAX_BYTES
    loaded = load_numpy_actor_npz(
        first_path,
        expected_sha256=first_evidence.sha256,
        expected_size_bytes=first_evidence.size_bytes,
    )
    assert loaded.evidence == first_evidence
    observations = np.linspace(
        -0.5,
        0.5,
        5 * NUMPY_ACTOR_OBSERVATION_DIM,
        dtype=np.float32,
    ).reshape(5, NUMPY_ACTOR_OBSERVATION_DIM)
    np.testing.assert_array_equal(loaded.actor(observations), actor(observations))
    with zipfile.ZipFile(io.BytesIO(first_bytes)) as archive:
        assert archive.namelist() == [
            "schema_version.npy",
            "action_high.npy",
            "action_low.npy",
            "actor_bias.npy",
            "actor_weight.npy",
            "hidden_0_bias.npy",
            "hidden_0_weight.npy",
            "hidden_1_bias.npy",
            "hidden_1_weight.npy",
        ]
        assert all(info.compress_type == zipfile.ZIP_STORED for info in archive.infolist())


def test_npz_loader_rejects_hash_size_noncanonical_and_oversized_inputs(
    tmp_path: Path,
) -> None:
    actor = _actor(19)
    canonical = tmp_path / "canonical.npz"
    evidence = save_numpy_actor_npz(actor, canonical)

    with pytest.raises(NumpyActorArtifactError, match="SHA-256"):
        load_numpy_actor_npz(canonical, expected_sha256="0" * 64)
    with pytest.raises(NumpyActorArtifactError, match="size differs"):
        load_numpy_actor_npz(canonical, expected_size_bytes=evidence.size_bytes + 1)
    with pytest.raises(ValueError, match="lowercase SHA-256"):
        load_numpy_actor_npz(canonical, expected_sha256="invalid")

    noncanonical = tmp_path / "noncanonical.npz"
    np.savez(
        noncanonical,
        schema_version=np.asarray(1, dtype=np.uint32),
        action_high=actor.action_high,
        action_low=actor.action_low,
        actor_bias=actor.actor_bias,
        actor_weight=actor.actor_weight,
        hidden_0_bias=actor.hidden_0_bias,
        hidden_0_weight=actor.hidden_0_weight,
        hidden_1_bias=actor.hidden_1_bias,
        hidden_1_weight=actor.hidden_1_weight,
    )
    with pytest.raises(NumpyActorArtifactError, match=r"canonical|ZIP metadata"):
        load_numpy_actor_npz(noncanonical)

    mutated = tmp_path / "mutated.npz"
    data = bytearray(canonical.read_bytes())
    data[len(data) // 2] ^= 0x01
    mutated.write_bytes(data)
    with pytest.raises(
        NumpyActorArtifactError,
        match=r"valid non-pickled NPZ|canonical",
    ):
        load_numpy_actor_npz(mutated)

    oversized = tmp_path / "oversized.npz"
    oversized.write_bytes(b"0" * (NUMPY_ACTOR_MAX_BYTES + 1))
    with pytest.raises(NumpyActorArtifactError, match="file size"):
        load_numpy_actor_npz(oversized)


def test_numpy_actor_module_has_no_import_time_torch_dependency() -> None:
    source = (PROJECT_ROOT / "controller_learning" / "rl" / "numpy_actor.py").read_text(
        encoding="utf-8"
    )
    converter = source.index("def numpy_actor_from_ppo_state_dict")
    assert "import torch" not in source[:converter]
    assert "controller_learning.rl.policy" not in source
