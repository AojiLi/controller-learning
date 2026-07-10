"""Tests for the versioned public-observation PPO feature encoder."""

from __future__ import annotations

from pathlib import Path

import gymnasium as gym
import jax
import jax.numpy as jnp
import numpy as np
import pytest
from gymnasium.vector import AutoresetMode

from controller_learning.config import load_project_config
from controller_learning.envs.car_racing import CarRacingEnv
from controller_learning.envs.observation import OBSERVATION_KEYS
from controller_learning.envs.vector_racing import VecCarRacingEnv
from controller_learning.rl.configuration import PpoObservationConfig
from controller_learning.rl.features import (
    LOCAL_TRACK_FEATURE_DIM,
    LOCAL_TRACK_FEATURE_SCHEMA_VERSION,
    LOCAL_TRACK_PREVIEW_POINTS,
    LocalTrackObservationVecEnv,
    encode_local_track_features_jax,
    encode_local_track_features_numpy,
    local_track_reference_jax,
    sample_local_track_preview_jax,
)
from controller_learning.tracks.generator import generate_track_candidate, pack_track
from controller_learning.tracks.specs import (
    generation_spec_from_project,
    track_capacity_from_project,
)

PROJECT_ROOT = Path(__file__).parents[3]
_PREVIEW_DISTANCE_M = 40.0
_MAX_SPEED_MPS = 15.0
_CONTROL_DT_S = 0.05
_MAX_STEERING_RAD = 0.6


def _square_observation(
    *,
    capacity: int = 8,
    progress: float = 0.0,
    position: tuple[float, float] = (0.0, 0.0),
    yaw: float = 0.0,
    padding: float = 0.0,
) -> dict[str, np.ndarray]:
    if capacity < 5:
        raise ValueError("square test observation requires capacity >= 5")
    center_valid = np.asarray(
        ((0.0, 0.0), (10.0, 0.0), (10.0, 10.0), (0.0, 10.0), (0.0, 0.0)),
        dtype=np.float32,
    )
    left_valid = center_valid + np.asarray((0.0, 1.0), dtype=np.float32)
    right_valid = center_valid + np.asarray((0.0, -1.0), dtype=np.float32)

    def padded(valid: np.ndarray) -> np.ndarray:
        result = np.full((capacity, 2), padding, dtype=np.float32)
        result[: valid.shape[0]] = valid
        return result

    mask = np.zeros(capacity, dtype=np.int8)
    mask[:5] = 1
    return {
        "position": np.asarray(position, dtype=np.float32),
        "yaw": np.asarray(yaw, dtype=np.float32),
        "velocity_body": np.asarray((7.5, -15.0), dtype=np.float32),
        "yaw_rate": np.asarray(2.0, dtype=np.float32),
        "steering_angle": np.asarray(0.3, dtype=np.float32),
        "track_progress": np.asarray(progress, dtype=np.float32),
        "centerline": padded(center_valid),
        "left_boundary": padded(left_valid),
        "right_boundary": padded(right_valid),
        "track_mask": mask,
        "track_length": np.asarray(40.0, dtype=np.float32),
    }


def _batched(*observations: dict[str, np.ndarray]) -> dict[str, jax.Array]:
    assert observations
    assert tuple(observations[0]) == OBSERVATION_KEYS
    return {
        name: jnp.asarray(np.stack([observation[name] for observation in observations]))
        for name in OBSERVATION_KEYS
    }


def _jax_features(observation: dict[str, jax.Array]) -> jax.Array:
    return encode_local_track_features_jax(
        observation,
        preview_points=LOCAL_TRACK_PREVIEW_POINTS,
        preview_distance_m=_PREVIEW_DISTANCE_M,
        max_speed_mps=_MAX_SPEED_MPS,
        control_dt_s=_CONTROL_DT_S,
        max_steering_angle_rad=_MAX_STEERING_RAD,
    )


def _numpy_features(observation: dict[str, np.ndarray]) -> np.ndarray:
    return encode_local_track_features_numpy(
        observation,
        preview_points=LOCAL_TRACK_PREVIEW_POINTS,
        preview_distance_m=_PREVIEW_DISTANCE_M,
        max_speed_mps=_MAX_SPEED_MPS,
        control_dt_s=_CONTROL_DT_S,
        max_steering_angle_rad=_MAX_STEERING_RAD,
    )


def test_versioned_feature_dimensions_and_field_order() -> None:
    observation = _square_observation()
    features = _numpy_features(observation)
    batched = _batched(observation)
    center, left, right = sample_local_track_preview_jax(
        batched,
        preview_points=LOCAL_TRACK_PREVIEW_POINTS,
        preview_distance_m=_PREVIEW_DISTANCE_M,
    )

    assert LOCAL_TRACK_FEATURE_SCHEMA_VERSION == 1
    assert LOCAL_TRACK_PREVIEW_POINTS == 16
    assert LOCAL_TRACK_FEATURE_DIM == 4 + 3 * 2 * LOCAL_TRACK_PREVIEW_POINTS == 100
    assert features.shape == (LOCAL_TRACK_FEATURE_DIM,)
    assert features.dtype == np.float32
    np.testing.assert_allclose(features[:4], (0.5, -1.0, 0.1, 0.5), atol=1.0e-7)
    np.testing.assert_allclose(
        features[4:36], np.asarray(center[0]).reshape(-1) / 40.0, atol=1.0e-7
    )
    np.testing.assert_allclose(features[36:68], np.asarray(left[0]).reshape(-1) / 40.0, atol=1.0e-7)
    np.testing.assert_allclose(
        features[68:100], np.asarray(right[0]).reshape(-1) / 40.0, atol=1.0e-7
    )
    assert np.isfinite(features).all()


def test_square_track_preview_has_hand_computable_values_and_reference() -> None:
    observation = _batched(_square_observation())
    center, left, right = sample_local_track_preview_jax(
        observation,
        preview_points=5,
        preview_distance_m=20.0,
    )
    expected_center = np.asarray(
        ((0.0, 0.0), (5.0, 0.0), (10.0, 0.0), (10.0, 5.0), (10.0, 10.0)),
        dtype=np.float32,
    )
    np.testing.assert_allclose(center[0], expected_center, atol=1.0e-6)
    np.testing.assert_allclose(left[0], expected_center + np.asarray((0.0, 1.0)), atol=1.0e-6)
    np.testing.assert_allclose(right[0], expected_center + np.asarray((0.0, -1.0)), atol=1.0e-6)

    reference, tangent = local_track_reference_jax(observation)
    np.testing.assert_allclose(reference[0], (0.0, 0.0), atol=1.0e-6)
    np.testing.assert_allclose(tangent[0], (1.0, 0.0), atol=1.0e-6)
    np.testing.assert_allclose(center[:, 0], reference, atol=1.0e-6)


def test_preview_wraps_across_explicit_closure_seam() -> None:
    observation = _batched(_square_observation(progress=0.875))
    center, _, _ = sample_local_track_preview_jax(
        observation,
        preview_points=3,
        preview_distance_m=10.0,
    )
    np.testing.assert_allclose(
        center[0],
        ((0.0, 5.0), (0.0, 0.0), (5.0, 0.0)),
        atol=1.0e-6,
    )


def test_body_frame_transform_and_reference_are_consistent() -> None:
    single = _square_observation(position=(1.0, 2.0), yaw=np.pi / 2.0)
    observation = _batched(single)
    center, _, _ = sample_local_track_preview_jax(
        observation,
        preview_points=3,
        preview_distance_m=10.0,
    )
    reference, tangent = local_track_reference_jax(observation)
    relative = np.asarray(reference[0]) - single["position"]
    expected_body = np.asarray((relative[1], -relative[0]), dtype=np.float32)
    np.testing.assert_allclose(center[0, 0], expected_body, atol=1.0e-6)
    np.testing.assert_allclose(tangent[0], (1.0, 0.0), atol=1.0e-6)


def test_padding_is_ignored_by_numpy_and_jax_encoders() -> None:
    clean = _square_observation(capacity=8, padding=0.0)
    poisoned = _square_observation(capacity=8, padding=np.nan)

    np.testing.assert_array_equal(_numpy_features(poisoned), _numpy_features(clean))
    jax_clean = _jax_features(_batched(clean))
    jax_poisoned = _jax_features(_batched(poisoned))
    np.testing.assert_array_equal(jax_poisoned, jax_clean)
    assert bool(jnp.all(jnp.isfinite(jax_poisoned)))


@pytest.mark.parametrize(
    ("mutation", "message"),
    (
        (lambda value: value.pop("yaw"), "missing=\\['yaw'\\]"),
        (lambda value: value.update(private_state=np.asarray(1.0)), "extra=\\['private_state'\\]"),
        (
            lambda value: value.__setitem__("position", np.zeros(3, dtype=np.float32)),
            "position.*shape",
        ),
        (
            lambda value: value.__setitem__(
                "track_mask", np.asarray((1, 1, 0, 1, 1, 0, 0, 0), dtype=np.int8)
            ),
            "contiguous valid prefix",
        ),
        (
            lambda value: value["centerline"].__setitem__((4, 0), 1.0),
            "explicitly closed",
        ),
        (
            lambda value: value.__setitem__("yaw_rate", np.asarray(np.inf, dtype=np.float32)),
            "finite values",
        ),
    ),
)
def test_numpy_encoder_rejects_malformed_public_schema(mutation, message: str) -> None:
    observation = _square_observation()
    mutation(observation)
    with pytest.raises(ValueError, match=message):
        _numpy_features(observation)


def test_jax_encoder_rejects_wrong_static_schema_and_feature_version() -> None:
    observation = _batched(_square_observation())
    malformed = dict(observation)
    malformed["centerline"] = malformed["centerline"][:, :-1]
    with pytest.raises(ValueError, match=r"observation field.*shape"):
        _jax_features(malformed)

    with pytest.raises(ValueError, match="requires preview_points=16"):
        encode_local_track_features_jax(
            observation,
            preview_points=15,
            preview_distance_m=40.0,
            max_speed_mps=15.0,
            control_dt_s=0.05,
            max_steering_angle_rad=0.6,
        )


def test_numpy_and_jax_encoders_have_close_batched_parity() -> None:
    observations = (
        _square_observation(progress=0.0),
        _square_observation(progress=0.2375, position=(7.0, -2.0), yaw=0.31),
        _square_observation(progress=0.9999, position=(-4.0, 3.0), yaw=-2.1),
    )
    expected = np.stack([_numpy_features(observation) for observation in observations])
    actual = np.asarray(jax.jit(_jax_features)(_batched(*observations)))
    assert actual.shape == (3, LOCAL_TRACK_FEATURE_DIM)
    assert actual.dtype == np.float32
    assert np.isfinite(actual).all()
    np.testing.assert_allclose(actual, expected, rtol=2.0e-6, atol=2.0e-6)


@pytest.mark.parametrize("track_seed", [3317, 8297])
def test_numpy_encoder_accepts_measured_official_train_length_rounding(track_seed: int) -> None:
    project = load_project_config(PROJECT_ROOT)
    track = pack_track(
        generate_track_candidate(track_seed, generation_spec_from_project(project)),
        track_capacity_from_project(project),
    )
    env = CarRacingEnv(
        project_config=project,
        level_id=1,
        track=track,
        backend="cpu_reference",
    )
    try:
        observation, _info = env.reset(seed=track_seed)
        features = _numpy_features(observation)
    finally:
        env.close()

    assert features.shape == (LOCAL_TRACK_FEATURE_DIM,)
    assert np.isfinite(features).all()


def test_wrapper_runs_compiled_encoder_on_cpu_reference_environment() -> None:
    project = load_project_config(PROJECT_ROOT)
    track = pack_track(
        generate_track_candidate(42, generation_spec_from_project(project)),
        track_capacity_from_project(project),
    )
    base = VecCarRacingEnv(
        num_envs=1,
        project_config=project,
        level_id=1,
        tracks=(track,),
        backend="cpu_reference",
    )
    env = LocalTrackObservationVecEnv(
        base,
        config=PpoObservationConfig(
            preview_points=16,
            preview_distance_m=40.0,
            max_speed_mps=15.0,
        ),
    )
    try:
        observation, initial_info = env.reset(seed=17)
        assert env.unwrapped is base
        assert env.metadata["autoreset_mode"] is AutoresetMode.NEXT_STEP
        assert env.feature_schema_version == LOCAL_TRACK_FEATURE_SCHEMA_VERSION
        assert env.single_observation_space.shape == (LOCAL_TRACK_FEATURE_DIM,)
        assert env.observation_space.shape == (1, LOCAL_TRACK_FEATURE_DIM)
        assert isinstance(observation, jax.Array)
        assert observation.shape == (1, LOCAL_TRACK_FEATURE_DIM)
        assert observation.dtype == jnp.float32
        assert bool(jnp.all(jnp.isfinite(observation)))
        assert env.observation_space.contains(np.asarray(observation))

        next_observation, reward, terminated, truncated, info = env.step(
            jnp.zeros((1, 2), dtype=jnp.float32)
        )
        assert next_observation.shape == (1, LOCAL_TRACK_FEATURE_DIM)
        assert bool(jnp.all(jnp.isfinite(next_observation)))
        assert reward.shape == terminated.shape == truncated.shape == (1,)
        assert tuple(info) == tuple(initial_info)
    finally:
        env.close()


def test_wrapper_validates_config_and_exact_public_input_schema() -> None:
    project = load_project_config(PROJECT_ROOT)
    track = pack_track(
        generate_track_candidate(43, generation_spec_from_project(project)),
        track_capacity_from_project(project),
    )

    def base_env() -> VecCarRacingEnv:
        return VecCarRacingEnv(
            num_envs=1,
            project_config=project,
            level_id=1,
            tracks=(track,),
            backend="cpu_reference",
        )

    base = base_env()
    try:
        with pytest.raises(ValueError, match="must be 16 for feature schema v1"):
            LocalTrackObservationVecEnv(
                base,
                config=PpoObservationConfig(
                    preview_points=15,
                    preview_distance_m=40.0,
                    max_speed_mps=15.0,
                ),
            )
    finally:
        base.close()

    base = base_env()
    try:
        with pytest.raises(ValueError, match=r"must be 15\.0 for the formal vehicle"):
            LocalTrackObservationVecEnv(
                base,
                config=PpoObservationConfig(
                    preview_points=16,
                    preview_distance_m=40.0,
                    max_speed_mps=14.0,
                ),
            )
    finally:
        base.close()

    base = base_env()
    modified = gym.vector.VectorWrapper(base)
    modified.single_observation_space = gym.spaces.Box(
        low=-np.inf,
        high=np.inf,
        shape=(1,),
        dtype=np.float32,
    )
    try:
        with pytest.raises(TypeError, match="public Dict schema"):
            LocalTrackObservationVecEnv(
                modified,
                config=PpoObservationConfig(
                    preview_points=16,
                    preview_distance_m=40.0,
                    max_speed_mps=15.0,
                ),
            )
    finally:
        base.close()
