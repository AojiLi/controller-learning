"""Tests for deterministic episode identity and restricted public info."""

from __future__ import annotations

from dataclasses import FrozenInstanceError, replace

import numpy as np
import pytest

from controller_learning.envs.episode import (
    PUBLIC_INFO_KEYS,
    EpisodeIdentity,
    build_reset_info,
    build_step_info,
    initialize_episode_identities,
    masked_next_episode,
    track_id_from_track,
    unbatch_public_info,
)
from controller_learning.envs.race_core import (
    RaceState,
    RaceStep,
    RaceTermination,
    TrackProjection,
)
from controller_learning.tracks.types import Track


def _track(*, seed: int, version: str = "trackgen-v1") -> Track:
    max_points = 5
    max_checkpoints = 1
    centerline = np.asarray(((0, 0), (1, 0), (1, 1), (0, 1), (0, 0)), dtype=np.float32)
    tangent = np.asarray(((1, 0), (0, 1), (-1, 0), (0, -1), (1, 0)), dtype=np.float32)
    cumulative_s = np.asarray((0, 1, 2, 3, 4), dtype=np.float32)
    checkpoint_center = np.zeros((max_checkpoints, 2), dtype=np.float32)
    checkpoint_tangent = np.asarray(((1, 0),), dtype=np.float32)
    checkpoint_s = np.asarray((4,), dtype=np.float32)
    return Track(
        seed=seed,
        generator_version=version,
        centerline_m=centerline,
        left_boundary_m=centerline,
        right_boundary_m=centerline,
        tangent=tangent,
        curvature_1pm=np.zeros(max_points, dtype=np.float32),
        cumulative_s_m=cumulative_s,
        track_mask=np.ones(max_points, dtype=np.bool_),
        checkpoint_center_m=checkpoint_center,
        checkpoint_tangent=checkpoint_tangent,
        checkpoint_s_m=checkpoint_s,
        checkpoint_mask=np.ones(max_checkpoints, dtype=np.bool_),
        start_pose=np.zeros(3, dtype=np.float32),
        point_count=max_points,
        checkpoint_count=max_checkpoints,
        length_m=4.0,
        width_m=7.0,
    )


def _race_step(
    reasons: np.ndarray,
    success: np.ndarray,
    elapsed_steps: np.ndarray,
) -> RaceStep:
    world_count = reasons.shape[0]
    zeros = np.zeros(world_count, dtype=np.float32)
    zero_int = np.zeros(world_count, dtype=np.int32)
    zero_bool = np.zeros(world_count, dtype=np.bool_)
    zero_xy = np.zeros((world_count, 2), dtype=np.float32)
    state = RaceState(
        previous_position_m=zero_xy,
        segment_index=zero_int,
        projected_s_m=zeros,
        unwrapped_s_m=zeros,
        legal_progress_m=zeros,
        next_checkpoint_index=zero_int,
        elapsed_steps=elapsed_steps,
    )
    projection = TrackProjection(
        segment_index=zero_int,
        segment_fraction=zeros,
        projected_s_m=zeros,
        closest_point_m=np.full_like(zero_xy, 98765.0),
        tangent=zero_xy,
        lateral_error_m=np.full_like(zeros, 12345.0),
        distance_m=zeros,
    )
    return RaceStep(
        state=state,
        projection=projection,
        reward=zeros,
        terminated=reasons != np.int32(RaceTermination.NONE),
        truncated=zero_bool,
        termination_reason=reasons,
        success=success,
        off_track=zero_bool,
        invalid_action=zero_bool,
        timeout=zero_bool,
        checkpoint_crossed=zero_bool,
        forward_progress_m=zeros,
        effective_half_width_m=zeros,
    )


def test_initialization_is_reproducible_domain_separated_and_batch_stable() -> None:
    first = initialize_episode_identities(123456, 4)
    again = initialize_episode_identities(np.uint32(123456), 4)
    smaller_batch = initialize_episode_identities(123456, 2)

    assert first.root_seed == np.uint32(123456)
    np.testing.assert_array_equal(first.world_index, np.arange(4, dtype=np.uint32))
    np.testing.assert_array_equal(first.episode_counter, np.zeros(4, dtype=np.uint32))
    np.testing.assert_array_equal(first.episode_seed, again.episode_seed)
    np.testing.assert_array_equal(first.controller_seed, again.controller_seed)
    np.testing.assert_array_equal(first.episode_seed[:2], smaller_batch.episode_seed)
    np.testing.assert_array_equal(first.controller_seed[:2], smaller_batch.controller_seed)
    assert np.all(first.episode_seed != first.controller_seed)


def test_seed_contract_has_stable_known_values() -> None:
    identities = initialize_episode_identities(123456, 4)

    # Locks the public SeedSequence entropy/spawn-path contract against accidental changes.
    assert identities.episode_seed.tolist() == [3080179479, 2422650640, 1073322509, 314032930]
    assert identities.controller_seed.tolist() == [
        2539469541,
        2170861154,
        3879327873,
        843055831,
    ]


def test_identity_is_frozen_and_owns_read_only_arrays() -> None:
    identity = initialize_episode_identities(7, 2)

    with pytest.raises(FrozenInstanceError):
        identity.root_seed = np.uint32(8)  # type: ignore[misc]
    with pytest.raises(ValueError, match="read-only"):
        identity.episode_seed[0] = np.uint32(3)


def test_masked_next_episode_only_changes_selected_worlds() -> None:
    current = initialize_episode_identities(991, 5)
    mask = np.asarray((False, True, False, True, False), dtype=np.bool_)
    advanced = masked_next_episode(current, mask)

    np.testing.assert_array_equal(advanced.episode_counter, (0, 1, 0, 1, 0))
    np.testing.assert_array_equal(advanced.world_index, current.world_index)
    assert advanced.root_seed == current.root_seed
    for field in ("episode_seed", "controller_seed"):
        before = getattr(current, field)
        after = getattr(advanced, field)
        np.testing.assert_array_equal(after[~mask], before[~mask])
        assert np.all(after[mask] != before[mask])
    assert np.all(advanced.episode_seed != advanced.controller_seed)

    unchanged = masked_next_episode(current, np.zeros(5, dtype=np.bool_))
    np.testing.assert_array_equal(unchanged.episode_counter, current.episode_counter)
    np.testing.assert_array_equal(unchanged.episode_seed, current.episode_seed)
    np.testing.assert_array_equal(unchanged.controller_seed, current.controller_seed)


@pytest.mark.parametrize("root_seed", (-1, 2**32, 1.5, True))
def test_initialization_rejects_invalid_root_seed(root_seed: object) -> None:
    expected_error = TypeError if isinstance(root_seed, (float, bool)) else ValueError
    with pytest.raises(expected_error):
        initialize_episode_identities(root_seed, 1)  # type: ignore[arg-type]


@pytest.mark.parametrize("num_envs", (0, -1, 1.5, True))
def test_initialization_rejects_invalid_world_count(num_envs: object) -> None:
    expected_error = TypeError if isinstance(num_envs, (float, bool)) else ValueError
    with pytest.raises(expected_error):
        initialize_episode_identities(1, num_envs)  # type: ignore[arg-type]


def test_identity_and_mask_shapes_and_dtypes_are_validated() -> None:
    current = initialize_episode_identities(1, 2)
    with pytest.raises(ValueError, match="shape"):
        masked_next_episode(current, np.asarray((True,), dtype=np.bool_))
    with pytest.raises(TypeError, match="dtype bool"):
        masked_next_episode(current, np.asarray((1, 0), dtype=np.int32))
    with pytest.raises(ValueError, match="episode_counter must have shape"):
        EpisodeIdentity(
            root_seed=np.uint32(1),
            world_index=np.arange(2, dtype=np.uint32),
            episode_counter=np.zeros(1, dtype=np.uint32),
            episode_seed=current.episode_seed,
            controller_seed=current.controller_seed,
        )

    exhausted = replace(
        current,
        episode_counter=np.full(2, np.iinfo(np.uint32).max, dtype=np.uint32),
    )
    with pytest.raises(OverflowError, match="cannot advance"):
        masked_next_episode(exhausted, np.asarray((False, True), dtype=np.bool_))


def test_reset_info_has_only_whitelisted_leading_arrays() -> None:
    identity = initialize_episode_identities(17, 2)
    tracks = (_track(seed=41), _track(seed=99))
    info = build_reset_info(identity, tracks, "v0.1")

    assert tuple(info) == PUBLIC_INFO_KEYS
    assert info["episode_seed"].dtype == np.uint32
    assert info["controller_seed"].dtype == np.uint32
    assert info["track_id"].dtype.kind == "U"
    assert info["benchmark_version"].dtype.kind == "U"
    assert info["termination_reason"].dtype == np.int32
    assert info["lap_completed"].dtype == np.bool_
    assert info["lap_time_s"].dtype == np.float32
    assert all(value.shape == (2,) for value in info.values())
    assert info["track_id"].tolist() == ["trackgen-v1:41", "trackgen-v1:99"]
    assert info["benchmark_version"].tolist() == ["v0.1", "v0.1"]
    np.testing.assert_array_equal(info["termination_reason"], RaceTermination.NONE)
    assert not info["lap_completed"].any()
    np.testing.assert_array_equal(info["lap_time_s"], 0.0)
    assert track_id_from_track(tracks[0]) == "trackgen-v1:41"


def test_step_info_exposes_outcome_without_race_or_backend_internals() -> None:
    identity = initialize_episode_identities(17, 2)
    tracks = (_track(seed=41), _track(seed=99))
    race_step = _race_step(
        np.asarray((RaceTermination.SUCCESS, RaceTermination.OFF_TRACK), dtype=np.int32),
        np.asarray((True, False), dtype=np.bool_),
        np.asarray((123, 4), dtype=np.int32),
    )
    info = build_step_info(identity, tracks, "v0.1", race_step, 0.05)

    assert tuple(info) == PUBLIC_INFO_KEYS
    np.testing.assert_array_equal(
        info["termination_reason"],
        (RaceTermination.SUCCESS, RaceTermination.OFF_TRACK),
    )
    np.testing.assert_array_equal(info["lap_completed"], (True, False))
    np.testing.assert_allclose(info["lap_time_s"], (6.15, 0.0), rtol=0.0, atol=1e-6)
    assert info["lap_time_s"].dtype == np.float32
    assert all(value.dtype != np.dtype(object) for value in info.values())

    forbidden_names = {
        "projection",
        "physics",
        "validator",
        "pool",
        "saturation",
        "lateral_error_m",
        "closest_point_m",
    }
    assert forbidden_names.isdisjoint(info)
    assert not any(np.any(value == 12345.0) for value in info.values())
    assert not any(np.any(value == 98765.0) for value in info.values())


def test_step_info_validates_race_shapes_values_and_control_period() -> None:
    identity = initialize_episode_identities(17, 2)
    tracks = (_track(seed=41), _track(seed=99))
    valid = _race_step(
        np.asarray((RaceTermination.NONE, RaceTermination.SUCCESS), dtype=np.int32),
        np.asarray((False, True), dtype=np.bool_),
        np.asarray((1, 2), dtype=np.int32),
    )

    with pytest.raises(ValueError, match="termination_reason must have shape"):
        build_step_info(
            identity, tracks, "v0.1", valid._replace(termination_reason=np.zeros(1)), 0.05
        )
    with pytest.raises(ValueError, match="unknown RaceTermination"):
        build_step_info(
            identity,
            tracks,
            "v0.1",
            valid._replace(termination_reason=np.asarray((0, 99), dtype=np.int32)),
            0.05,
        )
    with pytest.raises(ValueError, match="finite and positive"):
        build_step_info(identity, tracks, "v0.1", valid, 0.0)
    with pytest.raises(ValueError, match="one value per world"):
        build_reset_info(identity, tracks[:1], "v0.1")


def test_unbatch_returns_python_scalars_and_rejects_nonpublic_info() -> None:
    identity = initialize_episode_identities(17, 2)
    tracks = (_track(seed=41), _track(seed=99))
    vector_info = build_step_info(
        identity,
        tracks,
        "v0.1",
        _race_step(
            np.asarray((RaceTermination.NONE, RaceTermination.SUCCESS), dtype=np.int32),
            np.asarray((False, True), dtype=np.bool_),
            np.asarray((1, 20), dtype=np.int32),
        ),
        0.05,
    )

    scalar = unbatch_public_info(vector_info, 1)
    assert tuple(scalar) == PUBLIC_INFO_KEYS
    assert type(scalar["episode_seed"]) is int
    assert type(scalar["controller_seed"]) is int
    assert type(scalar["track_id"]) is str
    assert type(scalar["benchmark_version"]) is str
    assert type(scalar["termination_reason"]) is int
    assert type(scalar["lap_completed"]) is bool
    assert type(scalar["lap_time_s"]) is float
    assert scalar["track_id"] == "trackgen-v1:99"
    assert scalar["lap_completed"] is True
    assert scalar["lap_time_s"] == pytest.approx(1.0)

    with pytest.raises(ValueError, match="public whitelist"):
        unbatch_public_info({**vector_info, "projection": np.zeros(2)})
    with pytest.raises(IndexError):
        unbatch_public_info(vector_info, 2)
