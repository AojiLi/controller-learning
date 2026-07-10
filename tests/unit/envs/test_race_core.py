from __future__ import annotations

from collections.abc import Sequence

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from controller_learning.envs.race_core import (
    RaceCoreConfig,
    RaceState,
    RaceTermination,
    body_to_world,
    masked_reset_race_state,
    project_to_track,
    reset_race_state,
    step_race_core,
    world_to_body,
    wrap_angle,
)
from controller_learning.tracks.types import Track, stack_tracks

MAX_TRACK_POINTS = 8
MAX_CHECKPOINTS = 4


def _make_track(
    points: Sequence[Sequence[float]],
    checkpoints: Sequence[tuple[Sequence[float], Sequence[float], float]],
    *,
    seed: int = 1,
    width_m: float = 4.0,
) -> Track:
    valid_points = np.asarray(points, dtype=np.float32)
    point_count = valid_points.shape[0]
    assert point_count <= MAX_TRACK_POINTS
    assert np.array_equal(valid_points[0], valid_points[-1])

    segment_vectors = np.diff(valid_points, axis=0)
    segment_lengths = np.linalg.norm(segment_vectors, axis=1)
    segment_tangents = segment_vectors / segment_lengths[:, None]
    valid_tangents = np.concatenate((segment_tangents, segment_tangents[:1]), axis=0)
    cumulative_s = np.concatenate(
        (np.zeros(1, dtype=np.float32), np.cumsum(segment_lengths, dtype=np.float32))
    )
    normals = np.stack((-valid_tangents[:, 1], valid_tangents[:, 0]), axis=1)

    def padded(values: np.ndarray, shape: tuple[int, ...]) -> np.ndarray:
        result = np.zeros(shape, dtype=np.float32)
        result[: values.shape[0]] = values
        return result

    checkpoint_count = len(checkpoints)
    checkpoint_center = np.zeros((MAX_CHECKPOINTS, 2), dtype=np.float32)
    checkpoint_tangent = np.zeros((MAX_CHECKPOINTS, 2), dtype=np.float32)
    checkpoint_s = np.zeros(MAX_CHECKPOINTS, dtype=np.float32)
    for index, (center, tangent, distance) in enumerate(checkpoints):
        checkpoint_center[index] = center
        checkpoint_tangent[index] = tangent
        checkpoint_s[index] = distance

    centerline = padded(valid_points, (MAX_TRACK_POINTS, 2))
    return Track(
        seed=seed,
        generator_version="test-v1",
        centerline_m=centerline,
        left_boundary_m=padded(
            valid_points + 0.5 * width_m * normals,
            (MAX_TRACK_POINTS, 2),
        ),
        right_boundary_m=padded(
            valid_points - 0.5 * width_m * normals,
            (MAX_TRACK_POINTS, 2),
        ),
        tangent=padded(valid_tangents, (MAX_TRACK_POINTS, 2)),
        curvature_1pm=np.zeros(MAX_TRACK_POINTS, dtype=np.float32),
        cumulative_s_m=padded(cumulative_s, (MAX_TRACK_POINTS,)),
        track_mask=np.arange(MAX_TRACK_POINTS) < point_count,
        checkpoint_center_m=checkpoint_center,
        checkpoint_tangent=checkpoint_tangent,
        checkpoint_s_m=checkpoint_s,
        checkpoint_mask=np.arange(MAX_CHECKPOINTS) < checkpoint_count,
        start_pose=np.array([*valid_points[0], 0.0], dtype=np.float32),
        point_count=point_count,
        checkpoint_count=checkpoint_count,
        length_m=float(cumulative_s[-1]),
        width_m=width_m,
    )


def _loop_track(*, seed: int = 1, width_m: float = 4.0) -> Track:
    # The final segment approaches the start in +x, so the start/finish tangent is continuous.
    points = ((0, 0), (10, 0), (10, 10), (-10, 10), (-10, 0), (0, 0))
    checkpoints = (
        ((5, 0), (1, 0), 5.0),
        ((10, 5), (0, 1), 15.0),
        ((0, 0), (1, 0), 60.0),
    )
    return _make_track(points, checkpoints, seed=seed, width_m=width_m)


def _config(**changes: float | int) -> RaceCoreConfig:
    values: dict[str, float | int] = {
        "control_dt_s": 0.05,
        "vehicle_width_m": 1.6,
        "safety_margin_m": 0.1,
        "projection_backward_segments": 1,
        "projection_forward_segments": 2,
        "min_timeout_s": 60.0,
        "timeout_reference_speed_mps": 3.0,
    }
    values.update(changes)
    return RaceCoreConfig(**values)  # type: ignore[arg-type]


def test_angle_and_frame_transforms_are_inverse() -> None:
    angles = jnp.array([-4.0 * np.pi, -np.pi, np.pi, 3.0 * np.pi, 0.25])
    np.testing.assert_allclose(
        np.asarray(wrap_angle(angles)),
        np.array([0.0, -np.pi, -np.pi, -np.pi, 0.25]),
        atol=1e-6,
    )

    world_vectors = jnp.array(((1.0, 0.0), (2.0, -3.0)), dtype=jnp.float32)
    yaw = jnp.array((np.pi / 2.0, -0.7), dtype=jnp.float32)
    body_vectors = world_to_body(world_vectors, yaw)
    np.testing.assert_allclose(np.asarray(body_vectors[0]), (0.0, -1.0), atol=1e-6)
    np.testing.assert_allclose(
        np.asarray(body_to_world(body_vectors, yaw)),
        np.asarray(world_vectors),
        atol=1e-6,
    )


@pytest.mark.parametrize(
    ("field", "value"),
    (
        ("control_dt_s", 0.0),
        ("vehicle_width_m", -1.0),
        ("safety_margin_m", -0.1),
        ("projection_backward_segments", -1),
        ("projection_forward_segments", 0),
        ("timeout_reference_speed_mps", np.inf),
    ),
)
def test_config_rejects_invalid_rules(field: str, value: float | int) -> None:
    with pytest.raises(ValueError):
        _config(**{field: value})


def test_projection_reports_segment_distance_and_progress() -> None:
    batch = stack_tracks([_loop_track()])
    projection = project_to_track(
        batch,
        jnp.array(((3.0, 0.5),), dtype=jnp.float32),
        jnp.array((0,), dtype=jnp.int32),
        _config(),
    )

    assert int(projection.segment_index[0]) == 0
    assert float(projection.segment_fraction[0]) == pytest.approx(0.3)
    assert float(projection.projected_s_m[0]) == pytest.approx(3.0)
    assert float(projection.lateral_error_m[0]) == pytest.approx(0.5)
    assert float(projection.distance_m[0]) == pytest.approx(0.5)


def test_projection_window_prevents_hairpin_progress_jump() -> None:
    hairpin = _make_track(
        ((0, 0), (10, 0), (12, 3), (10, 1), (0, 1), (-2, 3), (0, 0)),
        (((0, 0), (1, 0), 31.0),),
    )
    projection = project_to_track(
        stack_tracks([hairpin]),
        jnp.array(((5.0, 0.9),), dtype=jnp.float32),
        jnp.array((0,), dtype=jnp.int32),
        _config(projection_forward_segments=1),
    )

    # The point is 0.1 m from remote segment 3 but prior topology only admits segments 5, 0, 1.
    assert int(projection.segment_index[0]) == 0
    assert float(projection.projected_s_m[0]) == pytest.approx(5.0)
    assert float(projection.distance_m[0]) == pytest.approx(0.9)


def test_reverse_then_forward_cannot_earn_progress_twice() -> None:
    batch = stack_tracks([_loop_track()])
    config = _config()
    state = reset_race_state(batch)

    first = step_race_core(batch, state, jnp.array(((2.0, 0.0),)), jnp.array((False,)), config)
    reverse = step_race_core(
        batch,
        first.state,
        jnp.array(((1.0, 0.0),)),
        jnp.array((False,)),
        config,
    )
    replay = step_race_core(
        batch,
        reverse.state,
        jnp.array(((2.0, 0.0),)),
        jnp.array((False,)),
        config,
    )
    new_progress = step_race_core(
        batch,
        replay.state,
        jnp.array(((3.0, 0.0),)),
        jnp.array((False,)),
        config,
    )

    assert float(first.forward_progress_m[0]) == pytest.approx(2.0)
    assert float(reverse.forward_progress_m[0]) == pytest.approx(0.0)
    assert float(replay.forward_progress_m[0]) == pytest.approx(0.0)
    assert float(new_progress.forward_progress_m[0]) == pytest.approx(1.0)
    assert float(new_progress.state.legal_progress_m[0]) == pytest.approx(3.0)


def test_checkpoints_are_ordered_and_finish_crossing_succeeds() -> None:
    batch = stack_tracks([_loop_track()])
    config = _config()
    reset = reset_race_state(batch)

    # Crossing checkpoint 1 while checkpoint 0 is still expected must not count.
    out_of_order_state = RaceState(
        previous_position_m=jnp.array(((10.0, 4.0),)),
        segment_index=jnp.array((1,), dtype=jnp.int32),
        projected_s_m=jnp.array((14.0,)),
        unwrapped_s_m=jnp.array((14.0,)),
        legal_progress_m=jnp.array((14.0,)),
        next_checkpoint_index=jnp.array((0,), dtype=jnp.int32),
        elapsed_steps=jnp.array((0,), dtype=jnp.int32),
    )
    skipped = step_race_core(
        batch,
        out_of_order_state,
        jnp.array(((10.0, 6.0),)),
        jnp.array((False,)),
        config,
    )
    assert int(skipped.state.next_checkpoint_index[0]) == 0

    positions = ((6, 0), (10, 4), (10, 6), (0, 10), (-10, 1), (-1, 0), (1, 0))
    result = None
    state = reset
    for position in positions:
        result = step_race_core(
            batch,
            state,
            jnp.asarray((position,), dtype=jnp.float32),
            jnp.array((False,)),
            config,
        )
        state = result.state

    assert result is not None
    assert bool(result.success[0])
    assert bool(result.terminated[0])
    assert not bool(result.truncated[0])
    assert int(result.termination_reason[0]) == RaceTermination.SUCCESS
    assert int(result.state.next_checkpoint_index[0]) == 3
    assert float(result.state.legal_progress_m[0]) == pytest.approx(60.0)


def test_effective_boundary_accounts_for_vehicle_width_and_margin() -> None:
    batch = stack_tracks([_loop_track(width_m=4.0)])
    config = _config(vehicle_width_m=1.6, safety_margin_m=0.2)
    state = reset_race_state(batch)

    inside = step_race_core(
        batch,
        state,
        jnp.array(((2.0, 0.99),)),
        jnp.array((False,)),
        config,
    )
    outside = step_race_core(
        batch,
        inside.state,
        jnp.array(((3.0, 1.01),)),
        jnp.array((False,)),
        config,
    )

    assert float(inside.effective_half_width_m[0]) == pytest.approx(1.0)
    assert not bool(inside.off_track[0])
    assert bool(outside.off_track[0])
    assert bool(outside.terminated[0])
    assert int(outside.termination_reason[0]) == RaceTermination.OFF_TRACK
    assert float(outside.reward[0]) < 0.0


def test_timeout_is_integer_truncation_and_loses_to_invalid_action() -> None:
    batch = stack_tracks([_loop_track()])
    config = _config(control_dt_s=1.0)
    state = reset_race_state(batch)._replace(elapsed_steps=jnp.array((58,), dtype=jnp.int32))

    before_timeout = step_race_core(
        batch,
        state,
        jnp.array(((0.0, 0.0),)),
        jnp.array((False,)),
        config,
    )
    timeout = step_race_core(
        batch,
        before_timeout.state,
        jnp.array(((0.0, 0.0),)),
        jnp.array((False,)),
        config,
    )
    invalid = step_race_core(
        batch,
        before_timeout.state,
        jnp.array(((0.0, 0.0),)),
        jnp.array((True,)),
        config,
    )

    assert not bool(before_timeout.truncated[0])
    assert bool(timeout.truncated[0])
    assert not bool(timeout.terminated[0])
    assert int(timeout.termination_reason[0]) == RaceTermination.TIMEOUT
    assert bool(invalid.terminated[0])
    assert not bool(invalid.truncated[0])
    assert int(invalid.termination_reason[0]) == RaceTermination.INVALID_ACTION


def test_termination_priority_is_invalid_then_offtrack_then_success_then_timeout() -> None:
    batch = stack_tracks([_loop_track()])
    config = _config(control_dt_s=1.0)
    final_state = RaceState(
        previous_position_m=jnp.array(((-1.0, 0.0),)),
        segment_index=jnp.array((4,), dtype=jnp.int32),
        projected_s_m=jnp.array((59.0,)),
        unwrapped_s_m=jnp.array((59.0,)),
        legal_progress_m=jnp.array((59.0,)),
        next_checkpoint_index=jnp.array((2,), dtype=jnp.int32),
        elapsed_steps=jnp.array((59,), dtype=jnp.int32),
    )

    success = step_race_core(
        batch,
        final_state,
        jnp.array(((1.0, 0.0),)),
        jnp.array((False,)),
        config,
    )
    offtrack = step_race_core(
        batch,
        final_state,
        jnp.array(((1.0, 2.0),)),
        jnp.array((False,)),
        config,
    )
    invalid = step_race_core(
        batch,
        final_state,
        jnp.array(((1.0, 2.0),)),
        jnp.array((True,)),
        config,
    )

    assert bool(success.success[0])
    assert not bool(success.truncated[0])
    assert int(success.termination_reason[0]) == RaceTermination.SUCCESS
    assert bool(offtrack.off_track[0])
    assert not bool(offtrack.truncated[0])
    assert int(offtrack.termination_reason[0]) == RaceTermination.OFF_TRACK
    assert bool(invalid.off_track[0])
    assert int(invalid.termination_reason[0]) == RaceTermination.INVALID_ACTION


def test_masked_reset_preserves_every_unmasked_field() -> None:
    batch = stack_tracks([_loop_track(seed=1), _loop_track(seed=2)])
    reset = reset_race_state(batch)
    current = RaceState(
        previous_position_m=jnp.array(((3.0, 4.0), (5.0, 6.0))),
        segment_index=jnp.array((2, 3), dtype=jnp.int32),
        projected_s_m=jnp.array((7.0, 8.0)),
        unwrapped_s_m=jnp.array((9.0, 10.0)),
        legal_progress_m=jnp.array((11.0, 12.0)),
        next_checkpoint_index=jnp.array((1, 2), dtype=jnp.int32),
        elapsed_steps=jnp.array((13, 14), dtype=jnp.int32),
    )
    result = masked_reset_race_state(current, reset, jnp.array((True, False)))

    for field in RaceState._fields:
        result_value = np.asarray(getattr(result, field))
        reset_value = np.asarray(getattr(reset, field))
        current_value = np.asarray(getattr(current, field))
        np.testing.assert_array_equal(result_value[0], reset_value[0])
        np.testing.assert_array_equal(result_value[1], current_value[1])


def test_batch_one_and_different_worlds_remain_independent() -> None:
    one_batch = stack_tracks([_loop_track(seed=1)])
    one = step_race_core(
        one_batch,
        reset_race_state(one_batch),
        jnp.array(((2.0, 0.0),)),
        jnp.array((False,)),
        _config(),
    )
    assert one.reward.shape == (1,)
    assert one.state.previous_position_m.shape == (1, 2)

    batch = stack_tracks([_loop_track(seed=2), _loop_track(seed=3)])
    state = reset_race_state(batch)._replace(segment_index=jnp.array((0, 1), dtype=jnp.int32))
    result = step_race_core(
        batch,
        state,
        jnp.array(((2.0, 0.0), (10.0, 2.0))),
        jnp.array((False, True)),
        _config(),
    )
    assert int(result.projection.segment_index[0]) == 0
    assert int(result.projection.segment_index[1]) == 1
    assert not bool(result.terminated[0])
    assert bool(result.terminated[1])
    assert int(result.termination_reason[1]) == RaceTermination.INVALID_ACTION


def test_one_compiled_executable_accepts_different_track_seeds() -> None:
    config = _config()
    first = jax.tree.map(jnp.asarray, stack_tracks([_loop_track(seed=11)]))
    second = jax.tree.map(jnp.asarray, stack_tracks([_loop_track(seed=999)]))
    first_state = reset_race_state(first)
    second_state = reset_race_state(second)

    compiled = (
        jax.jit(
            lambda tracks, state, position: step_race_core(
                tracks,
                state,
                position,
                jnp.zeros(position.shape[0], dtype=bool),
                config,
            )
        )
        .lower(first, first_state, jnp.array(((1.0, 0.0),)))
        .compile()
    )

    first_result = compiled(first, first_state, jnp.array(((1.0, 0.0),)))
    second_result = compiled(second, second_state, jnp.array(((2.0, 0.0),)))
    assert float(first_result.state.legal_progress_m[0]) == pytest.approx(1.0)
    assert float(second_result.state.legal_progress_m[0]) == pytest.approx(2.0)
