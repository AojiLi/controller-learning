"""Tests for public-trajectory metrics and canonical M8 metric artifacts."""

from __future__ import annotations

import hashlib
import io
import zipfile
from pathlib import Path
from types import MappingProxyType

import numpy as np
import pytest

from controller_learning.control import EpisodeRunResult
from controller_learning.evaluation.final_metrics import (
    FINAL_METRICS_EPISODE_COUNT,
    AggregateMetricSummary,
    EpisodeMetricSamples,
    FinalMetricsArtifactError,
    MetricActionLimits,
    build_final_metrics_data,
    canonical_final_metrics_bytes,
    compute_episode_metric_samples,
    load_final_metrics_npz,
    summarize_episode_metrics,
    summarize_final_metrics,
    summarize_metric_episodes,
    write_final_metrics_npz,
)
from controller_learning.evaluation.trajectory import (
    EpisodeTrajectory,
    RecordedControllerEpisode,
)


def _recorded_episode(
    *,
    centerline: np.ndarray,
    positions: np.ndarray,
    velocities: np.ndarray,
    actions: np.ndarray,
    compute_times: tuple[float, ...],
    track_id: int = 101,
) -> RecordedControllerEpisode:
    center = np.asarray(centerline, dtype=np.float32)
    positions = np.asarray(positions, dtype=np.float32)
    velocities = np.asarray(velocities, dtype=np.float32)
    actions = np.asarray(actions, dtype=np.float32)
    step_count = int(actions.shape[0])
    assert positions.shape == velocities.shape == (step_count, 2)
    frame_position = np.concatenate((center[:1], positions), axis=0)
    frame_velocity = np.concatenate((np.zeros((1, 2), dtype=np.float32), velocities), axis=0)
    track_length = float(np.sum(np.linalg.norm(np.diff(center.astype(np.float64), axis=0), axis=1)))
    reset_info = MappingProxyType(
        {
            "episode_seed": 1101,
            "controller_seed": 2202,
            "track_id": track_id,
            "benchmark_version": "0.1",
            "termination_reason": 0,
            "lap_completed": False,
            "lap_time_s": 0.0,
        }
    )
    final_info = MappingProxyType(
        {
            **dict(reset_info),
            "termination_reason": 1,
            "lap_completed": True,
            "lap_time_s": float(step_count * 0.05),
        }
    )
    terminated = np.zeros(step_count, dtype=np.bool_)
    terminated[-1] = True
    trajectory = EpisodeTrajectory(
        reset_info=reset_info,
        final_info=final_info,
        centerline_m=center,
        left_boundary_m=center,
        right_boundary_m=center,
        track_mask=np.ones(center.shape[0], dtype=np.bool_),
        track_length_m=track_length,
        position_m=frame_position,
        yaw_rad=np.zeros(step_count + 1, dtype=np.float32),
        velocity_body_mps=frame_velocity,
        yaw_rate_rad_s=np.zeros(step_count + 1, dtype=np.float32),
        steering_angle_rad=np.zeros(step_count + 1, dtype=np.float32),
        track_progress=np.linspace(0.0, 1.0, step_count + 1, dtype=np.float32),
        action=actions,
        reward=np.zeros(step_count, dtype=np.float32),
        terminated=terminated,
        truncated=np.zeros(step_count, dtype=np.bool_),
    )
    result = EpisodeRunResult(
        steps=step_count,
        total_reward=0.0,
        terminated=True,
        truncated=False,
        final_info=final_info,
        debug_commands=(),
        compute_times_s=compute_times,
    )
    return RecordedControllerEpisode(result=result, trajectory=trajectory)


def _square() -> np.ndarray:
    return np.asarray(((0, 0), (10, 0), (10, 10), (0, 10), (0, 0)), dtype=np.float32)


def _sample(
    *,
    track_id: int,
    reset_seed: int,
    speed: np.ndarray,
    lateral: np.ndarray,
    action: np.ndarray | None = None,
    steering_saturated: np.ndarray | None = None,
    longitudinal_saturated: np.ndarray | None = None,
) -> EpisodeMetricSamples:
    speed = np.asarray(speed, dtype=np.float64)
    count = int(speed.size)
    return EpisodeMetricSamples(
        track_id=track_id,
        reset_seed=reset_seed,
        compute_time_s=np.linspace(0.001, 0.002, count, dtype=np.float64),
        speed_mps=speed,
        lateral_error_m=np.asarray(lateral, dtype=np.float64),
        requested_action=(
            np.zeros((count, 2), dtype=np.float32)
            if action is None
            else np.asarray(action, dtype=np.float32)
        ),
        steering_saturated=(
            np.zeros(count, dtype=np.bool_)
            if steering_saturated is None
            else np.asarray(steering_saturated, dtype=np.bool_)
        ),
        longitudinal_saturated=(
            np.zeros(count, dtype=np.bool_)
            if longitudinal_saturated is None
            else np.asarray(longitudinal_saturated, dtype=np.bool_)
        ),
    )


def _artifact_samples() -> tuple[EpisodeMetricSamples, ...]:
    return tuple(
        _sample(
            track_id=10_000 + index,
            reset_seed=index,
            speed=np.asarray((index + 1.0, index + 2.0)),
            lateral=np.asarray((-0.1 * index, 0.2 * index)),
            action=np.asarray(((0.0, 0.0), (0.01 * index, 0.1 * index))),
            steering_saturated=np.asarray((False, index % 2 == 0)),
            longitudinal_saturated=np.asarray((False, index % 3 == 0)),
        )
        for index in range(FINAL_METRICS_EPISODE_COUNT)
    )


def test_public_episode_metrics_match_hand_computed_post_step_values() -> None:
    episode = _recorded_episode(
        centerline=_square(),
        positions=np.asarray(((2.0, 1.0), (4.0, -2.0))),
        velocities=np.asarray(((3.0, 4.0), (0.0, 2.0))),
        actions=np.asarray(((0.5, 4.0), (0.6, -9.0))),
        compute_times=(0.001, 0.002),
    )
    samples = compute_episode_metric_samples(
        episode,
        reset_seed=0,
        action_limits=MetricActionLimits(
            max_steering_angle_rad=0.5,
            max_acceleration_mps2=4.0,
            max_deceleration_mps2=8.0,
        ),
    )

    np.testing.assert_array_equal(samples.compute_time_s, (0.001, 0.002))
    np.testing.assert_allclose(samples.speed_mps, (5.0, 2.0), rtol=0.0, atol=0.0)
    np.testing.assert_allclose(samples.lateral_error_m, (1.0, -2.0), rtol=0.0, atol=0.0)
    np.testing.assert_array_equal(samples.requested_action, episode.trajectory.action)
    np.testing.assert_array_equal(samples.steering_saturated, (False, True))
    np.testing.assert_array_equal(samples.longitudinal_saturated, (False, True))
    np.testing.assert_allclose(samples.steering_rate_rad_s, (2.000000476837158,), rtol=0.0)
    np.testing.assert_allclose(samples.acceleration_rate_mps3, (-260.0,), rtol=0.0, atol=0.0)

    summary = summarize_episode_metrics(samples)
    assert summary.transition_count == 2
    assert summary.action_delta_count == 1
    assert summary.mean_speed_mps == 3.5
    assert summary.lateral_error_rms_m == pytest.approx(np.sqrt(2.5))
    assert summary.lateral_error_abs_p95_m == pytest.approx(1.95)
    assert summary.lateral_error_abs_max_m == 2.0
    assert summary.steering_saturation_rate == 0.5
    assert summary.longitudinal_saturation_rate == 0.5
    assert summary.steering_rate_rms_rad_s == pytest.approx(2.000000476837158)
    assert summary.acceleration_rate_rms_mps3 == 260.0
    for array in (
        samples.compute_time_s,
        samples.speed_mps,
        samples.lateral_error_m,
        samples.requested_action,
        samples.steering_saturated,
        samples.longitudinal_saturated,
    ):
        assert not array.flags.writeable


def test_projection_window_wraps_across_the_centerline_topology_seam() -> None:
    angles = np.linspace(0.0, 2.0 * np.pi, 25, dtype=np.float64)
    centerline = np.stack((10.0 * np.cos(angles), 10.0 * np.sin(angles)), axis=1).astype(np.float32)
    centerline[-1] = centerline[0]

    def offset_midpoint(segment: int) -> np.ndarray:
        start = centerline[segment].astype(np.float64)
        end = centerline[segment + 1].astype(np.float64)
        tangent = (end - start) / np.linalg.norm(end - start)
        left_normal = np.asarray((-tangent[1], tangent[0]))
        return 0.5 * (start + end) + 0.2 * left_normal

    episode = _recorded_episode(
        centerline=centerline,
        positions=np.stack((offset_midpoint(23), offset_midpoint(0))),
        velocities=np.zeros((2, 2)),
        actions=np.zeros((2, 2)),
        compute_times=(0.001, 0.001),
    )
    samples = compute_episode_metric_samples(
        episode,
        reset_seed=0,
        action_limits=MetricActionLimits(0.5, 4.0, 8.0),
    )

    np.testing.assert_allclose(samples.lateral_error_m, (0.2, 0.2), rtol=0.0, atol=1.0e-6)


def test_saturation_is_strictly_outside_each_asymmetric_action_bound() -> None:
    upper_steer = np.nextafter(np.float32(0.5), np.float32(np.inf))
    lower_steer = np.nextafter(np.float32(-0.5), np.float32(-np.inf))
    upper_accel = np.nextafter(np.float32(4.0), np.float32(np.inf))
    lower_accel = np.nextafter(np.float32(-8.0), np.float32(-np.inf))
    episode = _recorded_episode(
        centerline=_square(),
        positions=np.asarray(((1, 0), (2, 0), (3, 0), (4, 0))),
        velocities=np.zeros((4, 2)),
        actions=np.asarray(
            (
                (-0.5, -8.0),
                (0.5, 4.0),
                (upper_steer, upper_accel),
                (lower_steer, lower_accel),
            ),
            dtype=np.float32,
        ),
        compute_times=(0.0, 0.0, 0.0, 0.0),
    )
    samples = compute_episode_metric_samples(
        episode,
        reset_seed=0,
        action_limits=MetricActionLimits(0.5, 4.0, 8.0),
    )

    np.testing.assert_array_equal(samples.steering_saturated, (False, False, True, True))
    np.testing.assert_array_equal(samples.longitudinal_saturated, (False, False, True, True))


def test_aggregate_uses_all_transitions_and_never_crosses_episode_action_boundaries() -> None:
    first = _sample(
        track_id=1,
        reset_seed=0,
        speed=np.asarray((100.0,)),
        lateral=np.asarray((100.0,)),
        action=np.asarray(((1000.0, 1000.0),)),
        steering_saturated=np.asarray((True,)),
    )
    second = _sample(
        track_id=2,
        reset_seed=1,
        speed=np.zeros(9),
        lateral=np.zeros(9),
        action=np.zeros((9, 2)),
    )
    first_summary = summarize_episode_metrics(first)
    second_summary = summarize_episode_metrics(second)
    aggregate = summarize_metric_episodes((first, second))

    assert isinstance(aggregate, AggregateMetricSummary)
    assert aggregate.episode_count == 2
    assert aggregate.transition_count == 10
    assert aggregate.action_delta_count == 8
    assert aggregate.mean_speed_mps == 10.0
    assert aggregate.lateral_error_abs_p95_m == pytest.approx(55.0)
    assert aggregate.lateral_error_abs_p95_m != pytest.approx(
        (first_summary.lateral_error_abs_p95_m + second_summary.lateral_error_abs_p95_m) / 2.0
    )
    assert aggregate.steering_saturation_rate == 0.1
    assert aggregate.steering_rate_rms_rad_s == 0.0
    assert aggregate.acceleration_rate_rms_mps3 == 0.0


def test_canonical_metrics_npz_is_deterministic_strict_and_summary_recomputable(
    tmp_path: Path,
) -> None:
    data = build_final_metrics_data("pid", _artifact_samples())
    first = canonical_final_metrics_bytes(data)
    second = canonical_final_metrics_bytes(data)
    first_path = tmp_path / "first.npz"
    second_path = tmp_path / "second.npz"
    first_artifact = write_final_metrics_npz(data, first_path)
    second_artifact = write_final_metrics_npz(data, second_path)

    assert first == second == first_path.read_bytes() == second_path.read_bytes()
    assert first_artifact.sha256 == second_artifact.sha256 == hashlib.sha256(first).hexdigest()
    assert first_artifact.size_bytes == second_artifact.size_bytes == len(first)
    loaded = load_final_metrics_npz(
        first_path,
        expected_sha256=first_artifact.sha256,
        expected_size_bytes=first_artifact.size_bytes,
    )
    assert loaded.data.controller_name == "pid"
    np.testing.assert_array_equal(loaded.data.episode_offsets, np.arange(0, 41, 2))
    np.testing.assert_array_equal(loaded.data.reset_seed, np.arange(20, dtype=np.uint32))
    assert not loaded.data.requested_action.flags.writeable
    assert summarize_final_metrics(loaded.data) == summarize_metric_episodes(_artifact_samples())

    expected_names = [
        "schema_version.npy",
        "benchmark_version.npy",
        "controller_name.npy",
        "track_id.npy",
        "reset_seed.npy",
        "episode_offsets.npy",
        "compute_time_s.npy",
        "speed_mps.npy",
        "lateral_error_m.npy",
        "requested_action.npy",
        "steering_saturated.npy",
        "longitudinal_saturated.npy",
    ]
    with zipfile.ZipFile(io.BytesIO(first), mode="r") as archive:
        assert archive.namelist() == expected_names
        assert all(item.compress_type == zipfile.ZIP_STORED for item in archive.infolist())
    with np.load(io.BytesIO(first), allow_pickle=False) as archive:
        assert archive["episode_offsets"].dtype == np.dtype("<i8")
        assert archive["compute_time_s"].dtype == np.dtype("<f8")
        assert archive["speed_mps"].dtype == np.dtype("<f8")
        assert archive["lateral_error_m"].dtype == np.dtype("<f8")
        assert archive["requested_action"].dtype == np.dtype("<f4")
        assert archive["steering_saturated"].dtype == np.dtype("|b1")


def test_metrics_loader_rejects_tampering_noncanonical_archives_and_insecure_paths(
    tmp_path: Path,
) -> None:
    data = build_final_metrics_data("ppo", _artifact_samples())
    canonical = tmp_path / "canonical.npz"
    artifact = write_final_metrics_npz(data, canonical)
    with pytest.raises(FinalMetricsArtifactError, match="SHA-256"):
        load_final_metrics_npz(canonical, expected_sha256="0" * 64)
    with pytest.raises(FinalMetricsArtifactError, match="size differs"):
        load_final_metrics_npz(canonical, expected_size_bytes=artifact.size_bytes + 1)
    with pytest.raises(ValueError, match="lowercase SHA-256"):
        load_final_metrics_npz(canonical, expected_sha256="invalid")

    tampered = tmp_path / "tampered.npz"
    content = bytearray(canonical.read_bytes())
    content[len(content) // 2] ^= 1
    tampered.write_bytes(content)
    with pytest.raises(FinalMetricsArtifactError, match=r"valid non-pickled|canonical"):
        load_final_metrics_npz(tampered)

    noncanonical = tmp_path / "noncanonical.npz"
    arrays = {
        "schema_version": np.asarray(1, dtype=np.uint32),
        "benchmark_version": np.asarray(b"0.1", dtype="|S16"),
        "controller_name": np.asarray(b"ppo", dtype="|S32"),
        "track_id": data.track_id,
        "reset_seed": data.reset_seed,
        "episode_offsets": data.episode_offsets,
        "compute_time_s": data.compute_time_s,
        "speed_mps": data.speed_mps,
        "lateral_error_m": data.lateral_error_m,
        "requested_action": data.requested_action,
        "steering_saturated": data.steering_saturated,
        "longitudinal_saturated": data.longitudinal_saturated,
    }
    np.savez(noncanonical, **arrays)
    with pytest.raises(FinalMetricsArtifactError, match=r"ZIP metadata|canonical"):
        load_final_metrics_npz(noncanonical)

    source_link = tmp_path / "source-link.npz"
    source_link.symlink_to(canonical)
    with pytest.raises(FinalMetricsArtifactError, match="non-symlink"):
        load_final_metrics_npz(source_link)

    real_directory = tmp_path / "real"
    real_directory.mkdir()
    directory_link = tmp_path / "linked"
    directory_link.symlink_to(real_directory, target_is_directory=True)
    with pytest.raises(FinalMetricsArtifactError, match=r"non-symlink|symbolic"):
        write_final_metrics_npz(data, directory_link / "metrics.npz")


def test_metric_artifact_rejects_wrong_row_binding_and_invalid_samples() -> None:
    samples = list(_artifact_samples())
    samples[1] = _sample(
        track_id=samples[1].track_id,
        reset_seed=7,
        speed=np.ones(1),
        lateral=np.zeros(1),
    )
    with pytest.raises(ValueError, match=r"0\.\.19"):
        build_final_metrics_data("pid", samples)
    with pytest.raises(ValueError, match="exactly 20"):
        build_final_metrics_data("pid", _artifact_samples()[:-1])
    with pytest.raises(ValueError, match="non-negative"):
        _sample(
            track_id=1,
            reset_seed=0,
            speed=np.asarray((-1.0,)),
            lateral=np.zeros(1),
        )


def test_final_metrics_source_has_no_private_simulation_geometry_dependencies() -> None:
    source = (
        Path(__file__).parents[3] / "controller_learning" / "evaluation" / "final_metrics.py"
    ).read_text(encoding="utf-8")
    for forbidden in (
        "controller_learning.tracks",
        "controller_learning.envs",
        "race_core",
        "mjx_warp",
        "CpuVehicle",
    ):
        assert forbidden not in source
