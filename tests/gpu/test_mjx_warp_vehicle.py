"""GPU integration tests for the native-batch MJX-Warp vehicle adapter."""

from __future__ import annotations

from pathlib import Path

import jax
import jax.numpy as jnp
import mujoco
import numpy as np
import pytest

from controller_learning.config import load_vehicle_config
from controller_learning.physics import CpuVehicle
from controller_learning.physics.mjx_warp import MjxWarpShapeError, MjxWarpVehicle

PROJECT_ROOT = Path(__file__).parents[2]
pytestmark = pytest.mark.gpu


@pytest.fixture(scope="module")
def vehicle_config():
    return load_vehicle_config(PROJECT_ROOT / "configs" / "vehicle.toml")


@pytest.fixture(scope="module")
def one_world_vehicle(vehicle_config):
    return MjxWarpVehicle.create(vehicle_config, num_worlds=1)


@pytest.fixture(scope="module")
def four_world_vehicle(vehicle_config):
    return MjxWarpVehicle.create(vehicle_config, num_worlds=4)


def test_shared_mjcf_is_normalized_only_for_the_warp_conversion(one_world_vehicle) -> None:
    autoreset = int(mujoco.mjtDisableBit.mjDSBL_AUTORESET)

    assert one_world_vehicle.source_disableflags & autoreset
    assert not one_world_vehicle.effective_host_disableflags & autoreset
    assert not one_world_vehicle.effective_warp_disableflags & autoreset
    assert (
        one_world_vehicle.source_disableflags ^ one_world_vehicle.effective_host_disableflags
    ) == autoreset
    assert one_world_vehicle.host_model.opt.timestep == pytest.approx(
        one_world_vehicle.config.simulation.physics_dt_s
    )
    assert one_world_vehicle.physics_substeps_per_control == 10
    assert one_world_vehicle.naconmax == 16
    assert one_world_vehicle.njmax == 64
    assert one_world_vehicle.njmax_nnz > 0


def test_one_world_step_has_current_state_and_device_diagnostics(one_world_vehicle) -> None:
    state = one_world_vehicle.initial_state()
    initial = jax.device_get(one_world_vehicle.read_state(state))

    state, applied, diagnostics = one_world_vehicle.step(
        state,
        jnp.zeros((1, 2), dtype=jnp.float32),
    )
    jax.block_until_ready(state.data.qpos)
    current = jax.device_get(one_world_vehicle.read_state(state))

    assert state.data.qpos.shape == (1, 13)
    assert state.data.qvel.shape == (1, 12)
    assert state.data.ctrl.shape == (1, 6)
    assert initial.position_world_m[0] == pytest.approx((-1.35, 0.0, 0.34), abs=1e-6)
    assert initial.chassis_position_world_m[0] == pytest.approx((0.0, 0.0, 0.56), abs=1e-6)
    assert initial.quaternion_wxyz[0] == pytest.approx((1.0, 0.0, 0.0, 0.0), abs=1e-7)
    assert initial.velocity_body_mps[0] == pytest.approx((0.0, 0.0, 0.0), abs=1e-7)
    assert current.time_s[0] == pytest.approx(0.05, abs=1e-7)

    for _ in range(19):
        state, applied, diagnostics = one_world_vehicle.step(
            state,
            jnp.zeros((1, 2), dtype=jnp.float32),
        )
    jax.block_until_ready(state.data.qpos)
    diagnostics = jax.device_get(diagnostics)

    assert bool(diagnostics.finite)
    assert np.all(np.asarray(diagnostics.finite_per_world))
    assert bool(diagnostics.time_monotonic)
    assert np.all(np.asarray(diagnostics.time_monotonic_per_world))
    assert not bool(diagnostics.contact_overflow)
    assert not bool(diagnostics.constraint_overflow)
    assert not bool(diagnostics.unexpected_contact)
    assert 0 < int(diagnostics.peak_nacon) < one_world_vehicle.naconmax
    assert 0 < int(diagnostics.peak_ncollision) < one_world_vehicle.naconmax
    assert 0 < int(diagnostics.peak_nefc) < one_world_vehicle.njmax
    assert float(diagnostics.maximum_penetration_m) < 0.005
    assert np.all(np.asarray(diagnostics.wheel_ground_contact_fraction) >= 0.8)
    assert float(diagnostics.maximum_wheel_contact_gap_s) <= 0.05
    assert float(diagnostics.maximum_quaternion_norm_error) < 1e-5
    assert np.asarray(applied.final_wheel_torque_nm) == pytest.approx(np.zeros((1, 4)))


def test_action_clipping_rate_limit_and_torque_match_the_cpu_contract(
    one_world_vehicle,
) -> None:
    state, applied, diagnostics = one_world_vehicle.step(
        one_world_vehicle.initial_state(),
        ((10.0, 10.0),),
    )
    jax.block_until_ready(state.data.qpos)
    applied = jax.device_get(applied)

    assert applied.steering_angle_rad[0] == pytest.approx(0.6, abs=1e-6)
    assert applied.longitudinal_acceleration_mps2[0] == pytest.approx(4.0)
    assert applied.steering_target_rad[0] == pytest.approx(0.06, abs=1e-6)
    assert applied.final_wheel_torque_nm[0] == pytest.approx((408.0,) * 4, abs=1e-3)
    assert int(applied.saturation_count[0]) == 2
    assert not bool(applied.invalid_action[0])
    assert bool(diagnostics.finite)
    assert np.asarray(diagnostics.finite_per_world).shape == (1,)
    assert np.all(np.asarray(diagnostics.finite_per_world))


def test_native_batch_keeps_worlds_independent(four_world_vehicle) -> None:
    state = four_world_vehicle.initial_state()
    actions = jnp.asarray(
        ((0.0, 0.0), (0.0, 1.0), (0.0, 2.0), (0.0, 4.0)),
        dtype=jnp.float32,
    )

    for _ in range(40):
        state, _, diagnostics = four_world_vehicle.step(state, actions)
    jax.block_until_ready(state.data.qpos)
    current = jax.device_get(four_world_vehicle.read_state(state))
    diagnostics = jax.device_get(diagnostics)

    assert np.all(np.diff(current.position_world_m[:, 0]) > 1.0)
    assert np.all(np.diff(current.velocity_body_mps[:, 0]) > 1.0)
    assert bool(diagnostics.finite)
    assert np.asarray(diagnostics.finite_per_world).shape == (4,)
    assert np.all(np.asarray(diagnostics.finite_per_world))
    assert not bool(diagnostics.contact_overflow)
    assert not bool(diagnostics.constraint_overflow)
    assert not bool(diagnostics.unexpected_contact)


def test_masked_reset_changes_only_selected_worlds(four_world_vehicle) -> None:
    state = four_world_vehicle.initial_state()
    actions = jnp.asarray(
        ((0.0, 0.5), (0.1, 1.0), (-0.1, 2.0), (0.2, 3.0)),
        dtype=jnp.float32,
    )
    for _ in range(10):
        state, _, _ = four_world_vehicle.step(state, actions)
    jax.block_until_ready(state.data.qpos)
    before = np.asarray(state.data.qpos)

    mask = np.asarray((False, True, False, True))
    reset = four_world_vehicle.masked_reset(state, mask)
    after = np.asarray(reset.data.qpos)
    initial = np.asarray(four_world_vehicle.initial_state().data.qpos)

    np.testing.assert_array_equal(after[~mask], before[~mask])
    np.testing.assert_array_equal(after[mask], initial[mask])
    np.testing.assert_array_equal(
        np.asarray(reset.control_step_count),
        np.asarray((10, 0, 10, 0)),
    )

    baseline_next, _, _ = four_world_vehicle.step(
        state,
        jnp.zeros((4, 2), dtype=jnp.float32),
    )
    fresh_next, _, _ = four_world_vehicle.step(
        four_world_vehicle.initial_state(),
        jnp.zeros((4, 2), dtype=jnp.float32),
    )
    reset_next, _, diagnostics = four_world_vehicle.step(
        reset,
        jnp.zeros((4, 2), dtype=jnp.float32),
    )
    jax.block_until_ready(reset_next.data.qpos)
    assert bool(diagnostics.finite)
    np.testing.assert_array_equal(
        np.asarray(reset_next.control_step_count),
        np.asarray((11, 1, 11, 1)),
    )
    field_tolerances = {"qpos": 1e-6, "qvel": 1e-4, "qacc": 1e-3}
    for field, tolerance in field_tolerances.items():
        actual = np.asarray(getattr(reset_next.data, field))
        baseline = np.asarray(getattr(baseline_next.data, field))
        fresh = np.asarray(getattr(fresh_next.data, field))
        np.testing.assert_allclose(
            actual[~mask],
            baseline[~mask],
            rtol=1e-6,
            atol=tolerance,
        )
        np.testing.assert_allclose(
            actual[mask],
            fresh[mask],
            rtol=1e-6,
            atol=tolerance,
        )


@pytest.mark.parametrize("actions", [(0.0, 0.0), np.zeros((1, 3)), np.zeros((2, 2))])
def test_action_batch_shape_is_enforced(one_world_vehicle, actions) -> None:
    with pytest.raises(MjxWarpShapeError, match="batched actions must have shape"):
        one_world_vehicle.step(one_world_vehicle.initial_state(), actions)


def test_reset_mask_shape_is_enforced(one_world_vehicle) -> None:
    with pytest.raises(MjxWarpShapeError, match="reset mask must have shape"):
        one_world_vehicle.masked_reset(one_world_vehicle.initial_state(), (True, False))


def _scheduled_action(step: int) -> tuple[float, float]:
    if step < 10:
        return (0.0, 0.0)
    if step < 30:
        return (0.0, 1.5)
    if step < 50:
        return (0.15, 0.5)
    if step < 60:
        return (0.0, 0.0)
    if step < 80:
        return (0.0, -3.0)
    return (0.0, 0.0)


def test_batch_one_matches_cpu_reference_over_fixed_rollout(
    vehicle_config,
    one_world_vehicle,
) -> None:
    cpu = CpuVehicle(vehicle_config)
    gpu_state = one_world_vehicle.initial_state()
    cpu_states = [cpu.state()]
    gpu_states = [jax.device_get(one_world_vehicle.read_state(gpu_state))]

    for step in range(100):
        action = _scheduled_action(step)
        cpu_states.append(cpu.step(action))
        gpu_state, _, diagnostics = one_world_vehicle.step(gpu_state, (action,))
        jax.block_until_ready(gpu_state.data.qpos)
        assert bool(diagnostics.finite)
        assert not bool(diagnostics.contact_overflow)
        assert not bool(diagnostics.constraint_overflow)
        gpu_states.append(jax.device_get(one_world_vehicle.read_state(gpu_state)))

    def cpu_array(attribute: str) -> np.ndarray:
        return np.asarray([getattr(state, attribute) for state in cpu_states])

    def gpu_array(attribute: str) -> np.ndarray:
        return np.asarray([getattr(state, attribute)[0] for state in gpu_states])

    position_error = gpu_array("position_world_m") - cpu_array("position_world_m")
    velocity_error = gpu_array("velocity_body_mps") - cpu_array("velocity_body_mps")
    angular_error = gpu_array("angular_velocity_body_rad_s") - cpu_array(
        "angular_velocity_body_rad_s"
    )
    front_steering_error = gpu_array("front_steering_angles_rad") - cpu_array(
        "front_steering_angles_rad"
    )
    wheel_error = gpu_array("wheel_angular_velocity_rad_s") - cpu_array(
        "wheel_angular_velocity_rad_s"
    )
    yaw_error = np.arctan2(
        np.sin(gpu_array("yaw_rad") - cpu_array("yaw_rad")),
        np.cos(gpu_array("yaw_rad") - cpu_array("yaw_rad")),
    )
    cpu_quaternion = cpu_array("quaternion_wxyz").astype(np.float64)
    gpu_quaternion = gpu_array("quaternion_wxyz").astype(np.float64)
    cpu_quaternion /= np.linalg.norm(cpu_quaternion, axis=1, keepdims=True)
    gpu_quaternion /= np.linalg.norm(gpu_quaternion, axis=1, keepdims=True)
    quaternion_dot = np.abs(np.sum(cpu_quaternion * gpu_quaternion, axis=1))
    attitude_error = 2.0 * np.arccos(np.clip(quaternion_dot, 0.0, 1.0))

    assert np.sqrt(np.mean(np.square(position_error[:, :2]))) <= 5e-4
    assert np.max(np.abs(position_error[:, :2])) <= 2e-3
    assert np.max(np.abs(position_error[:, 2])) <= 5e-4
    assert np.sqrt(np.mean(np.square(attitude_error))) <= 5e-5
    assert np.max(np.abs(attitude_error)) <= 2e-4
    assert np.max(np.abs(yaw_error)) <= 2e-4
    assert np.sqrt(np.mean(np.square(velocity_error))) <= 5e-4
    assert np.max(np.abs(velocity_error)) <= 2e-3
    assert np.sqrt(np.mean(np.square(angular_error))) <= 5e-4
    assert np.max(np.abs(angular_error)) <= 2e-3
    assert np.max(np.abs(front_steering_error)) <= 2e-4
    assert np.max(np.abs(wheel_error)) <= 5e-3
    assert gpu_array("time_s")[-1] == pytest.approx(5.0, abs=1e-6)
