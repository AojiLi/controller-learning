"""Unit tests for MJX-Warp adapter compatibility boundaries."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import jax.numpy as jnp
import numpy as np
import pytest

from controller_learning.config import load_vehicle_config
from controller_learning.physics import mjx_warp
from controller_learning.physics.actuation import wheel_torques_for_acceleration
from controller_learning.physics.model import load_vehicle_model

PROJECT_ROOT = Path(__file__).parents[3]


@pytest.fixture
def vehicle_config():
    return load_vehicle_config(PROJECT_ROOT / "configs" / "vehicle.toml")


def test_create_passes_the_configured_physics_timestep_to_the_shared_model(
    monkeypatch,
    vehicle_config,
) -> None:
    class ExpectedCall(Exception):
        pass

    def stop_after_model_call(config, *, physics_dt_s=None):
        assert config is vehicle_config
        assert physics_dt_s == vehicle_config.simulation.physics_dt_s
        raise ExpectedCall

    fake_device = SimpleNamespace(device_kind="NVIDIA test device")
    monkeypatch.setattr(mjx_warp.jax, "devices", lambda platform: [fake_device])
    monkeypatch.setattr(mjx_warp, "load_vehicle_model", stop_after_model_call)

    with pytest.raises(ExpectedCall):
        mjx_warp.MjxWarpVehicle.create(vehicle_config, num_worlds=1)


def test_create_translates_an_unavailable_gpu_backend_to_domain_error(
    monkeypatch,
    vehicle_config,
) -> None:
    def unavailable_backend(platform):
        raise RuntimeError(f"backend {platform} is unavailable")

    monkeypatch.setattr(mjx_warp.jax, "devices", unavailable_backend)

    with pytest.raises(
        mjx_warp.MjxWarpCompatibilityError,
        match="JAX cannot access an NVIDIA GPU",
    ):
        mjx_warp.MjxWarpVehicle.create(vehicle_config, num_worlds=1)


def test_jax_action_clipping_matches_the_public_physical_contract(vehicle_config) -> None:
    actions = jnp.asarray(
        (
            (10.0, 10.0),
            (-10.0, -10.0),
            (0.1, 1.0),
            (np.nan, np.inf),
        ),
        dtype=jnp.float32,
    )
    steering, acceleration, target, saturation, invalid = mjx_warp._clip_actions(
        vehicle_config,
        actions,
        jnp.zeros(4, dtype=jnp.float32),
    )

    np.testing.assert_allclose(np.asarray(steering), (0.6, -0.6, 0.1, 0.0), atol=1e-6)
    np.testing.assert_allclose(np.asarray(acceleration), (4.0, -8.0, 1.0, 0.0), atol=1e-6)
    np.testing.assert_allclose(np.asarray(target), (0.06, -0.06, 0.06, 0.0), atol=1e-6)
    np.testing.assert_array_equal(np.asarray(saturation), (2, 2, 0, 0))
    np.testing.assert_array_equal(np.asarray(invalid), (False, False, False, True))


@pytest.mark.parametrize(
    ("acceleration", "longitudinal_velocity", "wheel_velocity"),
    [
        (4.0, 0.0, (0.0, 0.0, 0.0, 0.0)),
        (-8.0, 0.0, (0.0, 0.0, 0.0, 0.0)),
        (-8.0, 0.099, (0.499, -0.499, 0.0, 0.0)),
        (-8.0, 0.1, (0.499, -0.499, 0.0, 0.0)),
        (-8.0, -0.1, (0.499, -0.499, 0.0, 0.0)),
        (-8.0, 0.0, (0.5, -0.5, 0.5, -0.5)),
    ],
)
def test_jax_wheel_torque_matches_numpy_at_direction_thresholds(
    vehicle_config,
    acceleration,
    longitudinal_velocity,
    wheel_velocity,
) -> None:
    model, indices = load_vehicle_model(vehicle_config)
    qpos = np.zeros((1, model.nq), dtype=np.float32)
    qpos[:, 3] = 1.0
    qvel = np.zeros((1, model.nv), dtype=np.float32)
    qvel[:, 0] = longitudinal_velocity
    qvel[:, list(indices.wheel_dofs)] = wheel_velocity
    data = SimpleNamespace(qpos=jnp.asarray(qpos), qvel=jnp.asarray(qvel))

    actual = mjx_warp._wheel_torques(
        vehicle_config,
        data,
        jnp.asarray((acceleration,), dtype=jnp.float32),
        jnp.asarray(indices.wheel_dofs, dtype=jnp.int32),
    )
    expected = wheel_torques_for_acceleration(
        vehicle_config,
        acceleration,
        wheel_velocity,
        longitudinal_velocity,
    )

    np.testing.assert_allclose(np.asarray(actual[0]), expected, rtol=0.0, atol=1e-3)
