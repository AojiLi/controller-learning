"""Native leading-dimension MJX-Warp backend for the physical four-wheel vehicle."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, NamedTuple

import jax
import jax.numpy as jnp
import mujoco
from mujoco import mjx

from controller_learning.config import VehicleConfig
from controller_learning.physics.actuation import (
    VEHICLE_STOP_THRESHOLD_M_S,
    WHEEL_STOP_THRESHOLD_RAD_S,
)
from controller_learning.physics.model import VehicleModelIndices, load_vehicle_model

SUPPORTED_MUJOCO_VERSION = "3.10.0"
DEFAULT_CONTACTS_PER_WORLD = 16
DEFAULT_CONSTRAINTS_PER_WORLD = 64


class MjxWarpCompatibilityError(RuntimeError):
    """Raised when the locked MJX-Warp contract is unavailable."""


class MjxWarpShapeError(ValueError):
    """Raised when a batched action or reset mask has the wrong shape."""


class MjxWarpVehicleState(NamedTuple):
    """Dynamic physics state plus control-boundary state owned by the environment."""

    data: Any
    steering_target_rad: jax.Array
    control_step_count: jax.Array
    wheel_no_contact_substeps: jax.Array


class MjxWarpAppliedAction(NamedTuple):
    """Batched clipped action values used by one control step."""

    steering_angle_rad: jax.Array
    longitudinal_acceleration_mps2: jax.Array
    steering_target_rad: jax.Array
    final_wheel_torque_nm: jax.Array
    saturation_count: jax.Array
    invalid_action: jax.Array


class MjxWarpStepDiagnostics(NamedTuple):
    """Physics-substep diagnostics accumulated over one control period."""

    finite: jax.Array
    finite_per_world: jax.Array
    time_monotonic: jax.Array
    time_monotonic_per_world: jax.Array
    peak_nacon: jax.Array
    peak_ncollision: jax.Array
    peak_nefc: jax.Array
    contact_overflow: jax.Array
    constraint_overflow: jax.Array
    unexpected_contact: jax.Array
    maximum_penetration_m: jax.Array
    wheel_ground_contact_fraction: jax.Array
    maximum_wheel_contact_gap_s: jax.Array
    maximum_quaternion_norm_error: jax.Array
    maximum_abs_roll_pitch_rad: jax.Array
    maximum_abs_vertical_speed_mps: jax.Array


class MjxWarpVehicleStateView(NamedTuple):
    """Current public vehicle state derived directly from integrated qpos and qvel."""

    time_s: jax.Array
    position_world_m: jax.Array
    chassis_position_world_m: jax.Array
    quaternion_wxyz: jax.Array
    roll_rad: jax.Array
    pitch_rad: jax.Array
    yaw_rad: jax.Array
    velocity_body_mps: jax.Array
    angular_velocity_body_rad_s: jax.Array
    steering_angle_rad: jax.Array
    front_steering_angles_rad: jax.Array
    wheel_angular_velocity_rad_s: jax.Array


def _rotation_matrix_wxyz(quaternion: jax.Array) -> jax.Array:
    w, x, y, z = (quaternion[..., index] for index in range(4))
    return jnp.stack(
        (
            1.0 - 2.0 * (y * y + z * z),
            2.0 * (x * y - w * z),
            2.0 * (x * z + w * y),
            2.0 * (x * y + w * z),
            1.0 - 2.0 * (x * x + z * z),
            2.0 * (y * z - w * x),
            2.0 * (x * z - w * y),
            2.0 * (y * z + w * x),
            1.0 - 2.0 * (x * x + y * y),
        ),
        axis=-1,
    ).reshape((*quaternion.shape[:-1], 3, 3))


def _rear_axle_kinematics(
    qpos: jax.Array,
    qvel: jax.Array,
    rear_axle_offset_body: jax.Array,
) -> tuple[jax.Array, jax.Array, jax.Array, jax.Array]:
    rotation = _rotation_matrix_wxyz(qpos[..., 3:7])
    rear_position = qpos[..., :3] + jnp.einsum("...ij,j->...i", rotation, rear_axle_offset_body)
    linear_velocity_body = jnp.einsum("...ij,...i->...j", rotation, qvel[..., :3])
    angular_velocity_body = qvel[..., 3:6]
    rear_velocity_body = linear_velocity_body + jnp.cross(
        angular_velocity_body,
        rear_axle_offset_body,
    )
    return rear_position, rear_velocity_body, angular_velocity_body, rotation


def _clip_actions(
    config: VehicleConfig,
    actions: jax.Array,
    previous_steering_target_rad: jax.Array,
) -> tuple[jax.Array, jax.Array, jax.Array, jax.Array, jax.Array]:
    invalid = ~jnp.all(jnp.isfinite(actions), axis=1)
    safe_actions = jnp.where(invalid[:, None], 0.0, actions)
    steering = jnp.clip(
        safe_actions[:, 0],
        -config.actuator.max_steering_angle_rad,
        config.actuator.max_steering_angle_rad,
    )
    acceleration = jnp.clip(
        safe_actions[:, 1],
        -config.actuator.max_deceleration_mps2,
        config.actuator.max_acceleration_mps2,
    )
    maximum_delta = config.actuator.max_steering_rate_rad_s * config.simulation.control_dt_s
    steering_target = jnp.clip(
        steering,
        previous_steering_target_rad - maximum_delta,
        previous_steering_target_rad + maximum_delta,
    )
    saturation_count = (safe_actions[:, 0] != steering).astype(jnp.int32) + (
        safe_actions[:, 1] != acceleration
    ).astype(jnp.int32)
    return steering, acceleration, steering_target, saturation_count, invalid


def _wheel_torques(
    config: VehicleConfig,
    data: Any,
    acceleration_mps2: jax.Array,
    wheel_dofs: jax.Array,
) -> jax.Array:
    wheel_velocity = data.qvel[:, wheel_dofs]
    quaternion = data.qpos[:, 3:7]
    rotation = _rotation_matrix_wxyz(quaternion)
    longitudinal_velocity = jnp.einsum("nij,ni->nj", rotation, data.qvel[:, :3])[:, 0]
    vehicle_direction = jnp.where(
        jnp.abs(longitudinal_velocity) >= VEHICLE_STOP_THRESHOLD_M_S,
        jnp.sign(longitudinal_velocity),
        0.0,
    )
    directions = jnp.where(
        jnp.abs(wheel_velocity) >= WHEEL_STOP_THRESHOLD_RAD_S,
        jnp.sign(wheel_velocity),
        vehicle_direction[:, None],
    )
    magnitude = (
        config.vehicle.mass_kg * jnp.abs(acceleration_mps2) * config.vehicle.wheel_radius_m / 4.0
    )
    return jnp.where(
        acceleration_mps2[:, None] > 0.0,
        magnitude[:, None],
        jnp.where(
            acceleration_mps2[:, None] < 0.0,
            -directions * magnitude[:, None],
            0.0,
        ),
    )


def _contact_diagnostics(
    data: Any,
    *,
    num_worlds: int,
    ground_geom: int,
    wheel_geoms: tuple[int, int, int, int],
) -> tuple[jax.Array, jax.Array, jax.Array]:
    implementation = data._impl
    capacity = implementation.naconmax
    active_count = jnp.minimum(implementation.nacon[0], capacity)
    active = jnp.arange(capacity, dtype=jnp.int32) < active_count
    geoms = implementation.contact__geom
    world_ids = jnp.clip(implementation.contact__worldid, 0, num_worlds - 1)
    allowed = jnp.zeros(capacity, dtype=bool)
    wheel_contact_columns = []
    for wheel_geom in wheel_geoms:
        matches = active & (
            ((geoms[:, 0] == ground_geom) & (geoms[:, 1] == wheel_geom))
            | ((geoms[:, 1] == ground_geom) & (geoms[:, 0] == wheel_geom))
        )
        allowed |= matches
        per_world = (
            jnp.zeros(num_worlds, dtype=jnp.int32).at[world_ids].max(matches.astype(jnp.int32))
        )
        wheel_contact_columns.append(per_world.astype(bool))
    wheel_contact = jnp.stack(wheel_contact_columns, axis=1)
    unexpected = jnp.any(active & ~allowed)
    penetration = jnp.max(jnp.where(active, jnp.maximum(-implementation.contact__dist, 0.0), 0.0))
    return wheel_contact, unexpected, penetration


def _current_pose_extrema(
    data: Any,
    rear_axle_offset_body: jax.Array,
) -> tuple[jax.Array, jax.Array, jax.Array]:
    _, rear_velocity, _, rotation = _rear_axle_kinematics(
        data.qpos,
        data.qvel,
        rear_axle_offset_body,
    )
    roll = jnp.arctan2(rotation[:, 2, 1], rotation[:, 2, 2])
    pitch = jnp.arctan2(
        -rotation[:, 2, 0],
        jnp.hypot(rotation[:, 0, 0], rotation[:, 1, 0]),
    )
    maximum_roll_pitch = jnp.max(jnp.maximum(jnp.abs(roll), jnp.abs(pitch)))
    maximum_vertical_speed = jnp.max(jnp.abs(rear_velocity[:, 2]))
    quaternion_error = jnp.max(jnp.abs(jnp.linalg.norm(data.qpos[:, 3:7], axis=1) - 1.0))
    return maximum_roll_pitch, maximum_vertical_speed, quaternion_error


def _validate_warp_data_contract(
    data: Any,
    *,
    num_worlds: int,
    naconmax: int,
    njmax: int,
) -> int:
    """Validate the locked MuJoCo 3.10 private fields used by device diagnostics."""

    implementation = getattr(data, "_impl", None)
    expected_shapes = {
        "nacon": (1,),
        "ncollision": (1,),
        "nefc": (num_worlds,),
        "contact__geom": (naconmax, 2),
        "contact__worldid": (naconmax,),
        "contact__dist": (naconmax,),
    }
    if implementation is None:
        raise MjxWarpCompatibilityError("MJX-Warp data does not expose the locked _impl contract")
    for field, expected_shape in expected_shapes.items():
        value = getattr(implementation, field, None)
        if value is None or value.shape != expected_shape:
            actual_shape = None if value is None else value.shape
            raise MjxWarpCompatibilityError(
                f"MJX-Warp field _impl.{field} must have shape {expected_shape}, got {actual_shape}"
            )
    if implementation.naconmax != naconmax or implementation.njmax != njmax:
        raise MjxWarpCompatibilityError("MJX-Warp data capacities do not match the request")
    njmax_nnz = getattr(implementation, "njmax_nnz", None)
    if not isinstance(njmax_nnz, int) or njmax_nnz <= 0:
        raise MjxWarpCompatibilityError("MJX-Warp data does not expose a valid njmax_nnz")
    return njmax_nnz


def _control_step(
    config: VehicleConfig,
    model: Any,
    indices: VehicleModelIndices,
    num_worlds: int,
    rear_axle_offset_body: jax.Array,
    state: MjxWarpVehicleState,
    actions: jax.Array,
) -> tuple[MjxWarpVehicleState, MjxWarpAppliedAction, MjxWarpStepDiagnostics]:
    steering, acceleration, steering_target, saturation_count, invalid = _clip_actions(
        config,
        actions,
        state.steering_target_rad,
    )
    steering_actuators = jnp.asarray(indices.steering_actuators, dtype=jnp.int32)
    drive_actuators = jnp.asarray(indices.drive_actuators, dtype=jnp.int32)
    wheel_dofs = jnp.asarray(indices.wheel_dofs, dtype=jnp.int32)
    batched_step = jax.vmap(mjx.step, in_axes=(None, 0))
    contact_samples = jnp.zeros((num_worlds, 4), dtype=jnp.int32)

    carry = (
        state.data,
        state.wheel_no_contact_substeps,
        contact_samples,
        jnp.ones(num_worlds, dtype=bool),
        jnp.ones(num_worlds, dtype=bool),
        jnp.asarray(0, dtype=jnp.int32),
        jnp.asarray(0, dtype=jnp.int32),
        jnp.asarray(0, dtype=jnp.int32),
        jnp.asarray(False),
        jnp.asarray(False),
        jnp.asarray(False),
        jnp.asarray(0.0, dtype=jnp.float32),
        jnp.asarray(0, dtype=jnp.int32),
        jnp.asarray(0.0, dtype=jnp.float32),
        jnp.asarray(0.0, dtype=jnp.float32),
        jnp.asarray(0.0, dtype=jnp.float32),
        jnp.zeros((num_worlds, 4), dtype=jnp.float32),
    )

    def substep(carry_values: tuple[Any, ...], _: None) -> tuple[tuple[Any, ...], None]:
        (
            data,
            no_contact_substeps,
            accumulated_contact,
            finite,
            time_monotonic,
            peak_nacon,
            peak_ncollision,
            peak_nefc,
            contact_overflow,
            constraint_overflow,
            unexpected_contact,
            maximum_penetration,
            maximum_gap_substeps,
            maximum_quaternion_error,
            maximum_roll_pitch,
            maximum_vertical_speed,
            _latest_torque,
        ) = carry_values
        torque = _wheel_torques(config, data, acceleration, wheel_dofs)
        control = data.ctrl.at[:, steering_actuators].set(steering_target[:, None])
        control = control.at[:, drive_actuators].set(torque)
        before_time = data.time
        data = batched_step(model, data.replace(ctrl=control))
        implementation = data._impl
        wheel_contact, unexpected, penetration = _contact_diagnostics(
            data,
            num_worlds=num_worlds,
            ground_geom=indices.ground_geom,
            wheel_geoms=indices.wheel_geoms,
        )
        no_contact_substeps = jnp.where(
            wheel_contact,
            0,
            no_contact_substeps + 1,
        )
        current_roll_pitch, current_vertical_speed, current_quaternion_error = (
            _current_pose_extrema(data, rear_axle_offset_body)
        )
        current_finite = (
            jnp.all(jnp.isfinite(data.qpos), axis=1)
            & jnp.all(jnp.isfinite(data.qvel), axis=1)
            & jnp.all(jnp.isfinite(data.qacc), axis=1)
            & jnp.all(jnp.isfinite(data.ctrl), axis=1)
            & jnp.isfinite(data.time)
        )
        current_nacon = implementation.nacon[0]
        current_ncollision = implementation.ncollision[0]
        current_nefc = jnp.max(implementation.nefc)
        return (
            data,
            no_contact_substeps,
            accumulated_contact + wheel_contact.astype(jnp.int32),
            finite & current_finite,
            time_monotonic & (data.time > before_time),
            jnp.maximum(peak_nacon, current_nacon),
            jnp.maximum(peak_ncollision, current_ncollision),
            jnp.maximum(peak_nefc, current_nefc),
            contact_overflow
            | (current_nacon > implementation.naconmax)
            | (current_ncollision > implementation.naconmax),
            constraint_overflow | (current_nefc > implementation.njmax),
            unexpected_contact | unexpected,
            jnp.maximum(maximum_penetration, penetration),
            jnp.maximum(maximum_gap_substeps, jnp.max(no_contact_substeps)),
            jnp.maximum(maximum_quaternion_error, current_quaternion_error),
            jnp.maximum(maximum_roll_pitch, current_roll_pitch),
            jnp.maximum(maximum_vertical_speed, current_vertical_speed),
            torque,
        ), None

    carry, _ = jax.lax.scan(
        substep,
        carry,
        xs=None,
        length=round(config.simulation.control_dt_s / config.simulation.physics_dt_s),
    )
    (
        data,
        no_contact_substeps,
        accumulated_contact,
        finite,
        time_monotonic,
        peak_nacon,
        peak_ncollision,
        peak_nefc,
        contact_overflow,
        constraint_overflow,
        unexpected_contact,
        maximum_penetration,
        maximum_gap_substeps,
        maximum_quaternion_error,
        maximum_roll_pitch,
        maximum_vertical_speed,
        final_torque,
    ) = carry
    physics_substeps = round(config.simulation.control_dt_s / config.simulation.physics_dt_s)
    next_state = MjxWarpVehicleState(
        data=data,
        steering_target_rad=steering_target,
        control_step_count=state.control_step_count + 1,
        wheel_no_contact_substeps=no_contact_substeps,
    )
    applied = MjxWarpAppliedAction(
        steering_angle_rad=steering,
        longitudinal_acceleration_mps2=acceleration,
        steering_target_rad=steering_target,
        final_wheel_torque_nm=final_torque,
        saturation_count=saturation_count,
        invalid_action=invalid,
    )
    diagnostics = MjxWarpStepDiagnostics(
        finite=jnp.all(finite),
        finite_per_world=finite,
        time_monotonic=jnp.all(time_monotonic),
        time_monotonic_per_world=time_monotonic,
        peak_nacon=peak_nacon,
        peak_ncollision=peak_ncollision,
        peak_nefc=peak_nefc,
        contact_overflow=contact_overflow,
        constraint_overflow=constraint_overflow,
        unexpected_contact=unexpected_contact,
        maximum_penetration_m=maximum_penetration,
        wheel_ground_contact_fraction=accumulated_contact / physics_substeps,
        maximum_wheel_contact_gap_s=(maximum_gap_substeps * config.simulation.physics_dt_s),
        maximum_quaternion_norm_error=maximum_quaternion_error,
        maximum_abs_roll_pitch_rad=maximum_roll_pitch,
        maximum_abs_vertical_speed_mps=maximum_vertical_speed,
    )
    return next_state, applied, diagnostics


@dataclass(slots=True)
class MjxWarpVehicle:
    """Compiled MJX-Warp vehicle system with one native leading batch dimension."""

    config: VehicleConfig
    num_worlds: int
    host_model: mujoco.MjModel
    model: Any
    indices: VehicleModelIndices
    naconmax: int
    njmax: int
    njmax_nnz: int
    device: Any
    source_disableflags: int
    effective_host_disableflags: int
    effective_warp_disableflags: int
    _initial_data: Any
    _step_function: Any

    @classmethod
    def create(
        cls,
        config: VehicleConfig,
        *,
        num_worlds: int,
        contacts_per_world: int = DEFAULT_CONTACTS_PER_WORLD,
        constraints_per_world: int = DEFAULT_CONSTRAINTS_PER_WORLD,
    ) -> MjxWarpVehicle:
        """Load the shared MJCF and create a native MJX-Warp batch on NVIDIA GPU."""

        if mujoco.__version__ != SUPPORTED_MUJOCO_VERSION:
            raise MjxWarpCompatibilityError(
                f"MJX-Warp adapter requires MuJoCo {SUPPORTED_MUJOCO_VERSION}, "
                f"got {mujoco.__version__}"
            )
        if num_worlds <= 0:
            raise ValueError("num_worlds must be positive")
        if contacts_per_world <= 0 or constraints_per_world <= 0:
            raise ValueError("contact and constraint capacities must be positive")
        try:
            devices = jax.devices("gpu")
        except RuntimeError as error:
            raise MjxWarpCompatibilityError("JAX cannot access an NVIDIA GPU") from error
        if not devices:
            raise MjxWarpCompatibilityError("JAX cannot access an NVIDIA GPU")
        device = devices[0]
        if "NVIDIA" not in str(getattr(device, "device_kind", "")).upper():
            raise MjxWarpCompatibilityError(
                f"MJX-Warp adapter requires an NVIDIA GPU, got {device!s}"
            )
        host_model, indices = load_vehicle_model(
            config,
            physics_dt_s=config.simulation.physics_dt_s,
        )
        source_disableflags = int(host_model.opt.disableflags)
        autoreset_bit = int(mujoco.mjtDisableBit.mjDSBL_AUTORESET)
        if not source_disableflags & autoreset_bit:
            raise MjxWarpCompatibilityError(
                "shared MJCF must disable MuJoCo autoreset before Warp normalization"
            )
        host_model.opt.disableflags = source_disableflags & ~autoreset_bit
        effective_host_disableflags = int(host_model.opt.disableflags)
        model = mjx.put_model(host_model, impl="warp", device=device)
        naconmax = contacts_per_world * num_worlds
        njmax = constraints_per_world

        def make_one(_: jax.Array) -> Any:
            return mjx.make_data(
                host_model,
                impl="warp",
                device=device,
                naconmax=naconmax,
                naccdmax=0,
                njmax=njmax,
            )

        make_batch = jax.jit(jax.vmap(make_one))
        initial_data = make_batch(jnp.arange(num_worlds, dtype=jnp.int32))
        jax.block_until_ready(initial_data.qpos)
        njmax_nnz = _validate_warp_data_contract(
            initial_data,
            num_worlds=num_worlds,
            naconmax=naconmax,
            njmax=njmax,
        )
        rear_axle_offset = jnp.asarray(
            host_model.site_pos[indices.rear_axle_site],
            dtype=jnp.float32,
        )

        def step_function(
            state: MjxWarpVehicleState,
            actions: jax.Array,
        ) -> tuple[MjxWarpVehicleState, MjxWarpAppliedAction, MjxWarpStepDiagnostics]:
            return _control_step(
                config,
                model,
                indices,
                num_worlds,
                rear_axle_offset,
                state,
                actions,
            )

        return cls(
            config=config,
            num_worlds=num_worlds,
            host_model=host_model,
            model=model,
            indices=indices,
            naconmax=naconmax,
            njmax=njmax,
            njmax_nnz=njmax_nnz,
            device=device,
            source_disableflags=source_disableflags,
            effective_host_disableflags=effective_host_disableflags,
            effective_warp_disableflags=int(model.opt.disableflags),
            _initial_data=initial_data,
            _step_function=jax.jit(step_function),
        )

    @property
    def physics_substeps_per_control(self) -> int:
        """Return the fixed number of physics steps in one 20 Hz control step."""

        return self.config.simulation.physics_steps_per_control

    def initial_state(self) -> MjxWarpVehicleState:
        """Return an immutable initial batch state with exact integer time counters."""

        return MjxWarpVehicleState(
            data=self._initial_data,
            steering_target_rad=jnp.zeros(self.num_worlds, dtype=jnp.float32),
            control_step_count=jnp.zeros(self.num_worlds, dtype=jnp.int32),
            wheel_no_contact_substeps=jnp.zeros((self.num_worlds, 4), dtype=jnp.int32),
        )

    def step(
        self,
        state: MjxWarpVehicleState,
        actions: Any,
    ) -> tuple[MjxWarpVehicleState, MjxWarpAppliedAction, MjxWarpStepDiagnostics]:
        """Advance every world by one control period using standardized actions."""

        action_array = jnp.asarray(actions, dtype=jnp.float32)
        expected_shape = (self.num_worlds, 2)
        if action_array.shape != expected_shape:
            raise MjxWarpShapeError(
                f"batched actions must have shape {expected_shape}, got {action_array.shape}"
            )
        return self._step_function(state, action_array)

    def lower_step(self, state: MjxWarpVehicleState, actions: Any) -> Any:
        """Lower one control step so benchmarks can separate compilation from execution."""

        action_array = jnp.asarray(actions, dtype=jnp.float32)
        expected_shape = (self.num_worlds, 2)
        if action_array.shape != expected_shape:
            raise MjxWarpShapeError(
                f"batched actions must have shape {expected_shape}, got {action_array.shape}"
            )
        return self._step_function.lower(state, action_array)

    def read_state(self, state: MjxWarpVehicleState) -> MjxWarpVehicleStateView:
        """Read current public state without relying on pre-integration derived fields."""

        rear_offset = jnp.asarray(
            self.host_model.site_pos[self.indices.rear_axle_site],
            dtype=jnp.float32,
        )
        position, velocity, angular_velocity, rotation = _rear_axle_kinematics(
            state.data.qpos,
            state.data.qvel,
            rear_offset,
        )
        roll = jnp.arctan2(rotation[:, 2, 1], rotation[:, 2, 2])
        pitch = jnp.arctan2(
            -rotation[:, 2, 0],
            jnp.hypot(rotation[:, 0, 0], rotation[:, 1, 0]),
        )
        yaw = jnp.arctan2(rotation[:, 1, 0], rotation[:, 0, 0])
        steering = state.data.qpos[:, jnp.asarray(self.indices.steering_qpos)]
        wheel_velocity = state.data.qvel[:, jnp.asarray(self.indices.wheel_dofs)]
        return MjxWarpVehicleStateView(
            time_s=(
                state.control_step_count.astype(jnp.float32) * self.config.simulation.control_dt_s
            ),
            position_world_m=position,
            chassis_position_world_m=state.data.qpos[:, :3],
            quaternion_wxyz=state.data.qpos[:, 3:7],
            roll_rad=roll,
            pitch_rad=pitch,
            yaw_rad=yaw,
            velocity_body_mps=velocity,
            angular_velocity_body_rad_s=angular_velocity,
            steering_angle_rad=jnp.mean(steering, axis=1),
            front_steering_angles_rad=steering,
            wheel_angular_velocity_rad_s=wheel_velocity,
        )

    def masked_reset(
        self,
        state: MjxWarpVehicleState,
        reset_mask: Any,
    ) -> MjxWarpVehicleState:
        """Reset selected worlds while preserving every unmasked public dynamic field."""

        mask = jnp.asarray(reset_mask, dtype=bool)
        if mask.shape != (self.num_worlds,):
            raise MjxWarpShapeError(
                f"reset mask must have shape ({self.num_worlds},), got {mask.shape}"
            )
        initial = self._initial_data

        def choose(current: jax.Array, reset: jax.Array) -> jax.Array:
            expanded_mask = mask.reshape((self.num_worlds,) + (1,) * (current.ndim - 1))
            return jnp.where(expanded_mask, reset, current)

        data = state.data.replace(
            time=choose(state.data.time, initial.time),
            qpos=choose(state.data.qpos, initial.qpos),
            qvel=choose(state.data.qvel, initial.qvel),
            act=choose(state.data.act, initial.act),
            qacc_warmstart=choose(
                state.data.qacc_warmstart,
                initial.qacc_warmstart,
            ),
            ctrl=choose(state.data.ctrl, initial.ctrl),
            qfrc_applied=choose(state.data.qfrc_applied, initial.qfrc_applied),
            xfrc_applied=choose(state.data.xfrc_applied, initial.xfrc_applied),
            qacc=choose(state.data.qacc, initial.qacc),
        )
        return MjxWarpVehicleState(
            data=data,
            steering_target_rad=jnp.where(mask, 0.0, state.steering_target_rad),
            control_step_count=jnp.where(mask, 0, state.control_step_count),
            wheel_no_contact_substeps=jnp.where(
                mask[:, None],
                0,
                state.wheel_no_contact_substeps,
            ),
        )


__all__ = [
    "DEFAULT_CONSTRAINTS_PER_WORLD",
    "DEFAULT_CONTACTS_PER_WORLD",
    "MjxWarpAppliedAction",
    "MjxWarpCompatibilityError",
    "MjxWarpShapeError",
    "MjxWarpStepDiagnostics",
    "MjxWarpVehicle",
    "MjxWarpVehicleState",
    "MjxWarpVehicleStateView",
]
