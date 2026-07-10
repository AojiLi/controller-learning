"""Deterministic CPU MuJoCo reference for the physical four-wheel vehicle."""

from __future__ import annotations

from dataclasses import dataclass, replace
from math import atan2, hypot, isclose

import mujoco
import numpy as np
from numpy.typing import NDArray

from controller_learning.config import VehicleConfig
from controller_learning.physics.actuation import (
    AppliedVehicleAction,
    VehicleActionError,
    map_vehicle_action,
    wheel_torques_for_acceleration,
)
from controller_learning.physics.model import load_vehicle_model


class VehicleSimulationError(RuntimeError):
    """Raised when the CPU vehicle produces invalid state or MuJoCo warnings."""


@dataclass(frozen=True, slots=True)
class VehicleState:
    """Physical vehicle state expressed at the rear-axle reference site."""

    time_s: float
    position_world_m: tuple[float, float, float]
    chassis_position_world_m: tuple[float, float, float]
    quaternion_wxyz: tuple[float, float, float, float]
    roll_rad: float
    pitch_rad: float
    yaw_rad: float
    velocity_body_mps: tuple[float, float, float]
    angular_velocity_body_rad_s: tuple[float, float, float]
    steering_angle_rad: float
    front_steering_angles_rad: tuple[float, float]
    wheel_angular_velocity_rad_s: tuple[float, float, float, float]

    @property
    def longitudinal_velocity_mps(self) -> float:
        """Return rear-axle forward velocity in the vehicle frame."""

        return self.velocity_body_mps[0]

    @property
    def lateral_velocity_mps(self) -> float:
        """Return rear-axle leftward velocity in the vehicle frame."""

        return self.velocity_body_mps[1]

    @property
    def yaw_rate_rad_s(self) -> float:
        """Return body-local angular velocity around +z."""

        return self.angular_velocity_body_rad_s[2]


@dataclass(frozen=True, slots=True)
class ContactMetrics:
    """Current contact diagnostics grouped by physical wheel."""

    wheel_ground_contact_count: tuple[int, int, int, int]
    wheel_normal_force_n: tuple[float, float, float, float]
    unexpected_contact_count: int
    maximum_penetration_m: float
    total_contact_count: int


@dataclass(frozen=True, slots=True)
class StepDiagnostics:
    """Physics-substep diagnostics accumulated over the latest control period."""

    physics_step_count: int
    maximum_penetration_m: float
    wheel_ground_contact_fraction: tuple[float, float, float, float]
    mean_wheel_normal_force_n: tuple[float, float, float, float]
    maximum_unexpected_contact_count: int
    maximum_wheel_contact_gap_s: float
    maximum_abs_roll_pitch_rad: float
    maximum_abs_vertical_speed_mps: float


def _tuple(values: NDArray[np.floating]) -> tuple[float, ...]:
    return tuple(float(value) for value in values)


def _warning_count(data: mujoco.MjData) -> int:
    return sum(int(warning.number) for warning in data.warning)


class CpuVehicle:
    """Advance the packaged vehicle by one 20 Hz Controller period at a time."""

    def __init__(self, config: VehicleConfig, *, physics_dt_s: float | None = None) -> None:
        self.config = config
        self.model, self.indices = load_vehicle_model(config, physics_dt_s=physics_dt_s)
        self.data = mujoco.MjData(self.model)
        self.physics_steps_per_control = round(
            config.simulation.control_dt_s / self.model.opt.timestep
        )
        self._steering_target_rad = 0.0
        self._last_applied_action = AppliedVehicleAction(
            steering_angle_rad=0.0,
            longitudinal_acceleration_mps2=0.0,
            steering_target_rad=0.0,
            wheel_torque_nm=(0.0, 0.0, 0.0, 0.0),
            steering_saturated=False,
            longitudinal_saturated=False,
        )
        self._last_step_diagnostics = StepDiagnostics(
            physics_step_count=0,
            maximum_penetration_m=0.0,
            wheel_ground_contact_fraction=(0.0, 0.0, 0.0, 0.0),
            mean_wheel_normal_force_n=(0.0, 0.0, 0.0, 0.0),
            maximum_unexpected_contact_count=0,
            maximum_wheel_contact_gap_s=0.0,
            maximum_abs_roll_pitch_rad=0.0,
            maximum_abs_vertical_speed_mps=0.0,
        )
        self.reset()

    @property
    def last_applied_action(self) -> AppliedVehicleAction:
        """Return the action and actuator targets used by the latest control step."""

        return self._last_applied_action

    @property
    def last_step_diagnostics(self) -> StepDiagnostics:
        """Return substep extrema and averages for the latest control step."""

        return self._last_step_diagnostics

    @property
    def warning_count(self) -> int:
        """Return the cumulative number of MuJoCo simulation warnings."""

        return _warning_count(self.data)

    def reset(
        self,
        *,
        rear_axle_pose: tuple[float, float, float] | NDArray[np.floating] | None = None,
    ) -> VehicleState:
        """Restore a deterministic rear-axle pose without a hidden settling warmup.

        ``rear_axle_pose`` is ``(x_m, y_m, yaw_rad)``.  Omitting it preserves the
        original MJCF pose used by the M1/M2 reference evidence.
        """

        mujoco.mj_resetData(self.model, self.data)
        if rear_axle_pose is not None:
            pose = np.asarray(rear_axle_pose, dtype=np.float64)
            if pose.shape != (3,) or not np.isfinite(pose).all():
                raise ValueError("rear_axle_pose must be a finite shape-(3,) value")
            yaw = float(pose[2])
            cosine = float(np.cos(yaw))
            sine = float(np.sin(yaw))
            rear_offset = self.model.site_pos[self.indices.rear_axle_site]
            rotated_rear_xy = np.asarray(
                (
                    cosine * rear_offset[0] - sine * rear_offset[1],
                    sine * rear_offset[0] + cosine * rear_offset[1],
                )
            )
            self.data.qpos[:2] = pose[:2] - rotated_rear_xy
            self.data.qpos[3:7] = (
                np.cos(0.5 * yaw),
                0.0,
                0.0,
                np.sin(0.5 * yaw),
            )
        self.data.ctrl.fill(0.0)
        self._steering_target_rad = 0.0
        self._last_applied_action = replace(
            self._last_applied_action,
            steering_angle_rad=0.0,
            longitudinal_acceleration_mps2=0.0,
            steering_target_rad=0.0,
            wheel_torque_nm=(0.0, 0.0, 0.0, 0.0),
            steering_saturated=False,
            longitudinal_saturated=False,
        )
        self._wheel_no_contact_steps = np.zeros(4, dtype=np.int64)
        mujoco.mj_forward(self.model, self.data)
        state = self.state()
        contact = self.contact_metrics()
        self._last_step_diagnostics = StepDiagnostics(
            physics_step_count=0,
            maximum_penetration_m=contact.maximum_penetration_m,
            wheel_ground_contact_fraction=tuple(
                float(count > 0) for count in contact.wheel_ground_contact_count
            ),
            mean_wheel_normal_force_n=contact.wheel_normal_force_n,
            maximum_unexpected_contact_count=contact.unexpected_contact_count,
            maximum_wheel_contact_gap_s=0.0,
            maximum_abs_roll_pitch_rad=max(abs(state.roll_rad), abs(state.pitch_rad)),
            maximum_abs_vertical_speed_mps=abs(state.velocity_body_mps[2]),
        )
        return state

    def _current_chassis_rotation(self) -> NDArray[np.float64]:
        """Return the current body-to-world rotation directly from free-joint qpos."""

        rotation = np.empty(9, dtype=np.float64)
        mujoco.mju_quat2Mat(rotation, self.data.qpos[3:7])
        return rotation.reshape(3, 3)

    def state(self) -> VehicleState:
        """Read state using the project +x-forward, +y-left, yaw-CCW convention."""

        chassis_rotation = self._current_chassis_rotation()
        roll = atan2(chassis_rotation[2, 1], chassis_rotation[2, 2])
        pitch = atan2(
            -chassis_rotation[2, 0],
            hypot(chassis_rotation[0, 0], chassis_rotation[1, 0]),
        )
        yaw = atan2(chassis_rotation[1, 0], chassis_rotation[0, 0])
        rear_axle_offset_body = self.model.site_pos[self.indices.rear_axle_site]
        rear_axle_position_world = self.data.qpos[:3] + chassis_rotation @ rear_axle_offset_body
        angular_velocity_body = self.data.qvel[3:6]
        rear_axle_velocity_body = chassis_rotation.T @ self.data.qvel[:3] + np.cross(
            angular_velocity_body, rear_axle_offset_body
        )
        steering = self.data.qpos[list(self.indices.steering_qpos)]
        wheel_velocity = self.data.qvel[list(self.indices.wheel_dofs)]
        return VehicleState(
            time_s=float(self.data.time),
            position_world_m=_tuple(rear_axle_position_world),
            chassis_position_world_m=_tuple(self.data.qpos[:3]),
            quaternion_wxyz=_tuple(self.data.qpos[3:7]),
            roll_rad=roll,
            pitch_rad=pitch,
            yaw_rad=yaw,
            velocity_body_mps=_tuple(rear_axle_velocity_body),
            angular_velocity_body_rad_s=_tuple(angular_velocity_body),
            steering_angle_rad=float(np.mean(steering)),
            front_steering_angles_rad=(float(steering[0]), float(steering[1])),
            wheel_angular_velocity_rad_s=(
                float(wheel_velocity[0]),
                float(wheel_velocity[1]),
                float(wheel_velocity[2]),
                float(wheel_velocity[3]),
            ),
        )

    def contact_metrics(self) -> ContactMetrics:
        """Report allowed wheel-ground contacts and any unexpected collision pair."""

        wheel_by_geom = {
            geom_id: wheel_index for wheel_index, geom_id in enumerate(self.indices.wheel_geoms)
        }
        wheel_counts = [0, 0, 0, 0]
        wheel_normal_force = [0.0, 0.0, 0.0, 0.0]
        unexpected = 0
        maximum_penetration = 0.0
        for contact_index in range(self.data.ncon):
            contact = self.data.contact[contact_index]
            geom_a, geom_b = int(contact.geom[0]), int(contact.geom[1])
            maximum_penetration = max(maximum_penetration, max(0.0, -float(contact.dist)))
            wheel_geom = None
            if geom_a == self.indices.ground_geom and geom_b in wheel_by_geom:
                wheel_geom = geom_b
            elif geom_b == self.indices.ground_geom and geom_a in wheel_by_geom:
                wheel_geom = geom_a
            if wheel_geom is None:
                unexpected += 1
            else:
                wheel_index = wheel_by_geom[wheel_geom]
                wheel_counts[wheel_index] += 1
                contact_force = np.zeros(6, dtype=np.float64)
                mujoco.mj_contactForce(
                    self.model,
                    self.data,
                    contact_index,
                    contact_force,
                )
                wheel_normal_force[wheel_index] += max(0.0, float(contact_force[0]))
        return ContactMetrics(
            wheel_ground_contact_count=(
                wheel_counts[0],
                wheel_counts[1],
                wheel_counts[2],
                wheel_counts[3],
            ),
            wheel_normal_force_n=(
                wheel_normal_force[0],
                wheel_normal_force[1],
                wheel_normal_force[2],
                wheel_normal_force[3],
            ),
            unexpected_contact_count=unexpected,
            maximum_penetration_m=maximum_penetration,
            total_contact_count=int(self.data.ncon),
        )

    def step(self, action: tuple[float, float] | NDArray[np.floating]) -> VehicleState:
        """Advance exactly one Controller period using the standardized actuator mapping."""

        before = self.state()
        applied = map_vehicle_action(
            self.config,
            action,
            previous_steering_target_rad=self._steering_target_rad,
            wheel_angular_velocity_rad_s=before.wheel_angular_velocity_rad_s,
            longitudinal_velocity_mps=before.longitudinal_velocity_mps,
        )
        self._steering_target_rad = applied.steering_target_rad
        self.data.ctrl[list(self.indices.steering_actuators)] = applied.steering_target_rad
        warnings_before = self.warning_count
        expected_time = self.data.time + self.config.simulation.control_dt_s
        latest_torque = applied.wheel_torque_nm
        wheel_contact_samples = np.zeros(4, dtype=np.float64)
        wheel_normal_force_sum = np.zeros(4, dtype=np.float64)
        maximum_penetration = 0.0
        maximum_unexpected_contact_count = 0
        maximum_wheel_contact_gap_steps = 0
        maximum_abs_roll_pitch = 0.0
        maximum_abs_vertical_speed = 0.0
        for _ in range(self.physics_steps_per_control):
            current = self.state()
            latest_torque = wheel_torques_for_acceleration(
                self.config,
                applied.longitudinal_acceleration_mps2,
                current.wheel_angular_velocity_rad_s,
                current.longitudinal_velocity_mps,
            )
            self.data.ctrl[list(self.indices.drive_actuators)] = latest_torque
            mujoco.mj_step(self.model, self.data)
            if not (
                np.isfinite(self.data.qpos).all()
                and np.isfinite(self.data.qvel).all()
                and np.isfinite(self.data.qacc).all()
                and np.isfinite(self.data.ctrl).all()
            ):
                raise VehicleSimulationError("MuJoCo produced non-finite vehicle state")
            substep_state = self.state()
            substep_contact = self.contact_metrics()
            wheel_contact_samples += np.asarray(
                [count > 0 for count in substep_contact.wheel_ground_contact_count],
                dtype=np.float64,
            )
            in_contact = np.asarray(
                [count > 0 for count in substep_contact.wheel_ground_contact_count],
                dtype=bool,
            )
            self._wheel_no_contact_steps = np.where(
                in_contact,
                0,
                self._wheel_no_contact_steps + 1,
            )
            maximum_wheel_contact_gap_steps = max(
                maximum_wheel_contact_gap_steps,
                int(np.max(self._wheel_no_contact_steps)),
            )
            wheel_normal_force_sum += np.asarray(
                substep_contact.wheel_normal_force_n,
                dtype=np.float64,
            )
            maximum_penetration = max(
                maximum_penetration,
                substep_contact.maximum_penetration_m,
            )
            maximum_unexpected_contact_count = max(
                maximum_unexpected_contact_count,
                substep_contact.unexpected_contact_count,
            )
            maximum_abs_roll_pitch = max(
                maximum_abs_roll_pitch,
                abs(substep_state.roll_rad),
                abs(substep_state.pitch_rad),
            )
            maximum_abs_vertical_speed = max(
                maximum_abs_vertical_speed,
                abs(substep_state.velocity_body_mps[2]),
            )
        if self.warning_count != warnings_before:
            raise VehicleSimulationError("MuJoCo reported a simulation warning")
        if not isclose(self.data.time, expected_time, rel_tol=0.0, abs_tol=1e-10):
            raise VehicleSimulationError(
                "MuJoCo simulation time did not advance by one control period"
            )
        self._last_applied_action = replace(applied, wheel_torque_nm=latest_torque)
        self._last_step_diagnostics = StepDiagnostics(
            physics_step_count=self.physics_steps_per_control,
            maximum_penetration_m=maximum_penetration,
            wheel_ground_contact_fraction=tuple(
                float(value / self.physics_steps_per_control) for value in wheel_contact_samples
            ),
            mean_wheel_normal_force_n=tuple(
                float(value / self.physics_steps_per_control) for value in wheel_normal_force_sum
            ),
            maximum_unexpected_contact_count=maximum_unexpected_contact_count,
            maximum_wheel_contact_gap_s=(maximum_wheel_contact_gap_steps * self.model.opt.timestep),
            maximum_abs_roll_pitch_rad=maximum_abs_roll_pitch,
            maximum_abs_vertical_speed_mps=maximum_abs_vertical_speed,
        )
        return self.state()


__all__ = [
    "ContactMetrics",
    "CpuVehicle",
    "StepDiagnostics",
    "VehicleActionError",
    "VehicleSimulationError",
    "VehicleState",
]
