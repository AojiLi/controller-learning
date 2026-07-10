"""Narrow private adapters for the two approved four-wheel execution paths."""

from __future__ import annotations

from typing import Any, Literal, NamedTuple, Protocol

import numpy as np

from controller_learning.config import VehicleConfig
from controller_learning.physics import CpuVehicle
from controller_learning.physics.mjx_warp import MjxWarpVehicle

VehicleBackend = Literal["cpu_reference", "mjx_warp"]


class VehicleDriverError(RuntimeError):
    """Raised when a private environment vehicle driver cannot honor its contract."""


class VehicleDriverShapeError(ValueError):
    """Raised when a batch passed to a vehicle driver has the wrong shape."""


class AppliedActionBatch(NamedTuple):
    """Backend-neutral public action result arrays."""

    steering_angle_rad: Any
    longitudinal_acceleration_mps2: Any
    steering_target_rad: Any
    saturation_count: Any
    invalid_action: Any


class CpuDriverState(NamedTuple):
    """Explicit time ownership for the mutable batch-one CPU reference."""

    control_step_count: np.ndarray


class CpuVehicleStateView(NamedTuple):
    """Batch-one array view matching the MJX-Warp public state field names."""

    time_s: np.ndarray
    position_world_m: np.ndarray
    chassis_position_world_m: np.ndarray
    quaternion_wxyz: np.ndarray
    roll_rad: np.ndarray
    pitch_rad: np.ndarray
    yaw_rad: np.ndarray
    velocity_body_mps: np.ndarray
    angular_velocity_body_rad_s: np.ndarray
    steering_angle_rad: np.ndarray
    front_steering_angles_rad: np.ndarray
    wheel_angular_velocity_rad_s: np.ndarray


class VehicleStep(NamedTuple):
    """One private physics transition used by the shared Challenge wrapper."""

    state: Any
    applied: AppliedActionBatch
    diagnostics: Any


class _VehicleDriver(Protocol):
    """The intentionally tiny, package-private M4 physics boundary."""

    backend: VehicleBackend
    num_worlds: int

    def initial_state(self, rear_axle_pose: Any) -> Any: ...

    def read_state(self, state: Any) -> Any: ...

    def step(self, state: Any, actions: Any) -> VehicleStep: ...

    def masked_reset(self, state: Any, mask: Any, rear_axle_pose: Any) -> Any: ...

    def close(self) -> None: ...


def _cpu_view(vehicle: CpuVehicle) -> CpuVehicleStateView:
    state = vehicle.state()
    return CpuVehicleStateView(
        time_s=np.asarray((state.time_s,), dtype=np.float32),
        position_world_m=np.asarray((state.position_world_m,), dtype=np.float32),
        chassis_position_world_m=np.asarray(
            (state.chassis_position_world_m,),
            dtype=np.float32,
        ),
        quaternion_wxyz=np.asarray((state.quaternion_wxyz,), dtype=np.float32),
        roll_rad=np.asarray((state.roll_rad,), dtype=np.float32),
        pitch_rad=np.asarray((state.pitch_rad,), dtype=np.float32),
        yaw_rad=np.asarray((state.yaw_rad,), dtype=np.float32),
        velocity_body_mps=np.asarray((state.velocity_body_mps,), dtype=np.float32),
        angular_velocity_body_rad_s=np.asarray(
            (state.angular_velocity_body_rad_s,),
            dtype=np.float32,
        ),
        steering_angle_rad=np.asarray((state.steering_angle_rad,), dtype=np.float32),
        front_steering_angles_rad=np.asarray(
            (state.front_steering_angles_rad,),
            dtype=np.float32,
        ),
        wheel_angular_velocity_rad_s=np.asarray(
            (state.wheel_angular_velocity_rad_s,),
            dtype=np.float32,
        ),
    )


class _CpuReferenceDriver:
    """Batch-one CPU API/reference path; never a formal benchmark backend."""

    backend: VehicleBackend = "cpu_reference"
    num_worlds = 1

    def __init__(self, config: VehicleConfig) -> None:
        self._vehicle = CpuVehicle(config)

    @staticmethod
    def _pose(rear_axle_pose: Any) -> np.ndarray:
        pose = np.asarray(rear_axle_pose, dtype=np.float64)
        if pose.shape != (1, 3) or not np.isfinite(pose).all():
            raise VehicleDriverShapeError(
                f"rear-axle pose batch must contain one finite (x, y, yaw) row, got {pose.shape}"
            )
        return pose

    def initial_state(self, rear_axle_pose: Any) -> CpuDriverState:
        pose = self._pose(rear_axle_pose)
        self._vehicle.reset(rear_axle_pose=pose[0])
        return CpuDriverState(control_step_count=np.zeros(1, dtype=np.int32))

    def read_state(self, state: CpuDriverState) -> CpuVehicleStateView:
        del state
        return _cpu_view(self._vehicle)

    def step(self, state: CpuDriverState, actions: Any) -> VehicleStep:
        try:
            action_array = np.asarray(actions, dtype=np.float32)
        except (TypeError, ValueError) as error:
            raise VehicleDriverShapeError("actions must be convertible to float32") from error
        if action_array.shape != (1, 2):
            raise VehicleDriverShapeError(
                f"batched actions must have shape (1, 2), got {action_array.shape}"
            )
        invalid = ~np.all(np.isfinite(action_array), axis=1)
        safe_action = np.where(invalid[:, None], 0.0, action_array)
        self._vehicle.step(safe_action[0])
        applied = self._vehicle.last_applied_action
        return VehicleStep(
            state=CpuDriverState(control_step_count=state.control_step_count + 1),
            applied=AppliedActionBatch(
                steering_angle_rad=np.asarray((applied.steering_angle_rad,), dtype=np.float32),
                longitudinal_acceleration_mps2=np.asarray(
                    (applied.longitudinal_acceleration_mps2,),
                    dtype=np.float32,
                ),
                steering_target_rad=np.asarray(
                    (applied.steering_target_rad,),
                    dtype=np.float32,
                ),
                saturation_count=np.asarray((applied.saturation_count,), dtype=np.int32),
                invalid_action=invalid,
            ),
            diagnostics=self._vehicle.last_step_diagnostics,
        )

    def masked_reset(
        self,
        state: CpuDriverState,
        mask: Any,
        rear_axle_pose: Any,
    ) -> CpuDriverState:
        reset_mask = np.asarray(mask, dtype=bool)
        if reset_mask.shape != (1,):
            raise VehicleDriverShapeError(
                f"reset mask must have shape (1,), got {reset_mask.shape}"
            )
        pose = self._pose(rear_axle_pose)
        if bool(reset_mask[0]):
            self._vehicle.reset(rear_axle_pose=pose[0])
            return CpuDriverState(control_step_count=np.zeros(1, dtype=np.int32))
        return state

    def close(self) -> None:
        """The CPU MuJoCo reference owns no external viewer or process."""


class _MjxWarpDriver:
    """Formal native-leading-dimension MJX-Warp path."""

    backend: VehicleBackend = "mjx_warp"

    def __init__(self, config: VehicleConfig, num_worlds: int) -> None:
        self._vehicle = MjxWarpVehicle.create(config, num_worlds=num_worlds)
        self.num_worlds = num_worlds

    def initial_state(self, rear_axle_pose: Any) -> Any:
        return self._vehicle.initial_state(rear_axle_pose)

    def read_state(self, state: Any) -> Any:
        return self._vehicle.read_state(state)

    def step(self, state: Any, actions: Any) -> VehicleStep:
        next_state, applied, diagnostics = self._vehicle.step(state, actions)
        return VehicleStep(
            state=next_state,
            applied=AppliedActionBatch(
                steering_angle_rad=applied.steering_angle_rad,
                longitudinal_acceleration_mps2=applied.longitudinal_acceleration_mps2,
                steering_target_rad=applied.steering_target_rad,
                saturation_count=applied.saturation_count,
                invalid_action=applied.invalid_action,
            ),
            diagnostics=diagnostics,
        )

    def masked_reset(self, state: Any, mask: Any, rear_axle_pose: Any) -> Any:
        return self._vehicle.masked_reset(state, mask, rear_axle_pose)

    def close(self) -> None:
        """MJX-Warp state is device-owned and needs no explicit shutdown."""


def create_vehicle_driver(
    backend: VehicleBackend,
    config: VehicleConfig,
    *,
    num_worlds: int,
) -> _VehicleDriver:
    """Create one of the only two approved private vehicle drivers."""

    if isinstance(num_worlds, bool) or not isinstance(num_worlds, int) or num_worlds <= 0:
        raise ValueError("num_worlds must be a positive integer")
    if backend == "cpu_reference":
        if num_worlds != 1:
            raise VehicleDriverError("cpu_reference is restricted to num_worlds=1")
        return _CpuReferenceDriver(config)
    if backend == "mjx_warp":
        return _MjxWarpDriver(config, num_worlds)
    raise VehicleDriverError(f"unsupported vehicle backend: {backend!r}")


__all__ = [
    "AppliedActionBatch",
    "CpuDriverState",
    "CpuVehicleStateView",
    "VehicleBackend",
    "VehicleDriverError",
    "VehicleDriverShapeError",
    "VehicleStep",
    "create_vehicle_driver",
]
