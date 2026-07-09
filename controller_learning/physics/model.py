"""Load and validate the shared four-wheel MuJoCo model."""

from __future__ import annotations

from dataclasses import dataclass
from importlib.resources import as_file, files
from math import isclose, isfinite

import mujoco
import numpy as np
from numpy.typing import NDArray

from controller_learning.config import VehicleConfig

CHASSIS_BODY_NAME = "chassis"
GROUND_GEOM_NAME = "ground"
REAR_AXLE_SITE_NAME = "rear_axle_reference"
STEERING_JOINT_NAMES = ("front_left_steer", "front_right_steer")
WHEEL_JOINT_NAMES = (
    "front_left_wheel",
    "front_right_wheel",
    "rear_left_wheel",
    "rear_right_wheel",
)
WHEEL_BODY_NAMES = tuple(f"{name}_body" for name in WHEEL_JOINT_NAMES)
WHEEL_GEOM_NAMES = tuple(f"{name}_geom" for name in WHEEL_JOINT_NAMES)
STEERING_ACTUATOR_NAMES = (
    "front_left_steer_position",
    "front_right_steer_position",
)
DRIVE_ACTUATOR_NAMES = (
    "front_left_drive",
    "front_right_drive",
    "rear_left_drive",
    "rear_right_drive",
)


class VehicleModelError(ValueError):
    """Raised when the MJCF model violates the project vehicle contract."""


@dataclass(frozen=True, slots=True)
class VehicleModelIndices:
    """Named MuJoCo indexes used by CPU and future GPU adapters."""

    chassis_body: int
    ground_geom: int
    rear_axle_site: int
    steering_joints: tuple[int, int]
    wheel_joints: tuple[int, int, int, int]
    wheel_bodies: tuple[int, int, int, int]
    wheel_geoms: tuple[int, int, int, int]
    steering_actuators: tuple[int, int]
    drive_actuators: tuple[int, int, int, int]
    steering_qpos: tuple[int, int]
    steering_dofs: tuple[int, int]
    wheel_qpos: tuple[int, int, int, int]
    wheel_dofs: tuple[int, int, int, int]


def _named_ids(
    model: mujoco.MjModel, object_type: mujoco.mjtObj, names: tuple[str, ...]
) -> tuple[int, ...]:
    ids = tuple(mujoco.mj_name2id(model, object_type, name) for name in names)
    missing = [name for name, object_id in zip(names, ids, strict=True) if object_id < 0]
    if missing:
        raise VehicleModelError(f"MJCF is missing named objects: {', '.join(missing)}")
    return ids


def vehicle_model_indices(model: mujoco.MjModel) -> VehicleModelIndices:
    """Resolve and validate the stable named-index contract."""

    chassis_body = _named_ids(model, mujoco.mjtObj.mjOBJ_BODY, (CHASSIS_BODY_NAME,))[0]
    ground_geom = _named_ids(model, mujoco.mjtObj.mjOBJ_GEOM, (GROUND_GEOM_NAME,))[0]
    rear_axle_site = _named_ids(model, mujoco.mjtObj.mjOBJ_SITE, (REAR_AXLE_SITE_NAME,))[0]
    steering_joints = _named_ids(model, mujoco.mjtObj.mjOBJ_JOINT, STEERING_JOINT_NAMES)
    wheel_joints = _named_ids(model, mujoco.mjtObj.mjOBJ_JOINT, WHEEL_JOINT_NAMES)
    wheel_bodies = _named_ids(model, mujoco.mjtObj.mjOBJ_BODY, WHEEL_BODY_NAMES)
    wheel_geoms = _named_ids(model, mujoco.mjtObj.mjOBJ_GEOM, WHEEL_GEOM_NAMES)
    steering_actuators = _named_ids(
        model,
        mujoco.mjtObj.mjOBJ_ACTUATOR,
        STEERING_ACTUATOR_NAMES,
    )
    drive_actuators = _named_ids(model, mujoco.mjtObj.mjOBJ_ACTUATOR, DRIVE_ACTUATOR_NAMES)
    return VehicleModelIndices(
        chassis_body=chassis_body,
        ground_geom=ground_geom,
        rear_axle_site=rear_axle_site,
        steering_joints=(steering_joints[0], steering_joints[1]),
        wheel_joints=(wheel_joints[0], wheel_joints[1], wheel_joints[2], wheel_joints[3]),
        wheel_bodies=(wheel_bodies[0], wheel_bodies[1], wheel_bodies[2], wheel_bodies[3]),
        wheel_geoms=(wheel_geoms[0], wheel_geoms[1], wheel_geoms[2], wheel_geoms[3]),
        steering_actuators=(steering_actuators[0], steering_actuators[1]),
        drive_actuators=(
            drive_actuators[0],
            drive_actuators[1],
            drive_actuators[2],
            drive_actuators[3],
        ),
        steering_qpos=tuple(int(model.jnt_qposadr[joint]) for joint in steering_joints),
        steering_dofs=tuple(int(model.jnt_dofadr[joint]) for joint in steering_joints),
        wheel_qpos=tuple(int(model.jnt_qposadr[joint]) for joint in wheel_joints),
        wheel_dofs=tuple(int(model.jnt_dofadr[joint]) for joint in wheel_joints),
    )


def _initial_wheel_centers(
    model: mujoco.MjModel,
    indices: VehicleModelIndices,
) -> NDArray[np.float64]:
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    return np.asarray(data.xpos[list(indices.wheel_bodies)], dtype=np.float64)


def validate_vehicle_model(
    model: mujoco.MjModel,
    config: VehicleConfig,
    indices: VehicleModelIndices | None = None,
) -> VehicleModelIndices:
    """Check MJCF structure and physical constants against typed configuration."""

    resolved = indices or vehicle_model_indices(model)
    free_joint_count = int(np.count_nonzero(model.jnt_type == mujoco.mjtJoint.mjJNT_FREE))
    if free_joint_count != 1:
        raise VehicleModelError(
            f"vehicle must contain exactly one free joint, got {free_joint_count}"
        )
    if model.njnt != 7:
        raise VehicleModelError(
            f"vehicle must contain one free and six hinge joints, got {model.njnt}"
        )

    for joint_id in (*resolved.steering_joints, *resolved.wheel_joints):
        if model.jnt_type[joint_id] != mujoco.mjtJoint.mjJNT_HINGE:
            raise VehicleModelError("steering and wheel joints must all be hinges")
    for joint_id in resolved.steering_joints:
        if not np.allclose(model.jnt_axis[joint_id], (0.0, 0.0, 1.0), atol=1e-12):
            raise VehicleModelError("steering joint axes must point along body +z")
        expected_range = (
            -config.actuator.max_steering_angle_rad,
            config.actuator.max_steering_angle_rad,
        )
        if not np.allclose(model.jnt_range[joint_id], expected_range, atol=1e-12):
            raise VehicleModelError("steering joint range does not match vehicle config")
    for joint_id in resolved.wheel_joints:
        if not np.allclose(model.jnt_axis[joint_id], (0.0, 1.0, 0.0), atol=1e-12):
            raise VehicleModelError("wheel spin axes must point along body +y")

    total_mass = float(model.body_subtreemass[resolved.chassis_body])
    if not isclose(total_mass, config.vehicle.mass_kg, rel_tol=0.0, abs_tol=1e-9):
        raise VehicleModelError(
            f"MJCF total mass {total_mass} kg does not match config {config.vehicle.mass_kg} kg"
        )

    centers = _initial_wheel_centers(model, resolved)
    front_center = centers[:2].mean(axis=0)
    rear_center = centers[2:].mean(axis=0)
    wheelbase = float(front_center[0] - rear_center[0])
    front_track = float(centers[0, 1] - centers[1, 1])
    rear_track = float(centers[2, 1] - centers[3, 1])
    if not isclose(wheelbase, config.vehicle.wheelbase_m, rel_tol=0.0, abs_tol=1e-12):
        raise VehicleModelError("MJCF wheelbase does not match vehicle config")
    if not (
        isclose(front_track, config.vehicle.track_width_m, rel_tol=0.0, abs_tol=1e-12)
        and isclose(rear_track, config.vehicle.track_width_m, rel_tol=0.0, abs_tol=1e-12)
    ):
        raise VehicleModelError("MJCF track width does not match vehicle config")

    wheel_radii = model.geom_size[list(resolved.wheel_geoms), 0]
    if not np.allclose(wheel_radii, config.vehicle.wheel_radius_m, atol=1e-12):
        raise VehicleModelError("MJCF wheel radius does not match vehicle config")
    if not all(
        model.geom_type[geom] == mujoco.mjtGeom.mjGEOM_CYLINDER for geom in resolved.wheel_geoms
    ):
        raise VehicleModelError("all physical wheels must use cylinder collision geoms")

    expected_drive_limit = (
        config.vehicle.mass_kg
        * config.actuator.max_deceleration_mps2
        * config.vehicle.wheel_radius_m
        / 4.0
    )
    for actuator_id in resolved.drive_actuators:
        control_range = model.actuator_ctrlrange[actuator_id]
        negative_limit_ok = control_range[0] < -expected_drive_limit or isclose(
            control_range[0],
            -expected_drive_limit,
            rel_tol=0.0,
            abs_tol=1e-9,
        )
        positive_limit_ok = control_range[1] > expected_drive_limit or isclose(
            control_range[1],
            expected_drive_limit,
            rel_tol=0.0,
            abs_tol=1e-9,
        )
        if not (negative_limit_ok and positive_limit_ok):
            raise VehicleModelError("drive actuator range cannot supply configured maximum braking")

    if not isfinite(model.opt.timestep) or model.opt.timestep <= 0.0:
        raise VehicleModelError("model timestep must be finite and positive")
    ratio = config.simulation.control_dt_s / model.opt.timestep
    if not isclose(ratio, round(ratio), rel_tol=0.0, abs_tol=1e-9):
        raise VehicleModelError("control_dt / model timestep must be an integer")
    return resolved


def load_vehicle_model(
    config: VehicleConfig,
    *,
    physics_dt_s: float | None = None,
) -> tuple[mujoco.MjModel, VehicleModelIndices]:
    """Load the packaged MJCF and apply an allowed physics timestep override."""

    resource = files("controller_learning").joinpath("assets", "vehicle", "car.xml")
    with as_file(resource) as model_path:
        model = mujoco.MjModel.from_xml_path(str(model_path))
    if physics_dt_s is not None:
        model.opt.timestep = physics_dt_s
    indices = validate_vehicle_model(model, config)
    return model, indices
