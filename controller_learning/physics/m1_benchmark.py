"""Reproducible M1 CPU vehicle stability and timestep benchmark."""

from __future__ import annotations

import hashlib
import json
import math
import os
import platform
import subprocess
import sys
import time
from dataclasses import dataclass
from datetime import UTC, datetime
from importlib.metadata import version
from importlib.resources import as_file, files
from pathlib import Path
from typing import Any

import mujoco
import numpy as np
from numpy.typing import NDArray

from controller_learning.config import VehicleConfig, load_vehicle_config
from controller_learning.physics import CpuVehicle, VehicleSimulationError

CANDIDATE_TIMESTEPS_S = (0.01, 0.005, 0.002)
REFERENCE_TIMESTEP_S = 0.002
FORMAL_REPEATS = 3
PROTOCOL_VERSION = "m1-cpu-v1"
DYNAMIC_WHEEL_CONTACT_COVERAGE_MIN = 0.75
STATIC_WHEEL_CONTACT_COVERAGE_MIN = 0.98
MAXIMUM_WHEEL_CONTACT_GAP_S = 0.1
MINIMUM_MEAN_WHEEL_LOAD_RATIO = 0.8

SCENARIO_DURATIONS_S = {
    "rest": 10.0,
    "drop_settle": 5.0,
    "straight": 6.0,
    "steer_left": 7.0,
    "steer_right": 7.0,
    "brake": 8.0,
    "action_limits": 2.0,
    "contact_stress": 60.0,
}

# Telemetry column indexes. All scenarios sample at the common 20 Hz Controller boundary.
TIME = 0
POSITION = slice(1, 4)
ROLL = 4
PITCH = 5
YAW = 6
VELOCITY = slice(7, 10)
ANGULAR_VELOCITY = slice(10, 13)
STEERING = 13
WHEEL_VELOCITY = slice(14, 18)
PENETRATION = 18
WHEEL_CONTACT = slice(19, 23)
WHEEL_NORMAL_FORCE = slice(23, 27)
UNEXPECTED_CONTACT = 27
WARNING_COUNT = 28
INTERVAL_MAX_ROLL_PITCH = 29
INTERVAL_MAX_VERTICAL_SPEED = 30
INTERVAL_MAX_CONTACT_GAP = 31
APPLIED_STEERING = 32
APPLIED_ACCELERATION = 33
STEERING_TARGET = 34
SATURATION_COUNT = 35


@dataclass(frozen=True, slots=True)
class ScenarioRun:
    """One deterministic scenario result and its control-rate telemetry."""

    telemetry: NDArray[np.float64]
    metrics: dict[str, float | int | bool]
    checks: tuple[dict[str, Any], ...]
    telemetry_sha256: str


def scenario_action(
    name: str, time_s: float, longitudinal_velocity_mps: float
) -> tuple[float, float]:
    """Return the frozen open-loop or stress action for one scenario time."""

    if name in {"rest", "drop_settle"}:
        return (0.0, 0.0)
    if name == "straight":
        return (0.0, 2.0) if 1.0 <= time_s < 4.0 else (0.0, 0.0)
    if name in {"steer_left", "steer_right"}:
        if 1.0 <= time_s < 4.0:
            return (0.0, 1.5)
        if 4.0 <= time_s < 6.0:
            return (0.2 if name == "steer_left" else -0.2, 0.0)
        return (0.0, 0.0)
    if name == "brake":
        if 1.0 <= time_s < 5.0:
            return (0.0, 2.0)
        if 5.0 <= time_s < 7.0:
            return (0.0, -4.0)
        return (0.0, 0.0)
    if name == "action_limits":
        if time_s < 0.5:
            return (2.0, 10.0)
        if time_s < 1.0:
            return (-2.0, -20.0)
        return (0.0, 0.0)
    if name == "contact_stress":
        if time_s < 2.0:
            return (0.0, 0.0)
        steering = 0.2 * math.sin(2.0 * math.pi * (time_s - 2.0) / 4.0)
        acceleration = float(np.clip(2.0 * (8.0 - longitudinal_velocity_mps), -6.0, 3.0))
        return (steering, acceleration)
    raise ValueError(f"unknown M1 scenario: {name}")


def _telemetry_row(vehicle: CpuVehicle) -> list[float]:
    state = vehicle.state()
    diagnostics = vehicle.last_step_diagnostics
    applied = vehicle.last_applied_action
    return [
        state.time_s,
        *state.position_world_m,
        state.roll_rad,
        state.pitch_rad,
        state.yaw_rad,
        *state.velocity_body_mps,
        *state.angular_velocity_body_rad_s,
        state.steering_angle_rad,
        *state.wheel_angular_velocity_rad_s,
        diagnostics.maximum_penetration_m,
        *diagnostics.wheel_ground_contact_fraction,
        *diagnostics.mean_wheel_normal_force_n,
        float(diagnostics.maximum_unexpected_contact_count),
        float(vehicle.warning_count),
        diagnostics.maximum_abs_roll_pitch_rad,
        diagnostics.maximum_abs_vertical_speed_mps,
        diagnostics.maximum_wheel_contact_gap_s,
        applied.steering_angle_rad,
        applied.longitudinal_acceleration_mps2,
        applied.steering_target_rad,
        float(applied.saturation_count),
    ]


def _initialize_drop(vehicle: CpuVehicle) -> None:
    angle = math.radians(1.0)
    vehicle.data.qpos[2] += 0.02
    vehicle.data.qpos[3:7] = (math.cos(angle / 2.0), math.sin(angle / 2.0), 0.0, 0.0)
    mujoco.mj_forward(vehicle.model, vehicle.data)


def _nearest_index(telemetry: NDArray[np.float64], target_time_s: float) -> int:
    return int(np.argmin(np.abs(telemetry[:, TIME] - target_time_s)))


def _metric_checks(
    name: str,
    metrics: dict[str, float | int | bool],
    config: VehicleConfig,
) -> tuple[dict[str, Any], ...]:
    checks: list[dict[str, Any]] = []

    def add(metric: str, operator: str, limit: float | int | bool) -> None:
        value = metrics[metric]
        if operator == "<=":
            passed = float(value) <= float(limit)
        elif operator == ">=":
            passed = float(value) >= float(limit)
        elif operator == "==":
            passed = value == limit
        else:
            raise ValueError(f"unsupported threshold operator: {operator}")
        checks.append(
            {
                "id": f"{name}.{metric}",
                "metric": metric,
                "operator": operator,
                "limit": limit,
                "value": value,
                "passed": bool(passed),
            }
        )

    add("finite", "==", True)
    add("max_warning_count", "==", 0)
    add("max_unexpected_contact_count", "==", 0)
    contact_coverage_limit = (
        DYNAMIC_WHEEL_CONTACT_COVERAGE_MIN
        if name in {"steer_left", "steer_right", "contact_stress"}
        else STATIC_WHEEL_CONTACT_COVERAGE_MIN
    )
    add("minimum_steady_wheel_contact_coverage", ">=", contact_coverage_limit)
    add("maximum_contact_gap_s", "<=", MAXIMUM_WHEEL_CONTACT_GAP_S)
    add("maximum_penetration_m", "<=", 0.01 if name == "drop_settle" else 0.005)
    add("steady_penetration_p99_m", "<=", 0.002)
    add("maximum_abs_roll_pitch_rad", "<=", 0.15)
    add("steady_chassis_z_peak_to_peak_m", "<=", 0.005)
    add("steady_vertical_speed_p99_mps", "<=", 0.1)
    add("minimum_steady_mean_wheel_load_ratio", ">=", MINIMUM_MEAN_WHEEL_LOAD_RATIO)
    add("steady_mean_total_normal_force_relative_error", "<=", 0.1)

    if name == "rest":
        add("steady_xy_drift_m", "<=", 0.01)
        add("steady_yaw_drift_rad", "<=", 0.005)
        add("steady_speed_p99_mps", "<=", 0.02)
        add("steady_yaw_rate_p99_rad_s", "<=", 0.01)
        add("steady_chassis_z_peak_to_peak_m", "<=", 0.002)
        add("steady_roll_pitch_p99_rad", "<=", 0.01)
        add("steady_normal_force_relative_error", "<=", 0.02)
    elif name == "drop_settle":
        add("final_abs_roll_pitch_rad", "<=", 0.01)
        add("steady_chassis_z_peak_to_peak_m", "<=", 0.002)
    elif name == "straight":
        add("forward_displacement_m", ">=", 5.0)
        add("speed_at_acceleration_end_mps", ">=", 4.5)
        add("speed_at_acceleration_end_mps", "<=", 7.5)
        add("maximum_abs_lateral_displacement_m", "<=", 0.02)
        add("maximum_abs_yaw_rad", "<=", 0.01)
    elif name == "steer_left":
        add("final_lateral_displacement_m", ">=", 0.2)
        add("final_yaw_rad", ">=", 0.1)
        add("mean_commanded_steering_angle_rad", ">=", 0.17)
        add("maximum_commanded_steering_error_rad", "<=", 0.05)
    elif name == "steer_right":
        add("final_lateral_displacement_m", "<=", -0.2)
        add("final_yaw_rad", "<=", -0.1)
        add("mean_commanded_steering_angle_rad", "<=", -0.17)
        add("maximum_commanded_steering_error_rad", "<=", 0.05)
    elif name == "brake":
        add("speed_before_braking_mps", ">=", 5.0)
        add("final_abs_longitudinal_velocity_mps", "<=", 0.5)
        add("maximum_abs_post_braking_velocity_mps", "<=", 0.5)
        add("minimum_braking_velocity_mps", ">=", -0.2)
        add("braking_nonincrease_fraction", ">=", 0.95)
    elif name == "action_limits":
        steering_limit = config.actuator.max_steering_angle_rad
        add("maximum_applied_steering_rad", ">=", steering_limit)
        add("maximum_applied_steering_rad", "<=", steering_limit)
        add("minimum_applied_steering_rad", ">=", -steering_limit)
        add("minimum_applied_steering_rad", "<=", -steering_limit)
        add("maximum_applied_acceleration_mps2", ">=", config.actuator.max_acceleration_mps2)
        add("maximum_applied_acceleration_mps2", "<=", config.actuator.max_acceleration_mps2)
        add(
            "minimum_applied_acceleration_mps2",
            ">=",
            -config.actuator.max_deceleration_mps2,
        )
        add(
            "minimum_applied_acceleration_mps2",
            "<=",
            -config.actuator.max_deceleration_mps2,
        )
        add(
            "maximum_steering_target_step_rad",
            "<=",
            config.actuator.max_steering_rate_rad_s * config.simulation.control_dt_s + 1e-12,
        )
        add("minimum_saturation_count", ">=", 2)
    elif name == "contact_stress":
        add("maximum_speed_mps", "<=", 12.0)
        add("mean_stress_speed_mps", ">=", 7.0)
        add("stress_path_length_m", ">=", 400.0)
        add("stress_steering_rms_rad", ">=", 0.1)
    return tuple(checks)


def _scenario_metrics(
    name: str,
    telemetry: NDArray[np.float64],
    config: VehicleConfig,
    expected_weight_n: float,
) -> dict[str, float | int | bool]:
    steady_start_s = 1.2 if name == "action_limits" else 1.0
    steady_rows = telemetry[_nearest_index(telemetry, steady_start_s) :]
    base_position = steady_rows[0, POSITION]
    speed = np.linalg.norm(telemetry[:, VELOCITY][:, :2], axis=1)
    steady_speed = np.linalg.norm(steady_rows[:, VELOCITY][:, :2], axis=1)
    roll_pitch = np.abs(telemetry[:, [ROLL, PITCH]])
    steady_roll_pitch = np.abs(steady_rows[:, [ROLL, PITCH]])
    wheel_mean_normal_force = np.mean(steady_rows[:, WHEEL_NORMAL_FORCE], axis=0)
    normal_force = steady_rows[:, WHEEL_NORMAL_FORCE].sum(axis=1)
    static_wheel_load_n = expected_weight_n / 4.0
    metrics: dict[str, float | int | bool] = {
        "finite": bool(np.isfinite(telemetry).all()),
        "max_warning_count": int(np.max(telemetry[:, WARNING_COUNT])),
        "max_unexpected_contact_count": int(np.max(telemetry[:, UNEXPECTED_CONTACT])),
        "minimum_steady_wheel_contact_coverage": float(
            np.min(np.mean(steady_rows[:, WHEEL_CONTACT], axis=0))
        ),
        "maximum_contact_gap_s": float(np.max(steady_rows[:, INTERVAL_MAX_CONTACT_GAP])),
        "maximum_penetration_m": float(np.max(telemetry[:, PENETRATION])),
        "penetration_p99_m": float(np.percentile(telemetry[:, PENETRATION], 99)),
        "steady_penetration_p99_m": float(np.percentile(steady_rows[:, PENETRATION], 99)),
        "maximum_abs_roll_pitch_rad": float(np.max(telemetry[:, INTERVAL_MAX_ROLL_PITCH])),
        "maximum_speed_mps": float(np.max(speed)),
        "steady_chassis_z_peak_to_peak_m": float(np.ptp(steady_rows[:, POSITION][:, 2])),
        "steady_vertical_speed_p99_mps": float(
            np.percentile(steady_rows[:, INTERVAL_MAX_VERTICAL_SPEED], 99)
        ),
        "final_longitudinal_velocity_mps": float(telemetry[-1, VELOCITY.start]),
        "final_abs_longitudinal_velocity_mps": float(abs(telemetry[-1, VELOCITY.start])),
        "minimum_steady_mean_wheel_load_ratio": float(
            np.min(wheel_mean_normal_force) / static_wheel_load_n
        ),
        "steady_mean_total_normal_force_relative_error": float(
            abs(np.mean(normal_force) - expected_weight_n) / expected_weight_n
        ),
    }
    if name == "rest":
        xy_delta = steady_rows[:, POSITION][:, :2] - base_position[:2]
        yaw_delta = np.arctan2(
            np.sin(steady_rows[:, YAW] - steady_rows[0, YAW]),
            np.cos(steady_rows[:, YAW] - steady_rows[0, YAW]),
        )
        metrics.update(
            {
                "steady_xy_drift_m": float(np.max(np.linalg.norm(xy_delta, axis=1))),
                "steady_yaw_drift_rad": float(np.max(np.abs(yaw_delta))),
                "steady_speed_p99_mps": float(np.percentile(steady_speed, 99)),
                "steady_yaw_rate_p99_rad_s": float(
                    np.percentile(np.abs(steady_rows[:, ANGULAR_VELOCITY][:, 2]), 99)
                ),
                "steady_roll_pitch_p99_rad": float(np.percentile(steady_roll_pitch, 99)),
                "steady_normal_force_relative_error": float(
                    abs(np.mean(normal_force) - expected_weight_n) / expected_weight_n
                ),
            }
        )
    elif name == "drop_settle":
        metrics["final_abs_roll_pitch_rad"] = float(np.max(roll_pitch[-1]))
    elif name == "straight":
        acceleration_end = telemetry[_nearest_index(telemetry, 4.0)]
        metrics.update(
            {
                "forward_displacement_m": float(telemetry[-1, 1] - telemetry[0, 1]),
                "speed_at_acceleration_end_mps": float(acceleration_end[VELOCITY.start]),
                "maximum_abs_lateral_displacement_m": float(
                    np.max(np.abs(telemetry[:, 2] - telemetry[0, 2]))
                ),
                "maximum_abs_yaw_rad": float(np.max(np.abs(telemetry[:, YAW]))),
            }
        )
    elif name in {"steer_left", "steer_right"}:
        command_start = _nearest_index(telemetry, 4.5)
        command_end = _nearest_index(telemetry, 5.8)
        command_rows = telemetry[command_start:command_end]
        commanded_angle = 0.2 if name == "steer_left" else -0.2
        metrics.update(
            {
                "final_lateral_displacement_m": float(telemetry[-1, 2] - telemetry[0, 2]),
                "final_yaw_rad": float(telemetry[-1, YAW]),
                "mean_commanded_steering_angle_rad": float(np.mean(command_rows[:, STEERING])),
                "maximum_commanded_steering_error_rad": float(
                    np.max(np.abs(command_rows[:, STEERING] - commanded_angle))
                ),
            }
        )
    elif name == "brake":
        before_braking = _nearest_index(telemetry, 5.0)
        brake_end = _nearest_index(telemetry, 7.0)
        braking_speed = telemetry[before_braking : brake_end + 1, VELOCITY.start]
        post_braking_speed = telemetry[brake_end:, VELOCITY.start]
        metrics.update(
            {
                "speed_before_braking_mps": float(braking_speed[0]),
                "minimum_braking_velocity_mps": float(np.min(braking_speed)),
                "braking_nonincrease_fraction": float(np.mean(np.diff(braking_speed) <= 1e-6)),
                "maximum_abs_post_braking_velocity_mps": float(np.max(np.abs(post_braking_speed))),
            }
        )
    elif name == "action_limits":
        action_end = _nearest_index(telemetry, 1.0)
        action_rows = telemetry[1 : action_end + 1]
        metrics.update(
            {
                "maximum_applied_steering_rad": float(np.max(action_rows[:, APPLIED_STEERING])),
                "minimum_applied_steering_rad": float(np.min(action_rows[:, APPLIED_STEERING])),
                "maximum_applied_acceleration_mps2": float(
                    np.max(action_rows[:, APPLIED_ACCELERATION])
                ),
                "minimum_applied_acceleration_mps2": float(
                    np.min(action_rows[:, APPLIED_ACCELERATION])
                ),
                "maximum_steering_target_step_rad": float(
                    np.max(np.abs(np.diff(telemetry[:, STEERING_TARGET])))
                ),
                "minimum_saturation_count": int(np.min(action_rows[:, SATURATION_COUNT])),
            }
        )
    elif name == "contact_stress":
        stress_start = _nearest_index(telemetry, 5.0)
        stress_rows = telemetry[stress_start:]
        stress_speed = np.linalg.norm(stress_rows[:, VELOCITY][:, :2], axis=1)
        path_delta = np.diff(telemetry[:, POSITION][:, :2], axis=0)
        metrics.update(
            {
                "mean_stress_speed_mps": float(np.mean(stress_speed)),
                "stress_path_length_m": float(np.sum(np.linalg.norm(path_delta, axis=1))),
                "stress_steering_rms_rad": float(np.sqrt(np.mean(stress_rows[:, STEERING] ** 2))),
            }
        )
    return metrics


def run_scenario(config_path: Path, physics_dt_s: float, name: str) -> ScenarioRun:
    """Run one formal deterministic M1 scenario."""

    config = load_vehicle_config(config_path)
    vehicle = CpuVehicle(config, physics_dt_s=physics_dt_s)
    if name == "drop_settle":
        _initialize_drop(vehicle)
    rows = [_telemetry_row(vehicle)]
    control_steps = round(SCENARIO_DURATIONS_S[name] / config.simulation.control_dt_s)
    for control_step in range(control_steps):
        state = vehicle.state()
        schedule_time_s = control_step * config.simulation.control_dt_s
        action = scenario_action(name, schedule_time_s, state.longitudinal_velocity_mps)
        vehicle.step(action)
        rows.append(_telemetry_row(vehicle))
    telemetry = np.asarray(rows, dtype=np.float64)
    expected_weight_n = config.vehicle.mass_kg * abs(float(vehicle.model.opt.gravity[2]))
    metrics = _scenario_metrics(name, telemetry, config, expected_weight_n)
    checks = _metric_checks(name, metrics, config)
    digest = hashlib.sha256(np.ascontiguousarray(telemetry).tobytes()).hexdigest()
    return ScenarioRun(telemetry=telemetry, metrics=metrics, checks=checks, telemetry_sha256=digest)


def _check(check_id: str, value: float, operator: str, limit: float) -> dict[str, Any]:
    passed = value <= limit if operator == "<=" else value >= limit
    return {
        "id": check_id,
        "value": value,
        "operator": operator,
        "limit": limit,
        "passed": bool(passed),
    }


def _symmetry_checks(runs: dict[str, list[ScenarioRun]]) -> list[dict[str, Any]]:
    left = runs["steer_left"][0].metrics
    right = runs["steer_right"][0].metrics
    lateral_left = abs(float(left["final_lateral_displacement_m"]))
    lateral_right = abs(float(right["final_lateral_displacement_m"]))
    yaw_left = abs(float(left["final_yaw_rad"]))
    yaw_right = abs(float(right["final_yaw_rad"]))
    lateral_error = abs(lateral_left - lateral_right) / max(lateral_left, lateral_right, 1e-12)
    yaw_error = abs(yaw_left - yaw_right) / max(yaw_left, yaw_right, 1e-12)
    return [
        _check("steering.lateral_symmetry_relative_error", lateral_error, "<=", 0.1),
        _check("steering.yaw_symmetry_relative_error", yaw_error, "<=", 0.1),
    ]


def _convergence_checks(
    candidate: dict[str, list[ScenarioRun]],
    reference: dict[str, list[ScenarioRun]],
) -> tuple[dict[str, dict[str, float]], list[dict[str, Any]]]:
    metrics_by_scenario: dict[str, dict[str, float]] = {}
    checks: list[dict[str, Any]] = []
    for name in ("rest", "drop_settle", "straight", "steer_left", "steer_right", "brake"):
        actual = candidate[name][0].telemetry
        expected = reference[name][0].telemetry
        position_error = np.linalg.norm(
            actual[:, POSITION][:, :2] - expected[:, POSITION][:, :2], axis=1
        )
        velocity_error = np.linalg.norm(
            actual[:, VELOCITY][:, :2] - expected[:, VELOCITY][:, :2], axis=1
        )
        yaw_error = np.arctan2(
            np.sin(actual[:, YAW] - expected[:, YAW]),
            np.cos(actual[:, YAW] - expected[:, YAW]),
        )
        yaw_rate_error = actual[:, ANGULAR_VELOCITY][:, 2] - expected[:, ANGULAR_VELOCITY][:, 2]
        values = {
            "position_rmse_m": float(np.sqrt(np.mean(position_error**2))),
            "position_max_m": float(np.max(position_error)),
            "position_final_m": float(position_error[-1]),
            "velocity_rmse_mps": float(np.sqrt(np.mean(velocity_error**2))),
            "velocity_max_mps": float(np.max(velocity_error)),
            "yaw_rmse_rad": float(np.sqrt(np.mean(yaw_error**2))),
            "yaw_final_rad": float(abs(yaw_error[-1])),
            "yaw_rate_rmse_rad_s": float(np.sqrt(np.mean(yaw_rate_error**2))),
            "steady_z_difference_m": float(
                abs(actual[-1, POSITION.stop - 1] - expected[-1, POSITION.stop - 1])
            ),
            "penetration_p99_difference_m": float(
                abs(
                    np.percentile(actual[:, PENETRATION], 99)
                    - np.percentile(expected[:, PENETRATION], 99)
                )
            ),
        }
        metrics_by_scenario[name] = values
        prefix = f"convergence.{name}"
        if name in {"rest", "drop_settle"}:
            checks.extend(
                [
                    _check(
                        f"{prefix}.steady_z_difference_m",
                        values["steady_z_difference_m"],
                        "<=",
                        0.002,
                    ),
                    _check(
                        f"{prefix}.penetration_p99_difference_m",
                        values["penetration_p99_difference_m"],
                        "<=",
                        0.002,
                    ),
                ]
            )
        elif name in {"straight", "brake"}:
            checks.extend(
                [
                    _check(f"{prefix}.position_rmse_m", values["position_rmse_m"], "<=", 0.05),
                    _check(f"{prefix}.position_max_m", values["position_max_m"], "<=", 0.15),
                    _check(f"{prefix}.velocity_rmse_mps", values["velocity_rmse_mps"], "<=", 0.1),
                    _check(f"{prefix}.velocity_max_mps", values["velocity_max_mps"], "<=", 0.3),
                ]
            )
            if name == "straight":
                checks.append(
                    _check(f"{prefix}.yaw_final_rad", values["yaw_final_rad"], "<=", 0.01)
                )
        else:
            checks.extend(
                [
                    _check(f"{prefix}.position_rmse_m", values["position_rmse_m"], "<=", 0.1),
                    _check(f"{prefix}.position_final_m", values["position_final_m"], "<=", 0.25),
                    _check(f"{prefix}.yaw_rmse_rad", values["yaw_rmse_rad"], "<=", 0.02),
                    _check(f"{prefix}.yaw_final_rad", values["yaw_final_rad"], "<=", 0.05),
                    _check(f"{prefix}.velocity_rmse_mps", values["velocity_rmse_mps"], "<=", 0.2),
                    _check(
                        f"{prefix}.yaw_rate_rmse_rad_s",
                        values["yaw_rate_rmse_rad_s"],
                        "<=",
                        0.05,
                    ),
                ]
            )
    candidate_stress = candidate["contact_stress"][0].metrics
    reference_stress = reference["contact_stress"][0].metrics
    stress_values = {
        "wheel_contact_coverage_difference": abs(
            float(candidate_stress["minimum_steady_wheel_contact_coverage"])
            - float(reference_stress["minimum_steady_wheel_contact_coverage"])
        ),
        "maximum_contact_gap_difference_s": abs(
            float(candidate_stress["maximum_contact_gap_s"])
            - float(reference_stress["maximum_contact_gap_s"])
        ),
        "steady_penetration_p99_difference_m": abs(
            float(candidate_stress["steady_penetration_p99_m"])
            - float(reference_stress["steady_penetration_p99_m"])
        ),
        "vertical_speed_p99_difference_mps": abs(
            float(candidate_stress["steady_vertical_speed_p99_mps"])
            - float(reference_stress["steady_vertical_speed_p99_mps"])
        ),
        "mean_speed_difference_mps": abs(
            float(candidate_stress["mean_stress_speed_mps"])
            - float(reference_stress["mean_stress_speed_mps"])
        ),
        "path_length_difference_m": abs(
            float(candidate_stress["stress_path_length_m"])
            - float(reference_stress["stress_path_length_m"])
        ),
        "steering_rms_difference_rad": abs(
            float(candidate_stress["stress_steering_rms_rad"])
            - float(reference_stress["stress_steering_rms_rad"])
        ),
    }
    metrics_by_scenario["contact_stress"] = stress_values
    checks.extend(
        [
            _check(
                "convergence.contact_stress.wheel_contact_coverage_difference",
                stress_values["wheel_contact_coverage_difference"],
                "<=",
                0.05,
            ),
            _check(
                "convergence.contact_stress.maximum_contact_gap_difference_s",
                stress_values["maximum_contact_gap_difference_s"],
                "<=",
                0.02,
            ),
            _check(
                "convergence.contact_stress.steady_penetration_p99_difference_m",
                stress_values["steady_penetration_p99_difference_m"],
                "<=",
                0.001,
            ),
            _check(
                "convergence.contact_stress.vertical_speed_p99_difference_mps",
                stress_values["vertical_speed_p99_difference_mps"],
                "<=",
                0.05,
            ),
            _check(
                "convergence.contact_stress.mean_speed_difference_mps",
                stress_values["mean_speed_difference_mps"],
                "<=",
                0.1,
            ),
            _check(
                "convergence.contact_stress.path_length_difference_m",
                stress_values["path_length_difference_m"],
                "<=",
                1.0,
            ),
            _check(
                "convergence.contact_stress.steering_rms_difference_rad",
                stress_values["steering_rms_difference_rad"],
                "<=",
                0.01,
            ),
        ]
    )
    return metrics_by_scenario, checks


def select_largest_passing(candidate_pass: dict[float, bool]) -> float | None:
    """Select the largest confirmed timestep that passed every M1 gate."""

    return next((dt for dt in CANDIDATE_TIMESTEPS_S if candidate_pass.get(dt, False)), None)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _git(project_root: Path, *args: str) -> str:
    return subprocess.run(
        ("git", *args),
        cwd=project_root,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


def _cpu_name() -> str:
    try:
        for line in Path("/proc/cpuinfo").read_text().splitlines():
            if line.startswith("model name"):
                return line.split(":", 1)[1].strip()
    except OSError:
        pass
    return platform.processor() or "unknown"


def run_m1_benchmark(project_root: Path) -> dict[str, Any]:
    """Run the complete formal M1 protocol and return a strict-JSON report."""

    project_root = project_root.resolve()
    config_path = project_root / "configs" / "vehicle.toml"
    config = load_vehicle_config(config_path)
    control_dt_s = config.simulation.control_dt_s
    all_runs: dict[float, dict[str, list[ScenarioRun]] | None] = {}
    result_by_dt: dict[float, dict[str, Any]] = {}
    for physics_dt_s in CANDIDATE_TIMESTEPS_S:
        started = time.perf_counter()
        scenario_runs: dict[str, list[ScenarioRun]] = {}
        try:
            for name in SCENARIO_DURATIONS_S:
                scenario_runs[name] = [
                    run_scenario(config_path, physics_dt_s, name) for _ in range(FORMAL_REPEATS)
                ]
        except VehicleSimulationError as error:
            elapsed = time.perf_counter() - started
            all_runs[physics_dt_s] = None
            result_by_dt[physics_dt_s] = {
                "physics_dt_s": physics_dt_s,
                "physics_hz": round(1.0 / physics_dt_s),
                "substeps_per_control": round(control_dt_s / physics_dt_s),
                "scenarios": {},
                "runtime_error": {
                    "type": type(error).__name__,
                    "message": str(error),
                },
                "cpu_timing": {
                    "wall_time_s": elapsed,
                    "physics_steps": 0,
                    "physics_steps_per_second": 0.0,
                },
                "absolute_checks_passed": False,
                "absolute_failures": ["candidate.runtime_error"],
            }
            continue
        elapsed = time.perf_counter() - started
        all_runs[physics_dt_s] = scenario_runs
        scenario_results: dict[str, Any] = {}
        candidate_checks: list[dict[str, Any]] = []
        max_determinism_delta = 0.0
        control_steps = 0
        for name, runs in scenario_runs.items():
            baseline = runs[0].telemetry
            deltas = [float(np.max(np.abs(run.telemetry - baseline))) for run in runs[1:]]
            scenario_delta = max(deltas, default=0.0)
            max_determinism_delta = max(max_determinism_delta, scenario_delta)
            control_steps += (baseline.shape[0] - 1) * FORMAL_REPEATS
            candidate_checks.extend(runs[0].checks)
            scenario_results[name] = {
                "duration_s": SCENARIO_DURATIONS_S[name],
                "control_steps": int(baseline.shape[0] - 1),
                "metrics": runs[0].metrics,
                "checks": list(runs[0].checks),
                "telemetry_sha256": [run.telemetry_sha256 for run in runs],
                "determinism_max_abs_delta": scenario_delta,
            }
        determinism_check = _check(
            "determinism.max_abs_state_delta",
            max_determinism_delta,
            "<=",
            1e-12,
        )
        symmetry_checks = _symmetry_checks(scenario_runs)
        candidate_checks.extend([determinism_check, *symmetry_checks])
        substeps = round(control_dt_s / physics_dt_s)
        result_by_dt[physics_dt_s] = {
            "physics_dt_s": physics_dt_s,
            "physics_hz": round(1.0 / physics_dt_s),
            "substeps_per_control": substeps,
            "scenarios": scenario_results,
            "determinism": {
                "repeats": FORMAL_REPEATS,
                "max_abs_state_delta": max_determinism_delta,
                "passed": determinism_check["passed"],
            },
            "symmetry_checks": symmetry_checks,
            "cpu_timing": {
                "wall_time_s": elapsed,
                "physics_steps": control_steps * substeps,
                "physics_steps_per_second": control_steps * substeps / elapsed,
            },
            "absolute_checks_passed": all(check["passed"] for check in candidate_checks),
            "absolute_failures": [check["id"] for check in candidate_checks if not check["passed"]],
        }

    reference = all_runs[REFERENCE_TIMESTEP_S]
    candidate_pass: dict[float, bool] = {}
    for physics_dt_s in CANDIDATE_TIMESTEPS_S:
        scenario_runs = all_runs[physics_dt_s]
        result = result_by_dt[physics_dt_s]
        if reference is None:
            result["convergence_to_reference"] = {}
            result["convergence_checks"] = []
            result["convergence_passed"] = False
            result["convergence_failures"] = ["convergence.reference_runtime_error"]
            result["passed"] = False
            candidate_pass[physics_dt_s] = False
            continue
        if scenario_runs is None:
            result["convergence_to_reference"] = {}
            result["convergence_checks"] = []
            result["convergence_passed"] = False
            result["convergence_failures"] = ["convergence.candidate_runtime_error"]
            result["passed"] = False
            candidate_pass[physics_dt_s] = False
            continue
        convergence, convergence_checks = _convergence_checks(scenario_runs, reference)
        result["convergence_to_reference"] = convergence
        result["convergence_checks"] = convergence_checks
        result["convergence_passed"] = all(check["passed"] for check in convergence_checks)
        result["convergence_failures"] = [
            check["id"] for check in convergence_checks if not check["passed"]
        ]
        result["passed"] = bool(result["absolute_checks_passed"] and result["convergence_passed"])
        candidate_pass[physics_dt_s] = result["passed"]

    selected = select_largest_passing(candidate_pass)
    benchmark_vehicle = CpuVehicle(
        config,
        physics_dt_s=selected or REFERENCE_TIMESTEP_S,
    )
    model = benchmark_vehicle.model
    resource = files("controller_learning").joinpath("assets", "vehicle", "car.xml")
    with as_file(resource) as model_path:
        model_hash = _sha256(model_path)
    benchmark_source_hash = _sha256(Path(__file__).resolve())
    protocol = {
        "protocol_version": PROTOCOL_VERSION,
        "benchmark_source_sha256": benchmark_source_hash,
        "control_dt_s": control_dt_s,
        "candidate_dt_s": CANDIDATE_TIMESTEPS_S,
        "reference_dt_s": REFERENCE_TIMESTEP_S,
        "repeats": FORMAL_REPEATS,
        "scenario_durations_s": SCENARIO_DURATIONS_S,
        "schedule_timebase": "integer control-step index multiplied by control_dt_s",
        "contact_stability_definition": {
            "sampling": "every MuJoCo physics substep",
            "rigid_vehicle_basis": (
                "The v0.1 plant intentionally has no suspension. Dynamic wheel constraints may "
                "open briefly, so stability requires bounded per-wheel contact participation, "
                "bounded continuous gaps, sustained mean wheel load, low penetration, and low "
                "vertical motion rather than uninterrupted four-contact enforcement."
            ),
            "dynamic_minimum_per_wheel_contact_fraction": (DYNAMIC_WHEEL_CONTACT_COVERAGE_MIN),
            "non_dynamic_minimum_per_wheel_contact_fraction": (STATIC_WHEEL_CONTACT_COVERAGE_MIN),
            "maximum_continuous_per_wheel_contact_gap_s": MAXIMUM_WHEEL_CONTACT_GAP_S,
            "minimum_mean_wheel_load_ratio": MINIMUM_MEAN_WHEEL_LOAD_RATIO,
        },
        "selection_rule": (
            "largest candidate passing absolute, physical, determinism, symmetry, "
            "and convergence gates"
        ),
    }
    protocol_sha = hashlib.sha256(
        json.dumps(protocol, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    git_dirty = bool(_git(project_root, "status", "--porcelain"))
    physics_passed = selected is not None
    evidence_valid = not git_dirty
    m1_passed = physics_passed and evidence_valid
    report = {
        "schema_version": 1,
        "milestone": "M1",
        "status": "pass" if m1_passed else "fail",
        "generated_at_utc": datetime.now(UTC).isoformat(),
        "provenance": {
            "git_commit": _git(project_root, "rev-parse", "HEAD"),
            "git_dirty": git_dirty,
            "pixi_lock_sha256": _sha256(project_root / "pixi.lock"),
            "model_path": "controller_learning/assets/vehicle/car.xml",
            "model_sha256": model_hash,
            "vehicle_config_sha256": _sha256(config_path),
            "protocol_sha256": protocol_sha,
            "benchmark_source_sha256": benchmark_source_hash,
        },
        "runtime": {
            "os": platform.platform(),
            "kernel": platform.release(),
            "machine": platform.machine(),
            "cpu": _cpu_name(),
            "python_version": sys.version.split()[0],
            "mujoco_version": mujoco.__version__,
            "numpy_version": version("numpy"),
            "process_id": os.getpid(),
        },
        "model": {
            "nq": model.nq,
            "nv": model.nv,
            "nu": model.nu,
            "total_mass_kg": float(model.body_subtreemass[benchmark_vehicle.indices.chassis_body]),
            "integrator": mujoco.mjtIntegrator(model.opt.integrator).name,
            "solver": mujoco.mjtSolver(model.opt.solver).name,
            "iterations": model.opt.iterations,
            "cone": mujoco.mjtCone(model.opt.cone).name,
            "autoreset_disabled": bool(
                model.opt.disableflags & mujoco.mjtDisableBit.mjDSBL_AUTORESET
            ),
        },
        "protocol": protocol,
        "results": [result_by_dt[dt] for dt in CANDIDATE_TIMESTEPS_S],
        "selection": {
            "selected_physics_dt_s": selected,
            "physics_passed": physics_passed,
            "evidence_valid": evidence_valid,
            "m1_passed": m1_passed,
            "ready_for_m2": m1_passed,
            "candidate_pass": {str(dt): passed for dt, passed in candidate_pass.items()},
        },
    }
    return report


def write_m1_report(project_root: Path, output: Path) -> dict[str, Any]:
    """Run M1 and write its report using strict JSON without NaN values."""

    report = run_m1_benchmark(project_root)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2, sort_keys=True, allow_nan=False) + "\n")
    return report
