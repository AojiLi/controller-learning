"""Educational warm-started Frenet MPC using only public Challenge inputs."""

from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

import numpy as np
from numpy.typing import NDArray

from controller_learning.control import CenterlineReference, Controller, DebugDraw, wrap_angle

from .helpers import (
    HorizonReference,
    MpcControllerConfig,
    build_horizon_reference,
    deterministic_fallback_action,
)
from .solver import FrenetMpcSolver, MpcLimits, MpcRequest, MpcSolveResult


def _public_table(config: Mapping[str, Any], key: str) -> Mapping[str, Any]:
    value = config.get(key)
    if not isinstance(value, Mapping):
        raise ValueError(f"public config field {key!r} must be a table")
    return value


def _finite_positive(value: object, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{name} must be a number")
    result = float(value)
    if not math.isfinite(result) or result <= 0.0:
        raise ValueError(f"{name} must be finite and positive")
    return result


@dataclass(frozen=True, slots=True)
class _DebugState:
    position_m: NDArray[np.float64]
    projected_point_m: NDArray[np.float64]
    reference_points_m: NDArray[np.float64]
    predicted_points_m: NDArray[np.float64]
    mode: str
    solver_status: str
    target_speed_mps: float
    lateral_error_m: float
    heading_error_rad: float


class MpcController(Controller):
    """Track an observation-derived centerline with constrained kinematic-car NMPC."""

    def __init__(
        self,
        obs: Mapping[str, Any],
        info: Mapping[str, Any],
        config: Mapping[str, Any],
    ) -> None:
        del info
        self._reference = CenterlineReference.from_observation(obs)
        self._parameters = MpcControllerConfig.from_public_config(config)
        self._dt_s = _finite_positive(config.get("control_dt_s"), "control_dt_s")
        if not math.isclose(self._dt_s, 0.05, rel_tol=0.0, abs_tol=1.0e-12):
            raise ValueError("the v0.1 MPC requires the public 0.05 s control period")

        vehicle = _public_table(config, "vehicle")
        action_limits = _public_table(config, "action_limits")
        self._vehicle_width_m = _finite_positive(vehicle.get("vehicle_width_m"), "vehicle_width_m")
        self._limits = MpcLimits(
            wheelbase_m=_finite_positive(vehicle.get("wheelbase_m"), "wheelbase_m"),
            maximum_speed_mps=_finite_positive(vehicle.get("max_speed_mps"), "max_speed_mps"),
            maximum_steering_rad=_finite_positive(
                action_limits.get("max_steering_angle_rad"), "max_steering_angle_rad"
            ),
            maximum_steering_rate_rad_s=_finite_positive(
                action_limits.get("max_steering_rate_rad_s"),
                "max_steering_rate_rad_s",
            ),
            maximum_acceleration_mps2=_finite_positive(
                action_limits.get("max_acceleration_mps2"), "max_acceleration_mps2"
            ),
            maximum_deceleration_mps2=_finite_positive(
                action_limits.get("max_deceleration_mps2"), "max_deceleration_mps2"
            ),
        )
        if self._parameters.planning.maximum_speed_mps > self._limits.maximum_speed_mps:
            raise ValueError("MPC planning speed exceeds the public vehicle speed limit")
        if self._parameters.solver.maximum_wall_time_s > self._dt_s:
            raise ValueError("MPC solver wall-time limit cannot exceed the public control period")

        self._solver = FrenetMpcSolver(
            steps=self._parameters.horizon.steps,
            dt_s=self._dt_s,
            limits=self._limits,
            weights=self._parameters.weights,
            options=self._parameters.solver,
        )
        self._segment_hint: int | None = None
        self._previous_acceleration_mps2 = 0.0
        self._fallback_plan = np.empty((0, 2), dtype=np.float64)
        self._debug: _DebugState | None = None

    def _vehicle_state(
        self, obs: Mapping[str, Any]
    ) -> tuple[NDArray[np.float64], float, float, float]:
        try:
            position = np.asarray(obs["position"], dtype=np.float64)
            velocity_body = np.asarray(obs["velocity_body"], dtype=np.float64)
            yaw = float(np.asarray(obs["yaw"], dtype=np.float64))
            measured_steering = float(np.asarray(obs["steering_angle"], dtype=np.float64))
        except (KeyError, TypeError, ValueError, OverflowError) as error:
            raise ValueError("MPC Controller received a malformed public vehicle state") from error
        if (
            position.shape != (2,)
            or velocity_body.shape != (2,)
            or not np.isfinite(position).all()
            or not np.isfinite(velocity_body).all()
            or not np.isfinite((yaw, measured_steering)).all()
        ):
            raise ValueError("MPC Controller received a non-finite public vehicle state")
        speed = float(np.clip(velocity_body[0], 0.0, self._limits.maximum_speed_mps))
        return position, yaw, speed, measured_steering

    def _bounded_action(
        self, requested: NDArray[np.float64], measured_steering: float
    ) -> NDArray[np.float64]:
        values = np.asarray(requested, dtype=np.float64)
        if values.shape != (2,) or not np.isfinite(values).all():
            raise ValueError("MPC action request must contain two finite values")
        steering_center = float(
            np.clip(
                measured_steering,
                -self._limits.maximum_steering_rad,
                self._limits.maximum_steering_rad,
            )
        )
        steering_step = self._limits.maximum_steering_rate_rad_s * self._dt_s
        steering = float(
            np.clip(
                values[0],
                max(-self._limits.maximum_steering_rad, steering_center - steering_step),
                min(self._limits.maximum_steering_rad, steering_center + steering_step),
            )
        )
        acceleration = float(
            np.clip(
                values[1],
                -self._limits.maximum_deceleration_mps2,
                self._limits.maximum_acceleration_mps2,
            )
        )
        return np.asarray((steering, acceleration), dtype=np.float64)

    def _choose_action(
        self,
        result: MpcSolveResult,
        *,
        initial_state: NDArray[np.float64],
        horizon: HorizonReference,
        measured_steering: float,
    ) -> tuple[NDArray[np.float64], str]:
        if result.success and result.controls is not None:
            controls = np.asarray(result.controls, dtype=np.float64)
            if controls.shape != (self._parameters.horizon.steps, 2):
                raise RuntimeError("successful MPC result has the wrong control shape")
            self._fallback_plan = np.array(controls[1:], copy=True)
            return self._bounded_action(controls[0], measured_steering), "mpc"

        if self._fallback_plan.shape[0] > 0:
            requested = np.array(self._fallback_plan[0], copy=True)
            self._fallback_plan = np.array(self._fallback_plan[1:], copy=True)
            return self._bounded_action(requested, measured_steering), "shifted-plan"

        requested = deterministic_fallback_action(
            lateral_error_m=float(initial_state[0]),
            heading_error_rad=float(initial_state[1]),
            speed_mps=float(initial_state[2]),
            curvature_1pm=float(horizon.curvature_1pm[0]),
            target_speed_mps=float(horizon.target_speed_mps[0]),
            wheelbase_m=self._limits.wheelbase_m,
            config=self._parameters.fallback,
        )
        return self._bounded_action(requested, measured_steering), "feedback-fallback"

    @staticmethod
    def _predicted_world_points(
        result: MpcSolveResult,
        horizon: HorizonReference,
        lateral_error_m: float,
    ) -> NDArray[np.float64]:
        normals = np.stack((-horizon.tangent[:, 1], horizon.tangent[:, 0]), axis=1)
        if result.success and result.states is not None:
            errors = np.asarray(result.states[0], dtype=np.float64)
        else:
            errors = np.full(horizon.center_m.shape[0], lateral_error_m, dtype=np.float64)
        return np.asarray(horizon.center_m + errors[:, None] * normals, dtype=np.float64)

    def compute_control(
        self,
        obs: Mapping[str, Any],
        info: Mapping[str, Any] | None = None,
    ) -> NDArray[np.float32]:
        """Return one finite constrained steering/acceleration action."""

        del info
        position, yaw, speed, measured_steering = self._vehicle_state(obs)
        planning = self._parameters.planning
        projection = self._reference.project(
            position,
            hint_segment=self._segment_hint,
            backward_segments=planning.projection_backward_segments,
            forward_segments=planning.projection_forward_segments,
        )
        self._segment_hint = projection.segment_index
        reference_heading = math.atan2(float(projection.tangent[1]), float(projection.tangent[0]))
        heading_error = float(wrap_angle(yaw - reference_heading))
        initial_state = np.asarray(
            (projection.lateral_error_m, heading_error, speed), dtype=np.float64
        )
        horizon = build_horizon_reference(
            self._reference,
            start_s_m=projection.s_m,
            dt_s=self._dt_s,
            steps=self._parameters.horizon.steps,
            vehicle_width_m=self._vehicle_width_m,
            config=planning,
        )
        request = MpcRequest(
            initial_state=initial_state,
            curvature_1pm=horizon.curvature_1pm[:-1],
            target_speed_mps=horizon.target_speed_mps,
            effective_half_width_m=horizon.effective_half_width_m,
            previous_action=np.asarray(
                (measured_steering, self._previous_acceleration_mps2), dtype=np.float64
            ),
        )
        result = self._solver.solve(request)
        action, mode = self._choose_action(
            result,
            initial_state=initial_state,
            horizon=horizon,
            measured_steering=measured_steering,
        )
        self._previous_acceleration_mps2 = float(action[1])

        predicted = self._predicted_world_points(
            result,
            horizon,
            projection.lateral_error_m,
        )
        self._debug = _DebugState(
            position_m=np.array(position, copy=True),
            projected_point_m=np.array(projection.point_m, copy=True),
            reference_points_m=np.array(horizon.center_m, copy=True),
            predicted_points_m=np.array(predicted, copy=True),
            mode=mode,
            solver_status=result.status,
            target_speed_mps=float(horizon.target_speed_mps[0]),
            lateral_error_m=projection.lateral_error_m,
            heading_error_rad=heading_error,
        )

        output = np.asarray(action, dtype=np.float32)
        if output.shape != (2,) or not np.isfinite(output).all():
            raise RuntimeError("MPC Controller produced an invalid action")
        return output

    def render_callback(self, debug_draw: DebugDraw) -> None:
        """Draw public-geometry reference points and the latest predicted MPC path."""

        if self._debug is None:
            return
        state = self._debug
        stride = self._parameters.debug.prediction_stride
        reference_points = state.reference_points_m[::stride]
        predicted_points = state.predicted_points_m[::stride]
        debug_draw.line(
            state.position_m,
            state.projected_point_m,
            color=(1.0, 0.35, 0.2, 0.9),
            width=1.5,
        )
        debug_draw.points(reference_points, color=(0.15, 0.85, 0.35, 0.75), size=2.0)
        debug_draw.points(predicted_points, color=(0.15, 0.35, 1.0, 0.9), size=3.0)
        debug_draw.text(
            state.position_m + np.asarray((2.0, 2.0), dtype=np.float64),
            (
                f"{state.mode} ({state.solver_status})  "
                f"v*={state.target_speed_mps:.2f} m/s  "
                f"ey={state.lateral_error_m:+.2f} m  "
                f"epsi={state.heading_error_rad:+.2f} rad"
            ),
            color=(0.05, 0.05, 0.05),
        )
