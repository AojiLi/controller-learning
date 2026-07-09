"""Vehicle physics backends."""

from controller_learning.physics.actuation import (
    AppliedVehicleAction,
    VehicleActionError,
    map_vehicle_action,
    wheel_torques_for_acceleration,
)
from controller_learning.physics.cpu_reference import (
    ContactMetrics,
    CpuVehicle,
    StepDiagnostics,
    VehicleSimulationError,
    VehicleState,
)
from controller_learning.physics.model import (
    VehicleModelError,
    VehicleModelIndices,
    load_vehicle_model,
    validate_vehicle_model,
    vehicle_model_indices,
)

__all__ = [
    "AppliedVehicleAction",
    "ContactMetrics",
    "CpuVehicle",
    "StepDiagnostics",
    "VehicleActionError",
    "VehicleModelError",
    "VehicleModelIndices",
    "VehicleSimulationError",
    "VehicleState",
    "load_vehicle_model",
    "map_vehicle_action",
    "validate_vehicle_model",
    "vehicle_model_indices",
    "wheel_torques_for_acceleration",
]
