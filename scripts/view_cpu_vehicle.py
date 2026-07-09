"""Open MuJoCo's passive viewer for the M1 CPU four-wheel vehicle."""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import mujoco.viewer

from controller_learning.config import load_vehicle_config
from controller_learning.physics import CpuVehicle


def _demo_action(time_s: float) -> tuple[float, float]:
    if time_s < 1.0:
        return (0.0, 0.0)
    if time_s < 4.0:
        return (0.0, 1.5)
    if time_s < 8.0:
        return (0.15, 0.0)
    if time_s < 10.0:
        return (0.0, -3.0)
    return (0.0, 0.0)


def main() -> None:
    """Run a real-time rest or drive demonstration until timeout or viewer close."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--duration", type=float, default=12.0, help="Maximum simulated seconds")
    parser.add_argument(
        "--scenario",
        choices=("rest", "demo"),
        default="demo",
        help="Viewer action sequence",
    )
    args = parser.parse_args()
    if args.duration <= 0.0:
        parser.error("--duration must be positive")

    project_root = Path(__file__).resolve().parents[1]
    config = load_vehicle_config(project_root / "configs" / "vehicle.toml")
    vehicle = CpuVehicle(config)
    wall_start = time.perf_counter()
    with mujoco.viewer.launch_passive(vehicle.model, vehicle.data) as viewer:
        while viewer.is_running() and vehicle.data.time < args.duration:
            step_start = time.perf_counter()
            action = _demo_action(vehicle.data.time) if args.scenario == "demo" else (0.0, 0.0)
            vehicle.step(action)
            viewer.sync()
            elapsed = time.perf_counter() - step_start
            time.sleep(max(0.0, config.simulation.control_dt_s - elapsed))
    wall_elapsed = time.perf_counter() - wall_start
    print(f"simulated {vehicle.data.time:.2f} s in {wall_elapsed:.2f} s")


if __name__ == "__main__":
    main()
