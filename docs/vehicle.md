# CPU Four-Wheel Vehicle

M1 establishes the CPU MuJoCo reference plant for the same MJCF that M2 subsequently validated
through MJX-Warp. The M1 evidence itself proves CPU behavior only; see
[MJX-Warp GPU Vehicle](gpu.md) for GPU compatibility, throughput, and numerical agreement.

## Physical Contract

The simulation truth contains one free rigid chassis, four physical cylinder wheels, four wheel
spin joints, two front steering joints, four drive motors, and two steering position actuators. It
intentionally omits suspension, a differential, a gearbox, Pacejka tires, and aerodynamics. Vehicle
dimensions and actuator limits come from `configs/vehicle.toml`; the MJCF loader rejects structural
or numeric mismatches.

Coordinates use SI units. World and body `+x` point forward, `+y` points left, positive yaw is
counter-clockwise, and positive steering turns left. Public state is measured at the rear-axle
reference site. A public action is:

```text
[steering_angle_rad, longitudinal_acceleration_mps2]
```

Steering is angle- and rate-limited. Positive acceleration maps to equal torque at all four wheels.
Negative acceleration applies rotation-opposing brake torque that becomes zero at rest, so braking
cannot silently become reverse drive.

## CPU Timing and Diagnostics

One public control step is 0.05 seconds. `CpuVehicle` advances an integer number of MuJoCo physics
substeps and rejects non-finite state, simulation warnings, or an incorrect time increment.
Diagnostics are sampled after every physics substep, not only at the 20 Hz Controller boundary.
They include wheel contact participation, continuous contact gaps, wheel normal load, penetration,
unexpected collision pairs, roll/pitch, and vertical speed.

## M1 Timestep Protocol

The formal benchmark compares 0.010, 0.005, and 0.002 second physics steps. It runs rest, tilted
drop-and-settle, straight acceleration, mirrored steering, braking, out-of-range action, and a
60-second driven sinusoidal-steering stress scenario three times per candidate. It requires exact
repeatability, left/right symmetry, scenario-specific physical checks, and convergence against the
0.002 second reference. The largest candidate passing every gate is selected for M2.

The chassis is deliberately rigid and has no suspension, so a dynamic wheel constraint may open
briefly even while the wheel remains loaded on average. The protocol therefore does not require four
contacts at every instant. In steering and stress scenarios, every wheel must participate in contact
for at least 75% of physics substeps, no wheel may lose contact continuously for more than 0.1
seconds, and every wheel must carry at least 80% of its static mean load. These gates are combined
with strict penetration, vertical-motion, unexpected-contact, warning, and finite-state limits.
Rest and non-steering scenarios require at least 98% per-wheel contact participation.

The report embeds the benchmark source hash. Changing an action schedule, threshold, convergence
tolerance, or metric implementation therefore changes the protocol identity even if the human
protocol version is unchanged.

## Measured M1 Result

The reviewed M1 report selected 0.005 seconds (200 Hz) as the largest passing CPU candidate:

| Physics step | Result | Reason |
| --- | --- | --- |
| 0.010 s | Fail | Stress penetration and vertical-motion limits; stress convergence |
| 0.005 s | Pass, selected | All absolute, determinism, symmetry, and convergence gates |
| 0.002 s | Pass, reference | All gates |

At 0.005 seconds, the 60-second stress run maintained 79.42% minimum per-wheel physics-substep
contact participation, a 40 ms maximum continuous contact gap, 0.796 mm steady penetration P99,
7.936 m/s mean driven speed, and no MuJoCo warnings or unexpected contacts. The straight scenario
travelled 20.025 m, mirrored steering produced the expected turn signs, and braking finished at
0.085 m/s without reversing. These are CPU reference measurements, not GPU performance claims.

The complete machine-readable evidence is `benchmarks/v0.1/m1_cpu_report.json`.
