# CPU Four-Wheel Vehicle

M1 establishes the CPU MuJoCo reference plant used to debug the same MJCF that M2 will attempt to
run through MJX-Warp. Passing M1 proves CPU behavior only; it does not prove GPU compatibility,
throughput, or numerical agreement.

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
