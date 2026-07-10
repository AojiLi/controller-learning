# Configuration

Challenge, vehicle, and benchmark configuration are intentionally separate. Controller-specific
configuration will live under `controllers/<name>/config.toml` and cannot override these files.

M1 and M2 locked the vehicle timestep, actuator mapping, and formal GPU physics path from measured
CPU/GPU evidence. `track.toml` records the current evidence-backed M3 candidate: fixed arc-length
resolution and capacity, deterministic generator inputs, validation limits, and topology-local race
rules. Track width remains a Level rule in `levels/`; it is intentionally not duplicated in
`track.toml`. M3 must finish the geometry and race gates before these Track values become published
v0.1 benchmark constants.
