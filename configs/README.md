# Configuration

Challenge, vehicle, and benchmark configuration are intentionally separate. Controller-specific
configuration will live under `controllers/<name>/config.toml` and cannot override these files.

The numeric vehicle and simulation values in M0 are explicit candidates. M1 and M2 must measure
and update them before they become the v0.1 physical benchmark constants.
