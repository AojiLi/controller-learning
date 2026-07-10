# PID Controller

This example uses only the public observation, info, configuration, and `DebugDraw` interfaces.
It reconstructs a local centerline reference from the observed fixed-capacity arrays; it never
imports Race Core, TrackPool, simulator, or hidden progress state.

Longitudinal control previews centerline curvature, applies a braking envelope, and tracks the
resulting target speed with an anti-windup PID. Lateral control is a cascade: an outer lateral PID
requests a bounded heading correction, while an inner heading PD plus kinematic curvature
feedforward requests steering. The final command respects the public steering-angle, steering-rate,
acceleration, and deceleration limits.

The values in `config.toml` are one Controller-wide parameter set. They are not changed per Track.
