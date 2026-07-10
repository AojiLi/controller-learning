# MPC Controller

This example is a warm-started nonlinear model predictive Controller. The simulation truth remains
the physical four-wheel MuJoCo/MJX-Warp car; the Controller deliberately uses a smaller kinematic
car model in Frenet error coordinates as its prediction model.

The Controller derives centerline projection, curvature, speed references, and usable Track width
only from the public observation. Its state is lateral error, heading error, and forward speed. Its
inputs are the same steering-angle and longitudinal-acceleration actions used by every Controller.
The 20-step horizon covers one second at the public 20 Hz control rate.

CasADi constructs one fixed nonlinear program per episode and IPOPT solves it with new numerical
parameters at every control step. RK4 dynamics and hard speed, action, steering-rate, and future
Track-boundary constraints are enforced. A successful primal solution is shifted to warm-start the
next call. If IPOPT fails or reaches its wall-time limit, the Controller first consumes the
remaining controls from the last feasible plan, then uses a deterministic curvature-feedforward
feedback action derived from the same public geometry.

`DebugDraw` shows the sampled centerline reference and the most recent predicted path. Headless
evaluation does not invoke rendering. The included values are one Controller-wide configuration;
they are not changed per Track. Completion and timing claims require the separate M6 formal
benchmark and are not implied by this implementation alone.
