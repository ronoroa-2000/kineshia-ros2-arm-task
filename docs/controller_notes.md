# Controller notes: modules, settings and equations

These notes describe `planar_arm_control` (the controller). Angles are in
radians, lengths in metres. `L1, L2, L3 = 3.0, 2.0, 1.5`; `u` is
`ee_center_offset` (0.35). The provided `planar_arm.py` is never modified: it
is only used for forward kinematics, joint limits and the ground check.

## 1. Modules

| File | What it holds | ROS? |
|---|---|---|
| `controller_node.py` | The ROS layer: parameters, topics, `/stop`, the three action servers, the 50 Hz timer. It creates the parts below and calls them. | yes |
| `params.py` | `ControllerParams`: every setting as a plain field (read from the ROS parameters). | no |
| `arm_model.py` | `ArmModel`: the arm, gripper-centre IK (`solve_ik`), `ee_pos`, safety checks. | no |
| `sequencer.py` | `MotionSequencer`: the queue of steps, timing, tasks; `start_move`, `start_gripper`, `start_pick_place`, `step`, `stop`, `cancel`. Clock and status come in as arguments. | no |
| `backends.py` | `SimBackend`, `GazeboBackend`, `make_backend`: how a command leaves the controller. | light |
| `trajectory.py` | `TrapezoidalProfile`, `CartesianLineProfile`: the time laws. | no |
| `gripper_kinematics.py` | `tool_center`, `center_ik` (library IK), `heading_ik` (own IK). | no |
| `safety.py` | `SafetyChecker`, `arm_clearance`: limits, ground, clearance. | no |
| `route_planner.py` | `RoutePlanner`: via-point route around the target. | no |

```
controller_node -> ControllerParams, ArmModel, MotionSequencer, backend
MotionSequencer -> ArmModel, RoutePlanner, trajectory profiles
ArmModel        -> gripper_kinematics, SafetyChecker, planar_arm
RoutePlanner    -> trajectory, SafetyChecker
```

## 2. Settings (config/controller.yaml -> `ControllerParams` field)

| ROS parameter | Field | Unit | Meaning |
|---|---|---|---|
| `publish_rate_hz` | `rate` | Hz | control-loop rate |
| `max_joint_velocity` | `v_max` | rad/s | cruise speed of the leading joint |
| `max_joint_acceleration` | `a_max` | rad/s^2 | accel / decel |
| `approach_distance` | `approach_d` | m | back-off behind a target |
| `max_tip_velocity` | `v_tip` | m/s | gripper speed on straight lines |
| `max_tip_acceleration` | `a_tip` | m/s^2 | gripper acceleration on lines |
| `linear_step` | `line_step` | m | IK sample spacing on lines |
| `ee_center_offset` | `ee_center` | m | link-3 tip -> midpoint between the fingers: this point reaches the target |
| `ee_length` | `ee_length` | m | link-3 tip -> fingertips (ground check) |
| `ik_mode` | `ik_mode` | - | `heading` (own IK, joint 3 moves) or `library` (planar_arm IK, q3 stays 0) |
| `avoid_target` | `avoid` | - | keep the arm clear of the target while approaching |
| `clearance_links` | `clear_links` | m | min distance links -> target |
| `clearance_gripper` | `clear_grip` | m | min distance gripper -> target |
| `gripper_time` | `grip_time` | s | open / close time |
| `gripper_closed_position` | `grip_closed` | m | finger travel when closed |
| `unreachable_policy` | `policy` | - | `clip` or `reject` |
| `initial_joint_angles` | `initial_q` | rad | starting pose |
| `backend`, `control_mode` | `backend`, `control_mode` | - | `sim` / `gazebo`, `position` |

## 3. State of the sequencer (`MotionSequencer`)

| Field | Meaning |
|---|---|
| `q`, `qd` | current joint angles and velocities (what is sent) |
| `gripper` | finger position, 0 = open |
| `active` | `None`, `"move"` or `"gripper"` (what the timer is running) |
| `task` | the running command (progress, result, done event) |
| `step_label`, `action_frac` | name and progress (0..1) of the current step |
| `goal_xy` | target of the current move (for feedback) |

`ArmModel.reach_c = L1 + L2 + L3 + u = 6.85 m`: the largest radius a target can have.

## 4. Trapezoidal profile (`TrapezoidalProfile`)

```
D = max_i |qf_i - q0_i|          distance of the leading joint
if D >= v^2 / a:  t_a = v / a,        T = D / v + t_a,  v_peak = v
else (triangle):  t_a = sqrt(D / a),  T = 2 * t_a,      v_peak = a * t_a

Phase             s(t)                                  speed
0 <= t < t_a      0.5 * a * t^2                         a * t
t_a <= t < T-t_a  0.5*a*t_a^2 + v_peak*(t - t_a)        v_peak
T-t_a <= t <= T   D - 0.5 * a * (T - t)^2               a * (T - t)

All joints start and stop together:
q_i(t)  = q0_i + (qf_i - q0_i) * s(t) / D
qd_i(t) = (qf_i - q0_i) / D * speed(t)
```

## 5. Reach check and IK (`ArmModel.solve_ik`, `heading_ik`, `center_ik`)

The target is where the gripper centre (between the fingers) must end up:
`centre = tip + u * (cos phi, sin phi)`, `phi = q1 + q2 + q3`.

`ik_mode: heading` (`gripper_kinematics.heading_ik`), with the desired gripper
heading `phi`:

```
x_w = x_c - (L3 + u) cos(phi)
y_w = y_c - (L3 + u) sin(phi)
cos(q2) = (x_w^2 + y_w^2 - L1^2 - L2^2) / (2 L1 L2),      q2 = +/- acos(.)
q1 = atan2(y_w, x_w) - atan2(L2 sin q2, L1 + L2 cos q2)
q3 = phi - q1 - q2
```

`phi` is scanned every 0.5 degrees; candidates that break a joint limit or the
ground (links and fingertips) are dropped; the cheapest wins:
`cost = sum|q - q_now| + w * |phi - phi_pref|`, where `phi_pref` is the
outward direction, or the approach heading, which is kept constant on the
approach line so the gripper slides along its own axis. `phi` is then refined
locally.

`ik_mode: library` (`gripper_kinematics.center_ik`): the library solves for the
link-3 tip, so iterate `tip = target - u*dir(phi)`; `IK(tip) -> phi`; repeat
until it settles.

```
r = sqrt(x^2 + y^2);   clipped = r > L1 + L2 + L3 + u
clipped target = target * reach_c / r
```

An IK result is accepted only if: the joint limits hold, `y >= 0` for every
link point and for the fingertips, and `|| target - centre(q) || <= 1e-3`.

## 6. Path validation (`SafetyChecker.validate_profile`)

```
n = floor(T / 0.02) + 1;   t_k = k * T / (n - 1)
every sample q(t_k) must pass the joint limits and y >= 0, else it is rejected
```

## 7. Control loop and gripper (`controller_node._on_timer`, `MotionSequencer.step`)

```
timer period = 1 / publish_rate_hz;   t = now - t0;   a move is done when t >= T
frac = clip((now - t0) / T_g, 0, 1)
gripper(t) = g_from + (g_to - g_from) * frac            (linear open / close)
```

## 8. Approach from behind and pick-and-place (`MotionSequencer`, `RoutePlanner`)

Every target is reached in two steps so the gripper does not knock it:

```
q_goal = IK(target);   heading = q1 + q2 + q3          (direction the tool points)
pre    = target - d * (cos(heading), sin(heading))      d = approach_distance
1) joint-space move to pre (trapezoid)    2) straight line pre -> target
```

Straight line (`CartesianLineProfile`): `p_k = p0 + (p1 - p0) * s_k / L`, with
`s_k` every `linear_step`; `q_k = IK(p_k, guess = q_(k-1))`; rejected on IK
failure, joint limit, `y < 0` or a configuration jump. Timing: a trapezoid on
`s` with `v_tip`, `a_tip`, slowed so that `max_j |dq_j/ds| * v <= max_joint_velocity`.

Route to `pre` (`RoutePlanner.plan_route`): the target is treated as an
obstacle. Every joint-space leg is sampled; the distance from the target to
the links (>= `clearance_links`) and to the gripper axis
(>= `clearance_gripper`) must hold. If the direct leg fails, via points are
tried, cheapest first: one point on a polar grid (radius x azimuth), then the
polar route (pull in at the current azimuth, swing round, go out). The final
straight line is exempt: the gripper must reach the target.

Pick-and-place: approach pick -> close the gripper -> straight back to `pre`
-> approach place -> open the gripper -> straight back to `pre`.

## 9. Interfaces (`arm_interfaces`)

| Interface | Type | Goal / feedback / result |
|---|---|---|
| `/move_to_target` | `MoveToTarget` | goal: target point. feedback: position, distance remaining. result: success, clipped, final position. |
| `/pick_and_place` | `PickAndPlace` | goal: pick + place points. feedback: step, step index / total, progress, gripper position. |
| `/gripper` | `Gripper` | goal: close (true = pick) or open (false = place); no arm motion. |
| `/target_pose` | `PointStamped` | topic: move to a point |
| `/pick_place_request` | `PoseArray` | topic: pick + place |
| `/stop` | `Trigger` | service: halt where the arm is |
| `/controller_status` | `String` (JSON) | latched status |
| `/end_effector` | `PointStamped` | gripper centre |

One command at a time: a goal sent while busy is rejected. Cancel stops the
arm where it is. The node uses a `MultiThreadedExecutor`; a lock protects the
sequencer (timer, topics and action threads).
