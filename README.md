# 3-DoF planar arm: ROS 2 control, Gazebo pick-and-place, live telemetry GUI

A simulation-only ROS 2 system for a 3-DoF planar arm (links 3.0 / 2.0 / 1.5 m,
joint limits 0–180° / ±120° / ±120°, nothing below y = 0) with a two-finger
gripper. It picks a cylinder off a table and places it elsewhere, in Gazebo
(real physics) and RViz, with a PyQt5/PyQtGraph operator GUI. The provided
`planar_arm.py` is used as a black box and is **not modified**.

## Build and run

Needs ROS 2 Jazzy, Gazebo Harmonic (`ros_gz`), `numpy`, `PyQt5`, `pyqtgraph`.

```bash
cd ~/kineshia_ros2_arm_task
colcon build && source install/setup.bash
ros2 launch planar_arm_control bringup.launch.py     # Gazebo + controller (PID) + GUI
```

After about 10 s, send the test scenario (or use the GUI buttons):

```bash
ros2 action send_goal /pick_and_place arm_interfaces/action/PickAndPlace \
  "{pick: {x: 4.0, y: 2.0}, place: {x: -3.0, y: 3.0}}" --feedback
# edge case: target outside the workspace (clipped to the boundary)
ros2 action send_goal /move_to_target arm_interfaces/action/MoveToTarget \
  "{target: {x: 7.0, y: 3.0}}"
```

Launch arguments: `backend:=gazebo|sim`, `control_mode:=pid|position`,
`gz_gui`, `rviz`, `gui` (`true|false`; `rviz` is off by default, add `rviz:=true`;
the GUI's "Launch RViz" button starts its own second controller, so do not use it with `bringup`). `backend:=sim` runs without physics
(RViz only). Gazebo does not reset the object, so **restart the launch between
attempts**.
If the object is already at the place point when Gazebo opens, an old Gazebo
server survived the last run and the new launch attached to its world. Check
with `pgrep -af "gz sim"` and stop it with `pkill -f "gz sim"` before launching.

## Architecture

```
 gui_node ──action──► controller_node ──► backend ──cmd_pos (bridge)──► Gazebo
     ▲                  │      ▲  (swappable)                              │
     │                  ▼      └───────── PidTracker ◄── /joint_states ◄───┘
     └──── /joint_states, /tracking_error             scene_node ◄── object pose
 RViz ◄── robot_state_publisher ◄── /joint_states
```

The controller plans where the arm *should* be, Gazebo reports where it *is*,
and the PID closes the gap.

**Packages:** `arm_interfaces` (actions), `arm_description` (robot, world,
bridge, parameters), `planar_arm_control` (controller, planning, GUI, launch),
`arm_controllers` (empty skeleton).

**`arm_interfaces`:** `MoveToTarget`, `PickAndPlace` (feedback + cancel),
`PickOrPlace` (served as `/pick`, `/place`), `Gripper`.

**`arm_description`**
- `urdf/common_properties.xacro`: all sizes, limits, masses, Gazebo gains.
  `planar_arm`, `end_effector`, `gazebo`, `ros2_control` xacros hold the arm,
  gripper, Gazebo plugins and optional hardware description;
  `robot.urdf.xacro` includes them.
- `Gazebo/empty_world.sdf`: table, pick cylinder, green place zone.
- `config/bridge.yaml`: Gazebo↔ROS topics. `config/controller.yaml`: every
  controller setting incl. PID gains.
- `rviz/display.rviz`, `launch/gazebo.launch.py`, `launch/display.launch.py`.

**`planar_arm_control`, pure Python (no ROS, testable)**
- `params.py`: all settings in one dataclass.
- `trajectory.py`: `TrapezoidalProfile` (joint space) and `CartesianLineProfile`
  (straight line of the gripper centre via IK), both `sample(t) → (q, qd)`.
- `gripper_kinematics.py`: tool centre 0.35 m past link 3; library IK and own
  heading-aware IK. `arm_model.py`: FK/IK, reach check, clipping.
- `safety.py`: joint limits, ground, clearance. `route_planner.py`: via points
  around the target.
- `sequencer.py`: the step queue `approach → grip → retreat → approach →
  release → retreat`. `pid_tracking.py`: the PID loop.

**`planar_arm_control`, ROS layer**
- `controller_node.py`: serves the five actions (`/move_to_target`,
  `/pick_and_place`, `/gripper`, `/pick`, `/place`) and `/stop`; runs the 50 Hz loop.
- `backends.py`: the hardware-swap point; every backend implements
  `send(stamp, q, qd, gripper)` (`SimBackend`, `GazeboBackend`,
  `Ros2ControlBackend`; a Dynamixel driver is one more class).
- `scene_node.py`: RViz markers (mirrors the Gazebo object pose, or emulates a
  grasp in `sim` mode). `gui_node.py`: the GUI.
- Launch: `bringup` (default entry, includes `pick_place_demo`),
  `pick_place_demo` (full wiring), `controller_rviz`, `ros2_control`.

## Control loop (50 Hz)

```
q_ref, qd_ref = sequencer.step()
q_cmd = clip(q_ref + T_ff·qd_ref + PID(q_ref − q_measured), joint limits)
backend.send(q_cmd)   →  Gazebo
```

Each Gazebo joint has a PD servo that lags the command by about 0.4 s. The
velocity feedforward (`pid_ff_time`, 0.3 s) cancels most of it and the PID the
rest. It publishes `/tracking_error` and `/pid_correction`, skips correction on
stale `/joint_states`, and clamps and clears the integral when idle.

With Gazebo the controller runs on sim time (`use_sim_time`), so a slow
simulation slows the plan instead of letting the arm fall behind it. Without
that, at real-time factor 0.5 the arm knocked the object off the table.

Measured on the test scenario (headless Gazebo), RMS tracking error:

| Joint | position mode | PID mode |
|---|---|---|
| joint1 | 0.050 rad | 0.013 rad |
| joint2 | 0.111 rad | 0.017 rad |
| joint3 | 0.044 rad | 0.005 rad |

In both modes the object landed within about 0.1 m of (−3, 3).

## GUI threading

Qt runs on the main thread, `rclpy.spin` on a second one. ROS callbacks only
emit Qt signals; button clicks go through a thread-safe queue drained by a
timer on the ROS thread, so all rclpy calls stay on that thread. A 33 ms timer
redraws at about 30 fps. The GUI shows joint-angle plots, the live tracking-error
plot with RMS/peak, the arm drawn with its gripper, the end-effector path, the
target (and the clipped point), and Move / Pick / Place / Cancel buttons.


