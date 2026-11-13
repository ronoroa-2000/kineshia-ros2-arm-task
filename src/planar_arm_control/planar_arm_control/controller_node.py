#!/usr/bin/env python3
"""
controller_node.py - ROS 2 controller for the 3-DoF planar arm.

Pipeline:  target (x, y) -> IK -> validate -> trapezoidal joint trajectory
           -> fixed-rate timer samples it -> backend sends the command.

The planner (TrapezoidalProfile) knows nothing about ROS or hardware.
The backend (SimBackend / GazeboBackend / future Dynamixel) only knows how
to deliver a command, so it can be swapped without touching the planner.
"""
# =============================================================================
# REFERENCE NOTES  (variables and equations used in this file)
# =============================================================================
#
# 1. PARAMETERS  (from config/controller.yaml)
# -----------------------------------------------------------------------------
# publish_rate_hz          self._rate        Hz      control-loop rate
# max_joint_velocity       self._v_max       rad/s   cruise speed (v)
# max_joint_acceleration   self._a_max       rad/s^2 accel/decel (a)
# approach_height          self._approach_h  m       y-offset above pick/place
# gripper_time             self._grip_time   s       open/close time (T_g)
# gripper_closed_position  self._grip_closed m       finger travel when closed
# unreachable_policy       self._policy      -       "clip" or "reject"
# initial_joint_angles     self._q (start)   rad     starting pose [q1 q2 q3]
# backend / control_mode   -                 -       "sim"|"gazebo" / position
#
# 2. NODE STATE
# -----------------------------------------------------------------------------
# self._q, self._qd   current joint angles (rad) and velocities (rad/s)
# self._gripper       current finger position (0 = open)
# self._queue         pending actions (moves / gripper steps)
# self._active        None | "move" | "gripper"  (what the timer is running)
# self._profile       TrapezoidalProfile of the move in progress
# self._t0            start time (s) of the current action
# self._reach         L1 + L2 + L3 = 6.5 m, max workspace radius
#
# 3. TRAPEZOIDAL PROFILE  (TrapezoidalProfile)
# -----------------------------------------------------------------------------
#   D = max_i |qf_i - q0_i|          distance of the leading joint
#   if D >= v^2 / a:  t_a = v / a,        T = D / v + t_a,  v_peak = v
#   else (triangle):  t_a = sqrt(D / a),  T = 2 * t_a,      v_peak = a * t_a
#
#   Phase             s(t)                                  speed
#   0 <= t < t_a      0.5 * a * t^2                         a * t
#   t_a <= t < T-t_a  0.5*a*t_a^2 + v_peak*(t - t_a)        v_peak
#   T-t_a <= t <= T   D - 0.5 * a * (T - t)^2               a * (T - t)
#
#   All joints start and stop together:
#   q_i(t)  = q0_i + (qf_i - q0_i) * s(t) / D
#   qd_i(t) = (qf_i - q0_i) / D * speed(t)
#
# 4. REACH CHECK AND IK ACCEPTANCE  (_solve_ik)
# -----------------------------------------------------------------------------
#   r = sqrt(x^2 + y^2);  clipped = r > L1 + L2 + L3
#   clipped target = target * reach / r   (done by planar_arm.reachable_target)
#   IK result is accepted only if: joint limits hold, y >= 0 for every link
#   point, and || target - FK(q) || <= 1e-3.
#
# 5. PATH VALIDATION  (_validate_profile)
# -----------------------------------------------------------------------------
#   n = floor(T / 0.02) + 1;  t_k = k * T / (n - 1)
#   every sample q(t_k) must pass the joint limits and y >= 0, else rejected.
#
# 6. CONTROL LOOP AND GRIPPER  (_on_timer)
# -----------------------------------------------------------------------------
#   timer period = 1 / publish_rate_hz;  t = now - t0;  move done when t >= T
#   frac = clip((now - t0) / T_g, 0, 1)
#   gripper(t) = g_from + (g_to - g_from) * frac       (linear open/close)
#
# 7. PICK-AND-PLACE WAYPOINTS  (_on_pick_place)
# -----------------------------------------------------------------------------
#   above_pick  = (x_pick,  y_pick  + h);  above_place = (x_place, y_place + h)
#   above_pick -> pick -> close -> above_pick -> above_place -> place ->
#   open -> above_place.   Rejected if pick or place is farther than reach.
#
# 8. ACTIONS  (interfaces in the arm_interfaces package)
# -----------------------------------------------------------------------------
#   /move_to_target  MoveToTarget : goal = target point; feedback = position,
#                    distance_remaining; result = success, clipped, final pos
#   /pick_and_place  PickAndPlace : goal = pick + place points; feedback =
#                    step, step_index/total_steps, progress, gripper position
#   /gripper         Gripper      : goal = close (true: pick) or open (false:
#                    place); gripper only, no arm motion
#   "pick" step closes the gripper, "place" (release) step opens it.
#   One command at a time: a goal sent while busy is rejected. Cancel stops the
#   arm where it is. Shared state is protected by self._lock (timer, topics and
#   action threads); the node runs in a MultiThreadedExecutor.
# =============================================================================


import json
import threading
from abc import ABC, abstractmethod
from types import SimpleNamespace

import numpy as np
import rclpy
from arm_interfaces.action import Gripper, MoveToTarget, PickAndPlace
from geometry_msgs.msg import Point, PointStamped, PoseArray
from rclpy.action import ActionServer, CancelResponse, GoalResponse
from rclpy.callback_groups import ReentrantCallbackGroup
from rclpy.executors import ExternalShutdownException, MultiThreadedExecutor
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, QoSProfile
from sensor_msgs.msg import JointState
from std_msgs.msg import Float64, String
from std_srvs.srv import Trigger

# The provided kinematics library - do not modify it.
from planar_arm_control.planar_arm import PlanarArm

LINK_LENGTHS = [3.0, 2.0, 1.5]
ARM_JOINTS = ["joint1", "joint2", "joint3"]
FINGER_JOINTS = ["ee_finger_left_joint", "ee_finger_right_joint"]
VALIDATION_DT = 0.02  # s, spacing of samples used to check a planned path
FEEDBACK_PERIOD = 0.2  # s between action feedback messages (5 Hz)


# ======================================================================
# Planner: synchronised trapezoidal velocity profile (pure Python/NumPy)
# ======================================================================
class TrapezoidalProfile:
    """
    All joints start and stop together. The joint with the largest
    distance D drives a trapezoidal (or triangular, if D is short)
    velocity profile; every other joint is scaled by dq_i / D, so the
    straight line in joint space is followed and no joint exceeds the
    limits v_max / a_max.
    """

    def __init__(self, q0, qf, v_max, a_max):
        self.q0 = np.asarray(q0, dtype=float)
        self.dq = np.asarray(qf, dtype=float) - self.q0
        self.D = float(np.max(np.abs(self.dq)))
        self.a = float(a_max)

        if self.D < 1e-9:                      # nothing to do
            self.T = 0.0
            self.ta = 0.0
            self.v_peak = 0.0
        elif self.D >= v_max ** 2 / a_max:     # trapezoid: accel, cruise, decel
            self.v_peak = float(v_max)
            self.ta = v_max / a_max
            self.T = self.D / v_max + self.ta
        else:                                  # triangle: never reaches v_max
            self.ta = float(np.sqrt(self.D / a_max))
            self.v_peak = a_max * self.ta
            self.T = 2.0 * self.ta

    def _distance_and_speed(self, t):
        """Distance travelled and speed of the leading joint at time t."""
        if t < self.ta:                        # accelerating
            return 0.5 * self.a * t ** 2, self.a * t
        if t < self.T - self.ta:               # cruising
            return (0.5 * self.a * self.ta ** 2
                    + self.v_peak * (t - self.ta)), self.v_peak
        td = self.T - t                        # decelerating
        return self.D - 0.5 * self.a * td ** 2, self.a * td

    def sample(self, t):
        """Return (q, qd) as arrays at time t seconds from the start."""
        if self.D < 1e-9:
            return self.q0.copy(), np.zeros_like(self.q0)
        t = min(max(t, 0.0), self.T)
        s, v = self._distance_and_speed(t)
        frac = s / self.D
        return self.q0 + self.dq * frac, self.dq / self.D * v


# ======================================================================
# Command backends (swap these for hardware later)
# ======================================================================
class ArmBackend(ABC):
    @abstractmethod
    def send(self, stamp, q, qd, gripper):
        """Deliver one command. q/qd: 3 arm joints, gripper: finger position."""


class SimBackend(ArmBackend):
    """No physics: just publish the commanded state as /joint_states."""

    def __init__(self, node):
        self._pub = node.create_publisher(JointState, "/joint_states", 10)

    def send(self, stamp, q, qd, gripper):
        msg = JointState()
        msg.header.stamp = stamp
        msg.name = ARM_JOINTS + FINGER_JOINTS
        msg.position = [float(v) for v in q] + [float(gripper)] * 2
        msg.velocity = [float(v) for v in qd] + [0.0, 0.0]
        self._pub.publish(msg)


class GazeboBackend(ArmBackend):
    """Publish position commands bridged to Gazebo (see config/bridge.yaml)."""

    def __init__(self, node):
        self._pubs = {
            name: node.create_publisher(
                Float64, f"/planar_arm/{name}/cmd_pos", 10)
            for name in ARM_JOINTS + FINGER_JOINTS
        }

    def send(self, stamp, q, qd, gripper):
        values = [float(v) for v in q] + [float(gripper)] * 2
        for pub, value in zip(self._pubs.values(), values):
            pub.publish(Float64(data=value))


# Future: class DynamixelBackend(ArmBackend): write goal positions over the bus.


# ======================================================================
# Controller node
# ======================================================================
class ControllerNode(Node):
    def __init__(self):
        super().__init__("controller_node")
        self.arm = PlanarArm(LINK_LENGTHS)
        self._limits = self.arm.joint_limits
        self._reach = sum(self.arm.link_lengths)

        # ---- parameters ------------------------------------------------
        self.declare_parameter("publish_rate_hz", 50.0)
        self.declare_parameter("control_mode", "position")  # velocity: stretch goal
        self.declare_parameter("backend", "sim")            # sim | gazebo
        self.declare_parameter("max_joint_velocity", 1.0)   # rad/s
        self.declare_parameter("max_joint_acceleration", 2.0)  # rad/s^2
        self.declare_parameter("approach_height", 1.0)      # y offset above pick/place
        self.declare_parameter("gripper_time", 0.5)         # s to open/close
        self.declare_parameter("gripper_closed_position", 0.1)  # m, finger travel
        self.declare_parameter("unreachable_policy", "clip")    # clip | reject
        self.declare_parameter("initial_joint_angles", [0.0, 0.0, 0.0])

        p = self.get_parameter
        self._rate = float(p("publish_rate_hz").value)
        self._v_max = max(float(p("max_joint_velocity").value), 1e-3)
        self._a_max = max(float(p("max_joint_acceleration").value), 1e-3)
        self._approach_h = float(p("approach_height").value)
        self._grip_time = max(float(p("gripper_time").value), 1e-3)
        self._grip_closed = float(p("gripper_closed_position").value)
        self._policy = str(p("unreachable_policy").value)

        mode = str(p("control_mode").value)
        if mode != "position":
            self.get_logger().warn(
                f"control_mode '{mode}' not implemented yet; using position")

        # ---- state -------------------------------------------------------
        q0 = np.array(p("initial_joint_angles").value, dtype=float)
        if not (self.arm.within_joint_limits(q0) and self.arm.arm_above_base(q0)):
            self.get_logger().warn("initial_joint_angles invalid; using zeros")
            q0 = np.zeros(3)
        self._q = q0
        self._qd = np.zeros(3)
        self._gripper = 0.0          # 0 = open
        self._queue = []             # pending actions
        self._active = None          # None | "move" | "gripper"
        self._profile = None
        self._t0 = 0.0
        self._grip_from = 0.0
        self._grip_to = 0.0
        self._step_label = ""
        self._last_info = {}

        # ---- shared-state helpers for the action servers --------------------
        # Timer, topic callbacks and action threads all touch the state above,
        # so every one of them takes this lock.
        self._lock = threading.RLock()
        self._task = None            # running command (topic or action goal)
        self._reserved = False       # action goal accepted, not started yet
        self._action_frac = 0.0      # progress 0..1 of the current queue item
        self._goal_xy = None         # target of the current move (feedback)
        self._goal_tasks = {}        # action goal id -> its task
        self._last_fb = 0.0

        # ---- backend -----------------------------------------------------
        backend = str(p("backend").value)
        if backend == "gazebo":
            self._backend = GazeboBackend(self)
        else:
            if backend != "sim":
                self.get_logger().warn(f"unknown backend '{backend}'; using sim")
            self._backend = SimBackend(self)

        # ---- ROS interfaces ----------------------------------------------
        self.create_subscription(PointStamped, "/target_pose", self._on_target, 10)
        self.create_subscription(
            PoseArray, "/pick_place_request", self._on_pick_place, 10)
        self.create_service(Trigger, "/stop", self._on_stop)

        # Action servers. A reentrant group lets the execute callbacks wait
        # (in their own threads) without blocking the control timer.
        action_group = ReentrantCallbackGroup()
        self._action_servers = [
            ActionServer(
                self, action_type, name,
                execute_callback=self._execute_action,
                goal_callback=self._on_goal,
                cancel_callback=self._on_cancel,
                handle_accepted_callback=self._on_accepted,
                callback_group=action_group)
            for action_type, name in ((MoveToTarget, "/move_to_target"),
                                      (PickAndPlace, "/pick_and_place"),
                                      (Gripper, "/gripper"))
        ]
        self._ee_pub = self.create_publisher(PointStamped, "/end_effector", 10)
        latched = QoSProfile(depth=1, durability=DurabilityPolicy.TRANSIENT_LOCAL)
        self._status_pub = self.create_publisher(String, "/controller_status", latched)

        self.create_timer(1.0 / self._rate, self._on_timer)
        self._publish_status("idle", "ready")
        self.get_logger().info(
            f"controller_node started (backend={backend}, {self._rate:.0f} Hz)")

    # ------------------------------------------------------------------
    # helpers
    # ------------------------------------------------------------------
    def _now_s(self):
        return self.get_clock().now().nanoseconds * 1e-9

    def _clip(self, q):
        return np.clip(q, self._limits[:, 0], self._limits[:, 1])

    def _publish_status(self, state, message, info=None):
        if info is not None:
            self._last_info = info
        payload = {"state": state, "step": self._step_label,
                   "message": message, **self._last_info}
        self._status_pub.publish(String(data=json.dumps(payload)))

    # ------------------------------------------------------------------
    # inverse kinematics + validation
    # ------------------------------------------------------------------
    def _solve_ik(self, xy, allow_clip):
        """Return (q_goal or None, info). info always describes the outcome."""
        target = np.array(xy, dtype=float)
        clipped = bool(np.linalg.norm(target) > self._reach + 1e-9)
        info = {"target": target.tolist(), "clipped": clipped,
                "used_target": target.tolist()}

        if clipped:
            if not allow_clip:
                info["message"] = (f"target {tuple(target)} is outside the "
                                   f"workspace (reach {self._reach:.2f})")
                return None, info
            target = np.asarray(self.arm.reachable_target(target), dtype=float)
            info["used_target"] = target.tolist()

        q = np.array(
            self.arm.inverse_kinematics(target, initial_guess=self._q), dtype=float)

        if not self.arm.within_joint_limits(q):
            info["message"] = "IK solution violates joint limits"
            return None, info
        if not self.arm.arm_above_base(q):
            info["message"] = "IK solution puts the arm below y = 0"
            return None, info
        err = float(np.linalg.norm(target - np.asarray(self.arm.end_effector(q))))
        if err > 1e-3:
            info["message"] = f"no valid IK solution (error {err:.3f})"
            return None, info

        info["message"] = ("target clipped to workspace boundary"
                           if clipped else "ok")
        return q, info

    def _validate_profile(self, profile):
        """Check every sample of the path against limits and the ground."""
        n = max(2, int(profile.T / VALIDATION_DT) + 1)
        for t in np.linspace(0.0, profile.T, n):
            q = self._clip(profile.sample(t)[0])
            if not self.arm.within_joint_limits(q):
                return False, f"joint limit violated at t={t:.2f}s"
            if not self.arm.arm_above_base(q):
                return False, f"arm would go below y = 0 at t={t:.2f}s"
        return True, ""

    # ------------------------------------------------------------------
    # task bookkeeping (one running command at a time)
    # ------------------------------------------------------------------
    def _busy(self):
        return self._active is not None or self._reserved

    def _new_task(self, kind, goal_handle=None, total_steps=1,
                  done_message=""):
        """Start tracking a command. kind: "move" | "pick_place" | "gripper"."""
        self._last_info = {}
        self._task = SimpleNamespace(
            kind=kind, goal_handle=goal_handle, done=threading.Event(),
            success=False, message="", failed_step="", clipped=False,
            step_index=0, total_steps=total_steps, done_message=done_message)
        return self._task

    def _finish_task(self, success, message, failed_step=""):
        """Mark the running command as finished (wakes the action thread)."""
        task = self._task
        if task is None:
            return
        task.success = success
        task.message = message
        task.failed_step = failed_step
        task.clipped = bool(self._last_info.get("clipped", False))
        task.done.set()
        self._task = None

    def _halt_motion(self):
        self._queue.clear()
        self._profile = None
        self._active = None
        self._qd = np.zeros(3)
        self._step_label = ""
        self._goal_xy = None

    # ------------------------------------------------------------------
    # action queue (single moves and pick-and-place share this)
    # ------------------------------------------------------------------
    def _begin_next_action(self):
        if not self._queue:
            task = self._task
            if task is not None and task.done_message:
                message = task.done_message
            elif task is not None and task.kind == "pick_place":
                message = "pick and place complete"
            elif self._last_info.get("clipped"):
                message = ("target outside the workspace; moved to the "
                           "nearest boundary point")
            else:
                message = "target reached"
            self._active = None
            self._step_label = ""
            self._goal_xy = None
            self._publish_status("idle", "done")
            self._finish_task(True, message)
            return

        action = self._queue.pop(0)
        self._step_label = action["label"]
        self._action_frac = 0.0
        if self._task is not None:
            self._task.step_index += 1

        if action["type"] == "gripper":
            self._grip_from = self._gripper
            self._grip_to = action["value"]
            self._t0 = self._now_s()
            self._active = "gripper"
            self._publish_status("gripping", action["label"])
            return

        # move
        q_goal, info = self._solve_ik(action["xy"], action["allow_clip"])
        if q_goal is None:
            self._abort(info["message"], info)
            return
        profile = TrapezoidalProfile(self._q, q_goal, self._v_max, self._a_max)
        ok, why = self._validate_profile(profile)
        if not ok:
            self._abort(why, info)
            return
        self._goal_xy = info["used_target"]
        self._profile = profile
        self._t0 = self._now_s()
        self._active = "move"
        self._publish_status("moving", info["message"], info)

    def _abort(self, message, info=None):
        self.get_logger().warn(f"aborted: {message}")
        step = self._step_label
        self._queue.clear()
        self._profile = None
        self._active = None
        self._qd = np.zeros(3)
        self._publish_status("error", message, info)
        self._finish_task(False, message, step)

    def _reject_request(self, step, message, info):
        """A request that is invalid before any motion starts."""
        self.get_logger().warn(f"request rejected: {message}")
        self._step_label = step
        self._publish_status("error", message, info)
        self._finish_task(False, message, step)

    # ------------------------------------------------------------------
    # starting commands (call with self._lock held)
    # ------------------------------------------------------------------
    def _start_move(self, xy, goal_handle=None):
        task = self._new_task("move", goal_handle)
        self._queue = [{
            "type": "move", "label": "move_to_target",
            "xy": (float(xy[0]), float(xy[1])),
            "allow_clip": self._policy == "clip",
        }]
        self._begin_next_action()
        return task

    def _start_gripper(self, close, goal_handle=None):
        """Gripper only: close (pick) or open (place), no arm motion."""
        task = self._new_task(
            "gripper", goal_handle,
            done_message="gripper closed" if close else "gripper opened")
        self._queue = [{
            "type": "gripper",
            "label": "grip" if close else "release",
            "value": self._grip_closed if close else 0.0,
        }]
        self._begin_next_action()
        return task

    def _build_pick_place(self, pick, place):
        """Return (queue, bad_step, message, info); queue is None on error."""
        for name, pt in (("pick", pick), ("place", place)):
            if np.hypot(*pt) > self._reach:      # never pick/place at a clipped point
                info = {"target": list(pt), "clipped": True,
                        "used_target": np.asarray(
                            self.arm.reachable_target(pt)).tolist()}
                message = (f"{name} target {pt} is outside the workspace "
                           f"(reach {self._reach:.2f})")
                return None, name, message, info

        h = self._approach_h
        above_pick = (pick[0], pick[1] + h)
        above_place = (place[0], place[1] + h)

        def move(label, xy):
            return {"type": "move", "label": label, "xy": xy, "allow_clip": False}

        def grip(label, value):
            return {"type": "gripper", "label": label, "value": value}

        queue = [
            move("above_pick", above_pick),
            move("pick_descend", pick),
            grip("grip", self._grip_closed),     # pick: close the gripper
            move("pick_lift", above_pick),
            move("transfer", above_place),
            move("place_descend", place),
            grip("release", 0.0),                # place: open the gripper
            move("retreat", above_place),
        ]
        return queue, "", "", None

    def _start_pick_place(self, pick, place, goal_handle=None):
        pick = (float(pick[0]), float(pick[1]))
        place = (float(place[0]), float(place[1]))
        queue, step, message, info = self._build_pick_place(pick, place)
        task = self._new_task("pick_place", goal_handle,
                              total_steps=len(queue) if queue else 8)
        if queue is None:
            self._reject_request(step, message, info)
            return task
        self._queue = queue
        self._begin_next_action()
        return task

    # ------------------------------------------------------------------
    # topic / service callbacks
    # ------------------------------------------------------------------
    def _on_target(self, msg):
        with self._lock:
            if self._busy():
                self._publish_status("error", "busy: send /stop first")
                return
            self._start_move((msg.point.x, msg.point.y))

    def _on_pick_place(self, msg):
        with self._lock:
            if self._busy():
                self._publish_status("error", "busy: send /stop first")
                return
            if len(msg.poses) != 2:
                self._publish_status(
                    "error", "need exactly 2 poses: [pick, place]")
                return
            pick = (msg.poses[0].position.x, msg.poses[0].position.y)
            place = (msg.poses[1].position.x, msg.poses[1].position.y)
            self._start_pick_place(pick, place)

    def _on_stop(self, request, response):
        with self._lock:
            self._halt_motion()
            self._finish_task(False, "stopped by operator")
            self._publish_status("idle", "stopped by operator")
        response.success = True
        response.message = "stopped"
        return response

    # ------------------------------------------------------------------
    # action server callbacks (/move_to_target and /pick_and_place)
    # ------------------------------------------------------------------
    def _on_goal(self, goal_request):
        """Accept only when idle; the reservation closes the race window."""
        with self._lock:
            if self._busy():
                self.get_logger().warn("action goal rejected: controller busy")
                return GoalResponse.REJECT
            self._reserved = True
        return GoalResponse.ACCEPT

    def _on_cancel(self, goal_handle):
        return CancelResponse.ACCEPT

    def _on_accepted(self, goal_handle):
        request = goal_handle.request
        with self._lock:
            self._reserved = False
            if isinstance(request, MoveToTarget.Goal):
                task = self._start_move(
                    (request.target.x, request.target.y), goal_handle)
            elif isinstance(request, Gripper.Goal):
                task = self._start_gripper(request.close, goal_handle)
            else:
                task = self._start_pick_place(
                    (request.pick.x, request.pick.y),
                    (request.place.x, request.place.y), goal_handle)
            self._goal_tasks[bytes(goal_handle.goal_id.uuid)] = task
        goal_handle.execute()

    def _execute_action(self, goal_handle):
        """Runs in its own thread: waits for the control timer to finish."""
        with self._lock:
            task = self._goal_tasks.pop(bytes(goal_handle.goal_id.uuid), None)
        request = goal_handle.request
        if isinstance(request, MoveToTarget.Goal):
            result = MoveToTarget.Result()
        elif isinstance(request, Gripper.Goal):
            result = Gripper.Result()
        else:
            result = PickAndPlace.Result()

        if task is None:
            self._fill_result(result, False, "internal error: no task")
            goal_handle.abort()
            return result

        while rclpy.ok():
            if task.done.wait(timeout=0.05):         # finished by the timer
                break
            if goal_handle.is_cancel_requested:
                with self._lock:
                    if self._task is task:
                        self._halt_motion()
                        self._publish_status("idle", "canceled by client")
                        self._finish_task(False, "canceled")
                self._fill_result(result, False, "canceled")
                goal_handle.canceled()
                return result

        if not task.done.is_set():                   # node is shutting down
            self._fill_result(result, False, "node shutting down")
            goal_handle.abort()
            return result

        self._fill_result(result, task.success, task.message,
                          task.clipped, task.failed_step)
        if task.success:
            goal_handle.succeed()
        else:
            goal_handle.abort()
        return result

    def _fill_result(self, result, success, message, clipped=False,
                     failed_step=""):
        ee = self.arm.end_effector(self._q)
        result.success = success
        result.message = message
        if hasattr(result, "final_position"):
            result.final_position = Point(x=float(ee[0]), y=float(ee[1]))
        if hasattr(result, "gripper_position"):
            result.gripper_position = float(self._gripper)
        if hasattr(result, "clipped"):
            result.clipped = clipped
        if hasattr(result, "failed_step"):
            result.failed_step = failed_step

    def _publish_feedback(self, ee, now):
        """Throttled feedback for the running action goal (lock held)."""
        task = self._task
        if task is None or task.goal_handle is None:
            return
        if now - self._last_fb < FEEDBACK_PERIOD:
            return
        self._last_fb = now
        position = Point(x=float(ee[0]), y=float(ee[1]))

        if task.kind == "move":
            feedback = MoveToTarget.Feedback()
            feedback.state = "moving" if self._active else "idle"
            feedback.current_position = position
            if self._goal_xy is not None:
                feedback.distance_remaining = float(
                    np.hypot(self._goal_xy[0] - ee[0],
                             self._goal_xy[1] - ee[1]))
        elif task.kind == "gripper":
            feedback = Gripper.Feedback()
            feedback.gripper_position = float(self._gripper)
            feedback.progress = float(self._action_frac)
        else:
            feedback = PickAndPlace.Feedback()
            feedback.step = self._step_label
            feedback.step_index = int(task.step_index)
            feedback.total_steps = int(task.total_steps)
            done_steps = max(task.step_index - 1, 0)
            feedback.progress = float(min(
                (done_steps + self._action_frac) / max(task.total_steps, 1),
                1.0))
            feedback.current_position = position
            feedback.gripper_position = float(self._gripper)
        task.goal_handle.publish_feedback(feedback)

    # ------------------------------------------------------------------
    # fixed-rate control loop
    # ------------------------------------------------------------------
    def _on_timer(self):
        with self._lock:
            now = self._now_s()

            if self._active == "move":
                t = now - self._t0
                q, qd = self._profile.sample(t)
                self._q = self._clip(q)
                self._qd = qd
                duration = self._profile.T
                self._action_frac = min(t / duration, 1.0) if duration > 0 else 1.0
                if t >= duration:
                    self._qd = np.zeros(3)
                    self._profile = None
                    self._active = None
                    self._begin_next_action()

            elif self._active == "gripper":
                frac = min(max((now - self._t0) / self._grip_time, 0.0), 1.0)
                self._action_frac = frac
                self._gripper = self._grip_from + (self._grip_to - self._grip_from) * frac
                if frac >= 1.0:
                    self._active = None
                    self._begin_next_action()

            stamp = self.get_clock().now().to_msg()
            self._backend.send(stamp, self._q, self._qd, self._gripper)

            ee = self.arm.end_effector(self._q)
            msg = PointStamped()
            msg.header.stamp = stamp
            msg.header.frame_id = "world"
            msg.point.x, msg.point.y = float(ee[0]), float(ee[1])
            self._ee_pub.publish(msg)

            self._publish_feedback(ee, now)


def main(args=None):
    rclpy.init(args=args)
    node = ControllerNode()
    # Several threads: action execute callbacks wait while the timer keeps running.
    executor = MultiThreadedExecutor(num_threads=4)
    executor.add_node(node)
    try:
        executor.spin()
    except (KeyboardInterrupt, ExternalShutdownException):
        pass
    finally:
        executor.shutdown()
        node.destroy_node()
        rclpy.try_shutdown()


if __name__ == "__main__":
    main()


