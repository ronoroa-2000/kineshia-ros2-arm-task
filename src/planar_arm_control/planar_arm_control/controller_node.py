#!/usr/bin/env python3
"""
controller_node.py - ROS 2 controller for the 3-DoF planar arm.

This file is only the ROS layer: parameters, topics, services, the three
action servers and the fixed-rate timer. Everything else lives in modules
that are created here and called from here:

    params.py        ControllerParams   the settings (ROS parameters)
    arm_model.py     ArmModel           arm, gripper-centre IK, safety checks
    sequencer.py     MotionSequencer    queue of steps, timing, tasks
    backends.py      make_backend       how a command leaves (sim / Gazebo)

and, used by those modules:

    trajectory.py    TrapezoidalProfile, CartesianLineProfile (time laws)
    gripper_kinematics.py  gripper-centre kinematics, the two IK modes
    safety.py        joint-limit, ground and clearance checks
    route_planner.py via-point route around the target

Variables, equations and the design are described in docs/controller_notes.md.
"""

import json
import math
import threading
from functools import partial

import rclpy
from arm_interfaces.action import (
    Gripper, MoveToTarget, PickAndPlace, PickOrPlace)
from geometry_msgs.msg import Point, PointStamped, PoseArray
from rclpy.action import ActionServer, CancelResponse, GoalResponse
from rclpy.callback_groups import ReentrantCallbackGroup
from rclpy.executors import ExternalShutdownException, MultiThreadedExecutor
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, QoSProfile
from std_msgs.msg import String
from std_srvs.srv import Trigger

from planar_arm_control.arm_model import ArmModel
from planar_arm_control.backends import make_backend
from planar_arm_control.params import ControllerParams
from planar_arm_control.sequencer import MotionSequencer

FEEDBACK_PERIOD = 0.2  # s between action feedback messages (5 Hz)


class ControllerNode(Node):
    def __init__(self):
        super().__init__("controller_node")

        # ---- the parts ---------------------------------------------------
        self.p = ControllerParams.from_node(self)
        self.model = ArmModel(self.p)
        self.seq = MotionSequencer(
            self.model, self.p, clock=self._now_s,
            on_status=self._publish_status, on_warn=self.get_logger().warn)
        self._backend = make_backend(self.p.backend, self)

        # Timer, topic callbacks and action threads all use the sequencer,
        # which is not thread-safe, so every one of them takes this lock.
        self._lock = threading.RLock()
        self._goal_tasks = {}        # action goal id -> its task
        self._last_fb = 0.0

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
                handle_accepted_callback=partial(self._on_accepted, mode=mode),
                callback_group=action_group)
            for action_type, name, mode in (
                (MoveToTarget, "/move_to_target", None),
                (PickAndPlace, "/pick_and_place", None),
                (Gripper, "/gripper", None),
                (PickOrPlace, "/pick", "pick"),      # close the gripper at a point
                (PickOrPlace, "/place", "place"))    # open the gripper at a point
        ]
        self._ee_pub = self.create_publisher(PointStamped, "/end_effector", 10)
        latched = QoSProfile(depth=1, durability=DurabilityPolicy.TRANSIENT_LOCAL)
        self._status_pub = self.create_publisher(String, "/controller_status", latched)

        self.create_timer(1.0 / self.p.rate, self._on_timer)
        self.seq.report("idle", "ready")
        self.get_logger().info(
            f"controller_node started (backend={self.p.backend}, "
            f"{self.p.rate:.0f} Hz)")

    def _now_s(self):
        return self.get_clock().now().nanoseconds * 1e-9

    def _publish_status(self, payload):
        """on_status callback of the sequencer: publish it as JSON."""
        self._status_pub.publish(String(data=json.dumps(payload)))

    # ------------------------------------------------------------------
    # topic / service callbacks
    # ------------------------------------------------------------------
    def _on_target(self, msg):
        with self._lock:
            if self.seq.busy():
                self.seq.report("error", "busy: send /stop first")
                return
            self.seq.start_move((msg.point.x, msg.point.y))

    def _on_pick_place(self, msg):
        with self._lock:
            if self.seq.busy():
                self.seq.report("error", "busy: send /stop first")
                return
            if len(msg.poses) != 2:
                self.seq.report(
                    "error", "need exactly 2 poses: [pick, place]")
                return
            pick = (msg.poses[0].position.x, msg.poses[0].position.y)
            place = (msg.poses[1].position.x, msg.poses[1].position.y)
            self.seq.start_pick_place(pick, place)

    def _on_stop(self, request, response):
        with self._lock:
            self.seq.stop("stopped by operator")
        response.success = True
        response.message = "stopped"
        return response

    # ------------------------------------------------------------------
    # action server callbacks (/move_to_target, /pick_and_place, /gripper)
    # ------------------------------------------------------------------
    def _on_goal(self, goal_request):
        """Accept only when idle; the reservation closes the race window."""
        with self._lock:
            if self.seq.busy():
                self.get_logger().warn("action goal rejected: controller busy")
                return GoalResponse.REJECT
            self.seq.reserved = True
        return GoalResponse.ACCEPT

    def _on_cancel(self, goal_handle):
        return CancelResponse.ACCEPT

    def _on_accepted(self, goal_handle, mode=None):
        request = goal_handle.request
        with self._lock:
            self.seq.reserved = False
            if isinstance(request, MoveToTarget.Goal):
                task = self.seq.start_move(
                    (request.target.x, request.target.y), goal_handle)
            elif isinstance(request, Gripper.Goal):
                task = self.seq.start_gripper(request.close, goal_handle)
            elif isinstance(request, PickOrPlace.Goal):
                xy = (request.target.x, request.target.y)
                task = (self.seq.start_pick(xy, goal_handle) if mode == "pick"
                        else self.seq.start_place(xy, goal_handle))
            else:
                task = self.seq.start_pick_place(
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
        elif isinstance(request, PickOrPlace.Goal):
            result = PickOrPlace.Result()
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
                    self.seq.cancel(task)
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
        ee = self.seq.ee()
        result.success = success
        result.message = message
        if hasattr(result, "final_position"):
            result.final_position = Point(x=float(ee[0]), y=float(ee[1]))
        if hasattr(result, "gripper_position"):
            result.gripper_position = float(self.seq.gripper)
        if hasattr(result, "clipped"):
            result.clipped = clipped
        if hasattr(result, "failed_step"):
            result.failed_step = failed_step

    def _publish_feedback(self, ee, now):
        """Throttled feedback for the running action goal (lock held)."""
        seq = self.seq
        task = seq.task
        if task is None or task.handle is None:
            return
        if now - self._last_fb < FEEDBACK_PERIOD:
            return
        self._last_fb = now
        position = Point(x=float(ee[0]), y=float(ee[1]))

        if task.kind == "move":
            feedback = MoveToTarget.Feedback()
            feedback.state = "moving" if seq.active else "idle"
            feedback.current_position = position
            if seq.goal_xy is not None:
                feedback.distance_remaining = float(math.hypot(
                    seq.goal_xy[0] - ee[0], seq.goal_xy[1] - ee[1]))
        elif task.kind == "gripper":
            feedback = Gripper.Feedback()
            feedback.gripper_position = float(seq.gripper)
            feedback.progress = float(seq.action_frac)
        else:                       # "pick_place" and "grasp" (pick / place)
            feedback = (PickAndPlace.Feedback() if task.kind == "pick_place"
                        else PickOrPlace.Feedback())
            feedback.step = seq.step_label
            feedback.step_index = int(task.step_index)
            feedback.total_steps = int(task.total_steps)
            done_steps = max(task.step_index - 1, 0)
            feedback.progress = float(min(
                (done_steps + seq.action_frac) / max(task.total_steps, 1),
                1.0))
            feedback.current_position = position
            feedback.gripper_position = float(seq.gripper)
        task.handle.publish_feedback(feedback)

    # ------------------------------------------------------------------
    # fixed-rate control loop
    # ------------------------------------------------------------------
    def _on_timer(self):
        with self._lock:
            now = self._now_s()
            self.seq.step()

            stamp = self.get_clock().now().to_msg()
            self._backend.send(
                stamp, self.seq.q, self.seq.qd, self.seq.gripper)

            ee = self.seq.ee()                       # gripper centre
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
