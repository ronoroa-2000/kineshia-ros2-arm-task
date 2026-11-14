#!/usr/bin/env python3
"""
gui_node.py - operator GUI for the 3-DoF planar arm (PyQt5 + PyQtGraph).

What it does:
  - "Launch RViz" / "Stop RViz" buttons: start and stop
    `ros2 launch planar_arm_control controller_rviz.launch.py`
    (controller_node + robot model + RViz).
  - Live graphs:
      * one graph per joint (joint1, joint2, joint3): angle over time,
        from /joint_states, the three sharing one time axis
      * end-effector trajectory (x-y path) from /end_effector, with the
        workspace limit (reach arc) and the ground line (y = 0). The arm
        itself is drawn live on this plot (forward kinematics of the
        joint angles), links coloured like the URDF, with the gripper
        (palm + two fingers) at the tip showing its opening. The last requested
        target is marked with an X and, if it was outside the workspace,
        the point actually used is marked with a triangle.
      * tracking error (q_ref - q_measured, per joint) from /tracking_error,
        published by the controller's PID tracking loop (pid_tracking.py),
        with the RMS / peak error of the plotted window
  - Small read-outs: end-effector position and the controller status
    (/controller_status).
  - Action buttons (ROS 2 action clients of controller_node):
      * Move to target  -> /move_to_target  (MoveToTarget)
      * Pick at target  -> /pick   (PickOrPlace): go to the target, close the gripper
      * Place at target -> /place  (PickOrPlace): go to the target, open the gripper
      * Cancel          -> cancels the running goal
    All three use the same target x / y boxes.
    with a progress bar, live feedback text and the final result.

Threading model (the key idea):
  - Main thread  : Qt event loop (window, plots, buttons).
  - Second thread: rclpy.spin(node)  (receives ROS messages).
  - The ROS callbacks never touch widgets. They emit Qt signals, and Qt
    delivers them safely to slots that run on the main thread.
  - In the other direction, button clicks only put a request on a
    thread-safe queue; a timer on the ROS thread reads it and talks to ROS,
    so all rclpy calls happen on the ROS thread.

Run (from a terminal where ROS and this workspace are sourced):
  ros2 run planar_arm_control gui_node

The "Launch RViz" button starts its own controller (controller_rviz.launch.py).
If a controller is already running (for example from bringup or the
ros2_control demo), do not use that button: use the Move / Pick / Place
buttons directly, they talk to whichever controller is running.
"""

import json
import math
import os
import queue
import signal
import subprocess
import sys
import threading
import time
from collections import deque

import rclpy
from action_msgs.msg import GoalStatus
from geometry_msgs.msg import PointStamped
from PyQt5 import QtCore, QtGui, QtWidgets
from rclpy.action import ActionClient
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, QoSProfile
from sensor_msgs.msg import JointState
from std_msgs.msg import Float64MultiArray, String

try:
    import pyqtgraph as pg
except ImportError:
    sys.exit("pyqtgraph is missing. Install it with:\n"
             "  sudo apt install python3-pyqtgraph")

try:
    from planar_arm_control.planar_arm import PlanarArm
except ImportError:
    sys.exit("planar_arm_control not found. Build and source the workspace:\n"
             "  colcon build && source install/setup.bash")

try:
    from arm_interfaces.action import MoveToTarget, PickOrPlace
except ImportError:
    sys.exit("arm_interfaces not found. Build and source the workspace:\n"
             "  colcon build && source install/setup.bash")

JOINTS = ["joint1", "joint2", "joint3"]
# Same colours as the URDF links (rgba_link1..3 in common_properties.xacro),
# converted from 0..1 to 0..255: joint N moves link N.
COLORS = [(51, 102, 204),    # joint1 / link1 blue   (0.2 0.4 0.8)
          (230, 128, 26),    # joint2 / link2 orange (0.9 0.5 0.1)
          (51, 179, 77)]     # joint3 / link3 green  (0.2 0.7 0.3)
LINK_LENGTHS = [3.0, 2.0, 1.5]
EE_CENTER_OFFSET = 0.35     # link-3 tip -> midpoint between the fingers
REACH = sum(LINK_LENGTHS) + EE_CENTER_OFFSET  # max radius of the target point
# Gripper dimensions in metres (same values as common_properties.xacro).
# Tool frame: u along the tool (link 3 direction), v across the fingers.
EE_PALM_LEN = 0.15          # palm size along the tool
EE_PALM_WIDTH = 0.5         # palm size across the fingers
EE_FINGER_LEN = 0.4
EE_FINGER_W = 0.05
EE_FINGER_OFFSET = 0.21     # finger centre from the tool axis when open
LAUNCH_CMD = ["ros2", "launch", "planar_arm_control",
              "controller_rviz.launch.py"]
ANGLE_BUFFER = 1000      # samples kept per joint curve (20 s at 50 Hz)
TRAJ_BUFFER = 3000       # end-effector points kept (60 s at 50 Hz)
ERROR_BUFFER = 1000      # tracking-error samples kept per joint (20 s at 50 Hz)
REDRAW_MS = 33           # plots redraw at about 30 fps
GOAL_STATUS = {GoalStatus.STATUS_SUCCEEDED: "SUCCEEDED",
               GoalStatus.STATUS_CANCELED: "CANCELED",
               GoalStatus.STATUS_ABORTED: "ABORTED"}


# ======================================================================
# ROS side
# ======================================================================
class RosBridge(QtCore.QObject):
    """Signals used to pass data from the ROS thread to the Qt thread."""

    joint_state = QtCore.pyqtSignal(float, list, float)  # t, angles, gripper
    end_effector = QtCore.pyqtSignal(float, float)  # x, y
    tracking_error = QtCore.pyqtSignal(float, list)  # time (s), [e1, e2, e3]
    status = QtCore.pyqtSignal(str)                 # JSON string
    # Action signals
    servers_ready = QtCore.pyqtSignal(bool)         # both servers available
    action_started = QtCore.pyqtSignal(str)         # description of the goal
    action_feedback = QtCore.pyqtSignal(str, float)  # text, progress 0..1
    action_done = QtCore.pyqtSignal(str, bool)      # result text, success


class GuiRosNode(Node):
    def __init__(self, bridge):
        super().__init__("gui_node")
        self._bridge = bridge
        self.create_subscription(
            JointState, "/joint_states", self._on_joint_state, 10)
        self.create_subscription(
            PointStamped, "/end_effector", self._on_end_effector, 10)
        self.create_subscription(
            Float64MultiArray, "/tracking_error", self._on_tracking_error, 10)
        # Latched topic: we receive the last status even if we start late.
        latched = QoSProfile(
            depth=1, durability=DurabilityPolicy.TRANSIENT_LOCAL)
        self.create_subscription(
            String, "/controller_status", self._on_status, latched)

        # ---- action clients -------------------------------------------
        self._move_client = ActionClient(self, MoveToTarget, "/move_to_target")
        self._pick_client = ActionClient(self, PickOrPlace, "/pick")
        self._place_client = ActionClient(self, PickOrPlace, "/place")
        self._goal_handle = None         # goal currently running (if any)
        self._first_distance = None      # for the move progress estimate
        self._servers_ready = None
        # Requests from the Qt thread arrive here; a timer on the ROS thread
        # drains the queue, so rclpy is only ever called from the ROS thread.
        self._requests = queue.Queue()
        self.create_timer(0.05, self._process_requests)

    # ---- called from the Qt thread: only queue a request ---------------
    def request_move(self, x, y):
        self._requests.put(("move", (x, y)))

    def request_pick(self, x, y):
        self._requests.put(("pick", (x, y)))

    def request_place(self, x, y):
        self._requests.put(("place", (x, y)))

    def request_cancel(self):
        self._requests.put(("cancel", None))

    # ---- everything below runs on the ROS thread -----------------------
    def _process_requests(self):
        ready = (self._move_client.server_is_ready()
                 and self._pick_client.server_is_ready()
                 and self._place_client.server_is_ready())
        if ready != self._servers_ready:
            self._servers_ready = ready
            self._bridge.servers_ready.emit(ready)

        while True:
            try:
                kind, data = self._requests.get_nowait()
            except queue.Empty:
                break
            if kind == "move":
                self._send_move(*data)
            elif kind in ("pick", "place"):
                self._send_grasp(kind, *data)
            elif kind == "cancel":
                self._cancel_goal()

    def _send_move(self, x, y):
        if not self._move_client.server_is_ready():
            self._bridge.action_done.emit("move server not available", False)
            return
        goal = MoveToTarget.Goal()
        goal.target.x, goal.target.y = float(x), float(y)
        self._first_distance = None
        self._bridge.action_started.emit(f"Move to ({x:.2f}, {y:.2f})")
        future = self._move_client.send_goal_async(
            goal, feedback_callback=self._on_move_feedback)
        future.add_done_callback(self._on_goal_response)

    def _send_grasp(self, kind, x, y):
        """kind "pick": go to (x, y) and close the gripper; "place": open it."""
        client = self._pick_client if kind == "pick" else self._place_client
        if not client.server_is_ready():
            self._bridge.action_done.emit(f"{kind} server not available", False)
            return
        goal = PickOrPlace.Goal()
        goal.target.x, goal.target.y = float(x), float(y)
        self._bridge.action_started.emit(
            f"{kind.capitalize()} at ({x:.2f}, {y:.2f})")
        future = client.send_goal_async(
            goal, feedback_callback=self._on_grasp_feedback)
        future.add_done_callback(self._on_goal_response)

    def _cancel_goal(self):
        if self._goal_handle is None:
            return
        self._goal_handle.cancel_goal_async()    # result arrives as CANCELED

    def _on_goal_response(self, future):
        handle = future.result()
        if not handle.accepted:
            self._bridge.action_done.emit(
                "goal rejected (controller busy?)", False)
            return
        self._goal_handle = handle
        handle.get_result_async().add_done_callback(self._on_result)

    def _on_move_feedback(self, msg):
        fb = msg.feedback
        if self._first_distance is None:
            self._first_distance = max(fb.distance_remaining, 1e-6)
        progress = 1.0 - fb.distance_remaining / self._first_distance
        text = (f"{fb.state}: {fb.distance_remaining:.2f} m to go  "
                f"(at {fb.current_position.x:.2f}, "
                f"{fb.current_position.y:.2f})")
        self._bridge.action_feedback.emit(text, max(0.0, min(progress, 1.0)))

    def _on_grasp_feedback(self, msg):
        fb = msg.feedback
        text = (f"step {fb.step_index}/{fb.total_steps}: {fb.step}   "
                f"gripper {fb.gripper_position:.3f} m")
        self._bridge.action_feedback.emit(text, fb.progress)

    def _on_result(self, future):
        wrapped = future.result()
        result = wrapped.result
        name = GOAL_STATUS.get(wrapped.status, str(wrapped.status))
        text = f"{name}: {result.message}"
        if getattr(result, "clipped", False):
            text += "  (target was clipped to the workspace boundary)"
        if getattr(result, "failed_step", ""):
            text += f"  [failed step: {result.failed_step}]"
        self._goal_handle = None
        self._bridge.action_done.emit(
            text, wrapped.status == GoalStatus.STATUS_SUCCEEDED)

    # These run on the ROS thread: only emit signals, never touch widgets.
    def _on_joint_state(self, msg):
        index = {name: i for i, name in enumerate(msg.name)}
        try:
            pos = [msg.position[index[j]] for j in JOINTS]
        except (KeyError, IndexError):
            return
        t = msg.header.stamp.sec + msg.header.stamp.nanosec * 1e-9
        # Gripper opening: position of the left finger joint (0 = open).
        gripper = 0.0
        i = index.get("ee_finger_left_joint")
        if i is not None and i < len(msg.position):
            gripper = msg.position[i]
        self._bridge.joint_state.emit(t, pos, gripper)

    def _on_end_effector(self, msg):
        self._bridge.end_effector.emit(msg.point.x, msg.point.y)

    def _on_tracking_error(self, msg):
        if len(msg.data) >= len(JOINTS):
            self._bridge.tracking_error.emit(
                time.monotonic(), [float(v) for v in msg.data[:len(JOINTS)]])

    def _on_status(self, msg):
        self._bridge.status.emit(msg.data)


# ======================================================================
# Qt side
# ======================================================================
class MainWindow(QtWidgets.QMainWindow):
    def __init__(self, bridge, node):
        super().__init__()
        self.setWindowTitle("Planar arm - controller GUI")
        self.resize(1100, 960)

        self._node = node            # only request_*() are called from here
        self._servers_ready = False  # both action servers available
        self._action_busy = False    # an action goal is in progress
        self._proc = None            # the `ros2 launch` process
        self._stopping = False       # stop requested, process not gone yet
        self._t0 = None              # first message time (x-axis origin)
        self._last_rx = 0.0          # wall time of the last joint state
        self._dirty_angles = False   # new data waiting to be drawn
        self._dirty_ee = False
        self._arm = PlanarArm(LINK_LENGTHS)   # forward kinematics for drawing
        self._q_latest = None        # newest joint angles (rad)
        self._gripper_latest = 0.0   # newest finger position (m, 0 = open)
        self._dirty_arm = False
        self._t = deque(maxlen=ANGLE_BUFFER)
        self._angles = [deque(maxlen=ANGLE_BUFFER) for _ in JOINTS]
        self._ee_x = deque(maxlen=TRAJ_BUFFER)
        self._ee_y = deque(maxlen=TRAJ_BUFFER)
        self._err_t0 = None          # first tracking-error time (x origin)
        self._err_t = deque(maxlen=ERROR_BUFFER)
        self._err = [deque(maxlen=ERROR_BUFFER) for _ in JOINTS]
        self._dirty_err = False

        self._build_ui()
        self._update_buttons()
        self._update_action_buttons()

        bridge.joint_state.connect(self._on_joint_state)
        bridge.end_effector.connect(self._on_end_effector)
        bridge.tracking_error.connect(self._on_tracking_error)
        bridge.status.connect(self._on_status)
        bridge.servers_ready.connect(self._on_servers_ready)
        bridge.action_started.connect(self._on_action_started)
        bridge.action_feedback.connect(self._on_action_feedback)
        bridge.action_done.connect(self._on_action_done)

        # Redraw timer: draws at a fixed rate, no matter how fast data comes.
        self._redraw_timer = QtCore.QTimer(self)
        self._redraw_timer.timeout.connect(self._redraw)
        self._redraw_timer.start(REDRAW_MS)

        # Slow timer: checks whether the launched process is still alive.
        self._poll_timer = QtCore.QTimer(self)
        self._poll_timer.timeout.connect(self._poll_process)
        self._poll_timer.start(500)

    # ---------------- layout ----------------
    def _build_ui(self):
        central = QtWidgets.QWidget()
        self.setCentralWidget(central)
        layout = QtWidgets.QVBoxLayout(central)

        # Top row: launch / stop buttons + indicators
        top = QtWidgets.QHBoxLayout()
        self._launch_btn = QtWidgets.QPushButton("Launch RViz")
        self._launch_btn.clicked.connect(self._start_launch)
        self._stop_btn = QtWidgets.QPushButton("Stop RViz")
        self._stop_btn.clicked.connect(self._stop_launch)
        self._launch_lbl = QtWidgets.QLabel("launch: stopped")
        self._conn_lbl = QtWidgets.QLabel("no data on /joint_states")
        top.addWidget(self._launch_btn)
        top.addWidget(self._stop_btn)
        top.addWidget(self._launch_lbl)
        top.addStretch(1)
        top.addWidget(self._conn_lbl)
        layout.addLayout(top)

        # Plots side by side: one graph per joint (stacked, sharing the
        # time axis) | end-effector trajectory
        plots = QtWidgets.QHBoxLayout()
        angle_col = QtWidgets.QVBoxLayout()
        self._angle_plots = []
        self._angle_curves = []
        for i, (name, color) in enumerate(zip(JOINTS, COLORS)):
            plot = pg.PlotWidget()
            plot.setLabel("left", name, units="rad")
            plot.showGrid(x=True, y=True)
            if i < len(JOINTS) - 1:                  # time numbers only at the bottom
                plot.getAxis("bottom").setStyle(showValues=False)
            else:
                plot.setLabel("bottom", "time", units="s")
            if self._angle_plots:                    # zoom / pan all three together
                plot.setXLink(self._angle_plots[0])
            self._angle_plots.append(plot)
            self._angle_curves.append(
                plot.plot(pen=pg.mkPen(color, width=2)))
            angle_col.addWidget(plot, 2)

        # Tracking error of the three joints (reference - measured), live.
        self._err_plot = pg.PlotWidget(title="Tracking error (ref - measured)")
        self._err_plot.getAxis("left").enableAutoSIPrefix(False)   # keep rad
        self._err_plot.setLabel("left", "error", units="rad")
        self._err_plot.setLabel("bottom", "time", units="s")
        self._err_plot.showGrid(x=True, y=True)
        self._err_plot.addLegend(offset=(-10, 5))
        self._err_plot.addLine(y=0, pen=pg.mkPen((150, 150, 150), width=1))
        self._err_curves = [
            self._err_plot.plot(pen=pg.mkPen(color, width=2), name=name)
            for name, color in zip(JOINTS, COLORS)]
        angle_col.addWidget(self._err_plot, 3)        # the tallest of the four
        self._err_lbl = QtWidgets.QLabel(
            "tracking error: no data on /tracking_error "
            "(needs backend gazebo or ros2_control)")
        angle_col.addWidget(self._err_lbl)

        self._ee_plot = pg.PlotWidget(title="End-effector trajectory")
        self._ee_plot.setLabel("left", "y", units="m")
        self._ee_plot.setLabel("bottom", "x", units="m")
        self._ee_plot.showGrid(x=True, y=True)
        self._ee_plot.setAspectLocked(True)       # equal x / y scale
        self._draw_workspace(self._ee_plot)
        self._ee_curve = self._ee_plot.plot(pen=pg.mkPen("y", width=2))
        # The arm: one thick line per link (URDF colours) + joint dots.
        # Items drawn later appear on top of earlier ones.
        self._link_curves = [
            self._ee_plot.plot(pen=pg.mkPen(color, width=8))
            for color in COLORS]
        self._joint_dots = self._ee_plot.plot(
            pen=None, symbol="o", symbolSize=10, symbolBrush="w")
        # The end effector (gripper): palm + two fingers, drawn as filled
        # shapes at the tip of link 3, turning with the arm.
        self._palm_item = self._make_shape((100, 100, 100))
        self._finger_items = [self._make_shape((204, 51, 51))
                              for _ in range(2)]
        self._ee_dot = self._ee_plot.plot(
            pen=None, symbol="o", symbolSize=10, symbolBrush="r")
        # Last requested target (X) and, if clipped, the point actually used.
        self._target_mark = self._ee_plot.plot(
            pen=None, symbol="x", symbolSize=16, symbolBrush="m")
        self._clip_mark = self._ee_plot.plot(
            pen=None, symbol="t", symbolSize=14, symbolBrush="r")
        # Fixed view so the picture does not jump around as the arm moves.
        self._ee_plot.setRange(xRange=(-REACH - 1.5, REACH + 1.5),
                               yRange=(-1.0, REACH + 1.5), padding=0)

        plots.addLayout(angle_col, 1)
        plots.addWidget(self._ee_plot, 1)
        layout.addLayout(plots, stretch=1)

        layout.addWidget(self._build_action_panel())

        # Bottom read-outs
        self._ee_lbl = QtWidgets.QLabel("end effector: -")
        self._status_lbl = QtWidgets.QLabel("controller: -")
        self._status_lbl.setWordWrap(True)
        self._grip_lbl = QtWidgets.QLabel("gripper: -")
        layout.addWidget(self._ee_lbl)
        layout.addWidget(self._grip_lbl)
        layout.addWidget(self._status_lbl)

    @staticmethod
    def _make_spin(value):
        spin = QtWidgets.QDoubleSpinBox()
        spin.setRange(-10.0, 10.0)
        spin.setDecimals(2)
        spin.setSingleStep(0.1)
        spin.setSuffix(" m")
        spin.setValue(value)
        return spin

    def _build_action_panel(self):
        """Target boxes, the three action buttons, progress and result."""
        box = QtWidgets.QGroupBox("Actions (controller_node)")
        grid = QtWidgets.QGridLayout(box)

        # Row 0: the target position used by all three buttons
        self._move_x = self._make_spin(4.0)
        self._move_y = self._make_spin(2.0)
        grid.addWidget(QtWidgets.QLabel("Target"), 0, 0)
        grid.addWidget(QtWidgets.QLabel("x"), 0, 1)
        grid.addWidget(self._move_x, 0, 2)
        grid.addWidget(QtWidgets.QLabel("y"), 0, 3)
        grid.addWidget(self._move_y, 0, 4)

        # Row 1: move / pick (close gripper) / place (open gripper)
        self._move_btn = QtWidgets.QPushButton("Move to target")
        self._move_btn.clicked.connect(self._on_move_clicked)
        self._pick_btn = QtWidgets.QPushButton("Pick at target")
        self._pick_btn.setToolTip(
            "Go to the target, close the gripper, back out")
        self._pick_btn.clicked.connect(self._on_pick_clicked)
        self._place_btn = QtWidgets.QPushButton("Place at target")
        self._place_btn.setToolTip(
            "Go to the target, open the gripper, back out")
        self._place_btn.clicked.connect(self._on_place_clicked)
        buttons = QtWidgets.QHBoxLayout()
        for btn in (self._move_btn, self._pick_btn, self._place_btn):
            buttons.addWidget(btn)
        grid.addLayout(buttons, 1, 0, 1, 6)

        # Row 2..4: cancel + progress + feedback + result
        self._cancel_btn = QtWidgets.QPushButton("Cancel")
        self._cancel_btn.clicked.connect(self._on_cancel_clicked)
        self._progress = QtWidgets.QProgressBar()
        self._progress.setRange(0, 100)
        self._progress.setValue(0)
        self._server_lbl = QtWidgets.QLabel("action servers: not available")
        self._feedback_lbl = QtWidgets.QLabel("feedback: -")
        self._result_lbl = QtWidgets.QLabel("last result: -")
        self._result_lbl.setWordWrap(True)
        grid.addWidget(self._progress, 2, 0, 1, 5)
        grid.addWidget(self._cancel_btn, 2, 5)
        grid.addWidget(self._feedback_lbl, 3, 0, 1, 5)
        grid.addWidget(self._server_lbl, 3, 5)
        grid.addWidget(self._result_lbl, 4, 0, 1, 6)
        return box

    def _make_shape(self, rgb):
        """A filled polygon (updated every frame) inside the x-y plot."""
        item = QtWidgets.QGraphicsPolygonItem()
        item.setPen(pg.mkPen((230, 230, 230), width=1))
        item.setBrush(pg.mkBrush(*rgb))
        item.setZValue(10)                     # above the arm lines
        self._ee_plot.addItem(item)
        return item

    @staticmethod
    def _rect_points(tip, heading, u0, u1, v0, v1):
        """Corners of a rectangle given in the tool frame, as world points."""
        c, s = math.cos(heading), math.sin(heading)
        return QtGui.QPolygonF([
            QtCore.QPointF(tip[0] + u * c - v * s, tip[1] + u * s + v * c)
            for u, v in ((u0, v0), (u1, v0), (u1, v1), (u0, v1))])

    def _draw_gripper(self, tip, heading, gripper):
        """Palm and fingers at the arm tip; fingers move in by `gripper`."""
        self._palm_item.setPolygon(self._rect_points(
            tip, heading, 0.0, EE_PALM_LEN,
            -EE_PALM_WIDTH / 2, EE_PALM_WIDTH / 2))
        for item, side in zip(self._finger_items, (1, -1)):
            centre = side * (EE_FINGER_OFFSET - gripper)
            item.setPolygon(self._rect_points(
                tip, heading, EE_PALM_LEN, EE_PALM_LEN + EE_FINGER_LEN,
                centre - EE_FINGER_W / 2, centre + EE_FINGER_W / 2))

    @staticmethod
    def _draw_workspace(plot):
        """Dashed reach arc (upper half) and the ground line y = 0."""
        dashed = pg.mkPen((150, 150, 150), style=QtCore.Qt.DashLine)
        angles = [math.pi * k / 100 for k in range(101)]
        plot.plot([REACH * math.cos(a) for a in angles],
                  [REACH * math.sin(a) for a in angles], pen=dashed)
        plot.plot([-REACH, REACH], [0, 0], pen=pg.mkPen("w", width=1))

    # ---------------- ROS data slots (Qt thread) ----------------
    def _on_joint_state(self, t, angles, gripper):
        if self._t0 is None:
            self._t0 = t
        self._t.append(t - self._t0)
        for i in range(len(JOINTS)):
            self._angles[i].append(angles[i])
        self._last_rx = time.monotonic()
        self._dirty_angles = True
        self._q_latest = list(angles)
        self._gripper_latest = gripper
        self._dirty_arm = True

    def _on_tracking_error(self, t, errors):
        if self._err_t0 is None:
            self._err_t0 = t
        self._err_t.append(t - self._err_t0)
        for i in range(len(JOINTS)):
            self._err[i].append(errors[i])
        self._dirty_err = True

    def _on_end_effector(self, x, y):
        self._ee_x.append(x)
        self._ee_y.append(y)
        self._ee_lbl.setText(
            f"end effector (centre between fingers): x = {x:.3f}, y = {y:.3f}")
        self._dirty_ee = True

    def _on_status(self, text):
        try:
            d = json.loads(text)
        except ValueError:
            self._status_lbl.setText(f"controller: {text}")
            return
        line = (f"controller: {d.get('state', '?')} / "
                f"{d.get('step') or '-'} : {d.get('message', '')}")
        self._update_target_marks(d)
        if d.get("clipped"):
            target = d.get("target")
            used = [round(v, 2) for v in d.get("used_target", [])]
            line += f"   [target {target} clipped to {used}]"
        self._status_lbl.setText(line)

    def _update_target_marks(self, d):
        """Draw the requested target (X) and the clipped point (triangle)."""
        target = d.get("target")
        if not target or len(target) < 2:
            return
        self._target_mark.setData([target[0]], [target[1]])
        used = d.get("used_target")
        if d.get("clipped") and used and len(used) >= 2:
            self._clip_mark.setData([used[0]], [used[1]])
        else:
            self._clip_mark.setData([], [])

    # ---------------- action buttons and slots ----------------
    def _on_move_clicked(self):
        self._action_busy = True                 # block double clicks at once
        self._update_action_buttons()
        self._node.request_move(self._move_x.value(), self._move_y.value())

    def _on_pick_clicked(self):
        self._action_busy = True
        self._update_action_buttons()
        self._node.request_pick(self._move_x.value(), self._move_y.value())

    def _on_place_clicked(self):
        self._action_busy = True
        self._update_action_buttons()
        self._node.request_place(self._move_x.value(), self._move_y.value())

    def _on_cancel_clicked(self):
        self._node.request_cancel()
        self._feedback_lbl.setText("feedback: cancel requested...")

    def _update_action_buttons(self):
        can_start = self._servers_ready and not self._action_busy
        self._move_btn.setEnabled(can_start)
        self._pick_btn.setEnabled(can_start)
        self._place_btn.setEnabled(can_start)
        self._cancel_btn.setEnabled(self._action_busy)

    def _on_servers_ready(self, ready):
        self._servers_ready = ready
        self._server_lbl.setText(
            "action servers: ready" if ready
            else "action servers: not available")
        self._update_action_buttons()

    def _on_action_started(self, text):
        self._action_busy = True
        self._progress.setValue(0)
        self._feedback_lbl.setText(f"feedback: {text} ...")
        self._result_lbl.setStyleSheet("")
        self._result_lbl.setText("last result: running")
        self._update_action_buttons()

    def _on_action_feedback(self, text, progress):
        self._progress.setValue(int(max(0.0, min(progress, 1.0)) * 100))
        self._feedback_lbl.setText(f"feedback: {text}")

    def _on_action_done(self, text, success):
        self._action_busy = False
        if success:
            self._progress.setValue(100)
        self._result_lbl.setStyleSheet(
            "color: green" if success else "color: red")
        self._result_lbl.setText(f"last result: {text}")
        self._update_action_buttons()

    # ---------------- redraw ----------------
    def _redraw(self):
        if self._dirty_angles:
            self._dirty_angles = False
            t = list(self._t)
            for i in range(len(JOINTS)):
                self._angle_curves[i].setData(t, list(self._angles[i]))

        if self._dirty_err:
            self._dirty_err = False
            t = list(self._err_t)
            rms, peak = [], []
            for i in range(len(JOINTS)):
                e = list(self._err[i])
                self._err_curves[i].setData(t, e)
                rms.append(math.sqrt(sum(v * v for v in e) / len(e)))
                peak.append(max(abs(v) for v in e))
            self._err_lbl.setText(
                "error RMS / peak over the window (rad):  " + "   ".join(
                    f"{j}: {r:.3f} / {m:.3f}"
                    for j, r, m in zip(JOINTS, rms, peak)))

        if self._dirty_arm and self._q_latest is not None:
            self._dirty_arm = False
            pts = self._arm.forward_kinematics(self._q_latest)  # 4 points
            for i, curve in enumerate(self._link_curves):
                curve.setData([pts[i][0], pts[i + 1][0]],
                              [pts[i][1], pts[i + 1][1]])
            self._joint_dots.setData([p[0] for p in pts],
                                     [p[1] for p in pts])
            heading = sum(self._q_latest)        # link-3 direction
            self._draw_gripper(pts[-1], heading, self._gripper_latest)
            gap = 2 * (EE_FINGER_OFFSET - EE_FINGER_W / 2
                       - self._gripper_latest)
            self._grip_lbl.setText(
                f"gripper: fingers at {self._gripper_latest:.3f} m, "
                f"opening between fingers = {gap:.3f} m")

        if self._dirty_ee:
            self._dirty_ee = False
            xs, ys = list(self._ee_x), list(self._ee_y)
            self._ee_curve.setData(xs, ys)
            self._ee_dot.setData([xs[-1]], [ys[-1]])   # current position

        receiving = (time.monotonic() - self._last_rx) < 1.0
        self._conn_lbl.setText(
            "receiving /joint_states" if receiving
            else "no data on /joint_states")

    # ---------------- RViz launch / stop buttons ----------------
    def _is_running(self):
        return self._proc is not None and self._proc.poll() is None

    def _update_buttons(self):
        running = self._is_running()
        self._launch_btn.setEnabled(not running)
        self._stop_btn.setEnabled(running and not self._stopping)

    def _start_launch(self):
        if self._is_running():
            return
        try:
            # New session so we can stop the whole launch tree at once.
            self._proc = subprocess.Popen(LAUNCH_CMD, start_new_session=True)
        except FileNotFoundError:
            self._launch_lbl.setText(
                "launch: 'ros2' not found (source ROS first)")
            self._proc = None
            return
        self._stopping = False
        self._launch_lbl.setText(f"launch: running (pid {self._proc.pid})")
        self._update_buttons()

    def _stop_launch(self):
        if not self._is_running() or self._stopping:
            return
        self._stopping = True
        self._launch_lbl.setText("launch: stopping...")
        self._signal_group(signal.SIGINT)            # polite, like Ctrl+C
        QtCore.QTimer.singleShot(5000, self._force_kill)
        self._update_buttons()

    def _force_kill(self):
        if self._is_running():
            self._signal_group(signal.SIGKILL)

    def _signal_group(self, sig):
        try:
            os.killpg(os.getpgid(self._proc.pid), sig)
        except (ProcessLookupError, PermissionError):
            pass

    def _poll_process(self):
        if self._proc is not None and self._proc.poll() is not None:
            self._proc = None
            self._stopping = False
            self._launch_lbl.setText("launch: stopped")
        self._update_buttons()

    # ---------------- shutdown ----------------
    def closeEvent(self, event):
        if self._is_running():
            self._signal_group(signal.SIGINT)
            try:
                self._proc.wait(timeout=5)
            except subprocess.TimeoutExpired:
                self._signal_group(signal.SIGKILL)
        event.accept()


# ======================================================================
def _spin(node):
    try:
        rclpy.spin(node)
    except Exception:       # raised when rclpy shuts down under us
        pass


def main(args=None):
    rclpy.init(args=args)
    app = QtWidgets.QApplication(sys.argv)

    bridge = RosBridge()
    node = GuiRosNode(bridge)
    window = MainWindow(bridge, node)

    spin_thread = threading.Thread(target=_spin, args=(node,), daemon=True)
    spin_thread.start()

    window.show()
    exit_code = app.exec_()          # Qt event loop runs on the main thread

    rclpy.shutdown()
    spin_thread.join(timeout=2.0)
    node.destroy_node()
    sys.exit(exit_code)


if __name__ == "__main__":
    main()
