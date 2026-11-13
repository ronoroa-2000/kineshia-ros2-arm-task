#!/usr/bin/env python3
"""
test_gui_node.py - first test GUI for the planar arm (PyQt5 + PyQtGraph).

What it does:
  - "Launch RViz" / "Stop RViz" buttons: start and stop
    `ros2 launch planar_arm_control controller_rviz.launch.py`
    (controller_node + robot model + RViz).
  - Two live graphs:
      * joint angles over time (joint1, joint2, joint3) from /joint_states
      * end-effector trajectory (x-y path) from /end_effector, with the
        workspace limit (reach arc) and the ground line (y = 0)
  - Small read-outs: end-effector position and the controller status
    (/controller_status).
  - Action buttons (ROS 2 action clients of controller_node):
      * Move   -> /move_to_target  (MoveToTarget)
      * Pick   -> /gripper         (Gripper, close = true)
      * Place  -> /gripper         (Gripper, close = false)
      * Cancel -> cancels the running goal
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
  python3 test_gui_node.py
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
from PyQt5 import QtCore, QtWidgets
from rclpy.action import ActionClient
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, QoSProfile
from sensor_msgs.msg import JointState
from std_msgs.msg import String

try:
    import pyqtgraph as pg
except ImportError:
    sys.exit("pyqtgraph is missing. Install it with:\n"
             "  sudo apt install python3-pyqtgraph")

try:
    from arm_interfaces.action import Gripper, MoveToTarget
except ImportError:
    sys.exit("arm_interfaces not found. Build and source the workspace:\n"
             "  colcon build && source install/setup.bash")

JOINTS = ["joint1", "joint2", "joint3"]
# Same colours as the URDF links (rgba_link1..3 in common_properties.xacro),
# converted from 0..1 to 0..255: joint N moves link N.
COLORS = [(51, 102, 204),    # joint1 / link1 blue   (0.2 0.4 0.8)
          (230, 128, 26),    # joint2 / link2 orange (0.9 0.5 0.1)
          (51, 179, 77)]     # joint3 / link3 green  (0.2 0.7 0.3)
REACH = 3.0 + 2.0 + 1.5  # max workspace radius (sum of link lengths)
LAUNCH_CMD = ["ros2", "launch", "planar_arm_control",
              "controller_rviz.launch.py"]
ANGLE_BUFFER = 1000      # samples kept per joint curve (20 s at 50 Hz)
TRAJ_BUFFER = 3000       # end-effector points kept (60 s at 50 Hz)
REDRAW_MS = 33           # plots redraw at about 30 fps
GOAL_STATUS = {GoalStatus.STATUS_SUCCEEDED: "SUCCEEDED",
               GoalStatus.STATUS_CANCELED: "CANCELED",
               GoalStatus.STATUS_ABORTED: "ABORTED"}


# ======================================================================
# ROS side
# ======================================================================
class RosBridge(QtCore.QObject):
    """Signals used to pass data from the ROS thread to the Qt thread."""

    joint_state = QtCore.pyqtSignal(float, list)    # time (s), angles (rad)
    end_effector = QtCore.pyqtSignal(float, float)  # x, y
    status = QtCore.pyqtSignal(str)                 # JSON string
    # Action signals
    servers_ready = QtCore.pyqtSignal(bool)         # both servers available
    action_started = QtCore.pyqtSignal(str)         # description of the goal
    action_feedback = QtCore.pyqtSignal(str, float)  # text, progress 0..1
    action_done = QtCore.pyqtSignal(str, bool)      # result text, success


class GuiRosNode(Node):
    def __init__(self, bridge):
        super().__init__("test_gui_node")
        self._bridge = bridge
        self.create_subscription(
            JointState, "/joint_states", self._on_joint_state, 10)
        self.create_subscription(
            PointStamped, "/end_effector", self._on_end_effector, 10)
        # Latched topic: we receive the last status even if we start late.
        latched = QoSProfile(
            depth=1, durability=DurabilityPolicy.TRANSIENT_LOCAL)
        self.create_subscription(
            String, "/controller_status", self._on_status, latched)

        # ---- action clients -------------------------------------------
        self._move_client = ActionClient(self, MoveToTarget, "/move_to_target")
        self._gripper_client = ActionClient(self, Gripper, "/gripper")
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

    def request_gripper(self, close):
        self._requests.put(("gripper", close))

    def request_cancel(self):
        self._requests.put(("cancel", None))

    # ---- everything below runs on the ROS thread -----------------------
    def _process_requests(self):
        ready = (self._move_client.server_is_ready()
                 and self._gripper_client.server_is_ready())
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
            elif kind == "gripper":
                self._send_gripper(data)
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

    def _send_gripper(self, close):
        if not self._gripper_client.server_is_ready():
            self._bridge.action_done.emit("gripper server not available", False)
            return
        goal = Gripper.Goal()
        goal.close = bool(close)
        self._bridge.action_started.emit(
            "Pick: closing gripper" if close else "Place: opening gripper")
        future = self._gripper_client.send_goal_async(
            goal, feedback_callback=self._on_gripper_feedback)
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

    def _on_gripper_feedback(self, msg):
        fb = msg.feedback
        text = f"gripper position {fb.gripper_position:.3f} m"
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
        self._bridge.joint_state.emit(t, pos)

    def _on_end_effector(self, msg):
        self._bridge.end_effector.emit(msg.point.x, msg.point.y)

    def _on_status(self, msg):
        self._bridge.status.emit(msg.data)


# ======================================================================
# Qt side
# ======================================================================
class MainWindow(QtWidgets.QMainWindow):
    def __init__(self, bridge, node):
        super().__init__()
        self.setWindowTitle("Planar arm - test GUI")
        self.resize(1100, 760)

        self._node = node            # only request_*() are called from here
        self._servers_ready = False  # both action servers available
        self._action_busy = False    # an action goal is in progress
        self._proc = None            # the `ros2 launch` process
        self._stopping = False       # stop requested, process not gone yet
        self._t0 = None              # first message time (x-axis origin)
        self._last_rx = 0.0          # wall time of the last joint state
        self._dirty_angles = False   # new data waiting to be drawn
        self._dirty_ee = False
        self._t = deque(maxlen=ANGLE_BUFFER)
        self._angles = [deque(maxlen=ANGLE_BUFFER) for _ in JOINTS]
        self._ee_x = deque(maxlen=TRAJ_BUFFER)
        self._ee_y = deque(maxlen=TRAJ_BUFFER)

        self._build_ui()
        self._update_buttons()
        self._update_action_buttons()

        bridge.joint_state.connect(self._on_joint_state)
        bridge.end_effector.connect(self._on_end_effector)
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

        # Plots side by side: angles over time | end-effector trajectory
        plots = QtWidgets.QHBoxLayout()
        self._angle_plot = pg.PlotWidget(title="Joint angles")
        self._angle_plot.setLabel("left", "angle", units="rad")
        self._angle_plot.setLabel("bottom", "time", units="s")
        self._angle_plot.showGrid(x=True, y=True)
        self._angle_plot.addLegend()
        self._angle_curves = [
            self._angle_plot.plot(pen=pg.mkPen(color, width=2), name=name)
            for name, color in zip(JOINTS, COLORS)]

        self._ee_plot = pg.PlotWidget(title="End-effector trajectory")
        self._ee_plot.setLabel("left", "y", units="m")
        self._ee_plot.setLabel("bottom", "x", units="m")
        self._ee_plot.showGrid(x=True, y=True)
        self._ee_plot.setAspectLocked(True)       # equal x / y scale
        self._draw_workspace(self._ee_plot)
        self._ee_curve = self._ee_plot.plot(pen=pg.mkPen("y", width=2))
        self._ee_dot = self._ee_plot.plot(
            pen=None, symbol="o", symbolSize=10, symbolBrush="r")

        plots.addWidget(self._angle_plot)
        plots.addWidget(self._ee_plot)
        layout.addLayout(plots, stretch=1)

        layout.addWidget(self._build_action_panel())

        # Bottom read-outs
        self._ee_lbl = QtWidgets.QLabel("end effector: -")
        self._status_lbl = QtWidgets.QLabel("controller: -")
        self._status_lbl.setWordWrap(True)
        layout.addWidget(self._ee_lbl)
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
        """Buttons and read-outs for the two ROS 2 actions."""
        box = QtWidgets.QGroupBox("Actions (controller_node)")
        grid = QtWidgets.QGridLayout(box)

        # Row 0: move to a target
        self._move_x = self._make_spin(4.0)
        self._move_y = self._make_spin(2.0)
        self._move_btn = QtWidgets.QPushButton("Move to target")
        self._move_btn.clicked.connect(self._on_move_clicked)
        grid.addWidget(QtWidgets.QLabel("Target"), 0, 0)
        grid.addWidget(QtWidgets.QLabel("x"), 0, 1)
        grid.addWidget(self._move_x, 0, 2)
        grid.addWidget(QtWidgets.QLabel("y"), 0, 3)
        grid.addWidget(self._move_y, 0, 4)
        grid.addWidget(self._move_btn, 0, 5)

        # Row 1: gripper only - Pick closes it, Place opens it
        self._pick_btn = QtWidgets.QPushButton("Pick (close gripper)")
        self._pick_btn.clicked.connect(self._on_pick_clicked)
        self._place_btn = QtWidgets.QPushButton("Place (open gripper)")
        self._place_btn.clicked.connect(self._on_place_clicked)
        grid.addWidget(QtWidgets.QLabel("Gripper"), 1, 0)
        grid.addWidget(self._pick_btn, 1, 2, 1, 2)
        grid.addWidget(self._place_btn, 1, 4, 1, 2)

        # Row 3: cancel + progress + feedback + result
        self._cancel_btn = QtWidgets.QPushButton("Cancel")
        self._cancel_btn.clicked.connect(self._on_cancel_clicked)
        self._progress = QtWidgets.QProgressBar()
        self._progress.setRange(0, 100)
        self._progress.setValue(0)
        self._server_lbl = QtWidgets.QLabel("action servers: not available")
        self._feedback_lbl = QtWidgets.QLabel("feedback: -")
        self._result_lbl = QtWidgets.QLabel("last result: -")
        self._result_lbl.setWordWrap(True)
        grid.addWidget(self._cancel_btn, 2, 5)
        grid.addWidget(self._progress, 2, 0, 1, 5)
        grid.addWidget(self._server_lbl, 3, 5)
        grid.addWidget(self._feedback_lbl, 3, 0, 1, 5)
        grid.addWidget(self._result_lbl, 4, 0, 1, 6)
        return box

    @staticmethod
    def _draw_workspace(plot):
        """Dashed reach arc (upper half) and the ground line y = 0."""
        dashed = pg.mkPen((150, 150, 150), style=QtCore.Qt.DashLine)
        angles = [math.pi * k / 100 for k in range(101)]
        plot.plot([REACH * math.cos(a) for a in angles],
                  [REACH * math.sin(a) for a in angles], pen=dashed)
        plot.plot([-REACH, REACH], [0, 0], pen=pg.mkPen("w", width=1))

    # ---------------- ROS data slots (Qt thread) ----------------
    def _on_joint_state(self, t, angles):
        if self._t0 is None:
            self._t0 = t
        self._t.append(t - self._t0)
        for i in range(len(JOINTS)):
            self._angles[i].append(angles[i])
        self._last_rx = time.monotonic()
        self._dirty_angles = True

    def _on_end_effector(self, x, y):
        self._ee_x.append(x)
        self._ee_y.append(y)
        self._ee_lbl.setText(f"end effector: x = {x:.3f}, y = {y:.3f}")
        self._dirty_ee = True

    def _on_status(self, text):
        try:
            d = json.loads(text)
        except ValueError:
            self._status_lbl.setText(f"controller: {text}")
            return
        line = (f"controller: {d.get('state', '?')} / "
                f"{d.get('step') or '-'} : {d.get('message', '')}")
        if d.get("clipped"):
            target = d.get("target")
            used = [round(v, 2) for v in d.get("used_target", [])]
            line += f"   [target {target} clipped to {used}]"
        self._status_lbl.setText(line)

    # ---------------- action buttons and slots ----------------
    def _on_move_clicked(self):
        self._action_busy = True                 # block double clicks at once
        self._update_action_buttons()
        self._node.request_move(self._move_x.value(), self._move_y.value())

    def _on_pick_clicked(self):
        self._action_busy = True
        self._update_action_buttons()
        self._node.request_gripper(True)         # pick = close

    def _on_place_clicked(self):
        self._action_busy = True
        self._update_action_buttons()
        self._node.request_gripper(False)        # place = open

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
