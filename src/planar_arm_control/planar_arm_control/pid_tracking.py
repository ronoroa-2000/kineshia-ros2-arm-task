"""
pid_tracking.py - PID trajectory-tracking loop (outer loop around the backend).

The sequencer produces the reference trajectory q_ref(t). The arm (Gazebo or
real hardware) reports where it actually is on /joint_states. This module
closes the loop between the two:

    error        e  = q_ref - q_measured                  (per joint)
    correction   c  = Kp*e + Ki*integral(e) + Kd*de/dt
    feedforward  f  = T_ff * qd_ref      (cancels the servo's lag on a ramp)
    command      q_cmd = clip(q_ref + f + c, joint limits)

controller_node calls PidTracker.correct() every tick and sends q_cmd to the
backend instead of q_ref. The error and the correction are published for
the live error plot:

    /tracking_error   std_msgs/Float64MultiArray   [e1, e2, e3]  (rad)
    /pid_correction   std_msgs/Float64MultiArray   [c1, c2, c3]  (rad)

Only the three arm joints are tracked (not the gripper).

Safety:
  - no correction while no fresh /joint_states arrived (stale > max_age)
  - the integral is clamped (anti-windup) and reset whenever the arm is idle
  - the PID correction is limited to +/- max_correction, and q_cmd to the limits

Selected with  control_mode: pid  (needs backend gazebo or ros2_control: with
the `sim` backend /joint_states is the command itself, so there is no error).
"""

import threading

import numpy as np
from sensor_msgs.msg import JointState
from std_msgs.msg import Float64MultiArray

ARM_JOINTS = ("joint1", "joint2", "joint3")


class JointPid:
    """One PID with a clamped integral and a low-pass filtered derivative."""

    def __init__(self, kp, ki, kd, i_limit, d_filter=0.2):
        self.kp, self.ki, self.kd = kp, ki, kd
        self.i_limit = i_limit
        self.d_filter = d_filter       # 0..1, weight of the newest derivative
        self.reset()

    def reset(self):
        self.integral = 0.0
        self._prev_error = None
        self._d = 0.0

    def update(self, error, dt):
        if dt <= 0.0:
            return self.kp * error
        self.integral = float(np.clip(
            self.integral + error * dt, -self.i_limit, self.i_limit))
        if self._prev_error is not None:
            raw = (error - self._prev_error) / dt
            self._d += self.d_filter * (raw - self._d)
        self._prev_error = error
        return self.kp * error + self.ki * self.integral + self.kd * self._d


class PidTracker:
    def __init__(self, node, params, joint_limits):
        self._node = node
        self.enabled = params.control_mode == "pid"
        if self.enabled and params.backend == "sim":
            node.get_logger().warn(
                "control_mode 'pid' needs measured joint states; the sim "
                "backend has none (it publishes the command itself). "
                "PID disabled.")
            self.enabled = False

        self._monitor = params.backend != "sim"
        self._limits = np.asarray(joint_limits, dtype=float)      # (3, 2)
        self._max_corr = params.pid_max_correction
        self._ff_time = params.pid_ff_time
        self._max_age = params.pid_max_age
        self._pids = [
            JointPid(params.pid_kp[i], params.pid_ki[i], params.pid_kd[i],
                     params.pid_i_limit)
            for i in range(3)]

        self._lock = threading.Lock()
        self._meas = None                 # (q measured, monotonic time)
        self._last_t = None

        self._err_pub = node.create_publisher(
            Float64MultiArray, "/tracking_error", 10)
        self._corr_pub = node.create_publisher(
            Float64MultiArray, "/pid_correction", 10)
        # The subscription that feeds the loop: where the arm really is.
        node.create_subscription(
            JointState, "/joint_states", self._on_joint_states, 10)

        if self.enabled:
            node.get_logger().info(
                f"PID tracking on: kp={list(params.pid_kp)} "
                f"ki={list(params.pid_ki)} kd={list(params.pid_kd)}")

    # ---- measurement -----------------------------------------------------
    def _now(self):
        """Seconds on the node clock (simulation time with use_sim_time)."""
        return self._node.get_clock().now().nanoseconds * 1e-9

    def _on_joint_states(self, msg):
        pos = dict(zip(msg.name, msg.position))
        try:
            q = np.array([pos[j] for j in ARM_JOINTS], dtype=float)
        except KeyError:
            return
        with self._lock:
            self._meas = (q, self._now())

    def measured(self):
        """Latest measured joint angles, or None if none / stale."""
        with self._lock:
            meas = self._meas
        if meas is None or self._now() - meas[1] > self._max_age:
            return None
        return meas[0]

    # ---- the loop ----------------------------------------------------------
    def correct(self, q_ref, qd_ref, active):
        """
        Return the corrected command for the reference q_ref (3 joints);
        qd_ref is the reference velocity (for the feedforward term).
        active: True while the sequencer is running a motion; when idle the
        integral is cleared so it does not wind up while standing still.
        """
        q_ref = np.asarray(q_ref, dtype=float)
        if not self.enabled:
            # plain position mode: still publish the error (for comparison)
            q_meas = self.measured() if self._monitor else None
            if q_meas is not None:
                self._err_pub.publish(
                    Float64MultiArray(data=(q_ref - q_meas).tolist()))
            return q_ref

        now = self._now()
        dt = 0.0 if self._last_t is None else now - self._last_t
        self._last_t = now

        q_meas = self.measured()
        if q_meas is None:                       # no feedback: do not guess
            self._reset()
            return q_ref
        error = q_ref - q_meas
        corr = np.array([self._pids[i].update(float(error[i]), dt)
                         for i in range(3)])
        if not active:
            self._reset()
            corr = np.zeros(3)
        corr = np.clip(corr, -self._max_corr, self._max_corr)

        self._err_pub.publish(Float64MultiArray(data=error.tolist()))
        self._corr_pub.publish(Float64MultiArray(data=corr.tolist()))
        ff = self._ff_time * np.asarray(qd_ref, dtype=float)
        return np.clip(q_ref + ff + corr, self._limits[:, 0], self._limits[:, 1])

    def _reset(self):
        for pid in self._pids:
            pid.reset()
