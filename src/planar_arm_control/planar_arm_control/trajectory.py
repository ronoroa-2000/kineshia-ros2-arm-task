"""
trajectory.py - time-parameterised motion profiles (no ROS).

  TrapezoidalProfile     joint-space move, all joints synchronised
  CartesianLineProfile   straight line of the gripper centre, converted to
                         joint angles with IK and timed with a trapezoid
  PathError              raised when a planned path cannot be used

Both profiles share one interface: attribute T (duration in s) and
sample(t) -> (q, qd), so the controller can use either.
"""

import math

import numpy as np

from planar_arm_control.gripper_kinematics import center_ik, tool_center


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


class PathError(Exception):
    """A planned path cannot be used (limits, ground, IK jump, ...)."""


class CartesianLineProfile:
    """
    Straight line for the tool tip from its current position to p1.

    The line is sampled every `step` metres. Each sample is converted to
    joint angles with IK, using the previous solution as the guess so the
    arm stays on one configuration. A trapezoid over the path length s
    (tip speed v_tip, acceleration a_tip) gives the timing; it is slowed
    down if a joint would exceed joint_v_max. Same interface as
    TrapezoidalProfile: attribute T and sample(t) -> (q, qd).
    """

    def __init__(self, arm, q0, p1, v_tip, a_tip, joint_v_max,
                 step=0.02, max_jump=0.3, offset=0.0, ik_fn=None):
        # offset: link-3 tip -> the point that follows the line (gripper centre)
        # ik_fn(centre, guess) -> q or None; default: library IK + iteration
        if ik_fn is None:
            def ik_fn(centre, guess):
                return center_ik(arm, centre, guess, offset)
        self.q0 = np.asarray(q0, dtype=float)
        p0 = tool_center(arm, self.q0, offset)
        p1 = np.asarray(p1, dtype=float)
        length = float(np.linalg.norm(p1 - p0))
        self._trivial = length < 1e-6
        if self._trivial:                       # already there
            self.T = 0.0
            return

        n = max(2, int(math.ceil(length / step)) + 1)
        s_k = np.linspace(0.0, length, n)
        points = p0 + (p1 - p0) * (s_k / length)[:, None]
        q_k = np.zeros((n, 3))
        q_k[0] = self.q0
        for k in range(1, n):
            where = f"({points[k][0]:.2f}, {points[k][1]:.2f})"
            q = ik_fn(points[k], q_k[k - 1])
            if q is None:
                raise PathError(f"no IK solution on the straight line at "
                                f"{where}")
            err = float(np.linalg.norm(
                points[k] - tool_center(arm, q, offset)))
            if err > 1e-3:
                raise PathError(f"no IK solution on the straight line at "
                                f"{where}")
            if not arm.within_joint_limits(q):
                raise PathError(f"joint limit reached on the straight "
                                f"line at {where}")
            if not arm.arm_above_base(q):
                raise PathError(f"straight line goes below y = 0 at "
                                f"{where}")
            if float(np.max(np.abs(q - q_k[k - 1]))) > max_jump:
                raise PathError(f"arm configuration flips on the straight "
                                f"line near {where}")
            q_k[k] = q

        dq_ds = np.gradient(q_k, s_k, axis=0)   # joint change per metre
        peak = max(float(np.max(np.abs(dq_ds))), 1e-9)
        v = min(v_tip, joint_v_max / peak)      # respect joint speed limit
        a = a_tip * (v / v_tip)
        self._s, self._q, self._dq_ds = s_k, q_k, dq_ds
        self._time = TrapezoidalProfile([0.0], [length], v, a)
        self.T = self._time.T

    def sample(self, t):
        """Return (q, qd) as arrays at time t seconds from the start."""
        if self._trivial:
            return self.q0.copy(), np.zeros(3)
        s, speed = self._time.sample(t)
        s, speed = float(s[0]), float(speed[0])
        q = np.array([np.interp(s, self._s, self._q[:, j])
                      for j in range(3)])
        qd = np.array([np.interp(s, self._s, self._dq_ds[:, j])
                       for j in range(3)]) * speed
        return q, qd
