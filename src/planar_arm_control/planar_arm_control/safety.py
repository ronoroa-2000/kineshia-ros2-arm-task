"""
safety.py - limit, ground and clearance checks for planned motions (no ROS).

  arm_clearance   distance from a point to the arm links and to the gripper
  SafetyChecker   checks a motion profile against the joint limits, the
                  ground (links and fingertips) and the clearance around a
                  target that the arm must not sweep over

planar_arm.py is only used for its limit and ground checks.
"""

import math

import numpy as np


def arm_clearance(arm, q_samples, point, ee_length):
    """
    Distance from `point` to the arm for every row of q_samples (n x 3):
    returns (to_links, to_gripper). The links are base-P1-P2-tip; the gripper
    is the segment from the link-3 tip to the fingertips (ee_length ahead).
    """
    q = np.atleast_2d(np.asarray(q_samples, dtype=float))
    L1, L2, L3 = arm.link_lengths
    th1 = q[:, 0]
    th2 = th1 + q[:, 1]
    th3 = th2 + q[:, 2]
    p0 = np.zeros((len(q), 2))
    p1 = p0 + L1 * np.stack([np.cos(th1), np.sin(th1)], axis=1)
    p2 = p1 + L2 * np.stack([np.cos(th2), np.sin(th2)], axis=1)
    p3 = p2 + L3 * np.stack([np.cos(th3), np.sin(th3)], axis=1)
    far = p3 + ee_length * np.stack([np.cos(th3), np.sin(th3)], axis=1)
    point = np.asarray(point, dtype=float)

    def dist(a, b):                          # point to segment a-b, per row
        ab = b - a
        t = np.clip(np.sum((point - a) * ab, axis=1)
                    / np.maximum(np.sum(ab * ab, axis=1), 1e-12), 0.0, 1.0)
        return np.linalg.norm(point - (a + t[:, None] * ab), axis=1)

    links = np.minimum(np.minimum(dist(p0, p1), dist(p1, p2)), dist(p2, p3))
    return links, dist(p3, far)


class SafetyChecker:
    def __init__(self, arm, ee_length, clearance_links, clearance_gripper,
                 sample_dt=0.02):
        """
        ee_length          link-3 tip -> fingertips (m), for the ground check
        clearance_links    min distance links -> obstacle (m)
        clearance_gripper  min distance gripper -> obstacle (m)
        sample_dt          spacing (s) of the samples used to check a path
        """
        self.arm = arm
        self.ee_length = ee_length
        self.clearance_links = clearance_links
        self.clearance_gripper = clearance_gripper
        self.sample_dt = sample_dt

    def clip(self, q):
        """Force q into the joint limits (guards against rounding errors)."""
        return np.clip(q, self.arm.joint_limits[:, 0],
                       self.arm.joint_limits[:, 1])

    def gripper_above_ground(self, q):
        """The fingertips (end of the gripper) must not go below y = 0."""
        tip = np.asarray(self.arm.end_effector(q), dtype=float)
        phi = float(np.sum(q))
        far = tip + self.ee_length * np.array([math.cos(phi), math.sin(phi)])
        return bool(far[1] >= -1e-6)

    def validate_profile(self, profile):
        """Check every sample of the path against limits and the ground."""
        n = max(2, int(profile.T / self.sample_dt) + 1)
        for t in np.linspace(0.0, profile.T, n):
            q = self.clip(profile.sample(t)[0])
            if not self.arm.within_joint_limits(q):
                return False, f"joint limit violated at t={t:.2f}s"
            if not self.arm.arm_above_base(q):
                return False, f"arm would go below y = 0 at t={t:.2f}s"
            if not self.gripper_above_ground(q):
                return False, f"gripper would go below y = 0 at t={t:.2f}s"
        return True, ""

    def clear_of(self, profile, obstacle):
        """True if the arm keeps its distance from `obstacle` along profile."""
        n = max(3, int(profile.T / 0.03) + 1)
        q = np.array([self.clip(profile.sample(t)[0])
                      for t in np.linspace(0.0, profile.T, n)])
        links, grip = arm_clearance(self.arm, q, obstacle, self.ee_length)
        # if the arm already starts closer than the limit, it must not get
        # any closer than where it started
        lim_l = min(self.clearance_links, links[0] - 1e-3)
        lim_g = min(self.clearance_gripper, grip[0] - 1e-3)
        return bool(np.all(links >= lim_l) and np.all(grip >= lim_g))
