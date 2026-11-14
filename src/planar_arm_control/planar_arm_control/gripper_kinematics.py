"""
gripper_kinematics.py - gripper-centre kinematics built on top of planar_arm.PlanarArm.

Pure Python/NumPy (no ROS). planar_arm.py itself is never modified; it is
only used for forward kinematics, joint limits and the ground check.

  tool_center  forward kinematics of the point between the fingers
  center_ik    IK via the library's IK (iterates, joint 3 stays at 0)
  heading_ik   our own analytic IK with the gripper heading as free choice
"""

import contextlib
import io
import math

import numpy as np


def tool_center(arm, q, offset):
    """Gripper centre: the link-3 tip pushed `offset` m along the tool axis."""
    tip = np.asarray(arm.end_effector(q), dtype=float)
    phi = float(np.sum(q))                       # tool direction q1 + q2 + q3
    return tip + offset * np.array([math.cos(phi), math.sin(phi)])


def center_ik(arm, center, guess, offset, iterations=30, tol=1e-7):
    """
    IK for the gripper CENTRE (the library solves for the link-3 tip).
    tip = center - offset * (cos phi, sin phi), but phi comes from the
    solution itself, so iterate: IK(tip) -> phi -> new tip, until it settles.
    """
    center = np.asarray(center, dtype=float)
    tip = center.copy()
    q = np.asarray(guess, dtype=float)
    for _ in range(iterations):
        with contextlib.redirect_stdout(io.StringIO()):     # library prints
            q = np.array(arm.inverse_kinematics(tip, initial_guess=guess),
                         dtype=float)
        phi = float(np.sum(q))
        new_tip = center - offset * np.array([math.cos(phi), math.sin(phi)])
        if float(np.linalg.norm(new_tip - tip)) < tol:
            break
        tip = new_tip
    return q


def heading_ik(arm, center, guess, offset, ee_length, pref=None,
               w_head=8.0, coarse=721, fine=41):
    """
    Analytic IK for the gripper CENTRE with the gripper heading phi as the
    free choice (textbook form). The provided planar_arm IK is NOT used here.

      1. wrist = centre - (L3 + u) * (cos phi, sin phi)       u = offset
      2. cos(q2) = (xw^2 + yw^2 - L1^2 - L2^2) / (2 L1 L2),  q2 = +/- acos
      3. q1 = atan2(yw, xw) - atan2(L2 sin q2, L1 + L2 cos q2)
      4. q3 = phi - q1 - q2

    phi is scanned over the full circle (centred on the direction of the
    target, so the workspace edge phi = direction is always a sample); every
    candidate must respect the joint limits and keep the arm and the
    fingertips above y = 0. Among the valid ones the cheapest wins:
        cost = sum|q - guess|  +  w_head * |phi - pref|
    pref defaults to the direction from the base to the target (gripper
    pointing outward); pass the approach heading to keep it constant along
    a line. phi is then refined locally so joints change smoothly.
    Returns q (array of 3) or None if there is no valid solution.
    """
    L1, L2, L3 = arm.link_lengths
    lim = arm.joint_limits
    center = np.asarray(center, dtype=float)
    guess = np.asarray(guess, dtype=float)
    psi = math.atan2(center[1], center[0])
    pref = psi if pref is None else float(pref)

    def search(phis):
        best_cost, best_q, best_phi = np.inf, None, None
        wrist = center[:, None] - (L3 + offset) * np.stack(
            [np.cos(phis), np.sin(phis)])
        c2 = ((wrist[0] ** 2 + wrist[1] ** 2 - L1 ** 2 - L2 ** 2)
              / (2.0 * L1 * L2))
        reachable = np.abs(c2) <= 1.0 + 1e-9
        c2 = np.clip(c2, -1.0, 1.0)
        for sign in (1.0, -1.0):                       # the two elbows
            q2 = sign * np.arccos(c2)
            q1 = (np.arctan2(wrist[1], wrist[0])
                  - np.arctan2(L2 * np.sin(q2), L1 + L2 * np.cos(q2)))
            q3 = phis - q1 - q2
            q = arm.wrap_angle(np.stack([q1, q2, q3], axis=1))   # n x 3
            valid = reachable & np.all(
                (q >= lim[:, 0] - 1e-9) & (q <= lim[:, 1] + 1e-9), axis=1)
            th1 = q[:, 0]
            th2 = th1 + q[:, 1]
            th3 = th2 + q[:, 2]
            y1 = L1 * np.sin(th1)
            y2 = y1 + L2 * np.sin(th2)
            y3 = y2 + L3 * np.sin(th3)
            y_far = y3 + ee_length * np.sin(th3)       # fingertips
            valid &= ((y1 >= -1e-6) & (y2 >= -1e-6)
                      & (y3 >= -1e-6) & (y_far >= -1e-6))
            move = np.sum(np.abs(arm.wrap_angle(q - guess)), axis=1)
            tilt = np.abs(arm.wrap_angle(phis - pref))
            cost = np.where(valid, move + w_head * tilt, np.inf)
            i = int(np.argmin(cost))
            if cost[i] < best_cost:
                best_cost, best_q, best_phi = cost[i], q[i], float(phis[i])
        return best_cost, best_q, best_phi

    cost, q, phi = search(psi + np.linspace(-np.pi, np.pi, coarse))
    if q is None:
        return None
    step = 2.0 * np.pi / (coarse - 1)
    cost2, q2, _ = search(phi + np.linspace(-step, step, fine))
    return q2 if (q2 is not None and cost2 <= cost) else q
