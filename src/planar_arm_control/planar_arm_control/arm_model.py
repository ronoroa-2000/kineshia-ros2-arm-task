"""
arm_model.py - the arm as the controller sees it (no ROS).

ArmModel bundles the provided PlanarArm with the gripper-centre kinematics
(gripper_kinematics.py) and the safety checks (safety.py), so the rest of
the code asks one object: "where is the gripper centre?", "which joint
angles reach this point?".

  ee_pos(q)                    gripper centre for joint angles q
  ik_center(centre, guess, h)  joint angles for a gripper centre, or None
  solve_ik(xy, ...)            same, plus reach check / clipping, validation
                               and a message that explains the outcome

planar_arm.py itself is only used, never modified.
"""

import numpy as np

from planar_arm_control.gripper_kinematics import (
    center_ik, heading_ik, tool_center)
# The provided kinematics library - do not modify it.
from planar_arm_control.planar_arm import PlanarArm
from planar_arm_control.safety import SafetyChecker

LINK_LENGTHS = [3.0, 2.0, 1.5]


class ArmModel:
    def __init__(self, params, link_lengths=LINK_LENGTHS):
        self.params = params
        self.arm = PlanarArm(link_lengths)
        self.reach = float(sum(self.arm.link_lengths))      # 6.5 m
        # The target point is the gripper centre, so it can lie this far out.
        self.reach_c = self.reach + params.ee_center
        self.safety = SafetyChecker(
            self.arm, params.ee_length, params.clear_links, params.clear_grip)

    def ee_pos(self, q):
        """Position of the end effector = centre between the fingers."""
        return tool_center(self.arm, q, self.params.ee_center)

    def ik_center(self, center, guess, heading=None):
        """
        Joint angles for the gripper centre, or None (see ik_mode).
        heading: keep the gripper pointing this way (strong preference),
        used on the approach line; None = prefer pointing outward.
        """
        p = self.params
        if p.ik_mode == "heading":
            return heading_ik(self.arm, center, guess, p.ee_center,
                              p.ee_length, pref=heading,
                              w_head=8.0 if heading is None else 20.0)
        return center_ik(self.arm, center, guess, p.ee_center)

    def solve_ik(self, xy, allow_clip, guess, heading=None):
        """
        IK so that the gripper CENTRE reaches xy.
        Return (q_goal or None, info). info always describes the outcome.
        """
        target = np.array(xy, dtype=float)
        dist = float(np.linalg.norm(target))
        clipped = bool(dist > self.reach_c + 1e-9)
        info = {"target": target.tolist(), "clipped": clipped,
                "used_target": target.tolist()}

        if clipped:
            if not allow_clip:
                info["message"] = (f"target {tuple(target)} is outside the "
                                   f"workspace (reach {self.reach_c:.2f})")
                return None, info
            target = target * (self.reach_c / dist)    # nearest boundary point
            info["used_target"] = target.tolist()

        q = self.ik_center(target, guess, heading)
        if q is None:
            info["message"] = ("no IK solution within the joint limits and "
                               "the ground constraint")
            return None, info

        if not self.arm.within_joint_limits(q):
            info["message"] = "IK solution violates joint limits"
            return None, info
        if not self.arm.arm_above_base(q):
            info["message"] = "IK solution puts the arm below y = 0"
            return None, info
        if not self.safety.gripper_above_ground(q):
            info["message"] = "IK solution puts the gripper below y = 0"
            return None, info
        err = float(np.linalg.norm(target - self.ee_pos(q)))
        if err > 1e-3:
            info["message"] = f"no valid IK solution (error {err:.3f})"
            return None, info

        info["message"] = ("target clipped to workspace boundary"
                           if clipped else "ok")
        return q, info
