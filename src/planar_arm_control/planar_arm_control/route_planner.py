"""
route_planner.py - route search around the target (no ROS).

Before the arm goes to the point behind a target, the route is checked
against the clearance around that target. If the direct joint-space move
would sweep over it, via points are tried (cheapest first): one point on a
polar grid (radius x azimuth around the base), then the "polar route"
(pull in at the current azimuth, swing round, then go out).
"""

import math

import numpy as np

from planar_arm_control.trajectory import TrapezoidalProfile


class RoutePlanner:
    def __init__(self, arm, safety, ik_fn, ee_fn, v_max, a_max):
        """
        safety  a SafetyChecker
        ik_fn   ik_fn(centre, guess) -> joint angles or None (gripper centre)
        ee_fn   ee_fn(q) -> position of the gripper centre
        v_max, a_max  joint speed / acceleration used to time each leg
        """
        self.arm = arm
        self.safety = safety
        self.ik_fn = ik_fn
        self.ee_fn = ee_fn
        self.v_max = v_max
        self.a_max = a_max

    def leg_ok(self, qa, qb, obstacle):
        profile = TrapezoidalProfile(qa, qb, self.v_max, self.a_max)
        return (self.safety.clear_of(profile, obstacle)
                and self.safety.validate_profile(profile)[0])

    def plan_route(self, q_from, q_to, obstacle):
        """
        Joint-space legs from q_from to q_to that never bring the arm closer
        than the clearance to `obstacle` (the target). Returns the list of
        via-point joint configurations (empty = direct), or None.
        """
        obstacle = np.asarray(obstacle, dtype=float)
        if self.leg_ok(q_from, q_to, obstacle):
            return []

        centre_now = self.ee_fn(q_from)
        az_now = math.atan2(centre_now[1], centre_now[0])
        az_obs = math.atan2(obstacle[1], obstacle[0])
        radii = [1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 5.0]
        candidates = [[(r, math.radians(a))] for r in radii
                      for a in range(0, 181, 15)]
        candidates += [[(r, az_now), (r, az_obs)] for r in radii]

        scored = []
        for cand in candidates:
            chain, guess, ok = [q_from], q_from, True
            for r, az in cand:
                qv = self.ik_fn((r * math.cos(az), r * math.sin(az)), guess)
                if qv is None:
                    ok = False
                    break
                chain.append(qv)
                guess = qv
            if not ok:
                continue
            chain.append(np.asarray(q_to, dtype=float))
            cost = sum(float(np.max(np.abs(b - a)))
                       for a, b in zip(chain[:-1], chain[1:]))
            scored.append((cost + 0.2 * len(cand), chain))

        for _, chain in sorted(scored, key=lambda item: item[0]):
            if all(self.leg_ok(a, b, obstacle)
                   for a, b in zip(chain[:-1], chain[1:])):
                return chain[1:-1]
        return None
