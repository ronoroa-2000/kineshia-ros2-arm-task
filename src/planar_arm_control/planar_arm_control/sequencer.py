"""
sequencer.py - runs one command at a time, step by step (no ROS).

MotionSequencer turns a command (move to a target, close / open the gripper,
pick and place) into a queue of steps and executes them one after another:

    start_move / start_gripper / start_pick_place   build the queue
    start_pick / start_place                        pick or place at one point
    step()                                          advance one control tick
    stop() / cancel(task)                           halt where it is

After every step() the current command to send is in q, qd and gripper.
There is no ROS here: time comes from the `clock` function that is passed in
(so a test can use a fake clock), and status messages / warnings leave
through the on_status / on_warn callbacks. The class is not thread-safe; the
node takes its lock around every call.

A Task is the bookkeeping for one command (progress, result, a done event
that an action server can wait on); `handle` is opaque to this module.
"""

import math
import threading

import numpy as np

from planar_arm_control.route_planner import RoutePlanner
from planar_arm_control.trajectory import (
    CartesianLineProfile, PathError, TrapezoidalProfile)


class Task:
    """One command: kind is "move", "pick_place", "grasp" or "gripper"."""

    def __init__(self, kind, handle=None, total_steps=1, done_message=""):
        self.kind = kind
        self.handle = handle                 # e.g. an action goal handle
        self.done = threading.Event()        # set when the command finished
        self.success = False
        self.message = ""
        self.failed_step = ""
        self.clipped = False
        self.step_index = 0
        self.total_steps = total_steps
        self.done_message = done_message


class MotionSequencer:
    def __init__(self, model, params, clock, on_status, on_warn):
        """
        model      ArmModel (kinematics, IK, safety checks)
        params     ControllerParams
        clock      clock() -> seconds
        on_status  on_status(dict) called whenever the status changes
        on_warn    on_warn(str) called for warnings
        """
        self.model = model
        self.p = params
        self.arm = model.arm
        self.safety = model.safety
        self._clock = clock
        self._on_status = on_status
        self._warn = on_warn
        self._planner = RoutePlanner(
            self.arm, self.safety, model.ik_center, model.ee_pos,
            params.v_max, params.a_max)

        q0 = np.array(params.initial_q, dtype=float)
        if not (self.arm.within_joint_limits(q0)
                and self.arm.arm_above_base(q0)):
            on_warn("initial_joint_angles invalid; using zeros")
            q0 = np.zeros(3)

        # what to send (updated by step())
        self.q = q0
        self.qd = np.zeros(3)
        self.gripper = 0.0                   # finger position, 0 = open
        # what is going on (read by the node for feedback)
        self.task = None                     # running command
        self.reserved = False                # an action goal was accepted
        self.active = None                   # None | "move" | "gripper"
        self.step_label = ""
        self.action_frac = 0.0               # progress 0..1 of this step
        self.goal_xy = None                  # target of the current move
        # internal
        self._queue = []
        self._profile = None
        self._t0 = 0.0
        self._grip_from = 0.0
        self._grip_to = 0.0
        self._last_info = {}
        self._pre_point = None               # point "behind" the last target
        self._approach_heading = None        # gripper heading of that approach

    # ------------------------------------------------------------------
    # status and small helpers
    # ------------------------------------------------------------------
    def report(self, state, message, info=None):
        """Send a status message through the on_status callback."""
        if info is not None:
            self._last_info = info
        self._on_status({"state": state, "step": self.step_label,
                         "message": message, **self._last_info})

    def busy(self):
        return self.active is not None or self.reserved

    def ee(self):
        """Gripper centre for the current joint angles."""
        return self.model.ee_pos(self.q)

    def _solve(self, xy, allow_clip, guess=None, heading=None):
        return self.model.solve_ik(
            xy, allow_clip, self.q if guess is None else guess, heading)

    def _default_info(self, xy):
        """Status info for a step that has none of its own."""
        if self._last_info:
            return dict(self._last_info)
        return {"target": [float(xy[0]), float(xy[1])], "clipped": False,
                "used_target": [float(xy[0]), float(xy[1])], "message": "ok"}

    # ------------------------------------------------------------------
    # task bookkeeping (one running command at a time)
    # ------------------------------------------------------------------
    def _new_task(self, kind, handle=None, total_steps=1, done_message=""):
        self._last_info = {}
        self.task = Task(kind, handle, total_steps, done_message)
        return self.task

    def _finish_task(self, success, message, failed_step=""):
        """Mark the running command as finished (wakes whoever waits on it)."""
        task = self.task
        if task is None:
            return
        task.success = success
        task.message = message
        task.failed_step = failed_step
        task.clipped = bool(self._last_info.get("clipped", False))
        task.done.set()
        self.task = None

    def _halt_motion(self):
        self._queue.clear()
        self._profile = None
        self.active = None
        self.qd = np.zeros(3)
        self.step_label = ""
        self.goal_xy = None

    def stop(self, message="stopped by operator"):
        """Halt where the arm is and end the running command as failed."""
        self._halt_motion()
        self._finish_task(False, message)
        self.report("idle", message)

    def cancel(self, task):
        """Cancel `task` (an action goal) if it is still the running one."""
        if self.task is task:
            self._halt_motion()
            self.report("idle", "canceled by client")
            self._finish_task(False, "canceled")

    def _abort(self, message, info=None):
        self._warn(f"aborted: {message}")
        step = self.step_label
        self._queue.clear()
        self._profile = None
        self.active = None
        self.qd = np.zeros(3)
        self.report("error", message, info)
        self._finish_task(False, message, step)

    def _reject_request(self, step, message, info):
        """A request that is invalid before any motion starts."""
        self._warn(f"request rejected: {message}")
        self.step_label = step
        self.report("error", message, info)
        self._finish_task(False, message, step)

    # ------------------------------------------------------------------
    # starting commands
    # ------------------------------------------------------------------
    def start_move(self, xy, handle=None):
        task = self._new_task("move", handle, total_steps=2)
        self._queue = [{
            "type": "approach_target", "label": "move_to_target",
            "xy": (float(xy[0]), float(xy[1])),
            "allow_clip": self.p.policy == "clip",
        }]
        self._begin_next_action()
        return task

    def start_gripper(self, close, handle=None):
        """Gripper only: close (pick) or open (place), no arm motion."""
        task = self._new_task(
            "gripper", handle,
            done_message="gripper closed" if close else "gripper opened")
        self._queue = [{
            "type": "gripper",
            "label": "grip" if close else "release",
            "value": self.p.grip_closed if close else 0.0,
        }]
        self._begin_next_action()
        return task

    def _outside_reach(self, name, pt):
        """(message, info) if pt is beyond the reach, else None.

        A pick or place is never done at a clipped point."""
        reach_c = self.model.reach_c
        dist = float(np.hypot(*pt))
        if dist <= reach_c:
            return None
        info = {"target": list(pt), "clipped": True,
                "used_target": (np.asarray(pt) * reach_c / dist).tolist()}
        message = (f"{name} target {pt} is outside the workspace "
                   f"(reach {reach_c:.2f})")
        return message, info

    def _build_pick_place(self, pick, place):
        """Return (queue, bad_step, message, info); queue is None on error."""
        for name, pt in (("pick", pick), ("place", place)):
            bad = self._outside_reach(name, pt)
            if bad is not None:
                return None, name, bad[0], bad[1]

        def approach(label, xy):
            return {"type": "approach_target", "label": label, "xy": xy,
                    "allow_clip": False}

        def retreat(label):                      # straight back to "behind"
            return {"type": "line", "label": label, "xy": None}

        def grip(label, value):
            return {"type": "gripper", "label": label, "value": value}

        queue = [
            approach("pick", pick),              # behind pick, then straight in
            grip("grip", self.p.grip_closed),    # pick: close the gripper
            retreat("pick_retreat"),
            approach("place", place),            # behind place, then straight in
            grip("release", 0.0),                # place: open the gripper
            retreat("place_retreat"),
        ]
        return queue, "", "", None

    def start_pick(self, xy, handle=None):
        """Pick at xy: approach it, close the gripper, back straight out."""
        return self._start_grasp(xy, True, handle)

    def start_place(self, xy, handle=None):
        """Place at xy: approach it, open the gripper, back straight out."""
        return self._start_grasp(xy, False, handle)

    def _start_grasp(self, xy, close, handle):
        name = "pick" if close else "place"
        xy = (float(xy[0]), float(xy[1]))
        # approach_target counts as two steps, then the gripper and the retreat
        task = self._new_task("grasp", handle, total_steps=4,
                              done_message=f"{name} complete")
        bad = self._outside_reach(name, xy)
        if bad is not None:
            self._reject_request(name, bad[0], bad[1])
            return task
        self._queue = [
            {"type": "approach_target", "label": name, "xy": xy,
             "allow_clip": False},
            {"type": "gripper", "label": "grip" if close else "release",
             "value": self.p.grip_closed if close else 0.0},
            {"type": "line", "label": f"{name}_retreat", "xy": None},
        ]
        self._begin_next_action()
        return task

    def start_pick_place(self, pick, place, handle=None):
        pick = (float(pick[0]), float(pick[1]))
        place = (float(place[0]), float(place[1]))
        queue, step, message, info = self._build_pick_place(pick, place)
        # an approach_target counts as two steps (approach + straight line)
        steps = (sum(2 if a["type"] == "approach_target" else 1
                     for a in queue) if queue else 8)
        task = self._new_task("pick_place", handle, total_steps=steps)
        if queue is None:
            self._reject_request(step, message, info)
            return task
        self._queue = queue
        self._begin_next_action()
        return task

    # ------------------------------------------------------------------
    # the queue (single moves and pick-and-place share it)
    # ------------------------------------------------------------------
    def _begin_next_action(self):
        if not self._queue:
            task = self.task
            if task is not None and task.done_message:
                message = task.done_message
            elif task is not None and task.kind == "pick_place":
                message = "pick and place complete"
            elif self._last_info.get("clipped"):
                message = ("target outside the workspace; moved to the "
                           "nearest boundary point")
            else:
                message = "target reached"
            self.active = None
            self.step_label = ""
            self.goal_xy = None
            self.report("idle", "done")
            self._finish_task(True, message)
            return

        action = self._queue.pop(0)
        self.step_label = action["label"]
        self.action_frac = 0.0
        # an "approach_target" is replaced by real steps below
        if self.task is not None and action["type"] != "approach_target":
            self.task.step_index += 1

        if action["type"] == "gripper":
            self._grip_from = self.gripper
            self._grip_to = action["value"]
            self._t0 = self._clock()
            self.active = "gripper"
            self.report("gripping", action["label"])
            return

        if action["type"] == "approach_target":
            self._expand_approach(action)
            return

        # "move" (joint space) or "line" (straight line of the gripper centre)
        p = self.p
        label = action["label"]
        if action["type"] == "line":
            p1 = action["xy"] if action["xy"] is not None else self._pre_point
            info = action.get("info") or self._default_info(p1)
            heading = action.get("heading", self._approach_heading)
            try:
                profile = CartesianLineProfile(
                    self.arm, self.q, p1, p.v_tip, p.a_tip, p.v_max,
                    p.line_step, offset=p.ee_center,
                    ik_fn=lambda c, g: self.model.ik_center(c, g, heading))
            except PathError as err:
                self._abort(f"{label}: {err}", info)
                return
        elif action.get("q") is not None:            # precomputed (route legs)
            q_goal = np.asarray(action["q"], dtype=float)
            info = action.get("info") or self._default_info(
                self.model.ee_pos(q_goal))
            profile = TrapezoidalProfile(self.q, q_goal, p.v_max, p.a_max)
        else:
            q_goal, ik_info = self._solve(
                action["xy"], action["allow_clip"], action.get("guess"),
                action.get("heading"))
            info = action.get("info") or ik_info
            if q_goal is None:
                self._abort(f"{label}: {ik_info['message']}", info)
                return
            profile = TrapezoidalProfile(self.q, q_goal, p.v_max, p.a_max)

        ok, why = self.safety.validate_profile(profile)
        if not ok:
            self._abort(f"{label}: {why}", info)
            return
        self.goal_xy = info["used_target"]
        self._profile = profile
        self._t0 = self._clock()
        self.active = "move"
        self.report("moving", info.get("message", "ok"), info)

    def _expand_approach(self, action):
        """
        Replace a target by: (1) move to a point BEHIND it, (2) drive in a
        straight line onto it. "Behind" = backed off along the gripper
        axis: the direction the tool points when the arm is at the target.
        """
        q_goal, info = self._solve(action["xy"], action["allow_clip"])
        label = action["label"]
        if q_goal is None:
            self._abort(f"{label}: {info['message']}", info)
            return
        target = np.asarray(info["used_target"], dtype=float)
        heading = float(np.sum(q_goal))              # q1 + q2 + q3
        direction = np.array([math.cos(heading), math.sin(heading)])
        pre = target - self.p.approach_d * direction
        self._pre_point = (float(pre[0]), float(pre[1]))
        self._approach_heading = heading     # kept constant on the way in/out

        # joint angles at the point behind the target (same configuration
        # as the goal: guess = q_goal, heading kept)
        q_pre, pre_info = self._solve(self._pre_point, False, q_goal, heading)
        if q_pre is None:
            self._abort(f"{label}: point behind the target: "
                        f"{pre_info['message']}", info)
            return

        # route to that point that keeps the arm away from the target
        via = []
        if self.p.avoid:
            via = self._planner.plan_route(self.q, q_pre, target)
            if via is None:
                self._abort(f"{label}: no route to the point behind the "
                            f"target that stays clear of it", info)
                return
        if self.task is not None:
            self.task.total_steps += len(via)        # extra via legs
        steps = [{"type": "move", "label": f"{label}_via{i + 1}",
                  "q": qv, "info": info} for i, qv in enumerate(via)]
        steps.append({"type": "move", "label": f"{label}_approach",
                      "q": q_pre, "info": info})
        steps.append({"type": "line", "label": f"{label}_linear",
                      "xy": (float(target[0]), float(target[1])),
                      "heading": heading, "info": info})
        self._queue[0:0] = steps
        self._begin_next_action()

    # ------------------------------------------------------------------
    # one control tick
    # ------------------------------------------------------------------
    def step(self):
        """Advance the running step to the current time; updates q, qd, gripper."""
        now = self._clock()

        if self.active == "move":
            t = now - self._t0
            q, qd = self._profile.sample(t)
            self.q = self.safety.clip(q)
            self.qd = qd
            duration = self._profile.T
            self.action_frac = min(t / duration, 1.0) if duration > 0 else 1.0
            if t >= duration:
                self.qd = np.zeros(3)
                self._profile = None
                self.active = None
                self._begin_next_action()

        elif self.active == "gripper":
            frac = min(max((now - self._t0) / self.p.grip_time, 0.0), 1.0)
            self.action_frac = frac
            self.gripper = (self._grip_from
                            + (self._grip_to - self._grip_from) * frac)
            if frac >= 1.0:
                self.active = None
                self._begin_next_action()
