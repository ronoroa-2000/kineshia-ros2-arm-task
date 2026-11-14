"""
params.py - the controller settings, as plain fields.

ControllerParams holds every setting with its default (the same values as
config/controller.yaml). The dataclass itself needs no ROS, so tests can
build it directly; from_node() declares the ROS parameters on a node and
reads them, applying the same guards as before.

  yaml / ROS parameter       field
  publish_rate_hz            rate           Hz      control-loop rate
  max_joint_velocity         v_max          rad/s   joint cruise speed
  max_joint_acceleration     a_max          rad/s^2 joint acceleration
  approach_distance          approach_d     m       back-off behind a target
  max_tip_velocity           v_tip          m/s     gripper speed on lines
  max_tip_acceleration       a_tip          m/s^2   gripper accel on lines
  linear_step                line_step      m       IK sample spacing on lines
  ee_center_offset           ee_center      m       link-3 tip -> between fingers
  ee_length                  ee_length      m       link-3 tip -> fingertips
  ik_mode                    ik_mode        -       heading | library
  avoid_target               avoid          -       keep clear of the target
  clearance_links            clear_links    m       links -> target
  clearance_gripper          clear_grip     m       gripper -> target
  gripper_time               grip_time      s       open / close time
  gripper_closed_position    grip_closed    m       finger travel when closed
  unreachable_policy         policy         -       clip | reject
  initial_joint_angles       initial_q      rad     starting pose
  backend                    backend        -       sim | gazebo
  control_mode               control_mode   -       position (only one so far)
"""

from dataclasses import dataclass, field


@dataclass
class ControllerParams:
    rate: float = 50.0
    control_mode: str = "position"
    backend: str = "sim"
    v_max: float = 1.0
    a_max: float = 2.0
    approach_d: float = 0.5
    v_tip: float = 0.5
    a_tip: float = 1.0
    line_step: float = 0.02
    ee_center: float = 0.35
    ee_length: float = 0.55
    ik_mode: str = "heading"
    avoid: bool = True
    clear_links: float = 0.3
    clear_grip: float = 0.25
    grip_time: float = 0.5
    grip_closed: float = 0.1
    policy: str = "clip"
    initial_q: list = field(default_factory=lambda: [0.0, 0.0, 0.0])

    @classmethod
    def from_node(cls, node):
        """Declare the ROS parameters on `node`, read them, return the fields."""
        d = cls()                                  # defaults live in one place
        declare = node.declare_parameter
        declare("publish_rate_hz", d.rate)
        declare("control_mode", d.control_mode)    # velocity: stretch goal
        declare("backend", d.backend)              # sim | gazebo
        declare("max_joint_velocity", d.v_max)
        declare("max_joint_acceleration", d.a_max)
        declare("approach_distance", d.approach_d)
        declare("max_tip_velocity", d.v_tip)
        declare("max_tip_acceleration", d.a_tip)
        declare("linear_step", d.line_step)
        declare("ee_center_offset", d.ee_center)
        declare("ee_length", d.ee_length)
        declare("ik_mode", d.ik_mode)              # heading | library
        declare("avoid_target", d.avoid)
        declare("clearance_links", d.clear_links)
        declare("clearance_gripper", d.clear_grip)
        declare("gripper_time", d.grip_time)
        declare("gripper_closed_position", d.grip_closed)
        declare("unreachable_policy", d.policy)    # clip | reject
        declare("initial_joint_angles", list(d.initial_q))

        get = node.get_parameter
        p = cls(
            rate=float(get("publish_rate_hz").value),
            control_mode=str(get("control_mode").value),
            backend=str(get("backend").value),
            v_max=max(float(get("max_joint_velocity").value), 1e-3),
            a_max=max(float(get("max_joint_acceleration").value), 1e-3),
            approach_d=float(get("approach_distance").value),
            v_tip=max(float(get("max_tip_velocity").value), 1e-3),
            a_tip=max(float(get("max_tip_acceleration").value), 1e-3),
            line_step=max(float(get("linear_step").value), 1e-3),
            ee_center=float(get("ee_center_offset").value),
            ee_length=float(get("ee_length").value),
            ik_mode=str(get("ik_mode").value),
            avoid=bool(get("avoid_target").value),
            clear_links=float(get("clearance_links").value),
            clear_grip=float(get("clearance_gripper").value),
            grip_time=max(float(get("gripper_time").value), 1e-3),
            grip_closed=float(get("gripper_closed_position").value),
            policy=str(get("unreachable_policy").value),
            initial_q=[float(v) for v in get("initial_joint_angles").value],
        )
        if p.ik_mode not in ("heading", "library"):
            node.get_logger().warn(
                f"unknown ik_mode '{p.ik_mode}'; using heading")
            p.ik_mode = "heading"
        if p.control_mode != "position":
            node.get_logger().warn(
                f"control_mode '{p.control_mode}' not implemented yet; "
                f"using position")
        return p
