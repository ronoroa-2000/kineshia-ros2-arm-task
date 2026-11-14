"""
backends.py - how a command leaves the controller (swap for hardware later).

The planner and the controller never talk to the arm directly; every tick
they call backend.send(stamp, q, qd, gripper). To use real hardware, add a
class with the same send() (for example a DynamixelBackend that writes goal
positions over the bus) and select it with the `backend` parameter.

  SimBackend     no physics: publishes the commanded state as /joint_states
  GazeboBackend  publishes position commands bridged to Gazebo
  Ros2ControlBackend  publishes joint positions to ros2_control controllers
                 (arm_controller, gripper_controller); the hardware behind
                 them (fake, or a Dynamixel driver) is chosen in the URDF
  make_backend   picks one by name (the `backend` parameter)
"""

from abc import ABC, abstractmethod

from sensor_msgs.msg import JointState
from std_msgs.msg import Float64, Float64MultiArray

ARM_JOINTS = ["joint1", "joint2", "joint3"]
FINGER_JOINTS = ["ee_finger_left_joint", "ee_finger_right_joint"]


class ArmBackend(ABC):
    @abstractmethod
    def send(self, stamp, q, qd, gripper):
        """Deliver one command. q/qd: 3 arm joints, gripper: finger position."""


class SimBackend(ArmBackend):
    """No physics: just publish the commanded state as /joint_states."""

    def __init__(self, node):
        self._pub = node.create_publisher(JointState, "/joint_states", 10)

    def send(self, stamp, q, qd, gripper):
        msg = JointState()
        msg.header.stamp = stamp
        msg.name = ARM_JOINTS + FINGER_JOINTS
        msg.position = [float(v) for v in q] + [float(gripper)] * 2
        msg.velocity = [float(v) for v in qd] + [0.0, 0.0]
        self._pub.publish(msg)


class GazeboBackend(ArmBackend):
    """Publish position commands bridged to Gazebo (see config/bridge.yaml)."""

    def __init__(self, node):
        self._pubs = {
            name: node.create_publisher(
                Float64, f"/planar_arm/{name}/cmd_pos", 10)
            for name in ARM_JOINTS + FINGER_JOINTS
        }

    def send(self, stamp, q, qd, gripper):
        values = [float(v) for v in q] + [float(gripper)] * 2
        for pub, value in zip(self._pubs.values(), values):
            pub.publish(Float64(data=value))


class Ros2ControlBackend(ArmBackend):
    """
    The conventional ROS 2 arm setup: positions go to forward_command
    controllers (config/ros2_controllers.yaml) of ros2_control, which pass
    them to the hardware interface chosen in the URDF. /joint_states then
    comes from joint_state_broadcaster, so this backend publishes none.
    """

    def __init__(self, node):
        self._arm = node.create_publisher(
            Float64MultiArray, "/arm_controller/commands", 10)
        self._gripper = node.create_publisher(
            Float64MultiArray, "/gripper_controller/commands", 10)

    def send(self, stamp, q, qd, gripper):
        self._arm.publish(Float64MultiArray(data=[float(v) for v in q]))
        # one driven finger joint; the right finger mimics it in the URDF
        self._gripper.publish(Float64MultiArray(data=[float(gripper)]))


# Future: class DynamixelBackend(ArmBackend): write goal positions over the bus.


def make_backend(name, node):
    """Create the backend selected by the `backend` parameter."""
    if name == "gazebo":
        return GazeboBackend(node)
    if name == "ros2_control":
        return Ros2ControlBackend(node)
    if name != "sim":
        node.get_logger().warn(f"unknown backend '{name}'; using sim")
    return SimBackend(node)
