"""
scene_node.py - the pick object, place zone and table as RViz markers.

Publishes visualization_msgs/MarkerArray on /scene_markers (frame "world").

  backend:=gazebo  Gazebo simulates the object (contact + friction). This
                   node only mirrors its pose from /pick_object/odometry
                   (bridged, see arm_description/config/bridge.yaml).
  backend:=sim     No physics. The node emulates a grasp: when the fingers
                   close with the object between them it follows the gripper
                   centre; when they open it stays where it was released.

Service /reset_scene (sim backend) puts the object back at its start point.
"""

import math

import rclpy
from nav_msgs.msg import Odometry
from rclpy.node import Node
from sensor_msgs.msg import JointState
from std_srvs.srv import Trigger
from visualization_msgs.msg import Marker, MarkerArray

from planar_arm_control.gripper_kinematics import tool_center
from planar_arm_control.planar_arm import PlanarArm

LINK_LENGTHS = [3.0, 2.0, 1.5]


class SceneNode(Node):
    def __init__(self):
        super().__init__("scene_node")
        d = self.declare_parameter
        self.backend = d("backend", "sim").value
        self.start = list(d("object_start", [4.0, 2.0]).value)
        self.radius = float(d("object_radius", 0.1).value)
        self.height = float(d("object_height", 0.15).value)
        self.zone = list(d("place_zone", [-3.0, 3.0]).value)
        self.table_z = float(d("table_height", 0.75).value)
        self.ee_center = float(d("ee_center_offset", 0.35).value)
        self.grasp_radius = float(d("grasp_radius", 0.12).value)
        self.grip_threshold = float(d("grip_threshold", 0.03).value)

        self.arm = PlanarArm(LINK_LENGTHS)
        self.obj_xy = [float(self.start[0]), float(self.start[1])]
        self.obj_yaw = 0.0
        self.obj_quat = (0.0, 0.0, 0.0, 1.0)     # gazebo mode only
        self.obj_z = self.height / 2.0
        self.held = False
        self.grasp_offset = (0.0, 0.0)           # object - gripper centre
        self.grasp_heading = 0.0
        self.yaw_at_grasp = 0.0

        self.pub = self.create_publisher(MarkerArray, "/scene_markers", 10)
        if self.backend == "gazebo":
            self.create_subscription(
                Odometry, "/pick_object/odometry", self._on_odom, 10)
        else:
            self.create_subscription(
                JointState, "/joint_states", self._on_joints, 10)
            self.create_service(Trigger, "/reset_scene", self._on_reset)
        self.create_timer(0.05, self._publish)
        self.get_logger().info(f"scene_node started (backend={self.backend})")

    # ---- gazebo: mirror the simulated object ---------------------------
    def _on_odom(self, msg):
        p, q = msg.pose.pose.position, msg.pose.pose.orientation
        self.obj_xy = [p.x, p.y]
        self.obj_z = p.z - self.table_z
        self.obj_quat = (q.x, q.y, q.z, q.w)

    # ---- sim: emulate the grasp -----------------------------------------
    def _on_joints(self, msg):
        pos = dict(zip(msg.name, msg.position))
        try:
            q = [pos["joint1"], pos["joint2"], pos["joint3"]]
        except KeyError:
            return
        finger = pos.get("ee_finger_left_joint", 0.0)
        centre = tool_center(self.arm, q, self.ee_center)
        heading = sum(q)
        closed = finger > self.grip_threshold
        if closed and not self.held:
            dx = self.obj_xy[0] - centre[0]
            dy = self.obj_xy[1] - centre[1]
            if math.hypot(dx, dy) < self.grasp_radius:
                self.held = True
                self.grasp_offset = (dx, dy)
                self.grasp_heading = heading
                self.yaw_at_grasp = self.obj_yaw
        elif not closed and self.held:
            self.held = False
        if self.held:
            dh = heading - self.grasp_heading
            c, s = math.cos(dh), math.sin(dh)
            ox, oy = self.grasp_offset
            self.obj_xy = [centre[0] + c * ox - s * oy,
                           centre[1] + s * ox + c * oy]
            self.obj_yaw = self.yaw_at_grasp + dh

    def _on_reset(self, _req, resp):
        self.held = False
        self.obj_xy = [float(self.start[0]), float(self.start[1])]
        self.obj_yaw = 0.0
        resp.success, resp.message = True, "object back at its start point"
        return resp

    # ---- markers ---------------------------------------------------------
    def _marker(self, mid, shape, xyz, scale, rgba, quat=(0.0, 0.0, 0.0, 1.0)):
        m = Marker()
        m.header.frame_id = "world"
        m.ns, m.id, m.type, m.action = "scene", mid, shape, Marker.ADD
        m.pose.position.x, m.pose.position.y, m.pose.position.z = xyz
        (m.pose.orientation.x, m.pose.orientation.y,
         m.pose.orientation.z, m.pose.orientation.w) = quat
        m.scale.x, m.scale.y, m.scale.z = scale
        m.color.r, m.color.g, m.color.b, m.color.a = rgba
        return m

    def _publish(self):
        quat = self.obj_quat if self.backend == "gazebo" else (
            0.0, 0.0, math.sin(self.obj_yaw / 2), math.cos(self.obj_yaw / 2))
        d = 2 * self.radius
        arr = MarkerArray()
        arr.markers = [
            # table top (surface at z = 0)
            self._marker(0, Marker.CUBE, (0.0, 3.0, -0.05), (14.0, 8.0, 0.1),
                         (0.55, 0.38, 0.2, 1.0)),
            self._marker(1, Marker.CYLINDER, (self.zone[0], self.zone[1], 0.002),
                         (0.5, 0.5, 0.004), (0.1, 0.8, 0.2, 1.0)),
            self._marker(2, Marker.CYLINDER,
                         (self.obj_xy[0], self.obj_xy[1], self.obj_z),
                         (d, d, self.height), (0.9, 0.1, 0.1, 1.0), quat),
        ]
        # stamp 0 = "latest transform", so it works with wall or sim time
        self.pub.publish(arr)


def main(args=None):
    rclpy.init(args=args)
    node = SceneNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    node.destroy_node()
    if rclpy.ok():
        rclpy.shutdown()


if __name__ == "__main__":
    main()
