"""
controller_rviz.launch.py - controller_node + robot model + RViz in one launch.

Starts:
  - robot_state_publisher  (URDF from arm_description, TF from /joint_states)
  - controller_node        (parameters from arm_description/config/controller.yaml)
  - rviz2                  (arm_description/rviz/display.rviz)

Usage:
  ros2 launch planar_arm_control controller_rviz.launch.py
  ros2 launch planar_arm_control controller_rviz.launch.py backend:=gazebo
"""

import os

import xacro
from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description():
    desc_share = get_package_share_directory('arm_description')

    xacro_path = os.path.join(desc_share, 'urdf', 'robot.urdf.xacro')
    controller_yaml = os.path.join(desc_share, 'config', 'controller.yaml')
    rviz_config = os.path.join(desc_share, 'rviz', 'display.rviz')

    robot_description = xacro.process_file(xacro_path).toxml()

    backend_arg = DeclareLaunchArgument(
        'backend', default_value='sim',
        description='Command backend for controller_node: sim | gazebo')

    robot_state_publisher = Node(
        package='robot_state_publisher',
        executable='robot_state_publisher',
        output='screen',
        parameters=[{'robot_description': robot_description}],
    )

    controller = Node(
        package='planar_arm_control',
        executable='controller_node',
        name='controller_node',
        output='screen',
        # The YAML is loaded first; the launch argument overrides `backend`.
        parameters=[controller_yaml,
                    {'backend': LaunchConfiguration('backend')}],
    )

    rviz = Node(
        package='rviz2',
        executable='rviz2',
        name='rviz2',
        arguments=['-d', rviz_config],
        output='screen',
    )

    return LaunchDescription([backend_arg, robot_state_publisher,
                              controller, rviz])
