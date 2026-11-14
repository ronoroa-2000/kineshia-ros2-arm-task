"""
ros2_control.launch.py - the arm the conventional ROS 2 way (demo).

  controller_node  --commands-->  forward_command controllers
                                       |
                              ros2_control_node (controller_manager)
                                       |
                    hardware interface from the URDF (fake arm here)
                                       |
                    joint_state_broadcaster --> /joint_states --> RViz, GUI

Starts: robot_state_publisher, ros2_control_node, the three controller
spawners, controller_node (backend: ros2_control) and, optionally, RViz.
For a real arm only the hardware plugin in urdf/ros2_control.urdf.xacro
changes; nothing else here does.

Usage:
  ros2 launch planar_arm_control ros2_control.launch.py
  ros2 launch planar_arm_control ros2_control.launch.py rviz:=false
"""

import os

import xacro
from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.conditions import IfCondition
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description():
    desc_share = get_package_share_directory('arm_description')

    xacro_path = os.path.join(desc_share, 'urdf', 'robot.urdf.xacro')
    controllers_yaml = os.path.join(desc_share, 'config', 'ros2_controllers.yaml')
    controller_yaml = os.path.join(desc_share, 'config', 'controller.yaml')
    rviz_config = os.path.join(desc_share, 'rviz', 'display.rviz')

    robot_description = xacro.process_file(
        xacro_path,
        mappings={'ros2_control': 'true', 'hardware': 'mock'}).toxml()

    rviz_arg = DeclareLaunchArgument(
        'rviz', default_value='true', description='Also open RViz')

    robot_state_publisher = Node(
        package='robot_state_publisher',
        executable='robot_state_publisher',
        output='screen',
        parameters=[{'robot_description': robot_description}],
    )

    # The controller manager loads the hardware described in the URDF
    control_node = Node(
        package='controller_manager',
        executable='ros2_control_node',
        output='screen',
        parameters=[controllers_yaml, {'robot_description': robot_description}],
    )

    spawners = [
        Node(package='controller_manager', executable='spawner',
             arguments=[name, '--controller-manager', '/controller_manager'],
             output='screen')
        for name in ('joint_state_broadcaster', 'arm_controller',
                     'gripper_controller')
    ]

    controller = Node(
        package='planar_arm_control',
        executable='controller_node',
        name='controller_node',
        output='screen',
        parameters=[controller_yaml, {'backend': 'ros2_control'}],
    )

    rviz = Node(
        package='rviz2',
        executable='rviz2',
        arguments=['-d', rviz_config],
        output='screen',
        condition=IfCondition(LaunchConfiguration('rviz')),
    )

    return LaunchDescription([rviz_arg, robot_state_publisher, control_node,
                              *spawners, controller, rviz])
