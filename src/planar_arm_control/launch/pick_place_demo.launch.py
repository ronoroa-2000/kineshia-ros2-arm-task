"""
pick_place_demo.launch.py - pick and place with a real object, in Gazebo and RViz.

  backend:=gazebo  Gazebo (physics, contact, friction) + controller + RViz.
                   RViz mirrors the arm from /joint_states and the object
                   from its Gazebo pose.
  backend:=sim     No Gazebo. RViz shows the arm and an object that follows
                   the gripper when it closes on it (scene_node emulates it).

Usage:
  ros2 launch planar_arm_control pick_place_demo.launch.py
  ros2 launch planar_arm_control pick_place_demo.launch.py backend:=sim
  ros2 launch planar_arm_control pick_place_demo.launch.py gz_gui:=false rviz:=false gui:=true

Then send the test scenario:
  ros2 action send_goal /pick_and_place arm_interfaces/action/PickAndPlace \
    "{pick: {x: 4.0, y: 2.0}, place: {x: -3.0, y: 3.0}}" --feedback
"""

import os

import xacro
from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription
from launch.conditions import IfCondition
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration, PythonExpression
from launch_ros.actions import Node


def generate_launch_description():
    desc = get_package_share_directory('arm_description')
    backend = LaunchConfiguration('backend')
    is_gz = PythonExpression(["'", backend, "' == 'gazebo'"])
    is_sim = PythonExpression(["'", backend, "' != 'gazebo'"])

    robot_description = xacro.process_file(
        os.path.join(desc, 'urdf', 'robot.urdf.xacro')).toxml()
    controller_yaml = os.path.join(desc, 'config', 'controller.yaml')
    rviz_config = os.path.join(desc, 'rviz', 'display.rviz')

    args = [
        DeclareLaunchArgument('backend', default_value='gazebo',
                              description='gazebo | sim'),
        DeclareLaunchArgument('gz_gui', default_value='true',
                              description='show the Gazebo window'),
        DeclareLaunchArgument('control_mode', default_value='position',
                              description='position | pid (PID tracking, gazebo only)'),
        DeclareLaunchArgument('rviz', default_value='true'),
        DeclareLaunchArgument('gui', default_value='false',
                              description='also start the Qt telemetry GUI'),
    ]

    # Gazebo: world + spawn + bridge + robot_state_publisher (sim time)
    gazebo = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            os.path.join(desc, 'launch', 'gazebo.launch.py')),
        launch_arguments={'gui': LaunchConfiguration('gz_gui')}.items(),
        condition=IfCondition(is_gz))

    # Without Gazebo the controller publishes /joint_states itself
    rsp_sim = Node(
        package='robot_state_publisher', executable='robot_state_publisher',
        parameters=[{'robot_description': robot_description}],
        condition=IfCondition(is_sim))

    controller = Node(
        package='planar_arm_control', executable='controller_node',
        name='controller_node', output='screen',
        parameters=[controller_yaml, {
            'backend': backend,
            'control_mode': LaunchConfiguration('control_mode'),
            # With Gazebo the controller runs on the simulation clock, so a
            # slow simulation (real-time factor < 1) slows the plan down too
            # instead of letting the arm fall behind it.
            'use_sim_time': is_gz}])

    scene = Node(
        package='planar_arm_control', executable='scene_node',
        name='scene_node', output='screen',
        parameters=[{'backend': backend}])

    rviz = Node(
        package='rviz2', executable='rviz2', arguments=['-d', rviz_config],
        parameters=[{'use_sim_time': is_gz}],
        condition=IfCondition(LaunchConfiguration('rviz')))

    gui = Node(
        package='planar_arm_control', executable='gui_node', name='gui_node',
        output='screen', condition=IfCondition(LaunchConfiguration('gui')))

    return LaunchDescription(args + [gazebo, rsp_sim, controller, scene,
                                     rviz, gui])
