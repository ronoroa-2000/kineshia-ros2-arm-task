"""
bringup.launch.py - controller + RViz + the telemetry GUI, without Gazebo.

Starts, via pick_place_demo.launch.py:
  - robot_state_publisher, controller_node, scene_node
  - RViz                                        (rviz:=true)
  - the Qt telemetry GUI                        (gui:=true)
  - Gazebo only with backend:=gazebo            (off by default)

Do NOT use the GUI's "Launch RViz" button here: RViz is already running.

Arguments (defaults in brackets):
  backend       gazebo | sim                [sim]
  control_mode  pid | position              [position]
  gz_gui        show the Gazebo window      [true]  (gazebo only)
  rviz          start RViz                  [true]
  gui           start the telemetry GUI     [true]

With backend:=sim there is no physics: /joint_states is the command itself,
so the tracking-error plot stays empty and PID is switched off. For physics,
the object and PID use backend:=gazebo control_mode:=pid.

Examples:
  ros2 launch planar_arm_control bringup.launch.py
  ros2 launch planar_arm_control bringup.launch.py backend:=gazebo control_mode:=pid rviz:=false
"""

import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration


def generate_launch_description():
    demo = os.path.join(get_package_share_directory('planar_arm_control'),
                        'launch', 'pick_place_demo.launch.py')
    defaults = {'backend': 'sim', 'control_mode': 'position',
                'gz_gui': 'true', 'rviz': 'true', 'gui': 'true'}
    args = [DeclareLaunchArgument(k, default_value=v)
            for k, v in defaults.items()]
    include = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(demo),
        launch_arguments={k: LaunchConfiguration(k) for k in defaults}.items())
    return LaunchDescription(args + [include])
