"""
bringup.launch.py - the whole system: controller + GUI (+ Gazebo, RViz, object).

Starts, via pick_place_demo.launch.py:
  - Gazebo with the table and the pick object   (backend:=gazebo)
  - robot_state_publisher, controller_node, scene_node
  - RViz                                        (rviz:=true; off by default, and do NOT
                                                 use the GUI's "Launch RViz" button
                                                 here: it starts a second controller)
  - the Qt telemetry GUI                        (gui:=true)

Arguments (defaults in brackets):
  backend       gazebo | sim                [gazebo]
  control_mode  pid | position              [pid]
  gz_gui        show the Gazebo window      [true]
  rviz          start RViz                  [false]
  gui           start the telemetry GUI     [true]

The GUI's tracking-error plot needs real joint feedback, so it only shows
data with backend:=gazebo. With backend:=sim /joint_states is the command
itself (zero error by construction), and PID is switched off.

Examples:
  ros2 launch planar_arm_control bringup.launch.py
  ros2 launch planar_arm_control bringup.launch.py control_mode:=position
  ros2 launch planar_arm_control bringup.launch.py backend:=sim
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
    defaults = {'backend': 'gazebo', 'control_mode': 'pid',
                'gz_gui': 'true', 'rviz': 'false', 'gui': 'true'}
    args = [DeclareLaunchArgument(k, default_value=v)
            for k, v in defaults.items()]
    include = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(demo),
        launch_arguments={k: LaunchConfiguration(k) for k in defaults}.items())
    return LaunchDescription(args + [include])
