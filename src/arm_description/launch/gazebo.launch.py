"""Launch Gazebo with the table world and spawn the planar arm on the table."""

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
    pkg_share = get_package_share_directory('arm_description')
    ros_gz_sim_share = get_package_share_directory('ros_gz_sim')

    world_path = os.path.join(pkg_share, 'Gazebo', 'empty_world.sdf')
    xacro_path = os.path.join(pkg_share, 'urdf', 'robot.urdf.xacro')
    bridge_config = os.path.join(pkg_share, 'config', 'bridge.yaml')
    rviz_config = os.path.join(pkg_share, 'rviz', 'display.rviz')

    robot_description = xacro.process_file(xacro_path).toxml()

    args = [
        DeclareLaunchArgument('rviz', default_value='false',
                              description='Also open RViz'),
        DeclareLaunchArgument('gui', default_value='true',
                              description='false = headless (server only, no Gazebo window)'),
        DeclareLaunchArgument('table_height', default_value='0.75',
                              description='Z of the table top (robot spawn height)'),
    ]

    # Gazebo server (+ GUI unless gui:=false), running immediately (-r)
    gz_args = PythonExpression([
        "('-r -s ' if '", LaunchConfiguration('gui'), "' == 'false' else '-r ')",
        " + '", world_path, "'",
    ])
    gz_sim = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            os.path.join(ros_gz_sim_share, 'launch', 'gz_sim.launch.py')),
        launch_arguments={'gz_args': gz_args}.items(),
    )

    robot_state_publisher = Node(
        package='robot_state_publisher',
        executable='robot_state_publisher',
        output='screen',
        parameters=[{'robot_description': robot_description,
                     'use_sim_time': True}],
    )

    # Spawn from the robot_description topic, on top of the table
    spawn = Node(
        package='ros_gz_sim',
        executable='create',
        output='screen',
        arguments=['-topic', 'robot_description',
                   '-name', 'planar_arm',
                   '-world', 'empty_world',
                   '-x', '0.0', '-y', '0.0',
                   '-z', LaunchConfiguration('table_height')],
    )

    bridge = Node(
        package='ros_gz_bridge',
        executable='parameter_bridge',
        output='screen',
        parameters=[{'config_file': bridge_config,
                     'use_sim_time': True}],
    )

    rviz = Node(
        package='rviz2',
        executable='rviz2',
        arguments=['-d', rviz_config],
        parameters=[{'use_sim_time': True}],
        condition=IfCondition(LaunchConfiguration('rviz')),
    )

    return LaunchDescription(args + [gz_sim, robot_state_publisher, spawn,
                                     bridge, rviz])
