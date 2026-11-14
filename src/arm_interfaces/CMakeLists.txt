cmake_minimum_required(VERSION 3.8)
project(arm_interfaces)

find_package(ament_cmake REQUIRED)
find_package(rosidl_default_generators REQUIRED)
find_package(geometry_msgs REQUIRED)
find_package(action_msgs REQUIRED)

rosidl_generate_interfaces(${PROJECT_NAME}
  "action/MoveToTarget.action"
  "action/PickAndPlace.action"
  "action/Gripper.action"
  "action/PickOrPlace.action"
  DEPENDENCIES geometry_msgs action_msgs
)

ament_package()
