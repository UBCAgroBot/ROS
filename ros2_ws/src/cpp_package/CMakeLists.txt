cmake_minimum_required(VERSION 3.8)
project(cpp_package)

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

# find dependencies
find_package(ament_cmake REQUIRED)
# uncomment the following section in order to fill in
# further dependencies manually.
# find_package(<dependency> REQUIRED)

find_package(rclcpp REQUIRED)
find_package(std_msgs REQUIRED)
find_package(sensor_msgs REQUIRED)
find_package(launch REQUIRED)
find_package(launch_ros REQUIRED)

add_executable(camera_node src/camera_node.cpp)
ament_target_dependencies(camera_node
  rclcpp
  std_msgs
  sensor_msgs
)

add_executable(proxy_node src/proxy_node.cpp)
ament_target_dependencies(proxy_node
  rclcpp
  std_msgs
)

add_executable(inference_node src/inference_node.cpp)
ament_target_dependencies(inference_node
  rclcpp
  std_msgs
)

add_executable(extermination_node src/extermination_node.cpp)
ament_target_dependencies(extermination_node
  rclcpp
  std_msgs
)

install(TARGETS
  camera_node
  proxy_node
  inference_node
  extermination_node
  DESTINATION lib/${PROJECT_NAME}
)


if(BUILD_TESTING)
  find_package(ament_lint_auto REQUIRED)
  # the following line skips the linter which checks for copyrights
  # comment the line when a copyright and license is added to all source files
  set(ament_cmake_copyright_FOUND TRUE)
  # the following line skips cpplint (only works in a git repo)
  # comment the line when this package is in a git repo and when
  # a copyright and license is added to all source files
  set(ament_cmake_cpplint_FOUND TRUE)
  ament_lint_auto_find_test_dependencies()
endif()

ament_package()
