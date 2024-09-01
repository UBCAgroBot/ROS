cmake_minimum_required(VERSION 3.5)
project(onnx_cpp)

if(NOT CMAKE_CXX_STANDARD)
  set(CMAKE_CXX_STANDARD 14)
endif()

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()


find_package(ament_cmake REQUIRED)
find_package(rclcpp REQUIRED)
find_package(std_msgs REQUIRED)

find_package(CURL REQUIRED)
find_package( OpenCV REQUIRED )
find_package(cv_bridge REQUIRED)
find_package(sensor_msgs REQUIRED)

add_executable(talker src/talker.cpp)
ament_target_dependencies(talker 
OpenCV 
cv_bridge 
rclcpp 
sensor_msgs 
std_msgs 
)

# /usr/local/include
add_executable(listener src/listener.cpp)
ament_target_dependencies(listener rclcpp std_msgs
OpenCV 
cv_bridge 
sensor_msgs 
)
include_directories(/usr/local/include)
target_link_libraries(listener onnxruntime)

install(TARGETS
  talker
  listener
  DESTINATION lib/${PROJECT_NAME})

ament_package()