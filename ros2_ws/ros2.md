# ROS2 Quick Start

## File structure

- ros2_ws
   - package1
   - package2


## To craete changes
1. Make a change
2. Colcon build
3. Source the package?

## To launch a package

1. Build the package -> 
Note: from ws level
colcon build
colcon build --packages-select <package_name>

2. Source ros2 installation and package setup file:
Note: from ws level
. install/setup.bash

3. run launch file
Note: you can either launch at workspace level or in the launch folder
ros2 launch <launch_file_name>
ros2 launch <package_name> <launch_file_name>
