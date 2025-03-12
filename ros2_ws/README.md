# Python ROS2 workspace


## Overview

This is a ROS2 workspace for Python-based implementation.


## package list
- `custom_interface`: This package contains custom message and service definitions for interfacing with other ROS2 nodes.
- `python_package`: This package handles all camera inference, post-processing, and output functionalities.

For more detailed information regarding each node, please refer to the README file located in each package directory. These files contain specific instructions on parameters, usage examples, and other relevant details.


## Prerequisites

To set up and run this workspace, you have several options:

- **Development Containers in VSCode:** Please refer to the internal wiki page titled "VSCode Dev Containers Guide" for comprehensive instructions.
- **SSH Access to Jetson:** For remote access, consult the "Jetson Remote Access" wiki page for detailed guidance.
- **Local Installation (not recommended):** To run the workspace locally, you need to install ROS2 along with various other libraries. Please note that running ROS2 natively on Windows is not supported.


## Run instructions
### Run with sample images using a launch file
> **Note:** This section is still a work in progress. More detailed instructions will be added soon.

### With sample images
> **Warning:** The paths mentioned in the instructions are absolute paths based on the Jetson's directory. These arguments can be changed to be relative to your current directory. 

1. Navigate to the `/ROS/ros2_ws` directory.
2. Create temporary variables to store the paths values:
    ```bash
    IMAGE_FOLDER_PATH=/home/user/ROS/assets/maize
    MODEL_WEIGHT_PATH=/home/user/ROS/models/maize/Maize.pt
    ```
3. Build the packages:
    ```bash    
    # Install dependencies
    rosdep install --from-paths src --ignore-src -r -y

    # Build the workspace
    colcon build --symlink-install --packages-select custom_interfaces python_package 

    # why --symlink install: https://answers.ros.org/question/371822/what-is-the-use-of-symlink-install-in-ros2-colcon-build/
    ```
4. Run the `picture_node` or `video_node`:
    ```bash
    source install/setup.bash
    ros2 run python_package picture_node --ros-args -p static_image_path:=$IMAGE_FOLDER_PATH -p frame_rate:=1
    ```

    For video node, run: 
    ```
    VIDEO_PATH=write/path/here
    ros2 run python_package video_node --ros-args -p static_image_path:=$VIDEO_PATH -p frame_rate:=10
    ```
5. Run the `inference_node` in a new terminal:
    ```bash
    MODEL_WEIGHT_PATH=/home/user/ROS/models/maize/Maize.pt
    source install/setup.bash
    ros2 run python_package inference_node --ros-args -p weights_path:=$MODEL_WEIGHT_PATH 
    ```
6. Run the `extermination_node` in a new terminal:
    ```bash
    source install/setup.bash
    ros2 run python_package extermination_node --ros-args -p use_display_node:=False
    ```


### Getting the plant count using the service
#### Prerequisites:
- Extermination node must be active
```bash
source install/setup.bash
ros2 service call /reset_tracker custom_interfaces/srv/GetRowPlantCount "{}"
```
## Common trouble shooting
Some common troubleshooting steps include:

- **Restart:** ~~Have you tried turning it off and on again?~~ Perform a "clean" colcon build by removing the build and install directories, then run `colcon build` again. Alternatively, you can use `colcon-clean`.
- **Re-source:** Did you [re-source](https://ros2-tutorial.readthedocs.io/en/latest/source_after_build.html) the environment in your terminal after rebuilding?
- **Directory Check:** Ensure you are in the correct directory.
- **File Check:** Verify that you made changes to the actual source files, not the build or install files. 

#### ROS2 not found
Source ROS2 when opening a terminal. It's recommended to add this command to the bashrc file:
```bash
# this is to source ROS2 in the terminal
source /opt/ros/humbe/setup.bash

# this is to add the command to bashrch
echo "source /opt/ros/humbe/setup.bash" >> ~/.bashrc
```


#### CV/CUDA error when running the extermination node. 
This node defaults to using cv to display the output. Depending on how you decided to run this (container, ssh, etc) this error might occur because it can't create a display window. To prevent this error simply turn the display off by adding the argument `--ros-args -p use_display_node:=false` to the command.
```bash 
ros2 run python_package extermination_node --ros-args -p use_display_node:=false
```
## License

This project is licensed under the MIT License. See the LICENSE file for more details.