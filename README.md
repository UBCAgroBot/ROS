# ROS Computer Vision System

Developed by UBC AgroBot Systems Architecture Team

## Background Information

This repository contains a ROS2-based computer vision system for agricultural robotics. The system processes visual data to enable autonomous navigation and crop identification for the AgroBot. It includes both hardware integration components and software algorithms for real-time image processing and decision-making.

## Overview
![Node Diagram](assets/node_diagram.svg)

The system consists of four main components:
1. **Camera Input Processing Unit**: Captures data from the Zed Camera for inference.
2. **Inference Unit**: This is the main processing hub, running the core ROS nodes and computer vision algorithms.
3. **Inference Result Handler**: Manages inference data, performs preprocessing, and handles object tracking.
4. **Signal Combining**: Combines signals from both the left and right cameras to send to the designated module.

On the Jetson, the ROS `proxy_node` sends commands to the Arduino for physical actuation.

## Quickstart

To quickly start the system, use a ROS launch file to launch the entire stack.

First, ensure you have a ROS2 environment. Open the command palette in VS Code and select `Dev Containers: Rebuild and Reopen in Container`.

Our project is built and run inside the `ros2_ws` directory. Navigate to it by running:
```
cd ros2_ws
```

To build and run the stack, execute the following shell script:
```
./launch.sh
```

Congratulations 🎉! The entire stack is now running locally on your computer.

## Other Available Utilities

### Assets

The `assets` folder contains all image and video sources for testing:
- Video files should be named `video.mp4`.
- Images can have any name but must end with `.jpg` or `.png`.

Maize model dimensions: 1024x448 (448 is the width).

### `action_scripts`

The `scripts` folder contains various action scripts used in the project.

### `python_wip`

This folder contains experimental Python code related to computer vision that is still under development and may be pushed in the future.