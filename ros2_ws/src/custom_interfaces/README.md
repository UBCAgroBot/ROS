
# Custom Interface Package

This interface uses the ROS2 tutorial on custom interfaces:
[ROS2 Custom Interfaces Tutorial](https://docs.ros.org/en/humble/Tutorials/Beginner-Client-Libraries/Custom-ROS2-Interfaces.html)

Instructions to use the custom message or services can be found here:
[ROS2 Custom Interfaces Tutorial](https://docs.ros.org/en/humble/Tutorials/Beginner-Client-Libraries/Custom-ROS2-Interfaces.html)


## Re building the custom message interface
> **Note:** Ensure that this is run from the main workspace directory (`ros2_ws`).

```sh
colcon build --packages-select custom_interfaces
```

## Fun ROS2 Commands
> **Note:** These should also be run from the main workspace directory


### Show All Messages and Services
Use the `-m` flag to show only messages:
```sh
ros2 interface list
```

### Show Message and Service Definitions
```sh
ros2 interface show custom_interfaces/msg/Custom
```

### Topic Commands
```sh
### Topic Commands
List all active topics:
```sh
ros2 topic list
```

Show information about a specific topic:
```sh
ros2 topic info /topic_name
```

Echo messages from a specific topic:
```sh
ros2 topic echo /topic_name
```

### Troubleshooting
If you get "the message type `custom_interface/msg/Custom` is invalid", source the ROS2 install/setup.bash and try to run the commands again.