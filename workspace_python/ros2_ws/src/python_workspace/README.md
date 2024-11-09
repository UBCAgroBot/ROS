
# python_workspace package
This package contains nodes that gets image data, runs inference on them and also converts it into output needed for the extermination team. 

## Node list 
> **Note:** All example commands are run under the `workspace_python/ros2_ws` directory.


| Node Type         | Description | Subscribes To | Publishes To                      | Example command |
|-------------------|-------------|---------------|-----------------------------------|-----------------|
| Picture Node      | Captures and processes static images | -             | `/input_image`        | `ros2 run python_workspace picture_node --ros-args -p static_image_path:='./../../assets/maize' -p frame_rate:=1`|
| Inference Node    | Runs inference on input images | `/input_image`  | - `/inference_out` <br> - `/output_img` | `ros2 run python_workspace inference_node --ros-args -p weights_path:='../../models/maize/Maize.pt'`|
| Extermination Node| Processes inference results, displays and sends binary output for exttermination team | `/inference_out` | external binary        | `ros2 run python_workspace extermination_node`|



Other nodes are still a WIP

## List of Topics and Data Types

| Topic Name                  | Description                          | Data Type          |
|-----------------------------|--------------------------------------|--------------------|
| `/input_image`              | custom msg that sends raw image, processed image and velocity          | `custom_interface/msg/ImageInput`|
| `/inference_out` | custom msg that passes inference output and raw image to extermination node | `custom_interface/msg/InferenceOutput` |
| `/output_img`               | Processed output image               | `sensor_msgs/Image`|



## Argument List

| Node Type         | Argument Name       | Argument Type | Sample Argument Value                              | Description                                      |
|-------------------|---------------------|---------------|----------------------------------------------------|--------------------------------------------------|
| Picture Node      | `static_image_path` | `string`      | `-p static_image_path:='./../../assets/maize'`     | Path to the static image to be processed. Can be relative or absolute         |
|                   | `frame_rate`        | `int`         | `-p frame_rate:=1`                                 | Frame rate for processing the image              |
| Inference Node    | `weights_path`      | `string`      | `-p weights_path:='../../models/maize/Maize.pt'`   | Path to the model weights for inference. Can be relative or absolute          |
| Extermination Node| `use_display_node`  | `bool`        | `-p use_display_node:=false`                       | Flag to enable or disable the display of results |


## Additional information
### Compiling new Changes
```bash
colcon build --packages-select custom_interface python_workspace #not needed if the initial build uses  --symlink-install
source install/setup.bash
```

### using RQT to view 
#### Installing RQT on ROS 2 Humble

To install RQT and its plugins on ROS 2 Humble, use the following command:

```bash
sudo apt update
sudo apt install ~nros-humble-rqt*
```

This command will install RQT along with all available plugins for ROS 2 Humble.

#### Running RQT

To start RQT, simply run:

```bash
rqt
```

You can now use RQT to visualize and interact with various ROS topics and nodes.

To view the image topic : 

1. go to plugins/visualization/image view. a new box will open up
2. click refresh topic 
3. select the name of the image topic you want to see

