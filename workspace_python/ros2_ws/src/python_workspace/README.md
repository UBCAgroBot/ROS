
### List of Nodes

| Node Type         | Subscribes To | Publishes To                      | Example command |
|-------------------|----------------|-----------------------------------|------|
| Picture Node      | -                | `/input_image`        | `ros2 run python_workspace picture_node --ros-args -p static_image_path:='./../assets/maize' -p loop:=-1 -p frame_rate:=1`|
| Inference Node    | `/input_image`            | -`/inference_out` <br> - `/output_img`             | `ros2 run python_workspace inference_node --ros-args -p weights_path:='../models/maize/Maize.pt'`|
| Extermination Node    | `/inference_out`            | external binary        | `ros2 run python_workspace extermination_node`|


### List of Topics and Data Types

| Topic Name                  | Description                          | Data Type          |
|-----------------------------|--------------------------------------|--------------------|
| `/input_image`              | custom msg that sends raw image, processed image and velocity          | `custom_interface/msg/ImageInput`|
| `/inference_out` | custom msg that passes inference output and raw image to extermination node | `custom_interface/msg/InferenceOutput` |
| `/output_img`               | Processed output image               | `sensor_msgs/Image`|

### Other commands
#### Running the camera node:
`ros2 run python_workspace camera_node --ros-args -p static_image_path:='/home/user/Desktop/ROS/Models/Maize Model/sample_maize_images' -p loop:=-1 -p frame_rate:=10 -p model_dimensions:=[640,640]`

#### Running the jetson_node
ros2 run python_workspace jetson_node

#### Running the extermination node
```ros2 run python_workspace extermination_node```

Without any additional arguments, this will also show a window with the drawn bounding boxes.
To turn it off, 
add the argument `--ros-args -p use_display_node:=false` to the command.
#### Running the picture node:
File paths are relative to the working directory

`ros2 run python_workspace picture_node --ros-args -p static_image_path:='./../assets/maize' -p loop:=-1 -p frame_rate:=1`

#### Running the inference node:

The path for weights (.pt) is relative to the ROS/workspace_python/ros2_ws directory. 


```bash
ros2 run python_workspace inference_node --ros-args -p weights_path:='../models/maize/Maize.pt'
```

### Compiling new Changes
```bash
colcon build --packages-select custom_interface python_workspace
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

