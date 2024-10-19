
### List of Nodes

| Node Type         | Subscribes To | Publishes To                      | Example command |
|-------------------|----------------|-----------------------------------|------|
| Picture Node      | `/picture/command`                | `/input_image`        | `ros2 run python_workspace picture_node --ros-args -p static_image_path:='./../assets/maize' -p loop:=-1 -p frame_rate:=1`|
| Inference Node    | `/input_image`            | -`/bounding_box_coordinates` <br> - `/output_img`             | `ros2 run python_workspace inference_node --ros-args -p weights_path:='./ros2_ws/src/python_workspace/python_workspace/scripts/yolo11n.pt'`|


### List of Topics and Data Types

| Topic Name                  | Description                          | Data Type          |
|-----------------------------|--------------------------------------|--------------------|
| `/input_image`              | Input image for processing           | `sensor_msgs/Image`|
| `/bounding_box_coordinates` | Coordinates of detected objects (xyxy)      | `std_msgs/Float32MultiArray` |
| `/output_img`               | Processed output image               | `sensor_msgs/Image`|



### Other commands
#### Running the camera node:
`ros2 run python_workspace camera_node --ros-args -p static_image_path:='/home/user/Desktop/ROS/Models/Maize Model/sample_maize_images' -p loop:=-1 -p frame_rate:=10 -p model_dimensions:=[640,640]`

#### Running the picture node:

`ros2 run python_workspace picture_node --ros-args -p static_image_path:='./../assets/maize' -p loop:=-1 -p frame_rate:=1`

#### Running the inference node:

The path for weights (.pt) is relative to the ROS/workspace_python/ros2_ws directory. 

```bash
ros2 run python_workspace inference_node --ros-args -p weights_path:='./ros2_ws/src/python_workspace/python_workspace/scripts/yolo11n.pt'
```
