Running the camera node:
ros2 run python_workspace camera_node --ros-args -p static_image_path:='/home/user/Desktop/ROS/Models/Maize Model/sample_maize_images' -p loop:=-1 -p frame_rate:=10 -p model_dimensions:=[640,640]

ros2 run python_workspace jetson_node

ros2 run python_workspace extermination_node