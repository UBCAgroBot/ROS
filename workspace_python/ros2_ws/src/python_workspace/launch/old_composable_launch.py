import os
from launch import LaunchDescription
from launch_ros.actions import PushRosNamespace
from launch_ros.actions import Node
from launch.substitutions import LaunchConfiguration

def generate_launch_description():
    return LaunchDescription([
        PushRosNamespace('camera_jetson_namespace'),  # Optional: Organize nodes under a namespace
        
        Node(
            package='your_camera_package',  # Replace with the actual package name for CameraNode
            namespace='camera',
            executable='camera_node',
            name='camera_node',
            output='screen',
            parameters=[
                {
                    'source_type': 'zed',  # Example parameter, change as necessary
                    'static_image_path': '/home/usr/Desktop/ROS/assets/IMG_1822_14.JPG',
                    'video_path': '/home/usr/Desktop/ROS/assets/video.mp4',
                    'loop': 0,  # Example value, adjust accordingly
                    'frame_rate': 30,
                    'model_dimensions': (448, 1024),
                    'camera_side': 'left',
                    'shift_constant': 1,
                    'roi_dimensions': [0, 0, 100, 100],
                    'precision': 'fp32',  # Example precision
                }
            ],
        ),
        
        Node(
            package='your_jetson_package',  # Replace with the actual package name for JetsonNode
            namespace='jetson',
            executable='jetson_node',
            name='jetson_node',
            output='screen',
            parameters=[
                {
                    'engine_path': '/home/user/Downloads/model.engine',  # Replace with the actual engine path
                    'strip_weights': False,
                    'precision': 'fp32',  # Example precision
                }
            ],
        ),
    ])
