from launch import LaunchDescription
from launch_ros.actions import ComposableNodeContainer
from launch_ros.descriptions import ComposableNode

def generate_launch_description():
    return LaunchDescription([
        ComposableNodeContainer(
            name='camera_container',
            namespace='',
            package='rclcpp_components',
            executable='component_container',
            composable_node_descriptions=[
                ComposableNode(
                    package='your_package_name',
                    plugin='your_package_name.CameraNode',
                    name='camera_node',
                    parameters=[{
                        'source_type': 'zed',
                        'static_image_path': '.../assets/',
                        'video_path': '.../assets/video.mp4',
                        'loop': 0,
                        'frame_rate': 30,
                        'model_type': 'maize',
                    }],
                ),
            ],
            output='screen',
        ),
    ])
