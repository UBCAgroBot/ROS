# Both nodes can be loaded into a composable container, and the preprocessed CUDA memory pointer can be passed directly from ImagePreprocessingNode to InferenceNode.

from launch_ros.actions import ComposableNodeContainer
from launch_ros.descriptions import ComposableNode

def generate_launch_description():
    return LaunchDescription([
        ComposableNodeContainer(
            name='gpu_composable_container',
            namespace='',
            package='rclcpp_components',
            executable='component_container_mt',  # Multi-threaded container for composable nodes
            composable_node_descriptions=[
                ComposableNode(
                    package='my_package',
                    plugin='my_package::ImagePreprocessingNode',
                    name='image_preprocessing_node'
                ),
                ComposableNode(
                    package='my_package',
                    plugin='my_package::InferenceNode',
                    name='inference_node'
                ),
            ],
            output='screen',
        ),
    ])

# In this setup, ImagePreprocessingNode preprocesses the image on the GPU, and InferenceNode receives the same CUDA memory buffer without any overhead due to inter-process communication or copying back and forth between GPU and CPU. The nodes remain within the same process and can share GPU memory pointers efficiently.