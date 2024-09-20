from launch_ros.actions import ComposableNodeContainer
from launch_ros.descriptions import ComposableNode
from launch import LaunchDescription
import threading

sync_event = threading.Event()

def generate_launch_description():
    return LaunchDescription([
        ComposableNodeContainer(
            name='gpu_composable_container',
            namespace='',
            package='rclpy_components',
            executable='component_container_mt',  # Multi-threaded container for composable nodes
            composable_node_descriptions=[
                ComposableNode(
                    package='my_package',
                    plugin='my_package.ImagePreprocessingNode',
                    name='image_preprocessing_node',
                    parameters=[
                        {"sync_event": sync_event}
                    ]
                ),
                ComposableNode(
                    package='my_package',
                    plugin='my_package.InferenceNode',
                    name='inference_node',
                    parameters=[
                        {"preprocess_node": "image_preprocessing_node", "sync_event": sync_event}
                    ]
                ),
            ],
            output='screen',
        ),
    ])

# Threading Event: The synchronization between the preprocessing and inference nodes is handled by a threading event (sync_event). The event is set by the image preprocessing node once the image is copied to CUDA memory and is ready for inference. The inference node waits for this event to be set, indicating that the preprocessing is done, and clears the event after starting inference. This ensures that each image is processed in order and no data is overwritten or lost.

# Callback-Based Image Processing: The image preprocessing node processes each incoming image in the callback from the ROS2 image subscription. After processing the image, it signals the inference node to begin the inference process.

# Periodic Inference Check: The inference node runs on a timer that periodically checks whether a new image is ready by checking the threading event. Once the event is set (indicating that an image is ready), it retrieves the CUDA memory pointer and performs inference.