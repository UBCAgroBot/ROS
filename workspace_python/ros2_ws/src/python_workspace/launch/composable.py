## for composable:
from launch import LaunchDescription
from launch_ros.actions import LifecycleNode
from launch_ros.actions import ComposableNodeContainer
from launch_ros.descriptions import ComposableNode

def generate_launch_description():
    container = ComposableNodeContainer(
        name='my_container',
        namespace='',
        package='rclcpp_components',
        executable='component_container_mt',
        composable_node_descriptions=[
            ComposableNode(
                package='my_package',
                plugin='mypackage.ComposablePublisherNode',
                name='composable_publisher_node'
            ),
        ],
        output='screen',
    )

    lifecycle_manager = LifecycleNode(
        package='my_package',
        executable='lifecycle_node',
        name='lifecycle_manager_node',
        namespace='',
        output='screen',
    )

    return LaunchDescription([
        container,
        lifecycle_manager
    ])

# new basic:
from launch import LaunchDescription
from launch_ros.actions import ComposableNodeContainer
from launch_ros.descriptions import ComposableNode

def generate_launch_description():
    return LaunchDescription([
        ComposableNodeContainer(
            name='my_container',
            namespace='',
            package='rclcpp_components',
            executable='component_container',
            composable_node_descriptions=[
                ComposableNode(
                    package='my_package',
                    plugin='MyNode',
                    name='my_node'
                )
            ],
            output='screen',
        ),
    ])

# This launch file will load MyNode into the container at runtime.

## basic pt 2
from launch_ros.actions import ComposableNodeContainer
from launch_ros.descriptions import ComposableNode

def generate_launch_description():
    return LaunchDescription([
        ComposableNodeContainer(
            name='my_composable_container',
            namespace='',
            package='rclcpp_components',
            executable='component_container_mt',  # Multi-threaded container
            composable_node_descriptions=[
                ComposableNode(
                    package='my_package',
                    plugin='my_package.IndoorNavNode',
                    name='indoor_nav'
                ),
                ComposableNode(
                    package='my_package',
                    plugin='my_package.LiDARNode',
                    name='lidar_node'
                ),
            ],
            output='screen',
        ),
    ])

## cuda composable:
from launch_ros.actions import ComposableNodeContainer
from launch_ros.descriptions import ComposableNode
from launch import LaunchDescription

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
                    plugin='my_package.image_preprocessing_node',
                    name='image_preprocessing_node'
                ),
                ComposableNode(
                    package='my_package',
                    plugin='my_package.inference_node',
                    name='inference_node',
                    parameters=[
                        {"preprocess_node": "image_preprocessing_node"}
                    ]
                ),
            ],
            output='screen',
        ),
    ])



from launch import LaunchDescription
from launch_ros.actions import ComposableNodeContainer
from launch_ros.descriptions import ComposableNode

def generate_launch_description():
    return LaunchDescription([
        ComposableNodeContainer(
            name='my_python_container',
            namespace='',
            package='rclpy_components',
            executable='component_container_mt',  # Multi-threaded container
            composable_node_descriptions=[
                ComposableNode(
                    package='my_package',
                    plugin='my_python_node.MyPythonNode',
                    name='my_python_node'
                )
            ],
            output='screen',
        ),
    ])
    
### ll


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

# cuda concurrent
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

# CUDA Streams for Asynchronous Execution: Both the preprocessing and inference nodes now use CUDA streams to allow for non-blocking GPU operations.

#     Image Preprocessing Node: Executes memcpy_htod_async to copy the image to GPU memory without blocking the CPU. The preprocessing stream is synchronized with stream.synchronize() to ensure the operation completes.
#     Inference Node: Executes inference in its own stream using execute_async_v2. The stream is synchronized with stream.synchronize() to ensure that inference completes before retrieving the results.

# Concurrency: By using a multi-threaded composable node container (component_container_mt), both nodes can process different images in parallel. This means the preprocessing node can start working on the next image while the inference node is still running on the previous one.