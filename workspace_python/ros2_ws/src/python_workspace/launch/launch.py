from launch_ros.actions import PushRosNamespace
from launch import LaunchDescription
from launch_ros.actions import Node
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration

def generate_launch_description():
    return LaunchDescription([
        # Declare launch arguments
        DeclareLaunchArgument(
            'engine_path', default_value='/home/user/Downloads/model.engine',
            description='Path to the TensorRT engine file'),

        DeclareLaunchArgument(
            'use_display_node', default_value='True',
            description='Whether to use the display node in ExterminationNode'),
        
        DeclareLaunchArgument(
            'camera_side', default_value='left',
            description='Camera side (left or right)'),

        PushRosNamespace('camera_jetson_namespace'),  # Optional: Organize nodes under a namespace
        Node(
            package='python_workspace',
            executable='jetson_node',
            name='jetson_node',
            output='screen',
            parameters=[
                {'engine_path': LaunchConfiguration('engine_path')},
                {'strip_weights': LaunchConfiguration('strip_weights')},
                {'precision': LaunchConfiguration('precision')}
            ]
        ),
        Node(
            package='python_workspace',
            executable='extermination_node',
            name='extermination_node',
            output='screen',
            parameters=[
                {'use_display_node': LaunchConfiguration('use_display_node')},
                {'side': LaunchConfiguration('side')}
            ]
        ),
        Node(
            package='python_workspace',
            executable='camera_node',
            name='camera_node',
            output='screen',
            parameters=[
                {'camera_side': LaunchConfiguration('camera_side')},
            ]
        ),
    ])