from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node

def generate_launch_description():
    # Declare launch arguments to allow for parameter substitution
    return LaunchDescription([
        # Camera Node arguments
        DeclareLaunchArgument('camera_side', default_value='left', description='Choose left or right camera'),
        DeclareLaunchArgument('shift_constant', default_value='0', description='Shift value for ROI'),
        DeclareLaunchArgument('roi_dimensions', default_value='[0, 0, 100, 100]', description='ROI dimensions as [x1, y1, x2, y2]'),
        
        # Proxy Node argument
        DeclareLaunchArgument('usb_port', default_value='/dev/ttyACM0', description='USB port for the serial connection'),

        # Inference Node arguments
        DeclareLaunchArgument('weights_path', default_value='./src/python_package/python_package/scripts/best.onnx', description='Path to the weights file'),
        DeclareLaunchArgument('precision', default_value='fp32', description='Precision for the inference model (e.g., fp32, fp16)'),
        
        # Extermination Node arguments
        DeclareLaunchArgument('use_display_node', default_value='True', description='Enable display node for visualization'),
        DeclareLaunchArgument('camera_side_exterm', default_value='left', description='Camera side for extermination node (left or right)'),
        
        # Camera Node
        Node(
            package='python_package',
            executable='camera_node',
            name='camera_node',
            parameters=[
                {'camera_side': LaunchConfiguration('camera_side')},
                {'shift_constant': LaunchConfiguration('shift_constant')},
                {'roi_dimensions': LaunchConfiguration('roi_dimensions')}
            ],
            output='screen'
        ),
        
        # Proxy Node
        Node(
            package='python_package',
            executable='proxy_node',
            name='proxy_node',
            parameters=[
                {'usb_port': LaunchConfiguration('usb_port')}
            ],
            output='screen'
        ),

        # Inference Node
        Node(
            package='python_package',
            executable='inference_node',
            name='inference_node',
            parameters=[
                {'weights_path': LaunchConfiguration('weights_path')},
                {'precision': LaunchConfiguration('precision')},
                {'camera_side': LaunchConfiguration('camera_side')}
            ],
            output='screen'
        ),

        # Extermination Node
        Node(
            package='python_package',
            executable='extermination_node',
            name='extermination_node',
            parameters=[
                {'use_display_node': LaunchConfiguration('use_display_node')},
                {'camera_side': LaunchConfiguration('camera_side_exterm')}
            ],
            output='screen'
        )
    ])