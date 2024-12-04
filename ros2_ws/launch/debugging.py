from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch_ros.actions import Node
from launch.substitutions import LaunchConfiguration

def generate_launch_description():
    return LaunchDescription([
        DeclareLaunchArgument('static_image_path', default_value='/home/user/ROS/assets/maize/IMG_1822_14.JPG',description='The path to the static image for CameraNode'),
        DeclareLaunchArgument('loop', default_value='-1', description='0 = do not loop, >0 = # of loops, -1 = loop forever'),
        DeclareLaunchArgument('frame_rate', default_value='30', description='Frame rate for CameraNode publishing'),
        DeclareLaunchArgument('precision', default_value='fp32', description='Precision for the inference data input'),
        DeclareLaunchArgument('weights_path', default_value='/home/user/ROS/models/maize/Maize.onnx', description='Path to the model weights file (must be absolute!)'),
        DeclareLaunchArgument('camera_side', default_value='left', description='Side of the Zed Camera for inference'),
        DeclareLaunchArgument('use_display_node', default_value='False', description='Toggle for using the display'),
        
        # Picture Node
        Node(
            package='python_workspace',
            executable='picture_node',
            name='picture_node',
            parameters=[
                {'static_image_path': LaunchConfiguration('static_image_path')},
                {'loop': LaunchConfiguration('loop')},
                {'frame_rate': LaunchConfiguration('frame_rate')},
            ],
            output='screen'
        ),
        
        # Inference Node
        Node(
            package='python_workspace',
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
            package='python_workspace',
            executable='extermination_node',
            name='extermination_node',
            parameters=[
                {'use_display_node': LaunchConfiguration('use_display_node')},
                {'camera_side': LaunchConfiguration('camera_side')}
            ],
            output='screen'
        ),
        
    ])