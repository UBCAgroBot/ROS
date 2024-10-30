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
            'strip_weights', default_value='False',
            description='Whether to strip weights from the model'),
        
        DeclareLaunchArgument(
            'precision', default_value='fp32',
            description='Inference precision (fp32, fp16)'),

        DeclareLaunchArgument(
            'use_display_node', default_value='True',
            description='Whether to use the display node in ExterminationNode'),
        
        DeclareLaunchArgument(
            'lower_range', default_value='[78, 158, 124]',
            description='Lower HSV range for color segmentation'),
        
        DeclareLaunchArgument(
            'upper_range', default_value='[60, 255, 255]',
            description='Upper HSV range for color segmentation'),
        
        DeclareLaunchArgument(
            'min_area', default_value='100',
            description='Minimum area for object detection'),
        
        DeclareLaunchArgument(
            'min_confidence', default_value='0.5',
            description='Minimum confidence for object detection'),
        
        DeclareLaunchArgument(
            'roi_list', default_value='[0, 0, 100, 100]',
            description='Region of interest for detection'),
        
        DeclareLaunchArgument(
            'publish_rate', default_value='10',
            description='Publishing rate for the extermination node'),
        
        DeclareLaunchArgument(
            'side', default_value='left',
            description='Side of the camera (left or right)'),

        DeclareLaunchArgument(
            'frame_rate', default_value='30',
            description='Frame rate for CameraNode publishing'),
        
        DeclareLaunchArgument(
            'model_dimensions', default_value='[448, 1024]',
            description='Model dimensions for CameraNode preprocessing'),
        
        DeclareLaunchArgument(
            'camera_side', default_value='left',
            description='Camera side (left or right)'),
        
        DeclareLaunchArgument(
            'shift_constant', default_value='1',
            description='Shift constant for camera adjustments'),
        
        DeclareLaunchArgument(
            'roi_dimensions', default_value='[0, 0, 100, 100]',
            description='Region of interest dimensions for CameraNode'),
        
        DeclareLaunchArgument(
            'precision', default_value='fp32',
            description='Precision for image processing (fp32 or fp16)'),

        # Launch JetsonNode
        Node(
            package='your_package_name',
            executable='jetson_node',
            name='jetson_node',
            output='screen',
            parameters=[
                {'engine_path': LaunchConfiguration('engine_path')},
                {'strip_weights': LaunchConfiguration('strip_weights')},
                {'precision': LaunchConfiguration('precision')}
            ]
        ),
        
        # Launch ExterminationNode
        Node(
            package='your_package_name',
            executable='extermination_node',
            name='extermination_node',
            output='screen',
            parameters=[
                {'use_display_node': LaunchConfiguration('use_display_node')},
                {'lower_range': LaunchConfiguration('lower_range')},
                {'upper_range': LaunchConfiguration('upper_range')},
                {'min_area': LaunchConfiguration('min_area')},
                {'min_confidence': LaunchConfiguration('min_confidence')},
                {'roi_list': LaunchConfiguration('roi_list')},
                {'publish_rate': LaunchConfiguration('publish_rate')},
                {'side': LaunchConfiguration('side')}
            ]
        ),

        # Launch CameraNode
        Node(
            package='your_package_name',
            executable='camera_node',
            name='camera_node',
            output='screen',
            parameters=[
                {'frame_rate': LaunchConfiguration('frame_rate')},
                {'model_dimensions': LaunchConfiguration('model_dimensions')},
                {'camera_side': LaunchConfiguration('camera_side')},
                {'shift_constant': LaunchConfiguration('shift_constant')},
                {'roi_dimensions': LaunchConfiguration('roi_dimensions')},
                {'precision': LaunchConfiguration('precision')}
            ]
        ),
    ])
