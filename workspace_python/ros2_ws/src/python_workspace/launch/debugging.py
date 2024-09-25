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
            'static_image_path', default_value='/home/usr/Desktop/ROS/assets/IMG_1822_14.JPG',
            description='The path to the static image for CameraNode'),

        DeclareLaunchArgument(
            'loop', default_value='-1',
            description='0 = do not loop, >0 = # of loops, -1 = loop forever'),

        DeclareLaunchArgument(
            'frame_rate', default_value='30',
            description='Frame rate for CameraNode publishing'),
        
        DeclareLaunchArgument(
            'model_dimensions', default_value='[640, 640]',
            description='Model dimensions for CameraNode preprocessing'),
        
        DeclareLaunchArgument(
            'shift_constant', default_value='1',
            description='Shift constant for camera adjustments'),
        
        DeclareLaunchArgument(
            'roi_dimensions', default_value='[0, 0, 100, 100]',
            description='Region of interest dimensions for CameraNode'),
        
        DeclareLaunchArgument(
            'precision', default_value='fp32',
            description='Precision for image processing (fp32 or fp16)'),

        PushRosNamespace('camera_jetson_namespace'),  # Optional: Organize nodes under a namespace

        # Launch JetsonNode
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
        
        # Launch ExterminationNode
        Node(
            package='python_workspace',
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
            package='python_workspace',
            executable='picture_node',
            name='picture_node',
            output='screen',
            parameters=[
                {'static_image_path': LaunchConfiguration('static_image_path')},
                {'loop': LaunchConfiguration('loop')},
                {'frame_rate': LaunchConfiguration('frame_rate')},
                {'model_dimensions': LaunchConfiguration('model_dimensions')},
                {'shift_constant': LaunchConfiguration('shift_constant')},
                {'roi_dimensions': LaunchConfiguration('roi_dimensions')},
                {'precision': LaunchConfiguration('precision')}
            ]
        ),
    ])
