# document launch parameters

import os
from launch import LaunchDescription
from launch_ros.actions import Node
from launch.substitutions import LaunchConfiguration
from launch.actions import DeclareLaunchArgument
from launch.conditions import IfCondition

# update parameters
# specify side to camera_node and jetson_node for left/right subscriber/topic
# should be two composition containers for left/right
def generate_launch_description():
    return LaunchDescription([
        DeclareLaunchArgument(
            'model_path',
            default_value='/default/path/to/model.trt',
            description='Path to the TensorRT engine file'
        ),
        DeclareLaunchArgument(
            'source_type',
            default_value='static_image',
            description='Type of source: static_image, video, zed'
        ),
        DeclareLaunchArgument(
            'static_image_path',
            default_value='/path/to/static/image.jpg',
            description='Path to the static image file'
        ),
        DeclareLaunchArgument(
            'video_path',
            default_value='/path/to/video.mp4',
            description='Path to the video file'
        ),
        DeclareLaunchArgument(
            'loop',
            default_value='-1',
            description='Number of times to loop the video (-1 for infinite, 0 for 1 loop, >0 for # of loops )'
        ),
        DeclareLaunchArgument(
            'frame_rate',
            default_value='30',
            description='Desired frame rate for publishing'
        ),
        DeclareLaunchArgument(
            'model_type',
            default_value='maize',
            description='The model architecture being used'
        ),
        DeclareLaunchArgument(
            'use_display_node',
            default_value='True',
            description='Whether to launch cv2 display screens'
        ),
        DeclareLaunchArgument(
            'lower_range',
            default_value='[78, 158, 124]',
            description='Lower HSV range for color filtering'
        ),
        DeclareLaunchArgument(
            'upper_range',
            default_value='[60, 255, 255]',
            description='The model architecture being used'
        ),
        DeclareLaunchArgument(
            'min_area',
            default_value='100',
            description='The minimum area for a contour to be considered'
        ),
        DeclareLaunchArgument(
            'min_confidence',
            default_value='0.5',
            description='The minimum confidence for a detection to be considered'
        ),
        DeclareLaunchArgument(
            'roi_list',
            default_value='[0,0,100,100]',
            description='The region of interest for the camera'
        ),
        DeclareLaunchArgument(
            'publish_rate',
            default_value='10',
            description='The rate at which to publish the bounding boxes'
        ),
        # example to toggle nodes on/off
        # DeclareLaunchArgument(
        #     'use_display_node',
        #     default_value='true',
        #     description='Whether to launch the display node'
        # ),
        Node(
            package='python_workspace',
            executable='camera_node', # not sure if should match setup.py
            name='camera_node',
            output='screen', # idk what this does
            parameters=[
                {'source_type': LaunchConfiguration('source_type')},
                {'static_image_path': LaunchConfiguration('static_image_path')},
                {'video_path': LaunchConfiguration('video_path')},
                {'loop': LaunchConfiguration('loop')},
                {'frame_rate': LaunchConfiguration('frame_rate')},
                {'model_type': LaunchConfiguration('model_type')},
            ]
        ),
        Node(
            package='python_workspace',
            executable='jetson_node',
            name='jetson_node',
            output='screen',
            parameters=[{'model_path': LaunchConfiguration('model_path')}],
        ),
        Node(
            package='python_workspace',
            executable='extermination',
            name='extermination_node',
            output='screen',
            # condition=IfCondition(LaunchConfiguration('use_display_node'))
            parameters=[
                {'use_display_node': LaunchConfiguration('use_display_node')},
                {'lower_range': LaunchConfiguration('lower_range')},
                {'upper_range': LaunchConfiguration('upper_range')},
                {'min_area': LaunchConfiguration('min_area')},
                {'min_confidence': LaunchConfiguration('min_confidence')},
                {'roi_list': LaunchConfiguration('roi_list')},
                {'publish_rate': LaunchConfiguration('publish_rate')},
            ]
        ),
    ])