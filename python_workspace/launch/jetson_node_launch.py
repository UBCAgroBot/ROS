# # document launch parameters

# from launch import LaunchDescription
# from launch_ros.actions import Node

# def generate_launch_description():
#     return LaunchDescription([
#         Node(
#             package='node_test',
#             executable='jetson_node',
#             name='jetson_node',
#             output='screen',
#         ),
#     ])

# # node_test/launch/jetson_node_launch.py

# from launch import LaunchDescription
# from launch_ros.actions import Node

# def generate_launch_description():
#     return LaunchDescription([
#         Node(
#             package='node_test',
#             executable='jetson_node',
#             name='jetson_node',
#             output='screen',
#         ),
#         Node(
#             package='node_test',
#             executable='bounding_box_publisher',
#             name='bounding_box_publisher',
#             output='screen',
#         ),
#     ])

# ## modified to support CLI for specifying model path:
# # node_test/launch/jetson_node_launch.py

# from launch import LaunchDescription
# from launch_ros.actions import Node
# from launch.substitutions import LaunchConfiguration
# from launch.actions import DeclareLaunchArgument

# def generate_launch_description():
#     return LaunchDescription([
#         DeclareLaunchArgument(
#             'model_path',
#             default_value='/default/path/to/model.trt',
#             description='Path to the TensorRT engine file'
#         ),
#         Node(
#             package='node_test',
#             executable='jetson_node',
#             name='jetson_node',
#             output='screen',
#             parameters=[{'model_path': LaunchConfiguration('model_path')}],
#         ),
#         Node(
#             package='node_test',
#             executable='bounding_box_publisher',
#             name='bounding_box_publisher',
#             output='screen',
#         ),
#     ])

# # node_test/launch/jetson_node_launch.py

# from launch import LaunchDescription
# from launch_ros.actions import Node
# from launch.substitutions import LaunchConfiguration
# from launch.actions import DeclareLaunchArgument

# def generate_launch_description():
#     return LaunchDescription([
#         DeclareLaunchArgument(
#             'model_path',
#             default_value='/default/path/to/model.trt',
#             description='Path to the TensorRT engine file'
#         ),
#         Node(
#             package='node_test',
#             executable='jetson_node',
#             name='jetson_node',
#             output='screen',
#             parameters=[{'model_path': LaunchConfiguration('model_path')}],
#         ),
#         Node(
#             package='node_test',
#             executable='bounding_box_publisher',
#             name='bounding_box_publisher',
#             output='screen',
#         ),
#     ])


# # node_test/launch/jetson_node_launch.py

# from launch import LaunchDescription
# from launch_ros.actions import Node
# from launch.substitutions import LaunchConfiguration
# from launch.actions import DeclareLaunchArgument
# from launch.conditions import IfCondition

# def generate_launch_description():
#     return LaunchDescription([
#         DeclareLaunchArgument(
#             'model_path',
#             default_value='/default/path/to/model.trt',
#             description='Path to the TensorRT engine file'
#         ),
#         DeclareLaunchArgument(
#             'use_display_node',
#             default_value='true',
#             description='Whether to launch the display node'
#         ),
#         Node(
#             package='node_test',
#             executable='jetson_node',
#             name='jetson_node',
#             output='screen',
#             parameters=[{'model_path': LaunchConfiguration('model_path')}],
#         ),
#         Node(
#             package='node_test',
#             executable='bounding_box_publisher',
#             name='bounding_box_publisher',
#             output='screen',
#         ),
#         Node(
#             package='node_test',
#             executable='display_node',
#             name='display_node',
#             output='screen',
#             condition=IfCondition(LaunchConfiguration('use_display_node'))
#         ),
#     ])

# # node_test/launch/jetson_node_launch.py

# from launch import LaunchDescription
# from launch_ros.actions import Node
# from launch.substitutions import LaunchConfiguration
# from launch.actions import DeclareLaunchArgument
# from launch.conditions import IfCondition

# def generate_launch_description():
#     return LaunchDescription([
#         DeclareLaunchArgument(
#             'model_path',
#             default_value='/default/path/to/model.trt',
#             description='Path to the TensorRT engine file'
#         ),
#         DeclareLaunchArgument(
#             'source_type',
#             default_value='static_image',
#             description='Type of source: static_image, video, zed'
#         ),
#         DeclareLaunchArgument(
#             'static_image_path',
#             default_value='/path/to/static/image.jpg',
#             description='Path to the static image file'
#         ),
#         DeclareLaunchArgument(
#             'video_path',
#             default_value='/path/to/video.mp4',
#             description='Path to the video file'
#         ),
#         DeclareLaunchArgument(
#             'use_display_node',
#             default_value='true',
#             description='Whether to launch the display node'
#         ),
#         Node(
#             package='node_test',
#             executable='camera_node',
#             name='camera_node',
#             output='screen',
#             parameters=[
#                 {'source_type': LaunchConfiguration('source_type')},
#                 {'static_image_path': LaunchConfiguration('static_image_path')},
#                 {'video_path': LaunchConfiguration('video_path')}
#             ],
#         ),
#         Node(
#             package='node_test',
#             executable='jetson_node',
#             name='jetson_node',
#             output='screen',
#             parameters=[{'model_path': LaunchConfiguration('model_path')}],
#         ),
#         Node(
#             package='node_test',
#             executable='bounding_box_publisher',
#             name='bounding_box_publisher',
#             output='screen',
#         ),
#         Node(
#             package='node_test',
#             executable='display_node',
#             name='display_node',
#             output='screen',
#             condition=IfCondition(LaunchConfiguration('use_display_node'))
#         ),
#     ])

# # node_test/launch/extermination_node_launch.py

# from launch import LaunchDescription
# from launch_ros.actions import Node

# def generate_launch_description():
#     return LaunchDescription([
#         Node(
#             package='node_test',
#             executable='camera_node',
#             name='camera_node',
#             parameters=[{'source_type': 'zed'}]
#         ),
#         Node(
#             package='node_test',
#             executable='jetson_node',
#             name='jetson_node',
#             parameters=[{'model_path': '/path/to/your/model.trt'}]
#         ),
#         Node(
#             package='node_test',
#             executable='extermination_node',
#             name='extermination_node',
#         )
#     ])

# ## for composable:
# # example_launch.py
# from launch import LaunchDescription
# from launch_ros.actions import LifecycleNode
# from launch_ros.actions import ComposableNodeContainer
# from launch_ros.descriptions import ComposableNode

# def generate_launch_description():
#     container = ComposableNodeContainer(
#         name='my_container',
#         namespace='',
#         package='rclcpp_components',
#         executable='component_container_mt',
#         composable_node_descriptions=[
#             ComposableNode(
#                 package='my_package',
#                 plugin='mypackage.ComposablePublisherNode',
#                 name='composable_publisher_node'
#             ),
#         ],
#         output='screen',
#     )

#     lifecycle_manager = LifecycleNode(
#         package='my_package',
#         executable='lifecycle_node',
#         name='lifecycle_manager_node',
#         namespace='',
#         output='screen',
#     )

#     return LaunchDescription([
#         container,
#         lifecycle_manager
#     ])

# ## parameter handling:
# # example_launch.py
# from launch import LaunchDescription
# from launch_ros.actions import LifecycleNode
# from launch_ros.actions import ComposableNodeContainer
# from launch_ros.descriptions import ComposableNode
# from launch.actions import LogInfo, EmitEvent
# from launch.events.lifecycle import ChangeState
# from lifecycle_msgs.msg import Transition

# def generate_launch_description():
#     # Define parameters to pass to the composable node
#     composable_params = {
#         'message': 'This is a custom message from launch!',
#         'publish_frequency': 1.0
#     }
    
#     container = ComposableNodeContainer(
#         name='my_container',
#         namespace='',
#         package='rclcpp_components',
#         executable='component_container_mt',
#         composable_node_descriptions=[
#             ComposableNode(
#                 package='my_package',
#                 plugin='mypackage.ComposablePublisherNode',
#                 name='composable_publisher_node',
#                 parameters=[composable_params]
#             ),
#             ComposableNode(
#                 package='my_package',
#                 plugin='mypackage.ComposablePublisherNode',
#                 name='additional_publisher_node',
#                 parameters=[{'message': 'This is the second node!', 'publish_frequency': 2.5}]
#             )
#         ],
#         output='screen',
#     )

#     lifecycle_manager = LifecycleNode(
#         package='my_package',
#         executable='lifecycle_node',
#         name='lifecycle_manager_node',
#         namespace='',
#         output='screen',
#         parameters=[{'lifecycle_action': 'configure'}]
#     )
    
#     # Manually trigger lifecycle transitions after nodes are up
#     configure_event = EmitEvent(
#         event=ChangeState(
#             lifecycle_node_matcher=launch.events.matches_action(lifecycle_manager),
#             transition_id=Transition.TRANSITION_CONFIGURE
#         )
#     )

#     activate_event = EmitEvent(
#         event=ChangeState(
#             lifecycle_node_matcher=launch.events.matches_action(lifecycle_manager),
#             transition_id=Transition.TRANSITION_ACTIVATE
#         )
#     )

#     deactivate_event = EmitEvent(
#         event=ChangeState(
#             lifecycle_node_matcher=launch.events.matches_action(lifecycle_manager),
#             transition_id=Transition.TRANSITION_DEACTIVATE
#         )
#     )

#     shutdown_event = EmitEvent(
#         event=ChangeState(
#             lifecycle_node_matcher=launch.events.matches_action(lifecycle_manager),
#             transition_id=Transition.TRANSITION_SHUTDOWN
#         )
#     )

#     return LaunchDescription([
#         LogInfo(msg="Starting the container and lifecycle manager..."),
#         container,
#         lifecycle_manager,
#         LogInfo(msg="Configuring the lifecycle node..."),
#         configure_event,
#         LogInfo(msg="Activating the lifecycle node..."),
#         activate_event,
#         LogInfo(msg="Deactivating the lifecycle node..."),
#         deactivate_event,
#         LogInfo(msg="Shutting down the lifecycle node..."),
#         shutdown_event,
#     ])

# # import os
# # from launch import LaunchDescription
# # from launch.actions import ExecuteProcess, LogInfo, RegisterEventHandler
# # from launch.event_handlers import OnShutdown
# # from launch_ros.actions import Node
# # from launch.substitutions import LocalSubstitution

# # def generate_launch_description():
# #     camera_node = Node(
# #         package='node_test',
# #         executable='camera',
# #         name='camera_node',
# #     )
# #     jetson_node = Node(
# #         package='node_test',
# #         executable='jetson',
# #         name='jetson_node',
# #     )
# #     display_node = Node(
# #         package='node_test',
# #         executable='display',
# #         name='display_node',
# #     )

# #     def shutdown_func_with_echo_side_effect(event, context):
# #         os.system('echo [os.system()] Shutdown callback function can echo this way.')
# #         return [
# #             LogInfo(msg='Shutdown callback was called for reason "{}"'.format(event.reason)),
# #             ExecuteProcess(cmd=['echo', 'However, this echo will fail.'])
# #             ]
    
# #     ld = LaunchDescription(
# #         [
# #             camera_node,
# #             display_node,
# #             jetson_node,
# #             RegisterEventHandler(OnShutdown(on_shutdown=shutdown_func_with_echo_side_effect))
# #         ]
# #     )

# #     return ld