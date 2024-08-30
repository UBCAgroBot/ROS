import os
from launch import LaunchDescription
from launch.actions import ExecuteProcess, LogInfo, RegisterEventHandler
from launch.event_handlers import OnShutdown
from launch_ros.actions import Node
from launch.substitutions import LocalSubstitution

def generate_launch_description():
    camera_node = Node(
        package='node_test',
        executable='camera',
        name='camera_node',
    )
    jetson_node = Node(
        package='node_test',
        executable='jetson',
        name='jetson_node',
    )
    display_node = Node(
        package='node_test',
        executable='display',
        name='display_node',
    )

    def shutdown_func_with_echo_side_effect(event, context):
        os.system('echo [os.system()] Shutdown callback function can echo this way.')
        return [
            LogInfo(msg='Shutdown callback was called for reason "{}"'.format(event.reason)),
            ExecuteProcess(cmd=['echo', 'However, this echo will fail.'])
            ]
    
    ld = LaunchDescription(
        [
            camera_node,
            display_node,
            jetson_node,
            RegisterEventHandler(OnShutdown(on_shutdown=shutdown_func_with_echo_side_effect))
        ]
    )

    return ld