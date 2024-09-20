## parameter handling:
from launch import LaunchDescription
from launch_ros.actions import LifecycleNode
from launch_ros.actions import ComposableNodeContainer
from launch_ros.descriptions import ComposableNode
from launch.actions import LogInfo, EmitEvent
from launch.events.lifecycle import ChangeState
from lifecycle_msgs.msg import Transition

def generate_launch_description():
    # Define parameters to pass to the composable node
    composable_params = {
        'message': 'This is a custom message from launch!',
        'publish_frequency': 1.0
    }
    
    container = ComposableNodeContainer(
        name='my_container',
        namespace='',
        package='rclcpp_components',
        executable='component_container_mt',
        composable_node_descriptions=[
            ComposableNode(
                package='my_package',
                plugin='mypackage.ComposablePublisherNode',
                name='composable_publisher_node',
                parameters=[composable_params]
            ),
            ComposableNode(
                package='my_package',
                plugin='mypackage.ComposablePublisherNode',
                name='additional_publisher_node',
                parameters=[{'message': 'This is the second node!', 'publish_frequency': 2.5}]
            )
        ],
        output='screen',
    )

    lifecycle_manager = LifecycleNode(
        package='my_package',
        executable='lifecycle_node',
        name='lifecycle_manager_node',
        namespace='',
        output='screen',
        parameters=[{'lifecycle_action': 'configure'}]
    )
    
    # Manually trigger lifecycle transitions after nodes are up
    configure_event = EmitEvent(
        event=ChangeState(
            lifecycle_node_matcher=launch.events.matches_action(lifecycle_manager),
            transition_id=Transition.TRANSITION_CONFIGURE
        )
    )

    activate_event = EmitEvent(
        event=ChangeState(
            lifecycle_node_matcher=launch.events.matches_action(lifecycle_manager),
            transition_id=Transition.TRANSITION_ACTIVATE
        )
    )

    deactivate_event = EmitEvent(
        event=ChangeState(
            lifecycle_node_matcher=launch.events.matches_action(lifecycle_manager),
            transition_id=Transition.TRANSITION_DEACTIVATE
        )
    )

    shutdown_event = EmitEvent(
        event=ChangeState(
            lifecycle_node_matcher=launch.events.matches_action(lifecycle_manager),
            transition_id=Transition.TRANSITION_SHUTDOWN
        )
    )

    return LaunchDescription([
        LogInfo(msg="Starting the container and lifecycle manager..."),
        container,
        lifecycle_manager,
        LogInfo(msg="Configuring the lifecycle node..."),
        configure_event,
        LogInfo(msg="Activating the lifecycle node..."),
        activate_event,
        LogInfo(msg="Deactivating the lifecycle node..."),
        deactivate_event,
        LogInfo(msg="Shutting down the lifecycle node..."),
        shutdown_event,
    ])