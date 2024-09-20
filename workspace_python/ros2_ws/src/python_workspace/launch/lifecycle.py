import rclpy
from rclpy.node import Node
from rclpy.lifecycle import LifecycleNode
from rclpy.lifecycle import State

class CameraNode(LifecycleNode):
    def __init__(self):
        super().__init__('camera_node')

    def on_configure(self, state: State):
        self.get_logger().info('Configuring the camera...')
        # Initialize camera
        return self.configure_success()

    def on_activate(self, state: State):
        self.get_logger().info('Activating the camera...')
        # Start the camera stream
        return self.activate_success()

    def on_deactivate(self, state: State):
        self.get_logger().info('Deactivating the camera...')
        # Stop the camera stream
        return self.deactivate_success()

    def on_cleanup(self, state: State):
        self.get_logger().info('Cleaning up resources...')
        # Clean up resources
        return self.cleanup_success()

    def on_shutdown(self, state: State):
        self.get_logger().info('Shutting down the camera...')
        # Shut down resources
        return self.shutdown_success()

from launch import LaunchDescription
from launch_ros.actions import LifecycleNode
from launch.actions import EmitEvent
from launch.events import matches_action
from launch_ros.events.lifecycle import ChangeState

from lifecycle_msgs.msg import Transition

def generate_launch_description():
    return LaunchDescription([
        LifecycleNode(
            package='your_package',
            executable='your_lifecycle_node',
            name='lifecycle_node_1',
            namespace='',
            output='screen'
        ),
        LifecycleNode(
            package='another_package',
            executable='another_lifecycle_node',
            name='lifecycle_node_2',
            namespace='',
            output='screen'
        ),
    ])

## automatic lifecycle transition launch:
from launch import LaunchDescription
from launch_ros.actions import LifecycleNode
from launch.actions import EmitEvent
from launch.events import matches_action
from launch_ros.events.lifecycle import ChangeState
from lifecycle_msgs.msg import Transition

def generate_launch_description():
    lifecycle_node_1 = LifecycleNode(
        package='your_package',
        executable='your_lifecycle_node',
        name='lifecycle_node_1',
        namespace='',
        output='screen'
    )

    return LaunchDescription([
        # Launch the lifecycle node
        lifecycle_node_1,

        # Emit an event to automatically configure the node
        EmitEvent(
            event=ChangeState(
                lifecycle_node_matcher=matches_action(lifecycle_node_1),
                transition_id=Transition.TRANSITION_CONFIGURE
            )
        ),

        # Emit an event to automatically activate the node after configuring
        EmitEvent(
            event=ChangeState(
                lifecycle_node_matcher=matches_action(lifecycle_node_1),
                transition_id=Transition.TRANSITION_ACTIVATE
            )
        ),
    ])

# ComposableNodeContainer: This is the container where composable nodes are dynamically loaded. It's similar to a "node manager" that manages composable nodes.
# ComposableNode: This describes your node (in this case, the CameraNode) which will be loaded into the container.