# lifecycle_node.py
import rclpy
from rclpy.lifecycle import LifecycleNode
from rclpy.lifecycle import State
from rclpy.lifecycle import TransitionCallbackReturn
from rclpy.executors import MultiThreadedExecutor

class LifecycleManagerNode(LifecycleNode):
    def __init__(self):
        super().__init__('lifecycle_manager_node')
        self.get_logger().info('LifecycleManagerNode has been created.')

    def on_configure(self, state: State) -> TransitionCallbackReturn:
        self.get_logger().info('on_configure() called')
        return TransitionCallbackReturn.SUCCESS

    def on_activate(self, state: State) -> TransitionCallbackReturn:
        self.get_logger().info('on_activate() called')
        return TransitionCallbackReturn.SUCCESS

    def on_deactivate(self, state: State) -> TransitionCallbackReturn:
        self.get_logger().info('on_deactivate() called')
        return TransitionCallbackReturn.SUCCESS

    def on_cleanup(self, state: State) -> TransitionCallbackReturn:
        self.get_logger().info('on_cleanup() called')
        return TransitionCallbackReturn.SUCCESS

    def on_shutdown(self, state: State) -> TransitionCallbackReturn:
        self.get_logger().info('on_shutdown() called')
        return TransitionCallbackReturn.SUCCESS

def main(args=None):
    rclpy.init(args=args)
    
    lifecycle_node = LifecycleManagerNode()

    executor = MultiThreadedExecutor()
    executor.add_node(lifecycle_node)

    try:
        rclpy.spin(lifecycle_node, executor=executor)
    except KeyboardInterrupt:
        pass

    lifecycle_node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
