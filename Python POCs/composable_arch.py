import rclpy
from rclpy.node import Node
from std_msgs.msg import String

class ComposablePublisherNode(Node):
    def __init__(self):
        super().__init__('composable_publisher_node')
        self.publisher_ = self.create_publisher(String, 'topic', 10)
        timer_period = 2.0  # seconds
        self.timer = self.create_timer(timer_period, self.timer_callback)
        self.get_logger().info("ComposablePublisherNode has been started.")

    def timer_callback(self):
        msg = String()
        msg.data = 'Hello from Composable Node!'
        self.publisher_.publish(msg)
        self.get_logger().info('Publishing: "%s"' % msg.data)

def main(args=None):
    rclpy.init(args=args)
    node = ComposablePublisherNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
