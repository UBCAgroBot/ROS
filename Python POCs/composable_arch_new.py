import rclpy
from rclpy.node import Node
from std_msgs.msg import String

class ComposablePublisherNode(Node):
    def __init__(self):
        super().__init__('composable_publisher_node')
        
        # Declare a parameter and use it in the node
        self.declare_parameter('message', 'Hello from Composable Node!')
        self.declare_parameter('publish_frequency', 2.0)
        
        self.publisher_ = self.create_publisher(String, 'topic', 10)
        
        # Fetch the parameters
        self.message = self.get_parameter('message').get_parameter_value().string_value
        self.publish_frequency = self.get_parameter('publish_frequency').get_parameter_value().double_value
        
        self.timer = self.create_timer(self.publish_frequency, self.timer_callback)
        self.get_logger().info("ComposablePublisherNode has been started with message: '%s'" % self.message)

    def timer_callback(self):
        msg = String()
        msg.data = self.message
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

# allows parameters