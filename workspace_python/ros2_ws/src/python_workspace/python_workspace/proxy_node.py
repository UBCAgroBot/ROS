import serial

import rclpy
from rclpy.time import Time
from rclpy.node import Node
from std_msgs.msg import Bool
from rclpy.executors import MultiThreadedExecutor
from std_msgs.msg import String,Header, Int8

class ProxyNode(Node):
    def __init__(self):
        super().__init__('proxy_node')
        
        self.declare_parameter('usb_port', '/dev/ttyACM0')
        self.port = self.get_parameter('usb_port').get_parameter_value().string_value
        
        try:
            self.ser = serial.Serial(self.port, 115200, timeout=1)
        except Exception as e:
            self.get_logger().error(f"Failed to connect to serial port: {e}")
        
        self.left_subscription = self.create_subscription(Int8, 'left_extermination_output', self.left_callback, 10)
        self.right_subscription = self.create_subscription(Int8, 'right_extermination_output', self.right_callback, 10)
    
    def left_callback(self, msg):
        result = str(msg.data)
        if result == "0":
            serialized_msg = str(0) + '\n'
            self.ser.wrie(serialized_msg.encode())
            self.get_logger().info(f'Sent to Arduino: 0 (left off)')
        else:
            serialized_msg = str(1) + '\n'
            self.ser.wrie(serialized_msg.encode())
            self.get_logger().info(f'Sent to Arduino: 1 (left on)')
    
    def right_callback(self, msg):
        result = str(msg.data)
        if result == "0":
            serialized_msg = str(2) + '\n'
            self.ser.wrie(serialized_msg.encode())
            self.get_logger().info(f'Sent to Arduino: 2 (right off)')
        else:
            serialized_msg = str(3) + '\n'
            self.ser.wrie(serialized_msg.encode())
            self.get_logger().info(f'Sent to Arduino: 3 (right on)')
    
def main(args=None):
    rclpy.init(args=args)
    proxy_node = ProxyNode()
    executor = MultiThreadedExecutor(num_threads=2)
    executor.add_node(proxy_node)
    try:
        executor.spin()
    except KeyboardInterrupt:
        print("Shutting down proxy node")
    finally:
        executor.shutdown()
        proxy_node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()