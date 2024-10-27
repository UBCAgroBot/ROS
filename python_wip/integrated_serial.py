import rclpy
from rclpy.node import Node
import serial
import time
from std_msgs.msg import String  # Example message type

class ArduinoSerialNode(Node):
    def __init__(self):
        super().__init__('arduino_serial_node')
        self.subscription = self.create_subscription(
            String,
            'your_topic_name',
            self.listener_callback,
            10
        )
        self.subscription  # prevent unused variable warning
        
        # Open serial port to Arduino
        self.ser = serial.Serial('/dev/ttyUSB0', 115200, timeout=1)  # Adjust USB port as needed
        time.sleep(2)  # Wait for Arduino to reset

    def listener_callback(self, msg):
        # Serialize and send the message to Arduino
        serialized_msg = msg.data + '\n'  # Add a newline as a delimiter
        self.ser.write(serialized_msg.encode())
        self.get_logger().info('Sent to Arduino: "%s"' % msg.data)

def main(args=None):
    rclpy.init(args=args)
    arduino_serial_node = ArduinoSerialNode()
    rclpy.spin(arduino_serial_node)
    arduino_serial_node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()

# pip3 install pyserial