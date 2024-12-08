import cv2
import sys
import os

import rclpy
from rclpy.node import Node
from cv_bridge import CvBridge

from sensor_msgs.msg import Image
sys.path.append(os.path.abspath('/home/user/ROS/python_wip'))

class DisplayNode(Node):
    def __init__(self):
        super().__init__('display_node')
        self.subscription = self.create_subscription(Image, 'annotated_image', self.listener_callback, 10)  # QoS profile depth)
        self.bridge = CvBridge() 
        self.get_logger().info("Intializing Display Node")
        self.window = "Left Camera"
    
    def image_callback(self, msg):
        # Convert ROS Image message to OpenCV image
        cv_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
        # Display the image using OpenCV
        cv2.imshow('Live Image Feed', cv_image)
        # Refresh the OpenCV window
        cv2.waitKey(1)  # Delay for 1ms

def main(args=None):
    rclpy.init(args=args)
    node = DisplayNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        node.get_logger().info('Shutting down Image Subscriber Node...')
    finally:
        node.destroy_node()
        rclpy.shutdown()
        cv2.destroyAllWindows()

if __name__ == '__main__':
    main()
