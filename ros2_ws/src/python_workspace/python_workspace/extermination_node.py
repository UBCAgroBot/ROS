import cv2
import sys
import os

import rclpy
from rclpy.time import Time
from rclpy.node import Node
from rclpy.executors import MultiThreadedExecutor
from cv_bridge import CvBridge

from sensor_msgs.msg import Image
from std_msgs.msg import Header, Int8, String
from custom_interfaces.msg import InferenceOutput

from .scripts.utils import ModelInference
sys.path.append(os.path.abspath('/home/user/ROS/python_wip'))
from display import display_annotated_image

class ExterminationNode(Node):
    def __init__(self):
        super().__init__('extermination_node')
        
        self.get_logger().info("Initializing Extermination Node")
        
        self.declare_parameter('use_display_node', True)
        self.declare_parameter('camera_side', 'left')
        self.use_display_node = self.get_parameter('use_display_node').get_parameter_value().bool_value
        self.camera_side = self.get_parameter('camera_side').get_parameter_value().string_value
        
        self.window = "Left Camera" if self.camera_side == "left" else "Right Camera" if self.use_display_node else None
        print(self.window)
        
        self.publishing_rate = 1.0
        self.lower_range = [78, 158, 124]
        self.upper_range = [60, 255, 255]
        self.minimum_area = 100
        self.minimum_confidence = 0.5
        self.boxes_present = 0
        self.model = ModelInference()
        self.bridge = CvBridge()
        self.boxes_msg = Int8()
        self.boxes_msg.data = 0
        
        self.inference_subscription = self.create_subscription(InferenceOutput, f'{self.camera_side}_inference_output', self.inference_callback, 10)
        self.box_publisher = self.create_publisher(Int8, f'{self.camera_side}_extermination_output', 10)
        self.timer = self.create_timer(self.publishing_rate, self.timer_callback)
        self.image_publisher = self.create_publisher(Image, 'annotated_image', 10)

    def inference_callback(self, msg):
        self.get_logger().info("Received Bounding Boxes")
        
        preprocessed_image = self.bridge.imgmsg_to_cv2(msg.preprocessed_image, desired_encoding='passthrough') # what is this needed for?
        raw_image = self.bridge.imgmsg_to_cv2(msg.raw_image, desired_encoding='passthrough')
        bounding_boxes = self.model.postprocess(msg.confidences.data,msg.bounding_boxes.data, raw_image,msg.velocity)
        # final_image = self.model.draw_boxes(raw_image,bounding_boxes,velocity=msg.velocity)
        labels = [f"{i}%" for i in msg.confidences.data]
        final_image = display_annotated_image(raw_image, bounding_boxes, labels)
        image_msg = self.bridge.cv2_to_imgmsg(final_image, encoding='bgr8')
        self.image_publisher.publish(image_msg)
        # if self.use_display_node:
        #     cv2.imshow("left window", final_image)
        #     cv2.waitKey(1)

        if len(bounding_boxes) > 0:
            self.boxes_present = 1
        else:
            self.boxes_present = 0
        
        self.boxes_msg = Int8()
        self.boxes_msg.data = self.boxes_present
    
    def timer_callback(self):
        self.box_publisher.publish(self.boxes_msg)
        self.get_logger().info("Published results to Proxy Node")

def main(args=None):
    rclpy.init(args=args)
    extermination_node = ExterminationNode()
    # executor = MultiThreadedExecutor(num_threads=1)
    # executor.add_node(extermination_node)
    # try:
    #     executor.spin()
    # except KeyboardInterrupt:
    #     print("Shutting down extermination node")
    # finally:
    #     executor.shutdown()
    #     extermination_node.destroy_node()
    #     rclpy.shutdown()
    try:
        rclpy.spin(extermination_node)
    except KeyboardInterrupt:
        extermination_node.get_logger().info("shutting down")
    finally:
        extermination_node.destroy_node()
        rclpy.shutdown()
        cv2.destroyAllWindows()

if __name__ == '__main__':
    main()