import cv2
# import pycuda.driver as cuda
# from tracker import *
# depth point cloud here...
# add object counter
from cv_bridge import CvBridge
import numpy as np

import rclpy
from rclpy.time import Time
from rclpy.node import Node
from rclpy.executors import MultiThreadedExecutor

from sensor_msgs.msg import Image
from std_msgs.msg import Int8
from custom_interfaces.msg import InferenceOutput
from custom_interfaces.srv import GetRowPlantCount

from .scripts.utils import postprocess, draw_boxes
from .scripts.tracker import EuclideanDistTracker

class ExterminationNode(Node):
    def __init__(self):
        super().__init__('extermination_node')
    
        self.declare_parameter('use_display_node', False)
        self.declare_parameter('camera_side', 'left')

        self.use_display_node = self.get_parameter('use_display_node').get_parameter_value().bool_value
        self.camera_side = self.get_parameter('camera_side').get_parameter_value().string_value
        self.window = "Left Camera" if self.camera_side == "left" else "Right Camera" if self.use_display_node else None

        self.publishing_rate = 1.0
        self.lower_range = [78, 158, 124]
        self.upper_range = [60, 255, 255]
        self.minimum_area = 100
        self.minimum_confidence = 0.5
        
        self.boxes_present = 0
        self.tracker = EuclideanDistTracker()
        self.bridge = CvBridge()
        self.boxes_msg = Int8()
        self.boxes_msg.data = 0

        self.inference_subscription = self.create_subscription(InferenceOutput, f'{self.camera_side}_inference_output', self.inference_callback, 10)
        self.box_publisher = self.create_publisher(Int8, f'{self.camera_side}_extermination_output', 10)
        self.timer = self.create_timer(self.publishing_rate, self.timer_callback)

        self.get_tracker_row_count_service = self.create_service(GetRowPlantCount, 'reset_tracker', self.get_tracker_row_count_callback)


    def get_tracker_row_count_callback(self,request,response):
        """
        When navigation requests this service, reset the tracker count so that it knows to start a new row. 
        return the tracker's current row count
        """
        row_count = self.tracker.reset()
        response.plant_count = row_count
        return response

    def inference_callback(self, msg):
        # self.get_logger().info("Received Bounding Boxes")

        preprocessed_image = self.bridge.imgmsg_to_cv2(msg.preprocessed_image, desired_encoding='passthrough')
        raw_image = self.bridge.imgmsg_to_cv2(msg.raw_image, desired_encoding='passthrough')

        confidence = np.reshape(msg.confidences.data, (-1))
        bboxes = np.reshape(msg.bounding_boxes.data, (-1,4))
        
        bounding_boxes = postprocess(confidence,bboxes, raw_image,msg.velocity)

        bbox_with_label = self.tracker.update(bounding_boxes)

        final_image = draw_boxes(raw_image,bbox_with_label,velocity=msg.velocity)


        if self.use_display_node:
            cv2.imshow(self.window, final_image)
            cv2.waitKey(10)

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
    executor = MultiThreadedExecutor(num_threads=1)
    executor.add_node(extermination_node)
    try:
        executor.spin()
    except KeyboardInterrupt:
        print("Shutting down extermination node")
    finally:
        executor.shutdown()
        extermination_node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()