import time, os
import cv2
import serial
# import pycuda.driver as cuda
# from tracker import *
# depth point cloud here...
# add object counter
from cv_bridge import CvBridge
import numpy as np

import rclpy
from rclpy.time import Time
from rclpy.node import Node
# from std_msgs.msg import Header, String, Integer
from rclpy.executors import MultiThreadedExecutor
from sensor_msgs.msg import Image
from custom_interfaces.msg import InferenceOutput                            # CHANGE
from .scripts.utils import postprocess, draw_boxes

# cuda.init()
# device = cuda.Device(0)
# cuda_driver_context = device.make_context()

class ExterminationNode(Node):
    def __init__(self):
        super().__init__('extermination_node')
    
        self.declare_parameter('use_display_node', True)
        # self.declare_parameter('lower_range', [78, 158, 124]) #todo: make this a parameter
        # self.declare_parameter('upper_range', [60, 255, 255])
        # self.declare_parameter('min_area', 100)
        # self.declare_parameter('min_confidence', 0.5)
        # self.declare_parameter('roi_list', [0,0,100,100])
        # self.declare_parameter('publish_rate', 10)
        # self.declare_parameter('side', 'left')
        
        self.use_display_node = self.get_parameter('use_display_node').get_parameter_value().bool_value
        # self.lower_range = self.get_parameter('lower_range').get_parameter_value().integer_array_value
        # self.upper_range = self.get_parameter('upper_range').get_parameter_value().integer_array_value
        # self.min_area = self.get_parameter('min_area').get_parameter_value().integer_value
        # self.min_confidence = self.get_parameter('min_confidence').get_parameter_value().double_value
        # self.roi_list = self.get_parameter('roi_list').get_parameter_value().integer_array_value
        # self.publish_rate = self.get_parameter('publishsource install/setup.bash
        # if self.side == "left":
        #     side_topic = 'left_array_data'
        #     if self.use_display_node:
        #         self.window = "Left Camera"
        # else:
        #     side_topic = 'right_array_data'
        #     if self.use_display_node:
        #         self.window = "Right Camera"

        self.boxes_present = 0
        self.window = "Left Camera"
        self.bridge = CvBridge()
        # Open serial port to Arduino
        self.ser = serial.Serial('/dev/ttyACM0', 115200, timeout=1)
        self.subscription = self.create_subscription(InferenceOutput, 'inference_out', self.inference_callback, 10)

        # Create a timer that calls the listener_callback every second
        self.timer = self.create_timer(1.0, self.timer_callback)

        time.sleep(2)  # Wait for Arduino to reset

    def inference_callback(self, msg):
        preprocessed_image = self.bridge.imgmsg_to_cv2(msg.preprocessed_image, desired_encoding='passthrough')
        raw_image = self.bridge.imgmsg_to_cv2(msg.raw_image, desired_encoding='passthrough')

        confidence = np.reshape(msg.confidences.data, (-1))
        bboxes = np.reshape(msg.bounding_boxes.data, (-1,4))

        bounding_boxes = postprocess(confidence,bboxes, raw_image,msg.velocity)
        
        final_image = draw_boxes(raw_image,bounding_boxes,velocity=msg.velocity)

        if self.use_display_node:
        # Create a CUDA window and display the cv_image
            cv2.imshow('CUDA Window', final_image)
            cv2.waitKey(10)

        if len(bounding_boxes) > 0:
            self.boxes_present = 1
        else:
            self.boxes_present = 0

    def timer_callback(self):
        # Serialize and send the message to Arduino
        serialized_msg = str(self.boxes_present) + '\n'  # Add a newline as a delimiter
        self.ser.write(serialized_msg.encode())
        self.get_logger().info(f'Sent to Arduino: {self.boxes_present}')

def main(args=None):
    rclpy.init(args=args)
    extermination_node = ExterminationNode()
    executor = MultiThreadedExecutor(num_threads=1)
    executor.add_node(extermination_node)
    try:
        executor.spin()
    finally:
        executor.shutdown()
        extermination_node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()