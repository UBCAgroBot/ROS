import time, sys, os
import cv2
import numpy as np
# from tracker import *
# add object counter

import rclpy
from rclpy.node import Node
from rclpy.executors import MultiThreadedExecutor
from sensor_msgs.msg import Image
from std_msgs.msg import Header, String, Integer
from cv_bridge import CvBridge, CvBridgeError

class ExterminationNode(Node):
    def __init__(self):
        super().__init__('extermination_node')
        self.declare_parameter('use_display_node', True)
        self.declare_parameter('lower_range', [78, 158, 124])
        self.declare_parameter('upper_range', [60, 255, 255])
        self.declare_parameter('min_area', 100)
        self.declare_parameter('min_confidence', 0.5)
        self.declare_parameter('roi_list', [0,0,100,100])
        self.declare_parameter('publish_rate', 10)
        
        self.use_display_node = self.get_parameter('use_display_node').get_parameter_value().bool_value
        self.lower_range = self.get_parameter('lower_range').get_parameter_value().integer_array_value
        self.upper_range = self.get_parameter('upper_range').get_parameter_value().integer_array_value
        self.min_area = self.get_parameter('min_area').get_parameter_value().integer_value
        self.min_confidence = self.get_parameter('min_confidence').get_parameter_value().double_value
        self.roi_list = self.get_parameter('roi_list').get_parameter_value().integer_array_value
        self.publish_rate = self.get_parameter('publish_rate').get_parameter_value().integer_value
        
        self.subscription_img = self.create_subscription(Image, 'image', self.image_callback, 10)
        self.subscription_bbox = self.create_subscription(String, 'bounding_boxes', self.bbox_callback, 10)
        self.timer = self.create_timer(0.1, self.timer_callback)
        self.left_array_publisher = self.create_publisher(Integer, 'left_array_data', self.publish_rate)
        self.right_array_publisher = self.create_publisher(Integer, 'right_array_data', self.publish_rate)
        self.bridge = CvBridge()
        self.left_image, self.right_image = None, None
        self.left.on, self.right.on = 0, 0
        self.left_boxes, self.right_boxes = None, None
        self.left_velocity, self.right_velocity = 0, 0
        self.left_x1, self.left_y1, self.left_x2, self.left_y2 = 0, 0, 0, 0
        self.right_x1, self.right_y1, self.right_x2, self.right_y2 = 0, 0, 0, 0
        self.fps, self.last_time, self.latency = 0.0, time.time(), 0.1
        self.roi_x, self.roi_y, self.roi_w, self.roi_h = self.roi_list
        
        if self.use_display_node:
            self.window1 = "Left Camera"
            self.window2 = "Right Camera"
            cv2.namedWindow(self.window1)
            cv2.namedWindow(self.window2)

    def image_callback(self, msg):
        if msg.header.side == "left":
            self.left_image = self.bridge.imgmsg_to_cv2(msg, "rgb8")
            # self.left_velocity = msg.header.velocity
            # self.left_x1 = self.roi_x + (self.left_velocity[0] * self.latency)
            # self.left_y1 = self.roi_y + (self.left_velocity[1] * self.latency)
            # self.left_x2 = self.left_x1 + self.roi_w
            # self.left_y2 = self.left_y1 + self.roi_h
        else:
            self.right_image = self.bridge.imgmsg_to_cv2(msg, "rgb8")
            # self.right_velocity = msg.header.velocity
            # self.right_velocity = msg.header.velocity
            # self.right_x1 = self.roi_x - (self.right_velocity[0] * self.latency)
            # self.right_y1 = self.roi_y - (self.right_velocity[1] * self.latency)
            # self.right_x2 = self.right_x1 + self.roi_w
            # self.right_y2 = self.right_y1 + self.roi_h
        self.get_logger().info(f"Received {msg.header.side} image data")

    def bbox_callback(self, msg):
        if msg.header.side == "left":
            self.left_boxes = (msg.data).split(";")
            self.process_data("left")
        else:
            self.right_boxes = (msg.data).split(";")
            self.process_data("right")
        self.get_logger().info(f"Received {msg.header.side} bounding box data")

    def timer_callback(self, side):
        Lmsg = Integer()
        Lmsg.data = self.left.on
        Rmsg = Integer()
        Rmsg.data = self.right.on
        self.left_array_publisher.publish(Lmsg)
        self.right_array_publisher.publish(Rmsg)

    def process_data(self, side):
        if side == "left":
            image = self.left_image
            boxes = self.left_boxes
        else:
            image = self.right_image
            boxes = self.right_boxes
        
        if image is not None and boxes is not None:
            for box in boxes:
                # hard confidence filter
                # if box.confidence > self.min_confidence:
                bounding_box = box.split(",")
                self.object_filter(image, bounding_box, side)

    # creates a color segmentation mask and filters small areas
    # ROI is the bounding box of the object (screening objects)
    # this function will take cropped image w/ bounding box to clean it up
    def object_filter(self, image, bounding_box, side):
        # Extract Region of interest (bounding box)
        x1, y1, w, h = bounding_box
        roi = image[y1:y1+h, x1:x1+w]

        # Apply color segmentation mask
        hsv = cv2.cvtColor(roi, cv2.COLOR_RGB2HSV)
        mask = cv2.inRange(hsv,tuple(self.lower_range),tuple(self.upper_range))
        # result = cv2.bitwise_and(self.image,self.image, mask=mask) 
        
        # Find contours to create bounding boxes from the mask
        _, mask = cv2.threshold(mask, 254, 255, cv2.THRESH_BINARY)
        contours, _ = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        detections = []
        for cnt in contours:
            # Calculate area and remove small elements
            area = cv2.contourArea(cnt)
            if area > self.min_area:
                x, y, w, h = cv2.boundingRect(cnt)
                detections.append((x, y, x + w, y + h))
        
        # !!! need to transform from roi coordinates to image coordinates
        # (relative to the image instead of the bounding)
        
        # if self.use_display_node:
        #     self.display(roi, detections, side)
        
        for detection in detections:
            self.verify_object(detection, side)
    
    # add logging for output here...
    # filter out bounding boxes that are not within the ROI
    def verify_object(self, bounding_box, side):
        x1, y1, x2, y2 = bounding_box
        if side == "left":
            roi_x1 = self.left_x1
            roi_y1 = self.left_y1
            roi_x2 = self.left_x2
            roi_y2 = self.left_y2
        else:
            roi_x1 = self.right_x1
            roi_y2 = self.right_y1
            roi_x2 = self.right_x2
            roi_y2 = self.right_y2
        
        if x1 >= roi_x1 and x2 <= roi_x2 and y1 >= roi_y1 and y2 <= roi_y2:
            if side == "left":
                self.left.on = 1
            else:
                self.right.on = 1
        else:
            if side == "left":
                self.left.on = 0
            else:
                self.right.on = 0

    # parameters for CNN display, color segmentaion, object bounding boxes at diff stages
    # propagate parameters for display
    def display(self, image, boxes, side):
        disp_image = image.copy()
        # Overlay bounding boxes on the image
        for box in boxes:
            x, y, w, h = box[0], box[1], box[2], box[3]
            self.get_logger().info(f'Bounding box: ({x}, {y}), ({w}, {h})') # confidence: {box.confidence}
            # Convert coordinates and dimensions from normalized to absolute
            # x, y, w, h = int(x * image.shape[1]), int(y * image.shape[0]), int(w * image.shape[1]), int(h * image.shape[0])
            # Draw the bounding box
            # cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)
            cv2.rectangle(disp_image, (int(box.x_min), int(box.y_min)), (int(box.x_max), int(box.y_max)), (0, 255, 0), 2)
            cv2.putText(disp_image, f"Conf: {box.confidence:.2f}", (int(box.x_min), int(box.y_min) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1, cv2.LINE_AA)
        
        # Calculate FPS
        toc = time.time()
        curr_fps = 1 / (toc - self.last_time)
        self.fps = curr_fps if self.fps == 0.0 else (self.fps*0.95 + curr_fps*0.05)
        self.last_time = toc

        # Display FPS on the image
        cv2.putText(disp_image, f"FPS: {round(self.fps)}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2, cv2.LINE_AA)
        
        if side == "left":
            cv2.imshow(self.window1, disp_image)
        else:
            cv2.imshow(self.window2, disp_image)
        
        key = cv2.waitKey(1)
        if key == 27:  # ESC key: quit program
            cv2.destroyAllWindows()
            sys.exit()
        

def main(args=None):
    rclpy.init(args=args)
    extermination_node = ExterminationNode()
    # try:
    #     rclpy.spin(display_node)
    # except SystemExit:    
    #     rclpy.logging.get_logger("Quitting").info('Done')
    executor = MultiThreadedExecutor()
    executor.add_node(extermination_node)
    executor.spin()
    extermination_node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()