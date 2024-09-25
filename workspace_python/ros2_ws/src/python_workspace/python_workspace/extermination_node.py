import time, os
import cv2
import pycuda.driver as cuda
import numpy as np
import cupy as cp
import torch
import torchvision
# from tracker import *
# depth point cloud here...
# add object counter

import rclpy
from rclpy.time import Time
from rclpy.node import Node
from std_msgs.msg import Header, String, Integer
from rclpy.executors import MultiThreadedExecutor
from sensor_msgs.msg import Image

cuda.init()
device = cuda.Device(0)
cuda_driver_context = device.make_context()

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
        self.declare_parameter('side', 'left')
        
        self.use_display_node = self.get_parameter('use_display_node').get_parameter_value().bool_value
        self.lower_range = self.get_parameter('lower_range').get_parameter_value().integer_array_value
        self.upper_range = self.get_parameter('upper_range').get_parameter_value().integer_array_value
        self.min_area = self.get_parameter('min_area').get_parameter_value().integer_value
        self.min_confidence = self.get_parameter('min_confidence').get_parameter_value().double_value
        self.roi_list = self.get_parameter('roi_list').get_parameter_value().integer_array_value
        self.publish_rate = self.get_parameter('publish_rate').get_parameter_value().integer_value
        self.side = self.get_parameter('side').get_parameter_value().string_value
        
        if self.side == "left":
            side_topic = 'left_array_data'
            if self.use_display_node:
                self.window = "Left Camera"
        else:
            side_topic = 'right_array_data'
            if self.use_display_node:
                self.window = "Right Camera"
        
        self.timer = self.create_timer(1.0 / self.publish_rate, self.timer_callback)
        self.subscription_pointer = self.create_subscription(String, 'inference_done', self.pointer_callback, 10)
        self.subscription_img = self.create_subscription(Image, 'image', self.image_callback, 10)
        self.array_publisher = self.create_publisher(Integer, side_topic, self.publish_rate)
        self.on, self.image, self.boxes = 0, None, []
        self.fps, self.last_time = 0.0, time.time()
        self.roi_x, self.roi_y, self.roi_w, self.roi_h = self.roi_list

    def pointer_callback(self, ipc_handle_msg):
        self.get_logger().info(f"Received IPC handle: {ipc_handle_str}")
        ipc_handle_str = ipc_handle_msg.data
        ipc_handle = cuda.IPC_handle(ipc_handle_str) # Re-create the CuPy array from IPC handle
        d_input_ptr = ipc_handle.open()  # Map the shared memory to GPU
        self.run_nms(d_input_ptr)
        ipc_handle.close() # Clean up IPC handle
    
    # use shared image queue
    def retrieve_image(self):
        # self.image = ...
        pass

    def timer_callback(self):
        msg = Integer()
        msg.data = self.on
        self.array_publisher.publish(msg)
    
    # Perform NMS using PyTorch's GPU-accelerated function
    def run_nms(self, detections, iou_threshold=0.9):
        boxes = detections[:, :4]  # Extract boxes
        scores = detections[:, 4]  # Extract scores
        keep_indices = torchvision.ops.nms(boxes, scores, iou_threshold) 
        bboxes = detections[keep_indices]
        image = self.retrieve_image()
        self.object_filter(image, bboxes)
        return 

    # creates a color segmentation mask and filters small areas within bounding box frame
    def object_filter(self, image, bboxes):
        detections = []
        for bbox in bboxes:
            x1, y1, w, h = bbox[:, :4]
            roi = image[y1:y1+h, x1:x1+w] # Extract Region of interest (bounding box)
            hsv = cv2.cvtColor(roi, cv2.COLOR_RGB2HSV)
            mask = cv2.inRange(hsv,tuple(self.lower_range),tuple(self.upper_range)) # Apply color segmentation mask
            # result = cv2.bitwise_and(self.image,self.image, mask=mask) 
            _, mask = cv2.threshold(mask, 254, 255, cv2.THRESH_BINARY)
            contours, _ = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE) # Find contours to create bounding boxes from the mask
            for cnt in contours:
                # Calculate area and remove small elements
                area = cv2.contourArea(cnt)
                if area > self.min_area:
                    x, y, w, h = cv2.boundingRect(cnt)
                    detections.append((x, y, x + w, y + h))
        
        self.verify_object(image, detections)
    
    # filter out bounding boxes that are not within the ROI
    # make sure roi_x1... are ints
    def verify_object(self, disp_image, bboxes):
        roi_x1, roi_y1, roi_x2, roi_y2 = self.roi_list
        original_height, original_width = original_image_shape
        model_height, model_width = model_dimensions
        
        shifted_x = roi_x + abs(velocity[0]) * shift_constant
        scale_x = roi_w / model_width
        scale_y = roi_h / model_height
        
        adjusted_bboxes = []
        for bbox in bboxes:
            x1, y1, x2, y2 = bbox
            # Bounding box is in [x_min, y_min, x_max, y_max] format

            # Step 1: Reverse the resize operation
            x_min = int(bbox[0] * scale_x)
            y_min = int(bbox[1] * scale_y)
            x_max = int(bbox[2] * scale_x)
            y_max = int(bbox[3] * scale_y)

            # Step 2: Reverse the cropping operation by adding the crop offsets (shifted_x, roi_y)
            x_min += shifted_x
            y_min += roi_y
            x_max += shifted_x
            y_max += roi_y

            # Ensure the bounding box doesn't exceed the original image dimensions
            x_min = max(0, min(x_min, original_width))
            y_min = max(0, min(y_min, original_height))
            x_max = max(0, min(x_max, original_width))
            y_max = max(0, min(y_max, original_height))

            # Append the adjusted bounding box
            adjusted_bboxes.append([x_min, y_min, x_max, y_max])

            if x1 >= roi_x1 and x2 <= roi_x2 and y1 >= roi_y1 and y2 <= roi_y2:
                self.on = 1
                if self.display:
                    pass
            
        if self.display:
            self.display(disp_image, adjusted_bboxes)

    
    def display(self, image, bboxes):
        # Calculate FPS
        toc = time.time()
        curr_fps = 1 / (toc - self.last_time)
        self.fps = curr_fps if self.fps == 0.0 else (self.fps*0.95 + curr_fps*0.05)
        self.last_time = toc
        
        cv2.putText(disp_image, f"FPS: {round(self.fps)}", (10, 30),
        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2, cv2.LINE_AA)
            
        cv2.namedWindow(self.window)
        
        for bbox in bboxes:
            
            cv2.rectangle(disp_image, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2)
        
        cv2.imshow(disp_image)
        key = cv2.waitKey(0)
        if key == 27:  # ESC key: quit program
            cv2.destroyAllWindows()
            return
    
    def display(self, final_boxes):
        image = self.image
        for box in final_boxes:
            x, y, w, h, confidence = box
            cv2.rectangle(image, (x+h, y+w), (x-w, y-h), (255, 0, 0), 2)
        cv2.imshow("Image", image)
        cv2.waitKey(0)

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