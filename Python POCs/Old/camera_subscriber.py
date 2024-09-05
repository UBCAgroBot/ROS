import cv2
import numpy as np
from matplotlib.colors import hsv_to_rgb
import matplotlib.pyplot as plt
from tracker import *

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from std_msgs.msg import Header, String, Integer
from cv_bridge import CvBridge, CvBridgeError

PUBLISH_RATE = 10
LIGHT_GREEN = (78, 158, 124)
DARK_GREEN = (60, 255, 255)
ROI_X = 0
ROI_Y = 0
ROI_W = 100
ROI_H = 100
UPDATE_RATE = 1

class CameraNode(Node):
    def __init__(self):
        super().__init__('camera_node') #type: ignore
        self.bridge = CvBridge()
        self.tracker = EuclideanDistTracker()
        self.queue = []
        self.image = None
        self.left.on, self.right.on = 0, 0
        
        self.left_camera_subscriber = self.create_subscription(Image, 'image_data', self.callback, 10)
        self.right_camera_subscriber = self.create_subscription(Image, 'image_data', self.callback, 10)
        self.timer = self.create_timer(0.1, self.timer_callback)
        self.left_array_publisher = self.create_publisher(Integer, 'left_array_data', 10)
        self.right_array_publisher = self.create_publisher(Integer, 'right_array_data', 10)

    def timer_callback(self):
        Lmsg = Integer()
        Lmsg.data = self.left.on
        Rmsg = Integer()
        Rmsg.data = self.right.on
        self.left_array_publisher.publish(Lmsg)
        self.right_array_publisher.publish(Rmsg)
    
    def callback(self, msg):
        try:
            cv_image = self.bridge.imgmsg_to_cv2(msg, "rgb8")
            self.queue.append(cv_image)
            self.preprocess_image()
        except CvBridgeError as e:
            print(e)
    
    def preprocess_image(self):
        if len(self.queue) == 2:
            left_image = self.queue.pop(0)
            right_image = self.queue.pop(0)
            stacked_image = cv2.hconcat([left_image, right_image])
            print(stacked_image.shape)
            padded_image = cv2.copyMakeBorder(stacked_image, 0, 0, 0, 0, cv2.BORDER_CONSTANT, value=[0, 0, 0]) # add padding to make it square
            print(padded_image.shape)
            resized_image = cv2.resize(padded_image, (255, 255))
            plt.imshow(resized_image)
            plt.show()
            self.image = resized_image
            self.model_inference()
    
    def model_inference(self):
        # model inference here...
        self.postprocess_image()
    
    def postprocess_image(self):
        # Read the image
        img = cv2.imread('your file name')
        print(img.shape)
        height = img.shape[0]
        width = img.shape[1]

        # Cut the image in half
        width_cutoff = width // 2
        s1 = img[:, :width_cutoff]
        s2 = img[:, width_cutoff:]
    
    # def object_filter(self, image, bounding_box):
    #     height, width = self.image.shape
    #     x1, y1, x2, y2 = bounding_box
        
    #     # Extract Region of interest
    #     roi = image[y1:y2, x1:x2]

    #     # Apply color segmentation mask
    #     hsv = cv2.cvtColor(self.image, cv2.COLOR_RGB2HSV)
    #     mask = cv2.inRange(hsv,LIGHT_GREEN,DARK_GREEN)
    #     result = cv2.bitwise_and(self.image,self.image, mask=mask) 
    #     plt.subplot(1, 2, 1)
    #     plt.imshow(mask, cmap="gray")
    #     plt.subplot(1, 2, 2)
    #     plt.imshow(result)
    #     plt.show()
    #     bbox = cv2.boundingRect(mask)
        
    #     if bbox is not None:
    #         x, y, w, h = bbox
    #         detections.append([x, y, w, h])
    #         boxes_ids = self.tracker.update(detections)
    #         for box_id in boxes_ids:
    #             x, y, w, h, id = box_id
    #             cv2.putText(roi, str(id), (x, y - 15), cv2.FONT_HERSHEY_PLAIN, 2, (255, 0, 0), 2)
    #             cv2.rectangle(roi, (x, y), (x + w, y + h), (0, 255, 0), 3)
        
    #     cv2.imshow("roi", roi)
    #     cv2.imshow("Frame", image)
    #     cv2.imshow("Mask", mask)
        
    #     _, mask = cv2.threshold(mask, 254, 255, cv2.THRESH_BINARY)
    #     contours, _ = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    #     detections = []
    #     for cnt in contours:
    #         # Calculate area and remove small elements
    #         area = cv2.contourArea(cnt)
    #         if area > 100:
    #             #cv2.drawContours(roi, [cnt], -1, (0, 255, 0), 2)
    #             x, y, w, h = cv2.boundingRect(cnt)
    #             # this function will take cropped image w/ bounding box to clean it up
    
    from object_filter import object_filter
    def object_filter(self, image, bounding_box):
        rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        detections = object_filter(rgb_image, bounding_box)
        for detection in detections:
            x, y, w, h = detection
            cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 3)
        
        cv2.imshow("Processed Image", image)
        cv2.waitKey(1)

    def verify_object(self, bounding_box, side):
        x1, y1, w, h = bounding_box
        x2 = x1 + w
        y2 = y1 + h
        if x1 >= ROI_X and x1 <= ROI_X + ROI_W and y1 >= ROI_Y and y1 <= ROI_Y + ROI_H:
            if side == "left":
                self.left.on = 1
            else:
                self.right.on = 1
        else:
            if side == "left":
                self.left.on = 0
            else:
                self.right.on = 0