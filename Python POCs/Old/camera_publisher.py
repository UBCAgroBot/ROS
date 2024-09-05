import cv2
import matplotlib.pyplot as plt
import numpy as np
from decord import VideoReader, cpu
import os
import time

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from cv_bridge import CvBridge, CvBridgeError

TYPE1 = "Webcam"
TYPE2 = "Video"
TYPE3 = "Image"
TYPE4 = "ZED Camera"
DELAY = 0.1

os.chdir("23-I-12_SysArch/Experiments/extermination_workspace")

class CameraNode(Node):
    def __init__(self):
        super().__init__('camera_node')
        self.initialize()
        self.image = None
        self.bridge = CvBridge()
        self.type = TYPE1
        self.on = True
        self.frame_publisher = self.create_publisher(Image, 'source', 10)
        
        while True:
            if self.type == TYPE1:
                self.image = self.source_1()
            elif self.type == TYPE2:
                self.image = self.source_2()
            elif self.type == TYPE3:
                self.image = self.source_3()
            elif self.type == TYPE4:
                self.image = self.source_4()
            
            if self.image is not None:
                msg = self.bridge.cv2_to_imgmsg(self.image, "bgr8")
                self.frame_publisher.publish(msg)
                time.sleep(DELAY)
                self.toggle_on_off()

    def source_1(self):
        cap = cv2.VideoCapture(0)
        while self.on:
            ret, frame = cap.read()
            if ret:
                self.image = frame
            if cv2.waitKey(1) & 0xFF == ord('q'):
                self.on = not self.on
                break
        cap.release()
    
    def source_2(self):
        vr = VideoReader('examples/flipping_a_pancake.mkv', ctx=cpu(0))
        index = 0
        while self.on:
            self.image = vr[index]
            index += 1
            time.sleep(DELAY)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                self.on = not self.on
                break

    def source_3(self):
        while self.on:
            self.image = cv2.imread("test.jpg")
            time.sleep(DELAY)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                self.on = not self.on
                break
            
    def source_4(self):
        import pyzed.sl as sl
        init = sl.InitParameters()
        cam = sl.Camera()
        init.camera_resolution = sl.RESOLUTION.HD1080
        init.camera_fps = 30
        
        if not cam.is_opened():
            print("Opening ZED Camera ")
        status = cam.open(init)
        if status != sl.ERROR_CODE.SUCCESS:
            print(repr(status))
            exit()
        
        runtime = sl.RuntimeParameters()
        mat = sl.Mat()
        
        while self.on:
            err = cam.grab(runtime)
            if err == sl.ERROR_CODE.SUCCESS:
                cam.retrieve_image(mat, sl.VIEW.LEFT_UNRECTIFIED)
                image = mat.get_data()
                self.image = cv2.cvtColor(image, cv2.COLOR_RGBA2RGB)
            
            if cv2.waitKey(1) & 0xFF == ord('q'):
                self.on = not self.on
                break
        
        self.cam.close()
        print("ZED Camera closed")

    def toggle_on_off(self):
        if cv2.waitKey(1) & 0xFF == ord('q'):
            self.on = not self.on
