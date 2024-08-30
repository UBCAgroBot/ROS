import time
import sys
import cv2
import os
from pathlib import Path
import pyzed.sl as sl

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from std_msgs.msg import Header, String
from cv_bridge import CvBridge, CvBridgeError
from example_interfaces.srv import Trigger

class CameraNode(Node):
    def __init__(self):
        super().__init__('image_publisher') #type: ignore
        self.bridge = CvBridge()
        self.download_path = str(Path.home() / 'Downloads')
        # replace self.camera with parameter
        self.index, self.camera, self.frames, self.type, self.total_data, self.done = 0, True, [], "8UC4", 0, False
        # replace with action server-client architecture later after verifying frame consistency
        self.off_publisher = self.create_publisher(String, 'off', 10)
        self.model_publisher = self.create_publisher(Image, 'image_data', 10)
        self.srv = self.create_service(Trigger, 'image_data_service', self.image_service)

        if self.camera == True:
            self.camera_publisher()
        else:
            self.type = "JPG"
            self.picture_publisher()
    
    def picture_publisher(self):
        for filename in os.listdir(self.download_path):
            img = cv2.imread(os.path.join(self.download_path, filename))
            if img is not None:
                self.index += 1
                self.frames.append(img)
                self.publish_image(img)
    
    def camera_publisher(self):
        init = sl.InitParameters()
        init.camera_resolution = sl.RESOLUTION.AUTO
        init.camera_fps = 30
        cam = sl.Camera()
        runtime = sl.RuntimeParameters()
        mat = sl.Mat()

        if not cam.is_opened():
            print("Opening ZED Camera ")
        status = cam.open(init)
        if status != sl.ERROR_CODE.SUCCESS:
            print(repr(status))
            exit()

        key = ''
        while key != 113:  # for 'q' key
            err = cam.grab(runtime)
            if err == sl.ERROR_CODE.SUCCESS:
                self.tic = time.perf_counter()
                cam.retrieve_image(mat, sl.VIEW.LEFT_UNRECTIFIED)
                self.index += 1
                self.frames.append(mat.get_data())
                cv2.imshow(f"ZED Camera Frame {self.index}", mat.get_data())
                self.publish_image(mat.get_data())
                key = cv2.waitKey(5)
            else:
                key = cv2.waitKey(5)
        self.done = True
        cv2.destroyAllWindows()
        cam.close()
        print("ZED Camera closed ")
    
    def image_service(self, request, response):
        req_frame = request.index
        image = self.frames[req_frame-1]
        try:
            image_msg = self.bridge.cv2_to_imgmsg(image, encoding=self.type)
        except CvBridgeError as e:
            self.get_logger().info(e)
            print(e)
            return
        
        response.success = True
        response.message = image_msg
        return response

    def publish_image(self, image):
        if self.done != True:
            header = Header()
            header.stamp = self.get_clock().now().to_msg()
            header.frame_id = str(self.index) 

            try:
                image_msg = self.bridge.cv2_to_imgmsg(image, encoding=self.type)
            except CvBridgeError as e:
                self.get_logger().info(e)
                print(e)
                
            image_msg.header = header
            image_msg.is_bigendian = 0 
            image_msg.step = image_msg.width * 3

            self.model_publisher.publish(image_msg)
            size = sys.getsizeof(image_msg)
            self.get_logger().info(f'Published image frame: {self.index} with message size {size} bytes')
            self.total_data += size
        else:
            self.off_publisher.publish("Done")
            self.display_metrics()
    
    def display_metrics(self):
        toc = time.perf_counter()
        bandwidth = self.total_data / (toc - self.tic)
        self.get_logger().info(f'Published {len(self.frames)} images in {toc - self.tic:0.4f} seconds with average network bandwidth of {round(bandwidth)} bytes per second')
        self.get_logger().info('Shutting down display node...')
        raise SystemExit

def main(args=None):
    rclpy.init(args=args)
    image_publisher = CameraNode()
    try:
        rclpy.spin(image_publisher)
    except SystemExit:
        rclpy.logging.get_logger("Quitting").info('Done')
    image_publisher.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()