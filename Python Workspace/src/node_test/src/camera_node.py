import cv2
import os
from pathlib import Path
import pyzed.sl as sl
import psutil

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from std_msgs.msg import Header, String
from cv_bridge import CvBridge, CvBridgeError

class CameraNode(Node):
    def __init__(self):
        super().__init__('camera_node') #type: ignore
        self.bridge = CvBridge()
        self.download_path = str(Path.home() / 'Downloads')
        # replace self.camera with parameter
        self.index, self.camera, self.type = 0, True, "rgb8"
        self.model_publisher = self.create_publisher(Image, 'image_data', 10)

        if self.camera == True:
            self.camera_publisher()
        else:
            # self.type = "JPG"
            self.picture_publisher()
    
    def picture_publisher(self):
        for filename in os.listdir(self.download_path):
            img = cv2.imread(os.path.join(self.download_path, filename))
            if img is not None:
                self.index += 1
                self.publish_image(img)
    
    def camera_publisher(self):
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
        
        pre_mem = psutil.Process().memory_percent()
        key = ''
        while key != 113:  # for 'q' key
            err = cam.grab(runtime)
            if err == sl.ERROR_CODE.SUCCESS:
                cam.retrieve_image(mat, sl.VIEW.LEFT_UNRECTIFIED)
                self.index += 1
                image = mat.get_data()
                # Convert the image to RGB using CUDA
                image = cv2.cvtColor(image, cv2.COLOR_RGBA2RGB)

                cv2.imshow("zed", image)
                self.publish_image(image)
                post_mem = psutil.Process().memory_percent()
                print(f"Memory usage: {(post_mem - pre_mem) * 100:.2f}%")
                key = cv2.waitKey(5)
            else:
                key = cv2.waitKey(5)
        cam.close()
        print("ZED Camera closed")
        
        # raise SystemExit
        raise KeyboardInterrupt

    def publish_image(self, image):
        header = Header()
        header.stamp = self.get_clock().now().to_msg()
        header.frame_id = str(self.index) 

        try:
            image_msg = self.bridge.cv2_to_imgmsg(image, encoding=self.type)
        except CvBridgeError as e:
            print(e)
            
        image_msg.header = header
        image_msg.is_bigendian = 0 
        image_msg.step = image_msg.width * 3

        self.model_publisher.publish(image_msg)
        self.get_logger().info(f'Published: {self.index}')

def main(args=None):
    rclpy.init(args=args)
    camera_node = CameraNode()
    try:
        rclpy.spin(camera_node)
    except KeyboardInterrupt:
        print("works")
        rclpy.logging.get_logger("Quitting").info('Done')
        camera_node.display_metrics()
    except SystemExit:
        print("works")
        camera_node.display_metrics()
        rclpy.logging.get_logger("Quitting").info('Done')

    camera_node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()