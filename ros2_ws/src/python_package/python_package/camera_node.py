import time
import cv2
import pyzed.sl as sl

import rclpy
from rclpy.time import Time
from rclpy.node import Node
from rclpy.executors import MultiThreadedExecutor
from cv_bridge import CvBridge

from std_msgs.msg import Header
from sensor_msgs.msg import Image
from custom_interfaces.msg import ImageInput

from .scripts.utils import ModelInference
from .scripts.image_source import ImageSource

class CameraNode(Node):
    def __init__(self):
        super().__init__('camera_node')

        self.get_logger().info("Initializing Camera Node")
        
        self.declare_parameter('camera_side', 'left')
        self.declare_parameter('roi_dimensions', (0, 0, 100, 100))  
        self.declare_parameter('source_type', 'zed')
        self.camera_side = self.get_parameter('camera_side').get_parameter_value().string_value
        self.roi_dimensions = self.get_parameter('roi_dimensions').get_parameter_value().integer_array_value
        self.source_type = self.get_parameter('source_type').get_parameter_value().string_value
        
        # path = ""
        
        # # initialize image_source type
        # image = ImageSource(self.source_type, path)

        self.shift_constant = 0
        self.frame_rate = 1
        self.model_dimensions = (640, 640)
        self.velocity = 0.0
        self.bridge = CvBridge()
        self.left_camera_serial_number = 26853647
        self.right_camera_serial_number = 0

        self.input_image_publisher = self.create_publisher(ImageInput, f'{self.camera_side}_image_input', 10)
        
        self.publish_zed_frames()
    
    def publish_zed_frames(self):
        zed = sl.Camera()
        init = sl.InitParameters()
        init.camera_resolution = sl.RESOLUTION.HD1080
        init.camera_fps = self.frame_rate
        
        if self.camera_side == 'left':
            init.set_from_serial_number(self.left_camera_serial_number)
        elif self.camera_side == 'right':
            init.set_from_serial_number(self.right_camera_serial_number)
            self.shift_constant *= -1
        
        if not zed.is_opened():
            self.get_logger().info("Initializing Zed Camera")
        status = zed.open(init)
        if status != sl.ERROR_CODE.SUCCESS:
            self.get_logger().error(f"Failed to open ZED camera: {str(status)}")
            return
        
        runtime = sl.RuntimeParameters()
        image_zed = sl.Mat()
        
        while rclpy.ok():
            err = zed.grab(runtime)
            if err == sl.ERROR_CODE.SUCCESS:
                zed.retrieve_image(image_zed, sl.VIEW.LEFT_UNRECTIFIED)
                if self.camera_side == 'left':
                    zed.retrieve_image(image_zed, sl.VIEW.LEFT_UNRECTIFIED)
                else:
                    zed.retrieve_image(image_zed, sl.VIEW.RIGHT_UNRECTIFIED)
                
                cv_image = image_zed.get_data()  
                self.preprocess_image(cv_image)
                time.sleep(1.0 / self.frame_rate)
                
            else:
                self.get_logger().error("Failed to grab ZED camera frame: {str(err)}")

        zed.close()
        self.get_logger().info("Closing Zed Camera")
    
    def preprocess_image(self, image):
        tic = time.perf_counter_ns()
        roi_x1, roi_y1, roi_x2, roi_y2 = self.roi_dimensions
        shifted_x1 = roi_x1 + self.shift_constant
        shifted_x2 = roi_x2 + self.shift_constant

        preprocessed_img = image[int(roi_y1):int(roi_y2), int(shifted_x1):int(shifted_x2), :3] # replace w/ util
        preprocessed_img = cv2.resize(preprocessed_img, self.model_dimensions) # check if necessary?
        preprocessed_img_msg = self.bridge.cv2_to_imgmsg(preprocessed_img, encoding='rgb8')

        raw_image = image[:, :, :3]
        raw_img_msg = self.bridge.cv2_to_imgmsg(raw_image, encoding='rgb8')
        
        image_input = ImageInput()
        image_input.header = Header()
        image_input.header.frame_id = 'static_image' # fix
        image_input.raw_image = raw_img_msg
        image_input.preprocessed_image = preprocessed_img_msg
        image_input.velocity = self.velocity
        self.input_image_publisher.publish(image_input)
        toc = time.perf_counter_ns()
        self.get_logger().info(f"Preprocessing: {(toc-tic)/1e6} ms")
        time.sleep(1/self.frame_rate)

def main(args=None):
    rclpy.init(args=args)
    camera_node = CameraNode()
    executor = MultiThreadedExecutor(num_threads=1)
    executor.add_node(camera_node)
    
    try:
        executor.spin()
    except KeyboardInterrupt:
        print("Shutting down camera node")
    finally:
        executor.shutdown()
        camera_node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
