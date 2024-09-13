import time, sys, os
import cv2
import pyzed.sl as sl
import numpy as np

import rclpy
from rclpy.time import Time
from rclpy.node import Node
from rclpy.executors import MultiThreadedExecutor
from sensor_msgs.msg import Image
from std_msgs.msg import Header
from cv_bridge import CvBridge

class CameraNode(Node):
    def __init__(self):
        super().__init__('camera_node')

        self.declare_parameter('source_type', 'zed')  # static_image, video, zed
        self.declare_parameter('static_image_path', '.../assets/')
        self.declare_parameter('video_path', '.../assets/video.mp4')
        self.declare_parameter('loop', 0)  # 0 = don't loop, >0 = # of loops, -1 = loop forever
        self.declare_parameter('frame_rate', 30)  # Desired frame rate for publishing
        self.declare_parameter('model_dimensions', (448, 1024))
        # self.declare_parameter('camera_serial_number', 1101010101)
        
        self.source_type = self.get_parameter('source_type').get_parameter_value().string_value
        self.static_image_path = self.get_parameter('static_image_path').get_parameter_value().string_value
        self.video_path = self.get_parameter('video_path').get_parameter_value().string_value
        self.loop = self.get_parameter('loop').get_parameter_value().integer_value
        self.frame_rate = self.get_parameter('frame_rate').get_parameter_value().integer_value
        self.dimensions = tuple(self.get_parameter('model_dimensions').get_parameter_value().integer_array_value)
        # self.serial_number = self.get_parameter('camera_serial_number').get_parameter_value().integer_value

        self.publisher = self.create_publisher(Image, 'image_data', 10)
        self.bridge = CvBridge()
        self.index = 0

        if self.source_type == 'static_image':
            self.publish_static_image()
        elif self.source_type == 'video':
            self.publish_video_frames()
        elif self.source_type == 'zed':
            self.publish_zed_frames()
        else:
            self.get_logger().error(f"Invalid source_type: {self.source_type}")
    
    def publish_static_image(self):
        if not os.path.exists(self.static_image_path):
            self.get_logger().error(f"Static image not found at {self.static_image_path}")
            return

        loops = 0
        while rclpy.ok() and (self.loop == -1 or loops < self.loop):
            for filename in os.listdir(self.static_image_path):
                if filename.endswith('.JPG') or filename.endswith('.png'):
                    # print("found") !! log properly for vid too
                    image = cv2.imread(os.path.join(self.static_image_path, filename), cv2.IMREAD_COLOR)
                    if image is not None:
                        self.index += 1
                        image = cv2.resize(image, (self.dimensions))
                        self.publish_image(image)
                time.sleep(1.0 / self.frame_rate) # delay to control framerate
            if self.loop > 0:
                loops += 1
    
    def publish_video_frames(self):
        if not os.path.exists(self.video_path):
            self.get_logger().error(f"Video file not found at {self.video_path}")
            return

        cap = cv2.VideoCapture(self.video_path)
        if not cap.isOpened():
            self.get_logger().error(f"Failed to open video: {self.video_path}")
            return
        
        loops = 0
        while rclpy.ok() and (self.loop == -1 or loops < self.loop):
            while cap.isOpened() and rclpy.ok():
                ret, frame = cap.read()
                if not ret:
                    break
                self.index += 1
                self.publish_image(frame)
                time.sleep(1.0 / self.frame_rate) # delay to control framerate
                
            if self.loop > 0:
                loops += 1
            
            if self.loop != -1:
                cap.set(cv2.CAP_PROP_POS_FRAMES, 0) # restart video
            
        cap.release()
    
    def publish_zed_frames(self):
        zed = sl.Camera()
        init = sl.InitParameters()
        init.camera_resolution = sl.RESOLUTION.HD1080
        init.camera_fps = 30 # do we need publisher delay if this param is here?
        # init.set_from_serial_number(self.serial_number) # or give side and manually convert

        if not zed.is_opened():
            print("Opening ZED Camera ")
        status = zed.open(init)
        if status != sl.ERROR_CODE.SUCCESS:
            self.get_logger().error(f"Failed to open ZED camera: {str(status)}")
            return
        
        runtime = sl.RuntimeParameters()
        image_zed = sl.Mat()
        
        sensors_data = sl.SensorsData()
        previous_velocity = np.array([0.0, 0.0, 0.0])
        previous_time = time.time()
        
        while rclpy.ok():
            err = zed.grab(runtime)
            if err == sl.ERROR_CODE.SUCCESS:
                self.index += 1
                tic = time.perf_counter_ns()
                zed.retrieve_image(image_zed, sl.VIEW.LEFT_UNRECTIFIED) # modify based on left/right
                zed.get_sensors_data(sensors_data, sl.TIME_REFERENCE.IMAGE)
                accel_data = sensors_data.get_imu_data().get_linear_acceleration()
                
                # Calculate velocity (v = u + at)
                current_time = time.time()
                delta_time = current_time - previous_time
                acceleration = np.array([accel_data[0], accel_data[1], accel_data[2]])
                velocity = previous_velocity + acceleration * delta_time
                previous_velocity = velocity
                previous_time = current_time
                
                # Take and transform image using CUDA operations
                cv_image = image_zed.get_data()              
                gpu_image = cv2.cuda_GpuMat()
                gpu_image.upload(cv_image)                
                gpu_image = cv2.cuda.cvtColor(gpu_image, cv2.COLOR_RGBA2RGB)
                # crop goes here...
                gpu_image = cv2.cuda.resize(gpu_image, self.dimensions)
                
                # Download the processed image from GPU to CPU memory
                rgb_image = gpu_image.download()
                toc = time.perf_counter_ns()
                self.get_logger().info(f"Preprocessing: {(toc - tic)/1e6} ms")
                self.publish_image(rgb_image)
                time.sleep(1.0 / self.frame_rate) # delay to control framerate
            else:
                self.get_logger().error("Failed to grab ZED camera frame: {str(err)}")

        zed.close()
        print("ZED Camera closed")        

    def publish_image(self, image):
        header = Header()
        header.stamp = self.get_clock().now().to_msg()
        header.frame_id = str(self.index) 
        # try packing velcoity information into header
        image_msg = self.bridge.cv2_to_imgmsg(image, encoding="rgb8") 

        image_msg.header = header
        image_msg.is_bigendian = 0 
        image_msg.step = image_msg.width * 3

        self.publisher.publish(image_msg)

def main(args=None):
    rclpy.init(args=args)
    camera_node = CameraNode()
    executor = MultiThreadedExecutor()
    executor.add_node(camera_node)
    executor.spin()
    camera_node.display_metrics()
    camera_node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()