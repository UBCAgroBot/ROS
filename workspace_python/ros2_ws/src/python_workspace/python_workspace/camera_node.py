import time, os
import cv2
import pyzed.sl as sl
import pycuda.driver as cuda
import cupy as cp
import numpy as np

import rclpy
from rclpy.time import Time
from rclpy.node import Node
from rclpy.executors import MultiThreadedExecutor
from sensor_msgs.msg import Image
from std_msgs.msg import Header, String
from cv_bridge import CvBridge

cuda.init()
device = cuda.Device(0)
cuda_driver_context = device.make_context()

class CameraNode(Node):
    def __init__(self):
        super().__init__('camera_node')

        # change defaults here
        self.declare_parameter('source_type', 'zed')  # static_image, video, zed
        self.declare_parameter('static_image_path', '/home/usr/Desktop/ROS/assets/IMG_1822_14.JPG')
        self.declare_parameter('video_path', '/home/usr/Desktop/ROS/assets/video.mp4')
        self.declare_parameter('loop', 0)  # 0 = don't loop, >0 = # of loops, -1 = loop forever
        self.declare_parameter('frame_rate', 30)  # Desired frame rate for publishing
        self.declare_parameter('model_dimensions', (448, 1024))
        self.declare_parameter('camera_side', 'left') # left, right
        self.declare_parameter('shift_constant', 1)
        self.declare_parameter('roi_dimensions', [0, 0, 100, 100])        
        # propagate fp16 option (.fp16 datatype for cupy)
        
        self.source_type = self.get_parameter('source_type').get_parameter_value().string_value
        self.static_image_path = self.get_parameter('static_image_path').get_parameter_value().string_value
        self.video_path = self.get_parameter('video_path').get_parameter_value().string_value
        self.loop = self.get_parameter('loop').get_parameter_value().integer_value
        self.frame_rate = self.get_parameter('frame_rate').get_parameter_value().integer_value
        self.dimensions = tuple(self.get_parameter('model_dimensions').get_parameter_value().integer_array_value)
        self.camera_side = self.get_parameter('camera_side').get_parameter_value().string_value
        self.shift_constant = self.get_parameter('shift_constant').get_parameter_value().integer_value
        self.roi_dimensions = self.get_parameter('roi_dimensions').get_parameter_value().integer_array_value

        self.pointer_publisher = self.create_publisher(String, 'preprocessing_done', 10)
        self.image_publisher = self.create_publisher(Image, 'image_data', 10)
        self.bridge = CvBridge()
        self.velocity = 0

        if self.source_type == 'static_image':
            self.publish_static_image()
        elif self.source_type == 'video':
            self.publish_video_frames()
        elif self.source_type == 'zed':
            self.publish_zed_frames()
        else:
            self.get_logger().error(f"Invalid source_type: {self.source_type}")
            return
    
    def publish_static_image(self):
        if not os.path.exists(self.static_image_path):
            self.get_logger().error(f"Static image not found at {self.static_image_path}")
            # raise FileNotFoundError(f"Static image not found at {self.static_image_path}")
            return
        
        images = []
        if os.path.isdir(self.static_image_path):
            for filename in os.listdir(self.static_image_path):
                if filename.endswith('.JPG') or filename.endswith('.png'):
                    images.append(os.path.join(self.static_image_path, filename))
        elif os.path.isfile(self.static_image_path):
            images.append(self.static_image_path)
        
        if len(images) == 0:
            self.get_logger().error(f"No images found at {self.static_image_path}")
            return

        loops = 0
        while rclpy.ok() and (self.loop == -1 or loops < self.loop):
            for filename in images:
                image = cv2.imread(filename, cv2.IMREAD_COLOR)
                if image is not None:
                    self.preprocess_image(image)
                    time.sleep(1.0 / self.frame_rate)
                else:
                    self.get_logger().error(f"Failed to read image: {filename}")
                    raise FileNotFoundError(f"Failed to read image: {filename}")
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
                self.preprocess_image(frame)
                time.sleep(1.0 / self.frame_rate) 
                
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
        
        if self.side == 'left':
            init.camera_linux_id = 0
        elif self.side == 'right':
            init.camera_linux_id = 1
            self.shift_constant *= -1
        
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
                if self.side == 'left':
                    zed.retrieve_image(image_zed, sl.VIEW.LEFT_UNRECTIFIED)
                else:
                    zed.retrieve_image(image_zed, sl.VIEW.RIGHT_UNRECTIFIED)
                zed.get_sensors_data(sensors_data, sl.TIME_REFERENCE.IMAGE)
                accel_data = sensors_data.get_imu_data().get_linear_acceleration()
                
                # Calculate velocity (v = u + at)
                current_time = time.time()
                delta_time = current_time - previous_time
                acceleration = np.array([accel_data[0], accel_data[1], accel_data[2]])
                velocity = previous_velocity + acceleration * delta_time
                self.velocity = velocity
                previous_velocity = velocity
                previous_time = current_time
                
                cv_image = image_zed.get_data()   
                self.preprocess_image(cv_image)
                time.sleep(1.0 / self.frame_rate)
                
            else:
                self.get_logger().error("Failed to grab ZED camera frame: {str(err)}")

        zed.close()
        print("ZED Camera closed")  
    
    def preprocess_image(self, image):
        tic = time.perf_counter_ns()
        
        roi_x, roi_y, roi_w, roi_h = self.roi_dimensions
        shifted_x = roi_x + abs(self.velocity[0]) * self.shift_constant
        
        gpu_image = cv2.cuda_GpuMat()
        gpu_image.upload(image)                
        
        gpu_image = cv2.cuda.cvtColor(gpu_image, cv2.COLOR_RGBA2RGB) # remove alpha channel
        gpu_image = gpu_image[roi_y:(roi_y+roi_h), shifted_x:(shifted_x+roi_w)] # crop the image to ROI
        gpu_image = cv2.cuda.resize(gpu_image, self.dimensions) # resize to model dimensions
        
        input_data = cp.asarray(gpu_image.download(), dtype=cp.float32)  # Now the image is on GPU memory as CuPy array
        input_data /= 255.0 # normalize to [0, 1]
        input_data = cp.transpose(input_data, (2, 0, 1)) # Transpose from HWC (height, width, channels) to CHW (channels, height, width)

        d_input_ptr = input_data.data.ptr  # Get device pointer of the CuPy array
        ipc_handle = cuda.mem_get_ipc_handle(d_input_ptr) # Create the IPC handle
    
        toc = time.perf_counter_ns()
        self.get_logger().info(f"Preprocessing: {(toc-tic)/1e6} ms")

        # Publish the IPC handle as a string (sending the handle reference as a string)
        ipc_handle_msg = String()
        ipc_handle_msg.data = str(ipc_handle.handle)
        self.pointer_publisher.publish(ipc_handle_msg)
    
    def publish_image(self, image):
        header = Header()
        header.stamp = self.get_clock().now().to_msg()
        # header.frame_id = str(self.index)  # encode the transformation data into header
        image_msg = self.bridge.cv2_to_imgmsg(image, encoding="rgb8") 
        image_msg.header = header
        image_msg.is_bigendian = 0 
        image_msg.step = image_msg.width * 3
        self.publisher.publish(image_msg)

def main(args=None):
    rclpy.init(args=args)
    camera_node = CameraNode()
    executor = MultiThreadedExecutor(num_threads=1)
    executor.add_node(camera_node)
    
    try:
        executor.spin()
    finally:
        executor.shutdown()
        camera_node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()