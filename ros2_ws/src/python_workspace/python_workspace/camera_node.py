import time, sys, os
import cv2
import pyzed.sl as sl
import numpy as np

import rclpy
from rclpy.time import Time
from rclpy.node import Node
from rclpy.executors import MultiThreadedExecutor
from sensor_msgs.msg import Image
from std_msgs.msg import Header, String
from cv_bridge import CvBridge, CvBridgeError

class CameraNode(Node):
    def __init__(self):
        super().__init__('camera_node')
        
        # os.chdir(Path(__file__).parent)
        
        # Declare parameters for the image source type, file paths, looping, and frame rate
        self.declare_parameter('source_type', 'zed')  # static_image, video, zed
        self.declare_parameter('static_image_path', '.../assets/')
        self.declare_parameter('video_path', '.../assets/video.mp4')
        self.declare_parameter('loop', 0)  # 0 = don't loop, >0 = # of loops, -1 = loop forever
        self.declare_parameter('frame_rate', 30)  # Desired frame rate for publishing
        self.declare_parameter('model_type', 'maize')
        
        # Retrieve the parameters
        self.source_type = self.get_parameter('source_type').get_parameter_value().string_value
        self.static_image_path = self.get_parameter('static_image_path').get_parameter_value().string_value
        self.video_path = self.get_parameter('video_path').get_parameter_value().string_value
        self.loop = self.get_parameter('loop').get_parameter_value().integer_value
        self.frame_rate = self.get_parameter('frame_rate').get_parameter_value().integer_value
        
        # width is 448?
        if self.get_parameter('model_type').get_parameter_value().string_value == 'maize':
            self.dimensions = (1024, 448)
        elif self.get_parameter('model_type').get_parameter_value().string_value == 'weed':
            self.dimensions = (1024, 448)
        else:
            self.dimensions = (1024, 448)

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
        # Load and publish the static image
        if not os.path.exists(self.static_image_path):
            self.get_logger().error(f"Static image not found at {self.static_image_path}")
            return

        loops = 0
        while rclpy.ok() and (self.loop == -1 or loops < self.loop):
            for filename in os.listdir(self.static_image_path):
                if filename.endswith('.jpg') or filename.endswith('.png'):
                    image = cv2.imread(os.path.join(self.static_image_path, filename), cv2.IMREAD_COLOR)
                    if image is not None:
                        self.index += 1
                        self.publish_image(image)
                        
            if self.loop > 0:
                loops += 1
    
    def publish_video_frames(self): # replace with decord later
        # Capture and publish video frames using CUDA
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
                self.publish_image(frame)
                
            if self.loop > 0:
                loops += 1
            
            if self.loop != -1:
                cap.set(cv2.CAP_PROP_POS_FRAMES, 0) # restart video
            
        cap.release()
    
    def publish_zed_frames(self):
        # Create a ZED camera object
        zed = sl.Camera()
        
        # Set configuration parameters
        init = sl.InitParameters()
        init.camera_resolution = sl.RESOLUTION.HD1080 # HD720
        init.camera_fps = 30

        # Open the ZED camera
        if not zed.is_opened():
            print("Opening ZED Camera ")
        status = zed.open(init)
        if status != sl.ERROR_CODE.SUCCESS:
            self.get_logger().error(f"Failed to open ZED camera: {str(status)}")
            return
        
        # Set runtime parameters after opening the camera
        runtime = sl.RuntimeParameters()
        
        # Prepare new image size to retrieve half-resolution images
        image_size = zed.get_camera_information().camera_configuration.resolution
        # image_size.width = image_size.width /2 # can we set any arbitary resolution here, instead of resizing later?
        # image_size.height = image_size.height /2
        image_zed = sl.Mat(image_size.width, image_size.height, sl.MAT_TYPE.U8_C3)
        
        sensors_data = sl.SensorsData()
        previous_velocity = np.array([0.0, 0.0, 0.0])
        previous_time = time.time()
        
        while rclpy.ok():
            err = zed.grab(runtime)
            if err == sl.ERROR_CODE.SUCCESS:
                self.index += 1
                tic = time.perf_counter()
                zed.retrieve_image(image_zed, sl.VIEW.LEFT_UNRECTIFIED, sl.MEM.CPU, image_size)
                zed.get_sensors_data(sensors_data, sl.TIME_REFERENCE.IMAGE)
                accel_data = sensors_data.get_imu_data().get_linear_acceleration()
                
                # Calculate velocity (v = u + at)
                current_time = time.time()
                delta_time = current_time - previous_time
                acceleration = np.array([accel_data[0], accel_data[1], accel_data[2]])
                velocity = previous_velocity + acceleration * delta_time
                previous_velocity = velocity
                previous_time = current_time
                
                cv_image = image_zed.get_data()  
            
                # Upload the ZED image to CUDA
                gpu_image = cv2.cuda_GpuMat()
                gpu_image.upload(cv_image)
                
                # Transform to BGR8 format and resize using CUDA 
                cv2.cuda.cvtColor(gpu_image, cv2.COLOR_BGR2RGB)
                cv2.cuda.resize(gpu_image, self.dimensions)
                
                # Convert the image to float32
                image_gpu = image_gpu.transpose((2, 0, 1)).astype(np.float32)
                image_gpu = np.expand_dims(image_gpu, axis=0)
                
                # # Transpose the image
                # image_gpu = cv2.cuda.transpose(image_gpu)
                # # Add a new dimension to the image
                # image_gpu = cv2.cuda_GpuMat((1,) + image_gpu.size(), image_gpu.type())
                # cv2.cuda.copyMakeBorder(image_gpu, 0, 1, 0, 0, cv2.BORDER_CONSTANT, image_gpu)
                
                # Download the processed image from GPU to CPU memory
                image_bgr = gpu_image.download()
                toc = time.perf_counter_ns()
                # self.preprocessing_time = (toc - tic)/1e6
                self.publish_image(image_bgr)
            else:
                self.get_logger().error("Failed to grab ZED camera frame: {str(err)}")

        zed.close()
        print("ZED Camera closed")        

    def publish_image(self, image):
        header = Header()
        header.stamp = self.get_clock().now().to_msg() # maybe use ros time
        header.frame_id = str(self.index) 
        # header.side = 'left' if self.index % 2 == 0 else 'right'
        
        if self.source_type != 'zed':
            # Use CUDA for loading and processing the image
            gpu_image = cv2.cuda_GpuMat()
            gpu_image.upload(image)
            
            # Convert to BGR8 format and resize using CUDA
            cv2.cuda.cvtColor(image, cv2.COLOR_BGR2RGB)
            cv2.cuda.resize(gpu_image, self.dimensions)
            
            image = gpu_image.download() # does this update outside scope?

        try:
            image_msg = self.bridge.cv2_to_imgmsg(image, encoding='rgb8')
        except CvBridgeError as e:
            print(e)
            
        image_msg.header = header
        image_msg.is_bigendian = 0 
        image_msg.step = image_msg.width * 3

        self.model_publisher.publish(image_msg)
        size = sys.getsizeof(image_msg)
        self.get_logger().info(f'Published image frame: {self.index} with message size {size} bytes')
        time.sleep(1.0 / self.frame_rate) # delay to control framerate

    def display_metrics(self):
        toc = time.perf_counter()
        bandwidth = self.total_data / (toc - self.tic)
        self.get_logger().info(f'Published {len(self.frames)} images in {toc - self.tic:0.4f} seconds with average network bandwidth of {round(bandwidth)} bytes per second')
        self.get_logger().info('Shutting down display node...')
        raise SystemExit

def main(args=None):
    rclpy.init(args=args)
    camera_node = CameraNode()
    # try:
    #     rclpy.spin(camera_node)
    # except KeyboardInterrupt:
    #     print("works")
    #     rclpy.logging.get_logger("Quitting").info('Done')
    #     camera_node.display_metrics()
    # except SystemExit:
    #     print("works")
    #     camera_node.display_metrics()
    #     rclpy.logging.get_logger("Quitting").info('Done')
    executor = MultiThreadedExecutor()
    executor.add_node(camera_node)
    executor.spin()
    camera_node.display_metrics()
    camera_node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()