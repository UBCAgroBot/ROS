import time, os
import cv2
import pycuda.driver as cuda
import cupy as cp
import queue

import rclpy
from rclpy.time import Time
from rclpy.node import Node
from rclpy.executors import MultiThreadedExecutor
from sensor_msgs.msg import Image
from std_msgs.msg import Header, String
from cv_bridge import CvBridge

class VideoNode(Node):
    def __init__(self):
        super().__init__('video_node')

        cuda.init()
        device = cuda.Device(0)
        self.cuda_driver_context = device.make_context()
        self.stream = cuda.Stream()

        self.declare_parameter('video_path', '/home/usr/Desktop/ROS/assets/video.mp4')
        self.declare_parameter('loop', 0)  # 0 = don't loop, >0 = # of loops, -1 = loop forever
        self.declare_parameter('frame_rate', 30)  # Desired frame rate for publishing
        self.declare_parameter('model_dimensions', (448, 1024))
        self.declare_parameter('shift_constant', 1)
        self.declare_parameter('roi_dimensions', [0, 0, 100, 100])  
        self.declare_paramter('precision', 'fp32')      

        self.video_path = self.get_parameter('video_path').get_parameter_value().string_value
        self.loop = self.get_parameter('loop').get_parameter_value().integer_value
        self.frame_rate = self.get_parameter('frame_rate').get_parameter_value().integer_value
        self.dimensions = tuple(self.get_parameter('model_dimensions').get_parameter_value().integer_array_value)
        self.shift_constant = self.get_parameter('shift_constant').get_parameter_value().integer_value
        self.roi_dimensions = self.get_parameter('roi_dimensions').get_parameter_value().integer_array_value
        self.precision = self.get_parameter('precision').get_parameter_value().string_value

        self.pointer_publisher = self.create_publisher(String, 'preprocessing_done', 10)
        self.image_service = self.create_service(Image, 'image_data', self.image_callback)

        self.velocity = [0, 0, 0]
        self.image_queue = queue.Queue()
        self.bridge = CvBridge()
        self.publish_video_frames()
        # propagate fp16 option (.fp16 datatype for cupy)
        if self.precision == 'fp32':
            pass
        elif self.precision == 'fp16':
            pass
        else:
            self.get_logger().error(f"Invalid precision: {self.precision}")

    def image_callback(self, response):
        self.get_logger().info("Received image request")
        if not self.image_queue.empty():
            image_data = self.image_queue.get()  # Get the image from the queue
            cv_image, velocity = image_data  # unpack tuple (image, velocity)

            # Convert OpenCV image to ROS2 Image message using cv_bridge
            ros_image = self.bridge.cv2_to_imgmsg(cv_image, encoding='rgb8')
            # Create a new header and include velocity in the stamp fields
            header = Header()
            current_time = self.get_clock().now().to_msg()
            header.stamp = current_time  # Set timestamp to current time
            header.frame_id = str(velocity)  # Set frame ID to velocity

            ros_image.header = header  # Attach the header to the ROS image message
            response.image = ros_image  # Set the response's image
            response.image = ros_image  # Set the response's image
            return response

        else:
            self.get_logger().error("Image queue is empty")
            return response

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
                self.image_queue.put((frame, self.velocity[0]))
                self.preprocess_image(frame)
                time.sleep(1.0 / self.frame_rate) 

            if self.loop > 0:
                loops += 1

            if self.loop != -1:
                cap.set(cv2.CAP_PROP_POS_FRAMES, 0) # restart video
        cap.release()

    def preprocess_image(self, image):
        tic = time.perf_counter_ns()

        roi_x, roi_y, roi_w, roi_h = self.roi_dimensions
        shifted_x = roi_x + abs(self.velocity[0]) * self.shift_constant

        gpu_image = cv2.cuda_GpuMat()
        gpu_image.upload(image)                

        gpu_image = cv2.cuda.cvtColor(gpu_image, cv2.COLOR_RGBA2RGB) # remove alpha channel
        gpu_image = gpu_image[roi_y:(roi_y+roi_h), shifted_x:(shifted_x+roi_w)] # crop the image to ROI
        gpu_image = cv2.cuda.resize(gpu_image, self.dimensions) # resize to model dimensions

        input_data = cp.asarray(gpu_image)  # Now the image is on GPU memory as CuPy array
        input_data = input_data.astype(cp.float32) / 255.0 # normalize to [0, 1]
        input_data = cp.transpose(input_data, (2, 0, 1)) # Transpose from HWC (height, width, channels) to CHW (channels, height, width)
        input_data = cp.ravel(input_data) # flatten the array

        d_input_ptr = input_data.data.ptr  # Get device pointer of the CuPy array
        ipc_handle = cuda.mem_get_ipc_handle(d_input_ptr) # Create the IPC handle

        toc = time.perf_counter_ns()
        self.get_logger().info(f"Preprocessing: {(toc-tic)/1e6} ms")

        # Publish the IPC handle as a string (sending the handle reference as a string)
        ipc_handle_msg = String()
        ipc_handle_msg.data = str(ipc_handle.handle)
        self.pointer_publisher.publish(ipc_handle_msg)

def main(args=None):
    rclpy.init(args=args)
    video_node = VideoNode()
    executor = MultiThreadedExecutor(num_threads=1)
    executor.add_node(video_node)

    try:
        executor.spin()
    finally:
        executor.shutdown()
        video_node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()