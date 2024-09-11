import time, sys, os
import cv2
import numpy as np

import tensorrt as trt
import pycuda.autoinit
import pycuda.driver as cuda

import rclpy
from rclpy.time import Time
from rclpy.node import Node
from rclpy.executors import MultiThreadedExecutor
from sensor_msgs.msg import Image
from std_msgs.msg import Header, String
from cv_bridge import CvBridge, CvBridgeError

class JetsonNode(Node):
    def __init__(self):
        super().__init__('jetson_node')
        self.declare_parameter('engine_path', '.../assets/model.trt')
        self.engine_path = self.get_parameter('engine_path').get_parameter_value().string_value

        self.camera_subscriber = self.create_subscription(Image, 'image_data', self.image_callback, 10)
        self.bbox_publisher = self.create_publisher(String, 'bounding_boxes', 10)
        self.bridge = CvBridge()
        self.engine = self.load_engine(self.engine_path)
        self.context = self.engine.create_execution_context()
        self.allocate_buffers()
        self.tensorrt_init()
    
    def load_engine(self, engine_file_path):
        if not os.path.exists(engine_file_path):
            self.get_logger().error(f"Engine file not found at {engine_file_path}")
            return None

        TRT_LOGGER = trt.Logger(trt.Logger.WARNING)
        with open(engine_file_path, "rb") as f, trt.Runtime(TRT_LOGGER) as runtime:
            engine = runtime.deserialize_cuda_engine(f.read())
        self.get_logger().info(f"Loaded engine from {engine_file_path}")
        return engine

    def allocate_buffers(self):
        # Allocate host (pinned) and device memory for input/output
        self.input_binding_idx = self.engine.get_binding_index("input_0")
        self.output_binding_idx = self.engine.get_binding_index("output_0")

        self.input_shape = self.engine.get_binding_shape(self.input_binding_idx)
        self.output_shape = self.engine.get_binding_shape(self.output_binding_idx)

        # Allocate device memory for input/output
        self.d_input = cuda.mem_alloc(trt.volume(self.input_shape) * np.dtype(np.float32).itemsize)
        self.d_output = cuda.mem_alloc(trt.volume(self.output_shape) * np.dtype(np.float32).itemsize)

        # Allocate host pinned memory for input/output
        self.h_input = cuda.pagelocked_empty(trt.volume(self.input_shape), dtype=np.float32)
        self.h_output = cuda.pagelocked_empty(trt.volume(self.output_shape), dtype=np.float32)

        # Create a CUDA stream for async execution
        self.stream = cuda.Stream()
    
    def image_callback(self, msg):
        now = self.get_clock().now()
        self.get_logger().info(f"Received: {msg.header.frame_id}")
        latency = now - Time.from_msg(msg.header.stamp)
        print(f"Latency: {latency.nanoseconds / 1e6} milliseconds")
        
        try:
            image  = self.bridge.imgmsg_to_cv2(msg, desired_encoding='rgb8')
        except CvBridgeError as e:
            print(e)
        
        self.preprocess(image)
    
    def preprocess(self, image):
        # Preprocess the image (e.g. normalize)
        input_data = image.astype(np.float32)
        input_data = np.transpose(input_data, (2, 0, 1))  # HWC to CHW
        input_data = np.expand_dims(input_data, axis=0)  # add batch dimension
        # Copy input data to pinned memory (host side)
        np.copyto(self.h_input, input_data.ravel())
    
    def run_inference(self):
        # Transfer data from host to device asynchronously
        cuda.memcpy_htod_async(self.d_input, self.h_input, self.stream)
        # Execute inference asynchronously
        self.context.execute_async_v2(bindings=[int(self.d_input), int(self.d_output)], stream_handle=self.stream.handle)
        # Transfer output data from device to host asynchronously
        cuda.memcpy_dtoh_async(self.h_output, self.d_output, self.stream)
        # Synchronize the stream to ensure the transfers are completed
        self.stream.synchronize()
        # Return the output from host memory
        self.postprocess()
        
        # self.inferencing_time = (toc-tic)/1e6
        # self.get_logger().info(f"Execution time: {self.time} milliseconds")

    def postprocess(self):
        # Postprocess the output and extract bounding boxes
        output = np.array(self.h_output).reshape(-1, 7)  # Reshape the output tensor to a 2D array
        self.get_logger().info(f'Detected bounding boxes: {output}')
        
        msg_list = []
        for detection in output:
            x_min, y_min, width, height = detection[3:7]  # Assume these are the bbox coordinates or [:5]
            # confidence = detection[2]
            msg = String()
            msg.join([str(x_min), str(y_min), str(width), str(height)], ',')
            msg_list.boxes.append(msg)
        
        self.bbox_publisher.publish(msg_list.join(';'))

def main(args=None):
    rclpy.init(args=args)
    jetson_node = JetsonNode()
    # try:
    #     rclpy.spin(jetson_node)
    # except KeyboardInterrupt:
    #     print("qq...")
    #     jetson_node.display_metrics()
    #     rclpy.logging.get_logger("Quitting").info('Done')
    # except SystemExit:   
    #     print("qqq...")
    #     jetson_node.display_metrics()
    #     rclpy.logging.get_logger("Quitting").info('Done')
    executor = MultiThreadedExecutor()
    executor.add_node(jetson_node)
    executor.spin()
    jetson_node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()