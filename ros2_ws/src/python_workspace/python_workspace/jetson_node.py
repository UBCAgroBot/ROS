import time, sys, os
import cv2
import numpy as np

import tensorrt as trt
import pycuda.driver as cuda
# import pycuda.autoinit
cuda.init()
device = cuda.Device(0)
cuda_driver_context = device.make_context()

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
        self.declare_parameter('engine_path', '/home/user/Downloads/model.engine')
        self.engine_path = self.get_parameter('engine_path').get_parameter_value().string_value
        
        self.camera_subscriber = self.create_subscription(Image, 'image_data', self.image_callback, 10)
        self.bbox_publisher = self.create_publisher(String, 'bounding_boxes', 10)
        self.bridge = CvBridge()
        
        self.engine = self.load_engine()
        self.allocate_buffers()
        self.exec_context = (self.engine).create_execution_context()
        self.arrival_time = 0
        
    def load_engine(self):
        if not os.path.exists(self.engine_path):
            self.get_logger().error(f"Engine file not found at {self.engine_path}")
            return None

        TRT_LOGGER = trt.Logger(trt.Logger.WARNING)
        with open(self.engine_path, "rb") as f, trt.Runtime(TRT_LOGGER) as runtime:
            engine = runtime.deserialize_cuda_engine(f.read())
        self.get_logger().info(f"Successfully loaded engine from {self.engine_path}")
        return engine
    
    # if strip_weights:  
    #     engine = load_stripped_engine_and_refit(engine_path, path_onnx_model)  
    # else:  
    #     engine = load_normal_engine(engine_path)
    # def load_stripped_engine_and_refit(engine_path, onnx_model_path):
    #     runtime = trt.Runtime(TRT_LOGGER)
    #     with open(engine_path, "rb") as engine_file:
    #         engine = runtime.deserialize_cuda_engine(engine_file.read())
    #         refitter = trt.Refitter(engine, TRT_LOGGER)
    #         parser_refitter = trt.OnnxParserRefitter(refitter, TRT_LOGGER)
    #         assert parser_refitter.refit_from_file(onnx_model_path)
    #         assert refitter.refit_cuda_engine()
    #         return engine
    # def load_normal_engine(engine_path):
    #     runtime = trt.Runtime(TRT_LOGGER)
    #     with open(engine_path, "rb") as plan:
    #         engine = runtime.deserialize_cuda_engine(plan.read())
    #         return engine

    def allocate_buffers(self):
        engine = self.engine
        # Allocate host (pinned) and device memory for input/output
        # self.input_binding_idx = engine.get_binding_index("input_0")
        # self.output_binding_idx = engine.get_binding_index("output_0")

        self.input_shape = engine.get_binding_shape(0) # self.input_binding_idx
        self.output_shape = engine.get_binding_shape(1) # self.output_binding_idx

        # Allocate device memory for input/output
        self.d_input = cuda.mem_alloc(trt.volume(self.input_shape) * np.dtype(np.float32).itemsize)
        self.d_output = cuda.mem_alloc(trt.volume(self.output_shape) * np.dtype(np.float32).itemsize)

        # Allocate host pinned memory for input/output
        self.h_input = cuda.pagelocked_empty(trt.volume(self.input_shape), dtype=np.float32)
        self.h_output = cuda.pagelocked_empty(trt.volume(self.output_shape), dtype=np.float32)

        # Create a CUDA stream for async execution
        self.stream = cuda.Stream()
    
    def image_callback(self, msg):
        self.arrival_time = Time.from_msg(msg.header.stamp)
        latency = self.get_clock().now() - Time.from_msg(msg.header.stamp)
        self.get_logger().info(f"Latency: {latency.nanoseconds / 1e6} milliseconds")
        
        try:
            image  = self.bridge.imgmsg_to_cv2(msg, desired_encoding="rgb8")
        except CvBridgeError as e:
            print(e)
        
        self.preprocess(image)
    
    def preprocess(self, image):
        tic = time.perf_counter_ns()
        # Preprocess the image (e.g. normalize)
        input_data = image.astype(np.float32)
        input_data = np.transpose(input_data, (2, 0, 1))  # HWC to CHW
        input_data = np.expand_dims(input_data, axis=0)  # add batch dimension
        # Copy input data to pinned memory (host side)
        np.copyto(self.h_input, input_data.ravel())
        toc = time.perf_counter_ns()
        self.get_logger().info(f"Preprocessing: {(toc-tic)/1e6} ")
        self.run_inference()
    
    def run_inference(self):
        tic = time.perf_counter_ns()
        cuda_driver_context.push()
        # Transfer data from host to device asynchronously
        cuda.memcpy_htod_async(self.d_input, self.h_input, self.stream)
        # Execute inference asynchronously
        self.exec_context.execute_async_v2(bindings=[int(self.d_input), int(self.d_output)], stream_handle=self.stream.handle)
        # Transfer output data from device to host asynchronously
        cuda.memcpy_dtoh_async(self.h_output, self.d_output, self.stream)
        # Synchronize the stream to ensure the transfers are completed
        self.stream.synchronize()
        # Return the output from host memory
        output = np.copy(np.array(self.h_output))
        cuda_driver_context.pop()
        toc = time.perf_counter_ns()
        self.get_logger().info(f"Execution time: {(toc-tic)/1e6}")
        self.postprocess(output)

    def postprocess(self, output):
        output = output.reshape(-1, 7) # Reshape the output tensor to a 2D array
        # self.get_logger().info(f'Detected bounding boxes: {output}')
        
        msg_list = []
        for detection in output:
            x_min, y_min, width, height = detection[3:7]  # Assume these are the bbox coordinates or [:5]
            # confidence = detection[2]
            # msg = ",".join([str(x_min), str(y_min), str(width), str(height)])
            # msg_list.append(msg)
            bbox_msg = String()
            # bbox_msg.data = ";".join(msg_list)
            bbox_msg.data = "[x, y, w, h]"
        self.bbox_publisher.publish(bbox_msg)
        latency = (self.get_clock().now() - self.arrival_time) 
        self.get_logger().info(f"End to end: {latency.nanoseconds / 1e6} milliseconds")

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