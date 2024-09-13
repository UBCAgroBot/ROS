import time, os, sys
import numpy as np
import cv2
import tensorrt as trt
import pycuda.driver as cuda

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

class JetsonNode(Node):
    def __init__(self):
        super().__init__('jetson_node')
        
        self.declare_parameter('engine_path', '/home/user/Downloads/model.engine')
        self.declare_parameter('strip_weights', 'False')
        self.declare_parameter('model_path', '/home/user/Downloads/model.onnx')
        
        self.model_path = self.get_parameter('model_path').get_parameter_value().string_value
        self.engine_path = self.get_parameter('engine_path').get_parameter_value().string_value
        self.strip_weights = self.get_parameter('strip_weights').get_parameter_value().bool_value
        
        self.camera_subscriber = self.create_subscription(Image, 'image_data', self.image_callback, 10)
        self.bbox_publisher = self.create_publisher(String, 'bounding_boxes', 10)
        self.bridge = CvBridge()
        self.arrival_time, self.image = 0, None
        
        if self.strip_weights:  
            self.engine = self.load_stripped_engine_and_refit()  
        else:  
            self.engine = self.load_normal_engine()
        
        self.allocate_buffers()
        self.exec_context = (self.engine).create_execution_context()

    def load_stripped_engine_and_refit(self):
        if not os.path.exists(self.engine_path):
            self.get_logger().error(f"Engine file not found at {self.engine_path}")
            return None
        
        if not os.path.exists(self.model_path):
            self.get_logger().error(f"Model file not found at {self.model_path}")
            return None
        
        TRT_LOGGER = trt.Logger(trt.Logger.WARNING)
        with open(self.engine_path, "rb") as f, trt.Runtime(TRT_LOGGER) as runtime:
            engine = runtime.deserialize_cuda_engine(f.read())
            refitter = trt.Refitter(engine, TRT_LOGGER)
            parser_refitter = trt.OnnxParserRefitter(refitter, TRT_LOGGER)
            assert parser_refitter.refit_from_file(self.model_path)
            assert refitter.refit_cuda_engine()
            return engine
    
    def load_normal_engine(self):
        if not os.path.exists(self.engine_path):
            self.get_logger().error(f"Engine file not found at {self.engine_path}")
            return None

        TRT_LOGGER = trt.Logger(trt.Logger.WARNING)
        with open(self.engine_path, "rb") as f, trt.Runtime(TRT_LOGGER) as runtime:
            engine = runtime.deserialize_cuda_engine(f.read())
        self.get_logger().info(f"Successfully loaded engine from {self.engine_path}")
        return engine

    # fixed allocation: does not account for multiple bindings/batch sizes (single input -> output tensor)
    def allocate_buffers(self):
        engine = self.engine

        self.input_shape = engine.get_binding_shape(0)
        self.output_shape = engine.get_binding_shape(1)

        # Allocate device memory for input/output
        self.d_input = cuda.mem_alloc(trt.volume(self.input_shape) * np.dtype(np.float32).itemsize)
        self.d_output = cuda.mem_alloc(trt.volume(self.output_shape) * np.dtype(np.float32).itemsize)

        # Allocate host pinned memory for input/output (pinned memory for input/output buffers)
        self.h_input = cuda.pagelocked_empty(trt.volume(self.input_shape), dtype=np.float32)
        self.h_output = cuda.pagelocked_empty(trt.volume(self.output_shape), dtype=np.float32)

        # Create a CUDA stream for async execution
        self.stream = cuda.Stream()
    
    def image_callback(self, msg):
        self.arrival_time = Time.from_msg(msg.header.stamp)
        image  = self.bridge.imgmsg_to_cv2(msg, desired_encoding="rgb8")
        self.image = image
        latency = self.get_clock().now() - Time.from_msg(msg.header.stamp)
        self.get_logger().info(f"Latency: {latency.nanoseconds / 1e6} ms")
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
        self.get_logger().info(f"Preprocessing: {(toc-tic)/1e6} ms")
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
        output = np.copy(self.h_output)
        cuda_driver_context.pop()
        toc = time.perf_counter_ns()
        self.get_logger().info(f"Inference: {(toc-tic)/1e6} ms")
        self.postprocess(output)
    
    # output shape: (1, 5, 8400)
    def postprocess(self, output):
        tic = time.perf_counter_ns()
        num_detections = len(output) // 5
        output = np.reshape(output, (num_detections, 5))
        
        width = 640
        height = 640
        conf_threshold = 0.9
        nms_threshold = 0.1
        boxes = []
        confidences = []
        
        for detection in output:
            # print(detection)
            obj_conf, x_center, y_center, bbox_width, bbox_height = detection[:]
            
            # Apply sigmoid to object confidence and class score
            obj_conf = 1 / (1 + np.exp(-obj_conf))  # Sigmoid for object confidence

            # Filter out weak predictions based on confidence threshold
            if obj_conf < conf_threshold:
                continue
            
            # Convert normalized values to absolute pixel values
            x_center_pixel = int(x_center)
            y_center_pixel = int(y_center)
            bbox_width_pixel = int(bbox_width)
            bbox_height_pixel = int(bbox_height)
            print(f"[{obj_conf}, {x_center_pixel}, {y_center_pixel}, {bbox_width_pixel}, {bbox_height_pixel} ]")
            
            # Calculate the top-left and bottom-right corners of the bounding box
            # top_left_x = int(x_center_pixel - bbox_width_pixel / 2)
            # top_left_y = int(y_center_pixel - bbox_height_pixel / 2)
            # bottom_right_x = int(x_center_pixel + bbox_width_pixel / 2)
            # bottom_right_y = int(y_center_pixel + bbox_height_pixel / 2)

            
            boxes.append([x_center_pixel, y_center_pixel, bbox_width_pixel, bbox_height_pixel])
            # confidences.append(confidence)
            # boxes.append([top_left_x, top_left_y, bottom_right_x, bottom_right_y])
            
            # Append the box, confidence, and class score
            # boxes.append([x_min, y_min, x_max, y_max])
            confidences.append(float(obj_conf))
        
        # # Apply Non-Maximum Suppression (NMS) to suppress overlapping boxes
        indices = cv2.dnn.NMSBoxes(boxes, confidences, conf_threshold, nms_threshold)

        final_boxes = []
        for i in indices:
            # Since indices is now likely a 1D list, no need for i[0]
            final_boxes.append([*boxes[i], confidences[i]])
        toc = time.perf_counter_ns()
        self.get_logger().info(f"Postprocessing: {(toc-tic)/1e6} ms")
        # self.display(final_boxes)
        # print(final_boxes)
        self.display(final_boxes)
    
    def display(self, final_boxes):
        image = self.image
        for box in final_boxes:
            x, y, w, h, confidence = box
            cv2.rectangle(image, (x, y), (x + w, y + h), (255, 0, 0), 2)
        cv2.imshow("Image", image)
        cv2.waitKey(0)

def main(args=None):
    rclpy.init(args=args)
    jetson_node = JetsonNode()
    executor = MultiThreadedExecutor()
    executor.add_node(jetson_node)
    executor.spin()
    jetson_node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()