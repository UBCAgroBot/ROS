import time, os
import tensorrt as trt
import pycuda.driver as cuda
import cupy as cp
import numpy as np

import rclpy
from rclpy.time import Time
from rclpy.node import Node
from std_msgs.msg import Header, String
from rclpy.executors import MultiThreadedExecutor

cuda.init()
device = cuda.Device(0)
cuda_driver_context = device.make_context()

class JetsonNode(Node):
    def __init__(self):
        super().__init__('jetson_node')   
        
        self.declare_parameter('engine_path', '/home/user/Downloads/model.engine')
        self.declare_parameter('strip_weights', 'False')
        self.declare_parameter('precision', 'fp32') # fp32, fp16, int8
        
        self.engine_path = self.get_parameter('engine_path').get_parameter_value().string_value
        self.strip_weights = self.get_parameter('strip_weights').get_parameter_value().bool_value
        self.precision = self.get_parameter('precision').get_parameter_value().string_value
        
        self.engine, self.context = self.load_normal_engine()
        self.stream = cuda.Stream()
        input_shape = (self.engine).get_binding_shape(0)
        output_shape = (self.engine).get_binding_shape(1)
        self.d_input = cuda.mem_alloc(trt.volume(input_shape) * np.dtype(np.float32).itemsize) # change to fp16, etc.
        self.d_output = cuda.mem_alloc(trt.volume(output_shape) * np.dtype(np.float32).itemsize) # Allocate device memory for input/output
        
        self.pointer_subscriber = self.create_publisher(String, 'preprocessing_done', self.pointer_callback, 10)
        self.pointer_publisher = self.create_publisher(String, 'inference_done', 10)
        self.arrival_time, self.type = 0, None, None
        self.warmup()
    
    def load_normal_engine(self):
        if not os.path.exists(self.engine_path):
            self.get_logger().error(f"Engine file not found at {self.engine_path}")
            return None

        TRT_LOGGER = trt.Logger(trt.Logger.WARNING)
        with open(self.engine_path, "rb") as f, trt.Runtime(TRT_LOGGER) as runtime:
            engine = runtime.deserialize_cuda_engine(f.read())
        context = engine.create_execution_context()
        self.get_logger().info(f"Successfully loaded engine from {self.engine_path}")
        return engine, context
    
    # after verifying this works pass in the already allocated memory in class instantiation
    def warmup(self):
        input_shape = self.context.get_binding_shape(0)
        input_size = np.prod(input_shape)
        output_shape = self.context.get_binding_shape(1)
        output_size = np.prod(output_shape)

        d_input = cuda.mem_alloc(input_size * np.float32().nbytes)
        d_output = cuda.mem_alloc(output_size * np.float32().nbytes)

        for _ in range(20):
            random_input = np.random.randn(*input_shape).astype(np.float32)
            cuda.memcpy_htod(d_input, random_input)
            self.context.execute(bindings=[int(d_input), int(d_output)])

        self.get_logger().info(f"Engine warmed up with 20 inference passes.")
    
    def pointer_callback(self, msg):
        # Convert the string back to bytes for the IPC handle
        ipc_handle_str = msg.data
        ipc_handle_bytes = bytes(int(ipc_handle_str[i:i+2], 16) for i in range(0, len(ipc_handle_str), 2))

        # Recreate the IPC handle using PyCUDA
        ipc_handle = cuda.IPCHandle(ipc_handle_bytes)

        # Map the IPC memory into the current process
        d_input = cuda.IPCMemoryHandle(ipc_handle)

        # Map the memory to the current context
        self.d_input = d_input.open(cuda.Context.get_current())
        self.get_logger().info(f"Received IPC handle: {ipc_handle_str}")

    def run_inference(self):
        tic = time.perf_counter_ns()
        cuda_driver_context.push()
        # Execute inference asynchronously
        self.exec_context.execute_async_v2(bindings=[int(self.d_input), int(self.d_output)], stream_handle=self.stream.handle)
        self.stream.synchronize() 
        cuda_driver_context.pop()
        toc = time.perf_counter_ns()
        self.get_logger().info(f"Inference: {(toc-tic)/1e6} ms")
        
        # # Assuming the result is in d_output, we could copy it back to host if needed
        # result = np.empty(self.input_shape, dtype=np.float32)
        # cuda.memcpy_dtoh(result, self.d_output)
        # self.get_logger().info(f"Inference complete. Output shape: {result.shape}")
        # Publish the IPC handles to postprocessing node

    def postprocess(self, output):
        tic = time.perf_counter_ns()
        num_detections = len(output) // 5
        output = np.reshape(output, (5, num_detections))
        # output = np.reshape(output, (num_detections, 5))
        
        width = 640
        height = 640
        conf_threshold = 0.9
        nms_threshold = 0.1
        boxes = []
        confidences = []
        
        # print("HERE!!!", output.shape)
        # print(output[:,:20])
        
        for i in range(output.shape[1]):
        # for detection in output:

            detection = output[..., i]
            
            # obj_conf, x_center, y_center, bbox_width, bbox_height = detection[:]
            x_center, y_center, bbox_width, bbox_height, obj_conf = detection[:]
            # x_min, 
            
            # Apply sigmoid to object confidence and class score
            # obj_conf = 1 / (1 + np.exp(-obj_conf))  # Sigmoid for object confidence

            # Filter out weak predictions based on confidence threshold
            if obj_conf < conf_threshold:
                continue
            print(detection)
            
            # Convert normalized values to absolute pixel values
            x_center_pixel = int(x_center)
            y_center_pixel = int(y_center)
            bbox_width_pixel = int(bbox_width)
            bbox_height_pixel = int(bbox_height)
            # print(f"[{obj_conf}, {x_center_pixel}, {y_center_pixel}, {bbox_width_pixel}, {bbox_height_pixel} ]")
            
            # Calculate the top-left and bottom-right corners of the bounding box
            top_left_x = int(x_center_pixel - bbox_width_pixel / 2)
            top_left_y = int(y_center_pixel - bbox_height_pixel / 2)
            bottom_right_x = int(x_center_pixel + bbox_width_pixel / 2)
            bottom_right_y = int(y_center_pixel + bbox_height_pixel / 2)

            
            # boxes.append([x_center_pixel, y_center_pixel, bbox_width_pixel, bbox_height_pixel])
            # confidences.append(confidence)
            boxes.append([top_left_x, top_left_y, bottom_right_x, bottom_right_y])
            
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
        self.display(final_boxes)
        # print(final_boxes)
        # self.display(boxes)
    
        # assign random tensor in cuda memory as dummy input for extermination node and try memory passing with the composition thing
    
    def display(self, final_boxes):
        image = self.image
        for box in final_boxes:
            x, y, w, h, confidence = box
            cv2.rectangle(image, (x+h, y+w), (x-w, y-h), (255, 0, 0), 2)
        cv2.imshow("Image", image)
        cv2.waitKey(0)

def main(args=None):
    rclpy.init(args=args)
    jetson_node = JetsonNode()
    executor = MultiThreadedExecutor(num_threads=1)
    executor.add_node(jetson_node)
    
    try:
        executor.spin()
    finally:
        executor.shutdown()
        jetson_node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()