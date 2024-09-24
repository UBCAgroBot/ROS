import time, os
import tensorrt as trt
import pycuda.driver as cuda
import cupy as cp
import numpy as np

import rclpy
from rclpy.time import Time
from rclpy.node import Node
from rclpy.components import NodeComponent
from rclpy.executors import MultiThreadedExecutor
from std_msgs.msg import Header, String

class JetsonNode(Node):
    def __init__(self):
        super().__init__('jetson_node')   
        
        cuda.init()
        device = cuda.Device(0)
        self.cuda_driver_context = device.make_context()
        
        # self.cuda_stream = None # ensure that each thread has own cuda context
        
        self.declare_parameter('engine_path', '/home/user/Downloads/model.engine')
        self.declare_parameter('strip_weights', 'False')
        self.declare_parameter('precision', 'fp32') # fp32, fp16
        
        self.engine_path = self.get_parameter('engine_path').get_parameter_value().string_value
        self.strip_weights = self.get_parameter('strip_weights').get_parameter_value().bool_value
        self.precision = self.get_parameter('precision').get_parameter_value().string_value
        
        self.engine, self.context = self.load_normal_engine()
        
        # could replace trt.volume with cp.prod # d_input = cuda.mem_alloc(input_size * np.float32().nbytes)
        self.stream = cuda.Stream()
        self.input_shape = (self.engine).get_binding_shape(0)
        self.output_shape = (self.engine).get_binding_shape(1)
        self.d_input = cuda.mem_alloc(trt.volume(self.input_shape) * cp.dtype(cp.float32).itemsize) # change to fp16, etc.
        self.d_output = cuda.mem_alloc(trt.volume(self.output_shape) * cp.dtype(cp.float32).itemsize) 
        
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
    
    def warmup(self):
        input_shape = self.input_shape

        for _ in range(20):
            random_input = np.random.randn(*input_shape).astype(np.float32)
            cuda.memcpy_htod(self.d_input, random_input)
            self.context.execute(bindings=[int(self.d_input), int(self.d_output)])

        self.get_logger().info(f"Engine warmed up with 20 inference passes.")
    
    def pointer_callback(self, ipc_handle_msg):
        
        # # future multi-threading implementation:
        # if self.cuda_stream is None:
        #     self.cuda_stream = cuda.Stream()
        #     cuda.Context.push()
        
        # self.process_with_cuda()
        
        self.cuda_driver_context.push()
        self.get_logger().info(f"Received IPC handle: {ipc_handle_str}")
        # # Convert the string back to bytes for the IPC handle
        # ipc_handle_str = msg.data
        # ipc_handle_bytes = bytes(int(ipc_handle_str[i:i+2], 16) for i in range(0, len(ipc_handle_str), 2))

        # # Recreate the IPC handle using PyCUDA
        # ipc_handle = cuda.IPCHandle(ipc_handle_bytes)

        # # Map the IPC memory into the current process
        # d_input = cuda.IPCMemoryHandle(ipc_handle)

        # # Map the memory to the current context
        # self.d_input = d_input.open(cuda.Context.get_current())
        
        # Re-create the CuPy array from IPC handle
        ipc_handle_str = ipc_handle_msg.data
        ipc_handle = cuda.IPC_handle(ipc_handle_str)
        d_input_ptr = ipc_handle.open()  # Map the shared memory to GPU
        # Run inference on the received image tensor
        result = self.run_inference(d_input_ptr)
        # Publish inference results
        self.publish_inference_result(result)
        # Clean up IPC handle
        
        ipc_handle.close()
        self.cuda_driver_context.pop()

    def run_inference(self, d_input_ptr):
        tic = time.perf_counter_ns()
        # self.cuda_driver_context.push()
        
        cuda.memcpy_dtod_async(self.d_input, d_input_ptr, cp.prod(self.input_shape) * cp.dtype(cp.float32).itemsize, self.stream) # Copy input data to the allocated memory in TensorRT (from the IPC pointer)
        self.exec_context.execute_async_v2(bindings=[int(self.d_input), int(self.d_output)], stream_handle=self.stream.handle) # Execute inference asynchronously
        output = cp.empty(self.output_shape, dtype=cp.float32)
        cuda.memcpy_dtod_async(output.data, self.d_output, self.stream) # Copy output to variable
        self.stream.synchronize() 
        
        # self.cuda_driver_context.pop()
        toc = time.perf_counter_ns()
        
        self.get_logger().info(f"Inference: {(toc-tic)/1e6} ms")   
        print(f'Output: {output} \n Shape: {output.shape}')
        
        d_output_ptr = output.data.ptr
        ipc_handle = cuda.mem_get_ipc_handle(d_output_ptr)
        ipc_handle_msg = String()
        ipc_handle_msg.data = str(ipc_handle.handle)
        self.pointer_publisher.publish(ipc_handle_msg)

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

def generate_node():
    return NodeComponent(JetsonNode, "jetson_node")

if __name__ == '__main__':
    main()