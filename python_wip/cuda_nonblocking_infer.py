import rclpy
from rclpy.node import Node
import tensorrt as trt
import pycuda.driver as cuda
import numpy as np
import threading

class InferenceNode(Node):
    def __init__(self, preprocess_node, sync_event):
        super().__init__('inference_node')
        self.preprocess_node = preprocess_node
        self.sync_event = sync_event  # Synchronization event
        self.create_timer(0.1, self.run_inference)  # Periodically check for new images
        
        # Load the TensorRT engine
        self.engine = self.load_engine('model.trt')
        self.context = self.engine.create_execution_context()

        # Allocate device memory for input/output bindings
        self.input_shape = (1, 3, 224, 224)
        self.input_size = np.prod(self.input_shape) * np.dtype(np.float32).itemsize
        self.d_output = cuda.mem_alloc(self.input_size)  # assuming output size is the same as input

        # Create a CUDA stream for asynchronous inference execution
        self.inference_stream = cuda.Stream()

    def load_engine(self, engine_path):
        with open(engine_path, "rb") as f:
            runtime = trt.Runtime(trt.Logger(trt.Logger.WARNING))
            return runtime.deserialize_cuda_engine(f.read())

    def run_inference(self):
        # Wait until the preprocessing node signals that a new image is ready
        if not self.sync_event.is_set():
            return  # No image ready for inference yet

        # Reset the event so that the preprocessing node can signal for the next image
        self.sync_event.clear()

        # Get the CUDA memory pointer and stream from the preprocessing node
        d_preprocessed_image = self.preprocess_node.get_cuda_memory()
        preprocess_stream = self.preprocess_node.get_cuda_stream()

        # Run inference using TensorRT asynchronously in its own stream
        bindings = [int(d_preprocessed_image), int(self.d_output)]
        self.context.execute_async_v2(bindings, self.inference_stream.handle)

        # Synchronize the inference stream to ensure inference is complete before proceeding
        self.inference_stream.synchronize()

        # Assuming the result is in d_output, we could copy it back to host if needed
        result = np.empty(self.input_shape, dtype=np.float32)
        cuda.memcpy_dtoh_async(result, self.d_output, self.inference_stream)
        self.inference_stream.synchronize()  # Ensure the copy operation is done
        self.get_logger().info(f"Inference complete. Output shape: {result.shape}")
