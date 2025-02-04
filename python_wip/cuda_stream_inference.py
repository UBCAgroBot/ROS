# inference_node.py
import rclpy
from rclpy.node import Node
from cuda_manager import CudaStreamManager
import pycuda.driver as cuda
import numpy as np

class InferenceNode(Node):
    def __init__(self, cuda_manager):
        super().__init__('inference_node')
        self.cuda_manager = cuda_manager

    def infer(self):
        self.get_logger().info("Waiting for preprocessing to complete...")
        self.cuda_manager.get_preprocess_event().synchronize()
        self.get_logger().info("Starting inference on GPU...")

        # Simulate inference on GPU
        data = np.random.randn(1024, 1024).astype(np.float32)
        gpu_data = cuda.mem_alloc(data.nbytes)
        cuda.memcpy_htod_async(gpu_data, data, self.cuda_manager.get_stream())
        
        # Signal inference completion
        self.cuda_manager.get_inference_event().record(self.cuda_manager.get_stream())
        self.get_logger().info("Inference complete.")

# post processing:
# postprocessing_node.py
import rclpy
from rclpy.node import Node
from cuda_manager import CudaStreamManager
import pycuda.driver as cuda
import numpy as np

class PostprocessingNode(Node):
    def __init__(self, cuda_manager):
        super().__init__('postprocessing_node')
        self.cuda_manager = cuda_manager

    def postprocess(self):
        self.get_logger().info("Waiting for inference to complete...")
        self.cuda_manager.get_inference_event().synchronize()
        self.get_logger().info("Starting postprocessing on GPU...")

        # Simulate postprocessing on GPU
        data = np.random.randn(1024, 1024).astype(np.float32)
        gpu_data = cuda.mem_alloc(data.nbytes)
        cuda.memcpy_htod_async(gpu_data, data, self.cuda_manager.get_stream())

        # Assume postprocessing is complete
        self.get_logger().info("Postprocessing complete.")
