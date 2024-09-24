import rclpy
from rclpy.node import Node
import pycuda.driver as cuda
import pycuda.autoinit
import numpy as np
import cv2
import cv2.cuda as cv2_cuda

class InferenceNode(Node):
    def __init__(self):
        super().__init__('inference_node')

        # Initialize CUDA context
        self.cuda_driver_context = cuda.Device(0).make_context()
        self.stream = cuda.Stream()

        # Allocate GPU memory for input and output tensors using cudaMalloc
        self.h_input = np.random.randn(1, 3, 224, 224).astype(np.float32)
        self.h_output = np.empty((1, 1000), dtype=np.float32)

        self.d_input = cuda.mem_alloc(self.h_input.nbytes)
        self.d_output = cuda.mem_alloc(self.h_output.nbytes)

        # Example image (allocate on GPU)
        self.cv_image = np.random.rand(480, 640, 3).astype(np.uint8)
        self.cv_cuda_image = cv2_cuda_GpuMat(self.cv_image.shape[0], self.cv_image.shape[1], cv2.CV_8UC3)

        # Upload image to GPU (device memory)
        self.cv_cuda_image.upload(self.cv_image)

        # Create CUDA IPC handle for output tensor and image
        self.output_ipc_handle = cuda.mem_get_ipc_handle(self.d_output)
        self.image_ipc_handle = cuda.mem_get_ipc_handle(self.cv_cuda_image.cudaPtr())

        # Publish the IPC handle to postprocessing node
        self.publisher_ = self.create_publisher(MemoryHandle, 'inference_done', 10)

    def run_inference(self):
        tic = time.perf_counter_ns()
        self.cuda_driver_context.push()

        # Transfer data to device asynchronously
        cuda.memcpy_htod_async(self.d_input, self.h_input, self.stream)

        # Execute inference asynchronously
        self.exec_context.execute_async_v2(bindings=[int(self.d_input), int(self.d_output)], stream_handle=self.stream.handle)
        self.stream.synchronize()

        self.cuda_driver_context.pop()
        toc = time.perf_counter_ns()

        self.get_logger().info(f"Inference done in: {(toc-tic)/1e6} ms")

        # Publish the IPC handles to postprocessing node
        msg = MemoryHandle()
        msg.tensor_ipc_handle = str(self.output_ipc_handle)
        msg.image_ipc_handle = str(self.image_ipc_handle)
        self.publisher_.publish(msg)
