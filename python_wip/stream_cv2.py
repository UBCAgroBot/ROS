import rclpy
from rclpy.node import Node
import pycuda.driver as cuda
import pycuda.autoinit
import cv2
import cv2.cuda as cv2_cuda

class PostprocessingNode(Node):
    def __init__(self):
        super().__init__('postprocessing_node')
        self.subscription = self.create_subscription(
            MemoryHandle,
            'inference_done',
            self.postprocess_callback,
            10
        )

        # Ensure same CUDA context or initialize a new one if needed
        self.cuda_driver_context = cuda.Device(0).make_context()

    def postprocess_callback(self, msg):
        # Get the IPC handle and shared image address
        ipc_handle_str = msg.ipc_handle
        ipc_handle = cuda.IPCMemoryHandle(ipc_handle_str)
        shared_image_address = msg.shared_image_address

        # Map the shared output tensor via CUDA IPC
        d_output = cuda.ipc_open_mem_handle(ipc_handle, pycuda.driver.mem_alloc(self.h_output.nbytes))

        # Access shared image directly from unified memory (no need to download)
        cv_cuda_image = cv2_cuda_GpuMat(480, 640, cv2.CV_8UC3)
        cv_cuda_image.upload(shared_image_address)

        # Example OpenCV CUDA operation: GaussianBlur
        blurred_image = cv2_cuda_image.gaussianBlur((5, 5), 0)

        # Postprocess the inference output and the blurred image
        cuda.memcpy_dtoh(self.h_output, d_output)
        self.stream.synchronize()

        output = np.copy(self.h_output)
        self.get_logger().info(f"Postprocessed tensor: {output}")

        # Clean up the IPC memory
        cuda.ipc_close_mem_handle(d_output)

# this uses unified memory...