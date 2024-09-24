import rclpy
from rclpy.node import Node
import pycuda.driver as cuda
import pycuda.autoinit
import cv2
import cv2.cuda as cv2_cuda

class PostprocessingNode(Node):
    def __init__(self):
        super().__init__('postprocessing_node')

        # Create CUDA context
        self.cuda_driver_context = cuda.Device(0).make_context()

        # Subscribe to inference_done topic to get IPC handles
        self.subscription = self.create_subscription(
            MemoryHandle,
            'inference_done',
            self.postprocess_callback,
            10
        )

    def postprocess_callback(self, msg):
        # Get the IPC handles for tensor and image
        tensor_ipc_handle_str = msg.tensor_ipc_handle
        image_ipc_handle_str = msg.image_ipc_handle

        # Open IPC memory handles for tensor and image
        tensor_ipc_handle = cuda.IPCMemoryHandle(tensor_ipc_handle_str)
        image_ipc_handle = cuda.IPCMemoryHandle(image_ipc_handle_str)

        d_output = cuda.ipc_open_mem_handle(tensor_ipc_handle, self.h_output.nbytes)
        d_image = cuda.ipc_open_mem_handle(image_ipc_handle, self.cv_image.nbytes)

        # Wrap the image GPU pointer into a GpuMat object for OpenCV CUDA operations
        cv_cuda_image = cv2_cuda_GpuMat(self.cv_image.shape[0], self.cv_image.shape[1], cv2.CV_8UC3)
        cv_cuda_image.upload(d_image)

        # Perform OpenCV CUDA operations on the image (e.g., GaussianBlur)
        blurred_image = cv2_cuda_image.gaussianBlur((5, 5), 0)

        # Retrieve inference result and postprocess
        cuda.memcpy_dtoh(self.h_output, d_output)
        self.stream.synchronize()

        output = np.copy(self.h_output)
        self.get_logger().info(f"Postprocessed tensor: {output}")

        # Clean up IPC memory handles
        cuda.ipc_close_mem_handle(d_output)
        cuda.ipc_close_mem_handle(d_image)
