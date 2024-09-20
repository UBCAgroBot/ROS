import rclpy
from rclpy.callback_groups import ReentrantCallbackGroup
from rclpy.executors import MultiThreadedExecutor
from rclpy.node import Node
from sensor_msgs.msg import Image
import cv2
import numpy as np
import asyncio  # Import asyncio for async handling
from trt_inference import TRTInference  # TensorRT inference class
import threading  # For concurrency control
import pycuda.driver as cuda  # Assuming CUDA is required for inference

class BoundingBoxNode(Node):
    def __init__(self):
        super().__init__('bounding_box_node')

        # Create reentrant callback groups for left and right image callbacks
        self.callback_group_left = ReentrantCallbackGroup()
        self.callback_group_right = ReentrantCallbackGroup()

        # Subscribers for left and right camera image topics, each with a different callback group
        self.left_image_subscriber = self.create_subscription(
            Image, '/camera/left/image_raw', self.left_image_callback, 10, callback_group=self.callback_group_left)

        self.right_image_subscriber = self.create_subscription(
            Image, '/camera/right/image_raw', self.right_image_callback, 10, callback_group=self.callback_group_right)

        # Initialize TensorRT inference engine
        self.trt_inference = TRTInference('model.engine')
        self.cuda_stream = cuda.Stream()

        # Buffer for left and right images (shared resource)
        self.left_image = None
        self.right_image = None

        # Add a lock to avoid race conditions on image access
        self.lock = threading.Lock()

    def left_image_callback(self, msg):
        frame = self.ros2_image_to_numpy(msg)
        with self.lock:
            self.left_image = frame

        # If both left and right images are available, start async inference
        if self.right_image is not None:
            asyncio.create_task(self.run_inference())

    def right_image_callback(self, msg):
        frame = self.ros2_image_to_numpy(msg)
        with self.lock:
            self.right_image = frame

        # If both left and right images are available, start async inference
        if self.left_image is not None:
            asyncio.create_task(self.run_inference())

    async def run_inference(self):
        # Lock the shared resource (images)
        with self.lock:
            # Batch the images for inference
            left_img = self.left_image
            right_img = self.right_image

            # Reset images to avoid reprocessing the same images
            self.left_image = None
            self.right_image = None

        # Run inference asynchronously
        if left_img is not None and right_img is not None:
            await self.async_inference(left_img, right_img)

    async def async_inference(self, left_img, right_img):
        # This method would use the CUDA stream to run TensorRT inference asynchronously.
        # Actual TensorRT inference code goes here...
        # Ensure that TensorRT uses the self.cuda_stream for asynchronous execution.
        ...

    def ros2_image_to_numpy(self, msg):
        img = np.frombuffer(msg.data, dtype=np.uint8).reshape(msg.height, msg.width, 3)
        return img

def main(args=None):
    rclpy.init(args=args)

    node = BoundingBoxNode()

    # Use a MultiThreadedExecutor with 2 threads
    executor = MultiThreadedExecutor(num_threads=2)
    executor.add_node(node)

    try:
        executor.spin()
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
