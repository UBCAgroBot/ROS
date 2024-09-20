import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
import pycuda.driver as cuda
import numpy as np
import cv2
import threading

class ImagePreprocessingNode(Node):
    def __init__(self, sync_event):
        super().__init__('image_preprocessing_node')
        self.sync_event = sync_event  # Synchronization event to signal inference node
        self.width = 224
        self.height = 224
        self.channels = 3
        
        # Allocate CUDA memory for the image
        self.cuda_mem = cuda.mem_alloc(self.width * self.height * self.channels * np.dtype(np.float32).itemsize)
        
        # Create a subscription to image topic (e.g., from a camera)
        self.subscription = self.create_subscription(
            Image,
            '/camera/image',
            self.image_callback,
            10
        )

    def image_callback(self, msg):
        # Convert ROS Image message to a numpy array (assuming the image is encoded in RGB8)
        image = np.frombuffer(msg.data, dtype=np.uint8).reshape(msg.height, msg.width, self.channels)
        image = cv2.resize(image, (self.width, self.height)).astype(np.float32) / 255.0
        
        # Copy image from host (CPU) to device (GPU)
        cuda.memcpy_htod(self.cuda_mem, image)
        
        # Signal the inference node that preprocessing is done
        self.sync_event.set()
        self.get_logger().info("Image preprocessed and copied to CUDA memory")

    def get_cuda_memory(self):
        return self.cuda_mem
