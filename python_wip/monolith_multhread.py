import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
import cv2
import numpy as np
import pycuda.driver as cuda
from threading import Thread
from trt_inference import TRTInference  # Assume custom TensorRT wrapper

class BoundingBoxNode(Node):
    def __init__(self):
        super().__init__('bounding_box_node')
        
        # Create subscriber to ZED camera image topic
        self.image_subscriber = self.create_subscription(Image, '/zed/image_raw', self.image_callback, 10)
        
        # Load TensorRT engine
        self.trt_inference = TRTInference('model.engine')  # Custom class for TensorRT inference
        self.cuda_stream = cuda.Stream()  # Use an asynchronous CUDA stream

        # Initialize threading for preprocessing, inference, and post-processing
        self.inference_thread = None

    def image_callback(self, msg):
        # Convert ROS2 Image to a NumPy array (assume it's in BGR format)
        frame = self.ros2_image_to_numpy(msg)

        # Asynchronous preprocessing and inference using a separate thread
        if self.inference_thread is None or not self.inference_thread.is_alive():
            self.inference_thread = Thread(target=self.run_inference, args=(frame,))
            self.inference_thread.start()

    def run_inference(self, frame):
        # Preprocessing: Resize, normalize, convert to TensorRT input format
        preprocessed_frame = self.preprocess_image(frame)

        # Asynchronously copy input to the GPU
        self.trt_inference.set_input_async(preprocessed_frame, stream=self.cuda_stream)

        # Run asynchronous inference
        self.trt_inference.infer_async(stream=self.cuda_stream)

        # Asynchronously copy output from the GPU
        output = self.trt_inference.get_output_async(stream=self.cuda_stream)

        # Post-process bounding boxes and other outputs (e.g., NMS)
        bboxes = self.post_process(output)

        # Optionally publish bounding boxes or visualize results
        self.publish_bounding_boxes(bboxes)

    def preprocess_image(self, frame):
        """Perform preprocessing steps (resize, normalize, etc.) using CUDA or OpenCV."""
        # Example resizing and normalization using OpenCV (offloadable to GPU using Numba or custom CUDA kernels)
        resized_frame = cv2.resize(frame, (self.trt_inference.input_width, self.trt_inference.input_height))
        normalized_frame = resized_frame.astype(np.float32) / 255.0  # Simple normalization
        return normalized_frame

    def post_process(self, output):
        """Post-process the TensorRT output to extract bounding boxes."""
        # Implement bounding box extraction, confidence thresholding, and optional NMS
        bboxes = self.extract_bounding_boxes(output)
        return bboxes

    def publish_bounding_boxes(self, bboxes):
        """Convert bounding boxes to a suitable ROS2 message and publish (optional)."""
        # You can add the code to publish the bounding boxes to a ROS2 topic here
        pass

    def ros2_image_to_numpy(self, msg):
        """Convert ROS2 Image message to a NumPy array (assumed to be in BGR format)."""
        # Assuming the image is in BGR8 encoding
        img = np.frombuffer(msg.data, dtype=np.uint8).reshape(msg.height, msg.width, 3)
        return img

def main(args=None):
    rclpy.init(args=args)
    node = BoundingBoxNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
