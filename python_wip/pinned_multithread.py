import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
import cv2
import numpy as np
import pycuda.driver as cuda
import pycuda.autoinit
from threading import Thread
from trt_inference import TRTInference  # Custom TensorRT wrapper

class BoundingBoxNode(Node):
    def __init__(self):
        super().__init__('bounding_box_node')
        
        # Subscribers for left and right camera image topics
        self.left_image_subscriber = self.create_subscription(Image, '/camera/left/image_raw', self.left_image_callback, 10)
        self.right_image_subscriber = self.create_subscription(Image, '/camera/right/image_raw', self.right_image_callback, 10)
        
        # Initialize TensorRT inference engine
        self.trt_inference = TRTInference('model.engine')  # Custom class for TensorRT inference
        self.cuda_stream = cuda.Stream()  # Use an asynchronous CUDA stream

        # Buffer for left and right images
        self.left_image = None
        self.right_image = None

        # Allocate pinned memory (page-locked) for faster memory transfers
        self.host_input = cuda.pagelocked_empty(shape=(2, self.trt_inference.input_height, self.trt_inference.input_width, 3), dtype=np.float32)  # Batch of 2 images
        
        # Allocate memory for output
        self.host_output = cuda.pagelocked_empty(shape=(self.trt_inference.output_size,), dtype=np.float32)

        # Initialize threading
        self.inference_thread = None

    def left_image_callback(self, msg):
        frame = self.ros2_image_to_numpy(msg)
        self.left_image = frame

        # If both left and right images are available, start inference
        if self.right_image is not None:
            if self.inference_thread is None or not self.inference_thread.is_alive():
                self.inference_thread = Thread(target=self.run_inference)
                self.inference_thread.start()

    def right_image_callback(self, msg):
        frame = self.ros2_image_to_numpy(msg)
        self.right_image = frame

        # If both left and right images are available, start inference
        if self.left_image is not None:
            if self.inference_thread is None or not self.inference_thread.is_alive():
                self.inference_thread = Thread(target=self.run_inference)
                self.inference_thread.start()

    def run_inference(self):
        # Batch left and right images
        self.preprocess_image(self.left_image, 0)  # Preprocess left image into batch slot 0
        self.preprocess_image(self.right_image, 1)  # Preprocess right image into batch slot 1
        
        # Asynchronously copy input batch to GPU
        self.trt_inference.set_input_async(self.host_input, stream=self.cuda_stream)

        # Run asynchronous inference
        self.trt_inference.infer_async(stream=self.cuda_stream)

        # Asynchronously copy output from GPU to host
        output = self.trt_inference.get_output_async(self.host_output, stream=self.cuda_stream)

        # Post-process bounding boxes and other outputs (e.g., NMS)
        bboxes = self.post_process(output)

        # Optionally publish bounding boxes or visualize results
        self.publish_bounding_boxes(bboxes)

        # Reset left and right images for the next batch
        self.left_image = None
        self.right_image = None

    def preprocess_image(self, frame, index):
        """Perform preprocessing steps (resize, normalize, etc.) and copy to the pinned memory."""
        resized_frame = cv2.resize(frame, (self.trt_inference.input_width, self.trt_inference.input_height))
        normalized_frame = resized_frame.astype(np.float32) / 255.0  # Simple normalization

        # Copy preprocessed image to the pinned memory buffer at the correct batch index
        self.host_input[index] = normalized_frame

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

# stream synchronization:
# Run asynchronous inference
self.trt_inference.infer_async(stream=self.cuda_stream)

# Asynchronously copy output from GPU to host
output = self.trt_inference.get_output_async(self.host_output, stream=self.cuda_stream)

# Synchronize stream to ensure all GPU operations are complete before proceeding
self.cuda_stream.synchronize()

# Now it's safe to process the results
bboxes = self.post_process(output)
