import rclpy
from rclpy.callback_groups import ReentrantCallbackGroup
from rclpy.executors import MultiThreadedExecutor
from rclpy.node import Node
from sensor_msgs.msg import Image
import cv2
import numpy as np
from trt_inference import TRTInference  # TensorRT inference class

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

        # Buffer for left and right images
        self.left_image = None
        self.right_image = None

    def left_image_callback(self, msg):
        frame = self.ros2_image_to_numpy(msg)
        self.left_image = frame

        # If both left and right images are available, start inference
        if self.right_image is not None:
            self.run_inference()

    def right_image_callback(self, msg):
        frame = self.ros2_image_to_numpy(msg)
        self.right_image = frame

        # If both left and right images are available, start inference
        if self.left_image is not None:
            self.run_inference()

    def run_inference(self):
        # Batch left and right images and run inference here
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

# ReentrantCallbackGroup: This callback group allows the same callback to be executed in parallel across multiple threads. Each subscription (left and right image) is assigned a different ReentrantCallbackGroup to ensure they can be processed concurrently.

# MultiThreadedExecutor: The MultiThreadedExecutor is created with a thread pool size of 2, allowing two callbacks (left and right image) to be processed in parallel.

#  parallelize preprocessing, inference, postprocessing, and publishing operations while ensuring proper GPU context and memory management, we can leverage ROS2's multithreaded executor with callback groups for different stages of the pipeline.

import rclpy
from rclpy.callback_groups import ReentrantCallbackGroup
from rclpy.executors import MultiThreadedExecutor
from rclpy.node import Node
from sensor_msgs.msg import Image
import cv2
import numpy as np
import pycuda.driver as cuda

from trt_inference import TRTInference  # TensorRT inference class

class BoundingBoxNode(Node):
    def __init__(self):
        super().__init__('bounding_box_node')

        # Initialize callback groups for parallel processing
        self.preprocess_group = ReentrantCallbackGroup()
        self.inference_group = ReentrantCallbackGroup()
        self.postprocess_group = ReentrantCallbackGroup()
        self.publish_group = ReentrantCallbackGroup()

        # Subscribers for left and right camera image topics, each in the preprocess group
        self.left_image_subscriber = self.create_subscription(
            Image, '/camera/left/image_raw', self.left_image_callback, 10, callback_group=self.preprocess_group)

        self.right_image_subscriber = self.create_subscription(
            Image, '/camera/right/image_raw', self.right_image_callback, 10, callback_group=self.preprocess_group)

        # Initialize TensorRT inference engine
        self.trt_inference = TRTInference('model.engine')

        # CUDA stream for asynchronous operations
        self.cuda_stream = cuda.Stream()

        # Buffers for left and right images
        self.left_image = None
        self.right_image = None

        # Create publishers to publish bounding box results
        self.bounding_boxes_publisher = self.create_publisher(
            Image, '/bounding_boxes', 10, callback_group=self.publish_group)

    def left_image_callback(self, msg):
        # Preprocess left image
        self.get_logger().info('Preprocessing left image...')
        frame = self.ros2_image_to_numpy(msg)
        self.left_image = self.preprocess_image(frame)

        # If both left and right images are ready, run inference
        if self.right_image is not None:
            self.create_task(self.run_inference, self.inference_group)

    def right_image_callback(self, msg):
        # Preprocess right image
        self.get_logger().info('Preprocessing right image...')
        frame = self.ros2_image_to_numpy(msg)
        self.right_image = self.preprocess_image(frame)

        # If both left and right images are ready, run inference
        if self.left_image is not None:
            self.create_task(self.run_inference, self.inference_group)

    def preprocess_image(self, image):
        # Example preprocessing: resize, normalize, etc.
        # Perform any necessary transformations for the model
        return cv2.resize(image, (224, 224))

    def run_inference(self):
        # Ensure both images are preprocessed before inference
        if self.left_image is not None and self.right_image is not None:
            self.get_logger().info('Running inference...')

            # Batch left and right images together
            batched_images = np.stack([self.left_image, self.right_image])

            # Run asynchronous inference on GPU using the CUDA stream
            self.trt_inference.infer_async(batched_images, stream=self.cuda_stream)

            # Asynchronously copy the inference results from GPU to host
            self.create_task(self.postprocess_results, self.postprocess_group)

    def postprocess_results(self):
        self.get_logger().info('Postprocessing results...')
        # Wait for inference and data transfers to complete
        self.cuda_stream.synchronize()

        # Retrieve and postprocess the results
        output = self.trt_inference.get_output()
        bboxes = self.post_process(output)

        # Publish the results
        self.create_task(lambda: self.publish_results(bboxes), self.publish_group)

    def post_process(self, output):
        # Convert raw output into bounding boxes or detection results
        return output  # Example post-processing step

    def publish_results(self, bboxes):
        self.get_logger().info('Publishing results...')
        # Publish bounding boxes or any other output format
        # Placeholder for publishing logic; replace with actual ROS message type
        msg = Image()  # Use the appropriate message type for bounding boxes
        self.bounding_boxes_publisher.publish(msg)

    def ros2_image_to_numpy(self, msg):
        # Convert ROS2 Image message to a NumPy array
        return np.frombuffer(msg.data, dtype=np.uint8).reshape(msg.height, msg.width, 3)

    def create_task(self, target_function, callback_group):
        # Helper function to execute tasks within the appropriate callback group
        self.get_logger().info('Creating async task...')
        self.executor.submit(target_function)

def main(args=None):
    rclpy.init(args=args)

    node = BoundingBoxNode()

    # Use a MultiThreadedExecutor with multiple threads for parallel callbacks
    executor = MultiThreadedExecutor(num_threads=4)
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
