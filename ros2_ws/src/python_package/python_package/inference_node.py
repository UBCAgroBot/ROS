import time
import os

import rclpy
from rclpy.node import Node
from cv_bridge import CvBridge

from sensor_msgs.msg import Image
from std_msgs.msg import Float32MultiArray
from rclpy.executors import MultiThreadedExecutor

from custom_interfaces.msg import ImageInput, InferenceOutput
from .scripts.utils import initialise_model, run_inference


class InferenceNode(Node):
    def __init__(self):
        super().__init__('inference_node')

        self.declare_parameter('weights_path', '/home/user/ROS/models/maize/Maize.onnx')
        self.declare_parameter('precision', 'fp32') # fp32, fp16 # todo: do something with strip_weights and precision
        self.declare_parameter('camera_side', 'left')

        self.camera_side = self.get_parameter('camera_side').get_parameter_value().string_value
        self.weights_path = self.get_parameter('weights_path').get_parameter_value().string_value
        if not os.path.isabs(self.weights_path):
            self.weights_path = os.path.join(os.getcwd(), self.weights_path)

        self.precision = self.get_parameter('precision').get_parameter_value().string_value

        self.model = initialise_model(self.weights_path, self.precision)
        self.bridge = CvBridge()

        self.image_subscription = self.create_subscription(ImageInput, f'{self.camera_side}_image_input', self.image_callback, 10)
        self.box_publisher = self.create_publisher(InferenceOutput,f'{self.camera_side}_inference_output', 10) 
    
    def image_callback(self, msg):
        opencv_img = self.bridge.imgmsg_to_cv2(msg.preprocessed_image, desired_encoding='passthrough')

        tic = time.perf_counter_ns()
        output_img, confidences, boxes = run_inference(self.model,opencv_img)
        toc = time.perf_counter_ns()

        output_msg = InferenceOutput()
        output_msg.num_boxes = len(boxes)
        output_msg.raw_image = msg.raw_image
        output_msg.velocity = msg.velocity
        output_msg.preprocessed_image = msg.preprocessed_image
        bounding_boxes = Float32MultiArray()
        confidences_msg = Float32MultiArray()
        bounding_boxes.data = []
        confidences_msg.data = []
        if len(boxes) != 0:
            bounding_boxes.data = boxes.reshape(-1).tolist()
            confidences_msg.data = confidences.reshape(-1).tolist()
            output_msg.bounding_boxes = bounding_boxes
            output_msg.confidences = confidences_msg
        self.box_publisher.publish(output_msg)
        
        self.get_logger().info(f"Inference: {(toc-tic)/1e6} ms")

def main(args=None):
    rclpy.init(args=args)
    inference_node = InferenceNode()
    executor = MultiThreadedExecutor(num_threads=1)
    executor.add_node(inference_node)

    try:
        executor.spin()
    except KeyboardInterrupt:
        print("Shutting down inference node")
    finally:
        executor.shutdown()
        inference_node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()