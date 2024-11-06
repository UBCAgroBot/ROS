import os
from cv_bridge import CvBridge
import time
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from std_msgs.msg import Int32MultiArray, Float32MultiArray
from .scripts.utils import ModelInference
from custom_interfaces.msg import ImageInput, InferenceOutput                            # CHANGE


# for a service implementation: https://robotics.stackexchange.com/questions/88791/ros2-how-to-call-a-service-from-the-callback-function-of-a-subscriber
class InferenceNode(Node):
    def __init__(self):
        super().__init__('inference_node')

        self.declare_parameter('weights_path', './src/python_workspace/python_workspace/scripts/best.onnx')
        self.declare_parameter('precision', 'fp32') # fp32, fp16 # todo: do something with strip_weights and precision
        
        self.weights_path = self.get_parameter('weights_path').get_parameter_value().string_value
        if not os.path.isabs(self.weights_path):
            self.weights_path = os.path.join(os.getcwd(), self.weights_path)


        self.precision = self.get_parameter('precision').get_parameter_value().string_value

        # instantiate the model here
        self.model = ModelInference(self.weights_path, self.precision)
        
        self.bridge = CvBridge()

        self.subscription = self.create_subscription(ImageInput, 'input_image', self.image_callback, 10)
        
        # create a publisher for the output image/boxes/extermination data
        self.box_publisher = self.create_publisher(InferenceOutput,'inference_out', 10)

        self.output_image_publisher = self.create_publisher(Image, 'output_img', 10)        
    
    def image_callback(self, msg):
        # print("============================================")
        opencv_img = self.bridge.imgmsg_to_cv2(msg.preprocessed_image, desired_encoding='passthrough')
        output_img, confidences, boxes = self.model.inference(opencv_img)

        # publish bounding box etc as inference output
        output_msg = InferenceOutput()
        # output_msg.preprocessed_img = msg.image
        output_msg.num_boxes = len(boxes)
        output_msg.raw_image = msg.raw_image
        output_msg.velocity = msg.velocity
        output_msg.preprocessed_image = msg.preprocessed_image
        bounding_boxes = Float32MultiArray()
        confidences_msg = Float32MultiArray()
        bounding_boxes.data = []
        confidences_msg.data = []
        if len(boxes) != 0:
            bounding_boxes.data = boxes
            confidences_msg.data = confidences
            output_msg.bounding_boxes = bounding_boxes
            output_msg.confidences = confidences_msg
        self.box_publisher.publish(output_msg)


        # convert the output image to a ROS2 image message
        # todo: this will be moved to the extermination node
        output_image_msg = self.bridge.cv2_to_imgmsg(output_img, encoding='rgb8')

        self.output_image_publisher.publish(output_image_msg)




def main(args=None):
    rclpy.init(args=args)
    inference_node = InferenceNode()
    rclpy.spin(inference_node)
    inference_node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()