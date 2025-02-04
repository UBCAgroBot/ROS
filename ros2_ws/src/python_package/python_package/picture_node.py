import time, os
import cv2
import random
import rclpy
from rclpy.time import Time
from rclpy.node import Node
from rclpy.executors import MultiThreadedExecutor
from sensor_msgs.msg import Image
from std_msgs.msg import Header, String, Int8
from cv_bridge import CvBridge
import queue
import numpy as np

from .scripts.utils import ModelInference

from custom_interfaces.msg import ImageInput                            # CHANGE
# node to publish static image data to the /image topic. 
# used for testing the inference pipelines.
class PictureNode(Node):
    def __init__(self):
        super().__init__('picture_node')

        self.declare_parameter('static_image_path', '/home/user/ROS/assets/maize/IMG_1822_14.JPG')
        self.declare_parameter('loop', -1)  # 0 = don't loop, >0 = # of loops, -1 = loop forever
        self.declare_parameter('frame_rate', 1)  # Desired frame rate for publishing
        
        self.static_image_path = self.get_parameter('static_image_path').get_parameter_value().string_value
        # Resolve the path programmatically
        if not os.path.isabs(self.static_image_path):
            self.static_image_path = os.path.join(os.getcwd(), self.static_image_path)

        self.loop = self.get_parameter('loop').get_parameter_value().integer_value
        self.frame_rate = self.get_parameter('frame_rate').get_parameter_value().integer_value

        self.bridge = CvBridge()
        self.model = ModelInference()

        self.image_list = self.get_images()
        self.loop_count = 0
        self.image_counter = 0

        self.input_image_publisher = self.create_publisher(ImageInput, 'input_image', 10)
        timer_period = 1/self.frame_rate  # publish every 0.5 seconds
        self.timer = self.create_timer(timer_period * 2, self.publish_static_image)

           
    def get_images(self)-> list[np.ndarray]:
        """
        Returns a list of images in the form of cv images 
        from the path specified by the static_image_path parameter.
        This is to simulate having a stream of camera images
        """
        # You can technically read the img file as binary data rather than using cv2 to read it and then convert it into a ROS2 image message... and then converting it back to a numpy array in the inference node. But since the picture node is only for testing the MVP and we're not using the GPU yet the overhead here should not matter.
        if not os.path.exists(self.static_image_path):
            self.get_logger().error(f"Static image not found at {self.static_image_path}")
            raise FileNotFoundError(f"Static image not found at {self.static_image_path}")
        
        filename = self.static_image_path
        
        image_paths = []
        if os.path.isfile(self.static_image_path) \
            and (filename.endswith('.JPG') or filename.endswith('.png')):
            filename = os.path.join(self.static_image_path, filename)
            image_paths.append(filename)
        elif os.path.isdir(self.static_image_path):
            for filename in os.listdir(self.static_image_path):
                if filename.endswith('.JPG') or filename.endswith('.png'):
                    filename = os.path.join(self.static_image_path, filename)
                    image_paths.append(filename)

        if len(image_paths) == 0:
            self.get_logger().error(f"No images found at {self.static_image_path}")
            return
        
        images = []
        for filename in image_paths:
            # read images
            image = cv2.imread(filename, cv2.IMREAD_COLOR)
            if image is None:
                self.get_logger().error(f"Failed to read image: {filename}")
                raise FileNotFoundError(f"Failed to read image: {filename}")
                        
            # add to the list of images
            images.append(image)

        return images

    
    def publish_static_image(self):
        """
        Publishes static images to the /image topic
        """
        array_size = len(self.image_list)

        # publish the image on the current pos to the image topic
        if self.loop == -1 or self.loop_count < self.loop:
            #get image from list of uploaded
            position = self.image_counter % array_size
            self.get_logger().info(f"Publishing image {position + 1}/{array_size}")

            raw_image = self.image_list[position]

            #todo:  create the message to publish
            postprocessed_img = self.model.preprocess(raw_image)
            postprocessed_img_msg = self.bridge.cv2_to_imgmsg(postprocessed_img, encoding='rgb8')
            raw_img_msg = self.bridge.cv2_to_imgmsg(raw_image, encoding='rgb8')

            image_input = ImageInput()
            image_input.header = Header()
            image_input.header.frame_id = 'static_image'
            image_input.raw_image = raw_img_msg
            image_input.preprocessed_image = postprocessed_img_msg
            image_input.velocity = random.uniform(0,1)

            # publish image and increment whatever is needed
            self.input_image_publisher.publish(image_input)
            self.get_logger().info(f"Published image {self.image_counter}")
            self.image_counter += 1
            self.loop_count = self.image_counter // array_size
            
        else:
            # stop the timer/quit the node
            self.timer.cancel()
            self.get_logger().info("Finished publishing images")
            self.destroy_node()


    def crop_with_velocity(self, image: np.ndarray, velocity: float):
        """
        Takes in an image and crops it with the velocity shift
        """
        return image

def main(args=None):
    rclpy.init(args=args)
    picture_node = PictureNode()
    executor = MultiThreadedExecutor(num_threads=1)
    executor.add_node(picture_node)
    
    try:
        executor.spin()
    except KeyboardInterrupt:
        print("Shutting down picture node")
        return
    finally:
        executor.shutdown()
        picture_node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()