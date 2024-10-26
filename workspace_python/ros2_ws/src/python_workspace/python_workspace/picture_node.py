import time, os
import cv2
import random
import rclpy
from rclpy.time import Time
from rclpy.node import Node
from rclpy.executors import MultiThreadedExecutor
from sensor_msgs.msg import Image
from std_msgs.msg import Header, String
from cv_bridge import CvBridge

from custom_interfaces.msg import ImageInput                            # CHANGE

# node to publish static image data to the /image topic. 
# used for testing the inference pipelines.
class PictureNode(Node):
    def __init__(self):
        super().__init__('picture_node')

        self.declare_parameter('static_image_path', '/home/usr/Desktop/ROS/assets/IMG_1822_14.JPG')
        self.declare_parameter('loop', 0)  # 0 = don't loop, >0 = # of loops, -1 = loop forever
        self.declare_parameter('frame_rate', 1)  # Desired frame rate for publishing

        
        self.static_image_path = self.get_parameter('static_image_path').get_parameter_value().string_value
        # Resolve the path programmatically
        if not os.path.isabs(self.static_image_path):
            self.static_image_path = os.path.join(os.getcwd(), self.static_image_path)

        self.loop = self.get_parameter('loop').get_parameter_value().integer_value
        self.frame_rate = self.get_parameter('frame_rate').get_parameter_value().integer_value

        self.bridge = CvBridge()

        self.image_list = self.get_images()
        self.loop_count = 0
        self.image_counter = 0

        self.input_image_publisher = self.create_publisher(ImageInput, 'input_image', 10)
        timer_period = 1/self.frame_rate  # publish every 0.5 seconds
        self.timer = self.create_timer(timer_period, self.publish_static_image)

           
    def get_images(self):
        """
        Returns a list of images in the form of ROS2 image messages from the path specified by the static_image_path parameter.
        """
        # You can technically read the img file as binary data rather than using cv2 to read it and then convert it into a ROS2 image message... and then converting it back to a numpy array in the inference node. But since the picture node is only for testing the MVP and we're not using the GPU yet the overhead here should not matter.
        if not os.path.exists(self.static_image_path):
            self.get_logger().error(f"Static image not found at {self.static_image_path}")
            raise FileNotFoundError(f"Static image not found at {self.static_image_path}")
        
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
        counter = 0
        for filename in image_paths:

            image = cv2.imread(filename, cv2.IMREAD_COLOR)
            ros_image = self.bridge.cv2_to_imgmsg(image, encoding="bgr8")
            if image is None:
                self.get_logger().error(f"Failed to read image: {filename}")
                raise FileNotFoundError(f"Failed to read image: {filename}")
            
            image_msg = ImageInput()
            # image_msg.header.frame_id = f'camera_frame{counter}'
            image_msg.image = ros_image
            image_msg.velocity = random.uniform(0, 2)

            counter +=1

            images.append(image_msg)

        return images

    
    def publish_static_image(self):
        """
        Publishes static images to the /image topic
        """
        array_size = len(self.image_list)

        # publish the image on the current pos to the image topic
        if self.loop == -1 or self.loop_count < self.loop:
            position = self.image_counter % array_size
            self.get_logger().info(f"Publishing image {position + 1}/{array_size}")
            self.input_image_publisher.publish(self.image_list[position])

            self.image_counter += 1
            self.loop_count = self.image_counter // array_size
        else:
            # stop the timer/quit the node
            self.timer.cancel()
            self.get_logger().info("Finished publishing images")
            self.destroy_node()

def main(args=None):
    rclpy.init(args=args)
    picture_node = PictureNode()
    executor = MultiThreadedExecutor(num_threads=1)
    executor.add_node(picture_node)
    
    try:
        executor.spin()
    finally:
        executor.shutdown()
        picture_node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()