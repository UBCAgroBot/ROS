import os
import cv2
import random
import rclpy
from rclpy.node import Node
from std_msgs.msg import String, Header
from sensor_msgs.msg import Image
from cv_bridge import CvBridge

from custom_interfaces.msg import ImageInput
from .scripts.utils import preprocess

class CameraNode(Node):
    def __init__(self):
        super().__init__('camera_node')

        # Parameters
        self.declare_parameter('video_path', '/home/vscode/workspace/assets/IMG_5947.MOV')
        self.declare_parameter('loop', -1)
        self.declare_parameter('frame_rate', 10)

        self.video_path = self.get_parameter('video_path').get_parameter_value().string_value
        if not os.path.isabs(self.video_path):
            self.video_path = os.path.join(os.getcwd(), self.video_path)
        self.loop = self.get_parameter('loop').get_parameter_value().integer_value
        self.frame_rate = self.get_parameter('frame_rate').get_parameter_value().integer_value

        self.bridge = CvBridge()
        self.loop_count = 0
        self.image_counter = 0
        self.vr = None
        self.timer = None
        self.robot_state = 'stopped'

        self.input_image_publisher = self.create_publisher(ImageInput, 'image_input', 10)
        self.state_subscription = self.create_subscription(
            String, '/robot_state', self.state_callback, 10
        )

        self.openVR()
        video_fps = self.vr.get(cv2.CAP_PROP_FPS)
        self.interval = max(1, int(video_fps / self.frame_rate)) if video_fps > 0 else 1
        self.counter = 0

    def state_callback(self, msg):
        self.robot_state = msg.data
        if self.robot_state == 'moving':
            if self.timer is None:
                timer_period = 1 / self.frame_rate
                self.timer = self.create_timer(timer_period, self.publish_video_frame)
        else:
            if self.timer is not None:
                self.timer.cancel()
                self.timer = None
            self.publish_static_image()

    def publish_video_frame(self):
        # Only publish if robot is moving
        if self.robot_state != 'moving':
            return
        count = 0
        interval = self.interval
        success, image = self.vr.read()
        while success:
            if count == interval:
                self.publish_image(image)
                return
            success, image = self.vr.read()
            count += 1

        # End of video
        if self.loop == -1 or self.loop_count < self.loop:
            self.openVR()
            self.loop_count += 1
            self.publish_video_frame()
        else:
            if self.timer is not None:
                self.timer.cancel()
            self.get_logger().info("Finished publishing images")
            self.destroy_node()

    def publish_static_image(self):
        # Publish the current frame as a static image
        success, image = self.vr.read()
        if success:
            self.publish_image(image)

    def publish_image(self, image):
        image_input = ImageInput()
        raw_image = image
        postprocessed_img = preprocess(raw_image)
        postprocessed_img_msg = self.bridge.cv2_to_imgmsg(postprocessed_img, encoding='rgb8')
        raw_img_msg = self.bridge.cv2_to_imgmsg(raw_image, encoding='rgb8')

        image_input.header = Header()
        image_input.header.frame_id = 'camera'
        image_input.raw_image = raw_img_msg
        image_input.preprocessed_image = postprocessed_img_msg
        image_input.velocity = random.uniform(0, 1)

        self.input_image_publisher.publish(image_input)
        self.get_logger().info(f"Published image {self.image_counter}")
        self.image_counter += 1

    def openVR(self):
        self.vr = cv2.VideoCapture(self.video_path)
        if not self.vr.isOpened():
            self.get_logger().error(f"Error: Could not open video at {self.video_path}")
            if self.timer is not None:
                self.timer.cancel()
            self.destroy_node()

def main(args=None):
    rclpy.init(args=args)
    node = CameraNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()