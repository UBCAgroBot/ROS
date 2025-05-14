import os
import cv2
import rclpy
from rclpy.node import Node
from std_msgs.msg import String
from sensor_msgs.msg import Image
from cv_bridge import CvBridge

class CameraNode(Node):
    def __init__(self):
        super().__init__('camera_node')

        # Parameters
        self.declare_parameter('input_type', 'video')  # 'video' or 'image'
        self.declare_parameter('input_path', '') # no video path yet  
        self.declare_parameter('frame_rate', 10)  # Frames per second for video

        # Get parameters
        self.input_type = self.get_parameter('input_type').get_parameter_value().string_value
        self.input_path = self.get_parameter('input_path').get_parameter_value().string_value
        self.frame_rate = self.get_parameter('frame_rate').get_parameter_value().integer_value

        # Resolve input path
        if not os.path.isabs(self.input_path):
            self.input_path = os.path.join(os.getcwd(), self.input_path)

        # Initialize input source
        self.bridge = CvBridge()
        self.robot_state = 'stopped'  # Default state
        self.video_capture = None

        if self.input_type == 'video':
            self.init_video()
        elif self.input_type == 'image':
            self.init_image()
        else:
            self.get_logger().error(f"Invalid input_type: {self.input_type}. Use 'video' or 'image'.")
            self.destroy_node()

        # ROS 2 communication
        self.state_subscription = self.create_subscription(
            String,
            '/robot_state',
            self.state_callback,
            10
        )
        self.image_publisher = self.create_publisher(Image, '/camera/image_raw', 10)

        self.get_logger().info('CameraNode has been started.')

    def init_video(self):
        """Initialize video input."""
        self.video_capture = cv2.VideoCapture(self.input_path)
        if not self.video_capture.isOpened():
            self.get_logger().error(f"Failed to open video: {self.input_path}")
            self.destroy_node()

    def init_image(self):
        """Initialize static image input."""
        if not os.path.exists(self.input_path):
            self.get_logger().error(f"Image file not found: {self.input_path}")
            self.destroy_node()
        self.static_image = cv2.imread(self.input_path)
        if self.static_image is None:
            self.get_logger().error(f"Failed to read image: {self.input_path}")
            self.destroy_node()

    def state_callback(self, msg):
        """Callback for /robot_state topic."""
        self.robot_state = msg.data
        self.get_logger().info(f'Received robot state: {self.robot_state}')

        if self.robot_state == 'moving' and self.input_type == 'video':
            self.start_continuous_capture()
        elif self.robot_state == 'stopped':
            self.capture_static_image()

    def start_continuous_capture(self):
        """Continuously capture and publish frames from the video."""
        self.get_logger().info('Robot is moving. Starting continuous video capture...')
        while self.robot_state == 'moving' and rclpy.ok():
            success, frame = self.video_capture.read()
            if success:
                self.publish_image(frame)
            else:
                self.get_logger().info('End of video reached. Restarting...')
                self.video_capture.set(cv2.CAP_PROP_POS_FRAMES, 0)

    def capture_static_image(self):
        """Publish a single static image."""
        self.get_logger().info('Robot is stopped. Publishing static image...')
        if self.input_type == 'image':
            self.publish_image(self.static_image)
        elif self.input_type == 'video':
            # Publish the last frame from the video
            success, frame = self.video_capture.read()
            if success:
                self.publish_image(frame)

    def publish_image(self, frame):
        """Convert and publish an image to the /camera/image_raw topic."""
        try:
            ros_image = self.bridge.cv2_to_imgmsg(frame, encoding='bgr8')
            self.image_publisher.publish(ros_image)
            self.get_logger().info('Published image to /camera/image_raw.')
        except Exception as e:
            self.get_logger().error(f'Failed to publish image: {e}')

    def destroy_node(self):
        """Clean up resources."""
        if self.video_capture is not None:
            self.video_capture.release()
        super().destroy_node()

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