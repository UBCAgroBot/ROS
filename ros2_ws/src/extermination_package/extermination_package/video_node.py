import os
import cv2
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from cv_bridge import CvBridge

# publish static image data to /image topic
# listens to /robot_status

class VideoNode(Node):
    def __init__(self):
        super().__init__('video_node')

        # Declare parameters
        self.declare_parameter('video_path', '/home/vscode/workspace/assets/IMG_5947.MOV')  # no video yet
        self.declare_parameter('loop', -1)  # -1 = loop forever, 0 = no loop, >0 = loop count
        self.declare_parameter('frame_rate', 10) 

        # Get parameters
        self.video_path = self.get_parameter('video_path').get_parameter_value().string_value
        self.loop = self.get_parameter('loop').get_parameter_value().integer_value
        self.frame_rate = self.get_parameter('frame_rate').get_parameter_value().integer_value

        # Resolve path
        if not os.path.isabs(self.video_path):
            self.video_path = os.path.join(os.getcwd(), self.video_path)

        # Initialize video capture
        self.bridge = CvBridge()
        self.loop_count = 0
        self.vr = None
        self.open_video()

        # ROS 2 publisher
        self.image_publisher = self.create_publisher(Image, '/camera/image_raw', 10)

        # Timer for publishing frames
        self.timer = self.create_timer(1.0 / self.frame_rate, self.publish_frame)

    def open_video(self):
        """Open the video file."""
        self.vr = cv2.VideoCapture(self.video_path)
        if not self.vr.isOpened():
            self.get_logger().error(f"Failed to open video: {self.video_path}")
            self.destroy_node()

    def publish_frame(self):
        """Publish a single frame from the video."""
        success, frame = self.vr.read()
        if success:
            # Convert frame to ROS 2 Image message
            ros_image = self.bridge.cv2_to_imgmsg(frame, encoding='bgr8')
            self.image_publisher.publish(ros_image)
            self.get_logger().info('Published a frame.')
        else:
            # Handle end of video
            if self.loop == -1 or self.loop_count < self.loop:
                self.loop_count += 1
                self.open_video()
            else:
                self.get_logger().info('Finished playing video.')
                self.destroy_node()

    def destroy_node(self):
        """Clean up resources."""
        if self.vr is not None:
            self.vr.release()
        super().destroy_node()

def main(args=None):
    rclpy.init(args=args)
    node = VideoNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()