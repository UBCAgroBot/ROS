import os
import cv2
import random
import rclpy
from rclpy.node import Node
from rclpy.executors import MultiThreadedExecutor

from std_msgs.msg import Header
from cv_bridge import CvBridge

from .scripts.utils import preprocess
from custom_interfaces.msg import ImageInput                            # CHANGE
# node to publish static image data to the /image topic. 
# used for testing the inference pipelines.
class VideoNode(Node):
    def __init__(self):
        super().__init__('video_node')

        self.declare_parameter('video_path', '/home/vscode/workspace/ros2_ws/src/extermination_package/extermination_package/video_node.py')
        self.declare_parameter('loop', -1)  # 0 = don't loop, >0 = # of loops, -1 = loop forever
        self.declare_parameter('frame_rate', 10)  # Desired frame rate for publishing
        self.declare_parameter('camera_side', 'left')
        self.camera_side = self.get_parameter('camera_side').get_parameter_value().string_value
        
        self.video_path = self.get_parameter('video_path').get_parameter_value().string_value
        # Resolve the path programmatically
        if not os.path.isabs(self.video_path):
            self.video_path = os.path.join(os.getcwd(), self.video_path)

        self.loop = self.get_parameter('loop').get_parameter_value().integer_value
        self.frame_rate = self.get_parameter('frame_rate').get_parameter_value().integer_value

        self.bridge = CvBridge()

        self.loop_count = 0
        self.image_counter = 0

        
 
         
        video_fps =  self.vr.get(cv2.CAP_PROP_FPS)
        self.interval = int(video_fps / self.frame_rate)

        self.counter = 0
        
        self.input_image_publisher = self.create_publisher(ImageInput, f'{self.camera_side}_image_input', 10)
        timer_period = 1/self.frame_rate  # publish every 0.5 seconds
        self.timer = self.create_timer(timer_period * 2, self.publish_static_image)
        self.openVR()

    
    def publish_static_image(self):
        """
        Publishes static images to the /image topic
        """
        count = 0
        interval = self.interval

        success, image = self.vr.read()
        while success:
            if count == interval:
                image_input = ImageInput()
                
                raw_image = image
                postprocessed_img = preprocess(raw_image)
                postprocessed_img_msg = self.bridge.cv2_to_imgmsg(postprocessed_img, encoding='rgb8')
                raw_img_msg = self.bridge.cv2_to_imgmsg(raw_image, encoding='rgb8')

                image_input.header = Header()
                image_input.header.frame_id = 'static_image'
                image_input.raw_image = raw_img_msg
                image_input.preprocessed_image = postprocessed_img_msg
                image_input.velocity = random.uniform(0,1)




                # publish image and increment whatever is needed
                self.input_image_publisher.publish(image_input)
                self.get_logger().info(f"Published image {self.image_counter}")
                self.image_counter += 1

                return
            
            success, image = self.vr.read()
            count += 1

        if self.loop == -1 or self.loop_count < self.loop:
            self.openVR()
            self.loop_count += 1
            self.publish_static_image()
        else:
            # stop the timer/quit the node
            self.timer.cancel()
            self.get_logger().info("Finished publishing images")
            self.destroy_node()
            


    def openVR(self):
        self.vr = cv2.VideoCapture(self.video_path)
        if not self.vr.isOpened():
            print("Error: Could not open video.")
            self.timer.cancel()
            self.destroy_node()

def main(args=None):
    rclpy.init(args=args)
    video_node = VideoNode()
    executor = MultiThreadedExecutor(num_threads=1)
    executor.add_node(video_node)
    
    try:
        executor.spin()
    except KeyboardInterrupt:
        print("Shutting down video node")
        return
    finally:
        executor.shutdown()
        video_node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()