import os
import cv2
import supervision as sv
from tabulate import tabulate
from timeloop import Timeloop
from datetime import timedelta

import rclpy
from rclpy.node import Node
from std_msgs.msg import String
from sensor_msgs.msg import Image
from cv_bridge import CvBridge, CvBridgeError
from example_interfaces.srv import Trigger

class DisplayNode(Node):
    def __init__(self):
        super().__init__('display_node') #type:ignore
        self.bridge = CvBridge()
        self.box_subscriber = self.create_subscription(String, 'bounding_boxes', self.callback, 10)
        self.client = self.create_client(Trigger, 'image_data_service')
        self.off_subscriber = self.create_subscription(String, 'off', self.display_metrics, 10)
        self.boxes, self.total_mem, self.total_cpu, self.total_exec, self.total_latency, self.frame_id, self.total_frames, self.fps, self.id, self.gpu, self.gpu_mem = None, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0
        self.metrics = [self.total_cpu, self.total_mem, self.total_exec, self.total_latency, self.gpu, self.gpu_mem]
    
    loop = Timeloop()
    @loop.job(interval=timedelta(seconds=1.000))
    def fps_counter(self):
        self.fps = self.totalframes
        self.total_frames = 0

    def callback(self, msg):
        self.boxes = msg.data
        
        metric_list = msg.header.split(' ')
        for index, metric in enumerate(self.metrics[1:len(self.metrics) - 1]):
            self.metric += metric_list[index + 1]
        self.total_frames = metric_list[0]
        
        req = Trigger.Request()
        future = self.client.call_async(req)
        future.add_done_callback(self.future_callback)
    
    # write launch parameter to change this type
    def future_callback(self, future):
        try:
            response = future.result()
            if response.success:
                image_data = response.message  # Assuming the message is the image data
                try:
                    cv_image = self.bridge.imgmsg_to_cv2(image_data, "8UC4")
                except CvBridgeError as e:
                    self.get_logger().info(e)
                    print(e)
                self.process_image(cv_image)
        except Exception as e:
            self.get_logger().error('Service call failed %r' % (e,))

    def process_image(self, cv_image):
        # assumed boxes is a list of bounding boxes represented as [x, y, w, h]
        # for box in self.boxes: #type: ignore
        #     x, y, w, h = box
        #     cv2.rectangle(cv_image, (x, y), (x + w, y + h), (0, 255, 0), 2)

        bounding_box_annotator = sv.BoundingBoxAnnotator()
        annotated_frame = bounding_box_annotator.annotate(
            scene=cv_image.copy(),
            detections=self.boxes
        )

        # Display the image in a new window
        cv2.imshow(f"FPS: {self.fps}", annotated_frame)

    def display_metrics(self):
        cv2.destroyAllWindows()
        print("="*40, "Average System Metrics", "="*40)
        avg_list = []
        for metric in self.metrics:
            avg_list.append(metric / self.total_frames)
        print(tabulate(avg_list, headers=("CPU", "Memory", "Execution Time", "Latency", "GPU", "GPU Memory")))
        raise SystemExit

def main(args=None):
    rclpy.init(args=args)
    display_node = DisplayNode()
    try:
        rclpy.spin(display_node)
    except SystemExit:    
        rclpy.logging.get_logger("Quitting").info('Done')
    display_node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()