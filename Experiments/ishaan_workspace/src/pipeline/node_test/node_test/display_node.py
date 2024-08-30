import cv2
import threading

import rclpy
from rclpy.node import Node
from std_msgs.msg import String
from sensor_msgs.msg import Image
from cv_bridge import CvBridge, CvBridgeError
from BoundingBox.msg import BoundingBox

from tqdm import tqdm
from tabulate import tabulate
from timeloop import Timeloop
from datetime import timedelta

class_labels = [
    'person', 'bicycle', 'car', 'motorbike', 'aeroplane', 'bus', 'train', 'truck', 'boat', 
    'traffic light', 'fire hydrant', 'stop sign', 'parking meter', 'bench', 'bird', 'cat', 
    'dog', 'horse', 'sheep', 'cow', 'elephant', 'bear', 'zebra', 'giraffe', 'backpack', 
    'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball', 
    'kite', 'baseball bat', 'baseball glove', 'skateboard', 'surfboard', 'tennis racket', 
    'bottle', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple', 
    'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake', 'chair', 
    'sofa', 'pottedplant', 'bed', 'diningtable', 'toilet', 'tvmonitor', 'laptop', 'mouse', 
    'remote', 'keyboard', 'cell phone', 'microwave', 'oven', 'toaster', 'sink', 'refrigerator', 
    'book', 'clock', 'vase', 'scissors', 'teddy bear', 'hair drier', 'toothbrush'
]

class Image_subscriber(Node):
    def __init__(self):
        super().__init__('image_subscriber')
        self.bridge = CvBridge()
        self.subscription = self.create_subscription(Image, 'image_data', self.image_callback, 10) # 10 or 1?
        self.subscription

    def image_callback(self, data):
        global img
        global frame_id
        global frame_count
        try:
            img = self.bridge.imgmsg_to_cv2(data, "bgr8")
            frame_id, total_frames = data.header.frame_id, total_frames + 1
        except CvBridgeError as e:
            print(e)
        
class Yolo_subscriber(Node):
    def __init__(self):
        super().__init__('yolo_subscriber') #type:ignore
        self.bridge = CvBridge()
        self.box_subscriber = self.create_subscription(String, 'bounding_boxes', self.process_image, 10)
        self.total_mem, self.total_cpu, self.total_exec, self.total_latency, self.gpu, self.gpu_mem, self.fps = 0, 0, 0, 0, 0, 0, 0
        self.metrics = [self.total_cpu, self.total_mem, self.total_exec, self.total_latency, self.gpu, self.gpu_mem]

        # rolling average over number of frames
        loop = Timeloop()
        @loop.job(interval=timedelta(milliseconds=1000))
        def fps_counter(self):
            global frame_count
            self.fps = frame_count
            frame_count = 0
        
        self.subscriber = self.create_subscription(BoundingBox,'bounding_boxes', self.bounding_box_callback, 10)
        self.subscriber  
        
    def process_callback(self, msg):
        global img
        
        boxes = msg.box
        classes = msg.class_num
        scores = msg.score
        
        metric_list = msg.header.split(' ')
        for index, metric in enumerate(self.metrics[1:len(self.metrics) - 1]): # just remove the first element for the total frames, not needed
            self.metric += metric_list[index + 1]
        # self.total_frames = metric_list[0] # delete

        for box, score, class_ in zip(boxes, scores, classes):
            # Draw the bounding box
            top, left, bottom, right = box[0], box[1], box[2], box[3] #xmin, ymin, xmax, ymax
            cv2.rectangle(img, (top, left), (bottom, right), (255, 255, 0))
            class_ = class_labels[class_]

            # Draw the class and score
            cv2.putText(img, f"{class_}: {score}", (top, left - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2, cv2.LINE_AA)
        
        cv2.imshow("Image", img)
        cv2.waitKey(3) # what is 3?

    def display_metrics(self):
        global frame_id
        cv2.destroyAllWindows()
        print("="*40, "Average System Metrics", "="*40)
        avg_list = []
        for metric in self.metrics:
            avg_list.append(metric / frame_id)
        print(tabulate(avg_list, headers=("CPU", "Memory", "Execution Time", "Latency", "GPU", "GPU Memory")))
        raise SystemExit

# better way to do this? blocking?
def metric_bar(cpu, ram, gpu, gpu_mem, exec, latency):
    with tqdm(total=100, desc='cpu%', position=1) as cpubar, tqdm(total=100, desc='ram%', position=0) as rambar, tqdm(total=100, desc='gpu%', position=2) as gpubar, tqdm(total=100, desc='gpu_mem%', position=3) as gpumembar, tqdm(total=100, desc='exec%', position=4) as execbar, tqdm(total=100, desc='latency%', position=5) as latencybar:
        while True:
            rambar.n=cpu
            cpubar.n=ram
            gpubar.n=gpu
            gpumembar.n=gpu_mem
            execbar.n=execbar
            latencybar.n=latency
            
            rambar.refresh()
            cpubar.refresh()
            gpumembar.refresh()
            gpubar.refresh()
            execbar.refresh()
            latencybar.refresh()

def main(args=None):
    rclpy.init(args=args)
    image_subscriber = Image_subscriber()
    yolo_subscriber = Yolo_subscriber()
    
    executor = rclpy.executors.MultiThreadedExecutor()
    executor.add_node(image_subscriber)
    executor.add_node(yolo_subscriber)
    
    executor_thread = threading.Thread(target=executor.spin, daemon=True)
    executor_thread.start()
    rate = yolo_subscriber.create_rate(2) #idk what this does
    
    try:
        while rclpy.ok():
            rate.sleep()
    except SystemExit:    
        yolo_subscriber.display_metrics()
        rclpy.logging.get_logger("Quitting").info('Done')

    rclpy.shutdown()
    executor_thread.join()

if __name__ == '__main__':
    main()

