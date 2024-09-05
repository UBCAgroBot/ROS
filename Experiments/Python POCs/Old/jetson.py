import time
import os
import cv2
import psutil
import GPUtil
import supervision as sv
from ultralytics import YOLO
from numba import jit
from typing import Callable, Any

import rclpy
from rclpy.time import Time
from rclpy.node import Node
from std_msgs.msg import Header, String
from sensor_msgs.msg import Image
from cv_bridge import CvBridge, CvBridgeError

class JetsonNode(Node):
    def __init__(self):
        super().__init__('jetson_node') #type:ignore
        self.bridge = CvBridge()
        self.model_publisher = self.create_publisher(String, 'bounding_boxes', 10)
        self.camera_subscriber = self.create_subscription(Image, 'image_data', self.callback, 10)
        self.off_subscriber = self.create_subscription(String, 'off', self.display_metrics, 10)
        self.frames, self.cpu, self.mem, self.time, self.latency, self.pid = 0, 0, 0, 0, 0, 0
        self.model = YOLO('yolov8s.pt')
        self.tensorrt_init()
    
    def tensorrt_init(self):
        pass
    
    def callback(self, msg):
        self.get_logger().info(f"Received: {msg.header}")
        now = self.get_clock().now()
        latency = now - Time.from_msg(msg.header.stamp)
        print(f"Message transmisison latency: {latency.nanoseconds / 1e6} milliseconds")
        self.latency, self.frame_id, self.frames = latency.nanoseconds / 1e6, msg.header.frame_id, self.frames + 1
        try:
            cv_image  = self.bridge.imgmsg_to_cv2(msg.data, encoding="8UC4") # or JPG
            self.detection(cv_image)
        except CvBridgeError as e:
            self.get_logger().info(e)
            print(e)

    def high_precision_sleep(self, duration):
        start_time = time.perf_counter()
        while True:
            elapsed_time = time.perf_counter() - start_time
            remaining_time = duration - elapsed_time
            if remaining_time <= 0:
                break
            if remaining_time > 0.02:  # Sleep for 5ms if remaining time is greater
                time.sleep(max(remaining_time/2, 0.0001))  # Sleep for the remaining time or minimum sleep interval
            else:
                pass

    def measure_CPU_GPU(self, pid):
        # If I have thread a and thread b under process P.
        proc = psutil.Process(pid)
        # Log the initial total time of process P: 
        initial_total_time = sum(proc.cpu_times())

        # log the initial time of each thread
        initial_thread_times = {'a': {'system': None, 'user': None}}
        for thread in proc.threads():
            initial_thread_times[psutil.Process(thread.id).name()]['system'] = thread.system_time
            initial_thread_times[psutil.Process(thread.id).name()]['user'] = thread.user_time
        
        # Determine total percent cpu usage over an interval
        total_cpu_percent = proc.cpu_percent(interval = 0.1)
        # grab the new total amount of time the process has used the cpu
        final_total_time = sum(proc.cpu_times())

        # grab the new system and user times for each thread
        final_thread_times = {'a': {'system': None, 'user': None}}
        for thread in proc.threads():
            final_thread_times[psutil.Process(thread.id).name()]['system'] = thread.system_time
            final_thread_times[psutil.Process(thread.id).name()]['user'] = thread.user_time
        
        # calculate how much cpu each thread used by...
        total_time_thread_a_used_cpu_over_time_interval = ((final_thread_times['a']['system']-initial_thread_times['a']['system']) + (final_thread_times['a']['user']-initial_thread_times['a']['user']))
        total_time_process_used_cpu_over_interval = final_total_time - initial_total_time
        percent_of_cpu_usage_utilized_by_thread_a = total_cpu_percent*(total_time_thread_a_used_cpu_over_time_interval/total_time_process_used_cpu_over_interval)
        
        gpu = GPUtil.getGPUs()[0] # firstGPU = GPU.getFirstAvailable()
        
        # Get 1 available GPU, ordered by GPU load ascending
        # print('First available weighted by GPU load ascending: '),
        # print(GPU.getAvailable(order='load', limit=1))
        
        # Get 1 available GPU, ordered by ID in descending order
        # print('Last available: '),
        # print(GPU.getAvailable(order='last', limit=1))  

        self.get_logger.info(f"CPU usage: {percent_of_cpu_usage_utilized_by_thread_a}%")
        self.get_logger.info(f"GPU usage: {gpu.load*100}%")
        self.get_logger.info(f"GPU VRAM usage: {(gpu.memoryUsed / gpu.memoryTotal) * 100}%")
        
        self.cpu = percent_of_cpu_usage_utilized_by_thread_a
        self.gpu = gpu.load*100
        self.gpu_mem = (gpu.memoryUsed / gpu.memoryTotal) * 100

    def detection(self, data):
        pid = os.getpid()
        tic = time.perf_counter_ns()
        pre_mem = psutil.Process(pid).memory_percent()
        self.measure_CPU_GPU(pid)
        result = self.model(data)[0]
        post_mem = psutil.Process(self.pid).memory_percent()
        toc = time.perf_counter_ns()
        detections = sv.Detections.from_ultralytics(result)
        self.get_logger.info(f"Memory usage: {post_mem - pre_mem}%")
        self.get_logger.info(f"Execution time: {(toc-tic)/1e6} milliseconds")
        self.mem, self.time = post_mem - pre_mem, (toc-tic)/1e6
        self.publish_result(detections)
    
    def publish_result(self, bounding_boxes):
        header = Header()
        header.metrics = f"{self.frame_id} {self.cpu} {self.mem} {self.time} {self.latency} {self.gpu} {self.gpu_mem} {self.frames} {self.fps} {self.id"
        msg = String()
        msg.data = bounding_boxes
        self.model_publisher.publish(msg)
        self.get_logger().info(msg.data)
    
    def display_metrics(self):
        self.get_logger.info(f'Frame loss: {(self.frames/self.last_frame_id):0.1f}')
        raise SystemExit

def main(args=None):
    rclpy.init(args=args)
    jetson_node = JetsonNode()
    try:
        rclpy.spin(jetson_node)
    except SystemExit:   
        rclpy.logging.get_logger("Quitting").info('Done')
    rclpy.spin(jetson_node)
    jetson_node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()