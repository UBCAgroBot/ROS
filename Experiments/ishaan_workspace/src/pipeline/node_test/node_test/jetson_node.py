import time
import os
import psutil

import numpy as np
import cv2

import tensorrt as trt
import pycuda.autoinit
import pycuda.driver as cuda

# from numba import jit
# argparse for choosing model

import rclpy
from rclpy.time import Time
from rclpy.node import Node
from std_msgs.msg import Header, String
from sensor_msgs.msg import Image
# from msg import BoundingBox
from cv_bridge import CvBridge, CvBridgeError

class JetsonNode(Node):
    def __init__(self):
        super().__init__('jetson_node') #type:ignore
        self.bridge = CvBridge()
        # self.inference = BoundingBox()
        self.image = cv2.cuda_GpuMat()
        
        # self.model_publisher = self.create_publisher(BoundingBox, 'bounding_boxes', 10)
        self.camera_subscriber = self.create_subscription(Image, 'image_data', self.callback, 10)
        self.camera_subscriber
        self.frames, self.cpu, self.mem, self.time, self.latency, self.pid, self.frame_id, self.save = 0, 0, 0, 0, 0, 0, 0, True
        self.preprocessing_time, self.inferencing_time, self.postprocessing_time = 0, 0, 0
        self.tensorrt_init()
    
    def tensorrt_init(self):
        try:
            path = "/home/user/AppliedAI/23-I-12_SysArch/Experiments/ishaan_workspace/src/pipeline/node_test/node_test"
            # print(path)
            os.chdir(path)
            # print(os.getcwd())
            # Create a TensorRT logger
            TRT_LOGGER = trt.Logger(trt.Logger.WARNING)

            # Create a TensorRT builder
            builder = trt.Builder(TRT_LOGGER)
            
            # Create a TensorRT network
            network = builder.create_network(flags=trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)
            
            # Load the ONNX model into the network
            with trt.OnnxParser(network, TRT_LOGGER) as parser:
                with open('yolov8x.onnx', 'rb') as model:
                    parser.parse(model.read())
            
            # If the network has no layers, print an error message and exit
            if network.num_layers == 0:
                self.get_logger().info("Error: The network has no layers")
                raise SystemExit

            # If the network has no output layers, manually set the last layer as the output
            if num_outputs == 0:
                last_layer = network.get_layer(network.num_layers - 1)
                network.mark_output(last_layer.get_output(0))

            # Create a builder config
            config = builder.create_builder_config()

            # Build the TensorRT engine
            self.model = builder.build_engine(network, config)
            
        except Exception as e:
            self.get_logger().info(f"Error: {e}")
            raise SystemExit
        finally:
            self.get_logger().info("Model loaded successfully")
    
    def callback(self, msg):
        now = self.get_clock().now()
        self.get_logger().info(f"Received: {msg.header.frame_id}")
        latency = now - Time.from_msg(msg.header.stamp)
        print(f"Latency: {latency.nanoseconds / 1e6} milliseconds")
        
        try:
            image  = self.bridge.imgmsg_to_cv2(msg, desired_encoding='rgb8')
        except CvBridgeError as e:
            print(e)
        
        self.latency, self.frame_id, self.frames = latency.nanoseconds / 1e6, msg.header.frame_id, self.frames + 1
        self.preprocessing(image)
    
    def preprocessing(self, image):
        tic = time.perf_counter_ns()
        
        image_gpu = self.image
        # Convert the numpy array to a CUDA GPU Mat
        image_gpu.upload(image)
        # Resize and normalize the image
        image_gpu = cv2.cuda.resize(image_gpu, (1920, 1088)) 
        # Convert the image to float32
        image_gpu = image_gpu.transpose((2, 0, 1)).astype(np.float32)
        image_gpu = np.expand_dims(image_gpu, axis=0)
        # Transpose the image
        image_gpu = cv2.cuda.transpose(image_gpu)
        # Add a new dimension to the image
        image_gpu = cv2.cuda_GpuMat((1,) + image_gpu.size(), image_gpu.type())
        cv2.cuda.copyMakeBorder(image_gpu, 0, 1, 0, 0, cv2.BORDER_CONSTANT, image_gpu)

        toc = time.perf_counter_ns()
        self.preprocessing_time = (toc - tic)/1e6
        self.detection(image_gpu)

    def detection(self, image_gpu):
        # pid = os.getpid()
        # print(pid)
        # Allocate device memory for the input and output data
        d_output = cuda.mem_alloc(1 * self.engine.get_binding_shape(0).volume() * np.dtype(np.float32).itemsize)

        # Execute the engine
        # pre_mem = psutil.Process(pid).memory_percent()
        # pre_cpu = psutil.Process(pid).cpu_percent(interval=None)
        tic = time.perf_counter_ns()
        self.context.execute(bindings=[int(image_gpu.ptr), int(d_output)])
        # post_cpu = psutil.Process(pid).cpu_percent(interval=None)
        # post_mem = psutil.Process(self.pid).memory_percent()
        toc = time.perf_counter_ns()

        # Copy the output data back to the host
        output = np.empty(self.engine.get_binding_shape(0), dtype=np.float32)
        cuda.memcpy_dtoh(output, d_output)
        
        # self.cpu = ((self.post_cpu-self.pre_cpu)/self.time) * 100
        # self.mem = post_mem - pre_mem
        # self.inferencing_time = (toc-tic)/1e6
        # self.gpu = 0 # gpu.load*100
        # self.gpu_mem = 0 # (gpu.memoryUsed / gpu.memoryTotal) * 100
        
        # self.get_logger().info(f"CPU usage: {self.cpu}%")
        # self.get_logger().info(f"GPU usage: {self.gpu}%")
        # self.get_logger().info(f"GPU VRAM usage: {self.gpu_mem}%")
        # self.get_logger().info(f"Memory usage: {self.mem}%")
        # self.get_logger().info(f"Execution time: {self.time} milliseconds")
        self.postprocessing(output)
        self.output.clear()
        
    
    def postprocessing(self, output):
        tic = time.perf_counter_ns()
        
        # Reshape the output tensor to a 2D array
        output = output.reshape(-1, 7)
        # control over confidence, min/max confidence, etc.
        # Split the output array into boxes, objectness scores, and class scores
        boxes = output[:, :4]  # Bounding box coordinates
        scores = output[:, 4]  # Objectness scores
        classes = output[:, 5:] # Class labels
        
        # Compute the class predictions
        classes = np.argmax(classes, axis=1)
        
        # keep = nms(boxes, scores, iou_threshold=0.5)  # Indices of boxes to keep after non-maximum suppression
        # self.publish_result(boxes[keep], scores[keep], classes[keep])
        # self.publish_result(boxes, scores, classes)
        
        toc = time.perf_counter_ns()
        self.postprocessing_time = (toc-tic)/1e6
        self.time = self.preprocessing_time + self.inferencing_time + self.postprocessing_time
        
    def publish_result(self, boxes, scores, classes):
        msg = BoundingBox()
        msg.box = boxes
        msg.score = scores
        msg.classes = classes
    
        header = Header()
        header.metrics = f"{self.frame_id} {self.cpu} {self.mem} {self.time} {self.latency} {self.gpu} {self.gpu_mem} {self.frames} {self.fps} {self.id}"
        
        self.get_logger().info(msg.data)
        self.publisher.publish(msg)
    
    def display_metrics(self):
        self.get_logger().info(f'Frame loss: {((self.frames/self.frame_id)*100):0.1f}')
        raise SystemExit


# multithreading:  post processing/publisher node can run seperately than pre and inference nodes
# def main(args=None):
#     rclpy.init(args=args)
#     image_subscriber = Image_subscriber()
#     yolo_subscriber = Yolo_subscriber()
    
#     executor = rclpy.executors.MultiThreadedExecutor()
#     executor.add_node(image_subscriber)
#     executor.add_node(yolo_subscriber)
    
#     executor_thread = threading.Thread(target=executor.spin, daemon=True)
#     executor_thread.start()
#     rate = yolo_subscriber.create_rate(2) #idk what this does
    
#     try:
#         while rclpy.ok():
#             rate.sleep()
#     except SystemExit:    
#         yolo_subscriber.display_metrics()
#         rclpy.logging.get_logger("Quitting").info('Done')

#     rclpy.shutdown()
#     executor_thread.join()

# if __name__ == '__main__':
#     main()


def main(args=None):
    rclpy.init(args=args)
    jetson_node = JetsonNode()
    try:
        rclpy.spin(jetson_node)
    except KeyboardInterrupt:
        print("qq...")
        jetson_node.display_metrics()
        torch._dynamo.reset()
        rclpy.logging.get_logger("Quitting").info('Done')
    except SystemExit:   
        print("qqq...")
        jetson_node.display_metrics()
        torch._dynamo.reset()
        rclpy.logging.get_logger("Quitting").info('Done')

    rclpy.spin(jetson_node)
    jetson_node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
    
    