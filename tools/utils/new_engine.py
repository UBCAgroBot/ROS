import os
import cv2

import tensorrt as trt
import pycuda.driver as cuda

import numpy as np
import cupy as cp
import warnings 
import time
import numpy as np
import torch

# strip_weights, precision

class TRTEngine:
    def __init__(self, trt_path, engine=None):
        warnings.filterwarnings("ignore")
        cuda.init()
        device = cuda.Device(0)
        self.cuda_driver_context = device.make_context()
        
        self.TRT_LOGGER = trt.Logger(trt.Logger.WARNING)
        self.engine = engine
        self.trt_path = trt_path
        
        self.engine, self.context = self.load_normal_engine()
        
        self.stream = cuda.Stream()
        self.input_shape = (self.engine).get_binding_shape(0)
        self.output_shape = (self.engine).get_binding_shape(1)
        
        self.d_input = cuda.mem_alloc(trt.volume(self.input_shape) * cp.dtype(cp.float32).itemsize) # change to fp16, etc.
        self.d_output = cuda.mem_alloc(trt.volume(self.output_shape) * cp.dtype(cp.float32).itemsize) 
        
        # Allocate host pinned memory for input/output (pinned memory for input/output buffers)
        self.h_input = cuda.pagelocked_empty(trt.volume(self.input_shape), dtype=np.float32)
        self.h_output = cuda.pagelocked_empty(trt.volume(self.output_shape), dtype=np.float32)
        
        self.confidence_threshold = 0.5
        self.iou_threshold = 0.9
        self.input_size = 640
        
        self.load_normal_engine()
        self.allocate_buffers()
        self.warmup()
        
    def load_normal_engine(self):
        if self.engine is None:
            assert self.trt_path is not None
            assert os.path.exists(self.trt_path)
            
            with open(self.trt_path, 'rb') as f, trt.Runtime(self.TRT_LOGGER) as runtime:
                engine = runtime.deserialize_cuda_engine(f.read())
            
            context = engine.create_execution_context()
            
        self.get_logger().info(f"Successfully loaded engine from {self.engine_path}")
        return engine, context
        
    def warmup(self, loop=20):
        input_shape = self.input_shape
        shape = (1, 3, self.input_size, self.input_size)
        image = np.zeros(shape, dtype='float32')
        
        for _ in range(loop):
            random_input = np.random.randn(*input_shape).astype(np.float32)
            cuda.memcpy_htod(self.d_input, random_input)
            self.context.execute(bindings=[int(self.d_input), int(self.d_output)])

        self.get_logger().info(f"Engine warmed up with 20 inference passes.")

# execute_async_v2? -> check inference examples

    # time the function
    def __call__(self, image):
        image, scale = self.resize(image, self.input_size)
        # image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = image.transpose((2, 0, 1))[::-1]
        image = image.astype('float32') / 255
        image = image[np.newaxis, ...]

        tic = time.perf_counter_ns()
        cuda.memcpy_htod(self.d_input, image)
        self.context.execute(bindings=[int(self.d_input), int(self.d_output)])
        cuda.memcpy_dtoh(self.h_output, self.d_output)
        toc = time.perf_counter_ns()
        
        duration = (toc - tic) / 1e6
        
        outputs = self.h_output
        outputs = np.transpose(np.squeeze(outputs[0]))

        # Lists to store the bounding boxes, scores, and class IDs of the detections
        boxes = []
        scores = []
        class_indices = []

        # Iterate over each row in the outputs array
        for i in range(outputs.shape[0]):
            # Extract the class scores from the current row
            classes_scores = outputs[i][4:]

            # Find the maximum score among the class scores
            max_score = np.amax(classes_scores)

            # If the maximum score is above the confidence threshold
            if max_score >= self.confidence_threshold:
                # Get the class ID with the highest score
                class_id = np.argmax(classes_scores)

                # Extract the bounding box coordinates from the current row
                image, y, w, h = outputs[i][0], outputs[i][1], outputs[i][2], outputs[i][3]

                # Calculate the scaled coordinates of the bounding box
                left = int((image - w / 2) / scale)
                top = int((y - h / 2) / scale)
                width = int(w / scale)
                height = int(h / scale)

                # Add the class ID, score, and box coordinates to the respective lists
                class_indices.append(class_id)
                scores.append(max_score)
                boxes.append([left, top, width, height])
        
        # Apply non-maximum suppression to filter out overlapping bounding boxes
        indices = cv2.dnn.NMSBoxes(boxes, scores, self.confidence_threshold, self.iou_threshold)
        
        # convert to xyxy

        # Iterate over the selected indices after non-maximum suppression
        confidences = []
        bboxes = []
        for i in indices:
            x, y, w, h = boxes[i]
            x1, y1 = x, y
            x2, y2 = x + w, y + h
            bboxes.append([x1, y1, x2, y2])
            confidences.append(scores[i])
        return confidences, bboxes

    @staticmethod
    def resize(image, input_size):
        shape = image.shape

        ratio = float(shape[0]) / shape[1]
        if ratio > 1:
            h = input_size
            w = int(h / ratio)
        else:
            w = input_size
            h = int(w * ratio)
        scale = float(h) / shape[0]
        resized_image = cv2.resize(image, (w, h))
        det_image = np.zeros((input_size, input_size, 3), dtype=np.uint8)
        det_image[:h, :w, :] = resized_image
        return det_image, scale