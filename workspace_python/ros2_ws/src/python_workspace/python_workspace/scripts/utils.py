import cv2
import time, os
import tensorrt as trt
import pycuda.driver as cuda
import cupy as cp
import numpy as np
from sensor_msgs.msg import Image
from ultralytics import YOLO
from ultralytics.engine.results import Results


#utility class for the python workspace


class ModelInference:
    """
    class that performs inference using a model
    """
    def __init__(self, weights_path, precision):

        # Initialize instance variables
        self.precision = precision

        if not os.path.exists(weights_path):
            print(f"weights file not found at {weights_path}")
            raise FileNotFoundError(f"weights file not found at {weights_path}")
        
        self.yolo = YOLO(weights_path)

    def preprocess(self, image: np.ndarray):
        """
        Takes in a numpy array that has been preprocessed 
        No preprocessing is nededed for YOLO
        """
        return image


    def inference(self, image_array: np.ndarray) -> Results:
        # https://docs.ultralytics.com/modes/predict/#inference-arguments
        """
        perform inference on the image array. 
        Returns list of predicted classes, their confidences and bounding boxes
        """
        height, width, _ = image_array.shape
        size = max(height, width)
        results =self.yolo.predict(image_array,imgsz=size) #we are only predicting one image


        if results[0] is not None: # todo: change this so the message only sends bb
            out_img, named_classes, confidences, boxes = self.get_attributes(results[0])

            return out_img,named_classes, confidences, boxes
        
        return image_array, [], [], []

    def get_attributes(self, result:Results) -> np.ndarray:
        """
        postprocess the Result that we got from prediction
        returns the image with bounding boxes
        class names, confidences and bounding boxes

        Since YOLO does this for us we just need to convert the BGR image to RGB.
        """
        out_img = cv2.cvtColor(result.plot(), cv2.COLOR_BGR2RGB)  # Convert BGR to RGB

        class_mapping = self.yolo.model.names
        named_classes = [class_mapping[int(i)] for i in result.boxes.cls]
        confidences = result.boxes.conf
        print(f"Bounding boxes (xyxy format): {result.boxes.xyxy}")
        boxes = result.boxes.xyxy.flatten().tolist()

        
        return out_img,named_classes, confidences, boxes
    
    def postprocess(self, image: np.ndarray):
        """
        Takes in a numpy array that has been preprocessed 
        No preprocessing is nededed for YOLO
        """
        return image
    
    def print_info(self):
        print(self.yolo.info())