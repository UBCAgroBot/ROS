import cv2
import os
import numpy as np
import cupy as cp
from typing import Tuple, List
from ultralytics import YOLO

class ModelInferenceCupy:
    ROI_ROWS = 300
    ROI_COLS = 300
    CONTOUR_AREA_THRESHOLD = 500
    POSTPROCESS_OUTPUT_SHAPE = (640, 640)

    def __init__(self, weights_path=None, precision=None):
        self.yolo = YOLO(weights_path) if weights_path else None
        self.precision = precision

    def preprocess(self, image: np.ndarray):
        # Transfer image to GPU
        gpu_image = cp.asarray(image)
        
        # Resize w CuPy
        resized = cv2.resize(cp.asnumpy(gpu_image), self.POSTPROCESS_OUTPUT_SHAPE)
        return cp.asarray(resized)

    def _convert_bboxes_to_pixel(self, bbox_array: np.ndarray, image_shape: Tuple[int, int]):
        height, width = image_shape[:2]
        bbox_gpu = cp.asarray(bbox_array)
        
        # Vectorized conversion w CuPy
        x1 = cp.floor(bbox_gpu[:, 0] * width).astype(cp.int32)
        y1 = cp.floor(bbox_gpu[:, 1] * height).astype(cp.int32)
        x2 = cp.ceil(bbox_gpu[:, 2] * width).astype(cp.int32)
        y2 = cp.ceil(bbox_gpu[:, 3] * height).astype(cp.int32)
        
        return cp.stack([x1, y1, x2, y2], axis=1).get()

    def object_filter(self, image: np.ndarray, bboxes: List[Tuple[int, int, int, int]]):
        gpu_image = cp.asarray(image)
        detections = []
        
        for bbox in bboxes:
            x1, y1, x2, y2 = bbox
            roi = gpu_image[y1:y2, x1:x2]
            
            # Convert to HSV 
            roi_cpu = cp.asnumpy(roi)  # Temporarily transfer for cv2
            hsv = cp.asarray(cv2.cvtColor(roi_cpu, cv2.COLOR_BGR2HSV))
            
            # Color segmentation
            lower_mask = hsv[:, :, 0] > 35
            upper_mask = hsv[:, :, 0] < 80
            saturation_mask = hsv[:, :, 1] > 50
            mask = lower_mask & upper_mask & saturation_mask
            
            if cp.sum(mask) > self.CONTOUR_AREA_THRESHOLD:
                # Processing the contours
                gray_image = cv2.cvtColor(cp.asnumpy(roi), cv2.COLOR_BGR2GRAY)
                gray_image = gray_image * cp.asnumpy(mask).astype(np.uint8)
                _, thresh = cv2.threshold(gray_image, 0, 255, cv2.THRESH_BINARY)
                
                contours, _ = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
                for cnt in contours:
                    area = cv2.contourArea(cnt)
                    if area > self.CONTOUR_AREA_THRESHOLD:
                        x, y, w, h = cv2.boundingRect(cnt)
                        detections.append((x + x1, y + y1, x + w + x1, y + h + y1))
        
        return detections

    def postprocess(self, confidence, bbox_array, raw_image: np.ndarray, velocity=0):
        detections = self._convert_bboxes_to_pixel(bbox_array, raw_image.shape)
        detections = self.object_filter(raw_image, detections)
        return self.verify_object(raw_image, detections, velocity)