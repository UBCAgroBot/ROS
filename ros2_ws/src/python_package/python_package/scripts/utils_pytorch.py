import cv2
import os
import numpy as np
from typing import Tuple, List
import torch
import torch.nn.functional as F
import torchvision.ops as ops
import torchvision.transforms as T

# Utility class for the Python workspace
class ModelInferencePytorch:
    # Constants
    ROI_ROWS = 300
    ROI_COLS = 300
    CONTOUR_AREA_THRESHOLD = 500
    POSTPROCESS_OUTPUT_SHAPE = (640, 640)

    """
    Class that performs inference using a PyTorch YOLOv5 model.
    """
    def __init__(self, weights_path=None, precision=None):
        if weights_path is None or precision is None:
            self.model = None
            return
            
        self.precision = precision
        if not os.path.exists(weights_path):
            raise FileNotFoundError(f"Weights file not found at {weights_path}")
            
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = torch.hub.load('ultralytics/yolov5', 'custom', path=weights_path)
        self.model.to(self.device)
        if precision == "fp16":
            self.model.half()
        self.model.eval()

        # Pre-define transforms
        self.preprocess_transform = T.Compose([
            T.ToTensor(),
            T.Lambda(lambda x: x.to(self.device)),
            T.Lambda(lambda x: x.half() if precision == "fp16" else x),
            T.Resize(self.POSTPROCESS_OUTPUT_SHAPE),
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

    def preprocess(self, image: np.ndarray):
        """Optimized preprocessing using PyTorch transforms"""
        # Convert BGR to RGB
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        # Process image with pre-defined transforms
        return self.preprocess_transform(image_rgb)

    def postprocess(self, confidences, bbox_array, raw_image: np.ndarray, velocity=0):
        """Optimized postprocessing using PyTorch operations"""
        # Inputs to tensors
        boxes = torch.from_numpy(bbox_array).to(self.device)
        scores = torch.from_numpy(confidences).to(self.device)
        
        # Apply NMS
        keep_indices = ops.nms(boxes, scores, iou_threshold=0.45)
        filtered_boxes = boxes[keep_indices]
        
        # Convert to HSV color space
        image_tensor = torch.from_numpy(raw_image).to(self.device)
        image_tensor = image_tensor.permute(2, 0, 1).unsqueeze(0)  # [1, C, H, W]
        
        # Color filtering
        hsv_tensor = self._bgr_to_hsv(image_tensor)
        
        filtered_detections = []
        for box in filtered_boxes:
            x1, y1, x2, y2 = map(int, box.cpu().tolist())
            roi = hsv_tensor[:, :, y1:y2, x1:x2]
            
            # Color thresholding
            hue_mask = (roi[:, 0] > 35/180) & (roi[:, 0] < 80/180)
            sat_mask = roi[:, 1] > 0.196  # 50/255
            
            if torch.sum(hue_mask & sat_mask) > self.CONTOUR_AREA_THRESHOLD:
                filtered_detections.append([x1, y1, x2, y2])
        
        # Verify objects within ROI
        return self._verify_objects_batch(filtered_detections, raw_image, velocity)

    def _bgr_to_hsv(self, x):
        """Convert BGR tensor to HSV"""
        # Normalize to [0,1]
        x = x.float() / 255.0
        
        # Reshape for easier computation
        x = x.squeeze(0).permute(1, 2, 0)
        
        # Extract channels
        b, g, r = x[..., 0], x[..., 1], x[..., 2]
        
        max_rgb, _ = torch.max(x, dim=-1)
        min_rgb, _ = torch.min(x, dim=-1)
        diff = max_rgb - min_rgb
        
        # Calculate Hue
        hue = torch.zeros_like(max_rgb)
        mask = diff != 0
        
        # Red is max
        mask_r = mask & (max_rgb == r)
        hue[mask_r] = (60 * (g[mask_r] - b[mask_r]) / diff[mask_r]) % 360
        
        # Green is max
        mask_g = mask & (max_rgb == g)
        hue[mask_g] = 60 * (2 + (b[mask_g] - r[mask_g]) / diff[mask_g])
        
        # Blue is max
        mask_b = mask & (max_rgb == b)
        hue[mask_b] = 60 * (4 + (r[mask_b] - g[mask_b]) / diff[mask_b])
        
        hue = hue / 360  # Normalize hue to [0,1]
        
        # Calculate Saturation
        sat = torch.zeros_like(max_rgb)
        mask = max_rgb != 0
        sat[mask] = diff[mask] / max_rgb[mask]
        
        # Stack channels
        hsv = torch.stack([hue, sat, max_rgb], dim=0).unsqueeze(0)
        return hsv

    def _verify_objects_batch(self, detections, raw_image, velocity):
        """Batch verification of objects within ROI"""
        if not detections:
            return []
            
        boxes = torch.tensor(detections, device=self.device)
        roi = self.get_roi_coordinates(raw_image)
        roi_tensor = torch.tensor(roi, device=self.device)
        
        # Apply velocity shift
        shifted_roi = roi_tensor.clone()
        shifted_roi[::2] -= velocity  # Shift x coordinates
        
        # Check if boxes are within ROI
        x1, y1, x2, y2 = boxes.t()
        sx1, sy1, sx2, sy2 = shifted_roi
        
        # Create masks for valid boxes
        valid_x = (x1 >= sx1) & (x2 <= sx2)
        valid_y = (y1 >= sy1) & (y2 <= sy2)
        valid_boxes = valid_x & valid_y
        
        return boxes[valid_boxes].cpu().numpy().tolist()

    def draw_boxes(self, image: np.ndarray, bboxes: list, with_roi=True, with_roi_shift=True, velocity=0) -> np.ndarray:
        """
        Given array of bounding box tuples and an image, draw the bounding boxes into the image.
        """
        velocity = int(velocity)

        if with_roi:
            x1, y1, x2, y2 = self.get_roi_coordinates(image)
            overlay = image.copy()
            cv2.rectangle(overlay, (x1, y1), (x2, y2), (0, 128, 255), -1)
            alpha = 0.3  # Transparency factor.
            image = cv2.addWeighted(overlay, alpha, image, 1 - alpha, 0)
            cv2.putText(image, 'Original ROI', (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (100, 100, 100), 2)

            if with_roi_shift and velocity != 0:
                x1_shifted = max(x1 - velocity, 0)
                x2_shifted = max(x2 - velocity, 0)
                image = cv2.rectangle(image, (x1_shifted, y1), (x2_shifted, y2), (128, 128, 128), 2)
                overlay = image.copy()
                cv2.rectangle(overlay, (x1_shifted, y1), (x2_shifted, y2), (128, 128, 128), -1)
                alpha = 0.5  # Transparency factor
                image = cv2.addWeighted(overlay, alpha, image, 1 - alpha, 0)
                cv2.putText(image, 'Shifted ROI', (x1_shifted, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (100, 100, 100), 2)

        color = tuple(np.random.randint(0, 256, 3).tolist())  # Generate a random color

        for bbox in bboxes:
            x1, y1, x2, y2 = map(int, bbox)
            print(f"Bounding box: ({x1}, {y1}), ({x2}, {y2})")
            image = cv2.rectangle(image, (x1, y1), (x2, y2), (255, 0, 0), 2)

        return image

    def get_roi_coordinates(self, image: np.ndarray):
        """
        Calculate the coordinates of the Region of Interest (ROI) in the given image.
        """
        rows, cols = image.shape[:2]

        top_left_y = (int(rows / 2) - int(self.ROI_ROWS / 2))
        top_left_x = (int(cols / 2) - int(self.ROI_COLS / 2))

        bottom_right_y = min(rows, top_left_y + self.ROI_COLS)
        bottom_right_x = min(cols, top_left_x + self.ROI_ROWS)

        return top_left_x, top_left_y, bottom_right_x, bottom_right_y

    def print_info(self):
        """
        Prints the model information.
        """
        print(self.model)