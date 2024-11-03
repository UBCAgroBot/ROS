import cv2
import os
import numpy as np
from typing import Tuple, List
import torch

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
        else:
            self.precision = precision
            if not os.path.exists(weights_path):
                print(f"Weights file not found at {weights_path}")
                raise FileNotFoundError(f"Weights file not found at {weights_path}")

            # Load the YOLOv5 model
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            self.model = torch.hub.load('ultralytics/yolov5', 'custom', path=weights_path)
            self.model.to(self.device)
            self.model.eval()

            # YOLOv5 handles preprocessing internally

    def preprocess(self, image: np.ndarray):
        """
        Preprocesses the input image for the YOLOv5 model.
        """
        # Pytorch expects RGB input, convert image from BGR (OpenCV format) to RGB
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        return image

    def inference(self, image_array: np.ndarray):
        """
        Perform inference on the provided image array using the YOLOv5 model.

        Returns:
            tuple: A tuple containing:
            - out_img (np.ndarray): The output image with bounding boxes drawn.
            - confidences (np.ndarray): An array of confidence scores for each detected object.
            - boxes (np.ndarray): An array of bounding boxes in the format [x1, y1, x2, y2].
        """
        if self.model is None:
            raise Exception("Model was not initialized; inference cannot be done.")

        # Preprocess the image
        image = self.preprocess(image_array)

        # Run inference
        results = self.model(image)

        # Extract bounding boxes, confidences, and class labels
        detections = results.xyxy[0]  # detections on the first image
        # Each detection is (x1, y1, x2, y2, confidence, class)

        if detections is not None and len(detections) > 0:
            detections = detections.cpu().numpy()

            boxes = detections[:, :4]       # x1, y1, x2, y2
            confidences = detections[:, 4]
            class_ids = detections[:, 5]

            # Draw bounding boxes on the image
            out_img = image_array.copy()
            for box in boxes:
                x1, y1, x2, y2 = box.astype(int)
                cv2.rectangle(out_img, (x1, y1), (x2, y2), (255, 0, 0), 2)

            return out_img, confidences, boxes
        else:
            return image_array, np.array([]), np.array([])

    def postprocess(self, confidences, bbox_array, raw_image: np.ndarray, velocity=0):
        """
        Postprocesses the bounding boxes to apply color segmentation
        and adjusts them to fit within a shifted region of interest (ROI).

        Returns:
            list: A list of refined bounding boxes in pixel coordinates (x1, y1, x2, y2).
        """
        detections = self.object_filter(raw_image, bbox_array)  # Color segmentation
        detections = self.verify_object(raw_image, detections, velocity)
        return detections

    # The rest of the methods remain the same as your original code

    def object_filter(self, image: np.ndarray, bboxes: List[Tuple[int, int, int, int]]) -> List[Tuple[int, int, int, int]]:
        """
        Filters objects in an image based on bounding boxes and color thresholds.
        """
        detections = []
        for bbox in bboxes:
            x1, y1, x2, y2 = bbox.astype(int)
            roi = image[y1:y2, x1:x2]
            
            # Perform color segmentation
            hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
            lower_hue = 35
            upper_hue = 80
            lower_mask = hsv[:, :, 0] > lower_hue
            upper_mask = hsv[:, :, 0] < upper_hue
            saturation_mask = hsv[:, :, 1] > 50
            mask = lower_mask & upper_mask & saturation_mask

            # Check if the mask has significant area
            if np.sum(mask) > self.CONTOUR_AREA_THRESHOLD:
                # Refine using contours
                gray_image = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
                gray_image = gray_image * mask.astype(np.uint8)
                _, thresh = cv2.threshold(gray_image, 0, 255, cv2.THRESH_BINARY)

                contours, _ = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
                for cnt in contours:
                    area = cv2.contourArea(cnt)
                    if area > self.CONTOUR_AREA_THRESHOLD:
                        x, y, w, h = cv2.boundingRect(cnt)
                        detections.append((x + x1, y + y1, x + w + x1, y + h + y1))
        return detections

    def verify_object(self, raw_image, bboxes, velocity=0):
        """
        Adjusts bounding boxes based on the region of interest (ROI) and velocity.
        """
        velocity = int(velocity)
        roi_x1, roi_y1, roi_x2, roi_y2 = self.get_roi_coordinates(image=raw_image)
        shifted_roi_x1, shifted_roi_y1, shifted_roi_x2, shifted_roi_y2 = roi_x1 - velocity, roi_y1, roi_x2 - velocity, roi_y2

        adjusted_bboxes = []
        for bbox in bboxes:
            x1, y1, x2, y2 = bbox

            # Check if bounding box is within shifted ROI
            if not ((x1 < shifted_roi_x1 and x2 < shifted_roi_x1) or
                    (x1 > shifted_roi_x2 and x2 > shifted_roi_x2) or
                    (y1 < roi_y1 and y2 < roi_y1) or
                    (y1 > roi_y2 and y2 > roi_y2)):
                x1 = max(shifted_roi_x1, min(x1, shifted_roi_x2))  # Set the coordinates to be inside ROI
                x2 = max(shifted_roi_x1, min(x2, shifted_roi_x2))
                y1 = max(shifted_roi_y1, min(y1, shifted_roi_y2))
                y2 = max(shifted_roi_y1, min(y2, shifted_roi_y2))

                adjusted_bboxes.append([x1, y1, x2, y2])

        return adjusted_bboxes

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