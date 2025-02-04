import numpy as np
import cv2

def object_filter(self, image, bboxes):
    detections = []
    h, w, _ = image.shape

    # Create a mask for all ROIs
    for bbox in bboxes:
        x1, y1, w, h = bbox
        x1, y1, x2, y2 = x1, y1, x1 + w, y1 + h  # Adjust to x2, y2 format
        
        # Ensure the ROI is within image bounds
        roi = image[max(0, y1):min(h, y1 + h), max(0, x1):min(w, x1 + w)] 
        hsv_roi = cv2.cvtColor(roi, cv2.COLOR_RGB2HSV)

        # Apply color segmentation mask
        mask = cv2.inRange(hsv_roi, tuple(self.lower_range), tuple(self.upper_range))
        _, mask = cv2.threshold(mask, 254, 255, cv2.THRESH_BINARY)

        contours, _ = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

        # Process contours in a vectorized manner
        areas = np.array([cv2.contourArea(cnt) for cnt in contours])
        valid_contours = contours[areas > self.min_area]

        for cnt in valid_contours:
            x, y, w, h = cv2.boundingRect(cnt)
            detections.append((x, y, x + w, y + h))

    self.verify_object(image, detections)

def verify_object(self, disp_image, bboxes):
    roi_x1, roi_y1, roi_x2, roi_y2 = self.roi_list
    original_height, original_width = original_image_shape
    model_height, model_width = model_dimensions

    shifted_x = roi_x + abs(velocity[0]) * shift_constant
    scale_x = roi_w / model_width
    scale_y = roi_h / model_height

    adjusted_bboxes = []
    
    # Convert bboxes to a NumPy array for vectorized operations
    bboxes_array = np.array(bboxes)

    # Reverse the resize operation
    x_min = (bboxes_array[:, 0] * scale_x).astype(int)
    y_min = (bboxes_array[:, 1] * scale_y).astype(int)
    x_max = (bboxes_array[:, 2] * scale_x).astype(int)
    y_max = (bboxes_array[:, 3] * scale_y).astype(int)

    # Reverse the cropping operation
    x_min += shifted_x
    y_min += roi_y
    x_max += shifted_x
    y_max += roi_y

    # Ensure the bounding box doesn't exceed the original image dimensions
    x_min = np.clip(x_min, 0, original_width)
    y_min = np.clip(y_min, 0, original_height)
    x_max = np.clip(x_max, 0, original_width)
    y_max = np.clip(y_max, 0, original_height)

    adjusted_bboxes = np.vstack((x_min, y_min, x_max, y_max)).T.tolist()

    # Check if adjusted bounding boxes are within the ROI
    for bbox in adjusted_bboxes:
        if bbox[0] >= roi_x1 and bbox[2] <= roi_x2 and bbox[1] >= roi_y1 and bbox[3] <= roi_y2:
            self.on = 1
