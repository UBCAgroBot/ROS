import cv2
import os
import numpy as np
from typing import Tuple, List
import torch
from ultralytics import YOLO
from ultralytics.engine.results import Results

#utility for the python workspace
# Constants
ROI_ROWS = 300
ROI_COLS = 300
CONTOUR_AREA_THRESHOLD = 500
POSTPROCESS_OUTPUT_SHAPE = (640, 640)



def initialise_model(weights_path=None, precision=None):
    """
    create model with the given weights
    """
    if not os.path.exists(weights_path):
        print(f"weights file not found at {weights_path}")
        raise FileNotFoundError(f"weights file not found at {weights_path}")
    
    #TODO: do something with the precision
    
    yolo = YOLO(weights_path)
    return yolo

def _resize_with_padding(image: np.ndarray):
    """
    Resize the image to the desired shape and pad the image with a constant color.
    """
    new_shape = POSTPROCESS_OUTPUT_SHAPE
    padding_color = (0, 255, 255)
    original_shape = (image.shape[1], image.shape[0])
    ratio = float(max(new_shape))/max(original_shape)
    new_size = tuple([int(x*ratio) for x in original_shape])
    image = cv2.resize(image, new_size)
    delta_w = new_shape[0] - new_size[0]
    delta_h = new_shape[1] - new_size[1]
    top, bottom = delta_h//2, delta_h-(delta_h//2)
    left, right = delta_w//2, delta_w-(delta_w//2)
    preprocessed_img = cv2.copyMakeBorder(image, top, bottom, left, right, cv2.BORDER_CONSTANT, value=padding_color)

    return preprocessed_img

def preprocess(image: np.ndarray):
    """
    Takes in a numpy array that has been preprocessed 
    No pre-processing is nededed for YOLO
    """
    resized_up = cv2.resize(image, POSTPROCESS_OUTPUT_SHAPE)

    return resized_up

def run_inference(model, image_array: np.ndarray) -> Results:
    # https://docs.ultralytics.com/modes/predict/#inference-arguments
    """
    Perform inference on the provided image array using the YOLO model.
    Note that YOLO handles normalization of the bounding boxes 
    
    Args:
        image_array (np.ndarray): The input image array on which inference is to be performed.
    
    Returns:
        tuple: A tuple containing:
        - out_img (np.ndarray): The output image with bounding boxes drawn.
        - confidences (list): A list of confidence scores for each detected object.
        - boxes (list): A list of bounding boxes in the format [x1, y1, x2, y2] normalized from 0 to 1.
    """
    if model is None:
        raise Exception("Model was not initialised, inference cannot be done")
    
    results = model.predict(image_array)
    
    if results[0] is not None: # todo: change this so the message only sends bb
        result = results[0]
        out_img = cv2.cvtColor(result.plot(), cv2.COLOR_BGR2RGB)  # Convert BGR to RGB

        boxes = result.boxes.xyxyn.cpu().numpy()
        confidences = result.boxes.conf.cpu().numpy()
        return out_img, confidences, boxes
    
    return image_array, np.array([]), np.array([])

def postprocess(confidence, bbox_array: np.ndarray,raw_image: np.ndarray, velocity=0):
    """
    Postprocesses the bounding boxes to convert them from normalized coordinates (xyxyn) to pixel coordinates (xyxy).
    Applies color segmentation to refine the bounding boxes and adjusts them to fit within a shifted region of interest (ROI).

    Args:
        confidence (list): List of confidence scores for each bounding box. 2D array of shape (N, 4). (N = no. of boxes)
        bbox_array (list): List of bounding boxes in normalized coordinates (xyxyn).
        raw_image (np.ndarray): The original input image.
        velocity (int, optional): The velocity to shift the ROI. Default is 0.

    Returns:
        list: A list of refined bounding boxes in pixel coordinates (xyxy).
    """
    
    detections = _convert_bboxes_to_pixel(bbox_array, raw_image.shape)
    # detections = _object_filter(raw_image, detections) #color segmentation
    # detections = _verify_object(raw_image, detections,velocity)

    return detections

def _convert_bboxes_to_pixel(bbox_array: np.ndarray, image_shape: Tuple[int, int]) -> List[Tuple[int, int, int, int]]:
    """
    Converts normalized bounding boxes to pixel coordinates

    Args:
        bbox_array (np.ndarray): 2D Array of bounding boxes in normalized coordinates
        image_shape (Tuple[int, int]): The shape of the image

    Returns:
        List[Tuple[int, int, int, int]]: List of bounding boxes in pixel coordinates.
    """
    height, width = image_shape[:2]
    detections = []
    for bbox in bbox_array:
        x1 = int(bbox[0] * width)
        y1 = int(bbox[1] * height)
        x2 = int(bbox[2] * width)
        y2 = int(bbox[3] * height)
        detections.append((x1, y1, x2, y2))
    return detections

# creates a color segmentation mask and filters small areas within bounding box frame
def _object_filter(image: np.ndarray, bboxes: List[Tuple[int, int, int, int]]) -> List[Tuple[int, int, int, int]]:
    """
    Filters objects in an image based on bounding boxes and color thresholds.
    Args:
        image (np.ndarray): The input image in which objects are to be filtered.
        bboxes (List[Tuple[int, int, int, int]]): A list of bounding boxes.

    Returns:
        List[Tuple[int, int, int, int]]: A list of filtered bounding boxes.
    """
    detections = []
    for bbox in bboxes:
        x1, y1, x2, y2 = bbox
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
        if np.sum(mask) > CONTOUR_AREA_THRESHOLD:
            # Refine bounding boxes using contours
            gray_image = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
            gray_image = gray_image * mask.astype(np.uint8)
            _, thresh = cv2.threshold(gray_image, 0, 255, cv2.THRESH_BINARY)

            contours, _ = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
            for cnt in contours:
                area = cv2.contourArea(cnt)
                if area > CONTOUR_AREA_THRESHOLD:
                    x, y, w, h = cv2.boundingRect(cnt)
                    detections.append((x + x1, y + y1, x + w + x1, y + h + y1))
    return detections

def _verify_object(raw_image, bboxes, velocity=0):
        """
        Adjusts bounding boxes based on the region of interest (ROI) and velocity.
        This function takes an image, a list of bounding boxes, and an optional velocity parameter.
        It adjusts the bounding boxes to ensure they are within the shifted ROI
        Parameters:
            raw_image (numpy.ndarray): The input image.
            bboxes (list of list of int): A list of bounding boxes, where each bounding box is represented
                                            as [x1, y1, x2, y2].
            velocity (int, optional): The velocity to shift the ROI. Default is 0.
        Returns:
            List[List[int]]: A list of adjusted bounding boxes that are within the shifted ROI.
        """
        
        velocity = int(velocity)
        roi_x1, roi_y1, roi_x2, roi_y2 = _get_roi_coordinates(image=raw_image)
        shifted_roi_x1, shifted_roi_y1, shifted_roi_x2, shifted_roi_y2 = roi_x1 - velocity, roi_y1, roi_x2 - velocity, roi_y2

        adjusted_bboxes = []
        for bbox in bboxes:
            x1, y1, x2, y2 = bbox
            
            # Check if bounding box is within shifted ROI
            if not ((x1 < shifted_roi_x1 and x2 < shifted_roi_x1) or
                    (x1 > shifted_roi_x2 and x2 > shifted_roi_x2) or
                    (y1 < roi_y1 and y2 < roi_y1) or
                    (y1 > roi_y2 and y2 > roi_y2)):
                x1 = max(shifted_roi_x1, min(x1, shifted_roi_x2)) #set the coordinates to be inside roi
                x2 = max(shifted_roi_x1, min(x2, shifted_roi_x2))
                y1 = max(shifted_roi_y1, min(y1, shifted_roi_y2))
                y2 = max(shifted_roi_y1, min(y2, shifted_roi_y2))

                adjusted_bboxes.append([x1, y1, x2, y2])

        return adjusted_bboxes


def draw_boxes(image: np.ndarray, bboxes: list, with_roi =False, with_roi_shift = False, velocity = 0) -> np.ndarray:
    """
    Given array of bounding box tuples and an image, draw the bounding boxes into the image. 
    If with_roi and with_roi shift is set to true, the ROI areas will also be drawn in. 
    velocity must be set if roi_shift is True
    
    Parameters:
    image (np.ndarray): The image on which to draw the bounding boxes.
    bboxes (list): A list of bounding boxes, where each bounding box is represented as a list of four values [x1, y1, x2, y2].
    
    Returns:
    np.ndarray: The image with bounding boxes drawn.
    """
    velocity = int(velocity)

    if with_roi:
        x1,y1,x2,y2 = _get_roi_coordinates(image)
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
        x1, y1, x2, y2, label = map(int, bbox)


        image = cv2.rectangle(image, (x1, y1), (x2, y2), (255, 0, 0), 2)
        label = f"Object {label}"
        cv2.putText(image, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)

    
    image = _resize_with_padding(image)

    return image

def _get_roi_coordinates(image:np.ndarray):
    """
    Calculate the coordinates of the Region of Interest (ROI) in the given image.
    Args:
        image (np.ndarray): The input image as a NumPy array.
    Returns:
        list: A list containing the coordinates of the top-left and bottom-right corners of the ROI 
            in the format [top_left_x, top_left_y, bottom_right_x, bottom_right_y].
    """
    rows, cols = image.shape[:2]

    top_left_y = (int(rows/2) - int(ROI_ROWS/2))
    top_left_x = (int(cols/2) - int(ROI_COLS/2))

    bottom_right_y = min(rows, top_left_y + ROI_COLS)
    bottom_right_x = min(cols, top_left_x + ROI_ROWS)

    return top_left_x, top_left_y, bottom_right_x, bottom_right_y

def print_info(model):
    """
    Prints the model information
    """
    print(model.info())