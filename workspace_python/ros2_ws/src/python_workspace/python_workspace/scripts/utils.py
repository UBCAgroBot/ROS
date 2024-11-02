import cv2
import os
import numpy as np
import torch
from ultralytics import YOLO
from ultralytics.engine.results import Results

#utility class for the python workspace
class ModelInference:
    ROI_ROWS = 300
    ROI_COLS = 300
    CONTOUR_AREA_THRESHOLD = 500
    POSTPROCESS_OUTPUT_SHAPE = (640, 640)

    """
    class that performs inference using a model
    """
    def __init__(self, weights_path=None, precision=None):
        if weights_path is None or precision is None:
            self.yolo = None
            return
        # Initialize instance variables
        else:
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
        resized_up = cv2.resize(image, self.POSTPROCESS_OUTPUT_SHAPE)

        return resized_up
    
    def initialise_model(self, weights_path):
        """
        Initialize the YOLO model with the given weights.
        Args:
            weights_path (str): The file path to the YOLO weights.
        Raises:
            FileNotFoundError: If the weights file does not exist at the specified path.
        Returns:
            None
        """

        if not os.path.exists(weights_path):
            print(f"weights file not found at {weights_path}")
            raise FileNotFoundError(f"weights file not found at {weights_path}")
        
        self.yolo = YOLO(weights_path)

    def inference(self, image_array: np.ndarray) -> Results:
        # https://docs.ultralytics.com/modes/predict/#inference-arguments
        """
        Perform inference on the provided image array using the YOLO model.
        
        Args:
            image_array (np.ndarray): The input image array on which inference is to be performed.
        
        Returns:
            tuple: A tuple containing:
            - out_img (np.ndarray): The output image with bounding boxes drawn.
            - confidences (list): A list of confidence scores for each detected object.
            - boxes (list): A list of bounding boxes in the format [x1, y1, x2, y2] normalized from 0 to 1.
        """
        if self.yolo is None:
            raise Exception("Model was not initialised, inference cannot be done")
        height, width, _ = image_array.shape
        size = max(height, width)
        results =self.yolo.predict(image_array,imgsz=size) #we are only predicting one image


        if results[0] is not None: # todo: change this so the message only sends bb
            result = results[0]
            out_img = cv2.cvtColor(result.plot(), cv2.COLOR_BGR2RGB)  # Convert BGR to RGB

            # print(f"Bounding boxes (xyxyn format): {result.boxes.xyxyn}")
            boxes = np.array(result.boxes.xyxyn.tolist()).reshape(-1)
            confidences = np.array(result.boxes.conf.tolist()).reshape(-1)
            return out_img, list(confidences), list(boxes)
        
        return image_array, [], []

    def postprocess(self, confidence, bbox_array,raw_image: np.ndarray, velocity=0):
        """
        Postprocesses the bounding boxes to convert them from normalized coordinates (xyxyn) to pixel coordinates (xyxy).
        Applies color segmentation to refine the bounding boxes and adjusts them to fit within a shifted region of interest (ROI).

        Args:
            confidence (list): List of confidence scores for each bounding box.
            bbox_array (list): List of bounding boxes in normalized coordinates (xyxyn).
            raw_image (np.ndarray): The original input image.
            velocity (int, optional): The velocity to shift the ROI. Default is 0.

        Returns:
            list: A list of refined bounding boxes in pixel coordinates (xyxy).
        """
        velocity = int(velocity)
        detections = []
        if len(bbox_array) > 0 and type(bbox_array[0]) != tuple: #CONVERT TO TUPLE
            for offset in range(int(len(bbox_array) / 4)):
                x1 = int(bbox_array[offset * 4] * raw_image.shape[1])
                y1 = int(bbox_array[offset * 4 + 1] * raw_image.shape[0])
                x2 = int(bbox_array[offset * 4 + 2] * raw_image.shape[1])
                y2 = int(bbox_array[offset * 4 + 3] * raw_image.shape[0])
                detections.append((x1, y1, x2, y2))
        else:
            for bbox in bbox_array: #resize so it first the original image
                x1,y1,x2,y2 = bbox

                x1 = int(x1 * raw_image.shape[1])
                y1 = int(y1 * raw_image.shape[0])
                x2 = int(x2 * raw_image.shape[1])
                y2 = int(y2 * raw_image.shape[0])

                detections.append((x1, y1, x2, y2))


        detections = self.object_filter(raw_image, detections) #color segmentation
        detections = self.verify_object(raw_image, detections,velocity)

        return detections


    # creates a color segmentation mask and filters small areas within bounding box frame
    def object_filter(self, image, bboxes):
        """
        Uses Color segmentation to create better boxes
        Filters objects in an image based on bounding boxes and color thresholds.
        Args:
            image (numpy.ndarray): The input image in which objects are to be filtered.
            bboxes (list of tuples): A list of bounding boxes, where each bounding box is represented 
                                     as a tuple of four integers (x1, y1, x2, y2).
        Returns:
            list of tuples: A list of filtered bounding boxes, where each bounding box is represented 
                            as a tuple of four integers (x1, y1, x2, y2) corresponding to the coordinates 
                            of the top-left and bottom-right corners.
        """
        
        detections = []
        for bbox in bboxes:
            x1, y1, x2, y2 = map(int, bbox)

            roi = image[y1:y2, x1:x2] # Extract Region of interest (bounding box)
            bbox_offset_y = y1
            bbox_offset_x = x1

            roi = image[y1:y2,x1:x2,:] # Extract Region of interest (bounding box)

            hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV) #convrt opencv bgr to hsv

            lower_mask = hsv[:,:,0] > 35 #refer to hue channel (in the colorbar)
            upper_mask = hsv[:,:,0] < 80 #refer to transparency channel (in the colorbar)
            saturation_mask = hsv[:,:,1] >50

            mask = upper_mask*lower_mask*saturation_mask
            blue = roi[:,:,0]*mask
            green = roi[:,:,1]*mask
            red = roi[:,:,2]*mask

            masked_brg = np.dstack((blue,green,red))
            gray_image = cv2.cvtColor(masked_brg, cv2.COLOR_BGR2GRAY)


            ret, thresh = cv2.threshold(gray_image, 0, 255, cv2.THRESH_BINARY)

            contours, hiearchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)


            for cnt in contours:
                area = cv2.contourArea(cnt)
                if area > self.CONTOUR_AREA_THRESHOLD:
                    print(area)
                    x, y, w, h = cv2.boundingRect(cnt)
                    detections.append((x + bbox_offset_x, y + bbox_offset_y, x + w + bbox_offset_x, y + h + bbox_offset_y))
        
        return detections
    

    def verify_object(self, raw_image, bboxes, velocity=0):
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
            list of list of int: A list of adjusted bounding boxes that are within the shifted ROI.
            """
            
            velocity = int(velocity)
            roi_x1,roi_y1,roi_x2,roi_y2 = self.get_roi_coordinates(image=raw_image)
            shifted_roi_x1 = roi_x1 - velocity
            shifted_roi_y1 = roi_y1
            shifted_roi_x2 = roi_x2 - velocity
            shifted_roi_y2 = roi_y2

            width = raw_image.shape[1]
            height = raw_image.shape[0]

            adjusted_bboxes = []
            for bbox in bboxes:
                x1, y1, x2, y2 = bbox
                # #     # Ensure the bounding box doesn't exceed the original image dimensions
                # x1 = max(0, min(x1, width))
                # x2 = max(0, min(x2, width))
                # y1 = max(0, min(y1, height))
                # y2 = max(0, min(y2, height))

                if ((x1 < shifted_roi_x1 and x2 < shifted_roi_x1) #remove boxes that are out of point
                or (x1 > shifted_roi_x2 and x2 > shifted_roi_x2)
                or (y1 < shifted_roi_y1 and y2 < shifted_roi_y1)
                or (y1 > shifted_roi_y2 and y2 > shifted_roi_y2)
                ):
                    pass
                else:
                    x1 = max(shifted_roi_x1, min(x1, shifted_roi_x2)) #set the coordinates to be inside roi
                    x2 = max(shifted_roi_x1, min(x2, shifted_roi_x2))
                    y1 = max(shifted_roi_y1, min(y1, shifted_roi_y2))
                    y2 = max(shifted_roi_y1, min(y2, shifted_roi_y2))

                    adjusted_bboxes.append([x1, y1, x2, y2])

            return adjusted_bboxes
    
    
    def draw_boxes(self, image: np.ndarray, bboxes: list, with_roi =True, with_roi_shift = True, velocity = 0) -> np.ndarray:
        """
        Given array of bounding box tuples and an image, draw the bouding boxes into the image. 
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
            x1,y1,x2,y2 = self.get_roi_coordinates(image)
            overlay = image.copy()
            cv2.rectangle(overlay, (x1, y1), (x2, y2), (0, 128, 255), -1)
            alpha = 0.3  # Transparency factor.
            image = cv2.addWeighted(overlay, alpha, image, 1 - alpha, 0)
            cv2.putText(image, 'Original ROI', (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (100, 100, 100), 2)

            if with_roi_shift and velocity != 0:
                x1_shifted = x1 - velocity if x1 > velocity else 0
                x2_shifted = x2 - velocity if x2 > velocity else 0
                image = cv2.rectangle(image, (x1_shifted, y1), (x2_shifted, y2), (128, 128, 128), 2)
                overlay = image.copy()
                cv2.rectangle(overlay, (x1_shifted, y1), (x2_shifted, y2), (128, 128, 128), -1)
                alpha = 0.5  # Transparency factor.
                image = cv2.addWeighted(overlay, alpha, image, 1 - alpha, 0)
                cv2.putText(image, 'Shifted ROI', (x1_shifted, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (100, 100, 100), 2)
        


        color = tuple(np.random.randint(0, 256, 3).tolist())  # Generate a random color
        for bbox in bboxes:
            x1, y1, x2, y2 = map(int, bbox)

            print(f"Bounding box: ({x1}, {y1}), ({x2}, {y2})")

            image = cv2.rectangle(image, (x1, y1), (x2, y2),(255, 0, 0), 2)

        

        return image
    
    def get_roi_coordinates(self, image:np.ndarray):
        """
        Calculate the coordinates of the Region of Interest (ROI) in the given image.
        Args:
            image (np.ndarray): The input image as a NumPy array.
        Returns:
            list: A list containing the coordinates of the top-left and bottom-right corners of the ROI 
                in the format [top_left_x, top_left_y, bottom_right_x, bottom_right_y].
        """
        rows = image.shape[0]
        cols = image.shape[1]

        top_left_y_coord = (int(rows/2) - int(self.ROI_ROWS/2))
        top_left_x_coord = (int(cols/2) - int(self.ROI_COLS/2))

        bottom_right_y = top_left_y_coord + self.ROI_COLS
        bottom_right_x = top_left_x_coord + self.ROI_ROWS

        if cols < self.ROI_COLS:
            bottom_right_x = cols
            top_left_x_coord = 0

        if rows < self.ROI_ROWS:
            bottom_right_y = rows
            top_left_y_coord = 0


        return [top_left_x_coord,top_left_y_coord,bottom_right_x,bottom_right_y]
    
    def print_info(self):
        print(self.yolo.info())