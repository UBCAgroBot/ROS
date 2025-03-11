import cv2
import os

def draw_bounding_boxes(image_path, bboxes):
    # Read the image using OpenCV
    img = cv2.imread(image_path)
    
    if img is None:
        print(f"Error loading image: {image_path}")
        return
    
    # Get the dimensions of the image
    height, width, _ = img.shape
    print(height)
    print(width)
    
    # Draw each bounding box on the image
    for bbox in bboxes:
        class_id, x_center, y_center, bbox_width, bbox_height = bbox
        
        # Convert normalized values to absolute pixel values
        x_center_pixel = int(x_center * width)
        y_center_pixel = int(y_center * height)
        bbox_width_pixel = int(bbox_width * width)
        bbox_height_pixel = int(bbox_height * height)
        
        # Calculate the top-left and bottom-right corners of the bounding box
        top_left_x = int(x_center_pixel - bbox_width_pixel / 2)
        top_left_y = int(y_center_pixel - bbox_height_pixel / 2)
        bottom_right_x = int(x_center_pixel + bbox_width_pixel / 2)
        bottom_right_y = int(y_center_pixel + bbox_height_pixel / 2)
        
        # Draw the bounding box (using green color and thickness of 2)
        cv2.rectangle(img, (top_left_x, top_left_y), (bottom_right_x, bottom_right_y), (0, 255, 0), 2)
    
        # Show the image with bounding boxes (press any key to close)
        cv2.imshow('Bounding Boxes', img)
        cv2.waitKey(10000)
        cv2.destroyAllWindows()

def read_bounding_boxes(txt_file):
    bboxes = []
    with open(txt_file, 'r') as file:
        for line in file.readlines():
            values = line.strip().split()
            class_id = int(values[0])
            x_center = float(values[1])
            y_center = float(values[2])
            bbox_width = float(values[3])
            bbox_height = float(values[4])
            bboxes.append((class_id, x_center, y_center, bbox_width, bbox_height))
    return bboxes

os.chdir("C:/Users/ishaa/Coding Projects/Applied-AI/ROS/assets/maize")
print(os.getcwd())
boxes = read_bounding_boxes("IMG_2884_18.txt")
print(boxes)
draw_bounding_boxes("IMG_2884_18.JPG", boxes)

def box_iou_batch(self, boxes_a, boxes_b):
    # Helper function to calculate the area of the boxes
    def box_area(box):
        return (box[2] - box[0]) * (box[3] - box[1])

    # Compute areas of both sets of boxes
    area_a = box_area(boxes_a.T)
    area_b = box_area(boxes_b.T)

    # Calculate top-left and bottom-right coordinates of the intersection box
    top_left = cp.maximum(boxes_a[:, None, :2], boxes_b[:, :2])
    bottom_right = cp.minimum(boxes_a[:, None, 2:], boxes_b[:, 2:])

    # Calculate intersection area
    area_inter = cp.prod(cp.clip(bottom_right - top_left, a_min=0, a_max=None), axis=2)

    # Return the IoU (Intersection over Union)
    return area_inter / (area_a[:, None] + area_b - area_inter)

def non_max_suppression(self, predictions, iou_threshold = 0.5):
    # need to cast memory?
    rows, _ = predictions.shape

    # Sort predictions by score in descending order
    sort_index = cp.flip(predictions[:, 4].argsort())
    predictions = predictions[sort_index]

    # Extract boxes and categories
    boxes = predictions[:, :4]
    categories = predictions[:, 5]

    # Compute IoUs between all boxes
    ious = self.box_iou_batch(boxes, boxes)
    ious = ious - cp.eye(rows)  # Remove self IoU

    # Initialize array to track which boxes to keep
    keep = cp.ones(rows, dtype=cp.bool)

    # Non-max suppression loop
    for index, (iou, category) in enumerate(zip(ious, categories)):
        if not keep[index]:
            continue

        # Identify boxes that overlap (IoU > threshold) and have the same category
        condition = (iou > iou_threshold) & (categories == category)
        keep = keep & ~condition

    # Sort the kept boxes back to the original order
    return keep[sort_index.argsort()]

def postprocess(self, output):
    tic = time.perf_counter_ns()
    num_detections = len(output) // 5
    output = np.reshape(output, (5, num_detections))
    # output = np.reshape(output, (num_detections, 5))
    
    width = 640
    height = 640
    conf_threshold = 0.9
    nms_threshold = 0.1
    boxes = []
    confidences = []
    
    # print("HERE!!!", output.shape)
    # print(output[:,:20])
    
    for i in range(output.shape[1]):
    # for detection in output:

        detection = output[..., i]
        
        # obj_conf, x_center, y_center, bbox_width, bbox_height = detection[:]
        x_center, y_center, bbox_width, bbox_height, obj_conf = detection[:]
        # x_min, 
        
        # Apply sigmoid to object confidence and class score
        # obj_conf = 1 / (1 + np.exp(-obj_conf))  # Sigmoid for object confidence

        # Filter out weak predictions based on confidence threshold
        if obj_conf < conf_threshold:
            continue
        print(detection)
        
        # Convert normalized values to absolute pixel values
        x_center_pixel = int(x_center)
        y_center_pixel = int(y_center)
        bbox_width_pixel = int(bbox_width)
        bbox_height_pixel = int(bbox_height)
        # print(f"[{obj_conf}, {x_center_pixel}, {y_center_pixel}, {bbox_width_pixel}, {bbox_height_pixel} ]")
        
        # Calculate the top-left and bottom-right corners of the bounding box
        top_left_x = int(x_center_pixel - bbox_width_pixel / 2)
        top_left_y = int(y_center_pixel - bbox_height_pixel / 2)
        bottom_right_x = int(x_center_pixel + bbox_width_pixel / 2)
        bottom_right_y = int(y_center_pixel + bbox_height_pixel / 2)
        
        # boxes.append([x_center_pixel, y_center_pixel, bbox_width_pixel, bbox_height_pixel])
        # confidences.append(confidence)
        boxes.append([top_left_x, top_left_y, bottom_right_x, bottom_right_y])
        
        # Append the box, confidence, and class score
        # boxes.append([x_min, y_min, x_max, y_max])
        confidences.append(float(obj_conf))
    
    # # Apply Non-Maximum Suppression (NMS) to suppress overlapping boxes
    indices = cv2.dnn.NMSBoxes(boxes, confidences, conf_threshold, nms_threshold)

    final_boxes = []
    for i in indices:
        # Since indices is now likely a 1D list, no need for i[0]
        final_boxes.append([*boxes[i], confidences[i]])
    toc = time.perf_counter_ns()
    self.get_logger().info(f"Postprocessing: {(toc-tic)/1e6} ms")
    self.display(final_boxes)
    # print(final_boxes)
    # self.display(boxes)

main("/home/user/workspace/models/maize/Maize.onnx", "/home/user/workspace/assets/maize/IMG_1822_14.JPG")

# if __name__ == "__main__":
#     parser = argparse.ArgumentParser()
#     parser.add_argument("--model", default="yolov8n.onnx", help="Input your ONNX model.")
#     parser.add_argument("--img", default=str(ASSETS / "bus.jpg"), help="Path to input image.")
#     args = parser.parse_args()
#     main(args.model, args.img)

#     def post_process(self, output, original_image, scales, conf_threshold=0.5, iou_threshold=0.4):
#         """
#         Post-process the model output to extract bounding boxes, confidence, and class scores.
#         Rescale the boxes back to the original image size.
#         :param output: Raw output from the model.
#         :param original_image: Original image for drawing bounding boxes.
#         :param scales: Scaling factors to map the boxes back to original size.
#         :param conf_threshold: Confidence score threshold for filtering detections.
#         :param iou_threshold: IOU threshold for non-maximum suppression (NMS).
#         :return: Image with annotated bounding boxes.
#         """
#         scale_x, scale_y = scales
#         boxes = output[0]
#         filtered_boxes = []

#         # Iterate over boxes and filter by confidence
#         for box in boxes:
#             x1, y1, x2, y2, score, class_id = box
#             if score >= conf_threshold:
#                 # Rescale box coordinates to the original image size
#                 x1 *= scale_x
#                 x2 *= scale_x
#                 y1 *= scale_y
#                 y2 *= scale_y
#                 filtered_boxes.append([x1, y1, x2, y2, score, class_id])

#         # Apply Non-Maximum Suppression (NMS)
#         filtered_boxes = self.nms(filtered_boxes, iou_threshold)

#         # Annotate the image with bounding boxes
#         for (x1, y1, x2, y2, score, class_id) in filtered_boxes:
#             # Draw bounding box
#             cv2.rectangle(original_image, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
#             # Put label and score
#             label = f"Class {int(class_id)}: {score:.2f}"
#             cv2.putText(original_image, label, (int(x1), int(y1)-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)

#         return original_image, filtered_boxes

#     def nms(self, boxes, iou_threshold):
#         """
#         Perform Non-Maximum Suppression (NMS) on the bounding boxes.
#         :param boxes: List of boxes in the format [x1, y1, x2, y2, score, class_id].
#         :param iou_threshold: Intersection-over-Union threshold for filtering overlapping boxes.
#         :return: Filtered list of bounding boxes after NMS.
#         """
#         if len(boxes) == 0:
#             return []
        
#         boxes = sorted(boxes, key=lambda x: x[4], reverse=True)  # Sort by confidence score
        
#         keep_boxes = []
#         while boxes:
#             chosen_box = boxes.pop(0)
#             keep_boxes.append(chosen_box)
#             boxes = [box for box in boxes if self.iou(chosen_box, box) < iou_threshold]
        
#         return keep_boxes

#     def iou(self, box1, box2):
#         """
#         Calculate Intersection over Union (IoU) between two boxes.
#         :param box1: First box in the format [x1, y1, x2, y2, score, class_id].
#         :param box2: Second box in the same format.
#         :return: IoU score.
#         """
#         x1 = max(box1[0], box2[0])
#         y1 = max(box1[1], box2[1])
#         x2 = min(box1[2], box2[2])
#         y2 = min(box1[3], box2[3])

#         inter_area = max(0, x2 - x1) * max(0, y2 - y1)
#         box1_area = (box1[2] - box1[0]) * (box1[3] - box1[1])
#         box2_area = (box2[2] - box2[0]) * (box2[3] - box2[1])
#         union_area = box1_area + box2_area - inter_area
        
#         return inter_area / union_area

import cv2
import os

def draw_bounding_boxes(image_path, bboxes):
    # Read the image using OpenCV
    img = cv2.imread(image_path)
    
    if img is None:
        print(f"Error loading image: {image_path}")
        return
    
    # Get the dimensions of the image
    height, width, _ = img.shape
    print(height)
    print(width)
    
    # Draw each bounding box on the image
    for bbox in bboxes:
        class_id, x_center, y_center, bbox_width, bbox_height = bbox
        
        # Convert normalized values to absolute pixel values
        x_center_pixel = int(x_center * width)
        y_center_pixel = int(y_center * height)
        bbox_width_pixel = int(bbox_width * width)
        bbox_height_pixel = int(bbox_height * height)
        
        # Calculate the top-left and bottom-right corners of the bounding box
        top_left_x = int(x_center_pixel - bbox_width_pixel / 2)
        top_left_y = int(y_center_pixel - bbox_height_pixel / 2)
        bottom_right_x = int(x_center_pixel + bbox_width_pixel / 2)
        bottom_right_y = int(y_center_pixel + bbox_height_pixel / 2)
        
        # Draw the bounding box (using green color and thickness of 2)
        cv2.rectangle(img, (top_left_x, top_left_y), (bottom_right_x, bottom_right_y), (0, 255, 0), 2)
    
        # Show the image with bounding boxes (press any key to close)
        cv2.imshow('Bounding Boxes', img)
        cv2.waitKey(10000)
        cv2.destroyAllWindows()

def read_bounding_boxes(txt_file):
    bboxes = []
    with open(txt_file, 'r') as file:
        for line in file.readlines():
            values = line.strip().split()
            class_id = int(values[0])
            x_center = float(values[1])
            y_center = float(values[2])
            bbox_width = float(values[3])
            bbox_height = float(values[4])
            bboxes.append((class_id, x_center, y_center, bbox_width, bbox_height))
    return bboxes

os.chdir("C:/Users/ishaa/Coding Projects/Applied-AI/ROS/assets/maize")
print(os.getcwd())
boxes = read_bounding_boxes("IMG_2884_18.txt")
print(boxes)
draw_bounding_boxes("IMG_2884_18.JPG", boxes)

def box_iou_batch(self, boxes_a, boxes_b):
    # Helper function to calculate the area of the boxes
    def box_area(box):
        return (box[2] - box[0]) * (box[3] - box[1])

    # Compute areas of both sets of boxes
    area_a = box_area(boxes_a.T)
    area_b = box_area(boxes_b.T)

    # Calculate top-left and bottom-right coordinates of the intersection box
    top_left = cp.maximum(boxes_a[:, None, :2], boxes_b[:, :2])
    bottom_right = cp.minimum(boxes_a[:, None, 2:], boxes_b[:, 2:])

    # Calculate intersection area
    area_inter = cp.prod(cp.clip(bottom_right - top_left, a_min=0, a_max=None), axis=2)

    # Return the IoU (Intersection over Union)
    return area_inter / (area_a[:, None] + area_b - area_inter)

def non_max_suppression(self, predictions, iou_threshold = 0.5):
    # need to cast memory?
    rows, _ = predictions.shape

    # Sort predictions by score in descending order
    sort_index = cp.flip(predictions[:, 4].argsort())
    predictions = predictions[sort_index]

    # Extract boxes and categories
    boxes = predictions[:, :4]
    categories = predictions[:, 5]

    # Compute IoUs between all boxes
    ious = self.box_iou_batch(boxes, boxes)
    ious = ious - cp.eye(rows)  # Remove self IoU

    # Initialize array to track which boxes to keep
    keep = cp.ones(rows, dtype=cp.bool)

    # Non-max suppression loop
    for index, (iou, category) in enumerate(zip(ious, categories)):
        if not keep[index]:
            continue

        # Identify boxes that overlap (IoU > threshold) and have the same category
        condition = (iou > iou_threshold) & (categories == category)
        keep = keep & ~condition

    # Sort the kept boxes back to the original order
    return keep[sort_index.argsort()]

def postprocess(self, output):
    tic = time.perf_counter_ns()
    num_detections = len(output) // 5
    output = np.reshape(output, (5, num_detections))
    # output = np.reshape(output, (num_detections, 5))
    
    width = 640
    height = 640
    conf_threshold = 0.9
    nms_threshold = 0.1
    boxes = []
    confidences = []
    
    # print("HERE!!!", output.shape)
    # print(output[:,:20])
    
    for i in range(output.shape[1]):
    # for detection in output:

        detection = output[..., i]
        
        # obj_conf, x_center, y_center, bbox_width, bbox_height = detection[:]
        x_center, y_center, bbox_width, bbox_height, obj_conf = detection[:]
        # x_min, 
        
        # Apply sigmoid to object confidence and class score
        # obj_conf = 1 / (1 + np.exp(-obj_conf))  # Sigmoid for object confidence

        # Filter out weak predictions based on confidence threshold
        if obj_conf < conf_threshold:
            continue
        print(detection)
        
        # Convert normalized values to absolute pixel values
        x_center_pixel = int(x_center)
        y_center_pixel = int(y_center)
        bbox_width_pixel = int(bbox_width)
        bbox_height_pixel = int(bbox_height)
        # print(f"[{obj_conf}, {x_center_pixel}, {y_center_pixel}, {bbox_width_pixel}, {bbox_height_pixel} ]")
        
        # Calculate the top-left and bottom-right corners of the bounding box
        top_left_x = int(x_center_pixel - bbox_width_pixel / 2)
        top_left_y = int(y_center_pixel - bbox_height_pixel / 2)
        bottom_right_x = int(x_center_pixel + bbox_width_pixel / 2)
        bottom_right_y = int(y_center_pixel + bbox_height_pixel / 2)
        
        # boxes.append([x_center_pixel, y_center_pixel, bbox_width_pixel, bbox_height_pixel])
        # confidences.append(confidence)
        boxes.append([top_left_x, top_left_y, bottom_right_x, bottom_right_y])
        
        # Append the box, confidence, and class score
        # boxes.append([x_min, y_min, x_max, y_max])
        confidences.append(float(obj_conf))
    
    # # Apply Non-Maximum Suppression (NMS) to suppress overlapping boxes
    indices = cv2.dnn.NMSBoxes(boxes, confidences, conf_threshold, nms_threshold)

    final_boxes = []
    for i in indices:
        # Since indices is now likely a 1D list, no need for i[0]
        final_boxes.append([*boxes[i], confidences[i]])
    toc = time.perf_counter_ns()
    self.get_logger().info(f"Postprocessing: {(toc-tic)/1e6} ms")
    self.display(final_boxes)
    # print(final_boxes)
    # self.display(boxes)

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

import torch

def verify_object(self, disp_image, bboxes):
    roi_x1, roi_y1, roi_x2, roi_y2 = self.roi_list
    original_height, original_width = original_image_shape
    model_height, model_width = model_dimensions

    shifted_x = roi_x + abs(velocity[0]) * shift_constant
    scale_x = roi_w / model_width
    scale_y = roi_h / model_height

    # Convert bounding boxes to a PyTorch tensor for GPU processing
    bboxes_tensor = torch.tensor(bboxes, dtype=torch.float32)

    # Reverse the resize operation using vectorized operations in PyTorch
    x_min = (bboxes_tensor[:, 0] * scale_x).to(torch.int32)
    y_min = (bboxes_tensor[:, 1] * scale_y).to(torch.int32)
    x_max = (bboxes_tensor[:, 2] * scale_x).to(torch.int32)
    y_max = (bboxes_tensor[:, 3] * scale_y).to(torch.int32)

    # Reverse the cropping operation
    x_min += shifted_x
    y_min += roi_y
    x_max += shifted_x
    y_max += roi_y

    # Ensure the bounding boxes don't exceed the original image dimensions
    x_min = torch.clamp(x_min, 0, original_width)
    y_min = torch.clamp(y_min, 0, original_height)
    x_max = torch.clamp(x_max, 0, original_width)
    y_max = torch.clamp(y_max, 0, original_height)

    # Stack the adjusted bounding boxes
    adjusted_bboxes = torch.stack((x_min, y_min, x_max, y_max), dim=1)

    # Check if adjusted bounding boxes are within the ROI
    for bbox in adjusted_bboxes:
        if bbox[0] >= roi_x1 and bbox[2] <= roi_x2 and bbox[1] >= roi_y1 and bbox[3] <= roi_y2:
            self.on = 1

# Using PyTorch Tensors: The bounding boxes are converted to a PyTorch tensor, which allows for GPU acceleration and efficient batch processing.

# Vectorized Operations: All calculations (scaling and clipping) are performed using PyTorch’s built-in functions, which are optimized for performance.

# Clamping: The torch.clamp function is used to ensure bounding box coordinates are within valid ranges, similar to np.clip.

# Condition Checks: The condition checks for bounding boxes being within the ROI are retained in the loop but could also be vectorized if needed. However, since it involves logic checks with conditions that might vary for each box, a loop is simpler here.

# vectorized example:
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
