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