import cv2
import os
from ultralytics import YOLO

def annotate_bounding_boxes(img, bboxes):
    height, width, _ = img.shape
    
    for bbox in bboxes:
        _, x_center, y_center, bbox_width, bbox_height = bbox
        
        x_center_pixel = int(x_center * width)
        y_center_pixel = int(y_center * height)
        bbox_width_pixel = int(bbox_width * width)
        bbox_height_pixel = int(bbox_height * height)
        
        top_left_x = int(x_center_pixel - bbox_width_pixel / 2)
        top_left_y = int(y_center_pixel - bbox_height_pixel / 2)
        bottom_right_x = int(x_center_pixel + bbox_width_pixel / 2)
        bottom_right_y = int(y_center_pixel + bbox_height_pixel / 2)
        
        cv2.rectangle(img, (top_left_x, top_left_y), (bottom_right_x, bottom_right_y), (0, 255, 0), 2)
    
    return img

def annotate_ultralytics_bounding_boxes(img, bboxes, file_name):
    for bbox in bboxes:
        top_left_x, top_left_y, bottom_right_x, bottom_right_y = bbox
        cv2.rectangle(img, (int(top_left_x), int(top_left_y)), (int(bottom_right_x), int(bottom_right_y)), (0, 0, 255), 2)
        img = annotate_legend(img, file_name)
        
    return img

def annotate_legend(img, file_name):
    height, width, _ = img.shape
    margin = 10
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.4
    thickness = 1
    line_type = cv2.LINE_AA

    # Define legend texts
    text_ultralytics = "Custom Model"
    text_ground_truth = "Ground Truth"
    random_text = file_name

    # Calculate text sizes to right-align the texts
    (w_ut, h_ut), _ = cv2.getTextSize(text_ultralytics, font, font_scale, thickness)
    (w_gt, h_gt), _ = cv2.getTextSize(text_ground_truth, font, font_scale, thickness)
    (w_rand, h_rand), _ = cv2.getTextSize(random_text, font, font_scale, thickness)

    # Set positions (from top-right, accounting for margin)
    x_ut = width - w_ut - margin
    y_ut = margin + h_ut

    x_gt = width - w_gt - margin
    y_gt = y_ut + h_ut + margin

    # Top-left for random text:
    x_rand = margin
    y_rand = margin + h_rand

    cv2.putText(img, text_ultralytics, (x_ut, y_ut), font, font_scale, (0, 0, 255), thickness, line_type)
    cv2.putText(img, text_ground_truth, (x_gt, y_gt), font, font_scale, (0, 255, 0), thickness, line_type)
    cv2.putText(img, random_text, (x_rand, y_rand), font, font_scale, (255, 255, 255), thickness, line_type)
    return img

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

def inference(images, model_path):
    if model_path:
        model = YOLO(model_path)
    else:
        current_dir = os.path.dirname(os.path.realpath(__file__))
        os.chdir(current_dir)
        os.chdir("..\\..")
        os.chdir("./models/maize")
        model = YOLO("Maize.pt")
    
    boxes_dict = {}
    for file_prefix, image in images.items():
        results = model(image)
        bounding_boxes = []
        for result in results:
            for box in result.boxes:
                box_list = box.xyxy[0]
                x_min, y_min, x_max, y_max = box_list
                bounding_boxes.append(
                    [x_min.item(), y_min.item(), x_max.item(), y_max.item()]
                )
        boxes_dict[file_prefix] = bounding_boxes
    
    return boxes_dict

def parse():
    current_dir = os.path.dirname(os.path.realpath(__file__))
    os.chdir(current_dir)
    os.chdir("..\\..")
    os.chdir("./assets/maize")
    image_path = os.getcwd()
    
    files = []
    image_dict = {}
    boxes_dict = {}
    index = 0
    for filename in os.listdir(image_path):
        if filename.endswith(".jpg") or filename.endswith(".JPG") or filename.endswith(".png"):
            prefix = filename.split(".")[0]
            image = cv2.imread(filename)
            if image is None:
                print(f"Error loading image: {image_path}")
            else:
                image_dict[prefix] = image
                textfilename = filename.split(".")[0] + ".txt"
                boxes_dict[prefix] = read_bounding_boxes(textfilename)
                files.append(prefix)
    
    yolo_boxes_dict = inference(image_dict, None)
    
    if len(files) == 0:
        raise ValueError(f"No images files found in {image_path}")
    
    image = image_dict[files[index % len(files)]]
    
    cv2.namedWindow('Bounding Box Display', cv2.WINDOW_NORMAL)
    
    key = 0
    while key != ord('q'):
        image = image_dict[files[index % len(files)]]
        boxes = boxes_dict[files[index % len(files)]]
        yolo_boxes = yolo_boxes_dict[files[index % len(files)]]
        annotated_image = annotate_bounding_boxes(image, boxes)
        annotated_image = annotate_ultralytics_bounding_boxes(annotated_image, yolo_boxes, files[index % len(files)])
        cv2.imshow('Bounding Box Display', annotated_image)
        
        key = cv2.waitKey(1)
        if key == ord('q'):
            cv2.destroyAllWindows()
            break
        elif key == ord('a'):
            index -= 1
        elif key == ord('d'):
            index += 1

parse()