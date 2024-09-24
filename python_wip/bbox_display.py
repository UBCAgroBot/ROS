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