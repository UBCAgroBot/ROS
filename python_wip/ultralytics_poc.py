from ultralytics import YOLO
import cv2

# Define ROI (Region of Interest) as (x_min, y_min, x_max, y_max)
roi = (100, 100, 800, 800)
roi_x1, roi_y1, roi_x2, roi_y2 = roi

# Load the pre-trained YOLO model
model = YOLO('C:/Users/ishaa/Coding Projects/ROS/models/maize/Maize.pt')

# Load the image
image = cv2.imread('C:/Users/ishaa/Coding Projects/ROS/assets/maize/IMG_1822_14.JPG')

# Make predictions on the image
results = model(image)
result = results[0]

print("Bounding boxes:")
boxes = result.boxes.xyxy  # xyxy format: [x_min, y_min, x_max, y_max]

for box in boxes:
    print(box)
    x_min, y_min, x_max, y_max = box.tolist()

    if x_min >= roi_x1 and x_max <= roi_x2 and y_min >= roi_y1 and y_max <= roi_y2:
        print(1)

    cv2.rectangle(image, (int(x_min), int(y_min)), (int(x_max), int(y_max)), (0, 255, 0), 2)

cv2.rectangle(image, (roi_x1, roi_y1), (roi_x2, roi_y2), (255, 0, 0), 2)

cv2.imshow("Image with ROI", image)
cv2.waitKey(0)
cv2.destroyAllWindows()

# scaling color filter bbox output -> can get cropped images from ultralytics API