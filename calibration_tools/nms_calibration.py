import cv2
import numpy as np

# Generate random bounding boxes for demonstration purposes
def generate_random_boxes(image_shape, num_boxes=10):
    boxes = []
    for _ in range(num_boxes):
        x1 = np.random.randint(0, image_shape[1] - 50)
        y1 = np.random.randint(0, image_shape[0] - 50)
        x2 = np.random.randint(x1 + 30, min(image_shape[1], x1 + 100))
        y2 = np.random.randint(y1 + 30, min(image_shape[0], y1 + 100))
        boxes.append([x1, y1, x2, y2])
    return np.array(boxes)

# Apply Non-Maximum Suppression
def nms(boxes, threshold):
    if len(boxes) == 0:
        return []

    boxes = boxes.astype(np.float32)
    pick = []

    x1 = boxes[:, 0]
    y1 = boxes[:, 1]
    x2 = boxes[:, 2]
    y2 = boxes[:, 3]

    area = (x2 - x1 + 1) * (y2 - y1 + 1)
    idxs = np.argsort(y2)

    while len(idxs) > 0:
        last = len(idxs) - 1
        i = idxs[last]
        pick.append(i)

        xx1 = np.maximum(x1[i], x1[idxs[:last]])
        yy1 = np.maximum(y1[i], y1[idxs[:last]])
        xx2 = np.minimum(x2[i], x2[idxs[:last]])
        yy2 = np.minimum(y2[i], y2[idxs[:last]])

        w = np.maximum(0, xx2 - xx1 + 1)
        h = np.maximum(0, yy2 - yy1 + 1)

        overlap = (w * h) / area[idxs[:last]]

        idxs = np.delete(idxs, np.concatenate(([last], np.where(overlap > threshold)[0])))

    return boxes[pick].astype(int)

# Draw bounding boxes on an image
def draw_boxes(image, boxes, color=(0, 255, 0), thickness=2):
    for (x1, y1, x2, y2) in boxes:
        cv2.rectangle(image, (x1, y1), (x2, y2), color, thickness)

# Callback function for trackbar
def update_threshold(x):
    global img_copy, original_boxes
    img_copy = img.copy()
    threshold = cv2.getTrackbarPos('NMS Threshold', 'Calibration Tool') / 100
    selected_boxes = nms(original_boxes, threshold)
    draw_boxes(img_copy, original_boxes, color=(0, 0, 255))  # Draw original boxes in red
    draw_boxes(img_copy, selected_boxes)  # Draw NMS-selected boxes in green
    cv2.imshow('Calibration Tool', img_copy)

# Load your image
img = cv2.imread('image.jpg')
img_copy = img.copy()

# Generate random bounding boxes
original_boxes = generate_random_boxes(img.shape)

# Create a window
cv2.namedWindow('Calibration Tool')

# Create a trackbar for threshold adjustment
cv2.createTrackbar('NMS Threshold', 'Calibration Tool', 50, 100, update_threshold)

# Initial display
update_threshold(50)

# Wait until user exits the window
cv2.waitKey(0)
cv2.destroyAllWindows()
