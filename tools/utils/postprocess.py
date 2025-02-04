# Ultralytics YOLO 🚀, AGPL-3.0 license

# need gpu
# nms

import argparse

import cv2.dnn
import numpy as np

from ultralytics.utils import ASSETS, yaml_load
from ultralytics.utils.checks import check_yaml

CLASSES = yaml_load(check_yaml("coco8.yaml"))["names"]
colors = np.random.uniform(0, 255, size=(len(CLASSES), 3))


def draw_bounding_box(img, class_id, confidence, x, y, x_plus_w, y_plus_h):
    """
    Draws bounding boxes on the input image based on the provided arguments.

    Args:
        img (numpy.ndarray): The input image to draw the bounding box on.
        class_id (int): Class ID of the detected object.
        confidence (float): Confidence score of the detected object.
        x (int): X-coordinate of the top-left corner of the bounding box.
        y (int): Y-coordinate of the top-left corner of the bounding box.
        x_plus_w (int): X-coordinate of the bottom-right corner of the bounding box.
        y_plus_h (int): Y-coordinate of the bottom-right corner of the bounding box.
    """
    label = f"{CLASSES[class_id]} ({confidence:.2f})"
    color = colors[class_id]
    cv2.rectangle(img, (x, y), (x_plus_w, y_plus_h), color, 2)
    cv2.putText(img, label, (x - 10, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)


def main(onnx_model, input_image):
    """
    Main function to load ONNX model, perform inference, draw bounding boxes, and display the output image.

    Args:
        onnx_model (str): Path to the ONNX model.
        input_image (str): Path to the input image.

    Returns:
        list: List of dictionaries containing detection information such as class_id, class_name, confidence, etc.
    """
    # Load the ONNX model
    model: cv2.dnn.Net = cv2.dnn.readNetFromONNX(onnx_model)

    # Read the input image
    original_image: np.ndarray = cv2.imread(input_image)
    [height, width, _] = original_image.shape

    # Prepare a square image for inference
    length = max((height, width))
    image = np.zeros((length, length, 3), np.uint8)
    image[0:height, 0:width] = original_image

    # Calculate scale factor
    scale = length / 640

    # Preprocess the image and prepare blob for model
    blob = cv2.dnn.blobFromImage(image, scalefactor=1 / 255, size=(640, 640), swapRB=True)
    model.setInput(blob)

    # Perform inference
    outputs = model.forward()

    # Prepare output array
    outputs = np.array([cv2.transpose(outputs[0])])
    rows = outputs.shape[1]

    boxes = []
    scores = []
    class_ids = []

    # Iterate through output to collect bounding boxes, confidence scores, and class IDs
    for i in range(rows):
        classes_scores = outputs[0][i][4:]
        (minScore, maxScore, minClassLoc, (x, maxClassIndex)) = cv2.minMaxLoc(classes_scores)
        if maxScore >= 0.25:
            box = [
                outputs[0][i][0] - (0.5 * outputs[0][i][2]),
                outputs[0][i][1] - (0.5 * outputs[0][i][3]),
                outputs[0][i][2],
                outputs[0][i][3],
            ]
            boxes.append(box)
            scores.append(maxScore)
            class_ids.append(maxClassIndex)

    # Apply NMS (Non-maximum suppression)
    result_boxes = cv2.dnn.NMSBoxes(boxes, scores, 0.25, 0.45, 0.5)

    detections = []

    # Iterate through NMS results to draw bounding boxes and labels
    for i in range(len(result_boxes)):
        index = result_boxes[i]
        box = boxes[index]
        detection = {
            "class_id": class_ids[index],
            "class_name": CLASSES[class_ids[index]],
            "confidence": scores[index],
            "box": box,
            "scale": scale,
        }
        detections.append(detection)
        draw_bounding_box(
            original_image,
            class_ids[index],
            scores[index],
            round(box[0] * scale),
            round(box[1] * scale),
            round((box[0] + box[2]) * scale),
            round((box[1] + box[3]) * scale),
        )

    # Display the image with bounding boxes
    cv2.imshow("image", original_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    return detections


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="yolov8n.onnx", help="Input your ONNX model.")
    parser.add_argument("--img", default=str(ASSETS / "bus.jpg"), help="Path to input image.")
    args = parser.parse_args()
    main(args.model, args.img)

    def post_process(self, output, original_image, scales, conf_threshold=0.5, iou_threshold=0.4):
        """
        Post-process the model output to extract bounding boxes, confidence, and class scores.
        Rescale the boxes back to the original image size.
        :param output: Raw output from the model.
        :param original_image: Original image for drawing bounding boxes.
        :param scales: Scaling factors to map the boxes back to original size.
        :param conf_threshold: Confidence score threshold for filtering detections.
        :param iou_threshold: IOU threshold for non-maximum suppression (NMS).
        :return: Image with annotated bounding boxes.
        """
        scale_x, scale_y = scales
        boxes = output[0]
        filtered_boxes = []

        # Iterate over boxes and filter by confidence
        for box in boxes:
            x1, y1, x2, y2, score, class_id = box
            if score >= conf_threshold:
                # Rescale box coordinates to the original image size
                x1 *= scale_x
                x2 *= scale_x
                y1 *= scale_y
                y2 *= scale_y
                filtered_boxes.append([x1, y1, x2, y2, score, class_id])

        # Apply Non-Maximum Suppression (NMS)
        filtered_boxes = self.nms(filtered_boxes, iou_threshold)

        # Annotate the image with bounding boxes
        for (x1, y1, x2, y2, score, class_id) in filtered_boxes:
            # Draw bounding box
            cv2.rectangle(original_image, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
            # Put label and score
            label = f"Class {int(class_id)}: {score:.2f}"
            cv2.putText(original_image, label, (int(x1), int(y1)-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)

        return original_image, filtered_boxes

    def nms(self, boxes, iou_threshold):
        """
        Perform Non-Maximum Suppression (NMS) on the bounding boxes.
        :param boxes: List of boxes in the format [x1, y1, x2, y2, score, class_id].
        :param iou_threshold: Intersection-over-Union threshold for filtering overlapping boxes.
        :return: Filtered list of bounding boxes after NMS.
        """
        if len(boxes) == 0:
            return []
        
        boxes = sorted(boxes, key=lambda x: x[4], reverse=True)  # Sort by confidence score
        
        keep_boxes = []
        while boxes:
            chosen_box = boxes.pop(0)
            keep_boxes.append(chosen_box)
            boxes = [box for box in boxes if self.iou(chosen_box, box) < iou_threshold]
        
        return keep_boxes

    def iou(self, box1, box2):
        """
        Calculate Intersection over Union (IoU) between two boxes.
        :param box1: First box in the format [x1, y1, x2, y2, score, class_id].
        :param box2: Second box in the same format.
        :return: IoU score.
        """
        x1 = max(box1[0], box2[0])
        y1 = max(box1[1], box2[1])
        x2 = min(box1[2], box2[2])
        y2 = min(box1[3], box2[3])

        inter_area = max(0, x2 - x1) * max(0, y2 - y1)
        box1_area = (box1[2] - box1[0]) * (box1[3] - box1[1])
        box2_area = (box2[2] - box2[0]) * (box2[3] - box2[1])
        union_area = box1_area + box2_area - inter_area
        
        return inter_area / union_area