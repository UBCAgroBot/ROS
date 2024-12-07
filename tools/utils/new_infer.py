import os
import cv2
from onnxruntime import InferenceSession
import warnings 
import numpy as np

class ONNXModel:
    def __init__(self, onnx_path, session=None):
        warnings.filterwarnings("ignore")
        self.session = session
        if self.session is None:
            assert onnx_path is not None
            assert os.path.exists(onnx_path)
            self.session = InferenceSession(onnx_path,
                                            providers=['CPUExecutionProvider']) # fix
        self.inputs = self.session.get_inputs()[0]
        self.confidence_threshold = 0.5
        self.iou_threshold = 0.9
        self.input_size = 640
        shape = (1, 3, self.input_size, self.input_size)
        image = np.zeros(shape, dtype='float32')
        for _ in range(10):
            self.session.run(output_names=None,
                            input_feed={self.inputs.name: image})

    def __call__(self, image):
        image, scale = self.resize(image, self.input_size)
        # image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = image.transpose((2, 0, 1))[::-1]
        image = image.astype('float32') / 255
        image = image[np.newaxis, ...]

        outputs = self.session.run(output_names=None,
                                   input_feed={self.inputs.name: image})
        outputs = np.transpose(np.squeeze(outputs[0]))

        # Lists to store the bounding boxes, scores, and class IDs of the detections
        boxes = []
        scores = []
        class_indices = []

        # Iterate over each row in the outputs array
        for i in range(outputs.shape[0]):
            # Extract the class scores from the current row
            classes_scores = outputs[i][4:]

            # Find the maximum score among the class scores
            max_score = np.amax(classes_scores)

            # If the maximum score is above the confidence threshold
            if max_score >= self.confidence_threshold:
                # Get the class ID with the highest score
                class_id = np.argmax(classes_scores)

                # Extract the bounding box coordinates from the current row
                image, y, w, h = outputs[i][0], outputs[i][1], outputs[i][2], outputs[i][3]

                # Calculate the scaled coordinates of the bounding box
                left = int((image - w / 2) / scale)
                top = int((y - h / 2) / scale)
                width = int(w / scale)
                height = int(h / scale)

                # Add the class ID, score, and box coordinates to the respective lists
                class_indices.append(class_id)
                scores.append(max_score)
                boxes.append([left, top, width, height])
        
        # Apply non-maximum suppression to filter out overlapping bounding boxes
        indices = cv2.dnn.NMSBoxes(boxes, scores, self.confidence_threshold, self.iou_threshold)
        
        # convert to xyxy

        # Iterate over the selected indices after non-maximum suppression
        confidences = []
        bboxes = []
        for i in indices:
            x, y, w, h = boxes[i]
            x1, y1 = x, y
            x2, y2 = x + w, y + h
            bboxes.append([x1, y1, x2, y2])
            confidences.append(scores[i])
        return confidences, bboxes

    @staticmethod
    def resize(image, input_size):
        shape = image.shape

        ratio = float(shape[0]) / shape[1]
        if ratio > 1:
            h = input_size
            w = int(h / ratio)
        else:
            w = input_size
            h = int(w * ratio)
        scale = float(h) / shape[0]
        resized_image = cv2.resize(image, (w, h))
        det_image = np.zeros((input_size, input_size, 3), dtype=np.uint8)
        det_image[:h, :w, :] = resized_image
        return det_image, scale