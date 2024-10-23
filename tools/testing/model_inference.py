# stream off:
from ultralytics import YOLO

# Load a model
model = YOLO("yolov8n.pt")  # pretrained YOLOv8n model

# Run batched inference on a list of images
results = model(["image1.jpg", "image2.jpg"])  # return a list of Results objects

# Process results list
for result in results:
    boxes = result.boxes  # Boxes object for bounding box outputs
    masks = result.masks  # Masks object for segmentation masks outputs
    keypoints = result.keypoints  # Keypoints object for pose outputs
    probs = result.probs  # Probs object for classification outputs
    obb = result.obb  # Oriented boxes object for OBB outputs
    result.show()  # display to screen
    result.save(filename="result.jpg")  # save to disk

# stream on:
from ultralytics import YOLO

# Load a model
model = YOLO("yolov8n.pt")  # pretrained YOLOv8n model

# Run batched inference on a list of images
results = model(["image1.jpg", "image2.jpg"], stream=True)  # return a generator of Results objects

# Process results generator
for result in results:
    boxes = result.boxes  # Boxes object for bounding box outputs
    masks = result.masks  # Masks object for segmentation masks outputs
    keypoints = result.keypoints  # Keypoints object for pose outputs
    probs = result.probs  # Probs object for classification outputs
    obb = result.obb  # Oriented boxes object for OBB outputs
    result.show()  # display to screen
    result.save(filename="result.jpg")  # save to disk


from ultralytics import YOLO

# Load a pretrained YOLOv8n model
model = YOLO("yolov8n.pt")

# Run inference on 'bus.jpg' with arguments
model.predict("bus.jpg", save=True, imgsz=320, conf=0.5)

from ultralytics import YOLO

# Load a pretrained YOLOv8n model
model = YOLO("yolov8n.pt")

# Run inference on an image
results = model("bus.jpg")  # results list

# View results
for r in results:
    print(r.boxes)  # print the Boxes object containing the detection bounding boxes

from PIL import Image

from ultralytics import YOLO

# Load a pretrained YOLOv8n model
model = YOLO("yolov8n.pt")

# Run inference on 'bus.jpg'
results = model(["bus.jpg", "zidane.jpg"])  # results list

# Visualize the results
for i, r in enumerate(results):
    # Plot results image
    im_bgr = r.plot()  # BGR-order numpy array
    im_rgb = Image.fromarray(im_bgr[..., ::-1])  # RGB-order PIL image

    # Show results to screen (in supported environments)
    r.show()

    # Save results to disk
    r.save(filename=f"results{i}.jpg")

import cv2
from ultralytics import YOLO
import time

model = YOLO("/home/user/ROS/models/maize/Maize.engine")
image = cv2.imread("/home/user/ROS/assets/maize/IMG_1822_14.JPG")

sum = 0
# stream = True?
for _ in range(100):
    tic = time.perf_counter_ns()
    result = model.predict(
        image,  # batch=8 of the same image
        verbose=False,
        device="cuda",
    )
    elapsed_time = (time.perf_counter_ns() - tic) / 1e6
    print(f"Elapsed time: {(elapsed_time):.2f} ms")
    sum += elapsed_time
    annotated_frame = result[0].plot()
    cv2.imshow("YOLOv8 Inference", annotated_frame)
    if cv2.waitKey(1) & 0xFF == ord("q"):
            break

avearage_time = (sum - 2660) / 100
print(f"Average time: {avearage_time:.2f} ms")
cv2.destroyAllWindows()

# source, conf, iou, imgz, half, visualize, agnostic_nms
# visualization arguments:

import os
import cv2
import numpy as np
import onnxruntime as ort
import time
import logging

# Set up logging
logging.basicConfig(format='%(message)s', level=logging.INFO)

class Model:
    def __init__(self, model_path=None, onnx_model=None, input_size=(640, 640)):
        """
        Initialize the model by loading the ONNX model with GPU (CUDA) support.
        :param model_path: Path to the ONNX model file.
        :param onnx_model: ONNX model object if passed directly.
        :param input_size: Expected input size for the model (width, height).
        """
        self.input_size = input_size  # Model's expected input size

        if model_path:
            logging.info(f"ONNX: starting from '{model_path}' with input shape (1, 3, {input_size[0]}, {input_size[1]}) BCHW")
            self.session = ort.InferenceSession(model_path, providers=['CUDAExecutionProvider'])
        elif onnx_model:
            self.session = ort.InferenceSession(onnx_model.SerializeToString(), providers=['CUDAExecutionProvider'])
        else:
            raise ValueError("Either model_path or onnx_model must be provided.")

        # Input and output information from the ONNX model
        self.input_name = self.session.get_inputs()[0].name
        self.output_names = [output.name for output in self.session.get_outputs()]

        logging.info(f"ONNX: loaded successfully, using CUDA (GPU)")

    def load_images(self, image_dir='images/'):
        """
        Load images from the specified directory.
        :param image_dir: Directory containing the images to load.
        :return: List of loaded image file paths.
        """
        image_files = [os.path.join(image_dir, img) for img in os.listdir(image_dir) if img.endswith(('.png', '.jpg', '.jpeg'))]
        return image_files

    def preprocess_image(self, image_path):
        """
        Preprocess the image: resize, normalize, and convert to the required format for the model.
        :param image_path: Path to the image file.
        :return: Preprocessed image ready for inference, original image, and scaling factors.
        """
        img = cv2.imread(image_path)
        original_size = img.shape[:2]  # Original size (height, width)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        resized = cv2.resize(img, self.input_size)
        # Normalize the image (assuming mean=0.5, std=0.5 for demonstration)
        normalized = resized / 255.0
        normalized = (normalized - 0.5) / 0.5
        # HWC to CHW format for model input
        input_tensor = np.transpose(normalized, (2, 0, 1)).astype(np.float32)
        input_tensor = np.expand_dims(input_tensor, axis=0)  # Add batch dimension

        # Compute scaling factors to map back to original size
        scale_x = original_size[1] / self.input_size[0]
        scale_y = original_size[0] / self.input_size[1]

        return input_tensor, img, (scale_x, scale_y)

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

    def infer(self, image_dir='images/'):
        """
        Perform inference on all images in the specified directory and display results with bounding boxes.
        :param image_dir: Directory containing images to run inference on.
        :return: None. Displays images with bounding boxes.
        """
        image_files = self.load_images(image_dir)
        results = {}

        for image_file in image_files:
            input_tensor, original_image, scales = self.preprocess_image(image_file)

            # Log inference start
            logging.info(f"Predict: ONNX inference started on {image_file}")
            start_time = time.time()

            # Perform inference
            outputs = self.session.run(self.output_names, {self.input_name: input_tensor})

            # Measure inference time
            inference_time = time.time() - start_time
            logging.info(f"Inference complete in {inference_time:.2f} seconds.")

            # Post-process and display results
            annotated_image, boxes = self.post_process(outputs, original_image, scales)
            num_boxes = len(boxes)

            # Log number of bounding boxes and confidence
            for i, box in enumerate(boxes):
                logging.info(f"Box {i+1}: [x1={box[0]:.1f}, y1={box[1]:.1f}, x2={box[2]:.1f}, y2={box[3]:.1f}], score={box[4]:.2f}")

            logging.info(f"Total bounding boxes: {num_boxes}")

            # Show the image with bounding boxes
            cv2.imshow('Inference Result', annotated_image)
            cv2.waitKey(0)  # Press any key to continue
            cv2.destroyAllWindows()

# Example usage:
if __name__ == "__main__":
    model_path = 'model.onnx'
    model = Model(model_path=model_path)
    
    # Log ONNX loading details
    logging.info(f"ONNX: model loaded from '{model_path}', performing inference...")
    
    model.infer('test_images/')

import time
import onnx
import onnxruntime as ort
import numpy as np

# put random input shape into CUDA if using CUDA provider?
def verify_onnx(model_path, compared_outputs, model_dimensions, fp_16):
    print("Verifying the converted model")
    onnx_output, onnx_inference = predict_onnx(model_path, fp_16, model_dimensions)
    
    print("ONNX inference time:", onnx_inference, "ms")
    
    # Calculate MSE (Mean Squared Error)
    mse = np.mean((onnx_output - compared_outputs) ** 2)
    print("MSE between ONNX and TensorRT outputs:", mse)

    # Calculate MAE (Mean Absolute Error)
    mae = np.mean(np.abs(onnx_output - compared_outputs))
    print("MAE between ONNX and TensorRT outputs:", mae)
    return 

# any other chanes for fp_16 to work?
def predict_onnx(model_path, fp_16, input_shape):
    
    # random_input = np.random.randn(*input_shape).astype(np.float32)

    # # Run inference with ONNX
    # input_name = onnx_session.get_inputs()[0].name
    # onnx_output = onnx_session.run(None, {input_name: random_input})
    onnx_session = ort.InferenceSession(model_path,providers=["CUDAExecutionProvider"])
    
    if fp_16:
        random_input = np.random.randn(input_shape).astype(np.float16)
    else:
        random_input = np.random.randn(input_shape).astype(np.float32)
        
    input_name = onnx_session.get_inputs()[0].name
    tic = time.perf_counter_ns()
    # results_ort = session.run([out.name for out in session.get_outputs()], {session.get_inputs()[0].name: x_test})
    # results_ort = onnx_session.run([out.name for out in session.get_outputs()], {session.get_inputs()[0].name: model_test})
    onnx_output = onnx_session.run(None, {input_name: random_input})
    toc = time.perf_counter_ns()
    onnx_output = onnx_output[0]
    # onnx_output= np.array(onnx_output)
    return onnx_output, (toc - tic) / 1e6

# given the predictions from the original model and the converted model, check if they are consistent
# shape of predictions_original and converted_results should be the same
# only checks for the predicted class (aka the argmax)
# takes in two 2D arrays: first dimension is the number of samples,  second dimension is the number of classes and values correspond to confidence
def checkPredictionConsistency(predictions_original, converted_results):
    for n in range(predictions_original.shape[0]):
        if np.argmax(predictions_original[n]) != np.argmax(converted_results[n]):
            print(f"Original: {np.argmax(predictions_original[n])}, ONNX: {np.argmax(converted_results[n])}")
            print(f"{predictions_original[n]}, \n{converted_results[n]}")
            print("=====================================")
            raise ValueError("Predictions are not consistent")

    print("All predictions are consistent")

# given the predictions from the original model and the converted model, check if they are consistent
# shape of predictions_original and converted_results should be the same
# only checks for the difference in confidence
# takes in two 2D arrays: first dimension is the number of samples,  second dimension is the number of classes and values correspond to confidence
# tolerance: the maximum difference in confidence that is allowed
def checkConfidenceConsistency(predictions_original, converted_results, tolerance=1e-5):
    np.testing.assert_allclose(predictions_original, converted_results,atol=tolerance)
    # for n in range(predictions_original.shape[0]):
    #     if not np.allclose(predictions_original[n], converted_results[n], atol=tolerance):
    #         print(f"Original: \t {predictions_original[n]}, \nONNX: \t{converted_results[n]}")
    #         print("=====================================")
    #         return

    print("All confidence percentages are consistent")