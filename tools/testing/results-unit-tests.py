# should reference the results class?

import unittest
import numpy as np
from my_package.bbox_node import BBoxNode

# class for handling results with class methods for MSE, etc.

# get outputs of ultralytics and assert function diff between ultralytics and module vlaue is less than 0.5%
# unit tests for also length of list (object count)

def calculate_iou(boxA, boxB):
    """
    Calculate the Intersection over Union (IoU) between two bounding boxes.
    """
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])

    # Compute the area of intersection rectangle
    interArea = max(0, xB - xA) * max(0, yB - yA)

    # Compute the area of both bounding boxes
    boxAArea = (boxA[2] - boxA[0]) * (boxA[3] - boxA[1])
    boxBArea = (boxB[2] - boxB[0]) * (boxB[3] - boxB[1])

    # Compute the Intersection over Union (IoU)
    iou = interArea / float(boxAArea + boxBArea - interArea)

    return iou

def calculate_centroid(bbox):
    """
    Calculate the centroid of the bounding box.
    """
    x_min, y_min, x_max, y_max = bbox
    centroid_x = (x_min + x_max) / 2.0
    centroid_y = (y_min + y_max) / 2.0
    return centroid_x, centroid_y

class TestBBoxNode(unittest.TestCase):
    def setUp(self):
        # Create an instance of the BBoxNode
        self.node = BBoxNode()

        # Ground truth bounding box for comparison
        self.ground_truth_bbox = [110, 160, 210, 260]  # Example actual bounding box

    def test_bbox_output(self):
        # Simulate an image message and call run_inference
        predicted_bbox = self.node.run_inference(None)  # Normally, this would be the image message

        # Calculate IoU (Intersection over Union)
        iou = calculate_iou(predicted_bbox, self.ground_truth_bbox)
        self.assertGreater(iou, 0.5, "IoU is too low! Bounding box prediction is inaccurate.")

        # Calculate centroid offset
        predicted_centroid = calculate_centroid(predicted_bbox)
        ground_truth_centroid = calculate_centroid(self.ground_truth_bbox)

        offset_x = abs(predicted_centroid[0] - ground_truth_centroid[0])
        offset_y = abs(predicted_centroid[1] - ground_truth_centroid[1])

        self.assertLess(offset_x, 15, f"X centroid offset too large: {offset_x}")
        self.assertLess(offset_y, 15, f"Y centroid offset too large: {offset_y}")

    def test_performance_report(self):
        # Simulate inference results for multiple images
        predicted_bboxes = [
            [100, 150, 200, 250],
            [110, 160, 210, 260],
            [105, 155, 205, 255]
        ]
        ground_truth_bboxes = [
            [110, 160, 210, 260],
            [110, 160, 210, 260],
            [110, 160, 210, 260]
        ]

        total_iou = 0
        total_offset_x = 0
        total_offset_y = 0

        for pred_bbox, gt_bbox in zip(predicted_bboxes, ground_truth_bboxes):
            iou = calculate_iou(pred_bbox, gt_bbox)
            total_iou += iou

            pred_centroid = calculate_centroid(pred_bbox)
            gt_centroid = calculate_centroid(gt_bbox)

            total_offset_x += abs(pred_centroid[0] - gt_centroid[0])
            total_offset_y += abs(pred_centroid[1] - gt_centroid[1])

        # Generate performance metrics
        avg_iou = total_iou / len(predicted_bboxes)
        avg_offset_x = total_offset_x / len(predicted_bboxes)
        avg_offset_y = total_offset_y / len(predicted_bboxes)

        print(f"Performance Report:")
        print(f"Average IoU: {avg_iou:.2f}")
        print(f"Average Centroid Offset X: {avg_offset_x:.2f}")
        print(f"Average Centroid Offset Y: {avg_offset_y:.2f}")

        # Assert performance is within acceptable limits
        self.assertGreater(avg_iou, 0.5, "Average IoU is too low!")
        self.assertLess(avg_offset_x, 15, "Average X centroid offset is too large!")
        self.assertLess(avg_offset_y, 15, "Average Y centroid offset is too large!")

if __name__ == '__main__':
    unittest.main()

# colcon test --packages-select my_package

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

def preprocess_image(image_path, input_shape):
    """Preprocess image for inference."""
    image = cv2.imread(image_path)
    image_resized = cv2.resize(image, (input_shape[1], input_shape[0]))
    image = np.asarray(image_resized).astype(np.float32)
    return np.transpose(image, (2, 0, 1)) / 255.0  # CHW format and normalized

def run_benchmark(trt_model_path, test_images, ground_truth_path):
    """Run benchmark on the model."""
    # Load ground truth
    ground_truth_bboxes = load_ground_truth(ground_truth_path)

    # Initialize TensorRT inference
    trt_infer = TRTInference(trt_model_path)
    inputs, outputs, bindings, stream = trt_infer.allocate_buffers()

    inference_times = []
    iou_scores = []
    centroid_offsets = []

    for idx, img_path in enumerate(test_images):
        # Preprocess image
        image = preprocess_image(img_path, (300, 300))  # Adjust size as needed

        # Perform inference and measure time
        start_time = time.time()
        pred_bbox = trt_infer.infer(image, inputs, outputs, bindings, stream)
        inference_time = time.time() - start_time

        # Compute IoU, centroid offset
        gt_bbox = ground_truth_bboxes[idx]
        iou_score = iou(pred_bbox, gt_bbox)
        offset = calculate_centroid_offset(pred_bbox, gt_bbox)

        # Store results
        inference_times.append(inference_time)
        iou_scores.append(iou_score)
        centroid_offsets.append(offset)

    # Summary of benchmark
    print(f"Average Inference Time: {np.mean(inference_times):.4f} seconds")
    print(f"Average IoU: {np.mean(iou_scores) * 100:.2f}%")
    print(f"Average Centroid Offset: {np.mean(centroid_offsets):.2f}%")

if __name__ == "__main__":
    trt_model_path = "model.trt"  # Replace with your TensorRT model path
    test_images = ["test1.jpg", "test2.jpg"]  # Replace with your test images
    ground_truth_path = "ground_truth.txt"  # Replace with your ground truth file path

    run_benchmark(trt_model_path, test_images, ground_truth_path)

# Create performance report based on relative bounding box centroid for sample images (accuracy %, error offset %, etc.)
