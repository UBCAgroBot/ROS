import time
import numpy as np
import cv2
import pycuda.driver as cuda
import pycuda.autoinit
import tensorrt as trt
from onnxruntime import InferenceSession

# Helper functions
def load_ground_truth(file_path):
    """Load ground truth bounding boxes from text file."""
    with open(file_path, 'r') as f:
        bboxes = []
        for line in f:
            tokens = line.strip().split()
            cls, x_center, y_center, width, height = map(float, tokens)
            bboxes.append((cls, x_center, y_center, width, height))
    return bboxes

def iou(boxA, boxB):
    """Compute Intersection Over Union (IoU) between two bounding boxes."""
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])

    interArea = max(0, xB - xA + 1) * max(0, yB - yA + 1)
    boxAArea = (boxA[2] - boxA[0] + 1) * (boxA[3] - boxA[1] + 1)
    boxBArea = (boxB[2] - boxB[0] + 1) * (boxB[3] - boxB[1] + 1)

    iou = interArea / float(boxAArea + boxBArea - interArea)
    return iou

def calculate_centroid_offset(pred_box, gt_box):
    """Calculate percentage centroid offset between two boxes."""
    pred_center = (pred_box[0] + pred_box[2]) / 2, (pred_box[1] + pred_box[3]) / 2
    gt_center = (gt_box[0] + gt_box[2]) / 2, (gt_box[1] + gt_box[3]) / 2
    offset_x = abs(pred_center[0] - gt_center[0]) / (gt_box[2] - gt_box[0])
    offset_y = abs(pred_center[1] - gt_center[1]) / (gt_box[3] - gt_box[1])
    return (offset_x + offset_y) / 2 * 100

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