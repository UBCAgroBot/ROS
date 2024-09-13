import argparse
import tensorrt as trt
import pycuda.driver as cuda
import pycuda.autoinit  # Automatically initializes CUDA driver
# import numpy as np
import time
import torch

# allocates input/ouput buffers for the TensorRT engine inference
def allocate_buffers(engine):
    inputs = []
    outputs = []
    bindings = []
    stream = cuda.Stream()

    for binding in engine:
        size = trt.volume(engine.get_binding_shape(binding)) * engine.max_batch_size
        dtype = trt.nptype(engine.get_binding_dtype(binding))

        # Allocate host and device buffers
        host_mem = cuda.pagelocked_empty(size, dtype)
        device_mem = cuda.mem_alloc(host_mem.nbytes)

        # Append the device buffer to device bindings
        bindings.append(int(device_mem))

        # Append to the appropriate list
        if engine.binding_is_input(binding):
            inputs.append((host_mem, device_mem))
        else:
            outputs.append((host_mem, device_mem))

    return inputs, outputs, bindings, stream

# performs inference on the input data using the TensorRT engine
def infer(engine, inputs, outputs, bindings, stream, input_data):
    # Transfer input data to the device
    np.copyto(inputs[0][0], input_data.ravel())
    cuda.memcpy_htod_async(inputs[0][1], inputs[0][0], stream)

    # Execute the model
    context = engine.create_execution_context()
    context.execute_async_v2(bindings=bindings, stream_handle=stream.handle)

    # Transfer predictions back from the GPU
    cuda.memcpy_dtoh_async(outputs[0][0], outputs[0][1], stream)

    # Wait for the stream to complete the operation
    stream.synchronize()

    return outputs[0][0]

# tests a TensorRT engine file by performing inference and checking outputs
def test_trt_engine(trt_engine_path='model_trt.trt', input_shape=(1,3,224,224), input_data=None, expected_output=None):
    # Load the TensorRT engine from the file
    with open(trt_engine_path, "rb") as f:
        runtime = trt.Runtime(trt.Logger(trt.Logger.WARNING))
        engine = runtime.deserialize_cuda_engine(f.read())

    # Allocate buffers for inference
    inputs, outputs, bindings, stream = allocate_buffers(engine)

    # Generate random input data if not provided
    if input_data is None:
        input_data = np.random.rand(*input_shape).astype(np.float32)

    # Perform inference using the TensorRT engine
    output = infer(engine, inputs, outputs, bindings, stream, input_data)

    # Print the inference result
    print("Inference output:", output)

    # Compare with expected output if provided
    if expected_output is not None:
        if np.allclose(output, expected_output, rtol=1e-3, atol=1e-3):
            print("The inference result matches the expected output.")
            return True
        else:
            print("The inference result does not match the expected output.")
            return False
    else:
        print("No expected output provided. Unable to verify accuracy.")
        return True  # Pass as long as inference ran without errors

if __name__ == "__main__":
    print("Usage: python3 TensorRT_test.py <trt_engine_path> <input_shape> <input_data> <expected_output>")
    print("Example: python3 TensorRT_test.py model.trt (1, 3, 224, 224) None None")
    
    if len(sys.argv) < 2:
        test_trt_engine()
    else:
        for i in range(len(sys.argv), 5):
            sys.argv.append(None)
            test_trt_engine(*sys.argv[1:5])


def benchmark_trt_model(trt_engine_path):
    TRT_LOGGER = trt.Logger(trt.Logger.WARNING)
    with open(trt_engine_path, 'rb') as f, trt.Runtime(TRT_LOGGER) as runtime:
        engine = runtime.deserialize_cuda_engine(f.read())
    
    context = engine.create_execution_context()
    
    # Example input
    input_shape = (1, 3, 224, 224)
    input_data = torch.randn(input_shape).cuda()

    # Allocate buffers
    inputs, outputs, bindings = [], [], []
    for binding in engine:
        size = trt.volume(engine.get_binding_shape(binding)) * engine.max_batch_size
        dtype = trt.nptype(engine.get_binding_dtype(binding))
        host_mem = torch.empty(size, dtype=torch.float32).cuda()
        inputs.append(host_mem)
        bindings.append(int(host_mem.data_ptr()))

    # Execute inference
    start_time = time.time()
    context.execute_v2(bindings=bindings)
    end_time = time.time()

    # Report time
    latency = end_time - start_time
    print(f"Model Inference Time: {latency * 1000:.2f} ms")

benchmark_trt_model(args.model)

## new:
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

# Load TensorRT model
TRT_LOGGER = trt.Logger(trt.Logger.WARNING)

class TRTInference:
    def __init__(self, engine_path):
        self.engine = self.load_engine(engine_path)
        self.context = self.engine.create_execution_context()

    def load_engine(self, engine_path):
        with open(engine_path, 'rb') as f, trt.Runtime(TRT_LOGGER) as runtime:
            return runtime.deserialize_cuda_engine(f.read())

    def allocate_buffers(self):
        inputs = []
        outputs = []
        bindings = []
        stream = cuda.Stream()

        for binding in self.engine:
            size = trt.volume(self.engine.get_binding_shape(binding)) * self.engine.max_batch_size
            dtype = trt.nptype(self.engine.get_binding_dtype(binding))
            host_mem = cuda.pagelocked_empty(size, dtype)
            device_mem = cuda.mem_alloc(host_mem.nbytes)
            bindings.append(int(device_mem))

            if self.engine.binding_is_input(binding):
                inputs.append({'host': host_mem, 'device': device_mem})
            else:
                outputs.append({'host': host_mem, 'device': device_mem})

        return inputs, outputs, bindings, stream

    def infer(self, image, inputs, outputs, bindings, stream):
        np.copyto(inputs[0]['host'], image.ravel())

        # Transfer input data to the GPU.
        cuda.memcpy_htod_async(inputs[0]['device'], inputs[0]['host'], stream)

        # Run inference.
        self.context.execute_async_v2(bindings=bindings, stream_handle=stream.handle)

        # Transfer predictions back from GPU.
        cuda.memcpy_dtoh_async(outputs[0]['host'], outputs[0]['device'], stream)
        stream.synchronize()

        return outputs[0]['host']

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

# pip install pycuda onnxruntime numpy opencv-python

# Preprocessing: The image is resized and normalized to be input into the model.
# Inference: Uses TensorRT to make predictions on the preprocessed image.
# Metrics:

#     Intersection over Union (IoU) to measure accuracy.
#     Centroid offset, which checks the difference in the center of predicted and ground truth bounding boxes.
#     Inference time for each image.

# Ground Truth Parsing: The ground truth bounding boxes are read from the text file provided.

# Update paths for the .trt model, test images, and ground truth file.
# Ensure the bounding box coordinates are converted between the format of the ground truth and model output if necessary.
# Expand with more metrics such as precision, recall, or F1 score based on IoU thresholds if relevant.


import tensorrt as trt
import pycuda.driver as cuda
import pycuda.autoinit
import numpy as np

def allocate_buffers(engine):
    """
    Allocates input/output buffers for TensorRT engine inference.
    Args:
        engine: The TensorRT engine.
    Returns:
        inputs: List of input GPU buffers.
        outputs: List of output GPU buffers.
        bindings: List of bindings for the model.
        stream: CUDA stream for the inference.
    """
    inputs = []
    outputs = []
    bindings = []
    stream = cuda.Stream()

    for binding in engine:
        size = trt.volume(engine.get_binding_shape(binding)) * engine.max_batch_size
        dtype = trt.nptype(engine.get_binding_dtype(binding))

        # Allocate host and device buffers
        host_mem = cuda.pagelocked_empty(size, dtype)
        device_mem = cuda.mem_alloc(host_mem.nbytes)

        # Append the device buffer to device bindings
        bindings.append(int(device_mem))

        # Append to the appropriate list
        if engine.binding_is_input(binding):
            inputs.append((host_mem, device_mem))
        else:
            outputs.append((host_mem, device_mem))

    return inputs, outputs, bindings, stream

def infer(engine, inputs, outputs, bindings, stream, input_data):
    """
    Performs inference on the input data using the TensorRT engine.
    Args:
        engine: The TensorRT engine.
        inputs: List of input buffers.
        outputs: List of output buffers.
        bindings: List of bindings for the model.
        stream: CUDA stream for the inference.
        input_data: The data to be used as input for the model.
    Returns:
        output: The model's output.
    """
    # Transfer input data to the device
    np.copyto(inputs[0][0], input_data.ravel())
    cuda.memcpy_htod_async(inputs[0][1], inputs[0][0], stream)

    # Execute the model
    context = engine.create_execution_context()
    context.execute_async_v2(bindings=bindings, stream_handle=stream.handle)

    # Transfer predictions back from the GPU
    cuda.memcpy_dtoh_async(outputs[0][0], outputs[0][1], stream)

    # Wait for the stream to complete the operation
    stream.synchronize()

    return outputs[0][0]

def test_trt_engine(trt_engine_path, input_shape, input_data=None, expected_output=None):
    """
    Tests a TensorRT engine file by performing inference and checking outputs.
    Args:
        trt_engine_path: Path to the TensorRT engine file.
        input_shape: Shape of the input data.
        input_data: Optional input data. If None, random data will be generated.
        expected_output: Optional expected output. If provided, it will be compared to the TensorRT inference result.
    Returns:
        True if the engine works and inference results match the expected output (if provided), otherwise False.
    """
    # Load the TensorRT engine from the file
    with open(trt_engine_path, "rb") as f:
        runtime = trt.Runtime(trt.Logger(trt.Logger.WARNING))
        engine = runtime.deserialize_cuda_engine(f.read())

    # Allocate buffers for inference
    inputs, outputs, bindings, stream = allocate_buffers(engine)

    # Generate random input data if not provided
    if input_data is None:
        input_data = np.random.rand(*input_shape).astype(np.float32)

    # Perform inference using the TensorRT engine
    output = infer(engine, inputs, outputs, bindings, stream, input_data)

    # Print the inference result
    print("Inference output:", output)

    # Compare with expected output if provided
    if expected_output is not None:
        if np.allclose(output, expected_output, rtol=1e-3, atol=1e-3):
            print("The inference result matches the expected output.")
            return True
        else:
            print("The inference result does not match the expected output.")
            return False
    else:
        print("No expected output provided. Unable to verify accuracy.")
        return True  # Pass as long as inference ran without errors

# Example usage:
# Test TensorRT engine using random input
trt_engine_path = "model.trt"  # Path to your TensorRT engine file
input_shape = (1, 3, 224, 224)  # Adjust based on your model's input shape

test_trt_engine(trt_engine_path, input_shape)

input_data = np.random.rand(1, 3, 224, 224).astype(np.float32)  # Example input, replace with actual data
expected_output = np.random.rand(1, 1000).astype(np.float32)  # Example expected output (optional)
test_trt_engine("path_to_your_model.trt", (1, 3, 224, 224), input_data=input_data, expected_output=expected_output)

### new!

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

# Load TensorRT model
TRT_LOGGER = trt.Logger(trt.Logger.WARNING)

class TRTInference:
    def __init__(self, engine_path):
        self.engine = self.load_engine(engine_path)
        self.context = self.engine.create_execution_context()

    def load_engine(self, engine_path):
        with open(engine_path, 'rb') as f, trt.Runtime(TRT_LOGGER) as runtime:
            return runtime.deserialize_cuda_engine(f.read())

    def allocate_buffers(self):
        inputs = []
        outputs = []
        bindings = []
        stream = cuda.Stream()

        for binding in self.engine:
            size = trt.volume(self.engine.get_binding_shape(binding)) * self.engine.max_batch_size
            dtype = trt.nptype(self.engine.get_binding_dtype(binding))
            host_mem = cuda.pagelocked_empty(size, dtype)
            device_mem = cuda.mem_alloc(host_mem.nbytes)
            bindings.append(int(device_mem))

            if self.engine.binding_is_input(binding):
                inputs.append({'host': host_mem, 'device': device_mem})
            else:
                outputs.append({'host': host_mem, 'device': device_mem})

        return inputs, outputs, bindings, stream

    def infer(self, image, inputs, outputs, bindings, stream):
        np.copyto(inputs[0]['host'], image.ravel())

        # Transfer input data to the GPU.
        cuda.memcpy_htod_async(inputs[0]['device'], inputs[0]['host'], stream)

        # Run inference.
        self.context.execute_async_v2(bindings=bindings, stream_handle=stream.handle)

        # Transfer predictions back from GPU.
        cuda.memcpy_dtoh_async(outputs[0]['host'], outputs[0]['device'], stream)
        stream.synchronize()

        return outputs[0]['host']

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