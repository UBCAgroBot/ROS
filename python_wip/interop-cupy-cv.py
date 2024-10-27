import cupy as cp
import cv2
import numpy as np

# Create a custom CUDA stream using CuPy
cuda_stream = cp.cuda.Stream()

# Allocate a GPU array using CuPy
cupy_array = cp.random.random((224, 224, 3), dtype=cp.float32)

# Perform some CuPy operations on the custom stream
with cuda_stream:
    cupy_array = cp.sqrt(cupy_array)  # Example CuPy operation

# Sync the stream to ensure CuPy operations are done before OpenCV operation
cuda_stream.synchronize()

# Convert CuPy array to a NumPy array (on the CPU)
# OpenCV doesn't natively support CuPy arrays, so transfer data back to host
numpy_array = cp.asnumpy(cupy_array)

# Convert NumPy array to OpenCV's GPU Mat
gpu_mat = cv2.cuda_GpuMat()
gpu_mat.upload(numpy_array)

# Perform an OpenCV CUDA operation
# OpenCV CUDA functions generally don't support custom streams directly
gpu_mat = cv2.cuda.resize(gpu_mat, (128, 128))

# Optionally, download the result back to the CPU
result = gpu_mat.download()

# cropping in cupy:
import cupy as cp

# Assume you have a CuPy array (image) of shape (height, width, channels)
image = cp.random.rand(224, 224, 3).astype(cp.float32)  # Example image

# Define the crop region (x, y, width, height)
x, y, w, h = 50, 50, 100, 100

# Crop the image (this works similarly to NumPy slicing)
cropped_image = image[y:y+h, x:x+w, :]