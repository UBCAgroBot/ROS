import time
import torch
import torchvision.transforms as T
import cv2
import numpy as np
import cupy as cp

# Load image (this will be used for all methods)
image_path = 'image.png'
image = cv2.imread(image_path, cv2.IMREAD_UNCHANGED)

# ROI coordinates for cropping (just placeholders, you need to adjust)
roi_x, roi_y, roi_w, roi_h = 50, 50, 200, 200

### CuPy method
def preprocess_with_cupy():
    # Upload image to GPU with CuPy
    gpu_image = cp.asarray(image)

    # Remove alpha channel (RGBA -> RGB)
    gpu_image = gpu_image[:, :, :3]

    # Crop the image based on ROI
    gpu_image = gpu_image[roi_y:roi_y + roi_h, roi_x:roi_x + roi_w, :]

    # Resize the image to (224, 224)
    gpu_image = cp.array(cv2.resize(cp.asnumpy(gpu_image), (224, 224)))

    # Normalize pixel values to [0, 1] and transpose
    gpu_image = gpu_image.astype(cp.float32) / 255.0
    gpu_image = cp.transpose(gpu_image, (2, 0, 1))  # Transpose to (C, H, W)

    return gpu_image

### PyTorch method
def preprocess_with_pytorch():
    # Convert image to PyTorch tensor
    image_tensor = torch.from_numpy(image).cuda()

    # PyTorch transformation pipeline
    transform = T.Compose([
        T.Lambda(lambda x: x[:, :, :3]),  # Remove alpha channel
        T.Lambda(lambda x: x[roi_y:roi_y + roi_h, roi_x:roi_x + roi_w, :]),  # Crop
        T.Resize((224, 224)),
        T.ToTensor(),  # Convert to tensor and normalize
    ])

    # Apply transformations
    image_tensor = transform(image_tensor)

    return image_tensor

### OpenCV with CUDA method
def preprocess_with_opencv_cuda():
    # Upload image to GPU using cv2.cuda_GpuMat
    gpu_image = cv2.cuda_GpuMat()
    gpu_image.upload(image)

    # Remove alpha channel (RGBA -> RGB)
    gpu_image = gpu_image.get()[:, :, :3]
    gpu_image = cv2.cuda_GpuMat()
    gpu_image.upload(gpu_image)

    # Crop the image based on ROI
    gpu_image = gpu_image[roi_y:roi_y + roi_h, roi_x:roi_x + roi_w]

    # Resize the image to (224, 224)
    resized = cv2.cuda.resize(gpu_image, (224, 224))

    # Download back to CPU for normalization
    cpu_image = resized.download()

    # Normalize and transpose
    cpu_image = cpu_image.astype(np.float32) / 255.0
    cpu_image = np.transpose(cpu_image, (2, 0, 1))  # Transpose to (C, H, W)

    return cpu_image

# Measure execution time for CuPy
start_time = time.time()
gpu_image_cupy = preprocess_with_cupy()
cupy_time = time.time() - start_time

# Measure execution time for PyTorch
start_time = time.time()
gpu_image_pytorch = preprocess_with_pytorch()
pytorch_time = time.time() - start_time

# Measure execution time for OpenCV with CUDA
start_time = time.time()
gpu_image_opencv_cuda = preprocess_with_opencv_cuda()
opencv_cuda_time = time.time() - start_time

# Output results
print(f"CuPy Preprocessing Time: {cupy_time:.4f} seconds")
print(f"PyTorch Preprocessing Time: {pytorch_time:.4f} seconds")
print(f"OpenCV with CUDA Preprocessing Time: {opencv_cuda_time:.4f} seconds")

# CuPy Method: Uses CuPy for GPU-based operations. The image is uploaded to the GPU using CuPy, preprocessed (alpha channel removal, cropping, resizing, normalization), and the operation is timed.
# PyTorch Method: Uses PyTorch for GPU-based preprocessing, applying operations like cropping, resizing, and normalization using PyTorch’s transformation pipeline.
# OpenCV with CUDA Method: Uses OpenCV's CUDA functionality. The image is uploaded to the GPU with cv2.cuda_GpuMat and operations like cropping, resizing, and normalizing are performed.