from numba import cuda
import numpy as np

# Define a CUDA kernel function
@cuda.jit
def matrix_addition_kernel(a, b, result):
    i, j = cuda.grid(2)
    if i < result.shape[0] and j < result.shape[1]:
        result[i, j] = a[i, j] + b[i, j]

# Initialize NumPy arrays
a = np.random.rand(32, 32).astype(np.float32)
b = np.random.rand(32, 32).astype(np.float32)
result = np.zeros_like(a)

# Allocate arrays on the GPU
a_device = cuda.to_device(a)
b_device = cuda.to_device(b)
result_device = cuda.to_device(result)

# Define the grid size for the kernel execution
threads_per_block = (16, 16)
blocks_per_grid = (a.shape[0] // threads_per_block[0] + 1, a.shape[1] // threads_per_block[1] + 1)

# Launch the CUDA kernel
matrix_addition_kernel[blocks_per_grid, threads_per_block](a_device, b_device, result_device)

# Copy the result back to the host (CPU)
result = result_device.copy_to_host()

print(result)
