from numba import cuda
import numpy as np

@cuda.jit
def add_kernel(a, b, c):
    idx = cuda.grid(1)
    if idx < a.size:
        c[idx] = a[idx] + b[idx]

# Allocate data on the host
N = 1000
a = np.arange(N, dtype=np.float32)
b = np.arange(N, dtype=np.float32)
c = np.zeros_like(a)

# Allocate data on the device
a_gpu = cuda.to_device(a)
b_gpu = cuda.to_device(b)
c_gpu = cuda.device_array_like(a)

# Launch kernel
threads_per_block = 128
blocks_per_grid = (a.size + (threads_per_block - 1)) // threads_per_block
add_kernel[blocks_per_grid, threads_per_block](a_gpu, b_gpu, c_gpu)

# Copy result back to host
c_gpu.copy_to_host(c)

print(c)