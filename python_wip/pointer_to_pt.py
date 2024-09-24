import torch
import pycuda.driver as cuda

# Example CUDA buffer size (you should match this with your actual output size)
output_size = (1, 1000)  # Example: TensorRT output of shape [1, 1000]

# Convert CUDA memory pointer (from IPC) to a PyTorch tensor
def cuda_pointer_to_torch_tensor(cuda_ptr, shape, dtype=torch.float32):
    # Convert the raw pointer to PyTorch tensor (in GPU memory)
    tensor = torch.from_blob(cuda_ptr, shape, dtype=dtype, device='cuda')
    return tensor

# In your post-processing node, after receiving the CUDA IPC handle
ipc_handle = cuda.IPCHandle(ipc_handle_bytes)
d_output = ipc_handle.open(cuda.Context.get_current())

# Convert the CUDA device pointer to a PyTorch tensor
output_tensor = cuda_pointer_to_torch_tensor(d_output, output_size)
