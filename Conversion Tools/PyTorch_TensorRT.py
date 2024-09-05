import sys
import torch
from torch2trt import torch2trt
import pycuda.driver as cuda
import pycuda.autoinit
import numpy as np

def get_max_memory():
    total, free = cuda.mem_get_info()
    max_mem = free * 0.95

    print(f"Total GPU memory: {total / (1024**2)} MB")
    print(f"Free GPU memory: {free / (1024**2)} MB")
    print(f"Max memory to use: {max_mem / (1024**2)} MB")
    return max_mem

def convert_pt_to_trt(model_path='./model.pt', output_path='./model_trt.trt', FP16_mode=True, batch_size=1, input_shape=(1, 3, 224, 224)):
    print("Loading the PyTorch model")
    model = torch.load(model_path)
    model.eval()

    input_data = torch.randn(input_shape).cuda()
    print("Building TensorRT engine. This may take a few minutes.")
    model_trt = torch2trt(model, [input_data], fp16_mode=FP16_mode, max_batch_size=batch_size, max_workspace_size=get_max_memory())

    with open(output_path, 'wb') as f:
        f.write(model_trt.engine.serialize())
    
    print("Engine built successfully")
    print(f"Converted TensorRT engine saved at {output_path}")    
    return model_trt


if __name__ == "__main__":
    print("Usage: python3 PyTorch_TensorRT.py <model_path> <output_path> FP16_mode batch_size input_shape")
    print("Example: python3 PyTorch_TensorRT.py ./model.pt ./model_trt.trt True 1 (1, 3, 224, 224)")
    
    if len(sys.argv) < 2:
        convert_pt_to_trt()
    else:
        for i in range(len(sys.argv), 6):
            sys.argv.append(None)
            convert_pt_to_trt(*sys.argv[1:6])