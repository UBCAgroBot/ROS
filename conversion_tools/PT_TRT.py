import os
import time
import argparse
import torch
from torch2trt import torch2trt
import numpy as np
from ultralytics import YOLO

def convert_pt_to_trt(model_path='/home/user/Downloads/model.pt', output_path='/home/user/Downloads/model.pth', FP16_mode=False, input_shape=(1, 3, 448, 1024), verify=True):
    if not os.path.exists(model_path):
        raise ValueError(f"Model file not found at {model_path}")
    if not output_path.endswith(".pth"):
        raise ValueError("Output path should end with .pth")
    if input_shape is None:
        raise ValueError("Input shape is required for conversion")
    
    print("Loading the PyTorch model")
    model = YOLO(model_path).cuda().eval()
    
    if FP16_mode:
        data = torch.randn(input_shape).cuda().half()
        print("Building TensorRT engine. This may take a few minutes.")
        model_trt = torch2trt(model, [data], fp16_mode=True)
        model = model.half()
    else:
        data = torch.randn(input_shape).cuda()
        print("Building TensorRT engine. This may take a few minutes.")
        model_trt = torch2trt(model, [data])
    
    print("Engine built successfully")  
    
    if verify:
        tic = time.perf_counter_ns()
        output_trt = model_trt(data)
        toc = time.perf_counter_ns()
        print(f"TensorRT Inference time: {(toc - tic)/1e6}ms")
        
        tic = time.perf_counter_ns()
        output = model(data)
        toc = time.perf_counter_ns()
        print(f"PyTorch Inference time: {(toc - tic)/1e6}ms")
        
        # Mean Squared Error (MSE)
        mse = torch.mean((output - output_trt) ** 2)
        print(f"MSE between PyTorch and TensorRT outputs: {mse.item()}")

        # Mean Absolute Error (MAE)
        mae = torch.mean(torch.abs(output - output_trt))
        print(f"MAE between PyTorch and TensorRT outputs: {mae.item()}")
    
    torch.save(model_trt.state_dict(), output_path)
    print(f"Converted TensorRT engine saved at {output_path}")    
    return 

if __name__ == "__main__":
    print("Usage: python3 PT_TRT.py --model_path=/home/user/Downloads/model.onnx --output_path=/home/user/Downloads/model.engine --FP16_mode=False --input_shape=(1, 3, 448, 1024) --verify=True")
    
    parser = argparse.ArgumentParser(description='Convert PyTorch model to TensorRT')
    parser.add_argument('--modelpath', type=str, required=False, help='Path to the PyTorch model file (.pt)')
    parser.add_argument('--outputpath', type=str, required=False, help='Path to save the converted TensorRT model file (.trt)')
    parser.add_argument('--FP16_mode', type=bool, default=True, help='FP16 mode for TensorRT')
    parser.add_argument('--input_shape', type=tuple, default=(1, 3, 448, 1024), help='Input shape for TensorRT')
    parser.add_argument('--verify', type=bool, default=True, help='Verify the converted model')
    args = parser.parse_args()
    
    convert_pt_to_trt(args.modelpath, args.outputpath, args.FP16_mode, args.input_shape, args.verify)