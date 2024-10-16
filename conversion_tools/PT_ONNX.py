import os
import time
import argparse
import torch
import torch.onnx
import onnx
import numpy as np
from ultralytics import YOLO
OPSET_VERS = 13

def convert_pytorch_to_onnx(model_path="/home/user/Downloads/model.pt", output_path="/home/user/Downloads/model.onnx", FP16_mode=False, constant_folding=False, gs_optimize=False, model_dimensions=None, verify=True, verbose=False):
    if not os.path.exists(model_path):
        raise ValueError(f"Model file not found at {model_path}")
    if not output_path.endswith(".onnx"):
        raise ValueError("Output path should end with .onnx")
    if verify and model_dimensions is None:
        raise ValueError("Model dimensions are required for verification")
    
    print("Loading the PyTorch model")
    model = YOLO(model_path)
    
    model.eval().cuda()
    input_data = torch.randn(model_dimensions).cuda()
    
    if FP16_mode:
        model.half()
        input_data.half()
    
    tic = time.perf_counter_ns()
    torch_out = model(input_data)
    toc = time.perf_counter_ns()

    # names might be wrong
    print("Exporting model to ONNX format") 
    torch.onnx.export(model, input_data, output_path, 
                        verbose=verbose, 
                        opset_version=OPSET_VERS, 
                        export_params=True,
                        do_constant_folding=constant_folding,
                        input_names = ['input'], 
                        output_names = ['output'], 
                        )
    
    model = onnx.load(output_path)
    onnx.checker.check_model(model)
    
    if gs_optimize:
        from ONNX_GS import optimize_onnx
        output_path = optimize_onnx(output_path)
    
    print(model.graph)
    print("Model converted successfully")
    
    if verify:
        print("PyTorch inference time:", (toc - tic) / 1e6, "ms")
        from ONNX_Verify import verify_onnx
        verify_onnx(model_path, np.array(torch_out), model_dimensions) # torch.out.numpy()
    
    print(f"Converted ONNX model saved at {output_path}")
    return

if __name__ == "__main__":
    print("Usage: python3 PT_ONNX.py --model_path=/home/user/Downloads/model.onnx --output_path=/home/user/Downloads/model.engine --FP16=False --constant_folding=True --gs_optimize=False --model_dimensions=(1, 3, 448, 1024) --verify=True --verbose=False")
    
    parser = argparse.ArgumentParser(description='Convert PyTorch model to ONNX')
    parser.add_argument('--model_path', type=str, default="/home/user/Downloads/model.pt", required=False, help='Path to the PyTorch model file (.pt)')
    parser.add_argument('--output_path', type=str, default="/home/user/Downloads/model.onnx", required=False, help='Path to save the converted ONNX model file (.onnx)')
    parser.add_argument('--FP16', type=bool, default=False, help="FP16 precision mode")
    parser.add_argument('--constant_folding', type=bool, default=False, help="Apply constant folding opreation")
    parser.add_argument('--gs_optimize', type=bool, default=False, help='Use ONNX GraphSurgeon to optimize model after conversion')
    parser.add_argument('--model_dimensions', type=tuple, default=False, help="Model input dimensions")
    parser.add_argument('--verify', type=bool, default=True, help="Verify the converted model")
    parser.add_argument('--verbose', type=bool, default=False, help="Verbose mode")
    args = parser.parse_args()
    
    convert_pytorch_to_onnx(args.model_path, args.output_path, args.FP16, args.constant_folding, args.gs_optimize, args.model_dimensions, args.verify, args.verbose)