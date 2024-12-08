# c1, c2 = (int(x[0]), int(x[1])), (int(x[2]), int(x[3]))
# cv2.rectangle(img, c1, c2, color, thickness=2, lineType=cv2.LINE_AA)
# [batch size, channel number, height, width]
# cv2.dnn.NMSBoxes
# input shape (1,3,640,640) BCHW
# output shape (1,5,8400)

from ultralytics import YOLO

# Load a model
model = YOLO("yolov8n.pt")  # load an official model
model = YOLO("path/to/best.pt")  # load a custom trained model

# Export the model
model.export(format="onnx")

from ultralytics import YOLO

# Load the YOLOv8 model
model = YOLO("yolov8n.pt")

# Export the model to TensorRT format
model.export(format="engine")  # creates 'yolov8n.engine'

# Load the exported TensorRT model
tensorrt_model = YOLO("yolov8n.engine")

# Run inference
results = tensorrt_model("https://ultralytics.com/images/bus.jpg")

# logging style:
# PyTorch: starting from '/home/user/ROS/models/maize/Maize.pt' with input shape (1, 3, 640, 640) BCHW and output shape(s) (1, 5, 8400) (5.9 MB)
# ONNX: starting export with onnx 1.16.2 opset 12...
# ONNX: export success ✅ 1.7s, saved as '/home/user/ROS/models/maize/Maize.onnx' (11.7 MB)
# Export complete (3.8s)
# Results saved to /home/user/ROS/models/maize
# Predict:         yolo predict task=detect model=/home/user/ROS/models/maize/Maize.onnx imgsz=640  
# Validate:        yolo val task=detect model=/home/user/ROS/models/maize/Maize.onnx imgsz=640 data=config.yaml  

# settings for onnx: imgsz, half, dynamic, simplify, opset, batch
# engine: imgsz, half, dynamic, simplify, workspace, int8, batch

import os
import time
import argparse
import torch
import torch.onnx
import onnx
import numpy as np
from ultralytics import YOLO

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

# graphsurgeon
# import onnx_graphsurgeon as gs
# import onnx
# import argparse

# def optimize_onnx(model_path="/home/user/Downloads/model.onnx"):
#     print("Optimizing ONNX model")
#     model = onnx.load(model_path)
#     graph = gs.import_onnx(model)
    
#     print("Graph nodes before optimization:")
#     for node in graph.nodes:
#         print(node)

#     graph.cleanup().toposort()
#     graph.fold_constants()

#     model_path = model_path.replace(".onnx", "_optimized.onnx")
#     onnx.save(gs.export_onnx(graph), model_path)
#     return model_path

# if __name__ == "__main__":
#     print("Usage: python3 ONNX_GS.py --model_path=/home/user/Downloads/model.onnx")
    
#     parser = argparse.ArgumentParser(description='Optimize the ONNX model using GraphSurgeon')
#     parser.add_argument('--model_path', type=str, default="/home/user/Downloads/model.onnx", required=False, help='Path to the ONNX model file (.onnx)')
#     args = parser.parse_args()
    
#     optimize_onnx(args.model_path)

# # random chatgpt:
# import onnx
# import onnx_graphsurgeon as gs
# import numpy as np

# # Load the ONNX model
# onnx_model_path = "yolo_backbone.onnx"
# model = onnx.load(onnx_model_path)

# # Parse the model graph into GraphSurgeon graph
# graph = gs.import_onnx(model)

# # Display the graph nodes (optional, useful for inspection)
# print("Graph nodes before optimization:")
# for node in graph.nodes:
#     print(node)

# # Example: Remove Identity nodes (they are not needed for inference)
# graph.cleanup()

# # Example: Fold constant nodes
# # Constant folding can be used to simplify the graph by evaluating constant expressions at graph build time.
# for node in graph.nodes:
#     if node.op == "Add":
#         inputs_are_constants = all(isinstance(inp, gs.Constant) for inp in node.inputs)
#         if inputs_are_constants:
#             value = node.inputs[0].values + node.inputs[1].values
#             constant_node = gs.Constant(name=node.name, values=value)
#             graph.outputs = [constant_node]

# # Example: Fuse certain nodes, if applicable
# # In this case, you can fuse common patterns (like batch normalization, activation layers) if it's supported.
# # This is model-dependent, so it's an optional step. For simplicity, we omit specific fusion here.

# # Cleanup the graph to remove any orphaned nodes after the transformations
# graph.cleanup()
# graph.toposort()

# # Export the optimized ONNX model
# optimized_onnx_path = "yolo_backbone_optimized.onnx"
# onnx.save(gs.export_onnx(graph), optimized_onnx_path)

# print(f"Optimized model saved at {optimized_onnx_path}")
