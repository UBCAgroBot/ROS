import torch
from torch2trt import torch2trt
import pycuda.driver as cuda

def convert_torch_to_engine(model, x, engine_filename=None, fp16_mode=True):
    print("Parsing PyTorch model.")
    model_trt = torch2trt(model, [x], fp16_mode=fp16_mode)

    print("Building TensorRT engine. This may take a few minutes.")
    if engine_filename:
        with open(engine_filename, 'wb') as f:
            f.write(model_trt.engine.serialize())

    print("Completed building Engine.")
    return model_trt

model_name = input("Enter the path of the PyTorch model file (with .pth extension): ")
engine_name = input("Enter the name of the TensorRT engine file (without .engine extension): ")
precision = input("Enter the precision (FP32/FP16): ") # INT8 later
batch_size = int(input("Enter the maximum batch size: "))
input_dimensions = (224, 224) 

# Load the trained model
model = torch.load(model_name)
model.eval()

# Create example data
x = torch.randn((batch_size, 3, input_dimensions[0], input_dimensions[1])).cuda()

convert_torch_to_engine(model, x, f"{engine_name}.engine", precision == "FP16")