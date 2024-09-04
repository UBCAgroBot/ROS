import torch
import torchvision
import onnx

model_path = input("Enter the full/absolute path to the model: ")
model_name = input("Enter the name of the model: ")
input_dimensions = (224, 224)
model = torch.load(model_path)

model.eval()
x = torch.randn(1, 3, input_dimensions[0], input_dimensions[1])
traced_model = torch.jit.trace(model, x)

torch.onnx.export(traced_model,               
                  x,                         
                  f"{model_name}.onnx",   
                  export_params=True,        
                  opset_version=10,          
                  do_constant_folding=True,  
                  input_names = ['input'],   
                  output_names = ['output'], 
                  dynamic_axes={'input' : {0 : 'batch_size'},    
                                'output' : {0 : 'batch_size'}})

model = onnx.load(f"{model_name}.onnx")
print("Model converted successfully")
print(model.graph)

# improvements:
# Model Pruning: Pruning is a technique in deep learning where you remove the weights of certain neurons which are less important. This can help in reducing the size of the model and hence improve the performance during conversion.
# Quantization: Quantization is a process that reduces the numerical precision of the model's weights, which can lead to a significant reduction in both the memory requirement and computational cost of the model.
# Set the Appropriate Opset Version: The ONNX opset version corresponds to the set of operators and their versions supported. Newer opset versions can have optimizations that were not available in previous versions. You can set the opset version with the opset_version parameter in torch.onnx.export(). The latest version as of ONNX 1.8.0 is 13.