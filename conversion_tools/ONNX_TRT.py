import argparse
import onnx
import tensorrt as trt
import numpy as np
# import pycuda.driver as cuda
# import pycuda.autoinit

def get_max_memory():
    total, free = cuda.mem_get_info()
    max_mem = free * 0.95

    print(f"Total GPU memory: {total / (1024**2)} MB")
    print(f"Free GPU memory: {free / (1024**2)} MB")
    print(f"Max memory to use: {max_mem / (1024**2)} MB")
    return max_mem

def convert_onnx_to_trt(model_path, output_path, batch_size):
    # # Simplify the ONNX model (optional)
    # print("Loading the ONNX model")
    # onnx_model = onnx.load(model_path)
    # graph = gs.import_onnx(onnx_model)
    # graph.toposort()
    # onnx_model = gs.export_onnx(graph)
    
    TRT_LOGGER = trt.Logger(trt.Logger.WARNING)
    builder = trt.Builder(TRT_LOGGER)
    network = builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH))
    parser = trt.OnnxParser(network, TRT_LOGGER)
    
    with open(model_path, 'rb') as model_file:
        print("Parsing ONNX model")
        if not parser.parse(model_file.read()):
            for error in range(parser.num_errors):
                print(parser.get_error(error))
            return

    config = builder.create_builder_config()
    # config.fp16_mode = FP16_mode
    config.max_batch_size = batch_size
    config.max_workspace_size = 15

    print("Building TensorRT engine. This may take a few minutes.")
    engine = builder.build_cuda_engine(network, config)
    
    # Serialize the TensorRT engine to a file
    with open(output_path, 'wb') as f:
        f.write(engine.serialize())

    print("Engine built successfully")
    print(f"Converted TensorRT engine saved at {output_path}")    
    return engine, TRT_LOGGER

# def build_engine(self, onnx_file_path):
#     TRT_LOGGER = trt.Logger(trt.Logger.WARNING)
#     with trt.Builder(TRT_LOGGER) as builder, \
#             builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)) as network, \
#             trt.OnnxParser(network, TRT_LOGGER) as parser:

#         with open(onnx_file_path, 'rb') as model:
#             if not parser.parse(model.read()):
#                 for error in range(parser.num_errors):
#                     self.get_logger().error(parser.get_error(error))
#                 return None

#         config = builder.create_builder_config()
#         config.max_workspace_size = 1 << 30  # 1GB
#         builder.max_batch_size = 1

#         engine = builder.build_engine(network, config)
#         with open("model.trt", "wb") as f:
#             f.write(engine.serialize())
#         return engine


if __name__ == "__main__":
    print("Usage: python3 Onnx_TensorRT.py <model_path> <output_path> FP16_mode batch_size input_shape")
    print("Example: python3 Onnx_TensorRT.py ./model.onnx ./model_trt.trt True 1 (1, 3, 224, 224)")
    
    parser = argparse.ArgumentParser(description='Convert Onnx model to TensorRT')
    parser.add_argument('--modelpath', type=str, default="./model.onnx", required=False, help='Path to the PyTorch model file (.pt)')
    parser.add_argument('--outputpath', type=str, default="model_trt.trt", required=False, help='Path to save the converted TensorRT model file (.trt)')
    # parser.add_argument('--FP16_mode', type=bool, default=True, help='FP16 mode for TensorRT')
    parser.add_argument('--batch_size', type=int, default=1, help='Batch size for TensorRT')
    args = parser.parse_args()
    
    convert_onnx_to_trt(args.modelpath, args.outputpath, args.batch_size)