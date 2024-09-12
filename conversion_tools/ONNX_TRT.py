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

def convert_onnx_to_trt(model_path, output_path, FP16, INT8, strip_weights, batch_size, verbose):
    # # Simplify the ONNX model (optional)
    # print("Loading the ONNX model")
    # onnx_model = onnx.load(model_path)
    # graph = gs.import_onnx(onnx_model)
    # graph.toposort()
    # onnx_model = gs.export_onnx(graph)
    
    if verbose:
        TRT_LOGGER = trt.Logger(trt.Logger.VERBOSE)
    else:
        TRT_LOGGER = trt.Logger(trt.Logger.WARNING)
        
    builder = trt.Builder(TRT_LOGGER)
    config = builder.create_builder_config()
    
    # Set cache
    cache = config.create_timing_cache(b"")
    config.set_timing_cache(cache, ignore_mismatch=False)
    
    # Set max workspace
    max_workspace = (1 << 30) # 15
    config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, max_workspace)
    
    network = builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH))
    parser = trt.OnnxParser(network, TRT_LOGGER)
    
    with open(model_path, 'rb') as model_file:
        print("Parsing ONNX model")
        if not parser.parse(model_file.read()):
            for error in range(parser.num_errors):
                print(parser.get_error(error))
    
    inputs = [network.get_input(i) for i in range(network.num_inputs)]
    outputs = [network.get_output(i) for i in range(network.num_outputs)]
    
    for input in inputs:
        print(f"Model {input.name} shape: {input.shape} {input.dtype}")
    for output in outputs:
        print(f"Model {output.name} shape: {output.shape} {output.dtype}") 

    if FP16:
        config.set_flag(trt.BuilderFlag.FP16)
    elif INT8:
        config.set_flag(trt.BuilderFlag.INT8)
    
    if strip_weights:
        config.set_flag(trt.BuilderFlag.STRIP_PLAN)

    # if batch_size > 1:
    #     # https://docs.nvidia.com/deeplearning/tensorrt/developer-guide/index.html#opt_profiles
    #     profile = builder.create_optimization_profile()
    #     min_shape = [1] + shape_input_model[-3:]
    #     opt_shape = [int(max_batch_size/2)] + shape_input_model[-3:]
    #     max_shape = shape_input_model
    #     for input in inputs:
    #         profile.set_shape(input.name, min_shape, opt_shape, max_shape)
    #     config.add_optimization_profile(profile)

    print("Building TensorRT engine. This may take a few minutes.")
    engine_bytes = builder.build_serialized_network(network, config) 
    with open(output_path, "wb") as f:    
        f.write(engine_bytes)

    print("Engine built successfully")
    print(f"Converted TensorRT engine saved at {output_path}")    


if __name__ == "__main__":
    print("Usage: python3 ONNX_TRT.py --model_path=/home/user/Downloads/model.onnx --output_path=/home/user/Downloads/model.engine --FP16=False --INT8=False --strip_weights=False --batch_size=1 --verbose=False ")
    
    parser = argparse.ArgumentParser(description='Convert Onnx model to TensorRT')
    parser.add_argument('--model_path', type=str, default="/home/user/Downloads/model.onnx", required=False, help='Path to the PyTorch model file (.pt)')
    parser.add_argument('--output_path', type=str, default="/home/user/Downloads/model.engine", required=False, help='Path to save the converted TensorRT model file (.trt)')
    parser.add_argument('--FP16', type=bool, default=False, help="FP16 precision mode")
    parser.add_argument('--INT8', type=bool, default=False, help="INT8 precision mode")
    parser.add_argument('--strip_weights', type=bool, default=False, help="Strip unnecessary weights")
    parser.add_argument('--batch_size', type=int, default=1, help='Batch size')
    parser.add_argument('--verbose', type=bool, default=False, help="Verbose TensorRT logging")
    args = parser.parse_args()
    
    convert_onnx_to_trt(args.model_path, args.output_path, args.FP16, args.INT8, args.strip_weights, args.batch_size, args.verbose)