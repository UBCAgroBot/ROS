import argparse
import onnx
import tensorrt as trt
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

# precision can be FP32, FP16, or INT8. The batch size is the maximum number of samples that can be processed in a single inference. The get_max_memory() function calculates the maximum memory that can be used by the TensorRT engine. The convert_onnx_to_engine() function converts the ONNX model to a TensorRT engine and saves it to a file. The engine is built with the specified precision and batch size.
def convert_onnx_to_trt(model_path="./model.onnx", output_path="model_trt.trt", FP16_mode = True, batch_size=1, input_shape=(1, 3, 224, 224)):
    print("Loading the ONNX model")
    onnx_model = onnx.load(model_path)
    
    # # Simplify the ONNX model (optional)
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

    # config.max_workspace_size = 1 << 30  # Adjust as needed
    # builder.max_workspace_size = get_max_memory()
    # builder.fp16_mode = fp16_mode
    # builder.max_batch_size = batch_size
    config = builder.create_builder_config()
    config.fp16_mode = FP16_mode
    config.max_batch_size = batch_size
    config.max_workspace_size = get_max_memory()

    print("Building TensorRT engine. This may take a few minutes.")
    engine = builder.build_cuda_engine(network, config)
    
    # Serialize the TensorRT engine to a file
    with open(output_path, 'wb') as f:
        f.write(engine.serialize())

    print("Engine built successfully")
    print(f"Converted TensorRT engine saved at {output_path}")    
    return engine, TRT_LOGGER


if __name__ == "__main__":
    print("Usage: python3 Onnx_TensorRT.py <model_path> <output_path> FP16_mode batch_size input_shape")
    print("Example: python3 Onnx_TensorRT.py ./model.onnx ./model_trt.trt True 1 (1, 3, 224, 224)")
    
    if len(sys.argv) < 2:
        convert_onnx_to_trt()
    else:
        for i in range(len(sys.argv), 6):
            sys.argv.append(None)
            convert_onnx_to_trt(*sys.argv[1:6])

# improvements
# Use Dynamic Shapes: If your model supports it, using dynamic input shapes can improve the inference speed. This allows TensorRT to optimize the execution plan based on the actual input shapes at runtime.
# Use Streams for Concurrent Execution: If you are running multiple inferences concurrently, you can use CUDA streams to overlap the computation and data transfer of different inferences.
# Use TensorRT's Optimized Layers: Whenever possible, use TensorRT's optimized layers instead of custom layers. TensorRT has highly optimized implementations for many common layers.
# Enable Layer Fusion: Layer fusion combines multiple layers into a single operation, which can reduce memory access and improve speed. This is automatically done by TensorRT during the optimization process.
# Enable Kernel Auto-Tuning: TensorRT automatically selects the best CUDA kernels for the layers in your model. This process can take some time during the first run, but the results are cached and used for subsequent runs.
# Free GPU Memory After Use: After you are done with a TensorRT engine, you should free its memory to make it available for other uses. In Python, you can do this by deleting the engine and calling gc.collect().
# Use Streams for Concurrent Execution: If you are running multiple inferences concurrently, you can use CUDA streams to overlap the computation and data transfer of different inferences. This can reduce the peak memory usage.