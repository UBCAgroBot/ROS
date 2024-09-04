import tensorrt as trt
import pycuda.driver as cuda

def convert_onnx_to_engine(onnx_filename, engine_filename = None, max_batch_size = 32, max_workspace_size = 1 << 30, fp16_mode = True):
    logger = trt.Logger(trt.Logger.WARNING)
    with trt.Builder(logger) as builder, builder.create_network() as network, trt.OnnxParser(network, logger) as parser:
        builder.max_workspace_size = max_workspace_size
        builder.fp16_mode = fp16_mode
        builder.max_batch_size = max_batch_size

        print("Parsing ONNX file.")
        with open(onnx_filename, 'rb') as model:
            if not parser.parse(model.read()):
                for error in range(parser.num_errors):
                    print(parser.get_error(error))

        print("Completed parsing of ONNX file.")

        print("Building TensorRT engine. This may take a few minutes.")
        engine = builder.build_cuda_engine(network)

        if engine_filename:
            with open(engine_filename, 'wb') as f:
                f.write(engine.serialize())

        print("Completed building Engine.")
        return engine, logger

model_name = input("Enter the name of the ONNX model file (without .onnx extension): ")
engine_name = input("Enter the name of the TensorRT engine file (without .engine extension): ")
precision = input("Enter the precision (FP32/FP16): ") # INT8 later
batch_size = int(input("Enter the maximum batch size: "))

def get_max_memory():
    cuda.init()
    total, free = cuda.mem_get_info()
    max_mem = free * 0.95

    print(f"Total GPU memory: {total / (1024**2)} MB")
    print(f"Free GPU memory: {free / (1024**2)} MB")
    print(f"Max memory to use: {max_mem / (1024**2)} MB")
    return max_mem

convert_onnx_to_engine(f"{model_name}.onnx", f"{engine_name}.engine", batch_size, get_max_memory(), precision == "FP16")

# improvements
# Use Dynamic Shapes: If your model supports it, using dynamic input shapes can improve the inference speed. This allows TensorRT to optimize the execution plan based on the actual input shapes at runtime.
# Use Streams for Concurrent Execution: If you are running multiple inferences concurrently, you can use CUDA streams to overlap the computation and data transfer of different inferences.
# Use TensorRT's Optimized Layers: Whenever possible, use TensorRT's optimized layers instead of custom layers. TensorRT has highly optimized implementations for many common layers.
# Enable Layer Fusion: Layer fusion combines multiple layers into a single operation, which can reduce memory access and improve speed. This is automatically done by TensorRT during the optimization process.
# Enable Kernel Auto-Tuning: TensorRT automatically selects the best CUDA kernels for the layers in your model. This process can take some time during the first run, but the results are cached and used for subsequent runs.
# Free GPU Memory After Use: After you are done with a TensorRT engine, you should free its memory to make it available for other uses. In Python, you can do this by deleting the engine and calling gc.collect().
# Use Streams for Concurrent Execution: If you are running multiple inferences concurrently, you can use CUDA streams to overlap the computation and data transfer of different inferences. This can reduce the peak memory usage.