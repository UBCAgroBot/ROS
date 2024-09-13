import time
import tensorrt as trt
import pycuda.driver as cuda
import pycuda.autoinit
import numpy as np

# Load TensorRT engine
def load_engine(engine_file_path):
    TRT_LOGGER = trt.Logger(trt.Logger.WARNING)
    with open(engine_file_path, "rb") as f, trt.Runtime(TRT_LOGGER) as runtime:
        return runtime.deserialize_cuda_engine(f.read())

# Set up CUDA context and allocate memory
def infer_with_tensorrt(engine, random_input):
    context = engine.create_execution_context()

    # Allocate memory for input and output
    input_shape = random_input.shape
    input_size = trt.volume(input_shape) * random_input.itemsize

    d_input = cuda.mem_alloc(input_size)
    h_output = cuda.pagelocked_empty(trt.volume(engine.get_binding_shape(1)), dtype=np.float32)
    d_output = cuda.mem_alloc(h_output.nbytes)

    # Transfer input to GPU
    cuda.memcpy_htod(d_input, random_input)

    # Run inference
    tic = time.perf_counter_ns()
    context.execute(bindings=[int(d_input), int(d_output)])
    toc = time.perf_counter_ns()

    # Transfer prediction back to host
    cuda.memcpy_dtoh(h_output, d_output)

    return h_output, (toc - tic) / 1e6

def verify_trt(model_path, output_path, fp_16, input_shape):
    print("Verifying the converted model")
    if fp_16:
        random_input = np.random.randn(*input_shape).astype(np.float16)
    else:
        random_input = np.random.randn(*input_shape).astype(np.float32)
    
    # Load the TensorRT engine
    engine = load_engine(output_path)

    # Run inference
    trt_output, trt_inference = infer_with_tensorrt(engine, random_input)
    print("TensorRT inference time:", trt_inference, "ms")
        
    # Get ONNX output
    from ONNX_Verify import predict_onnx
    onnx_output, onnx_inference = predict_onnx(model_path, fp_16, input_shape)
    print("ONNX inference time:", onnx_inference, "ms")

    # Calculate MSE (Mean Squared Error)
    mse = np.mean((onnx_output - trt_output) ** 2)
    print("MSE between ONNX and TensorRT outputs:", mse)

    # Calculate MAE (Mean Absolute Error)
    mae = np.mean(np.abs(onnx_output - trt_output))
    print("MAE between ONNX and TensorRT outputs:", mae)