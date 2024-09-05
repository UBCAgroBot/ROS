import argparse
import tensorrt as trt
import pycuda.driver as cuda
import pycuda.autoinit  # Automatically initializes CUDA driver
# import numpy as np
import time
import torch

# allocates input/ouput buffers for the TensorRT engine inference
def allocate_buffers(engine):
    inputs = []
    outputs = []
    bindings = []
    stream = cuda.Stream()

    for binding in engine:
        size = trt.volume(engine.get_binding_shape(binding)) * engine.max_batch_size
        dtype = trt.nptype(engine.get_binding_dtype(binding))

        # Allocate host and device buffers
        host_mem = cuda.pagelocked_empty(size, dtype)
        device_mem = cuda.mem_alloc(host_mem.nbytes)

        # Append the device buffer to device bindings
        bindings.append(int(device_mem))

        # Append to the appropriate list
        if engine.binding_is_input(binding):
            inputs.append((host_mem, device_mem))
        else:
            outputs.append((host_mem, device_mem))

    return inputs, outputs, bindings, stream

# performs inference on the input data using the TensorRT engine
def infer(engine, inputs, outputs, bindings, stream, input_data):
    # Transfer input data to the device
    np.copyto(inputs[0][0], input_data.ravel())
    cuda.memcpy_htod_async(inputs[0][1], inputs[0][0], stream)

    # Execute the model
    context = engine.create_execution_context()
    context.execute_async_v2(bindings=bindings, stream_handle=stream.handle)

    # Transfer predictions back from the GPU
    cuda.memcpy_dtoh_async(outputs[0][0], outputs[0][1], stream)

    # Wait for the stream to complete the operation
    stream.synchronize()

    return outputs[0][0]

# tests a TensorRT engine file by performing inference and checking outputs
def test_trt_engine(trt_engine_path='model_trt.trt', input_shape=(1,3,224,224), input_data=None, expected_output=None):
    # Load the TensorRT engine from the file
    with open(trt_engine_path, "rb") as f:
        runtime = trt.Runtime(trt.Logger(trt.Logger.WARNING))
        engine = runtime.deserialize_cuda_engine(f.read())

    # Allocate buffers for inference
    inputs, outputs, bindings, stream = allocate_buffers(engine)

    # Generate random input data if not provided
    if input_data is None:
        input_data = np.random.rand(*input_shape).astype(np.float32)

    # Perform inference using the TensorRT engine
    output = infer(engine, inputs, outputs, bindings, stream, input_data)

    # Print the inference result
    print("Inference output:", output)

    # Compare with expected output if provided
    if expected_output is not None:
        if np.allclose(output, expected_output, rtol=1e-3, atol=1e-3):
            print("The inference result matches the expected output.")
            return True
        else:
            print("The inference result does not match the expected output.")
            return False
    else:
        print("No expected output provided. Unable to verify accuracy.")
        return True  # Pass as long as inference ran without errors

if __name__ == "__main__":
    print("Usage: python3 TensorRT_test.py <trt_engine_path> <input_shape> <input_data> <expected_output>")
    print("Example: python3 TensorRT_test.py model.trt (1, 3, 224, 224) None None")
    
    if len(sys.argv) < 2:
        test_trt_engine()
    else:
        for i in range(len(sys.argv), 5):
            sys.argv.append(None)
            test_trt_engine(*sys.argv[1:5])


def benchmark_trt_model(trt_engine_path):
    TRT_LOGGER = trt.Logger(trt.Logger.WARNING)
    with open(trt_engine_path, 'rb') as f, trt.Runtime(TRT_LOGGER) as runtime:
        engine = runtime.deserialize_cuda_engine(f.read())
    
    context = engine.create_execution_context()
    
    # Example input
    input_shape = (1, 3, 224, 224)
    input_data = torch.randn(input_shape).cuda()

    # Allocate buffers
    inputs, outputs, bindings = [], [], []
    for binding in engine:
        size = trt.volume(engine.get_binding_shape(binding)) * engine.max_batch_size
        dtype = trt.nptype(engine.get_binding_dtype(binding))
        host_mem = torch.empty(size, dtype=torch.float32).cuda()
        inputs.append(host_mem)
        bindings.append(int(host_mem.data_ptr()))

    # Execute inference
    start_time = time.time()
    context.execute_v2(bindings=bindings)
    end_time = time.time()

    # Report time
    latency = end_time - start_time
    print(f"Model Inference Time: {latency * 1000:.2f} ms")

benchmark_trt_model(args.model)