import tensorrt as trt
import pycuda.driver as cuda
import numpy as np
import cupy as cp
import time
import torch

# should add quantized and fp16 

# Load TensorRT model
TRT_LOGGER = trt.Logger(trt.Logger.WARNING)

class TRTEngine:
    def __init__(self, engine_path): # strip_weights, precision
        self.engine = self.load_engine(engine_path)
        self.context = self.engine.create_execution_context()
        # self.precision = precision
        self.stream = cuda.Stream()
        self.engine = None
        self.context = None
        self.h_input = None
        self.h_output = None
        self.d_input = None
        self.d_output = None
        
        self.allocate_buffers()

    def load_engine(self, engine_path):
        with open(engine_path, 'rb') as f, trt.Runtime(TRT_LOGGER) as runtime:
            return runtime.deserialize_cuda_engine(f.read())

    def allocate_buffers(self):
        engine = self.engine
        self.stream = cuda.Stream()
        
        self.input_shape = engine.get_binding_shape(0)
        self.output_shape = engine.get_binding_shape(1)

        # self.d_output = cuda.mem_alloc(trt.volume(self.output_shape) * cp.dtype(cp.float32).itemsize) 

        # Allocate device memory for input/output
        self.d_input = cuda.mem_alloc(trt.volume(self.input_shape) * np.dtype(np.float32).itemsize)
        self.d_output = cuda.mem_alloc(trt.volume(self.output_shape) * np.dtype(np.float32).itemsize)

        # Allocate host pinned memory for input/output (pinned memory for input/output buffers)
        self.h_input = cuda.pagelocked_empty(trt.volume(self.input_shape), dtype=np.float32) # cp.float
        self.h_output = cuda.pagelocked_empty(trt.volume(self.output_shape), dtype=np.float32)

    def warmup(self, engine_type):
        for _ in range(10):  # Warmup with 10 inference passes
            random_input = np.random.randn(*self.input_shape).astype(np.float32)
            cuda.memcpy_htod(self.d_input, random_input)
            # if engine_type == "torch_trt":
            #     self.engine(random_input)
            # else:
            self.context.execute(bindings=[int(self.d_input), int(self.d_output)])

    # time
    # performs inference on the input data using the TensorRT engine
    def infer(self, engine, inputs, outputs, bindings, stream, input_data):
        # Transfer input data to the device
        np.copyto(inputs[0][0], input_data.ravel())
        cuda.memcpy_htod_async(inputs[0][1], inputs[0][0], stream)
        start_time = time.perf_counter_ns()
        context.execute_async_v2(bindings=bindings, stream_handle=stream.handle)
        end_time = time.perf_counter_ns()
        
        # inference
        start_time = time.time()
        cuda.memcpy_htod(self.d_input, np.random.randn(*self.input_shape).astype(np.float32))
        self.context.execute(bindings=[int(self.d_input), int(self.d_output)])
        cuda.memcpy_dtoh(self.h_output, self.d_output)
        trt_normal_time = time.time() - start_time
        
        # Transfer predictions back from the GPU
        cuda.memcpy_dtoh_async(outputs[0][0], outputs[0][1], stream)

        # Wait for the stream to complete the operation
        stream.synchronize()

        # return outputs[0][0], (end_time - start_time)/1e6
        
        return self.h_output, trt_normal_time

    def infer_with_trt_stripped(self):
        self.engine = self.load_stripped_engine_and_refit()
        self.context = self.engine.create_execution_context()
        self.allocate_buffers(self.engine)

        # Warmup phase
        self.warmup("trt_stripped")

        # Run actual inference and measure time
        start_time = time.time()
        cuda.memcpy_htod(self.d_input, np.random.randn(*self.input_shape).astype(np.float32))
        self.context.execute(bindings=[int(self.d_input), int(self.d_output)])
        cuda.memcpy_dtoh(self.h_output, self.d_output)
        trt_stripped_time = time.time() - start_time

        return self.h_output, trt_stripped_time

def torch_trt_inference_v2():
    import torch
    import torch_tensorrt as torch_trt

    # Sample PyTorch model (ResNet18 in this case)
    model = torch.hub.load('pytorch/vision:v0.10.0', 'resnet18', pretrained=True)
    model.eval().cuda()

    # Example input tensor
    input_data = torch.randn((1, 3, 224, 224)).cuda()

    # Convert the PyTorch model to a Torch-TensorRT optimized model
    trt_model = torch_trt.compile(model, 
                                inputs=[torch_trt.Input(input_data.shape)], 
                                enabled_precisions={torch.float, torch.half})  # Use FP32 and FP16

    # Run inference
    with torch.no_grad():
        output = trt_model(input_data)

    print(output)

def load_torch_trt(self):
    model = TRTModule()
    model.load_state_dict(torch.load(self.engine_path))
    return model

def infer_with_torch_trt(self):
    self.engine = self.load_torch_trt()

    # Create random input tensor
    input_tensor = torch.randn(1, 3, 224, 224).cuda()

    # Warmup phase
    for _ in range(10):
        _ = self.engine(input_tensor)

    # Run actual inference and measure time
    start_time = time.time()
    output = self.engine(input_tensor)
    torch_trt_time = time.time() - start_time

    return output, torch_trt_time

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

if self.engine_path.endswith('.pth'):
    from torch2trt import TRTModule
    import torch
    self.engine = TRTModule()
    (self.engine).load_state_dict(torch.load(self.engine_path))
    self.type = "torch_trt"
else:
    self.get_logger().error("Invalid engine file format. Please provide a .trt, .engine, or .pth file")
    return None

def load_stripped_engine_and_refit(self):
    if not os.path.exists(self.engine_path):
        self.get_logger().error(f"Engine file not found at {self.engine_path}")
        return None
    
    if not os.path.exists(self.model_path):
        self.get_logger().error(f"Model file not found at {self.model_path}")
        return None
    
    TRT_LOGGER = trt.Logger(trt.Logger.WARNING)
    with open(self.engine_path, "rb") as f, trt.Runtime(TRT_LOGGER) as runtime:
        engine = runtime.deserialize_cuda_engine(f.read())
        refitter = trt.Refitter(engine, TRT_LOGGER)
        parser_refitter = trt.OnnxParserRefitter(refitter, TRT_LOGGER)
        assert parser_refitter.refit_from_file(self.model_path)
        assert refitter.refit_cuda_engine()
        return engine

def load_stripped_engine_and_refit(self):
    TRT_LOGGER = trt.Logger(trt.Logger.WARNING)
    with open(self.engine_path, "rb") as f, trt.Runtime(TRT_LOGGER) as runtime:
        engine = runtime.deserialize_cuda_engine(f.read())
        refitter = trt.Refitter(engine, TRT_LOGGER)
        # Refit weights here if necessary
        assert refitter.refit_cuda_engine()
        return engine

    def load_normal_engine(self):
        if not os.path.exists(self.engine_path):
            self.get_logger().error(f"Engine file not found at {self.engine_path}")
            return None

        TRT_LOGGER = trt.Logger(trt.Logger.WARNING)
        with open(self.engine_path, "rb") as f, trt.Runtime(TRT_LOGGER) as runtime:
            engine = runtime.deserialize_cuda_engine(f.read())
        context = engine.create_execution_context()
        self.get_logger().info(f"Successfully loaded engine from {self.engine_path}")
        return engine, context

# proper inference:
def run_inference(self, d_input_ptr):
    tic = time.perf_counter_ns()
    # self.cuda_driver_context.push()
    
    cuda.memcpy_dtod_async(self.d_input, d_input_ptr, cp.prod(self.input_shape) * cp.dtype(cp.float32).itemsize, self.stream) # Copy input data to the allocated memory in TensorRT (from the IPC pointer)
    self.exec_context.execute_async_v2(bindings=[int(self.d_input), int(self.d_output)], stream_handle=self.stream.handle) # Execute inference asynchronously
    output = cp.empty(self.output_shape, dtype=cp.float32)
    cuda.memcpy_dtod_async(output.data, self.d_output, self.stream) # Copy output to variable
    self.stream.synchronize() 
    
    # self.cuda_driver_context.pop()
    toc = time.perf_counter_ns()
    
    self.get_logger().info(f"Inference: {(toc-tic)/1e6} ms")   
    print(f'Output: {output} \n Shape: {output.shape}')