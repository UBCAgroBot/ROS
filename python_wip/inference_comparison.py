import time
import tensorrt as trt
import pycuda.driver as cuda
import cupy as cp

# warmup
self.engine_path = self.get_parameter('engine_path').get_parameter_value().string_value
self.strip_weights = self.get_parameter('strip_weights').get_parameter_value().bool_value
self.precision = self.get_parameter('precision').get_parameter_value().string_value

self.engine, self.context = self.load_normal_engine()

# could replace trt.volume with cp.prod # d_input = cuda.mem_alloc(input_size * np.float32().nbytes)
self.stream = cuda.Stream()
self.input_shape = (self.engine).get_binding_shape(0)
self.output_shape = (self.engine).get_binding_shape(1)
self.d_input = cuda.mem_alloc(trt.volume(self.input_shape) * cp.dtype(cp.float32).itemsize) # change to fp16, etc.
self.d_output = cuda.mem_alloc(trt.volume(self.output_shape) * cp.dtype(cp.float32).itemsize) 

self.pointer_subscriber = self.create_publisher(String, 'preprocessing_done', self.pointer_callback, 10)
self.pointer_publisher = self.create_publisher(String, 'inference_done', 10)
self.arrival_time, self.type = 0, None, None
self.warmup()

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

def warmup(self):
    input_shape = self.input_shape

    for _ in range(20):
        random_input = np.random.randn(*input_shape).astype(np.float32)
        cuda.memcpy_htod(self.d_input, random_input)
        self.context.execute(bindings=[int(self.d_input), int(self.d_output)])

    self.get_logger().info(f"Engine warmed up with 20 inference passes.")

    # Load TensorRT engine and create execution context (example)
    TRT_LOGGER = trt.Logger(trt.Logger.WARNING)

def load_engine(engine_file_path):
    with open(engine_file_path, "rb") as f, trt.Runtime(TRT_LOGGER) as runtime:
        return runtime.deserialize_cuda_engine(f.read())

engine = load_engine("model.trt")
context = engine.create_execution_context()

if self.engine_path.endswith('.trt') or self.engine_path.endswith('.engine'):
    if self.strip_weights:  
        self.engine = self.load_stripped_engine_and_refit() 
        self.type = "trt_stripped" 
    else:  
        self.engine = self.load_normal_engine()
        self.type = "trt_normal"
elif self.engine_path.endswith('.pth'):
    from torch2trt import TRTModule
    import torch
    self.engine = TRTModule()
    (self.engine).load_state_dict(torch.load(self.engine_path))
    self.type = "torch_trt"
else:
    self.get_logger().error("Invalid engine file format. Please provide a .trt, .engine, or .pth file")
    return None

if self.type == "trt_stripped" or self.type == "trt_normal":
    self.allocate_buffers()
    self.exec_context = (self.engine).create_execution_context()
else:
    self.inference_type = "torch_trt"

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

# fixed allocation: does not account for multiple bindings/batch sizes (single input -> output tensor)
def allocate_buffers(self):
    engine = self.engine
    # Create a CUDA stream for async execution
    self.stream = cuda.Stream()

    self.input_shape = engine.get_binding_shape(0)
    self.output_shape = engine.get_binding_shape(1)

    # Allocate device memory for input/output
    self.d_input = cuda.mem_alloc(trt.volume(self.input_shape) * np.dtype(np.float32).itemsize)
    self.d_output = cuda.mem_alloc(trt.volume(self.output_shape) * np.dtype(np.float32).itemsize)

    # Allocate host pinned memory for input/output (pinned memory for input/output buffers)
    self.h_input = cuda.pagelocked_empty(trt.volume(self.input_shape), dtype=np.float32)
    self.h_output = cuda.pagelocked_empty(trt.volume(self.output_shape), dtype=np.float32)

    # Example image (allocate on GPU)
    self.cv_image = np.random.rand(480, 640, 3).astype(np.uint8)
    self.cv_cuda_image = cv2_cuda_GpuMat(self.cv_image.shape[0], self.cv_image.shape[1], cv2.CV_8UC3)

    # Upload image to GPU (device memory)
    self.cv_cuda_image.upload(self.cv_image)

import time
import tensorrt as trt
import pycuda.driver as cuda
import pycuda.autoinit
import torch
import numpy as np
from torch2trt import TRTModule

# Define paths and settings
engine_path = "model.trt"  # Replace with actual paths for your engines
strip_weights = False  # Set True if using stripped weights for TensorRT
precision = "fp32"  # Set precision (fp32, fp16, int8)

class InferenceComparison:

    def __init__(self, engine_path, strip_weights, precision):
        self.engine_path = engine_path
        self.strip_weights = strip_weights
        self.precision = precision
        self.stream = cuda.Stream()
        self.engine = None
        self.context = None
        self.h_input = None
        self.h_output = None
        self.d_input = None
        self.d_output = None

    def allocate_buffers(self, engine):
        self.input_shape = engine.get_binding_shape(0)
        self.output_shape = engine.get_binding_shape(1)

        # Allocate device memory for input/output
        self.d_input = cuda.mem_alloc(trt.volume(self.input_shape) * np.dtype(np.float32).itemsize)
        self.d_output = cuda.mem_alloc(trt.volume(self.output_shape) * np.dtype(np.float32).itemsize)

        # Allocate host pinned memory for input/output (for asynchronous execution)
        self.h_input = cuda.pagelocked_empty(trt.volume(self.input_shape), dtype=np.float32)
        self.h_output = cuda.pagelocked_empty(trt.volume(self.output_shape), dtype=np.float32)

    def load_normal_engine(self):
        TRT_LOGGER = trt.Logger(trt.Logger.WARNING)
        with open(self.engine_path, "rb") as f, trt.Runtime(TRT_LOGGER) as runtime:
            engine = runtime.deserialize_cuda_engine(f.read())
        context = engine.create_execution_context()
        return engine, context

    def load_stripped_engine_and_refit(self):
        TRT_LOGGER = trt.Logger(trt.Logger.WARNING)
        with open(self.engine_path, "rb") as f, trt.Runtime(TRT_LOGGER) as runtime:
            engine = runtime.deserialize_cuda_engine(f.read())
            refitter = trt.Refitter(engine, TRT_LOGGER)
            # Refit weights here if necessary
            assert refitter.refit_cuda_engine()
            return engine

    def load_torch_trt(self):
        model = TRTModule()
        model.load_state_dict(torch.load(self.engine_path))
        return model

    def warmup(self, engine_type):
        for _ in range(10):  # Warmup with 10 inference passes
            random_input = np.random.randn(*self.input_shape).astype(np.float32)
            cuda.memcpy_htod(self.d_input, random_input)
            if engine_type == "torch_trt":
                self.engine(random_input)
            else:
                self.context.execute(bindings=[int(self.d_input), int(self.d_output)])

    def infer_with_trt_normal(self):
        self.engine, self.context = self.load_normal_engine()
        self.allocate_buffers(self.engine)

        # Warmup phase
        self.warmup("trt_normal")

        # Run actual inference and measure time
        start_time = time.time()
        cuda.memcpy_htod(self.d_input, np.random.randn(*self.input_shape).astype(np.float32))
        self.context.execute(bindings=[int(self.d_input), int(self.d_output)])
        cuda.memcpy_dtoh(self.h_output, self.d_output)
        trt_normal_time = time.time() - start_time

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


# Instantiate the comparison class
comparison = InferenceComparison(engine_path, strip_weights, precision)

# Run inference for each engine and compare times
output_normal, normal_time = comparison.infer_with_trt_normal()
print(f"Normal TensorRT Inference Time: {normal_time:.6f} seconds")

output_stripped, stripped_time = comparison.infer_with_trt_stripped()
print(f"Stripped TensorRT Inference Time: {stripped_time:.6f} seconds")

output_torch_trt, torch_trt_time = comparison.infer_with_torch_trt()
print(f"Torch2TRT Inference Time: {torch_trt_time:.6f} seconds")

## compare bounding box output to the expected from the file...