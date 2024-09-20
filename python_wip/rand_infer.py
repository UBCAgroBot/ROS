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

    # # Create CUDA IPC handle for output tensor and image
    # self.output_ipc_handle = cuda.mem_get_ipc_handle(self.d_output)
    # self.image_ipc_handle = cuda.mem_get_ipc_handle(self.cv_cuda_image.cudaPtr())