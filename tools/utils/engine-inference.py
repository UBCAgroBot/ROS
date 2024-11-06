import tensorrt as trt
import pycuda.driver as cuda
import pycuda.autoinit  # Automatically initializes CUDA driver
import numpy as np
import cupy as cp
import time
import torch

# should add quantized and fp16 

# Load TensorRT model
TRT_LOGGER = trt.Logger(trt.Logger.WARNING)

class TRTEngine:
    def __init__(self, engine_path):
        self.engine = self.load_engine(engine_path)
        self.context = self.engine.create_execution_context()
        # should also allocate buffers in this stage

    def load_engine(self, engine_path):
        with open(engine_path, 'rb') as f, trt.Runtime(TRT_LOGGER) as runtime:
            return runtime.deserialize_cuda_engine(f.read())

    def allocate_buffers(self):
        inputs = []
        outputs = []
        bindings = []
        stream = cuda.Stream()

        for binding in self.engine:
            size = trt.volume(self.engine.get_binding_shape(binding)) * self.engine.max_batch_size
            dtype = trt.nptype(self.engine.get_binding_dtype(binding))
            # Allocate host and device buffers?
            host_mem = cuda.pagelocked_empty(size, dtype)
            device_mem = cuda.mem_alloc(host_mem.nbytes)
            
            # host_mem = torch.empty(size, dtype=torch.float32).cuda()
            # inputs.append(host_mem)
            # bindings.append(int(host_mem.data_ptr()))
            
            # Append the device buffer to device binding
            bindings.append(int(device_mem))

            # Append to the appropriate list
            if self.engine.binding_is_input(binding):
                inputs.append({'host': host_mem, 'device': device_mem})
            else:
                outputs.append({'host': host_mem, 'device': device_mem})

        return inputs, outputs, bindings, stream

    # performs inference on the input data using the TensorRT engine
    def infer(engine, inputs, outputs, bindings, stream, input_data):
    # should assign buffers in self initialization earlier and reference just self
    
        # Transfer input data to the device
        np.copyto(inputs[0][0], input_data.ravel())
        cuda.memcpy_htod_async(inputs[0][1], inputs[0][0], stream)

        # Execute the model
        context = engine.create_execution_context()
        start_time = time.perf_counter_ns()
        context.execute_async_v2(bindings=bindings, stream_handle=stream.handle)
        end_time = time.perf_counter_ns()

        # Transfer predictions back from the GPU
        cuda.memcpy_dtoh_async(outputs[0][0], outputs[0][1], stream)

        # Wait for the stream to complete the operation
        stream.synchronize()

        return outputs[0][0], (end_time - start_time)/1e6