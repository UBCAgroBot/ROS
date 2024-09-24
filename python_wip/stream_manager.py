import pycuda.driver as cuda

class CudaStreamManager:
    def __init__(self):
        # Create separate streams for different stages
        self.preprocess_stream = cuda.Stream()
        self.inference_stream = cuda.Stream()
        self.postprocess_stream = cuda.Stream()
        
        # Create CUDA events for synchronization
        self.preprocess_done = cuda.Event()
        self.inference_done = cuda.Event()
        
    def get_preprocess_stream(self):
        return self.preprocess_stream

    def get_inference_stream(self):
        return self.inference_stream

    def get_postprocess_stream(self):
        return self.postprocess_stream

    def get_preprocess_event(self):
        return self.preprocess_done

    def get_inference_event(self):
        return self.inference_done