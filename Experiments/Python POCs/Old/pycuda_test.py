import timeit, os
import tensorrt as trt
import pycuda.driver as cuda
import pycuda.autoinit
from decord import VideoReader, cpu, gpu
import numpy as np
import torch
from torchvision.transforms import Normalize

BATCH_SIZE = 8 # try 2, 4, 16
target_dtype = np.float16

os.chdir('/home/user/AppliedAI/23-I-12_SysArch/Experiments/ishaan_workspace/src/pipeline/node_test/node_test')
f = open("resnet_engine_pytorch.trt", "rb")
runtime = trt.Runtime(trt.Logger(trt.Logger.WARNING)) 
engine = runtime.deserialize_cuda_engine(f.read())
context = engine.create_execution_context()

global frames
# We can utilize the libavfilter library to do frame resize for us
vr = VideoReader('City.mp4', ctx=gpu(0), width=640, height=480) # cpu while on same process?
print('video frames:', len(vr))
for i in range(len(vr)/BATCH_SIZE):
    frames = vr.get_batch(range(i*BATCH_SIZE, (i+1)*BATCH_SIZE)).asnumpy()
    print(frames.shape)
    print(frames.dtype)
    print(frames.context)
    
    timeit.timeit(lambda: preprocess_image(), number=1, globals=globals())
    
    timeit.timeit(lambda: predict(), number=1, globals=globals())

# allocate device memory
d_input = cuda.mem_alloc(1 * frames.nbytes)
output = np.empty([BATCH_SIZE, 1000], dtype = target_dtype) 
d_output = cuda.mem_alloc(1 * output.nbytes)
bindings = [int(d_input), int(d_output)]
stream = cuda.Stream()

def preprocess_image():
    global frames
    norm = Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    frames = norm(torch.from_numpy(frames).transpose(0,2).transpose(1,2))
    frames.astype(np.float16, copy=False)
    # preprocessed_images = np.array([preprocess_image(image) for image in input_batch])

def predict(batch): # result gets copied into output
    # transfer input data to device
    cuda.memcpy_htod_async(d_input, batch, stream)
    # execute model
    context.execute_async_v2(bindings, stream.handle, None)
    # transfer predictions back
    cuda.memcpy_dtoh_async(output, d_output, stream)
    # syncronize threads
    stream.synchronize()
    return output

# pytorch native bridge!!! (if slow)