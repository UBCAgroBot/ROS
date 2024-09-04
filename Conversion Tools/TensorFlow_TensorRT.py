import tensorflow as tf
from tensorflow.python.compiler.tensorrt import trt_convert as trt
import pycuda.driver as cuda

def convert_saved_model_to_engine(saved_model_dir, precision_mode='FP16', max_batch_size=1, max_workspace_size = 1 << 30):
    print("Converting TensorFlow SavedModel to TensorRT engine. This may take a few minutes.")

    # params = trt.DEFAULT_TRT_CONVERSION_PARAMS._replace(
    #     precision_mode=precision_mode,
    #     max_batch_size=max_batch_size
    # )

    params = trt.TrtConversionParams(
        precision_mode=precision_mode,
        max_batch_size=max_batch_size,
        max_workspace_size_bytes=max_workspace_size,
    )

    converter = trt.TrtGraphConverterV2(
        input_saved_model_dir=saved_model_dir,
        conversion_params=params
    )

    converter.convert()
    converter.summary()
    converter.save(saved_model_dir)
    print("Completed building Engine.")

saved_model_dir = input("Enter the path of the TensorFlow SavedModel directory: ")
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

convert_saved_model_to_engine(saved_model_dir, precision, batch_size, get_max_memory())