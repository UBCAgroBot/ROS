import argparse
import tensorflow as tf
from tf2trt import trt_convert as trt
import pycuda.driver as cuda
import pycuda.autoinit
import numpy as np

def get_max_memory():
    total, free = cuda.mem_get_info()
    max_mem = free * 0.95

    print(f"Total GPU memory: {total / (1024**2)} MB")
    print(f"Free GPU memory: {free / (1024**2)} MB")
    print(f"Max memory to use: {max_mem / (1024**2)} MB")
    return max_mem

def convert_tf_to_trt(model_path='./model.pb', output_path='./model_trt.trt', FP16_mode=True, batch_size=1, input_shape=(1, 3, 224, 224)):
    # params = trt.DEFAULT_TRT_CONVERSION_PARAMS._replace(max_workspace_size_bytes=(1<<30))
    params = trt.TrtConversionParams(
        precision_mode='FP16',
        max_batch_size=batch_size,
        max_workspace_size_bytes=get_max_memory(),
    )

    print("Loading the TensorFlow model")
    converter = trt.TrtGraphConverterV2(
        input_saved_model_dir=model_path,
        conversion_params=params
    )
    
    print("Building TensorRT engine. This may take a few minutes.")
    converter.convert()
    
    print("Engine built successfully")
    converter.summary()
    
    print(f"Converted TensorRT engine saved at {output_path}")    
    converter.save(output_path)

if __name__ == "__main__":
    print("Usage: python3 TensorFlow_TensorRT.py <model_path> <output_path> FP16_mode batch_size input_shape")
    print("Example: python3 TensorFlow_TensorRT.py ./model.pb ./model_trt.trt True 1 (1, 3, 224, 224)")
    
    if len(sys.argv) < 2:
        convert_tf_to_trt()
    else:
        for i in range(len(sys.argv), 6):
            sys.argv.append(None)
            convert_tf_to_trt(*sys.argv[1:6])