import os
import onnx
import logging

logging.basicConfig(format='%(message)s', level=logging.INFO)

def load_model(model_path):
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found at {model_path}")
    if not model_path.endswith(".onnx"):
        raise ValueError("Model path should end with .onnx")
    else:
        logging.info(f"ONNX model loaded successfully")
        return onnx.load(model_path)

def load_engine(engine_path):
    if not os.path.exists(engine_path):
        raise FileNotFoundError(f"Engine file not found at {engine_path}")
    if not engine_path.endswith(".engine"):
        raise ValueError("Engine path should end with .engine")
    else:
        import tensorrt as trt
        TRT_LOGGER = trt.Logger(trt.Logger.WARNING)
        with open(engine_path, "rb") as f, trt.Runtime(TRT_LOGGER) as runtime:
            engine = runtime.deserialize_cuda_engine(f.read())
        logging.info(f"TensorRT engine loaded successfully")
        return engine

def load_images(image_path):
    if not os.path.exists(image_path):
            raise ValueError(f"Images folder not found at {image_path}")
        
    if len(os.listdir(image_path)) == 0:
        raise ValueError(f"Images folder is empty")
    
    files = []
    os.chdir(image_path)
    for filename in os.listdir(image_path):
        if filename.endswith(".jpg") or filename.endswith(".JPG") or filename.endswith(".png"):
            files.append(image_path + '/' + filename)
    
    if len(files) == 0:
        raise ValueError(f"No images files found in {image_path}")
    
    logging.info(f"{len(files)} from {image_path} loaded successfully")
    return files