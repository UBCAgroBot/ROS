import time
import onnxruntime as ort
import numpy as np
import cv2
import tqdm
from load import load_model, load_images

# to do: add int8 support
# base path
def get_images(images=None, image_path="C:/", imgsz=640, gpu=True, precision="fp16"):
    images = load_images(image_path) if images == None else images = images
    
    if gpu:
        import cupy as cp
        if precision == "fp16":
            img_type = cp.float16
        elif precision == "fp32":
            img_type = cp.float32
        else:
            raise ValueError(f"{precision} is not supported, please pick either fp16 or fp32")
    else:
        if precision == "fp16":
            img_type = np.float16
        elif precision == "fp32":
            img_type = np.float32
        else:   
            raise ValueError(f"{precision} is not supported, please pick either fp16 or fp32")
    
    loaded_images = []
    for image_src in images:
        # should call preprocess utility...
        image = cv2.imread(image_src)
        image = cv2.resize(image, imgsz)
        loaded_images.append(image.astype(img_type)) 
    
    return loaded_images    

def predict(model, model_path, images, image_path, imgsz, confidence, gpu):
    model = load_model(model_path) if model == None else model = model
    images = get_images(images, image_path, imgsz, gpu)
    
    if gpu:
        import cupy as cp
        import pycuda.driver as cuda
        cuda.init()
        device = cuda.Device(0)
        cuda_driver_context = device.make_context()
        onnx_session = ort.InferenceSession(model,providers=["CUDAExecutionProvider"])
    else:
        onnx_session = ort.InferenceSessoin(model,providers=["CPUExecutionProvider"])
    
    results = []
    times = []
    for image in images:
        input_name = onnx_session.get_inputs()[0].name
        tic = time.perf_counter_ns()
        # results_ort = session.run([out.name for out in session.get_outputs()], {session.get_inputs()[0].name: x_test})
        # results_ort = onnx_session.run([out.name for out in session.get_outputs()], {session.get_inputs()[0].name: model_test})
        onnx_output = onnx_session.run(None, {input_name: image})
        toc = time.perf_counter_ns()
        onnx_output = onnx_output[0]
        inference_time = (toc - tic) / 1e6
        times.append(inference_time)
        # onnx_output= np.array(onnx_output)
        # cast to results class
        # instantiate results object, append to results list
    
    average_time = sum(times[1:])/(len(images)-1)
    return results

# are we automatically applying postprocessing/preprocessing?