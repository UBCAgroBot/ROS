import time
import onnxruntime as ort
import numpy as np
import cv2
import tqdm
from load import load_model, load_images

# to do: add int8 support
# base path
def get_images(images=None, image_path="C:/", imgsz=640, gpu=True, precision="fp32"):
    images = load_images(image_path) if images is None else images
    
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
        image = cv2.resize(image, (imgsz, imgsz))
        if gpu:
            image = cp.asarray(image).astype(img_type)
            image = cp.expand_dims(image, axis=0)
        else:
            image = image.astype(img_type)
            image = np.expand_dims(image, axis=0)
            image = np.transpose(image, (0, 3, 1, 2))
        loaded_images.append(image) 
    
    return loaded_images    

def predict(model, model_path, images, image_path, imgsz, confidence, gpu):
    model = load_model(model_path) if model is None else model
    images = get_images(images, image_path, imgsz, gpu)
    
    if gpu:
        import cupy as cp
        import pycuda.driver as cuda
        cuda.init()
        device = cuda.Device(0)
        cuda_driver_context = device.make_context()
        onnx_session = ort.InferenceSession(model,providers=["CUDAExecutionProvider"])
    else:
        onnx_session = ort.InferenceSession(model,providers=["CPUExecutionProvider"])
    
    scale = 1.6
    
    results = []
    times = []
    for image in images:
        input_name = onnx_session.get_inputs()[0].name
        
        tic = time.perf_counter_ns()
        # results_ort = onnx_session.run([out.name for out in session.get_outputs()], {session.get_inputs()[0].name: model_test})
        onnx_output = onnx_session.run(None, {input_name: image})
        toc = time.perf_counter_ns()
        
        inference_time = (toc - tic) / 1e6
        times.append(inference_time)
        
        outputs = np.array(onnx_output[0])
        rows = outputs.shape[1]
        
    boxes = []
    scores = []
    class_ids = []

    # Iterate through output to collect bounding boxes, confidence scores, and class IDs
    for i in range(rows):
        classes_scores = outputs[4:, i]
        if classes_scores.size == 0:
            continue
        maxClassIndex = np.argmax(classes_scores)
        maxScore = classes_scores[maxClassIndex]
        if maxScore >= confidence:
            x_center = outputs[0, i]
            y_center = outputs[1, i]
            width = outputs[2, i]
            height = outputs[3, i]
            x1 = x_center - (0.5 * width)
            y1 = y_center - (0.5 * height)
            x2 = x_center + (0.5 * width)
            y2 = y_center + (0.5 * height)
            box = [x1, y1, x2, y2]
            boxes.append(box)
            scores.append(float(maxScore))
            class_ids.append(maxClassIndex)

    # Apply NMS (Non-maximum suppression)
    indices = cv2.dnn.NMSBoxes(boxes, scores, confidence, 0.45)
    
    detections = []
    
    print(detections)
    
    # Iterate through NMS results to draw bounding boxes and labels
    if len(indices) > 0:
        for i in indices.flatten():
            box = boxes[i]
            score = scores[i]
            class_id = class_ids[i]
            detections.append((box, score, class_id))
    
    # # Iterate through NMS results to draw bounding boxes and labels
    # for i in range(len(result_boxes)):
    #     index = result_boxes[i]
    #     box = boxes[index]
    #     detections.append([round(box[0] * scale),
    #         round(box[1] * scale),
    #         round((box[0] + box[2]) * scale),
    #         round((box[1] + box[3]) * scale),])

        # cast to results class
        # instantiate results object, append to results list
        # append
    
    if len(times) > 1 and len(images) > 1:
        average_time = sum(times[1:])/(len(images)-1)
    else:
        average_time = times[0]
    
    for box in detections:
        print(box)
    
    return detections

# are we automatically applying postprocessing/preprocessing?
predict(model=None, images=None, confidence=0.1, model_path='/home/user/workspace/models/maize/Maize.onnx', image_path='/home/user/workspace/assets/maize', imgsz=640, gpu=False)