# stream off:
from ultralytics import YOLO

# Load a model
model = YOLO("yolov8n.pt")  # pretrained YOLOv8n model

# Run batched inference on a list of images
results = model(["image1.jpg", "image2.jpg"])  # return a list of Results objects

# Process results list
for result in results:
    boxes = result.boxes  # Boxes object for bounding box outputs
    masks = result.masks  # Masks object for segmentation masks outputs
    keypoints = result.keypoints  # Keypoints object for pose outputs
    probs = result.probs  # Probs object for classification outputs
    obb = result.obb  # Oriented boxes object for OBB outputs
    result.show()  # display to screen
    result.save(filename="result.jpg")  # save to disk

# stream on:
from ultralytics import YOLO

# Load a model
model = YOLO("yolov8n.pt")  # pretrained YOLOv8n model

# Run batched inference on a list of images
results = model(["image1.jpg", "image2.jpg"], stream=True)  # return a generator of Results objects

# Process results generator
for result in results:
    boxes = result.boxes  # Boxes object for bounding box outputs
    masks = result.masks  # Masks object for segmentation masks outputs
    keypoints = result.keypoints  # Keypoints object for pose outputs
    probs = result.probs  # Probs object for classification outputs
    obb = result.obb  # Oriented boxes object for OBB outputs
    result.show()  # display to screen
    result.save(filename="result.jpg")  # save to disk


from ultralytics import YOLO

# Load a pretrained YOLOv8n model
model = YOLO("yolov8n.pt")

# Run inference on 'bus.jpg' with arguments
model.predict("bus.jpg", save=True, imgsz=320, conf=0.5)

from ultralytics import YOLO

# Load a pretrained YOLOv8n model
model = YOLO("yolov8n.pt")

# Run inference on an image
results = model("bus.jpg")  # results list

# View results
for r in results:
    print(r.boxes)  # print the Boxes object containing the detection bounding boxes

from PIL import Image

from ultralytics import YOLO

# Load a pretrained YOLOv8n model
model = YOLO("yolov8n.pt")

# Run inference on 'bus.jpg'
results = model(["bus.jpg", "zidane.jpg"])  # results list

# Visualize the results
for i, r in enumerate(results):
    # Plot results image
    im_bgr = r.plot()  # BGR-order numpy array
    im_rgb = Image.fromarray(im_bgr[..., ::-1])  # RGB-order PIL image

    # Show results to screen (in supported environments)
    r.show()

    # Save results to disk
    r.save(filename=f"results{i}.jpg")

import cv2
from ultralytics import YOLO
import time

model = YOLO("/home/user/ROS/models/maize/Maize.engine")
image = cv2.imread("/home/user/ROS/assets/maize/IMG_1822_14.JPG")

sum = 0
# stream = True?
for _ in range(100):
    tic = time.perf_counter_ns()
    result = model.predict(
        image,  # batch=8 of the same image
        verbose=False,
        device="cuda",
    )
    elapsed_time = (time.perf_counter_ns() - tic) / 1e6
    print(f"Elapsed time: {(elapsed_time):.2f} ms")
    sum += elapsed_time
    annotated_frame = result[0].plot()
    cv2.imshow("YOLOv8 Inference", annotated_frame)
    if cv2.waitKey(1) & 0xFF == ord("q"):
            break

avearage_time = (sum - 2660) / 100
print(f"Average time: {avearage_time:.2f} ms")
cv2.destroyAllWindows()

# source, conf, iou, imgz, half, visualize, agnostic_nms
# visualization arguments:

import time
import onnx
import onnxruntime as ort
import numpy as np

# any other chanes for fp_16 to work?
def predict_onnx(model_path, fp_16, input_shape):
    
    # random_input = np.random.randn(*input_shape).astype(np.float32)

    # # Run inference with ONNX
    # input_name = onnx_session.get_inputs()[0].name
    # onnx_output = onnx_session.run(None, {input_name: random_input})
    onnx_session = ort.InferenceSession(model_path,providers=["CUDAExecutionProvider"])
    
    if fp_16:
        random_input = np.random.randn(input_shape).astype(np.float16)
    else:
        random_input = np.random.randn(input_shape).astype(np.float32)
        
    input_name = onnx_session.get_inputs()[0].name
    tic = time.perf_counter_ns()
    # results_ort = session.run([out.name for out in session.get_outputs()], {session.get_inputs()[0].name: x_test})
    # results_ort = onnx_session.run([out.name for out in session.get_outputs()], {session.get_inputs()[0].name: model_test})
    onnx_output = onnx_session.run(None, {input_name: random_input})
    toc = time.perf_counter_ns()
    onnx_output = onnx_output[0]
    # onnx_output= np.array(onnx_output)
    return onnx_output, (toc - tic) / 1e6