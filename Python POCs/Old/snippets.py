# try:
#     self.model = torch.jit.load("trt_model.ts").cuda()
# except Exception as e:
#     try:
#         model = torch.load('yolov5s.pt', model_math='fp32').eval().to("cuda") # replace with ONNX
#         self.model = torch_tensorrt.compile(model, inputs=[torch_tensortt([1, 3, 1280, 1280])], enabled_precisions={'torch.half'}, debug=True)
#         self.save = False
#     except Exception as e:
#         self.get_logger().info(f"Error: {e}")
#         raise SystemExit
# finally:
#     self.get_logger().info("Model loaded successfully")
    
#             normalized_image = resized_image / 255.0  # Normalize pixel values to [0, 1]
#         input_image = torch.from_numpy(normalized_image).permute(2, 0, 1).float()  # Convert to torch tensor and rearrange dimensions
#                 output = input_image.unsqueeze(0)  # Add a batch dimension
                
                
#                         img = img.transpose((2, 0, 1)).astype(np.float32)
#         img = np.expand_dims(img, axis=0)

from ultralytics import YOLO
import cv2, time
import numpy as np
from models.pycuda_api import TRTEngine
from tensorrt_infer_det_without_torch import inference

tensorrt_mode = 1
mask_detect_mode = 1
webcam_mode = 1

if mask_detect_mode:
    model = YOLO("model/mask_detect/best.pt")
    engine = TRTEngine("model/mask_detect/best.engine")
    target = "inference/face-mask-video.mp4"
else:
    model = YOLO("model/yolov8l.pt")
    engine = TRTEngine("model/yolov8l.engine")
    #names=model.names
    target = "inference/City.mp4"
    #target = "https://trafficvideo2.tainan.gov.tw/82520774"

# Run batched infernece on a list of images
#results = model.predict(img, stream=True) # return a list of Results objects
if webcam_mode:
    cap = cv2.VideoCapture(0)
else:
    cap = cv2.VideoCapture(target)
#CountFrame = 0
#dt=0
while True:
    try:
        r, img = cap.read()
        st = time.time()
        #img =cv2.resize(img, (800, 600))
        if not tensorrt_mode:
            results = model(source=img)
            img = results[0].plot() # annotated_frame
        else:
            img = inference(engine, img, mask_detect_mode)

        et = time.time()

        FPS = round(1/(et-st))
        cv2.putText(img, 'FPS=' + str(FPS), (20, 150),
                    cv2.FONT_HERSHEY_PLAIN, 5, (0, 0, 255), 2, cv2.LINE_AA)
        cv2.imshow("YOLOv8", img)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    except Exception as e:
        print(e)

cap.release()
cv2.destroyAllWindows()



def main():                                                                     
    args = parse_args()                                                         
    cam = Camera(args)                                                          
    cam.open()                                                                  
    if not cam.is_opened:                                                       
        sys.exit('Failed to open camera!')                                      
                                                                                
    import tensorflow as tf                                                     
    a = tf.constant([1.0, 2.0], name="a")                                       
    b = tf.constant([3.0, 4.0], name="b")                                       
    result = a + b                                                              
    with tf.Session() as sess:                                                  
        print(sess.run(result))                                                 
                                                                                
    import pycuda.driver as cuda                                                
    cuda_ctx = cuda.Device(0).make_context()  # GPU 0                           
                                                                                
    cls_dict = get_cls_dict('coco')                                             
    yolo_dim = int(args.model.split('-')[-1])  # 416 or 608                     
    trt_yolov3 = TrtYOLOv3(args.model, (yolo_dim, yolo_dim))                    
                                                                                
    cam.start()                                                                 
    open_window(WINDOW_NAME, args.image_width, args.image_height,               
                'Camera TensorRT YOLOv3 Demo')                                  
    vis = BBoxVisualization(cls_dict)                                           
    loop_and_detect(cam, trt_yolov3, conf_th=0.3, vis=vis)                      
                                                                                
    cuda_ctx.pop()                                                              
                                                                                
    cam.stop()                                                                  
    cam.release()                                                               
    cv2.destroyAllWindows()

"""trt_yolov3.py

This script demonstrates how to do real-time object detection with
TensorRT optimized YOLOv3 engine.
"""


import sys
import time
import argparse

import cv2
import tensorflow as tf
import pycuda.driver as cuda
import pycuda.autoinit  # This is needed for initializing CUDA driver

from utils.yolov3_classes import get_cls_dict
from utils.yolov3 import TrtYOLOv3
from utils.camera import add_camera_args, Camera
from utils.display import open_window, set_display, show_fps
from utils.visualization import BBoxVisualization


WINDOW_NAME = 'TrtYOLOv3Demo'


def parse_args():
    """Parse input arguments."""
    desc = ('Capture and display live camera video, while doing '
            'real-time object detection with TensorRT optimized '
            'YOLOv3 model on Jetson Nano')
    parser = argparse.ArgumentParser(description=desc)
    parser = add_camera_args(parser)
    parser.add_argument('--model', type=str, default='yolov3-416',
                        choices=['yolov3-288', 'yolov3-416', 'yolov3-608',
                                 'yolov3-tiny-288', 'yolov3-tiny-416'])
    args = parser.parse_args()
    return args


def loop_and_detect(cam, trt_yolov3, cuda_ctx, conf_th, vis):
    """Continuously capture images from camera and do object detection.

    # Arguments
      cam: the camera instance (video source).
      trt_yolov3: the TRT YOLOv3 object detector instance.
      cuda_ctx: cuda context for trt_yolov3
      conf_th: confidence/score threshold for object detection.
      vis: for visualization.
    """

    counter = tf.Variable(initial_value=0, dtype=tf.int32)
    step = tf.assign_add(counter, 1)
    config = tf.ConfigProto(gpu_options=tf.GPUOptions(allow_growth=True))
    sess = tf.Session(config=config)
    sess.run(tf.global_variables_initializer())

    full_scrn = False
    fps = 0.0
    tic = time.time()
    while True:
        if cv2.getWindowProperty(WINDOW_NAME, 0) < 0:
            break
        img = cam.read()
        sess.run(step)
        print(sess.run(counter))
        if img is not None:
            cuda_ctx.push()  # set this as the active context
            boxes, confs, clss = trt_yolov3.detect(img, conf_th)
            img = vis.draw_bboxes(img, boxes, confs, clss)
            img = show_fps(img, fps)
            cv2.imshow(WINDOW_NAME, img)
            toc = time.time()
            curr_fps = 1.0 / (toc - tic)
            # calculate an exponentially decaying average of fps number
            fps = curr_fps if fps == 0.0 else (fps*0.95 + curr_fps*0.05)
            tic = toc
        key = cv2.waitKey(1)
        if key == 27:  # ESC key: quit program
            break
        elif key == ord('F') or key == ord('f'):  # Toggle fullscreen
            full_scrn = not full_scrn
            set_display(WINDOW_NAME, full_scrn)

    cuda_ctx.detach()
    sess.close()


def main():
    args = parse_args()
    cam = Camera(args)
    cam.open()
    if not cam.is_opened:
        sys.exit('Failed to open camera!')

    cuda.init()
    cuda_ctx = cuda.Device(0).make_context()  # GPU 0

    cls_dict = get_cls_dict('coco')
    yolo_dim = int(args.model.split('-')[-1])  # 416 or 608
    trt_yolov3 = TrtYOLOv3(args.model, (yolo_dim, yolo_dim))

    cam.start()
    open_window(WINDOW_NAME, args.image_width, args.image_height,
                'Camera TensorRT YOLOv3 Demo')
    vis = BBoxVisualization(cls_dict)
    loop_and_detect(cam, trt_yolov3, cuda_ctx, conf_th=0.3, vis=vis)

    cam.stop()
    cam.release()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    main()
    
import argparse
    parser = argparse.ArgumentParser(
    description="Traffic Flow Analysis with YOLO and ByteTrack"
)

parser.add_argument(
    "--source_weights_path",
    required=True,
    help="Path to the source weights file",
    type=str,
)

parser.add_argument(
    "--source_video_path",
    required=True,
    help="Path to the source video file",
    type=str,
)

parser.add_argument(
    "--target_video_path",
    default=None,
    help="Path to the target video file (output)",
    type=str,
)

parser.add_argument(
    "--confidence_threshold",
    default=0.3,
    help="Confidence threshold for the model",
    type=float,
)

parser.add_argument(
    "--iou_threshold", default=0.7, help="IOU threshold for the model", type=float
)

args = parser.parse_args()

processor = VideoProcessor(
    source_weights_path=args.source_weights_path,
    source_video_path=args.source_video_path,
    target_video_path=args.target_video_path,
    confidence_threshold=args.confidence_threshold,
    iou_threshold=args.iou_threshold,
)

processor.process_video()

def draw_detections(image, detections):
    for *xyxy, conf, cls in detections:
        # Convert bounding box format from xyxy to xywh
        x, y, w, h = xyxy[0], xyxy[1], xyxy[2] - xyxy[0], xyxy[3] - xyxy[1]

        # Convert coordinates and dimensions from normalized to absolute
        x, y, w, h = int(x * image.shape[1]), int(y * image.shape[0]), int(w * image.shape[1]), int(h * image.shape[0])

        # Draw the bounding box
        cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)

        # Draw the class label
        label = f"{class_labels[int(cls)]} {conf:.2f}"
        cv2.putText(image, label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    return image

image = cv2.imread('image.jpg')
detections = model(image)  # Replace this with your YOLOv5 model inference code
image = draw_detections(image, detections)
cv2.imshow('image', image)
cv2.waitKey(0)
cv2.destroyAllWindows()

self.bbox_subscription = self.create_subscription(
    BoundingBox, 'bbox_data', self.bbox_callback, 10)
self.bbox_subscription

def bbox_callback(self, data):
    # Convert the ROS Image message to a CV image
    try:
        cv_image = self.bridge.imgmsg_to_cv2(data.image, "bgr8")
    except CvBridgeError as e:
        print(e)

    # Draw the bounding box and class label on the image
    for box in data.boxes:
        x, y, w, h = box.x, box.y, box.w, box.h
        cv2.rectangle(cv_image, (x, y), (x + w, y + h), (0, 255, 0), 2)
        label = class_labels[box.class_id]
        cv2.putText(cv_image, label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    # Display the image
    cv2.imshow('image', cv_image)
    cv2.waitKey(1)