from ultralytics import YOLO
from ultralytics.utils.benchmarks import benchmark

model = YOLO("/home/usr/Downloads/Maize.pt")

model.export(format='engine', imgsz=640, half=False, dynamic=False, simplify=False, workspace=8.0, int8=False, batch=1)

model.export(format='onnx', imgsz=640, half=False, dynamic=False, simplify=False, opset=12, batch=1) 

# onnxslim

# GPU benchmarking...
benchmark(model="yolov8n.pt", data="coco8.yaml", imgsz=640, half=False, device=0) # verbose, int8

## onnx: imgsz, half, dynamic, simplify, opset, batch
# tensorrt: imgsz, half, dynamic, simplify, workspace, int8, batch