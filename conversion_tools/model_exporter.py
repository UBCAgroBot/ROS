# c1, c2 = (int(x[0]), int(x[1])), (int(x[2]), int(x[3]))
# cv2.rectangle(img, c1, c2, color, thickness=2, lineType=cv2.LINE_AA)
# [batch size, channel number, height, width]
# cv2.dnn.NMSBoxes
# input shape (1,3,640,640) BCHW
# output shape (1,5,8400)

from ultralytics import YOLO

# Load a model
model = YOLO("yolov8n.pt")  # load an official model
model = YOLO("path/to/best.pt")  # load a custom trained model

# Export the model
model.export(format="onnx")

from ultralytics import YOLO

# Load the YOLOv8 model
model = YOLO("yolov8n.pt")

# Export the model to TensorRT format
model.export(format="engine")  # creates 'yolov8n.engine'

# Load the exported TensorRT model
tensorrt_model = YOLO("yolov8n.engine")

# Run inference
results = tensorrt_model("https://ultralytics.com/images/bus.jpg")

# logging style:
# PyTorch: starting from '/home/user/ROS/models/maize/Maize.pt' with input shape (1, 3, 640, 640) BCHW and output shape(s) (1, 5, 8400) (5.9 MB)
# ONNX: starting export with onnx 1.16.2 opset 12...
# ONNX: export success ✅ 1.7s, saved as '/home/user/ROS/models/maize/Maize.onnx' (11.7 MB)
# Export complete (3.8s)
# Results saved to /home/user/ROS/models/maize
# Predict:         yolo predict task=detect model=/home/user/ROS/models/maize/Maize.onnx imgsz=640  
# Validate:        yolo val task=detect model=/home/user/ROS/models/maize/Maize.onnx imgsz=640 data=config.yaml  

