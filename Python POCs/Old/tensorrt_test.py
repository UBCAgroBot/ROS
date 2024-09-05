from ultralytics import YOLO
import cv2
import os

# yolov8x_onnx.engine does not work -> UTF encoding error, retry 

os.chdir('/home/user/AppliedAI/23-I-12_SysArch/Experiments/ishaan_workspace/src/pipeline/node_test/node_test')
tensorrt_model = YOLO('yolov8x_pt.engine')
success = 1
vidObj = cv2.VideoCapture('City.mp4') 

while vidObj.isOpened():
    # Read a frame from the video
    success, frame = vidObj.read()
    
    if success:
        # Run YOLOv8 inference on the frame
        results = tensorrt_model(frame, stream=False) # set stream to false
        # for result in results:
            # annotated_frame = result[0].plot()
            # print(result[0].speed)
            # print(repr(result[1]))
            # cv2.imshow("YOLOv8 Inference", annotated_frame)
            # Break the loop if 'q' is pressed
            # if cv2.waitKey(1) & 0xFF == ord("q"):
            #     break
            # pass

        # Visualize the results on the frame
        # annotated_frame = results[0].plot()
        # print(results[0].speed)
        # print(repr(result[0]))
        # print(repr(result[1]))
        # Display the annotated frame
        # cv2.imshow("YOLOv8 Inference", annotated_frame)
        # # Break the loop if 'q' is pressed
        # if cv2.waitKey(1) & 0xFF == ord("q"):
        #     break
    else:
        # Break the loop if the end of the video is reached
        break

# Release the video capture object and close the display window
vidObj.release()
cv2.destroyAllWindows()

def load(self, file):
    f = open(file, "rb")
    runtime = trt.Runtime(trt.Logger(trt.Logger.WARNING)) 

    engine = runtime.deserialize_cuda_engine(f.read())
    self.context = engine.create_execution_context()
    
def allocate_memory(self, batch):
    self.output = np.empty(self.num_classes, dtype = self.target_dtype) # Need to set both input and output precisions to FP16 to fully enable FP16

    # Allocate device memory
    self.d_input = cuda.mem_alloc(1 * batch.nbytes)
    self.d_output = cuda.mem_alloc(1 * self.output.nbytes)

    self.bindings = [int(self.d_input), int(self.d_output)]

    self.stream = cuda.Stream()
    
def predict(self, batch): # result gets copied into output
    if self.stream is None:
        self.allocate_memory(batch)
        
    # Transfer input data to device
    cuda.memcpy_htod_async(self.d_input, batch, self.stream)
    # Execute model
    self.context.execute_async_v2(self.bindings, self.stream.handle, None)
    # Transfer predictions back
    cuda.memcpy_dtoh_async(self.output, self.d_output, self.stream)
    # Syncronize threads
    self.stream.synchronize()
    
    return self.output