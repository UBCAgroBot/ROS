import cv2
from ultralytics import YOLO
import time

model = YOLO("/home/user/ROS/models/maize/Maize.engine")
image = cv2.imread("/home/user/ROS/assets/maize/IMG_1822_14.JPG")

# stream = True?
for _ in range(100):
    tic = time.perf_counter_ns()
    result = model.predict(
        image,  # batch=8 of the same image
        verbose=False,
        device="cuda",
    )
    print(f"Elapsed time: {(time.perf_counter_ns() - tic) / 1e6:.2f} ms")
    annotated_frame = result[0].plot()
    cv2.imshow("YOLOv8 Inference", annotated_frame)
    if cv2.waitKey(1) & 0xFF == ord("q"):
            break

cv2.destroyAllWindows()