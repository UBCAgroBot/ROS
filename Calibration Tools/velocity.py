import cv2
import numpy as np
import os

os.chdir("23-I-12_SysArch/Experiments/Calibration Utilities")

class VelocityTracker:
    def __init__(self, initial_image, point_to_track=np.array([[300, 300]], dtype=np.float32)):
        self.point_to_track = point_to_track
        self.prev_image = cv2.cvtColor(initial_image, cv2.COLOR_BGR2GRAY)
        self.prev_timestamp = None

    def calculate_velocity(self, current_image, current_timestamp):
        current_image = cv2.cvtColor(current_image, cv2.COLOR_BGR2GRAY)
        new_points, status, error = cv2.calcOpticalFlowPyrLK(self.prev_image, current_image, self.point_to_track, None)
        distance = np.sqrt(np.sum((self.point_to_track - new_points) ** 2))

        if self.prev_timestamp is not None:
            time_diff = current_timestamp - self.prev_timestamp
            velocity = distance / time_diff if time_diff != 0 else 0
        else:
            velocity = 0

        cv2.circle(current_image, (new_points[0][0], new_points[0][1]), 5, (0, 255, 0), -1)
        cv2.putText(current_image, f'Velocity: {velocity}', (new_points[0][0], new_points[0][1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        cv2.imshow('Image', current_image)
        cv2.waitKey(0)

        # Update the previous image, point to track and timestamp
        self.prev_image = current_image.copy()
        self.point_to_track = new_points
        self.prev_timestamp = current_timestamp
        
        return velocity

    cv2.destroyAllWindows()

# from velocity import VelocityTracker

# initial_image = cv2.imread('initial_image.jpg')
# tracker = VelocityTracker(initial_image)

# # For each new image and timestamp
# new_image = cv2.imread('new_image.jpg')
# new_timestamp = 1234567890  # replace with your timestamp
# offset = tracker.calculate_velocity(new_image, new_timestamp)
offset = velocity * self.latency