# test_bbox_accuracy.py

import unittest
import numpy as np
import rosbag2_py
from some_custom_pkg.msg import BoundingBox  # Import your bounding box message type
from datetime import datetime

class TestBoundingBoxAccuracy(unittest.TestCase):
    def setUp(self):
        # Load the ground truth data from the .txt file
        self.ground_truth = self.load_ground_truth("ground_truth.txt")

        # Initialize rosbag reader
        storage_options = rosbag2_py.StorageOptions(uri="path/to/rosbag", storage_id="sqlite3")
        converter_options = rosbag2_py.ConverterOptions("", "")
        self.reader = rosbag2_py.SequentialReader()
        self.reader.open(storage_options, converter_options)

        # Set the topic to read bounding box messages
        self.reader.set_filter(rosbag2_py.TopicFilter(topic_name="/bbox_topic"))

    def load_ground_truth(self, filepath):
        """Load ground truth bounding boxes from a .txt file."""
        ground_truth_data = []
        with open(filepath, 'r') as file:
            for line in file:
                parts = line.strip().split()
                timestamp = float(parts[0])
                bbox = list(map(int, parts[1:5]))
                ground_truth_data.append((timestamp, bbox))
        return ground_truth_data

    def calculate_iou(self, bbox1, bbox2):
        """Calculate Intersection over Union (IoU) between two bounding boxes."""
        x_min1, y_min1, x_max1, y_max1 = bbox1
        x_min2, y_min2, x_max2, y_max2 = bbox2

        # Calculate intersection
        x_min_inter = max(x_min1, x_min2)
        y_min_inter = max(y_min1, y_min2)
        x_max_inter = min(x_max1, x_max2)
        y_max_inter = min(y_max1, y_max2)
        
        if x_min_inter >= x_max_inter or y_min_inter >= y_max_inter:
            return 0.0  # No overlap

        intersection_area = (x_max_inter - x_min_inter) * (y_max_inter - y_min_inter)

        # Calculate union
        area1 = (x_max1 - x_min1) * (y_max1 - y_min1)
        area2 = (x_max2 - x_min2) * (y_max2 - y_min2)
        union_area = area1 + area2 - intersection_area

        return intersection_area / union_area

    def test_bounding_box_accuracy(self):
        """Compare bounding box predictions in ROS bag with ground truth."""
        tolerance = 0.1  # IoU threshold for considering a match
        ground_truth_idx = 0  # Index for tracking ground truth entries

        while self.reader.has_next():
            (topic, msg, t) = self.reader.read_next()
            timestamp_ros = t / 1e9  # Convert nanoseconds to seconds
            bbox_ros = [msg.x_min, msg.y_min, msg.x_max, msg.y_max]

            # Find the closest ground truth bbox based on timestamp
            timestamp_gt, bbox_gt = self.ground_truth[ground_truth_idx]
            
            # If the ROS bag timestamp matches the ground truth timestamp, compare bboxes
            if abs(timestamp_ros - timestamp_gt) < 0.05:  # 50ms tolerance
                iou = self.calculate_iou(bbox_ros, bbox_gt)
                self.assertGreaterEqual(iou, tolerance, f"Low IoU ({iou}) at time {timestamp_ros}")

                # Optional: Calculate other metrics
                offset = np.linalg.norm(np.array(bbox_ros) - np.array(bbox_gt))
                self.assertLessEqual(offset, 10, f"High offset ({offset}) at time {timestamp_ros}")

                # Move to next ground truth entry
                ground_truth_idx += 1

            elif timestamp_ros < timestamp_gt:
                # If ROS bag timestamp is earlier than ground truth, continue to next ROS msg
                continue
            else:
                # If ROS bag timestamp is later, increment ground truth index to catch up
                ground_truth_idx += 1

            # Stop if we run out of ground truth data
            if ground_truth_idx >= len(self.ground_truth):
                break

    def tearDown(self):
        self.reader.close()

if __name__ == '__main__':
    unittest.main()


# ground truth format:
# <timestamp> <x_min> <y_min> <x_max> <y_max>
# <timestamp> <x_min> <y_min> <x_max> <y_max>

# metrics for comparison:
# Bounding Box Overlap: Calculate Intersection over Union (IoU) between the ground truth bounding box and the predicted bounding box.
# Coordinate Accuracy: Calculate the offset of each bounding box coordinate.
# FPS Consistency: Check that the timestamps in the bag data follow the expected frequency.