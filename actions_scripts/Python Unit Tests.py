import unittest
import numpy as np
from my_package.bbox_node import BBoxNode

def calculate_iou(boxA, boxB):
    """
    Calculate the Intersection over Union (IoU) between two bounding boxes.
    """
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])

    # Compute the area of intersection rectangle
    interArea = max(0, xB - xA) * max(0, yB - yA)

    # Compute the area of both bounding boxes
    boxAArea = (boxA[2] - boxA[0]) * (boxA[3] - boxA[1])
    boxBArea = (boxB[2] - boxB[0]) * (boxB[3] - boxB[1])

    # Compute the Intersection over Union (IoU)
    iou = interArea / float(boxAArea + boxBArea - interArea)

    return iou

def calculate_centroid(bbox):
    """
    Calculate the centroid of the bounding box.
    """
    x_min, y_min, x_max, y_max = bbox
    centroid_x = (x_min + x_max) / 2.0
    centroid_y = (y_min + y_max) / 2.0
    return centroid_x, centroid_y

class TestBBoxNode(unittest.TestCase):
    def setUp(self):
        # Create an instance of the BBoxNode
        self.node = BBoxNode()

        # Ground truth bounding box for comparison
        self.ground_truth_bbox = [110, 160, 210, 260]  # Example actual bounding box

    def test_bbox_output(self):
        # Simulate an image message and call run_inference
        predicted_bbox = self.node.run_inference(None)  # Normally, this would be the image message

        # Calculate IoU (Intersection over Union)
        iou = calculate_iou(predicted_bbox, self.ground_truth_bbox)
        self.assertGreater(iou, 0.5, "IoU is too low! Bounding box prediction is inaccurate.")

        # Calculate centroid offset
        predicted_centroid = calculate_centroid(predicted_bbox)
        ground_truth_centroid = calculate_centroid(self.ground_truth_bbox)

        offset_x = abs(predicted_centroid[0] - ground_truth_centroid[0])
        offset_y = abs(predicted_centroid[1] - ground_truth_centroid[1])

        self.assertLess(offset_x, 15, f"X centroid offset too large: {offset_x}")
        self.assertLess(offset_y, 15, f"Y centroid offset too large: {offset_y}")

    def test_performance_report(self):
        # Simulate inference results for multiple images
        predicted_bboxes = [
            [100, 150, 200, 250],
            [110, 160, 210, 260],
            [105, 155, 205, 255]
        ]
        ground_truth_bboxes = [
            [110, 160, 210, 260],
            [110, 160, 210, 260],
            [110, 160, 210, 260]
        ]

        total_iou = 0
        total_offset_x = 0
        total_offset_y = 0

        for pred_bbox, gt_bbox in zip(predicted_bboxes, ground_truth_bboxes):
            iou = calculate_iou(pred_bbox, gt_bbox)
            total_iou += iou

            pred_centroid = calculate_centroid(pred_bbox)
            gt_centroid = calculate_centroid(gt_bbox)

            total_offset_x += abs(pred_centroid[0] - gt_centroid[0])
            total_offset_y += abs(pred_centroid[1] - gt_centroid[1])

        # Generate performance metrics
        avg_iou = total_iou / len(predicted_bboxes)
        avg_offset_x = total_offset_x / len(predicted_bboxes)
        avg_offset_y = total_offset_y / len(predicted_bboxes)

        print(f"Performance Report:")
        print(f"Average IoU: {avg_iou:.2f}")
        print(f"Average Centroid Offset X: {avg_offset_x:.2f}")
        print(f"Average Centroid Offset Y: {avg_offset_y:.2f}")

        # Assert performance is within acceptable limits
        self.assertGreater(avg_iou, 0.5, "Average IoU is too low!")
        self.assertLess(avg_offset_x, 15, "Average X centroid offset is too large!")
        self.assertLess(avg_offset_y, 15, "Average Y centroid offset is too large!")

if __name__ == '__main__':
    unittest.main()

# colcon test --packages-select my_package
