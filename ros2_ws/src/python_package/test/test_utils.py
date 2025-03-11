import unittest
import cv2
import numpy as np
import os
from pathlib import Path
from python_package.scripts.utils import ModelInference

class TestModelInference(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # Set up paths
        cls.test_dir = Path(__file__).parent
        cls.data_dir = cls.test_dir / "test_data"
        cls.output_dir = cls.test_dir / "test_output"
        cls.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize model with test weights
        cls.model = ModelInference(
            weights_path="",  # TODO: Update with actual test weights path
            precision="fp16"
        )
        
        # Load test image
        cls.test_image = cv2.imread(str(cls.data_dir / "test_image.jpg"))
        if cls.test_image is None:
            raise FileNotFoundError("Test image not found")

    def test_inference_output_format(self):
        """Test if inference returns expected format and valid boxes"""
        out_img, confidences, boxes = self.model.inference(self.test_image)
        
        self.assertIsInstance(out_img, np.ndarray)
        self.assertIsInstance(confidences, np.ndarray)
        self.assertIsInstance(boxes, np.ndarray)
        
        if len(boxes) > 0:
            # Check if boxes are normalized (between 0 and 1)
            self.assertTrue(np.all(boxes >= 0))
            self.assertTrue(np.all(boxes <= 1))

    def test_roi_coordinates(self):
        """Test if ROI coordinates are valid"""
        roi_coords = self.model.get_roi_coordinates(self.test_image)
        x1, y1, x2, y2 = roi_coords
        
        height, width = self.test_image.shape[:2]
        
        # Check if coordinates are within image bounds
        self.assertTrue(0 <= x1 < width)
        self.assertTrue(0 <= y1 < height)
        self.assertTrue(x1 < x2 <= width)
        self.assertTrue(y1 < y2 <= height)

    def test_color_filtering(self):
        """Test color filtering on known test image"""
        # Create synthetic boxes for testing
        test_boxes = np.array([[0.2, 0.2, 0.4, 0.4],  # Normalized coordinates
                             [0.6, 0.6, 0.8, 0.8]])
        
        filtered_boxes = self.model.object_filter(self.test_image, test_boxes)
        
        self.assertIsInstance(filtered_boxes, list)
        if len(filtered_boxes) > 0:
            self.assertEqual(len(filtered_boxes[0]), 4)  # Each box should have 4 coordinates

    def test_box_drawing(self):
        """Test bounding box drawing functionality"""
        test_boxes = [[100, 100, 200, 200],
                     [300, 300, 400, 400]]
        
        # Test with ROI
        drawn_image = self.model.draw_boxes(
            self.test_image.copy(),
            test_boxes,
            with_roi=True,
            with_roi_shift=True,
            velocity=10
        )
        
        self.assertEqual(drawn_image.shape, self.test_image.shape)
        
        # Save output for visual inspection
        output_path = self.output_dir / "test_boxes.jpg"
        cv2.imwrite(str(output_path), drawn_image)

    def test_postprocessing(self):
        """Test complete postprocessing pipeline"""
        # Create test data
        test_confidences = np.array([0.8, 0.9])
        test_boxes = np.array([[0.2, 0.2, 0.4, 0.4],
                             [0.6, 0.6, 0.8, 0.8]])
        
        processed_boxes = self.model.postprocess(
            test_confidences,
            test_boxes,
            self.test_image,
            velocity=10
        )
        
        self.assertIsInstance(processed_boxes, list)
        if len(processed_boxes) > 0:
            # Check if processed boxes are in pixel coordinates
            height, width = self.test_image.shape[:2]
            for box in processed_boxes:
                x1, y1, x2, y2 = box
                self.assertTrue(0 <= x1 < width)
                self.assertTrue(0 <= y1 < height)
                self.assertTrue(x1 < x2 <= width)
                self.assertTrue(y1 < y2 <= height)

    def calculate_iou(box1, box2):
        """Calculate Intersection over Union"""
        x1 = max(box1[0], box2[0])
        y1 = max(box1[1], box2[1])
        x2 = min(box1[2], box2[2])
        y2 = min(box1[3], box2[3])

        intersection = max(0, x2 - x1) * max(0, y2 - y1)
        area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
        area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
        
        return intersection / (area1 + area2 - intersection)

if __name__ == '__main__':
    unittest.main()

# python3 workspace_python/ros2_ws/src/python_package/test/test_utils.py