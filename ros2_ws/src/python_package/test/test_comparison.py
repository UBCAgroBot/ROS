import unittest
import cv2
import numpy as np
import torch
import cupy as cp
import time
from pathlib import Path
from python_package.scripts.utils import ModelInference
from python_package.scripts.utils_cupy import ModelInferenceCupy
from python_package.scripts.utils_pytorch import ModelInferencePytorch

class TestPreprocessingMethods(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.test_dir = Path(__file__).parent
        cls.data_dir = cls.test_dir / "test_data"
        cls.output_dir = cls.test_dir / "test_output"
        cls.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize all models
        weights_path = "models/maize/Maize.pt"
        cls.model_cpu = ModelInference(weights_path=weights_path, precision="fp16")
        cls.model_cupy = ModelInferenceCupy(weights_path=weights_path, precision="fp16")
        cls.model_pytorch = ModelInferencePytorch(weights_path=weights_path, precision="fp16")
        
        # Load test image
        cls.test_image = cv2.imread(str(cls.data_dir / "assets/maize/IMG_1822_14.JPG"))
        if cls.test_image is None:
            raise FileNotFoundError("Test image not found")

    def test_preprocessing_speed(self):
        iterations = 100
        methods = {
            "CPU": self.model_cpu,
            "CuPy": self.model_cupy,
            "PyTorch": self.model_pytorch
        }
        
        times = {}
        for name, model in methods.items():
            start_time = time.time()
            for _ in range(iterations):
                _ = model.preprocess(self.test_image)
            times[name] = (time.time() - start_time) / iterations
        
        print(f"\nPreprocessing times (avg over {iterations} iterations):")
        base_time = times["CPU"]
        for name, t in times.items():
            print(f"{name} time: {t*1000:.2f} ms")
            if name != "CPU":
                print(f"{name} speedup: {base_time/t:.2f}x")

    def test_postprocessing_speed(self):
        iterations = 100
        
        # Test data
        test_confidences = np.array([0.8, 0.9])
        test_boxes = np.array([[0.2, 0.2, 0.4, 0.4],
                             [0.6, 0.6, 0.8, 0.8]])
        
        methods = {
            "CPU": self.model_cpu,
            "CuPy": self.model_cupy,
            "PyTorch": self.model_pytorch
        }
        
        times = {}
        outputs = {}
        for name, model in methods.items():
            start_time = time.time()
            for _ in range(iterations):
                outputs[name] = model.postprocess(
                    test_confidences,
                    test_boxes,
                    self.test_image,
                    velocity=10
                )
            times[name] = (time.time() - start_time) / iterations
        
        print(f"\nPostprocessing times (avg over {iterations} iterations):")
        base_time = times["CPU"]
        for name, t in times.items():
            print(f"{name} time: {t*1000:.2f} ms")
            if name != "CPU":
                print(f"{name} speedup: {base_time/t:.2f}x")
        
        # Save visualizations
        for name, output in outputs.items():
            vis_image = self.test_image.copy()
            if name == "CPU":
                vis_image = self.model_cpu.draw_boxes(vis_image, output, velocity=10)
            elif name == "CuPy":
                vis_image = self.model_cupy.draw_boxes(vis_image, output, velocity=10)
            else:
                vis_image = self.model_pytorch.draw_boxes(vis_image, output, velocity=10)
            cv2.imwrite(str(self.output_dir / f"{name.lower()}_output.jpg"), vis_image)

    def test_output_consistency(self):
        """Test if all implementations produce similar results"""
        test_boxes = np.array([[0.2, 0.2, 0.4, 0.4],
                             [0.6, 0.6, 0.8, 0.8]])
        test_confidences = np.array([0.8, 0.9])
        
        outputs = {
            "CPU": self.model_cpu.postprocess(test_confidences, test_boxes, self.test_image, velocity=10),
            "CuPy": self.model_cupy.postprocess(test_confidences, test_boxes, self.test_image, velocity=10),
            "PyTorch": self.model_pytorch.postprocess(test_confidences, test_boxes, self.test_image, velocity=10)
        }
        
        # Compare number of detections
        for name, output in outputs.items():
            print(f"\n{name} detections: {len(output)}")
            
        # Calculate IoU between implementations
        for name1 in outputs:
            for name2 in outputs:
                if name1 < name2:  # Avoid duplicate comparisons
                    iou = self.calculate_iou_batch(outputs[name1], outputs[name2])
                    print(f"IoU between {name1} and {name2}: {iou:.3f}")

    def calculate_iou_batch(self, boxes1, boxes2):
        """Calculate average IoU between two sets of boxes"""
        if not boxes1 or not boxes2:
            return 0.0
            
        ious = []
        for box1 in boxes1:
            for box2 in boxes2:
                iou = self.calculate_iou(box1, box2)
                ious.append(iou)
        return np.mean(ious) if ious else 0.0

    def calculate_iou(self, box1, box2):
        """Calculate IoU between two boxes"""
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