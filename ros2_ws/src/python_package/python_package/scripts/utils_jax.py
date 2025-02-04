import cv2
import os
import numpy as np
from typing import Tuple, List
import jax
import jax.numpy as jnp
from functools import partial

class ModelInferenceJax:
    # Constants
    ROI_ROWS = 300
    ROI_COLS = 300
    CONTOUR_AREA_THRESHOLD = 500
    POSTPROCESS_OUTPUT_SHAPE = (640, 640)
    
    def __init__(self, weights_path=None, precision=None):
        if weights_path is None or precision is None:
            self.model = None
            return
        else:
            self.precision = precision
            if not os.path.exists(weights_path):
                print(f"Weights file not found at {weights_path}")
                raise FileNotFoundError(f"Weights file not found at {weights_path}")

            # Note: Model loading would be handled separately
            # JAX doesn't have direct YOLO support, so you might need a custom solution
            self.device = jax.devices()[0]  # Get default device

    @partial(jax.jit, static_argnums=(0,))
    def preprocess(self, image: np.ndarray):
        """
        Preprocesses the input image using JAX acceleration.
        """
        # Convert image to float32 and normalize
        image = jnp.array(image, dtype=jnp.float32) / 255.0
        
        # Convert BGR to RGB (JAX implementation)
        image = image[..., ::-1]
        
        return image

    @partial(jax.jit, static_argnums=(0,))
    def color_threshold(self, hsv_image: jnp.ndarray) -> jnp.ndarray:
        """
        JAX-accelerated color thresholding
        """
        lower_hue = 35
        upper_hue = 80
        
        hue = hsv_image[..., 0]
        saturation = hsv_image[..., 1]
        
        lower_mask = hue > lower_hue
        upper_mask = hue < upper_hue
        saturation_mask = saturation > 50
        
        return jnp.logical_and(jnp.logical_and(lower_mask, upper_mask), saturation_mask)

    @partial(jax.jit, static_argnums=(0,))
    def object_filter(self, image: jnp.ndarray, bboxes: jnp.ndarray) -> List[Tuple[int, int, int, int]]:
        """
        JAX-accelerated object filtering based on color thresholds
        """
        detections = []
        
        def process_bbox(bbox):
            x1, y1, x2, y2 = [int(coord) for coord in bbox]
            roi = image[y1:y2, x1:x2]
            
            # Convert to HSV using JAX operations
            # Note: You'll need to implement HSV conversion in JAX
            hsv = self._bgr_to_hsv(roi)
            
            mask = self.color_threshold(hsv)
            
            if jnp.sum(mask) > self.CONTOUR_AREA_THRESHOLD:
                return (x1, y1, x2, y2)
            return None
        
        # Process each bbox
        filtered_boxes = jax.vmap(process_bbox)(bboxes)
        return [box for box in filtered_boxes if box is not None]

    @partial(jax.jit, static_argnums=(0,))
    def _bgr_to_hsv(self, bgr: jnp.ndarray) -> jnp.ndarray:
        """
        JAX implementation of BGR to HSV conversion
        
        Args:
            bgr: Input image in BGR format with values in [0, 1]
            
        Returns:
            HSV image with H in [0, 180], S and V in [0, 255]
        """
        # Separate BGR channels
        b, g, r = bgr[..., 2], bgr[..., 1], bgr[..., 0]
        
        # Calculate Value (V)
        v = jnp.maximum(jnp.maximum(r, g), b)
        
        # Calculate Saturation (S)
        diff = v - jnp.minimum(jnp.minimum(r, g), b)
        s = jnp.where(v == 0, 0, diff / v)
        
        # Calculate Hue (H)
        h = jnp.zeros_like(v)
        
        # When r is max
        r_max = (v == r)
        h = jnp.where(r_max, 60 * (g - b) / (diff + 1e-7), h)
        
        # When g is max
        g_max = (v == g)
        h = jnp.where(g_max, 120 + 60 * (b - r) / (diff + 1e-7), h)
        
        # When b is max
        b_max = (v == b)
        h = jnp.where(b_max, 240 + 60 * (r - g) / (diff + 1e-7), h)
        
        # Adjust negative values
        h = jnp.where(h < 0, h + 360, h)
        
        # Scale values to match OpenCV ranges
        h = h / 2  # Convert to [0, 180]
        s = s * 255  # Convert to [0, 255]
        v = v * 255  # Convert to [0, 255]
        
        return jnp.stack([h, s, v], axis=-1)

    def postprocess(self, confidences: jnp.ndarray, bbox_array: jnp.ndarray, 
                   raw_image: np.ndarray, velocity: float = 0) -> List[Tuple[int, int, int, int]]:
        """
        Postprocesses the bounding boxes using JAX acceleration where possible
        """
        detections = self.object_filter(raw_image, bbox_array)
        detections = self.verify_object(raw_image, detections, velocity)
        return detections

    @partial(jax.jit, static_argnums=(0,))
    def verify_object(self, image: jnp.ndarray, detections: List[Tuple[int, int, int, int]], 
                     velocity: float) -> List[Tuple[int, int, int, int]]:
        """
        JAX-accelerated object verification based on ROI and velocity
        
        Args:
            image: Input image
            detections: List of bounding boxes in format [(x1, y1, x2, y2), ...]
            velocity: Velocity value for ROI shift
            
        Returns:
            List of verified and adjusted bounding boxes
        """
        # Convert list of tuples to JAX array
        boxes = jnp.array(detections)
        
        # Get image dimensions
        height, width = image.shape[:2]
        
        # Calculate ROI coordinates
        roi_x1 = width // 2 - self.ROI_COLS // 2
        roi_y1 = height // 2 - self.ROI_ROWS // 2
        roi_x2 = roi_x1 + self.ROI_COLS
        roi_y2 = roi_y1 + self.ROI_ROWS
        
        # Calculate velocity-shifted ROI
        shifted_roi_x1 = roi_x1 - int(velocity)
        shifted_roi_x2 = roi_x2 - int(velocity)
        shifted_roi_y1 = roi_y1
        shifted_roi_y2 = roi_y2
        
        def process_box(box):
            x1, y1, x2, y2 = box
            
            # Check if box is outside shifted ROI
            outside_x = jnp.logical_or(
                jnp.logical_and(x1 < shifted_roi_x1, x2 < shifted_roi_x1),
                jnp.logical_and(x1 > shifted_roi_x2, x2 > shifted_roi_x2)
            )
            outside_y = jnp.logical_or(
                jnp.logical_and(y1 < shifted_roi_y1, y2 < shifted_roi_y1),
                jnp.logical_and(y1 > shifted_roi_y2, y2 > shifted_roi_y2)
            )
            is_outside = jnp.logical_or(outside_x, outside_y)
            
            # Clip coordinates to ROI boundaries
            x1_new = jnp.clip(x1, shifted_roi_x1, shifted_roi_x2)
            x2_new = jnp.clip(x2, shifted_roi_x1, shifted_roi_x2)
            y1_new = jnp.clip(y1, shifted_roi_y1, shifted_roi_y2)
            y2_new = jnp.clip(y2, shifted_roi_y1, shifted_roi_y2)
            
            # Create adjusted box
            adjusted_box = jnp.array([x1_new, y1_new, x2_new, y2_new])
            
            # Return adjusted box or None-indicator (zeros) if outside ROI
            return jnp.where(is_outside, jnp.zeros(4), adjusted_box)
        
        # Apply processing to all boxes using vmap
        processed_boxes = jax.vmap(process_box)(boxes)
        
        # Filter out invalid boxes (those that were outside ROI)
        valid_mask = jnp.any(processed_boxes != 0, axis=1)
        valid_boxes = processed_boxes[valid_mask]
        
        return [(int(x1), int(y1), int(x2), int(y2)) 
                for x1, y1, x2, y2 in valid_boxes.tolist()]