from ultralytics.utils import plotting
import cv2
import numpy as np
from typing import Union, Optional, List, Dict, Callable
import torch

# Mocking the colors function used in the provided code
def colors(idx, bgr=False):
    palette = [
        (255, 56, 56), (255, 157, 151), (255, 112, 112), (255, 178, 56), 
        (240, 240, 240), (20, 20, 255), (0, 255, 0), (0, 0, 255)
    ]
    c = palette[idx % len(palette)]
    return (c[2], c[1], c[0]) if bgr else c

# Example usage
def main():
    # Load an image using cv2
    image_path = "example.jpg"  # Replace with your image file path
    image = cv2.imread(image_path)
    if image is None:
        print("Failed to load image. Please check the image path.")
        return
    
    # Convert BGR to RGB
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # Prepare the inputs for the `plot_images` function
    images = np.expand_dims(image.transpose(2, 0, 1), axis=0)  # Shape: (1, C, H, W)
    batch_idx = np.array([0, 0])  # Example: Two bounding boxes for the first image
    cls = np.array([0, 1])  # Example classes
    bboxes = np.array([
        [50, 50, 200, 200],  # Bounding box 1: [x1, y1, x2, y2]
        [300, 100, 400, 250]  # Bounding box 2: [x1, y1, x2, y2]
    ], dtype=np.float32)
    confs = np.array([0.9, 0.8])  # Confidence scores
    names = {0: "Class A", 1: "Class B"}  # Class names

    # Plot the images with bounding boxes
    plotting.plot_images(
        images=images,
        batch_idx=batch_idx,
        cls=cls,
        bboxes=bboxes,
        confs=confs,
        fname="/home/user/Downloads/output.jpg",
        names=names,
        save=True
    )

    # Display the result using OpenCV
    output_image = cv2.imread("/home/user/Downloads/output.jpg")
    if output_image is not None:
        cv2.imshow("Image with Bounding Boxes", output_image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    else:
        print("Failed to display the output image.")

if __name__ == "__main__":
    main()