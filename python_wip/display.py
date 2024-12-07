import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from ultralytics import YOLO
import os
import time

# should take boolean for displaying roi or not, import path verify library for image source
# copy ultralytics color picker selection

def display_annotated_image(image, boxes, labels=None, colors=None, window_name="Annotated Image"):
    """
    Display an image with annotated bounding boxes in an OpenCV window using Arial font for labels.

    Args:
        image (numpy.ndarray): The input image as a NumPy array (H, W, C).
        boxes (list or numpy.ndarray): List of bounding boxes, each in the format [x1, y1, x2, y2].
        labels (list, optional): List of labels corresponding to the bounding boxes. Defaults to None.
        colors (list, optional): List of colors for each bounding box in BGR format. Defaults to random colors.
        window_name (str): Name of the OpenCV window. Defaults to "Annotated Image".
    """
    # Convert OpenCV image to PIL Image
    # image = cv2.imread(image_path)
    
    # Load Arial font
    try:
        font_path = "/usr/share/fonts/truetype/msttcorefonts/arial.ttf"
        font = ImageFont.truetype(font_path, size=14)
    except IOError:
        font = ImageFont.load_default()
    
    if labels is None:
        labels = ["12%" for _ in boxes]
    if colors is None:
        colors = [(0, 255, 0) for _ in boxes]
    
    tic = time.perf_counter_ns()
    annotated_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    pil_image = Image.fromarray(annotated_image)
    
    draw = ImageDraw.Draw(pil_image)
    
    for box, label, color in zip(boxes, labels, colors):
        x1, y1, x2, y2 = map(int, box)
        draw.rectangle([x1, y1, x2, y2], outline=(0, 255, 0), width=3)

        if label:
            # Get text size using font.getbbox()
            bbox = font.getbbox(label)
            text_width = bbox[2] - bbox[0]
            text_height = bbox[3] - bbox[1]
            
            # Right-align the text box with the bounding box
            text_x = x2 - text_width
            
            # Make the box bigger in the vertical direction
            padding = 5
            text_height += padding
            
            # Shift the font up slightly
            text_y = y1 - text_height - 2 if y1 - text_height - 2 > 0 else y1 + 2
            # Draw background rectangle for text
            draw.rectangle([text_x, text_y, text_x + text_width, text_y + text_height], fill=color)
            # Draw text
            draw.text((text_x, text_y), label, fill=(255, 255, 255), font=font)

    # Convert annotated image back to OpenCV format
    image = cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2BGR)
    toc = time.perf_counter_ns()
    pil_transform = toc-tic

    overlay = image.copy()
    output = image.copy()
    # Define rectangle parameters
    top_left = (100, 100)
    bottom_right = (300, 300)
    alpha = 0.5
    
    cv2.rectangle(overlay, top_left, bottom_right, (0, 255, 0), -1)
    cv2.addWeighted(overlay, alpha, output, 1 - alpha, 0, output)
    tic = time.perf_counter_ns()

    overlay = tic-toc
    
    return output
    
    print(f"pillow: {pil_transform/1e6} ms")
    print(f"overlay: {overlay/1e6} ms")
    
    # Display the image in an OpenCV window
    cv2.imshow(window_name, output)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

# Example usage
# if __name__ == "__main__":
    # # Create a blank white image
    # img = np.ones((400, 400, 3), dtype=np.uint8) * 255

    # # Define some bounding boxes and labels

    # # Call the function to display the image with annotations
    # display_annotated_image(img, example_boxes, example_labels)

# Example usage
example_boxes = [[50, 50, 150, 150], [200, 80, 300, 200]]
example_labels = ["Object A", "Object B"]
# # run_inference_and_display('python_wip/example.jpg', 'python_wip/Maize.pt')
display_annotated_image('python_wip/example.jpg', example_boxes, example_labels)