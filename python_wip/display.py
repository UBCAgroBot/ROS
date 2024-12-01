import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

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
    annotated_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    pil_image = Image.fromarray(annotated_image)
    draw = ImageDraw.Draw(pil_image)

    # Load Arial font
    try:
        font = ImageFont.truetype("arial.ttf", size=14)
    except IOError:
        font = ImageFont.load_default()

    if labels is None:
        labels = ["" for _ in boxes]
    if colors is None:
        colors = [tuple(np.random.randint(0, 256, size=3).tolist()) for _ in boxes]

    for box, label, color in zip(boxes, labels, colors):
        x1, y1, x2, y2 = map(int, box)
        draw.rectangle([x1, y1, x2, y2], outline=color, width=2)

        if label:
            # Get text size using font.getbbox()
            bbox = font.getbbox(label)
            text_width = bbox[2] - bbox[0]
            text_height = bbox[3] - bbox[1]
            text_x = x1
            text_y = y1 - text_height - 2 if y1 - text_height - 2 > 0 else y1 + 2
            # Draw background rectangle for text
            draw.rectangle([text_x, text_y, text_x + text_width, text_y + text_height], fill=color)
            # Draw text
            draw.text((text_x, text_y), label, fill=(255, 255, 255), font=font)

    # Convert annotated image back to OpenCV format
    annotated_image = cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2BGR)

    # Display the image in an OpenCV window
    cv2.imshow(window_name, annotated_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

# Example usage
if __name__ == "__main__":
    # Create a blank white image
    img = np.ones((400, 400, 3), dtype=np.uint8) * 255

    # Define some bounding boxes and labels
    example_boxes = [[50, 50, 150, 150], [200, 80, 300, 200]]
    example_labels = ["Object A", "Object B"]

    # Call the function to display the image with annotations
    display_annotated_image(img, example_boxes, example_labels)
