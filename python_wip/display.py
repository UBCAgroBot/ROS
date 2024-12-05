import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from ultralytics import YOLO
import os

def display_annotated_image(image_path, boxes, labels=None, colors=None, window_name="Annotated Image"):
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
    image = cv2.imread(image_path)
    annotated_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    pil_image = Image.fromarray(annotated_image)
    draw = ImageDraw.Draw(pil_image)

    # Load Arial font
    try:
        font_path = "/usr/share/fonts/truetype/msttcorefonts/arial.ttf"
        font = ImageFont.truetype(font_path, size=14)
    except IOError:
        font = ImageFont.load_default()

    if labels is None:
        labels = ["12%" for _ in boxes]
    if colors is None:
        colors = [tuple(np.random.randint(0, 256, size=3).tolist()) for _ in boxes]

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
    annotated_image = cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2BGR)

    # Display the image in an OpenCV window
    cv2.imshow(window_name, annotated_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

# Example usage
# if __name__ == "__main__":
    # # Create a blank white image
    # img = np.ones((400, 400, 3), dtype=np.uint8) * 255

    # # Define some bounding boxes and labels

    # # Call the function to display the image with annotations
    # display_annotated_image(img, example_boxes, example_labels)

def run_inference_and_display(image_path, model_path):
    """
    Run inference on an image using a custom local model and display the resulting annotated image.

    Args:
        image_path (str): Path to the input image.
        model_path (str): Path to the custom local model.
    """
    # Load the custom model
    model = YOLO(model_path)

    # Load the image
    image = cv2.imread(image_path)
    print(image.shape)
    if image is None:
        print("Error: Could not open or find the image.")
        return

    # Run inference
    results = model(image)
    results[0].show()
    print(results[0].boxes.xyxy)

    # # Extract bounding boxes, labels, and scores
    # boxes = results[0].boxes.xyxy.cpu().numpy()  # Bounding boxes
    # scores = results[0].boxes.conf.cpu().numpy()  # Confidence scores
    # labels = results[0].boxes.cls.cpu().numpy() 

    # # Annotate the image
    # annotated_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    # pil_image = Image.fromarray(annotated_image)
    # draw = ImageDraw.Draw(pil_image)

    # # Load Arial font
    # try:
    #     font = ImageFont.truetype("arial.ttf", size=14)
    # except IOError:
    #     font = ImageFont.load_default()

    # for box, label, score in zip(boxes, labels, scores):
    #     x1, y1, x2, y2 = map(int, box)
    #     color = tuple(np.random.randint(0, 256, size=3).tolist())
    #     draw.rectangle([x1, y1, x2, y2], outline=color, width=2)

    #     label_text = f"{int(label)}: {score:.2f}"
    #     bbox = font.getbbox(label_text)
    #     text_width = bbox[2] - bbox[0]
    #     text_height = bbox[3] - bbox[1]
    #     text_x = x1
    #     text_y = y1 - text_height - 2 if y1 - text_height - 2 > 0 else y1 + 2
    #     draw.rectangle([text_x, text_y, text_x + text_width, text_y + text_height], fill=color)
    #     draw.text((text_x, text_y), label_text, fill=(255, 255, 255), font=font)

    # # Convert annotated image back to OpenCV format
    # annotated_image = cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2BGR)

    # # Display the image in an OpenCV window
    # cv2.imshow("Annotated Image", annotated_image)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()

# Example usage
# example_boxes = [[50, 50, 150, 150], [200, 80, 300, 200]]
# example_labels = ["Object A", "Object B"]
# # run_inference_and_display('python_wip/example.jpg', 'python_wip/Maize.pt')
# display_annotated_image('python_wip/example.jpg', example_boxes, example_labels)