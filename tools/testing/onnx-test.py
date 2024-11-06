import os
import numpy as np
import onnxruntime as ort
import time
import logging
import tqdm

logging.basicConfig(format='%(message)s', level=logging.INFO)

# add compatibility with float 16?
# cupy, cp.float32, 16
# relative path
# make a dict for each image, with inference time
# add gpu support (conditional imports)

# should take gpu support in here

class Model:
    def __init__(self, model_path=None, onnx_model=None, input_size=(640, 640)):
        """
        Initialize the model by loading the ONNX model with GPU (CUDA) support.
        :param model_path: Path to the ONNX model file.
        :param onnx_model: ONNX model object if passed directly.
        :param input_size: Expected input size for the model (width, height).
        """
        self.input_size = input_size  # Model's expected input size

        if model_path:
            logging.info(f"ONNX: starting from '{model_path}' with input shape (1, 3, {input_size[0]}, {input_size[1]}) BCHW")
            self.session = ort.InferenceSession(model_path, providers=['CUDAExecutionProvider'])
        elif onnx_model:
            self.session = ort.InferenceSession(onnx_model.SerializeToString(), providers=['CUDAExecutionProvider'])
        else:
            raise ValueError("Either model_path or onnx_model must be provided.")

        # Input and output information from the ONNX model
        self.input_name = self.session.get_inputs()[0].name
        self.output_names = [output.name for output in self.session.get_outputs()]

        logging.info(f"ONNX: loaded successfully, using CUDA (GPU)")

    def load_images(self, image_dir='images/'):
        """
        Load images from the specified directory.
        :param image_dir: Directory containing the images to load.
        :return: List of loaded image file paths.
        """
        image_files = [os.path.join(image_dir, img) for img in os.listdir(image_dir) if img.endswith(('.png', '.jpg', '.jpeg'))]
        return image_files

    def infer(self, image_dir='images/'):
        """
        Perform inference on all images in the specified directory and display results with bounding boxes.
        :param image_dir: Directory containing images to run inference on.
        :return: None. Displays images with bounding boxes.
        """
        image_files = self.load_images(image_dir)
        results = {}

        # make tqdm status bar here and call the func inference call
        # from model_inference.py with appropriate args
        
        
        for image_file in image_files:
            input_tensor, original_image, scales = self.preprocess_image(image_file)

            # Log inference start
            logging.info(f"Predict: ONNX inference started on {image_file}")
            start_time = time.time()

            # Perform inference
            outputs = self.session.run(self.output_names, {self.input_name: input_tensor})

            # Measure inference time
            inference_time = time.time() - start_time
            logging.info(f"Inference complete in {inference_time:.2f} seconds.")

            # Post-process and display results
            annotated_image, boxes = self.post_process(outputs, original_image, scales)
            num_boxes = len(boxes)

            # Log number of bounding boxes and confidence
            for i, box in enumerate(boxes):
                logging.info(f"Box {i+1}: [x1={box[0]:.1f}, y1={box[1]:.1f}, x2={box[2]:.1f}, y2={box[3]:.1f}], score={box[4]:.2f}")

            logging.info(f"Total bounding boxes: {num_boxes}")

            # Show the image with bounding boxes
            cv2.imshow('Inference Result', annotated_image)
            cv2.waitKey(0)  # Press any key to continue
            cv2.destroyAllWindows()

# Example usage:
if __name__ == "__main__":
    model_path = 'model.onnx'
    model = Model(model_path=model_path)
    
    # Log ONNX loading details
    logging.info(f"ONNX: model loaded from '{model_path}', performing inference...")
    
    model.infer('test_images/')