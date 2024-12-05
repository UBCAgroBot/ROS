import time
import onnxruntime as ort
import numpy as np
import cv2
import logging
import onnx

logging.basicConfig(format='%(message)s', level=logging.INFO)


class Results:
    def __init__(self, model_path, image_path, imgsz=640, gpu=True, precision="fp16"):
        """
        Initialize the Results class for ONNX Runtime analysis.
        """
        self.model_path = model_path
        self.image_path = image_path
        self.imgsz = imgsz
        self.gpu = gpu
        self.precision = precision
        self.onnx_session = None
        self.results = []
        self.inference_times = []
        self.confidence_scores = []

        # Load the model during initialization
        self._load_model()

    def _load_model(self):
        """Load the ONNX model into an inference session."""
        providers = ["CUDAExecutionProvider"] if self.gpu else ["CPUExecutionProvider"]
        self.onnx_session = ort.InferenceSession(self.model_path, providers=providers)
        logging.info(f"Model loaded from {self.model_path} using {'GPU' if self.gpu else 'CPU'}.") # use gpu if available

    def _get_images(self):
        """Load images from the specified path."""
        images = []
        for image_src in self.image_path:
            image = cv2.imread(image_src)
            image = cv2.resize(image, self.imgsz)
            images.append(image.astype(np.float16) if self.precision == "fp16" else image.astype(np.float32))

        logging.info(f"{len(images)} images loaded and preprocessed.")
        return images
    
    def predict(self, confidence_threshold=0.5):
        """
        Perform inference and collect results.
        """
        images = self._get_images()
        input_name = self.onnx_session.get_inputs()[0].name

        self.results.clear()
        self.inference_times.clear()
        self.confidence_scores.clear()

        for image in images:
            image = np.expand_dims(image, axis=0)  # Add batch dimension
            tic = time.perf_counter_ns()
            outputs = self.onnx_session.run(None, {input_name: image})
            toc = time.perf_counter_ns()

            # Record inference time
            inference_time = (toc - tic) / 1e6  # Milliseconds
            self.inference_times.append(inference_time)

            # Process outputs (assuming the first output contains predictions)
            output = outputs[0]  # Replace with your post-processing logic
            filtered_output = [o for o in output if o[-1] >= confidence_threshold]
            self.results.append(filtered_output)
            self.confidence_scores.extend(o[-1] for o in filtered_output)

        avg_time = sum(self.inference_times) / len(self.inference_times)
        logging.info(f"Average inference time: {avg_time:.2f} ms")
        return self.results

    def display(self):
        """Display the results."""
        images= self._get_images()
        i = 0

        while True:
            image = images[i % len(images)].copy()
            for box in self.results[i % len(self.results)]:
                x1, y1, x2, y2, score = box[:5]
                color = (0, 255, 0) if score >= 0.5 else (0, 0, 255)
                cv2.rectangle(image, (int(x1), int(y1)), (int(x2), int(y2)), color, 2)
                cv2.putText(image, f"{score:.2f}", (int(x1), int(y1) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

            cv2.imshow('Inference Results', image)
            key = cv2.waitKey(0)

            if key == ord('q'):
                cv2.destroyAllWindows()
                break
            elif key == ord('a'):
                i -= 1
            elif key == ord('d'):
                i += 1
    
    def save(self, output_path):
        """Save the results to a file."""
        with open(output_path, 'w') as f:
            for result in self.results:
                f.write(','.join(str(r) for r in result) + '\n')
        logging.info(f"Results saved to {output_path}.")

    def validate(model_path):
        try:
            onnx_model = onnx.load(model_path)
            onnx.checker.check_model(onnx_model)
            print("The model is valid!")
            return True
        except onnx.checker.ValidationError as e:
            print("Model validation failed:", e)
            return False
        
    def compare(self, compared_results):
        """
        Compare the current results with the provided results.
        """
        pass
    


# sample usage

img1 = "IMG_1822_14.JPG"
img2 = "IMG_1828_02.JPG"

img = [img1, img2]

model = "Maize.onnx"
model_opt = "Maize_optimized.onnx"

# check image size 
cv2.imread(img1).shape

# check if optimized model is valid
# if not Results.validate(model_opt):
#     raise ValueError("The model is not valid. Please check the model.")
# else:
#     logging.info("The model is valid.")
#     result_analysis = Results(
#         model_path=model,
#         image_path=img,
#         imgsz=cv2.imread(img1).shape[:2],
#         gpu=True,
#         precision="fp16"
#     )
    # results_opt_analysis = Results(
    #     model_path=model_opt,
    #     image_path=img,
    #     imgsz=cv2.imread(img1).shape[:2],
    #     gpu=True,
    #     precision="fp16"
    # )

    # visulaize and compare

    # # Run predictions
    # results_pred = result_analysis.predict(confidence_threshold=0.5)
    # # results_opt_pred = results_opt_analysis.predict(confidence_threshold=0.5)

    # # Visualize results
    # result_analysis.show_results()
    # # results_opt_analysis.show_results()

# checking the original model
result_analysis = Results(
        model_path=model,
        image_path=img,
        imgsz=cv2.imread(img1).shape[:2],
        gpu=True,
        precision="fp16"
    )

result_analysis.predict(confidence_threshold=0.5)
result_analysis.display()
