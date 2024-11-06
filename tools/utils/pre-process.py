def preprocess_image(self, image_path):
    """
    Preprocess the image: resize, normalize, and convert to the required format for the model.
    :param image_path: Path to the image file.
    :return: Preprocessed image ready for inference, original image, and scaling factors.
    """
    img = cv2.imread(image_path)
    original_size = img.shape[:2]  # Original size (height, width)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    resized = cv2.resize(img, self.input_size)
    # Normalize the image (assuming mean=0.5, std=0.5 for demonstration)
    normalized = resized / 255.0
    normalized = (normalized - 0.5) / 0.5
    # HWC to CHW format for model input
    input_tensor = np.transpose(normalized, (2, 0, 1)).astype(np.float32)
    input_tensor = np.expand_dims(input_tensor, axis=0)  # Add batch dimension

    # Compute scaling factors to map back to original size
    scale_x = original_size[1] / self.input_size[0]
    scale_y = original_size[0] / self.input_size[1]

    return input_tensor, img, (scale_x, scale_y)