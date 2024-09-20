// Composable Node 1: Image Preprocessing

// This node would take input data, allocate GPU memory for the image, and perform preprocessing in CUDA.

class ImagePreprocessingNode : public rclcpp::Node {
public:
  ImagePreprocessingNode() : Node("image_preprocessing_node") {
    // Allocate GPU memory for image
    cudaMalloc(&d_image, width * height * sizeof(float));
    // Perform preprocessing (e.g., normalization) in CUDA
    preprocess_image<<<grid, block>>>(d_image, width, height);
  }
  
  // Function to get CUDA pointer to the preprocessed image
  float* get_preprocessed_image() {
    return d_image;  // Return pointer to GPU memory
  }

private:
  float* d_image;  // CUDA pointer for image
};
