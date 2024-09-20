// Composable Node 2: Inference Using TensorRT

// This node would receive the CUDA pointer from Node 1 and use it directly as input to the inference engine, avoiding any memory copy or serialization overhead.

class InferenceNode : public rclcpp::Node {
public:
  InferenceNode() : Node("inference_node") {
    // Set up TensorRT engine and allocate necessary memory
  }

  void run_inference(float* d_preprocessed_image) {
    // Run inference directly on the GPU memory
    trt_inference(d_preprocessed_image);
  }

private:
  // TensorRT inference engine and buffers
};
