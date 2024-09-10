Onnx TRT improvements:
# improvements
# Use Dynamic Shapes: If your model supports it, using dynamic input shapes can improve the inference speed. This allows TensorRT to optimize the execution plan based on the actual input shapes at runtime.
# Use Streams for Concurrent Execution: If you are running multiple inferences concurrently, you can use CUDA streams to overlap the computation and data transfer of different inferences.
# Use TensorRT's Optimized Layers: Whenever possible, use TensorRT's optimized layers instead of custom layers. TensorRT has highly optimized implementations for many common layers.
# Enable Layer Fusion: Layer fusion combines multiple layers into a single operation, which can reduce memory access and improve speed. This is automatically done by TensorRT during the optimization process.
# Enable Kernel Auto-Tuning: TensorRT automatically selects the best CUDA kernels for the layers in your model. This process can take some time during the first run, but the results are cached and used for subsequent runs.
# Free GPU Memory After Use: After you are done with a TensorRT engine, you should free its memory to make it available for other uses. In Python, you can do this by deleting the engine and calling gc.collect().
# Use Streams for Concurrent Execution: If you are running multiple inferences concurrently, you can use CUDA streams to overlap the computation and data transfer of different inferences. This can reduce the peak memory usage.

PT ONNX:
# Model Pruning: Pruning is a technique in deep learning where you remove the weights of certain neurons which are less important. This can help in reducing the size of the model and hence improve the performance during conversion.
# Quantization: Quantization is a process that reduces the numerical precision of the model's weights, which can lead to a significant reduction in both the memory requirement and computational cost of the model.

TF Onnx:
# Model Pruning: Pruning is a technique in deep learning where you remove the weights of certain neurons which are less important. This can help in reducing the size of the model and hence improve the performance during conversion.
# Quantization: Quantization is a process that reduces the numerical precision of the model's weights, which can lead to a significant reduction in both the memory requirement and computational cost of the model.
# Use the Latest ONNX Opset Version: The ONNX opset version corresponds to the set of operators and their versions supported. Newer opset versions can have optimizations that were not available in previous versions. You can set the opset version with the opset parameter in tf2onnx.convert.from_graph_def.


# General:
use the onnx graphsurgeon?