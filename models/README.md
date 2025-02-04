# Naming Conventions:
- Name: name the model based on the architecture
- Optimizations: name the optimization you performed (ex. pruned, quantized, etc.)
- Framework: name the framework the model was trained on

# Building Engines:
- TensorRT engines are optimized for the architecture/machine they are built on, use the self-service GitHub Actions pipeline to build engines for different architectures on the Jetson for consistent performance