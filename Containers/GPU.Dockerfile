# Use an official NVIDIA runtime image with CUDA
FROM nvidia/cuda:12.2.2-cudnn8-runtime-ubuntu20.04

# Set environment variables to prevent prompts during package installations
ENV DEBIAN_FRONTEND=noninteractive

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    git \
    ca-certificates \
    libjpeg-dev \
    libpng-dev \
    libavcodec-dev \
    libavformat-dev \
    libswscale-dev \
    libv4l-dev \
    libxvidcore-dev \
    libx264-dev \
    libgtk2.0-dev \
    pkg-config \
    python3 \
    python3-pip \
    python3-dev \
    libnvinfer-dev \
    libnvonnxparsers-dev \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Upgrade pip
RUN python3 -m pip install --upgrade pip

# Install CUDA-enabled Python packages
RUN pip install --no-cache-dir \
    numpy \
    scipy \
    scikit-learn \
    matplotlib \
    notebook \
    albumentations \
    opencv-python-headless \
    torch \
    torchvision \
    torchaudio \
    wandb \
    onnx \
    onnxruntime-gpu \
    optimum \
    cupy-cuda11x \
    numba \
    # pycuda \
    tensorflow-gpu \
    torch-tensorrt \
    tensorrt

# Expose port for Jupyter Notebook
EXPOSE 8888

# Set the default command to run shell
CMD ["jupyter", "notebook", "--ip=0.0.0.0", "--port=8888", "--no-browser", "--allow-root"]