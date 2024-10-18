FROM nvidia/cuda:12.2.2-cudnn8-runtime-ubuntu20.04

ENV DEBIAN_FRONTEND=noninteractive

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

RUN python3 -m pip install --upgrade pip

RUN pip install --upgrade --no-cache-dir --verbose \
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

EXPOSE 8888

CMD ["jupyter", "notebook", "--ip=0.0.0.0", "--port=8888", "--no-browser", "--allow-root"]