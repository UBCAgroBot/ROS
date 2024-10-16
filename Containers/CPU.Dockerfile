# Use an official Python runtime as a parent image
FROM python:3.10-slim-buster

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
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Upgrade pip
RUN python3 -m pip install --upgrade pip

# Install CPU-only Python packages
RUN pip3 install --upgrade --no-cache-dir --verbose \
    numpy \
    wandb \
    scipy \
    scikit-learn \
    matplotlib \
    notebook \
    albumentations \
    onnx \
    onnxruntime \
    # tensorflow-cpu \
    tensorflow-cpu-aws \
    torch --extra-index-url https://download.pytorch.org/whl/cpu

# Expose port for Jupyter Notebook
EXPOSE 8888

# Set the default command to run jupyter notebook
CMD ["jupyter", "notebook", "--ip=0.0.0.0", "--port=8888", "--no-browser", "--allow-root"]