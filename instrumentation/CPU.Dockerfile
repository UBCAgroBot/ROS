FROM python:3.10-slim-buster

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
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

RUN python3 -m pip install --upgrade pip

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

EXPOSE 8888

CMD ["jupyter", "notebook", "--ip=0.0.0.0", "--port=8888", "--no-browser", "--allow-root"]