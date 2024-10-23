FROM nvcr.io/nvidia/pytorch:24.09-py3

ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update && apt-get install -y --no-install-recommends \
    sudo \
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
    libboost-all-dev \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/* /tmp/* /var/tmp/*

RUN echo 'export CUDA_PATH=/usr/local/cuda-12.6/targets/x86_64-linux/include' >> ~/.bashrc \
    echo 'export LIBRARY_PATH=$LIBRARY_PATH:/usr/local/cuda-12.6/targets/x86_64-linux/lib' >> ~/.bashrc \
    echo 'export PATH=$PATH:/usr/local/cuda-12.6/bin' >> ~/.bashrc \
    echo 'export LD_LIBRARY_PATH=/usr/local/cuda-12.6/lib64/stubs' >> ~/.bashrc

RUN pip3 install --upgrade --no-cache-dir --verbose \
    numpy \
    scipy \
    scikit-learn \
    matplotlib \
    notebook \
    albumentations \
    wandb \
    onnx \
    optimum \
    ultralytics \
    onnxruntime-gpu \
    cupy-cuda12x

RUN apt-get update && apt install -y --no-install-recommends \
    libopencv-dev

ARG USERNAME=vscode
ARG USER_UID=1000
ARG USER_GID=$USER_UID

# Create and switch to user
RUN groupadd --gid $USER_GID $USERNAME \
    && useradd --uid $USER_UID --gid $USER_GID -m $USERNAME --shell /bin/bash \
    && echo "$USERNAME ALL=(ALL) NOPASSWD:ALL" >> /etc/sudoers.d/$USERNAME \
    && chmod 0440 /etc/sudoers.d/$USERNAME

# Set the default user to vscode
USER $USERNAME

# Create workspace so that user own this directory
RUN mkdir -p /home/$USERNAME/workspace
WORKDIR /home/$USERNAME/workspace

ENV LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/opt/hpcx/ucx/lib
ENV CPATH=$CPATH:/usr/local/cuda-12.6/targets/x86_64-linux/include
ENV LIBRARY_PATH=$LIBRARY_PATH:/usr/local/cuda-12.6/targets/x86_64-linux/lib
ENV PATH=$PATH:/usr/local/cuda-12.6/bin
ENV LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/local/cuda-12.6/lib64
ENV PATH=/home/vscode/.local/bin:$PATH

RUN pip3 install opencv-python==4.8.0.76 numpy==1.26.4 pycuda

EXPOSE 8888

ENTRYPOINT [ "/bin/bash" ]