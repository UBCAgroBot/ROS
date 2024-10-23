FROM python:3.10-slim-buster

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
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

ARG USERNAME=vscode
ARG USER_UID=1000
ARG USER_GID=$USER_UID

# Create and switch to user
RUN sudo groupadd --gid $USER_GID $USERNAME \
    && sudo useradd --uid $USER_UID --gid $USER_GID -m $USERNAME --shell /bin/bash \
    && echo "$USERNAME ALL=(ALL) NOPASSWD:ALL" >> /etc/sudoers.d/$USERNAME \
    && sudo chmod 0440 /etc/sudoers.d/$USERNAME

# Set the default user to vscode
USER $USERNAME

RUN python3 -m pip install --upgrade pip

# Create workspace so that user own this directory
RUN mkdir -p /home/$USERNAME/workspace
WORKDIR /home/$USERNAME/workspace

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
    ultralytics \
    optimum \
    torch --extra-index-url https://download.pytorch.org/whl/cpu \
    ultralytics 

EXPOSE 8888

ENTRYPOINT [ "/bin/bash" ]