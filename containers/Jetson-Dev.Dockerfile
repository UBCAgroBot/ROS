# CUDA: 12.2.1
# TensorRT: 8.6.1.2
# Ubuntu: 22.04
# cuDNN: 8.9.5
# Python: ?
# https://docs.nvidia.com/deeplearning/tensorrt/container-release-notes/index.html

# supposedly TF-TRT and Torch-TRT are not supported in these containers...
# if no work: FROM nvidia/cuda:12.2.2-cudnn8-runtime-ubuntu20.04

FROM nvcr.io/nvidia/tensorrt:23.09-py3

ENV DEBIAN_FRONTEND=noninteractive

COPY ./scripts /scripts

RUN /scripts/install-build-essential.sh

RUN /scripts/install-cmake.sh

RUN /scripts/install-script.sh

RUN /scripts/

ENV LANG=en_US.UTF-8

ENTRYPOINT ["/bin/bash"]