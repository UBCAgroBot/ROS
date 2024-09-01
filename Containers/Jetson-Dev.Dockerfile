# CUDA: 12.2.1
# TensorRT: 8.6.1.2
# Ubuntu: 22.04
# cuDNN: 8.9.5
# Python: ?
# https://docs.nvidia.com/deeplearning/tensorrt/container-release-notes/index.html

FROM nvcr.io/nvidia/tensorrt:23.09-py3

ENV DEBIAN_FRONTEND=noninteractive

COPY ./scripts /scripts

RUN /scripts/install-script.sh

ENTRYPOINT ["/bin/bash"]