# CUDA: 12.2.1
# cuDNN: 8.9.4
# TensorRT: 8.6.2
# Ubuntu 22.04
# Python 3.10
# Jetson Linux: 36.3
# Zed SDK: 4.1.3

FROM nvcr.io/nvidia/l4t-jetpack:r36.3.0

ENV DEBIAN_FRONTEND=noninteractive \
    LANGUAGE=en_US:en \
    LANG=en_US.UTF-8 \
    LC_ALL=en_US.UTF-8
    # OPENBLAS_CORETYPE=ARMV8 \
    # OPENCV_URL=https://nvidia.box.com/shared/static/ngp26xb9hb7dqbu6pbs7cs9flztmqwg0.gz \
    # OPENCV_VERSION=4.8.1 \
    # OPENCV_DEB=OpenCV-4.8.1-aarch64.tar.gz

COPY ./scripts /scripts

RUN /scripts/install-build-essential.sh

RUN /scripts/install-cmake.sh

# RUN /scripts/install-opencv-cuda.sh

RUN /scripts/install-zed.sh

RUN /scripts/install-ros2.sh

RUN /scripts/install-jetpack-packages.sh

ENTRYPOINT ["/bin/bash"]