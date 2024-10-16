FROM ubuntu:22.04

ENV DEBIAN_FRONTEND=noninteractive \
    LANGUAGE=en_US:en \
    LANG=en_US.UTF-8 \
    LC_ALL=en_US.UTF-8

RUN apt-get update && apt-get install -y \
    curl \
    gnupg2 \
    lsb-release \
    sudo \
    wget \
    python3.10

COPY ./scripts /scripts

RUN /scripts/install-cmake.sh

RUN /scripts/install-ros2.sh

ENTRYPOINT ["/bin/bash"]