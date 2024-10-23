FROM ubuntu:22.04

ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update && apt-get install -y \
    sudo \
    python3.10 \
    dos2unix \
    git \
    python3-pip \
    xauth \
    apt-utils \
    dialog \
    libnss3-tools \
    libx11-dev \
    git \
    xz-utils \
    zip \
    unzip \
    && apt clean \
    && rm -rf /var/lib/apt/lists/* /tmp/* /var/tmp/*

COPY ./scripts /scripts

RUN dos2unix /scripts/*.sh

RUN /scripts/install-build-essential.sh
RUN /scripts/install-cmake.sh
RUN /scripts/install-ros2.sh

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

RUN python3 -m pip install --upgrade pip

# Create workspace so that user own this directory
RUN mkdir -p /home/$USERNAME/workspace
WORKDIR /home/$USERNAME/workspace

ENV LANG=en_US.UTF-8

ENTRYPOINT ["/bin/bash"]