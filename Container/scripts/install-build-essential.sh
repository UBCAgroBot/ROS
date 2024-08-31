#!/usr/bin/env bash
set -ex

echo 'installing build-essential'

apt -y update && apt -y upgrade
apt-get install -y --no-install-recommends
    locales \
    locales-all \
    tzdata \
locale-gen en_US $LANG
update-locale LC_ALL=$LC_ALL LANG=$LANG
locale
apt-get install -y --no-install-recommends \
    build-essential \
    software-properties-common \
    apt-transport-https \
    ca-certificates \
    lsb-release \
    pkg-config \
    gnupg \
    git \
    gdb \
    wget \
    curl \
    nano \
    zip \
    unzip \
    time \
    sshpass \
    ssh-client
apt-get clean 
rm -rf /var/lib/apt/lists/* 
gcc --version 
g++ --version