#!/usr/bin/env bash
set -ex

echo 'installing ZED SDK'

apt-get update -y || true ; apt-get install --no-install-recommends lsb-release wget less zstd udev sudo apt-transport-https -y
echo "# R36 (release), REVISION: 3.0" > /etc/nv_tegra_release ;
wget -q --no-check-certificate -O ZED_SDK_Linux.run https://download.stereolabs.com/zedsdk/4.1/l4t36.3/jetsons
chmod +x ZED_SDK_Linux.run ; ./ZED_SDK_Linux.run silent skip_drivers
rm -rf /usr/local/zed/resources/* 
rm -rf ZED_SDK_Linux.run
rm -rf /var/lib/apt/lists/*
apt-get clean

echo 'installing pyzed'

apt-get update -y || true ; apt-get install --no-install-recommends python3 python3-pip python3-dev python3-setuptools build-essential -y
wget download.stereolabs.com/zedsdk/pyzed -O /usr/local/zed/get_python_api.py
python3 /usr/local/zed/get_python_api.py
python3 -m pip install cython wheel
python3 -m pip install numpy pyopengl *.whl
rm *.whl ; rm -rf /var/lib/apt/lists/* 
apt-get clean

ln -sf /usr/lib/aarch64-linux-gnu/tegra/libv4l2.so.0 /usr/lib/aarch64-linux-gnu/libv4l2.so