#!/usr/bin/env bash
set -ex

apt update && apt upgrade -y

echo 'installing opencv-python-cuda'

python3 -m pip install --upgrade pip
URL="https://github.com/cudawarped/opencv-python-cuda-wheels/releases/download/4.9.80_13%2F05%2F24/opencv_contrib_python_rolling-4.9.0.80-cp37-abi3-linux_x86_64.whl"
# URL="https://github.com/cudawarped/opencv-python-cuda-wheels/releases/download/4.8.0.20230804/opencv_contrib_python_rolling-4.8.0.20230804-cp36-abi3-linux_x86_64.whl"
FILENAME="opencv_contrib_python_rolling-4.9.0.80-cp37-abi3-linux_x86_64.whl"
wget -O $FILENAME $URL
pip3 install --no-cache-dir --verbose numpy $FILENAME
# python3 -c "import cv2; print(cv2.__version__); print(cv2.getBuildInformation())"

echo 'installing ROS2'

apt install -y --no-install-recommends software-properties-common 
add-apt-repository universe
apt update &&  apt install curl -y
curl -sSL https://raw.githubusercontent.com/ros/rosdistro/master/ros.key -o /usr/share/keyrings/ros-archive-keyring.gpg
echo "deb [arch=$(dpkg --print-architecture) signed-by=/usr/share/keyrings/ros-archive-keyring.gpg] http://packages.ros.org/ros2/ubuntu $(. /etc/os-release && echo $UBUNTU_CODENAME) main" | tee /etc/apt/sources.list.d/ros2.list > /dev/null
apt update && apt upgrade -y
apt install ros-humble-ros-base python3-argcomplete ros-dev-tools -y
echo "source /opt/ros/humble/setup.bash" >> ~/.bashrc
source /opt/ros/humble/setup.bash
printenv | grep -i ROS

echo 'installing ZED SDK'

apt-get update -y || true ; apt-get install --no-install-recommends lsb-release wget less zstd udev  apt-transport-https -y
wget --no-check-certificate -O ZED_SDK_Linux.run https://download.stereolabs.com/zedsdk/4.1/cu121/ubuntu22
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