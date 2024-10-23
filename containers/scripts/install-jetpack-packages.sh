#!/usr/bin/env bash
set -ex

echo 'installing GPU packages'

apt-get install -y --no-install-recommends libboost-all-dev

echo 'export CPATH=$CPATH:/usr/local/cuda-12.2/targets/aarch64-linux/include' >> ~/.bashrc
echo 'export LIBRARY_PATH=$LIBRARY_PATH:/usr/local/cuda-12.2/targets/aarch64-linux/lib' >> ~/.bashrc
echo 'export PATH=$PATH:/usr/local/cuda-12.2/bin' >> ~/.bashrc
echo 'export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/local/cuda-12.2/lib64' >> ~/.bashrc
# echo 'export LD_LIBRARY_PATH=/usr/local/lib:$LD_LIBRARY_PATH' >> ~/.bashrc
# echo 'export PYTHONPATH=/usr/local/lib/python3.10/site-packages/:$PYTHONPATH' >> ~/.bashrc

wget "http://jetson.webredirect.org/jp6/cu122/+f/54d/5094104195427/onnxruntime_gpu-1.17.0-cp310-cp310-linux_aarch64.whl#sha256=54d509410419542764627aebf6542734a7c5b4d260ad0860b650e05f38b1b47a"
wget "http://jetson.webredirect.org/jp6/cu122/+f/9aa/972ec292a04c6/opencv_contrib_python-4.10.0+6a181ce-cp310-cp310-linux_aarch64.whl#sha256=9aa972ec292a04c63b1106d042098d2026ad898daa8ca2860754c96a0bb3abfe"
wget "http://jetson.webredirect.org/jp6/cu122/+f/6c3/d010be06060e4/pycuda-2024.1-cp310-cp310-linux_aarch64.whl#sha256=6c3d010be06060e40073696630339d10187c0d0f752ab1962a5e615d0c747e9e"
wget "http://jetson.webredirect.org/jp6/cu122/+f/caa/8de487371c7f6/torch-2.4.0-cp310-cp310-linux_aarch64.whl#sha256=caa8de487371c7f66b566025700635c30728032e0a3acf1c3183cec7c2787f94"
wget "http://jetson.webredirect.org/jp6/cu122/+f/8c0/114b6c62bfa3d/torchvision-0.19.0a0+48b1edf-cp310-cp310-linux_aarch64.whl#sha256=8c0114b6c62bfa3d60d08b51f1467e0ea1ee4916e5b4b1084db50c2c1f345d93"
wget "http://jetson.webredirect.org/jp6/cu122/+f/7b4/730d4b147df15/cupy-13.0.0rc1-cp310-cp310-linux_aarch64.whl#sha256=7b4730d4b147df15f9cfd91adbe79fb683b586ce38dde9ed9fad4dc1cab11408"

pip3 install -U --no-cache-dir --verbose onnxruntime_gpu-1.17.0-cp310-cp310-linux_aarch64.whl
pip3 install -U --no-cache-dir --verbose opencv_contrib_python-4.10.0+6a181ce-cp310-cp310-linux_aarch64.whl
pip3 install -U --no-cache-dir --verbose pycuda-2024.1-cp310-cp310-linux_aarch64.whl
pip3 install -U --no-cache-dir --verbose torch-2.4.0-cp310-cp310-linux_aarch64.whl
pip3 install -U --no-cache-dir --verbose torchvision-0.19.0a0+48b1edf-cp310-cp310-linux_aarch64.whl
pip3 install -U --no-cache-dir --verbose cupy-13.0.0rc1-cp310-cp310-linux_aarch64.whl

pip3 install -U --no-cache-dir --verbose jetson-stats numpy onnx ultralytics argparse
pip3 show numpy && python3 -c 'import numpy; print(numpy.__version__)'
pip3 show onnx && python3 -c 'import onnx; print(onnx.__version__)'

apt-get install -y --no-install-recommends ros-humble-cv-bridge