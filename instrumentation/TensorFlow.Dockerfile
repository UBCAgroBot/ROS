FROM nvcr.io/nvidia/tensorflow:24.08-tf2-py3

COPY ../conversion_tools /conversion_tools

# ORT
RUN pip3 install -U --no-cache-dir --verbose jetson-stats numpy onnx pycuda argparse

ENTRYPOINT [ "/bin/bash" ]