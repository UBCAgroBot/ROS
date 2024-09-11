FROM nvcr.io/nvidia/pytorch:24.08-py3

COPY ../conversion_tools /conversion_tools

# ORT
RUN pip3 install -U --no-cache-dir --verbose jetson-stats numpy onnx pycuda ultralytics

ENTRYPOINT [ "/bin/bash" ]