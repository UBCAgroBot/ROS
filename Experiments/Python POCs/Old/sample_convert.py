import tensorrt as trt

TRT_LOGGER = trt.Logger(trt.Logger.WARNING)
trt_runtime = trt.Runtime(TRT_LOGGER)
def build_engine(onnx_path, shape = [1,224,224,3]):

   """
   This is the function to create the TensorRT engine
   Args:
      onnx_path : Path to onnx_file. 
      shape : Shape of the input of the ONNX file. 
  """
   with trt.Builder(TRT_LOGGER) as builder, builder.create_network(1) as network, builder.create_builder_config() as config, trt.OnnxParser(network, TRT_LOGGER) as parser:
       config.max_workspace_size = (256 << 20)
       with open(onnx_path, 'rb') as model:
           parser.parse(model.read())
       network.get_input(0).shape = shape
       engine = builder.build_engine(network, config)
       return engine

def save_engine(engine, file_name):
   buf = engine.serialize()
   with open(file_name, 'wb') as f:
       f.write(buf)
def load_engine(trt_runtime, plan_path):
   with open(plan_path, 'rb') as f:
       engine_data = f.read()
   engine = trt_runtime.deserialize_cuda_engine(engine_data)
   return engine

import engine as eng
import argparse
from onnx import ModelProto
import tensorrt as trt 
 
 engine_name = “resnet50.plan”
 onnx_path = "/path/to/onnx/result/file/"
 batch_size = 1 
 
 model = ModelProto()
 with open(onnx_path, "rb") as f:
    model.ParseFromString(f.read())
 
 d0 = model.graph.input[0].type.tensor_type.shape.dim[1].dim_value
 d1 = model.graph.input[0].type.tensor_type.shape.dim[2].dim_value
 d2 = model.graph.input[0].type.tensor_type.shape.dim[3].dim_value
 shape = [batch_size , d0, d1 ,d2]
 engine = eng.build_engine(onnx_path, shape= shape)
 eng.save_engine(engine, engine_name) 