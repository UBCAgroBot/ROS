import argparse
import tensorflow as tf
import tf2onnx
from tensorflow.python.tools import optimize_for_inference_lib

def convert_tf_to_onnx(model_path="/home/user/Downloads/model.pt", output_path="/home/user/Downloads/model.onnx", input_names=["input"], output_names=["output"], constant_folding=False):
    # use os module to check if path exists...
    print("Loading the TensorFlow model")
    with tf.io.gfile.GFile(model_path, "rb") as f:
        frozen_graph_def = tf.compat.v1.GraphDef()
        frozen_graph_def.ParseFromString(f.read())
    
    print("Exporting the model to ONNX format")
    onnx_model, _ = tf2onnx.convert.from_graph_def(
        frozen_graph_def,
        input_names=input_names,
        output_names=output_names,
        opset=13 
    )

    with open(output_path, "wb") as f:
        f.write(onnx_model.SerializeToString())
    
    model = onnx.load(output_path)
    onnx.checker.check_model(model)
    print("Model converted successfully")
    print(model.graph)
    
    print(f"Converted ONNX model saved at {output_path}")
    # return loadOnnxModel(onnx_model_str)

if __name__ == "__main__":
    print("Usage: python3 ONNX_TRT.py --model_path=/home/user/Downloads/model.onnx --output_path=/home/user/Downloads/model.engine --FP16=False --INT8=False --strip_weights=False --gs_optimize=False --verbose=False ")
    
    parser = argparse.ArgumentParser(description='Convert Onnx model to TensorRT')
    parser.add_argument('--model_path', type=str, default="/home/user/Downloads/model.onnx", required=False, help='Path to the ONNX model file (.onnx)')
    parser.add_argument('--output_path', type=str, default="/home/user/Downloads/model.engine", required=False, help='Path to save the converted TensorRT model file (.engine)')
    parser.add_argument('--FP16', type=bool, default=False, help="FP16 precision mode")
    parser.add_argument('--INT8', type=bool, default=False, help="INT8 precision mode")
    parser.add_argument('--strip_weights', type=bool, default=False, help="Strip unnecessary weights")
    parser.add_argument('--gs_optimize', type=bool, default=False, help='Use ONNX GraphSurgeon to optimize model first')
    parser.add_argument('--verbose', type=bool, default=False, help="Verbose TensorRT logging")
    args = parser.parse_args()
    
    convert_onnx_to_trt(args.model_path, args.output_path, args.FP16, args.INT8, args.strip_weights, args.gs_optimize, args.verbose)