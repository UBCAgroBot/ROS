import tensorflow as tf
from tensorflow.python.tools import optimize_for_inference_lib
import tf2onnx

model_path = input("Enter the full/absolute path to the model: ")
model_name = input("Enter the name of the model: ")

loaded = tf.saved_model.load(model_path)
print(list(loaded.signatures.keys()))  # Print the names of the signatures

signature_key = input("Please enter the name for the signature based on the above list: ")
infer = loaded.signatures[signature_key]

# Get the names of the input and output nodes
input_node_name = list(infer.structured_input_signature[1].keys())[0]
output_node_name = list(infer.structured_outputs.keys())[0]

# Convert the model
onnx_graph = tf2onnx.tfonnx.process_tf_graph(infer.graph, input_names=[input_node_name+":0"], output_names=[output_node_name+":0"])
model_proto = onnx_graph.make_model("test")
with open(f"{model_name}.onnx", "wb") as f:
    f.write(model_proto.SerializeToString())

print("Model converted successfully")

# improvements:
# Model Pruning: Pruning is a technique in deep learning where you remove the weights of certain neurons which are less important. This can help in reducing the size of the model and hence improve the performance during conversion.
# Quantization: Quantization is a process that reduces the numerical precision of the model's weights, which can lead to a significant reduction in both the memory requirement and computational cost of the model.
# Use the Latest ONNX Opset Version: The ONNX opset version corresponds to the set of operators and their versions supported. Newer opset versions can have optimizations that were not available in previous versions. You can set the opset version with the opset parameter in tf2onnx.convert.from_graph_def.