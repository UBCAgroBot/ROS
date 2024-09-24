import onnx_graphsurgeon as gs
import onnx
import argparse

def optimize_onnx(model_path="/home/user/Downloads/model.onnx"):
    print("Optimizing ONNX model")
    model = onnx.load(model_path)
    graph = gs.import_onnx(model)
    
    print("Graph nodes before optimization:")
    for node in graph.nodes:
        print(node)

    graph.cleanup().toposort()
    graph.fold_constants()

    model_path = model_path.replace(".onnx", "_optimized.onnx")
    onnx.save(gs.export_onnx(graph), model_path)
    return model_path

if __name__ == "__main__":
    print("Usage: python3 ONNX_GS.py --model_path=/home/user/Downloads/model.onnx")
    
    parser = argparse.ArgumentParser(description='Optimize the ONNX model using GraphSurgeon')
    parser.add_argument('--model_path', type=str, default="/home/user/Downloads/model.onnx", required=False, help='Path to the ONNX model file (.onnx)')
    args = parser.parse_args()
    
    optimize_onnx(args.model_path)

# random chatgpt:
import onnx
import onnx_graphsurgeon as gs
import numpy as np

# Load the ONNX model
onnx_model_path = "yolo_backbone.onnx"
model = onnx.load(onnx_model_path)

# Parse the model graph into GraphSurgeon graph
graph = gs.import_onnx(model)

# Display the graph nodes (optional, useful for inspection)
print("Graph nodes before optimization:")
for node in graph.nodes:
    print(node)

# Example: Remove Identity nodes (they are not needed for inference)
graph.cleanup()

# Example: Fold constant nodes
# Constant folding can be used to simplify the graph by evaluating constant expressions at graph build time.
for node in graph.nodes:
    if node.op == "Add":
        inputs_are_constants = all(isinstance(inp, gs.Constant) for inp in node.inputs)
        if inputs_are_constants:
            value = node.inputs[0].values + node.inputs[1].values
            constant_node = gs.Constant(name=node.name, values=value)
            graph.outputs = [constant_node]

# Example: Fuse certain nodes, if applicable
# In this case, you can fuse common patterns (like batch normalization, activation layers) if it's supported.
# This is model-dependent, so it's an optional step. For simplicity, we omit specific fusion here.

# Cleanup the graph to remove any orphaned nodes after the transformations
graph.cleanup()
graph.toposort()

# Export the optimized ONNX model
optimized_onnx_path = "yolo_backbone_optimized.onnx"
onnx.save(gs.export_onnx(graph), optimized_onnx_path)

print(f"Optimized model saved at {optimized_onnx_path}")
