import onnx
import onnx.helper
import os
import onnx_graphsurgeon as gs

if os.path.exists("Maize_nodes.txt"):
    os.remove("Maize_nodes.txt")

# Load the ONNX model
model = onnx.load("Maize.onnx")

# Extract the graph from the model
graph = model.graph

# Save the nodes in the graph
nodes = []
for node in graph.node:
    nodes.append(f"Node Type: {node.op_type}, Inputs: {node.input}, Outputs: {node.output}")

# Save the nodes to a text file
with open("Maize_nodes.txt", "w") as file:
    for node in nodes:
        file.write(node + "\n")


