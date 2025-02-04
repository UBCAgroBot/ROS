import onnx
import onnx.helper
import os
import onnx_graphsurgeon as gs
import argparse

def optimize_onnx(model_path="./Maize.onnx"):
    print("Optimizing ONNX model")
    
    # Load the ONNX model
    model = onnx.load(model_path)
    graph = gs.import_onnx(model)
    
    print("Graph nodes before optimization:")
    for node in graph.nodes:
        print(f"Index: {graph.nodes.index(node)}, Node Type: {node.op}, Node Name: {node.name}")
    
    # Skip nodes from index 1 to 100 while optimizing others
    for idx, node in enumerate(graph.nodes):
        if 1 <= idx <= 100:
            continue  # Skip nodes in this range
        
        # Example optimization: Fuse arithmetic operations (Add with Mul)
        if node.op == "Add":
            # Check if the input to this "Add" is a multiplication operation (or any other condition)
            # We need to check the node that produces the input tensors, not the tensors themselves
            input_node_0 = node.inputs[0].inputs[0] if isinstance(node.inputs[0], gs.Variable) and node.inputs[0].inputs else None
            
            if input_node_0 and input_node_0.op == "Mul":
                # Fuse the Add and Mul operations
                fused_node = gs.Node(op="Add", inputs=node.inputs[0].inputs, outputs=node.outputs)
                graph.nodes.append(fused_node)
                
                # Remove the original nodes that are now fused
                graph.nodes.remove(input_node_0)  # Remove the Mul node
                graph.nodes.remove(node)          # Remove the Add node
    
    # Clean up, fold constants, and toposort the graph
    graph.cleanup().toposort()
    graph.fold_constants()

    # Save the optimized model to a new file
    optimized_model_path = model_path.replace(".onnx", "_optimized.onnx")
    onnx.save(gs.export_onnx(graph), optimized_model_path)

    print(f"Optimized model saved to {optimized_model_path}")
    return optimized_model_path

if __name__ == "__main__":
    print("Usage: python3 ONNX_GS.py --model_path=./Maize.onnx")
    
    parser = argparse.ArgumentParser(description='Optimize the ONNX model using GraphSurgeon')
    parser.add_argument('--model_path', type=str, default="./Maize.onnx", required=False, help='Path to the ONNX model file (.onnx)')
    args = parser.parse_args()
    
    optimize_onnx(args.model_path)