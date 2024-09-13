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