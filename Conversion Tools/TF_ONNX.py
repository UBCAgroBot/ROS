import sys
import tensorflow as tf
import tf2onnx
# from tensorflow.python.tools import optimize_for_inference_lib
import onnx
# import onnxruntime as ort
import numpy as np
import pycuda.driver as cuda

OPSET_VERS = 13

# tf_predictions = tf_model.predict(x_test)
# results_tf_ort = mct.predictOnnx(x_test, session=onnx_sess)
# mct.checkPredictionConsistency(tf_predictions, results_tf_ort)
# mct.checkConfidenceConsistency(tf_predictions, results_tf_ort)
# given an array of test inputs and a path to onnx model or a session returns the predictions
def predictOnnx(x_test,session=None,dest_path=""):
    if session is None and dest_path == "":
        raise ValueError("No model or path provided, please specifiy one of them.")
    if session is None:
        session = loadOnnxModel(dest_path)

    results_ort = session.run([out.name for out in session.get_outputs()], {session.get_inputs()[0].name: x_test})
    return np.array(results_ort[0])

# given the predictions from the original model and the converted model, check if they are consistent
# shape of predictions_original and converted_results should be the same
# only checks for the predicted class (aka the argmax)
# takes in two 2D arrays: first dimension is the number of samples,  second dimension is the number of classes and values correspond to confidence
def checkPredictionConsistency(predictions_original, converted_results):
    for n in range(predictions_original.shape[0]):
        if np.argmax(predictions_original[n]) != np.argmax(converted_results[n]):
            print(f"Original: {np.argmax(predictions_original[n])}, ONNX: {np.argmax(converted_results[n])}")
            print(f"{predictions_original[n]}, \n{converted_results[n]}")
            print("=====================================")
            raise ValueError("Predictions are not consistent")

    print("All predictions are consistent")

# given the predictions from the original model and the converted model, check if they are consistent
# shape of predictions_original and converted_results should be the same
# only checks for the difference in confidence
# takes in two 2D arrays: first dimension is the number of samples,  second dimension is the number of classes and values correspond to confidence
# tolerance: the maximum difference in confidence that is allowed
def checkConfidenceConsistency(predictions_original, converted_results, tolerance=1e-5):
    np.testing.assert_allclose(predictions_original, converted_results,atol=tolerance)
    # for n in range(predictions_original.shape[0]):
    #     if not np.allclose(predictions_original[n], converted_results[n], atol=tolerance):
    #         print(f"Original: \t {predictions_original[n]}, \nONNX: \t{converted_results[n]}")
    #         print("=====================================")
    #         return

    print("All confidence percentages are consistent")

def convert_tf_to_onnx(model_path="./model.pb", output_path="./model.onnx", input_names=["input"], output_names=["output"], constant_folding=False):
    print("Loading the TensorFlow model")
    with tf.io.gfile.GFile(model_path, "rb") as f:
        frozen_graph_def = tf.compat.v1.GraphDef()
        frozen_graph_def.ParseFromString(f.read())
    
    print("Exporting the model to ONNX format")
    onnx_model, _ = tf2onnx.convert.from_graph_def(
        frozen_graph_def,  # The frozen graph definition
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
    if len(sys.argv) != 2:
        print("Usage: python convert_to_trt.py <model_path>")
        sys.exit(1)

    model_path = sys.argv[1]
    convert_tf_to_onnx(model_path)

# improvements:
# Model Pruning: Pruning is a technique in deep learning where you remove the weights of certain neurons which are less important. This can help in reducing the size of the model and hence improve the performance during conversion.
# Quantization: Quantization is a process that reduces the numerical precision of the model's weights, which can lead to a significant reduction in both the memory requirement and computational cost of the model.
# Use the Latest ONNX Opset Version: The ONNX opset version corresponds to the set of operators and their versions supported. Newer opset versions can have optimizations that were not available in previous versions. You can set the opset version with the opset parameter in tf2onnx.convert.from_graph_def.