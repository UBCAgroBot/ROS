import tensorflow as tf
import numpy as np
import tf2onnx
import onnx
import onnxruntime as ort
import torch


# assumes that the model only has a single input and a single output layer

OPSET_VERS = 13
# given a tensorflow model, convert it to onnx, save it to dest file and return an onnx inference session
# takes in a tensorflow model and a target path for the onnx model file
# onnx models are saved with a .onnx extension
def tf_to_onnx(model, dest_path): 
    input_signature = [tf.TensorSpec( model.input_shape, model.input.dtype, name="input")]
    # Use from_function for tf functions
    onnx_model, _ = tf2onnx.convert.from_keras(model, input_signature, opset=OPSET_VERS)
    onnx.save(onnx_model, dest_path)

    # Convert the model to a serialized string format
    onnx_model_str = onnx_model.SerializeToString()

    # Create an InferenceSession. This is the object that will run the model.
    return loadOnnxModel(onnx_model_str)


# given a torch model, convert it to onnx, save it to dest file and return an onnx inference session
def torch_to_onnx(model, example_input, dest_path):
    torch.onnx.export(model,               # model being run
                  example_input,                         # model input (or a tuple for multiple inputs)
                  dest_path,   # where to save the model (can be a file or file-like object)
                  export_params=True,        # store the trained parameter weights inside the model file
                  opset_version=OPSET_VERS,          # the ONNX version to export the model to
                #   do_constant_folding=True,  # whether to execute constant folding for optimization
                  dynamic_axes={'input' : {0 : 'batch_size'},    # variable length axes
                                'output' : {0 : 'batch_size'}})
    return loadOnnxModel(dest_path)

def loadOnnxModel(path, providers=["CUDAExecutionProvider"]):
    return ort.InferenceSession(path,providers=providers)
    

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

