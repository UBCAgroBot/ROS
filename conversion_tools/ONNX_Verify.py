import time
import onnx
import onnxruntime as ort
import numpy as np

# put random input shape into CUDA if using CUDA provider?
def verify_onnx(model_path, compared_outputs, model_dimensions, fp_16):
    print("Verifying the converted model")
    onnx_output, onnx_inference = predict_onnx(model_path, fp_16, model_dimensions)
    
    print("ONNX inference time:", onnx_inference, "ms")
    
    # Calculate MSE (Mean Squared Error)
    mse = np.mean((onnx_output - compared_outputs) ** 2)
    print("MSE between ONNX and TensorRT outputs:", mse)

    # Calculate MAE (Mean Absolute Error)
    mae = np.mean(np.abs(onnx_output - compared_outputs))
    print("MAE between ONNX and TensorRT outputs:", mae)
    return 

# any other chanes for fp_16 to work?
def predict_onnx(model_path, fp_16, input_shape):
    onnx_session = ort.InferenceSession(model_path,providers=["CUDAExecutionProvider"])
    
    if fp_16:
        random_input = np.random.randn(input_shape).astype(np.float16)
    else:
        random_input = np.random.randn(input_shape).astype(np.float32)
        
    input_name = onnx_session.get_inputs()[0].name
    tic = time.perf_counter_ns()
    # results_ort = session.run([out.name for out in session.get_outputs()], {session.get_inputs()[0].name: x_test})
    # results_ort = onnx_session.run([out.name for out in session.get_outputs()], {session.get_inputs()[0].name: model_test})
    onnx_output = onnx_session.run(None, {input_name: random_input})
    toc = time.perf_counter_ns()
    onnx_output = onnx_output[0]
    # onnx_output= np.array(onnx_output)
    return onnx_output, (toc - tic) / 1e6

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