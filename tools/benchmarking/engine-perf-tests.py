# combines calling the engine inference utility with system_metrics
# along with pre-processing and post-processing utilities
# should plot results cleanly, able to export summary, used in github?
# can also toggle unit tests functionality
# Update paths for the .trt model, test images, and ground truth file.
# Ensure the bounding box coordinates are converted between the format of the ground truth and model output if necessary.
# Expand with more metrics such as precision, recall, or F1 score based on IoU thresholds if relevant.

import os
# need imports for inference module
from utils import pre-process, postprocess

def verify_path(trt_engine_path=None):
    pass

# tests a TensorRT engine file by performing inference and checking outputs
def test_trt_engine(trt_engine_path='model_trt.trt', input_shape=(1,3,224,224), input_data=None, expected_output=None):
    engine = load_engine()
    inputs, outputs, bindings, stream = allocate_buffers(engine)


    # Example input
    input_shape = (1, 3, 224, 224)
    input_data = torch.randn(input_shape).cuda()
    
    # Generate random input data if not provided (should be cp)
    if input_data is None:
        input_data = np.random.rand(*input_shape).astype(np.float32)

    # Perform inference using the TensorRT engine
    output = infer(engine, inputs, outputs, bindings, stream, input_data)

    # Print the inference result
    print("Inference output:", output)

    # Compare with expected output if provided
    if expected_output is not None:
        if np.allclose(output, expected_output, rtol=1e-3, atol=1e-3):
            print("The inference result matches the expected output.")
            return True
        else:
            print("The inference result does not match the expected output.")
            return False
    else:
        print("No expected output provided. Unable to verify accuracy.")
        return True  # Pass as long as inference ran without errors