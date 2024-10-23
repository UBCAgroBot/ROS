import tensorrt as trt
import sys
import argparse

"""
takes in onnx model
converts to tensorrt
tensorrt model input size must be src pth input size 
"""

def cli():
    desc = 'Compile Onnx model to TensorRT'
    parser = argparse.ArgumentParser(description=desc)
    parser.add_argument('-m', '--model', default='', help='onnx file location')
    parser.add_argument('-fp', '--floatingpoint', type=int, default=16, help='floating point precision. 16 or 32')
    parser.add_argument('-o', '--output', default='', help='name of trt output file')
    args = parser.parse_args()
    model = 'coco_yolov5.onnx'
    fp = args.floatingpoint
    if fp != 16 and fp != 32:
        print('floating point precision must be 16 or 32')
        sys.exit()
    output = 'coco_yolov5-{}.trt'.format(fp)
    return {
        'model': model,
        'fp': fp,
        'output': output
    }

if __name__ == '__main__':
    args = cli()
    batch_size = 4
    model = '{}'.format(args['model'])
    output = '{}'.format(args['output'])
    logger = trt.Logger(trt.Logger.WARNING)
    explicit_batch = 1 << (int)(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)  # trt7
    with trt.Builder(logger) as builder, builder.create_network(explicit_batch) as network, trt.OnnxParser(network, logger) as parser:
        builder.max_workspace_size = 1 << 28
        builder.max_batch_size = batch_size
        if args['fp'] == 16:
            builder.fp16_mode = True
        with open(model, 'rb') as f:
            print('Beginning ONNX file parsing')
            if not parser.parse(f.read()):
                for error in range(parser.num_errors):
                    print("ERROR", parser.get_error(error))
        print("num layers:", network.num_layers)
        network.get_input(0).shape = [batch_size, 3, 608, 608]  # trt7
        # last_layer = network.get_layer(network.num_layers - 1)
        # network.mark_output(last_layer.get_output(0))
        # reshape input from 32 to 1
        engine = builder.build_cuda_engine(network)
        with open(output, 'wb') as f:
            f.write(engine.serialize())
        print("Completed creating Engine")

import os
import argparse
import onnx
import tensorrt as trt

def convert_onnx_to_trt(model_path="/home/user/Downloads/model.onnx", output_path="/home/user/Downloads/model.engine", FP16_mode=True, strip_weights=False, gs_optimize=False, verbose=False, verify=True):
    if not os.path.exists(model_path):
        raise ValueError(f"Model file not found at {model_path}")
    if not output_path.endswith(".engine"):
        raise ValueError("Output path should end with .engine")
    
    if gs_optimize:
        from ONNX_GS import optimize_onnx
        model_path = optimize_onnx(model_path)
    
    if verbose:
        TRT_LOGGER = trt.Logger(trt.Logger.VERBOSE)
    else:
        TRT_LOGGER = trt.Logger(trt.Logger.WARNING)
        
    builder = trt.Builder(TRT_LOGGER)
    config = builder.create_builder_config()
    
    # cache = config.create_timing_cache(b"")
    # config.set_timing_cache(cache, ignore_mismatch=False)
    
    max_workspace = (1 << 33) # 8GB
    config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, max_workspace)
    
    network = builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH))
    parser = trt.OnnxParser(network, TRT_LOGGER)
    
    with open(model_path, 'rb') as model_file:
        print("Parsing ONNX model")
        if not parser.parse(model_file.read()):
            for error in range(parser.num_errors):
                print(parser.get_error(error))
    
    inputs = [network.get_input(i) for i in range(network.num_inputs)]
    outputs = [network.get_output(i) for i in range(network.num_outputs)]
    input_shape = inputs[0].shape
    
    for input in inputs:
        print(f"Model {input.name} shape: {input.shape} {input.dtype}")
    for output in outputs:
        print(f"Model {output.name} shape: {output.shape} {output.dtype}") 

    if FP16_mode:
        config.set_flag(trt.BuilderFlag.FP16)
    # elif INT8:
    #     config.set_flag(trt.BuilderFlag.INT8)
    # Enable FP16 optimization if the device supports it
    # if builder.platform_has_fast_fp16:
    #     builder.fp16_mode = True
    
    if strip_weights:
        config.set_flag(trt.BuilderFlag.STRIP_PLAN)

    print("Building TensorRT engine. This may take a few minutes.")
    engine_bytes = builder.build_serialized_network(network, config) 
    with open(output_path, "wb") as f:    
        f.write(engine_bytes)

    if verify:
        from TRT_Verify import verify_trt
        verify_trt(model_path, output_path, FP16_mode, input_shape)

    print("Engine built successfully")
    print(f"Converted TensorRT engine saved at {output_path}")    

if __name__ == "__main__":
    print("Usage: python3 ONNX_TRT.py --model_path=/home/user/Downloads/model.onnx --output_path=/home/user/Downloads/model.engine --FP16=False --strip_weights=False --gs_optimize=False --verbose=False --verify=True")
    
    parser = argparse.ArgumentParser(description='Convert Onnx model to TensorRT')
    parser.add_argument('--model_path', type=str, default="/home/user/Downloads/model.onnx", required=False, help='Path to the ONNX model file (.onnx)')
    parser.add_argument('--output_path', type=str, default="/home/user/Downloads/model.engine", required=False, help='Path to save the converted TensorRT model file (.engine)')
    parser.add_argument('--FP16', type=bool, default=False, help="FP16 precision mode")
    # parser.add_argument('--INT8', type=bool, default=False, help="INT8 precision mode")
    parser.add_argument('--strip_weights', type=bool, default=False, help="Strip unnecessary weights")
    parser.add_argument('--gs_optimize', type=bool, default=False, help='Use ONNX GraphSurgeon to optimize model first')
    parser.add_argument('--verbose', type=bool, default=False, help="Verbose TensorRT logging")
    parser.add_argument('--verify', type=bool, default=True, help="Verify the converted engine output")
    args = parser.parse_args()
    
    convert_onnx_to_trt(args.model_path, args.output_path, args.FP16, args.strip_weights, args.gs_optimize, args.verbose)