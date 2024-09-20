# lifecycle_node.py
import rclpy
from rclpy.lifecycle import LifecycleNode
from rclpy.lifecycle import State
from rclpy.lifecycle import TransitionCallbackReturn
from rclpy.executors import MultiThreadedExecutor

class LifecycleManagerNode(LifecycleNode):
    def __init__(self):
        super().__init__('lifecycle_manager_node')
        
        # Declare parameters for the lifecycle node
        self.declare_parameter('lifecycle_action', 'activate')
        self.get_logger().info('LifecycleManagerNode has been created.')

    def on_configure(self, state: State) -> TransitionCallbackReturn:
        self.get_logger().info('on_configure() called')
        return TransitionCallbackReturn.SUCCESS

    def on_activate(self, state: State) -> TransitionCallbackReturn:
        self.get_logger().info('on_activate() called')
        return TransitionCallbackReturn.SUCCESS

    def on_deactivate(self, state: State) -> TransitionCallbackReturn:
        self.get_logger().info('on_deactivate() called')
        return TransitionCallbackReturn.SUCCESS

    def on_cleanup(self, state: State) -> TransitionCallbackReturn:
        self.get_logger().info('on_cleanup() called')
        return TransitionCallbackReturn.SUCCESS

    def on_shutdown(self, state: State) -> TransitionCallbackReturn:
        self.get_logger().info('on_shutdown() called')
        return TransitionCallbackReturn.SUCCESS

def main(args=None):
    rclpy.init(args=args)
    
    lifecycle_node = LifecycleManagerNode()

    executor = MultiThreadedExecutor()
    executor.add_node(lifecycle_node)

    try:
        rclpy.spin(lifecycle_node, executor=executor)
    except KeyboardInterrupt:
        pass

    lifecycle_node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()

import rclpy
from rclpy.lifecycle import Node as LifecycleNode
from rclpy.lifecycle import State
from rclpy.lifecycle import TransitionCallbackReturn
import pycuda.driver as cuda
import pycuda.autoinit
import tensorrt as trt
import numpy as np
import time

# TensorRT logger
TRT_LOGGER = trt.Logger(trt.Logger.WARNING)

class TRTLifecycleNode(LifecycleNode):
    def __init__(self):
        super().__init__('trt_lifecycle_node')
        self.engine = None
        self.context = None
        self.inputs = None
        self.outputs = None
        self.bindings = None
        self.stream = None

    def on_configure(self, state: State) -> TransitionCallbackReturn:
        """Load and deserialize TensorRT engine and allocate buffers."""
        self.get_logger().info("Configuring... Loading TensorRT engine.")

        try:
            # Load TensorRT engine
            engine_path = '/path/to/your/model.trt'
            self.engine = self.load_trt_engine(engine_path)
            self.context = self.engine.create_execution_context()

            # Allocate buffers
            self.inputs, self.outputs, self.bindings, self.stream = self.allocate_buffers()

            self.get_logger().info("TensorRT engine loaded and buffers allocated.")
            return TransitionCallbackReturn.SUCCESS

        except Exception as e:
            self.get_logger().error(f"Failed to load TensorRT engine: {e}")
            return TransitionCallbackReturn.FAILURE

    def on_activate(self, state: State) -> TransitionCallbackReturn:
        """Start inference process or any task in this state."""
        self.get_logger().info("Activating... ready for inference.")
        return TransitionCallbackReturn.SUCCESS

    def on_deactivate(self, state: State) -> TransitionCallbackReturn:
        """Handle node deactivation (optional)."""
        self.get_logger().info("Deactivating...")
        return TransitionCallbackReturn.SUCCESS

    def on_cleanup(self, state: State) -> TransitionCallbackReturn:
        """Clean up and release resources."""
        self.get_logger().info("Cleaning up... releasing resources.")
        self.context = None
        self.engine = None
        self.inputs = None
        self.outputs = None
        self.bindings = None
        self.stream = None
        return TransitionCallbackReturn.SUCCESS

    def on_shutdown(self, state: State) -> TransitionCallbackReturn:
        """Shutdown node gracefully."""
        self.get_logger().info("Shutting down...")
        return TransitionCallbackReturn.SUCCESS

    def load_trt_engine(self, engine_path):
        """Load and deserialize the TensorRT engine."""
        with open(engine_path, 'rb') as f, trt.Runtime(TRT_LOGGER) as runtime:
            engine_data = f.read()
            return runtime.deserialize_cuda_engine(engine_data)

    def allocate_buffers(self):
        """Allocate input/output buffers for TensorRT engine."""
        inputs = []
        outputs = []
        bindings = []
        stream = cuda.Stream()

        for binding in self.engine:
            binding_shape = self.engine.get_binding_shape(binding)
            size = trt.volume(binding_shape) * self.engine.max_batch_size
            dtype = trt.nptype(self.engine.get_binding_dtype(binding))

            # Allocate host and device memory for inputs/outputs
            host_mem = cuda.pagelocked_empty(size, dtype)
            device_mem = cuda.mem_alloc(host_mem.nbytes)

            bindings.append(int(device_mem))
            if self.engine.binding_is_input(binding):
                inputs.append({'host': host_mem, 'device': device_mem})
            else:
                outputs.append({'host': host_mem, 'device': device_mem})

        return inputs, outputs, bindings, stream

    def run_inference(self, input_data):
        """Run inference using the loaded TensorRT engine."""
        np.copyto(self.inputs[0]['host'], input_data.ravel())

        # Transfer input to device
        cuda.memcpy_htod_async(self.inputs[0]['device'], self.inputs[0]['host'], self.stream)

        # Run inference
        self.context.execute_async_v2(bindings=self.bindings, stream_handle=self.stream.handle)

        # Transfer predictions back from device
        cuda.memcpy_dtoh_async(self.outputs[0]['host'], self.outputs[0]['device'], self.stream)

        # Synchronize stream
        self.stream.synchronize()

        return self.outputs[0]['host']


def main(args=None):
    rclpy.init(args=args)
    node = TRTLifecycleNode()

    # Spin the node until shutdown
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
