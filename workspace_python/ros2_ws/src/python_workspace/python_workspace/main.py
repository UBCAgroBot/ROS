import rclpy
from rclpy.executors import MultiThreadedExecutor
from rclpy.components import ComponentManager
from camera_node import generate_node as generate_camera_node
from jetson_node import generate_node as generate_jetson_node

def main(args=None):
    rclpy.init(args=args)

    # Create a component manager (container)
    component_manager = ComponentManager()
    
    # Load the composable nodes into the component manager
    component_manager.add_node(generate_camera_node())
    component_manager.add_node(generate_jetson_node())

    executor = MultiThreadedExecutor()
    executor.add_node(component_manager)

    try:
        executor.spin()
    finally:
        executor.shutdown()
        component_manager.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
