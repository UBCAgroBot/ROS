import time
import rclpy
from rclpy.node import Node
from std_msgs.msg import String
from custom_interfaces.msg import InferenceOutput


class MainControlNode(Node):
    def __init__(self):
        super().__init__('main_control_node')
        self.get_logger().info("MainControlNode initializing...")

        # Bbox subscriber
        self.inference_sub = self.create_subscription(
            InferenceOutput, '/inference_output', self.inference_callback, 10
        ) 
        self.get_logger().info("Subscribed to /inference_output.")

        # Robot state publisher
        self.robot_state_pub = self.create_publisher(String, '/robot_state', 10)
        self.get_logger().info("Publisher created for /robot_state.")
        self.waiting_for_static = False
        self.get_logger().info("MainControlNode initialized and waiting for inference results.")

    def inference_callback(self, msg):
        num_weeds = msg.num_boxes
        self.get_logger().info(f"Received {num_weeds} weed(s) from inference node.")
        if num_weeds > 0 and not self.waiting_for_static:
            self.get_logger().info("Weed detected, requesting stop.")
            # TODO: request service to navigation control and wait until confirm stop
            time.sleep(3) # place holder
            self.publish_robot_state('stopped')
            self.waiting_for_static = True
            self.get_logger().info("Robot state set to 'stopped', waiting for static image.")
        
        # After receiving a static image and bbox
        elif self.waiting_for_static and num_weeds > 0:
            self.get_logger().info("Processing static image for weed elimination (send to post node/arduino).")
            # TODO: send to post-processing node/arduino
            time.sleep(3) # place holder
            
            # After done, reset
            self.get_logger().info("Eliminated, resuming navigation.")
            self.waiting_for_static = False
            # TODO: request service to navigation control and wait until confirm started moving again
            time.sleep(3) # place holder
            self.publish_robot_state('moving')
            self.get_logger().info("Robot state set to 'moving', system ready for next detection.")

    def publish_robot_state(self, state):
        msg = String()
        msg.data = state
        self.robot_state_pub.publish(msg)
        self.get_logger().info(f"Published /robot_state: {state}")

def main(args=None):
    rclpy.init(args=args)
    node = MainControlNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        print(f"Keyboard Interrupt, shutting down...")
        
        
if __name__ == '__main__':
    main()


