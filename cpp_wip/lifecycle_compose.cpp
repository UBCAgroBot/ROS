#include "rclcpp/rclcpp.hpp"
#include "rclcpp_lifecycle/lifecycle_node.hpp"

class MyLifecycleNode : public rclcpp_lifecycle::LifecycleNode {
public:
    MyLifecycleNode(const rclcpp::NodeOptions & options)
        : rclcpp_lifecycle::LifecycleNode("my_lifecycle_node", options) {}

    rclcpp_lifecycle::node_interfaces::LifecycleNodeInterface::CallbackReturn
    on_configure(const rclcpp_lifecycle::State &) {
        RCLCPP_INFO(this->get_logger(), "Configuring...");
        // Initialization code
        return rclcpp_lifecycle::node_interfaces::LifecycleNodeInterface::CallbackReturn::SUCCESS;
    }

    rclcpp_lifecycle::node_interfaces::LifecycleNodeInterface::CallbackReturn
    on_activate(const rclcpp_lifecycle::State &) {
        RCLCPP_INFO(this->get_logger(), "Activating...");
        // Activation code
        return rclcpp_lifecycle::node_interfaces::LifecycleNodeInterface::CallbackReturn::SUCCESS;
    }

    rclcpp_lifecycle::node_interfaces::LifecycleNodeInterface::CallbackReturn
    on_deactivate(const rclcpp_lifecycle::State &) {
        RCLCPP_INFO(this->get_logger(), "Deactivating...");
        // Deactivation code
        return rclcpp_lifecycle::node_interfaces::LifecycleNodeInterface::CallbackReturn::SUCCESS;
    }

    rclcpp_lifecycle::node_interfaces::LifecycleNodeInterface::CallbackReturn
    on_cleanup(const rclcpp_lifecycle::State &) {
        RCLCPP_INFO(this->get_logger(), "Cleaning up...");
        // Cleanup code
        return rclcpp_lifecycle::node_interfaces::LifecycleNodeInterface::CallbackReturn::SUCCESS;
    }

    rclcpp_lifecycle::node_interfaces::LifecycleNodeInterface::CallbackReturn
    on_shutdown(const rclcpp_lifecycle::State &) {
        RCLCPP_INFO(this->get_logger(), "Shutting down...");
        // Shutdown code
        return rclcpp_lifecycle::node_interfaces::LifecycleNodeInterface::CallbackReturn::SUCCESS;
    }
};

#include "rclcpp_components/register_node_macro.hpp"
RCLCPP_COMPONENTS_REGISTER_NODE(MyLifecycleNode)

// This code defines a lifecycle node that follows the component pattern, making it compatible with ROS2 composition.

// Load the Lifecycle Node Dynamically: You can load the lifecycle node into a container just like any other composable node. Use the same launch file as before, but specify the lifecycle node's package and plugin.

// Manage the Node’s Lifecycle: You can manage the node's state transitions using ROS2 lifecycle commands:

// ros2 lifecycle set /my_lifecycle_node configure
// ros2 lifecycle set /my_lifecycle_node activate
// ros2 lifecycle set /my_lifecycle_node deactivate
// ros2 lifecycle set /my_lifecycle_node cleanup
// ros2 lifecycle set /my_lifecycle_node shutdown