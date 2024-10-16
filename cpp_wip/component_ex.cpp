#include "rclcpp/rclcpp.hpp"

class MyNode : public rclcpp::Node {
public:
    MyNode(const rclcpp::NodeOptions & options)
        : Node("my_node", options) {
        RCLCPP_INFO(this->get_logger(), "MyNode started.");
    }
};

#include "rclcpp_components/register_node_macro.hpp"
RCLCPP_COMPONENTS_REGISTER_NODE(MyNode)

// This example defines a simple component node that can be dynamically loaded into a container.