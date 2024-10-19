#include "rclcpp/rclcpp.hpp"
#include "rclcpp_components/component_manager.hpp"

int main(int argc, char ** argv) {
    rclcpp::init(argc, argv);
    auto options = rclcpp::NodeOptions();
    auto component_manager = std::make_shared<rclcpp_components::ComponentManager>(options);
    rclcpp::spin(component_manager);
    rclcpp::shutdown();
    return 0;
}
