#include <rclcpp/rclcpp.hpp>
#include <std_msgs/msg/int8.hpp>
#include <stdexcept>
#include <string>
#include <sstream>
#include <iostream>
#include <thread>
#include <memory>
#include <serial/serial.h>  // Include a serial library like https://github.com/wjwwood/serial

// Serial Communication:
// Uses the serial library for communication. You may need to install it if not already available.
// Ensures that the serial port is properly initialized and handles exceptions gracefully.

// Subscriptions:
// Subscribes to left_extermination_output and right_extermination_output topics to receive commands.

// Message Serialization:
// Converts the received integer data into a string message suitable for the Arduino and appends a newline for proper parsing on the Arduino side.

// Error Handling:
// Ensures errors in serial communication or parameter retrieval are logged and handled.

// Threaded Execution:
// Uses MultiThreadedExecutor to allow callbacks to run in separate threads, improving responsiveness.

class ProxyNode : public rclcpp::Node
{
public:
    ProxyNode()
        : Node("proxy_node")
    {
        // Declare and get the USB port parameter
        this->declare_parameter<std::string>("usb_port", "/dev/ttyACM0");
        port_ = this->get_parameter("usb_port").as_string();

        // Try to connect to the serial port
        try
        {
            serial_port_ = std::make_unique<serial::Serial>(port_, 115200, serial::Timeout::simpleTimeout(1000));
            RCLCPP_INFO(this->get_logger(), "Connected to serial port: %s", port_.c_str());
        }
        catch (const std::exception &e)
        {
            RCLCPP_ERROR(this->get_logger(), "Failed to connect to serial port: %s", e.what());
            throw;
        }

        // Subscriptions
        left_subscription_ = this->create_subscription<std_msgs::msg::Int8>(
            "left_extermination_output", 10,
            std::bind(&ProxyNode::left_callback, this, std::placeholders::_1));
        right_subscription_ = this->create_subscription<std_msgs::msg::Int8>(
            "right_extermination_output", 10,
            std::bind(&ProxyNode::right_callback, this, std::placeholders::_1));
    }

private:
    void left_callback(const std_msgs::msg::Int8::SharedPtr msg)
    {
        send_serial_command(msg->data == 0 ? 0 : 1, "left");
    }

    void right_callback(const std_msgs::msg::Int8::SharedPtr msg)
    {
        send_serial_command(msg->data == 0 ? 2 : 3, "right");
    }

    void send_serial_command(int command, const std::string &side)
    {
        if (!serial_port_ || !serial_port_->isOpen())
        {
            RCLCPP_ERROR(this->get_logger(), "Serial port not open!");
            return;
        }

        // Serialize and send the message
        std::string serialized_msg = std::to_string(command) + "\n";
        try
        {
            serial_port_->write(serialized_msg);
            RCLCPP_INFO(this->get_logger(), "Sent to Arduino: %d (%s %s)", command,
                        side.c_str(), command % 2 == 0 ? "off" : "on");
        }
        catch (const std::exception &e)
        {
            RCLCPP_ERROR(this->get_logger(), "Failed to write to serial port: %s", e.what());
        }
    }

    // Members
    std::string port_;
    std::unique_ptr<serial::Serial> serial_port_;

    // ROS2 subscriptions
    rclcpp::Subscription<std_msgs::msg::Int8>::SharedPtr left_subscription_;
    rclcpp::Subscription<std_msgs::msg::Int8>::SharedPtr right_subscription_;
};

int main(int argc, char **argv)
{
    rclcpp::init(argc, argv);

    auto node = std::make_shared<ProxyNode>();
    rclcpp::executors::MultiThreadedExecutor executor;
    executor.add_node(node);

    try
    {
        executor.spin();
    }
    catch (const std::exception &e)
    {
        RCLCPP_ERROR(node->get_logger(), "Exception caught: %s", e.what());
    }
    catch (...)
    {
        RCLCPP_ERROR(node->get_logger(), "Unknown exception caught");
    }

    rclcpp::shutdown();
    return 0;
}
