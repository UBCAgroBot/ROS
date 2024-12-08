#include <rclcpp/rclcpp.hpp>
#include <sensor_msgs/msg/image.hpp>
#include <std_msgs/msg/int8.hpp>
#include <custom_interfaces/msg/inference_output.hpp>
#include <cv_bridge/cv_bridge.h>
#include <opencv2/opencv.hpp>
#include "utils/model_inference.hpp"

// C++ Equivalent of cv_bridge:
//      Used CvBridge to convert ROS Image messages to OpenCV Mat.
// Custom Utility Integration:
//      ModelInference class should provide postprocess and draw_boxes functions similar to Python.
// Multi-threaded Executor:
//      Ensures responsiveness by managing callbacks efficiently.

class ExterminationNode : public rclcpp::Node {
public:
    ExterminationNode() : Node("extermination_node") {
        RCLCPP_INFO(this->get_logger(), "Initializing Extermination Node");

        // Declare and retrieve parameters
        this->declare_parameter<bool>("use_display_node", true);
        this->declare_parameter<std::string>("camera_side", "left");
        use_display_node_ = this->get_parameter("use_display_node").as_bool();
        camera_side_ = this->get_parameter("camera_side").as_string();

        window_ = (camera_side_ == "left") ? "Left Camera" : "Right Camera";
        if (!use_display_node_) {
            window_ = "";
        }

        publishing_rate_ = 1.0;
        lower_range_ = cv::Scalar(78, 158, 124);
        upper_range_ = cv::Scalar(60, 255, 255);
        minimum_area_ = 100;
        minimum_confidence_ = 0.5;
        boxes_present_ = 0;

        model_ = std::make_shared<ModelInference>();
        bridge_ = std::make_shared<cv_bridge::CvBridge>();

        boxes_msg_.data = 0;

        // Create subscription and publisher
        inference_subscription_ = this->create_subscription<custom_interfaces::msg::InferenceOutput>(
            camera_side_ + "_inference_output", 10, 
            std::bind(&ExterminationNode::inference_callback, this, std::placeholders::_1));

        box_publisher_ = this->create_publisher<std_msgs::msg::Int8>(
            camera_side_ + "_extermination_output", 10);

        timer_ = this->create_wall_timer(
            std::chrono::milliseconds(static_cast<int>(1000 / publishing_rate_)),
            std::bind(&ExterminationNode::timer_callback, this));
    }

private:
    void inference_callback(const custom_interfaces::msg::InferenceOutput::SharedPtr msg) {
        RCLCPP_INFO(this->get_logger(), "Received Bounding Boxes");

        // Convert ROS image messages to OpenCV
        cv::Mat preprocessed_image = bridge_->imgmsg_to_cv2(msg->preprocessed_image, "passthrough");
        cv::Mat raw_image = bridge_->imgmsg_to_cv2(msg->raw_image, "passthrough");

        // Perform post-processing and draw bounding boxes
        auto bounding_boxes = model_->postprocess(
            msg->confidences, msg->bounding_boxes, raw_image, msg->velocity);

        cv::Mat final_image = model_->draw_boxes(raw_image, bounding_boxes, msg->velocity);

        // Display the image if required
        if (use_display_node_) {
            cv::imshow(window_, final_image);
            cv::waitKey(10);
        }

        // Update presence of bounding boxes
        boxes_present_ = !bounding_boxes.empty() ? 1 : 0;
        boxes_msg_.data = boxes_present_;
    }

    void timer_callback() {
        box_publisher_->publish(boxes_msg_);
        RCLCPP_INFO(this->get_logger(), "Published results to Proxy Node");
    }

    // Parameters
    bool use_display_node_;
    std::string camera_side_;
    std::string window_;

    // Model and preprocessing
    double publishing_rate_;
    cv::Scalar lower_range_;
    cv::Scalar upper_range_;
    int minimum_area_;
    double minimum_confidence_;
    int boxes_present_;

    std_msgs::msg::Int8 boxes_msg_;

    // Utilities
    std::shared_ptr<ModelInference> model_;
    std::shared_ptr<cv_bridge::CvBridge> bridge_;

    // ROS 2 communication
    rclcpp::Subscription<custom_interfaces::msg::InferenceOutput>::SharedPtr inference_subscription_;
    rclcpp::Publisher<std_msgs::msg::Int8>::SharedPtr box_publisher_;
    rclcpp::TimerBase::SharedPtr timer_;
};

int main(int argc, char **argv) {
    rclcpp::init(argc, argv);
    auto node = std::make_shared<ExterminationNode>();

    rclcpp::executors::MultiThreadedExecutor executor;
    executor.add_node(node);

    try {
        executor.spin();
    } catch (const std::exception &e) {
        RCLCPP_ERROR(node->get_logger(), "Shutting down Extermination Node: %s", e.what());
    }

    rclcpp::shutdown();
    return 0;
}
