#include <rclcpp/rclcpp.hpp>
#include <sensor_msgs/msg/image.hpp>
#include <std_msgs/msg/float32_multi_array.hpp>
#include <custom_interfaces/msg/image_input.hpp>
#include <custom_interfaces/msg/inference_output.hpp>
#include <cv_bridge/cv_bridge.h>
#include <opencv2/opencv.hpp>
#include <chrono>
#include "utils/model_inference.hpp"

// Custom ModelInference Class:

// Should support inference as a method accepting an input image and returning output image, confidences, and bounding boxes.
// Must handle loading models using the weights_path and precision.
// ROS 2 Message Handling:

// custom_interfaces::msg::ImageInput and custom_interfaces::msg::InferenceOutput must have fields matching the Python implementation.
// Conversion from Python's list (Float32MultiArray) is straightforward using std::vector.
// Timing:

// Uses std::chrono for precise inference timing.

class InferenceNode : public rclcpp::Node {
public:
    InferenceNode() : Node("inference_node") {
        RCLCPP_INFO(this->get_logger(), "Initializing Inference Node");

        // Declare and retrieve parameters
        this->declare_parameter<std::string>("weights_path", "/home/user/ROS/models/maize/Maize.onnx");
        this->declare_parameter<std::string>("precision", "fp32"); // Options: fp32, fp16
        this->declare_parameter<std::string>("camera_side", "left");

        weights_path_ = this->get_parameter("weights_path").as_string();
        precision_ = this->get_parameter("precision").as_string();
        camera_side_ = this->get_parameter("camera_side").as_string();

        model_ = std::make_shared<ModelInference>(weights_path_, precision_);
        bridge_ = std::make_shared<cv_bridge::CvBridge>();

        // Create subscription and publisher
        image_subscription_ = this->create_subscription<custom_interfaces::msg::ImageInput>(
            camera_side_ + "_image_input", 10,
            std::bind(&InferenceNode::image_callback, this, std::placeholders::_1));

        box_publisher_ = this->create_publisher<custom_interfaces::msg::InferenceOutput>(
            camera_side_ + "_inference_output", 10);
    }

private:
    void image_callback(const custom_interfaces::msg::ImageInput::SharedPtr msg) {
        RCLCPP_INFO(this->get_logger(), "Received Image");

        // Convert ROS image message to OpenCV format
        cv::Mat opencv_img = bridge_->imgmsg_to_cv2(msg->preprocessed_image, "passthrough");

        // Perform inference and measure time
        auto start = std::chrono::high_resolution_clock::now();
        cv::Mat output_img;
        std::vector<float> confidences;
        std::vector<float> boxes;
        model_->inference(opencv_img, output_img, confidences, boxes);
        auto end = std::chrono::high_resolution_clock::now();
        auto inference_time = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();

        // Prepare the output message
        custom_interfaces::msg::InferenceOutput output_msg;
        output_msg.num_boxes = boxes.size() / 4;  // Assuming each box has 4 values (x, y, width, height)
        output_msg.raw_image = msg->raw_image;
        output_msg.velocity = msg->velocity;
        output_msg.preprocessed_image = msg->preprocessed_image;

        // Fill bounding boxes and confidences
        std_msgs::msg::Float32MultiArray bounding_boxes;
        std_msgs::msg::Float32MultiArray confidences_msg;

        if (!boxes.empty()) {
            bounding_boxes.data = boxes;
            confidences_msg.data = confidences;
        }

        output_msg.bounding_boxes = bounding_boxes;
        output_msg.confidences = confidences_msg;

        // Publish the output message
        box_publisher_->publish(output_msg);

        RCLCPP_INFO(this->get_logger(), "Inference: %ld ms", inference_time);
    }

    // Parameters
    std::string weights_path_;
    std::string precision_;
    std::string camera_side_;

    // Model and utilities
    std::shared_ptr<ModelInference> model_;
    std::shared_ptr<cv_bridge::CvBridge> bridge_;

    // ROS 2 communication
    rclcpp::Subscription<custom_interfaces::msg::ImageInput>::SharedPtr image_subscription_;
    rclcpp::Publisher<custom_interfaces::msg::InferenceOutput>::SharedPtr box_publisher_;
};

int main(int argc, char **argv) {
    rclcpp::init(argc, argv);
    auto node = std::make_shared<InferenceNode>();

    rclcpp::executors::MultiThreadedExecutor executor;
    executor.add_node(node);

    try {
        executor.spin();
    } catch (const std::exception &e) {
        RCLCPP_ERROR(node->get_logger(), "Shutting down Inference Node: %s", e.what());
    }

    rclcpp::shutdown();
    return 0;
}
