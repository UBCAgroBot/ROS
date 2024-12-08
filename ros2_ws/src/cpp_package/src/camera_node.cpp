#include <chrono>
#include <opencv2/opencv.hpp>
#include <sl/Camera.hpp>

#include "rclcpp/rclcpp.hpp"
#include "std_msgs/msg/header.hpp"
#include "sensor_msgs/msg/image.hpp"
#include "custom_interfaces/msg/image_input.hpp"
#include "cv_bridge/cv_bridge.h"

using namespace std::chrono_literals;


// ZED SDK Integration:
//      Uses sl::Mat for accessing ZED images.
//      Handles different views based on camera_side.
// ROS 2 Standard Practices:
//      Uses rclcpp::Node for ROS integration.
//      Publishes to a custom ROS message type.
// OpenCV Conversion:
//      Converts ZED's sl::Mat to OpenCV's cv::Mat for processing.
// Performance Logging:
//      Measures and logs preprocessing time.

class CameraNode : public rclcpp::Node {
public:
    CameraNode() : Node("camera_node") {
        RCLCPP_INFO(this->get_logger(), "Initializing Camera Node");

        // Declare and get parameters
        this->declare_parameter<std::string>("camera_side", "left");
        this->declare_parameter<std::vector<int>>("roi_dimensions", {0, 0, 100, 100});
        this->declare_parameter<int>("frame_rate", 30);

        camera_side_ = this->get_parameter("camera_side").as_string();
        roi_dimensions_ = this->get_parameter("roi_dimensions").as_integer_array();
        frame_rate_ = this->get_parameter("frame_rate").as_int();

        model_dimensions_ = cv::Size(640, 640);
        velocity_ = 0.0;
        shift_constant_ = (camera_side_ == "left") ? 0 : -1;

        // Publishers
        image_publisher_ = this->create_publisher<custom_interfaces::msg::ImageInput>(
            camera_side_ + std::string("_image_input"), 10);

        // Start ZED camera processing
        start_camera();
    }

private:
    void start_camera() {
        sl::Camera zed;
        sl::InitParameters init_params;
        init_params.camera_resolution = sl::RESOLUTION::HD1080;
        init_params.camera_fps = frame_rate_;

        if (camera_side_ == "left") {
            init_params.set_from_serial_number(left_camera_serial_number_);
        } else if (camera_side_ == "right") {
            init_params.set_from_serial_number(right_camera_serial_number_);
        }

        auto status = zed.open(init_params);
        if (status != sl::ERROR_CODE::SUCCESS) {
            RCLCPP_ERROR(this->get_logger(), "Failed to open ZED camera: %s", sl::toString(status).c_str());
            return;
        }

        sl::RuntimeParameters runtime_params;
        sl::Mat image_zed;

        while (rclcpp::ok()) {
            if (zed.grab(runtime_params) == sl::ERROR_CODE::SUCCESS) {
                zed.retrieve_image(image_zed, (camera_side_ == "left") ? sl::VIEW::LEFT_UNRECTIFIED : sl::VIEW::RIGHT_UNRECTIFIED);
                cv::Mat cv_image = slMatToCvMat(image_zed);
                preprocess_and_publish(cv_image);
                std::this_thread::sleep_for(std::chrono::milliseconds(1000 / frame_rate_));
            } else {
                RCLCPP_ERROR(this->get_logger(), "Failed to grab ZED frame");
            }
        }

        zed.close();
        RCLCPP_INFO(this->get_logger(), "Closing ZED camera");
    }

    void preprocess_and_publish(const cv::Mat &image) {
        auto start_time = std::chrono::high_resolution_clock::now();

        // Extract ROI
        int x1 = roi_dimensions_[0] + shift_constant_;
        int y1 = roi_dimensions_[1];
        int x2 = roi_dimensions_[2] + shift_constant_;
        int y2 = roi_dimensions_[3];
        cv::Mat roi_image = image(cv::Rect(cv::Point(x1, y1), cv::Point(x2, y2)));

        // Resize and convert to ROS message
        cv::Mat preprocessed_image;
        cv::resize(roi_image, preprocessed_image, model_dimensions_);

        auto raw_image_msg = cv_bridge::CvImage(std_msgs::msg::Header(), "bgr8", image).toImageMsg();
        auto preprocessed_image_msg = cv_bridge::CvImage(std_msgs::msg::Header(), "bgr8", preprocessed_image).toImageMsg();

        // Publish custom message
        custom_interfaces::msg::ImageInput image_input;
        image_input.header.frame_id = "camera_frame";
        image_input.raw_image = *raw_image_msg;
        image_input.preprocessed_image = *preprocessed_image_msg;
        image_input.velocity = velocity_;

        image_publisher_->publish(image_input);

        auto end_time = std::chrono::high_resolution_clock::now();
        RCLCPP_INFO(this->get_logger(), "Preprocessing time: %.2f ms",
                    std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time).count());
    }

    cv::Mat slMatToCvMat(const sl::Mat &input) {
        int cv_type = (input.getDataType() == sl::MAT_TYPE::U8_C4) ? CV_8UC4 : CV_8UC3;
        return cv::Mat(input.getHeight(), input.getWidth(), cv_type, input.getPtr<sl::uchar1>(sl::MEM::CPU));
    }

    // Parameters
    std::string camera_side_;
    std::vector<int64_t> roi_dimensions_;
    int frame_rate_;

    // Camera settings
    int left_camera_serial_number_ = 26853647;
    int right_camera_serial_number_ = 0;
    double velocity_;
    int shift_constant_;
    cv::Size model_dimensions_;

    // ROS Publisher
    rclcpp::Publisher<custom_interfaces::msg::ImageInput>::SharedPtr image_publisher_;
};

int main(int argc, char **argv) {
    rclcpp::init(argc, argv);
    auto node = std::make_shared<CameraNode>();
    rclcpp::spin(node);
    rclcpp::shutdown();
    return 0;
}
