#include <rclcpp/rclcpp.hpp>
#include <sensor_msgs/msg/image.hpp>
#include <std_msgs/msg/header.hpp>
#include <std_msgs/msg/string.hpp>
#include <std_msgs/msg/int32.hpp>
#include <cv_bridge/cv_bridge.h>
#include <opencv2/opencv.hpp>
#include <pybind11/embed.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>
#include <iostream>
#include <vector>

namespace py = pybind11;
using namespace std::chrono_literals;

class CameraNode : public rclcpp::Node {
public:
    CameraNode() : Node("camera_node"), left_on_(0), right_on_(0) {
        py::initialize_interpreter();
        py::module_ object_filter = py::module_::import("object_filter");

        left_camera_subscriber_ = this->create_subscription<sensor_msgs::msg::Image>(
            "image_data", 10, std::bind(&CameraNode::callback, this, std::placeholders::_1));
        right_camera_subscriber_ = this->create_subscription<sensor_msgs::msg::Image>(
            "image_data", 10, std::bind(&CameraNode::callback, this, std::placeholders::_1));
        timer_ = this->create_wall_timer(
            100ms, std::bind(&CameraNode::timer_callback, this));
        left_array_publisher_ = this->create_publisher<std_msgs::msg::Int32>("left_array_data", 10);
        right_array_publisher_ = this->create_publisher<std_msgs::msg::Int32>("right_array_data", 10);
    }

private:
    void timer_callback() {
        auto left_msg = std_msgs::msg::Int32();
        left_msg.data = left_on_;
        auto right_msg = std_msgs::msg::Int32();
        right_msg.data = right_on_;
        left_array_publisher_->publish(left_msg);
        right_array_publisher_->publish(right_msg);
    }

    void callback(const sensor_msgs::msg::Image::SharedPtr msg) {
        try {
            cv::Mat cv_image = cv_bridge::toCvShare(msg, "rgb8")->image;
            queue_.push_back(cv_image);
            preprocess_image();
        } catch (cv_bridge::Exception& e) {
            RCLCPP_ERROR(this->get_logger(), "cv_bridge exception: %s", e.what());
        }
    }

    void preprocess_image() {
        if (queue_.size() == 2) {
            cv::Mat left_image = queue_.front();
            queue_.pop_front();
            cv::Mat right_image = queue_.front();
            queue_.pop_front();

            cv::Mat stacked_image;
            cv::hconcat(left_image, right_image, stacked_image);
            cv::Mat padded_image;
            cv::copyMakeBorder(stacked_image, padded_image, 0, 0, 0, 0, cv::BORDER_CONSTANT, cv::Scalar(0, 0, 0));
            cv::Mat resized_image;
            cv::resize(padded_image, resized_image, cv::Size(255, 255));

            cv::imshow("Resized Image", resized_image);
            cv::waitKey(1);

            image_ = resized_image;
            model_inference();
        }
    }

    void model_inference() {
        // Implement model inference here...
        postprocess_image();
    }

    void postprocess_image() {
        cv::Mat img = cv::imread("your_file_name");
        if (img.empty()) {
            RCLCPP_ERROR(this->get_logger(), "Could not read the image.");
            return;
        }
        int height = img.rows;
        int width = img.cols;

        int width_cutoff = width / 2;
        cv::Mat s1 = img(cv::Range::all(), cv::Range(0, width_cutoff));
        cv::Mat s2 = img(cv::Range::all(), cv::Range(width_cutoff, width));
    }

    void object_filter(cv::Mat& image, std::vector<int>& bounding_box) {
        py::array_t<uint8_t> input_image = py::array_t<uint8_t>(
            {image.rows, image.cols, image.channels()}, image.data);
        py::tuple bbox = py::make_tuple(bounding_box[0], bounding_box[1], bounding_box[2], bounding_box[3]);
        py::object obj_filter = py::module_::import("object_filter").attr("object_filter");
        std::vector<std::vector<int>> detections = obj_filter(input_image, bbox).cast<std::vector<std::vector<int>>>();

        for (const auto& detection : detections) {
            int x = detection[0];
            int y = detection[1];
            int w = detection[2];
            int h = detection[3];
            cv::rectangle(image, cv::Rect(x, y, w, h), cv::Scalar(0, 255, 0), 3);
        }

        cv::imshow("Processed Image", image);
        cv::waitKey(1);
    }

    void verify_object(std::vector<int>& bounding_box, const std::string& side) {
        int x1 = bounding_box[0];
        int y1 = bounding_box[1];
        int w = bounding_box[2];
        int h = bounding_box[3];
        int x2 = x1 + w;
        int y2 = y1 + h;

        if (x1 >= ROI_X && x1 <= ROI_X + ROI_W && y1 >= ROI_Y && y1 <= ROI_Y + ROI_H) {
            if (side == "left") {
                left_on_ = 1;
            } else {
                right_on_ = 1;
            }
        } else {
            if (side == "left") {
                left_on_ = 0;
            } else {
                right_on_ = 0;
            }
        }
    }

    rclcpp::Subscription<sensor_msgs::msg::Image>::SharedPtr left_camera_subscriber_;
    rclcpp::Subscription<sensor_msgs::msg::Image>::SharedPtr right_camera_subscriber_;
    rclcpp::Publisher<std_msgs::msg::Int32>::SharedPtr left_array_publisher_;
    rclcpp::Publisher<std_msgs::msg::Int32>::SharedPtr right_array_publisher_;
    rclcpp::TimerBase::SharedPtr timer_;
    int left_on_;
    int right_on_;
    std::deque<cv::Mat> queue_;
    cv::Mat image_;
};

int main(int argc, char* argv[]) {
    rclcpp::init(argc, argv);
    auto node = std::make_shared<CameraNode>();
    rclcpp::spin(node);
    rclcpp::shutdown();
    py::finalize_interpreter();
    return 0;
}
