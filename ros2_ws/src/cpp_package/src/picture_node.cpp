#include <chrono>
#include <fstream>
#include <iostream>
#include <memory>
#include <random>
#include <string>
#include <vector>
#include <filesystem>
#include <opencv2/opencv.hpp>

#include "rclcpp/rclcpp.hpp"
#include "sensor_msgs/msg/image.hpp"
#include "std_msgs/msg/header.hpp"
#include "custom_interfaces/msg/image_input.hpp"
#include "cv_bridge/cv_bridge.h"

using namespace std::chrono_literals;
namespace fs = std::filesystem;


// Parameter Handling:
// Parameters such as static_image_path, loop, and frame_rate are declared and fetched using the ROS2 parameter system.

// Image Preprocessing:
// Example preprocessing includes resizing the image. Additional transformations can be added as needed.

// ROS2 Message Conversion:
// OpenCV images are converted to ROS2 messages using the cv_bridge library.

// Dynamic Publishing:
// Publishes images at a configurable frame rate using a timer.

// Error Handling:
// Ensures that the node throws meaningful errors if the image path is invalid or no images are found.

// Random Velocity:
// A random velocity is generated for each image for testing purposes.

class PictureNode : public rclcpp::Node
{
public:
    PictureNode() : Node("picture_node"), loop_count_(0), image_counter_(0)
    {
        // Declare parameters
        declare_parameter<std::string>("static_image_path", "/home/user/ROS/assets/maize/IMG_1822_14.JPG");
        declare_parameter<int>("loop", -1);  // 0 = don't loop, >0 = # of loops, -1 = loop forever
        declare_parameter<int>("frame_rate", 1); // Desired frame rate for publishing

        static_image_path_ = get_parameter("static_image_path").as_string();
        loop_ = get_parameter("loop").as_int();
        frame_rate_ = get_parameter("frame_rate").as_int();

        if (!fs::exists(static_image_path_))
        {
            RCLCPP_ERROR(this->get_logger(), "Static image path does not exist: %s", static_image_path_.c_str());
            throw std::runtime_error("Static image path not found");
        }

        // Load images
        load_images();

        // Publisher
        image_publisher_ = this->create_publisher<custom_interfaces::msg::ImageInput>("input_image", 10);

        // Timer
        auto timer_period = std::chrono::duration<double>(1.0 / frame_rate_);
        timer_ = this->create_wall_timer(timer_period, std::bind(&PictureNode::publish_static_image, this));
    }

private:
    void load_images()
    {
        if (fs::is_regular_file(static_image_path_))
        {
            if (is_valid_image(static_image_path_))
            {
                images_.push_back(cv::imread(static_image_path_));
            }
        }
        else if (fs::is_directory(static_image_path_))
        {
            for (const auto &entry : fs::directory_iterator(static_image_path_))
            {
                if (entry.is_regular_file() && is_valid_image(entry.path()))
                {
                    images_.push_back(cv::imread(entry.path()));
                }
            }
        }

        if (images_.empty())
        {
            RCLCPP_ERROR(this->get_logger(), "No valid images found in path: %s", static_image_path_.c_str());
            throw std::runtime_error("No valid images found");
        }
    }

    bool is_valid_image(const fs::path &path)
    {
        auto ext = path.extension().string();
        return ext == ".JPG" || ext == ".jpg" || ext == ".png";
    }

    void publish_static_image()
    {
        if (loop_ == -1 || loop_count_ < loop_)
        {
            size_t position = image_counter_ % images_.size();
            RCLCPP_INFO(this->get_logger(), "Publishing image %zu/%zu", position + 1, images_.size());

            // Prepare the image
            auto raw_image = images_[position];
            auto preprocessed_image = preprocess_image(raw_image);

            // Convert to ROS2 messages
            auto raw_msg = cv_bridge::CvImage(std_msgs::msg::Header(), "bgr8", raw_image).toImageMsg();
            auto preprocessed_msg = cv_bridge::CvImage(std_msgs::msg::Header(), "bgr8", preprocessed_image).toImageMsg();

            // Publish message
            custom_interfaces::msg::ImageInput image_input_msg;
            image_input_msg.header.stamp = this->get_clock()->now();
            image_input_msg.header.frame_id = "static_image";
            image_input_msg.raw_image = *raw_msg;
            image_input_msg.preprocessed_image = *preprocessed_msg;
            image_input_msg.velocity = generate_random_velocity();

            image_publisher_->publish(image_input_msg);

            // Update counters
            image_counter_++;
            loop_count_ = image_counter_ / images_.size();
        }
        else
        {
            // Stop publishing
            timer_->cancel();
            RCLCPP_INFO(this->get_logger(), "Finished publishing images");
            rclcpp::shutdown();
        }
    }

    cv::Mat preprocess_image(const cv::Mat &image)
    {
        cv::Mat processed;
        cv::resize(image, processed, cv::Size(224, 224)); // Example preprocessing
        return processed;
    }

    float generate_random_velocity()
    {
        static std::random_device rd;
        static std::mt19937 gen(rd());
        static std::uniform_real_distribution<> dis(0.0, 1.0);
        return dis(gen);
    }

    // Parameters
    std::string static_image_path_;
    int loop_;
    int frame_rate_;

    // Images and counters
    std::vector<cv::Mat> images_;
    size_t loop_count_;
    size_t image_counter_;

    // ROS2 publishers and timers
    rclcpp::Publisher<custom_interfaces::msg::ImageInput>::SharedPtr image_publisher_;
    rclcpp::TimerBase::SharedPtr timer_;
};

int main(int argc, char **argv)
{
    rclcpp::init(argc, argv);
    auto picture_node = std::make_shared<PictureNode>();
    rclcpp::spin(picture_node);
    rclcpp::shutdown();
    return 0;
}
