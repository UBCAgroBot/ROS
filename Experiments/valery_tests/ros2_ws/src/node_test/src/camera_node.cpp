#include <chrono>
#include <functional>
#include <memory>
#include <string>

#include "rclcpp/rclcpp.hpp"
#include "std_msgs/msg/string.hpp"

#include <opencv2/opencv.hpp> // for image processing
#include <cv_bridge/cv_bridge.h>
#include "sensor_msgs/msg/image.hpp"

// #include <onnxruntime_c_api.h>

#include <sys/resource.h>

using namespace std::chrono_literals;
using namespace cv;

/* This example creates a subclass of Node and uses std::bind() to register a
 * member function as a callback from the timer. */

class MinimalPublisher : public rclcpp::Node
{
public:
  MinimalPublisher()
  : Node("camera_node"), count_(0)
  {
    camera_image_ = this->create_publisher<sensor_msgs::msg::Image>("camera_image", 10);
    timer_ = this->create_wall_timer(
      3000ms, std::bind(&MinimalPublisher::picture_publisher, this));
  }

private:
  void picture_publisher()
  {
    cv::Mat image = cv::imread("src/node_test/src/images/mnist_"+ std::to_string(count_ % 10)+ ".png");

    if (image.empty()) {
      RCLCPP_ERROR(this->get_logger(), "Image  NOT  found");
      return;
    }

    std_msgs::msg::Header header = std_msgs::msg::Header(); // empty header
    header.frame_id = "image_" + std::to_string(count_++ % 10); // time

    cv_bridge::CvImage img_bridge = cv_bridge::CvImage(header, sensor_msgs::image_encodings::BGR8, image);

    sensor_msgs::msg::Image out_image; // >> message to be sent
    img_bridge.toImageMsg(out_image); // from cv_bridge to sensor_msgs::Image

    auto message = std_msgs::msg::String();
    RCLCPP_INFO(this->get_logger(), "Publishing and working: '%s'", header.frame_id.c_str());
    camera_image_->publish(out_image);

    checkRAMUsage();

  }


// takes in image message and converts into datatype for onnx model
  void checkRAMUsage() const{
    struct rusage r_usage;

    // Get resource usage
    if (getrusage(RUSAGE_SELF, &r_usage) != 0) {
        RCLCPP_INFO(this->get_logger(), "Error: Unable to get resource usage.");
  }

    // Memory usage in kilobytes
    long memory_usage = r_usage.ru_maxrss;

    // Convert memory usage to megabytes
    double memory_usage_mb = static_cast<double>(memory_usage) / 1024.0;
    RCLCPP_INFO(this->get_logger(), "Memory Usage: %.2f", memory_usage_mb);
}


  rclcpp::TimerBase::SharedPtr timer_;
  rclcpp::Publisher<sensor_msgs::msg::Image>::SharedPtr camera_image_;
  size_t count_;
};


int main(int argc, char * argv[])
{
  rclcpp::init(argc, argv);
  rclcpp::spin(std::make_shared<MinimalPublisher>());
  rclcpp::shutdown();
  return 0;
}
