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


using namespace std::chrono_literals;
using namespace cv;

/* This example creates a subclass of Node and uses std::bind() to register a
 * member function as a callback from the timer. */

class MinimalPublisher : public rclcpp::Node
{
public:
  MinimalPublisher()
  : Node("minimal_publisher"), count_(0)
  {
    publisher_ = this->create_publisher<sensor_msgs::msg::Image>("topic", 10);
    timer_ = this->create_wall_timer(
      500ms, std::bind(&MinimalPublisher::timer_callback, this));
  }

private:
  void timer_callback()
  {
    cv::Mat image = cv::imread("ros2_ws/src/onnx_cpp/src/images/mnist_"+ std::to_string(count_ % 10)+ ".png");

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
    publisher_->publish(out_image);


  }
  rclcpp::TimerBase::SharedPtr timer_;
  rclcpp::Publisher<sensor_msgs::msg::Image>::SharedPtr publisher_;
  size_t count_;
};

int main(int argc, char * argv[])
{
  rclcpp::init(argc, argv);
  rclcpp::spin(std::make_shared<MinimalPublisher>());
  rclcpp::shutdown();
  return 0;
}
