#include <functional>
#include <memory>
#include <chrono>
#include <ctime>

#include "rclcpp/rclcpp.hpp"
#include "std_msgs/msg/string.hpp"

#include <opencv2/opencv.hpp> // for image processing
#include <cv_bridge/cv_bridge.h>
#include "sensor_msgs/msg/image.hpp"

#include <onnxruntime/core/session/onnxruntime_cxx_api.h>


using std::placeholders::_1;
using namespace cv;

std::string image_predict(sensor_msgs::msg::Image in_image);

class MinimalSubscriber : public rclcpp::Node
{
public:
  MinimalSubscriber()
  : Node("minimal_subscriber")
  {
    subscription_ = this->create_subscription<sensor_msgs::msg::Image>(
      "topic", 10, std::bind(&MinimalSubscriber::topic_callback, this, _1));
    Ort::Env env(ORT_LOGGING_LEVEL_WARNING, "ModelRunner");

    std::string instanceName{"Image classifier inference"};
    environment_ = std::make_unique<Ort::Env>(OrtLoggingLevel::ORT_LOGGING_LEVEL_WARNING,"ModelRunner");

    Ort::SessionOptions session_options;
    session_options.SetIntraOpNumThreads(1);
    session_ = std::make_unique<Ort::Session>(env, "/home/valery/ros2_cpp_publisher/src/onnx_cpp/src/model.onnx", session_options);
  }

private:
  void topic_callback(const sensor_msgs::msg::Image & msg) const

  {
    auto start = std::chrono::high_resolution_clock::now();
    std::string prediction = image_predict(msg);
    // std::vector<int> expected = {5,4,3,3,0,3,6,6,5,5};
    RCLCPP_INFO(this->get_logger(), "Image: '%s',  %s", msg.header.frame_id.c_str(), prediction.c_str());
    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> elapsed = end - start;
    RCLCPP_INFO(this->get_logger(), "Time: '%s'", std::to_string(elapsed.count()).c_str());


  }

  std::string image_predict(sensor_msgs::msg::Image image) const{
  // convert the image into a better format:
    std::vector<float> input_vector = process_image(image);

    // Get the input tensor shape
    std::vector<int64_t> input_tensor_shape = {1,32, 32, 3};  // Adjust this to match your input shape

    // Create the input tensor object from the data
    Ort::MemoryInfo memory_info = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);
    Ort::Value input_tensor = Ort::Value::CreateTensor<float>(memory_info, input_vector.data(), input_vector.size(), input_tensor_shape.data(), input_tensor_shape.size());

    std::vector<const char*> input_names = { "args_0" };
    std::vector<Ort::Value> input_tensors;
    input_tensors.emplace_back(std::move(input_tensor));

    std::vector<const char*> output_names = { "dense_4" };

    std::vector<Ort::Value> output_tensors = session_->Run(Ort::RunOptions{ nullptr }, input_names.data(), input_tensors.data(), input_tensors.size(), output_names.data(), output_names.size());

    // Get a pointer to the data in the first output tensor
    float* floatarr = output_tensors[0].GetTensorMutableData<float>();
    int cls_idx = std::max_element(floatarr, floatarr + 10) - floatarr;

    return "prediction: " + std::to_string(cls_idx);
  }

std::vector<float> process_image(sensor_msgs::msg::Image image) const{
  cv_bridge::CvImagePtr cv_ptr = cv_bridge::toCvCopy(image, sensor_msgs::image_encodings::BGR8);
  cv::Mat image_raw =  cv_ptr->image;

  cv::Mat scaledImage,img_greyscale;
  image_raw.convertTo(scaledImage, CV_32F, 2.0f / 255.0f, -1.0f);   // Scale image pixels: [0 255] -> [-1, 1]

  std::vector<float> img_vector;

  img_vector.assign((float*)scaledImage.datastart, (float*)scaledImage.dataend);
  
  return img_vector;
}

  rclcpp::Subscription<sensor_msgs::msg::Image>::SharedPtr subscription_;
  std::unique_ptr<Ort::Session> session_;
  std::unique_ptr<Ort::Env> environment_;
};




int main(int argc, char * argv[])
{
  rclcpp::init(argc, argv);
  rclcpp::spin(std::make_shared<MinimalSubscriber>());
  rclcpp::shutdown();
  return 0;
}


