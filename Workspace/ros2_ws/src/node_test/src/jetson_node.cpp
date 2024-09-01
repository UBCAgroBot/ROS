#include <functional>
#include <memory>
#include <chrono>
#include <ctime>
#include <fstream>

#include "rclcpp/rclcpp.hpp"
#include "std_msgs/msg/string.hpp"

#include <opencv2/opencv.hpp> // for image processing
#include <cv_bridge/cv_bridge.h>
#include "sensor_msgs/msg/image.hpp"
#include "custom_interface/msg/bounding_box.hpp"                                            // CHANGE

#include <onnxruntime/core/session/onnxruntime_cxx_api.h>
#include <sys/resource.h>


using std::placeholders::_1;
using namespace cv;

class MinimalSubscriber : public rclcpp::Node
{
  const int64_t INPUT_WIDTH = 640.0;
  const int64_t INPUT_HEIGHT = 640.0;
  const float SCORE_THRESHOLD = 0.2;
  const float NMS_THRESHOLD = 0.4;
  const float CONFIDENCE_THRESHOLD = 0.3;

public:
  MinimalSubscriber()
      : Node("jetson_node")
  {
    camera_image_ = this->create_subscription<sensor_msgs::msg::Image>(
        "camera_image", 10, std::bind(&MinimalSubscriber::topic_callback, this, _1));
    bounding_box_ = this->create_publisher<custom_interface::msg::BoundingBox>("bounding_box", 10);

    class_list_ = std::make_unique<std::vector<std::string>>(load_class_list());
    create_ort_session();
  }

private:
  std::vector<std::string> load_class_list()
{
    std::vector<std::string> class_list;
    std::ifstream ifs("src/node_test/src/classes.txt");
    std::string line;
    while (getline(ifs, line))
    {
        class_list.push_back(line);
    }
    return class_list;
}

  void create_ort_session()
  {
    Ort::Env env(ORT_LOGGING_LEVEL_ERROR, "ModelRunner");

    std::string instanceName{"Image classifier inference"};
    environment_ = std::make_unique<Ort::Env>(OrtLoggingLevel::ORT_LOGGING_LEVEL_WARNING, "ModelRunner");

    Ort::SessionOptions session_options;
    session_options.SetIntraOpNumThreads(1);
    session_ = std::make_unique<Ort::Session>(env, "src/node_test/src/yolov5s.onnx", session_options);
  }

  void topic_callback(const sensor_msgs::msg::Image &msg) const

  {
    auto start = std::chrono::high_resolution_clock::now();
    
    
    std::vector<float> input_vector = preprocess(msg);
    auto after_preprocess = std::chrono::high_resolution_clock::now();

    std::vector<Ort::Value> outputs = image_predict(input_vector);

    auto after_model = std::chrono::high_resolution_clock::now();

    custom_interface::msg::BoundingBox out_box = postprocessing(msg,&outputs);

    auto after_postprocess = std::chrono::high_resolution_clock::now();

    bounding_box_->publish(out_box);

    auto end = std::chrono::high_resolution_clock::now();

    std::chrono::duration<double> elapsed = end - start;
    RCLCPP_INFO(this->get_logger(), "Total time:\t\t'%s'", std::to_string(std::chrono::duration<double>(end-start).count()).c_str());
    RCLCPP_INFO(this->get_logger(), "Time for preprocess:\t\t%s'", std::to_string(std::chrono::duration<double>(after_preprocess-start).count()).c_str());
    RCLCPP_INFO(this->get_logger(), "Time for model:\t\t%s'", std::to_string(std::chrono::duration<double>(after_model-after_preprocess).count()).c_str());
    RCLCPP_INFO(this->get_logger(), "Time for postprocess:\t\t%s'", std::to_string(std::chrono::duration<double>(after_postprocess-after_model).count()).c_str());

    checkRAMUsage();

  }

  custom_interface::msg::BoundingBox postprocessing(const sensor_msgs::msg::Image &image,std::vector<Ort::Value>  *outputs) const {
    float* data = (*outputs)[0].GetTensorMutableData<float>();

    std::vector<int> boxes;
    std::vector<float> predictions;
    std::vector<int> class_nums;

    const int dimensions = 85;
    const int rows = 25200;

    float x_factor = image.width / INPUT_WIDTH;
    float y_factor = image.height / INPUT_HEIGHT;

    for (int i = 0; i < rows; ++i) {
      float confidence = data[4];
      if (confidence >= CONFIDENCE_THRESHOLD) {
        float * classes_scores = data + 5;
        cv::Mat scores(1, class_list_->size(), CV_32FC1, classes_scores);
        cv::Point class_id;
        double max_class_score;
        minMaxLoc(scores, 0, &max_class_score, 0, &class_id);
        if (max_class_score > SCORE_THRESHOLD) {
          predictions.push_back(confidence);
          class_nums.push_back(class_id.x);

          float x = data[0];
          float y = data[1];
          float w = data[2];
          float h = data[3];
          int left = int((x - 0.5 * w) * x_factor);
          int top = int((y - 0.5 * h) * y_factor);
          int width = int(w * x_factor);
          int height = int(h * y_factor);
          std::vector<int> box = {left, top, width, height};

          boxes.insert(boxes.end(), box.begin(), box.end());
        }
      }
      data += dimensions;
    }

    custom_interface::msg::BoundingBox out_box;
    out_box.box = boxes;
    out_box.scores = predictions;
    out_box.class_num = class_nums;

    return out_box;
  }

  std::vector<Ort::Value> image_predict(std::vector<float> input_vector) const
  {
    // Get the input tensor shape
    std::vector<int64_t> input_tensor_shape = {1,3, INPUT_WIDTH, INPUT_HEIGHT};// Adjust this to match your input shape

    // Create the input tensor object from the data
    Ort::MemoryInfo memory_info = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);
    Ort::Value input_tensor = Ort::Value::CreateTensor<float>(memory_info, input_vector.data(), input_vector.size(), input_tensor_shape.data(), input_tensor_shape.size());

    std::vector<const char *> input_names = {"images"};
    std::vector<Ort::Value> input_tensors;
    input_tensors.emplace_back(std::move(input_tensor));

    std::vector<const char *> output_names = {"output"};

    std::vector<Ort::Value> output_tensors = session_->Run(Ort::RunOptions{nullptr}, input_names.data(), input_tensors.data(), input_tensors.size(), output_names.data(), output_names.size());

    return output_tensors;
  }

  // takes in image message and converts into datatype for onnx model
  std::vector<float> preprocess(sensor_msgs::msg::Image image) const
  {
    cv_bridge::CvImagePtr cv_ptr = cv_bridge::toCvCopy(image, sensor_msgs::image_encodings::BGR8);
    cv::Mat image_raw = cv_ptr->image;

    cv::Mat resizedImage;
    cv::resize(image_raw, resizedImage, cv::Size(640, 640));

    resizedImage.convertTo(resizedImage, CV_32F, 2.0f / 255.0f, -1.0f);

    std::vector<float> img_vector;
    img_vector.assign((float *)resizedImage.datastart, (float *)resizedImage.dataend);

    return img_vector;
  }

  // takes in image message and converts into datatype for onnx model
  void checkRAMUsage() const
  {
    struct rusage r_usage;

    // Get resource usage
    if (getrusage(RUSAGE_SELF, &r_usage) != 0)
    {
      RCLCPP_INFO(this->get_logger(), "Error: Unable to get resource usage.");
    }

    // Memory usage in kilobytes
    long memory_usage = r_usage.ru_maxrss;

    // Convert memory usage to megabytes
    double memory_usage_mb = static_cast<double>(memory_usage) / 1024.0;
    RCLCPP_INFO(this->get_logger(), "Memory Usage: %.2f", memory_usage_mb);
  }
  
  rclcpp::Publisher<custom_interface::msg::BoundingBox>::SharedPtr bounding_box_;
  rclcpp::Subscription<sensor_msgs::msg::Image>::SharedPtr camera_image_;

  std::unique_ptr<Ort::Session> session_;
  std::unique_ptr<Ort::Env> environment_;

  std::unique_ptr<std::vector<std::string>> class_list_;
};

int main(int argc, char *argv[])
{
  rclcpp::init(argc, argv);
  rclcpp::spin(std::make_shared<MinimalSubscriber>());
  rclcpp::shutdown();
  return 0;
}
