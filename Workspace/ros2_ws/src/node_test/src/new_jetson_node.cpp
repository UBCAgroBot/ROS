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
#include <engine.h>


// TODO - merge existing CMAKE with that of https://github.com/cyrusbehr/tensorrt-cpp-api
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
  MinimalSubscriber(char* arguments[])
      : Node("jetson_node")
  {
    camera_image_ = this->create_subscription<sensor_msgs::msg::Image>(
        "camera_image", 10, std::bind(&MinimalSubscriber::topic_callback, this, _1));
    bounding_box_ = this->create_publisher<custom_interface::msg::BoundingBox>("bounding_box", 10);
    class_list_ = std::make_unique<std::vector<std::string>>(load_class_list()); 
    create_ort_session();
    tensorrt_engine_ = create_tensorrt_engine();
    std::string onnx_model_path = arguments[3];      // TODO: might have to load paths from ENV variables
    std::string tensor_rt_model_path = arguments[4]; // TODO: this might lead to accessing invalid place in array
    load_tensorrt_engine_from_onnx(onnx_model_path, tensor_rt_model_path);
  }

private:
  rclcpp::Publisher<custom_interface::msg::BoundingBox>::SharedPtr bounding_box_; 
  rclcpp::Subscription<sensor_msgs::msg::Image>::SharedPtr camera_image_;
  Engine tensorrt_engine_;

  // TODO: cleanup ONNX code
  std::unique_ptr<Ort::Session> session_;
  std::unique_ptr<Ort::Env> environment_;

  std::unique_ptr<std::vector<std::string>> class_list_;
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

  void load_tensorrt_engine_from_onnx(Options &options, std::string &onnx_path, std::string &tensor_rt_model_path) {
    // TODO - can the following normalization setup for YOLO be triggered in response to an environment variable?
    //        We probably won't be switching the model we use while in-competition
    const bool mock_env_var_for_yolo = true;

    std::array<float, 3> subVals{0.f, 0.f, 0.f};
    std::array<float, 3> divVals{1.f, 1.f, 1.f};
    bool normalize = true;

    // If an onnx path is specified, we want to generate a (new) trt file based on that onnx model
    if (!onnx_path.empty())) {
      bool succ = engine.buildLoadNetwork(onnx_path, subVals, divVals, normalize);
      if (!succ) {
        throw std::runtime_error("Unable to build or load TRT engine");
      }
    } else {
      bool succ = engine.loadNetwork(tensor_rt_model_path, subVals, divVals, normalize);
      if (!succ) {
        throw std::runtime_error("Unable to load TRT engine");
      }
    }
  }

  bool does_tensorrt_file_exist(Options &options, std::string &onnx_path) 
  {
    const std::string tensorrt_path = get_tensorrt_path_from_oonx(options, onnx_path);
    return Util::doesFileExist(tensorrt_path);
  }

  // TODO - consider moving this to another file
  Engine create_tensorrt_engine()
  {
    const options = get_tensorrt_options();
    // IMPROVEMENT: How can I make the type of the engine dyanmic here?
    // A declaration and instantiation in one line?? My god...
    Engine<auto> = engine(options);
    return engine;
  }

  Options create_tensorrt_engine() {
    Options options;
    // Go with FP::16 initially since we care about performance.
    // TODO: Benchmark against FP::32 ?
    options.precision = Precision::FP16;
    // Specify path to calibration data if using INT8 precision
    options.calibrationDataDirectoryPath = "";
    // Specify batch size to optimize for
    // Q: For the purposes of the competition, will our batch size be equivalent to the number of 
    //    frames per second?
    options.optBatchSize = 1;
    // Specify max batch size we plan to run inference on (for the purposes of allocating enough memory ahead of time)
    // Q: What would this value be for the competition?
    options.maxBatchSize = 1;
    return options;
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
    
    
    cv::Mat input_vector = preprocess(msg);
    auto after_preprocess = std::chrono::high_resolution_clock::now();

    std::vector<std::vector<float>> outputs = image_predict(input_vector);

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

  custom_interface::msg::BoundingBox postprocessing(const sensor_msgs::msg::Image &image, std::vector<std::vector<std::vector<float>>> *outputs) const {
    std::vector<float> flat_data;
    flatten(*outputs, flat_data);

    float* data = flat_data.data();

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

  std::vector<<std::vector<std::vector<float>>> image_predict(cv::Mat input_vector) const
  {
    // Upload the image to GPU memory
    cv::cuda::GpuMat img;
    img.upload(input_vector);

    // Keep this line of code if model expects RGB image
    cv::cuda::cvtColor(img, img, cv::COLOR_BGR2RGB);

    std::vector<std::vector<std::vector<float>>> featureVectors;

    // TODO - implement batching for better performance
    std::vector<std::vector<cv::cuda::GpuMat>> inputs;
    std::vector<cv::cuda::GpuMat> input;
    input.emplace_back(std::move(img))
    inputs.emplace_back(std::move(input))
    engine.runInference(inputs, featureVectors)

    return featureVectors;
  }

  // takes in image message and converts into datatype for onnx model
  std::vector<float> preprocess(sensor_msgs::msg::Image image) const
  {
    cv_bridge::CvImagePtr cv_ptr = cv_bridge::toCvCopy(image, sensor_msgs::image_encodings::BGR8);
    cv::Mat image_raw = cv_ptr->image;

    cv::Mat resizedImage;
    cv::resize(image_raw, resizedImage, cv::Size(640, 640));

    resizedImage.convertTo(resizedImage, CV_32F, 2.0f / 255.0f, -1.0f);

    return resizedImage;
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

    // Returns peak memory in KB used by process during its execution
    long memory_usage = r_usage.ru_maxrss;

    // Convert memory usage to megabytes
    double memory_usage_mb = static_cast<double>(memory_usage) / 1024.0;
    RCLCPP_INFO(this->get_logger(), "Memory Usage: %.2f", memory_usage_mb);
  }
};

void flatten(const std::vector<std::vector<std::vector<float>>>& nested, std::vector<float>& flat) {
    for (const auto& outer : nested) {
        for (const auto& inner : outer) {
            flat.insert(flat.end(), inner.begin(), inner.end());
        }
    }
}

int main(int argc, char *argv[])
{
  // Q: Can i pass a the onnx model path as an argument to the minimal subscriber constructor?
  rclcpp::init(argc, argv);
  // Assume that the onnx and tensorrt paths can be passed in the constructor here
  rclcpp::spin(std::make_shared<MinimalSubscriber>(argv));
  rclcpp::shutdown();
  return 0;
}
