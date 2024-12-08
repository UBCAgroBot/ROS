#include <rclcpp/rclcpp.hpp>
#include <std_msgs/msg/string.hpp>
#include <sensor_msgs/msg/image.hpp>
#include <cv_bridge/cv_bridge.h>
#include <opencv2/opencv.hpp>
#include <opencv2/core/cuda.hpp>
#include <opencv2/cudaimgproc.hpp>
#include <opencv2/cudawarping.hpp>
#include <pycuda.driver.hpp>
#include <queue>

// Key Points:

// CUDA: I used cv::cuda::GpuMat for GPU operations similar to the OpenCV CUDA functions in Python.

// ROS2: The ROS2 publisher and service setup uses std_msgs::msg::String and sensor_msgs::msg::Image for handling messages.

// Queue: The C++ queue (std::queue) is used to manage image frames.

// Precision Handling: The precision-related code was mentioned as a placeholder. In C++, you can implement FP16 handling with CUDA directly or using external libraries like CuPy (if available).

// Time Handling: Time is measured using the standard C++ chrono library.


class VideoNode : public rclcpp::Node
{
public:
    VideoNode() : Node("video_node")
    {
        cudaSetDevice(0);
        cuCtxCreate(&cuda_context_, 0, 0);
        
        declare_parameter<std::string>("video_path", "/home/usr/Desktop/ROS/assets/video.mp4");
        declare_parameter<int>("loop", 0);
        declare_parameter<int>("frame_rate", 30);
        declare_parameter<std::vector<int>>("model_dimensions", {448, 1024});
        declare_parameter<int>("shift_constant", 1);
        declare_parameter<std::vector<int>>("roi_dimensions", {0, 0, 100, 100});
        declare_parameter<std::string>("precision", "fp32");

        video_path_ = get_parameter("video_path").get_parameter_value().get<std::string>();
        loop_ = get_parameter("loop").get_parameter_value().get<int>();
        frame_rate_ = get_parameter("frame_rate").get_parameter_value().get<int>();
        dimensions_ = get_parameter("model_dimensions").get_parameter_value().get<std::vector<int>>();
        shift_constant_ = get_parameter("shift_constant").get_parameter_value().get<int>();
        roi_dimensions_ = get_parameter("roi_dimensions").get_parameter_value().get<std::vector<int>>();
        precision_ = get_parameter("precision").get_parameter_value().get<std::string>();

        pointer_publisher_ = this->create_publisher<std_msgs::msg::String>("preprocessing_done", 10);
        image_service_ = this->create_service<sensor_msgs::srv::Image>("image_data", std::bind(&VideoNode::image_callback, this, std::placeholders::_1));

        velocity_ = {0, 0, 0};
        image_queue_ = std::queue<std::pair<cv::Mat, int>>();

        publish_video_frames();

        if (precision_ == "fp32")
        {
            // Handle fp32 precision
        }
        else if (precision_ == "fp16")
        {
            // Handle fp16 precision
        }
        else
        {
            RCLCPP_ERROR(this->get_logger(), "Invalid precision: %s", precision_.c_str());
        }
    }

private:
    void image_callback(const std::shared_ptr<sensor_msgs::srv::Image::Request> request,
                        std::shared_ptr<sensor_msgs::srv::Image::Response> response)
    {
        RCLCPP_INFO(this->get_logger(), "Received image request");

        if (!image_queue_.empty())
        {
            auto image_data = image_queue_.front(); // Get the image from the queue
            image_queue_.pop();  // Remove from queue
            cv::Mat cv_image = image_data.first;
            int velocity = image_data.second;

            // Convert OpenCV image to ROS2 Image message using cv_bridge
            cv_bridge::CvImage cv_image_msg;
            cv_image_msg.header.stamp = this->get_clock()->now();
            cv_image_msg.header.frame_id = std::to_string(velocity);
            cv_image_msg.encoding = "rgb8";
            cv_image_msg.image = cv_image;

            response->image = *cv_image_msg.toImageMsg();
            return;
        }
        else
        {
            RCLCPP_ERROR(this->get_logger(), "Image queue is empty");
            return;
        }
    }

    void publish_video_frames()
    {
        if (!std::filesystem::exists(video_path_))
        {
            RCLCPP_ERROR(this->get_logger(), "Video file not found at %s", video_path_.c_str());
            return;
        }

        cv::VideoCapture cap(video_path_);
        if (!cap.isOpened())
        {
            RCLCPP_ERROR(this->get_logger(), "Failed to open video: %s", video_path_.c_str());
            return;
        }

        int loops = 0;
        while (rclcpp::ok() && (loop_ == -1 || loops < loop_))
        {
            while (cap.isOpened() && rclcpp::ok())
            {
                cv::Mat frame;
                bool ret = cap.read(frame);
                if (!ret)
                {
                    break;
                }

                image_queue_.push({frame, velocity_[0]});
                preprocess_image(frame);
                std::this_thread::sleep_for(std::chrono::milliseconds(1000 / frame_rate_));
            }

            if (loop_ > 0)
            {
                loops++;
            }

            if (loop_ != -1)
            {
                cap.set(cv::CAP_PROP_POS_FRAMES, 0); // restart video
            }
        }
        cap.release();
    }

    void preprocess_image(const cv::Mat &image)
    {
        auto tic = std::chrono::high_resolution_clock::now();

        int roi_x = roi_dimensions_[0], roi_y = roi_dimensions_[1], roi_w = roi_dimensions_[2], roi_h = roi_dimensions_[3];
        int shifted_x = roi_x + std::abs(velocity_[0]) * shift_constant_;

        cv::cuda::GpuMat gpu_image;
        gpu_image.upload(image);

        cv::cuda::cvtColor(gpu_image, gpu_image, cv::COLOR_RGBA2RGB); // Remove alpha channel
        gpu_image = gpu_image(cv::Rect(shifted_x, roi_y, roi_w, roi_h)); // Crop the image to ROI
        cv::cuda::resize(gpu_image, gpu_image, cv::Size(dimensions_[1], dimensions_[0])); // Resize to model dimensions

        cv::Mat output;
        gpu_image.download(output);

        // Here we would use GPU-based libraries (CuPy or CUDA directly in C++) to handle preprocessing for model
        // Assuming normalization and transposition occur here

        auto toc = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double, std::milli> duration = toc - tic;
        RCLCPP_INFO(this->get_logger(), "Preprocessing: %f ms", duration.count());

        // Example for publishing the IPC handle (just sending a dummy value)
        std_msgs::msg::String ipc_handle_msg;
        ipc_handle_msg.data = "dummy_ipc_handle"; // Placeholder for IPC handle
        pointer_publisher_->publish(ipc_handle_msg);
    }

private:
    std::string video_path_;
    int loop_;
    int frame_rate_;
    std::vector<int> dimensions_;
    int shift_constant_;
    std::vector<int> roi_dimensions_;
    std::string precision_;
    std::queue<std::pair<cv::Mat, int>> image_queue_;
    std::vector<int> velocity_;

    rclcpp::Publisher<std_msgs::msg::String>::SharedPtr pointer_publisher_;
    rclcpp::Service<sensor_msgs::srv::Image>::SharedPtr image_service_;
    CUcontext cuda_context_;
};

int main(int argc, char **argv)
{
    rclcpp::init(argc, argv);
    rclcpp::spin(std::make_shared<VideoNode>());
    rclcpp::shutdown();
    return 0;
}
