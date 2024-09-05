#include <rclcpp/rclcpp.hpp>
#include <sensor_msgs/msg/image.hpp>
#include <opencv2/opencv.hpp>

class preprocessing_node : public rclcpp::Node
{
public:
    preprocessing_node() : Node("image_concatenator_node")
    {
        // Subscribe to left and right image topics
        left_image_subscriber_ = create_subscription<sensor_msgs::msg::Image>(
            "left_image_data", 10, std::bind(&preprocessing_node::leftImageCallback, this, std::placeholders::_1));
        right_image_subscriber_ = create_subscription<sensor_msgs::msg::Image>(
            "right_image_data", 10, std::bind(&preprocessing_node::rightImageCallback, this, std::placeholders::_1));

        // Advertise concatenated and resized image topic
        concatenated_image_publisher_ = create_publisher<sensor_msgs::msg::Image>("concatenated_image_data", 10);

        // Advertise preprocessed image topic
        preprocessed_image_publisher_ = create_publisher<sensor_msgs::msg::Image>("preprocessed_image", 10);
    }

private:
    void leftImageCallback(const sensor_msgs::msg::Image::SharedPtr msg)
    {
        left_image_ = cv_bridge::toCvCopy(msg, "rgb8")->image;
        processAndPublish();
    }

    void rightImageCallback(const sensor_msgs::msg::Image::SharedPtr msg)
    {
        right_image_ = cv_bridge::toCvCopy(msg, "rgb8")->image;
        processAndPublish();
    }

    void processAndPublish()
    {
        if (left_image_.empty() || right_image_.empty())
            return;

        cv::Mat concatenated_image;
        cv::vconcat(left_image_, right_image_, concatenated_image);

        cv::Size target_size(640, 480);
        cv::resize(concatenated_image, concatenated_image, target_size);

        // Publish concatenated and resized image
        sensor_msgs::msg::Image::SharedPtr output_msg = cv_bridge::CvImage(std_msgs::msg::Header(), "rgb8", concatenated_image).toImageMsg();
        concatenated_image_publisher_->publish(output_msg);

        // Publish preprocessed image
        preprocessed_image_publisher_->publish(output_msg);
    }

    rclcpp::Subscription<sensor_msgs::msg::Image>::SharedPtr left_image_subscriber_;
    rclcpp::Subscription<sensor_msgs::msg::Image>::SharedPtr right_image_subscriber_;
    rclcpp::Publisher<sensor_msgs::msg::Image>::SharedPtr concatenated_image_publisher_;
    rclcpp::Publisher<sensor_msgs::msg::Image>::SharedPtr preprocessed_image_publisher_;

    cv::Mat left_image_;
    cv::Mat right_image_;
};

int main(int argc, char *argv[])
{
    rclcpp::init(argc, argv);
    auto node = std::make_shared<preprocessing_node>();
    rclcpp::spin(node);
    rclcpp::shutdown();
    return 0;
}