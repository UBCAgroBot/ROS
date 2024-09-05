#include <opencv2/opencv.hpp>
#include <pybind11/embed.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>
#include <fstream>
#include <sstream>
#include <iostream>

namespace py = pybind11;

class ImageTester {
public:
    ImageTester() {
        py::initialize_interpreter();
        py::module_ object_filter = py::module_::import("object_filter");
    }

    ~ImageTester() {
        py::finalize_interpreter();
    }

    void run_tests(const std::string& image_folder) {
        // List all image files and their corresponding annotation files
        std::vector<std::string> image_files = {
            "IMG_1822_14.JPG", "IMG_1848_27.JPG", "IMG_1877_39.JPG"
            // Add more files as needed
        };

        for (const auto& image_file : image_files) {
            std::string image_path = image_folder + "/" + image_file;
            std::string annotation_path = image_folder + "/" + image_file.substr(0, image_file.find_last_of('.')) + ".txt";

            cv::Mat image = cv::imread(image_path);
            if (image.empty()) {
                std::cerr << "Could not read the image: " << image_path << std::endl;
                continue;
            }

            std::vector<std::vector<int>> bounding_boxes = load_annotations(annotation_path);
            preprocess_image(image);
            object_filter(image, bounding_boxes);
            postprocess_image(image);
        }
    }

private:
    void preprocess_image(cv::Mat& image) {
        cv::Mat resized_image;
        cv::resize(image, resized_image, cv::Size(255, 255));
        cv::imshow("Resized Image", resized_image);
        cv::waitKey(1);
        image = resized_image;
    }

    void object_filter(cv::Mat& image, const std::vector<std::vector<int>>& bounding_boxes) {
        py::array_t<uint8_t> input_image = py::array_t<uint8_t>(
            {image.rows, image.cols, image.channels()}, image.data);

        py::object obj_filter = py::module_::import("object_filter").attr("object_filter");
        std::vector<std::vector<int>> detections = obj_filter(input_image, bounding_boxes).cast<std::vector<std::vector<int>>>();

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

    void postprocess_image(cv::Mat& image) {
        int width_cutoff = image.cols / 2;
        cv::Mat s1 = image(cv::Range::all(), cv::Range(0, width_cutoff));
        cv::Mat s2 = image(cv::Range::all(), cv::Range(width_cutoff, image.cols));
        cv::imshow("Postprocessed Image - Part 1", s1);
        cv::imshow("Postprocessed Image - Part 2", s2);
        cv::waitKey(1);
    }

    std::vector<std::vector<int>> load_annotations(const std::string& annotation_file) {
        std::vector<std::vector<int>> bounding_boxes;
        std::ifstream infile(annotation_file);
        std::string line;
        while (std::getline(infile, line)) {
            std::istringstream iss(line);
            int class_id;
            float x, y, width, height;
            if (!(iss >> class_id >> x >> y >> width >> height)) {
                break;
            }
            int img_width = 255; // Assuming the images are resized to 255x255
            int img_height = 255;
            int x1 = static_cast<int>(x * img_width - (width * img_width / 2));
            int y1 = static_cast<int>(y * img_height - (height * img_height / 2));
            int w = static_cast<int>(width * img_width);
            int h = static_cast<int>(height * img_height);
            bounding_boxes.push_back({x1, y1, w, h});
        }
        return bounding_boxes;
    }
};

int main() {
    ImageTester tester;
    tester.run_tests("../sample_maize_images"); // Adjust the path as needed
    return 0;
}
