// preprocessing_node.cpp
#include <rclcpp/rclcpp.hpp>
#include "cuda_stream_manager.hpp"

class PreprocessingNode : public rclcpp::Node {
public:
    PreprocessingNode(const CudaStreamManagerPtr& cuda_manager)
        : Node("preprocessing_node"), cuda_manager_(cuda_manager) {}

    void preprocess() {
        // Perform GPU preprocessing here using cuda_manager_->getStream()

        // Signal that preprocessing is done
        cudaEventRecord(cuda_manager_->getPreprocessEvent(), cuda_manager_->getStream());
    }

private:
    CudaStreamManagerPtr cuda_manager_;
};

// Register as a composable node
#include "rclcpp_components/register_node_macro.hpp"
RCLCPP_COMPONENTS_REGISTER_NODE(PreprocessingNode)
