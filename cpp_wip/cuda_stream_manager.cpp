// cuda_stream_manager.hpp
#pragma once
#include <cuda_runtime.h>
#include <memory>

class CudaStreamManager {
public:
    CudaStreamManager() {
        // Create a single CUDA stream
        cudaStreamCreate(&stream_);

        // Create CUDA events
        cudaEventCreate(&preprocess_done_);
        cudaEventCreate(&inference_done_);
    }

    ~CudaStreamManager() {
        // Destroy CUDA stream and events
        cudaStreamDestroy(stream_);
        cudaEventDestroy(preprocess_done_);
        cudaEventDestroy(inference_done_);
    }

    cudaStream_t getStream() const {
        return stream_;
    }

    cudaEvent_t& getPreprocessEvent() {
        return preprocess_done_;
    }

    cudaEvent_t& getInferenceEvent() {
        return inference_done_;
    }

private:
    cudaStream_t stream_;
    cudaEvent_t preprocess_done_, inference_done_;
};

using CudaStreamManagerPtr = std::shared_ptr<CudaStreamManager>;
