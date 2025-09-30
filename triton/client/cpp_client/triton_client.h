#pragma once

#include "config.h"
#include "utils.h"
#include "grpc_client.h"
#include "yolo_postprocess.h"
#include <memory>
#include <string>
#include <vector>
#include <opencv2/opencv.hpp>

namespace triton_client {

class TritonClient {
public:
    explicit TritonClient(const ClientConfig& config);
    ~TritonClient();
    
    // Connection management
    bool connect();
    void disconnect();
    
    // Model information
    void get_model_info();
    
    // Inference methods
    void run_dummy_inference();
    void run_image_inference(const std::string& image_path, const std::string& output_path = "");
    void run_video_inference(const std::string& video_path, const std::string& output_path = "");
    
private:
    ClientConfig config_;
    std::unique_ptr<GrpcClient> grpc_client_;
    std::unique_ptr<YoloPostprocessor> yolo_postprocessor_;
    std::vector<std::string> class_names_;
    
    // Helper methods
    void load_class_names();
    bool is_server_live();
    bool is_server_ready();
    bool is_model_ready(const std::string& model_name);
    
    // Inference helpers
    std::vector<float> prepare_input_tensor(const cv::Mat& image);
    std::vector<Detection> run_inference(const std::vector<float>& input_tensor, 
                                        int orig_width, int orig_height);
    void process_video_frame(cv::Mat& frame, PerformanceStats& stats, 
                           std::vector<std::vector<Detection>>& frame_history);
};

} // namespace triton_client
