#pragma once

#include <string>
#include <vector>

namespace triton_client {

struct ModelConfig {
    std::string name = "yolov11";
    int input_width = 800;
    int input_height = 800;
    float confidence_threshold = 0.20f;
    float iou_threshold = 0.65f;
    int max_detections = 1024;
    float draw_confidence = 0.20f;
};

struct ServerConfig {
    std::string url = "localhost:8001";
    float timeout = 0.0f;  // 0 means no timeout
    bool verbose = false;
};

struct VideoConfig {
    float fps = 24.0f;
    int max_history = 2;
    int distance_threshold = 50;
    int line_thickness = 4;
    float font_scale = 0.6f;
    int font_thickness = 2;
};

struct PathsConfig {
    std::string data_yaml = "./data.yaml";
    std::string output_dir = "./output";
    std::string media_dir = "./media";
};

struct ClientConfig {
    ModelConfig model;
    ServerConfig server;
    VideoConfig video;
    PathsConfig paths;
};

} // namespace triton_client
